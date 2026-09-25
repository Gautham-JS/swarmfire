"""
Shared PPO + TrXL training internals.

This is everything from the old single-file `train_parallel.py` that is
NOT specific to a particular environment: the actor-critic, the rollout
buffer, the reward normaliser, and the main train()/evaluate() loops.

Each benchmark (wildfire, memory maze, ...) gets its own thin script that:
  1. defines a Config subclass with its own env-specific fields,
  2. defines make_env_fn(cfg, rank) -> thunk returning a gymnasium.Env,
  3. optionally defines domain_metrics_fn(info) -> dict for extra
     per-episode wandb scalars,
  4. calls train(cfg, make_env_fn, extractor_kwargs=..., domain_metrics_fn=...)

Because every script funnels through this one train() function, the wandb
metric names, shapes, and logging cadence are identical across benchmarks
-- runs from different environments overlay cleanly in the same project
(differentiate them with cfg.benchmark_name, logged into wandb config and
used as a tag).

Changes vs. previous version
----------------------------
 1. Reward normalisation no longer subtracts the mean. New cfg.reward_norm:
    "returns" (default; scale by std of discounted returns, no centering)
    or "none" (raw rewards, recommended first test for Memory Maze).
 2. Bootstrap value (get_value) no longer mutates the extractor memory.
 3. The critic reads the PRE-step memory in rollout, so it matches the
    memory snapshot used in the PPO update (old_values == new_values at
    the first update step).
 4. GAE uses dones[t] (was dones[t+1] except for the last step).
 5. Non-finite loss / gradients skip the update instead of poisoning the
    weights; offending params are logged. Grad norm is logged to wandb.
 6. evaluate() runs in eval mode and restores the training memory
    afterwards; train() no longer zeroes memory after evaluation.
 7. Advantages are normalised once per rollout, not per minibatch.
 8. target_kl is checked against the current epoch's mean KL.

NOT included: proper bootstrapping on time-limit truncation (SB3's
SubprocVecEnv merges terminated/truncated into `dones`).
"""

import os
import time
import random
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from stable_baselines3.common.vec_env import SubprocVecEnv
from policies.DenseGTrXLH import TrXLExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# 
# Base config -- fields common to every benchmark. Subclass this per-env
# and add whatever extra fields that environment needs (world size, maze
# size, phase weights, ...).
# 

@dataclass
class BaseConfig:
    run_id:          str = None
    benchmark_name:  str = "unnamed"   # logged to wandb config + used as a tag

    n_envs:          int = 8

    # TrXL
    features_dim:    int   = 256
    memory_len:      int   = 128
    n_layers:        int   = 4
    n_heads:         int   = 4
    d_ff_multiplier: int   = 2
    dropout:         float = 0.1

    seed:            int = None

    # PPO
    total_timesteps: int   = 2_000_000
    n_steps:         int   = 512
    batch_size:      int   = 256
    n_epochs:        int   = 10
    learning_rate:   float = 1e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_coef:       float = 0.2
    ent_coef:        float = 0.0001
    vf_coef:         float = 0.5
    max_grad_norm:   float = 0.3
    target_kl:       float = 0.03

    # Reward handling: "returns" = divide by running std of discounted
    # returns (no mean subtraction); "none" = use raw rewards.
    reward_norm:         str = "returns"
    # Abort if more than this many minibatch updates per rollout are skipped
    # because of non-finite loss/gradients.
    max_skipped_updates: int = 20

    # Checkpointing
    checkpoint_freq: int = 50_000
    checkpoint_dir:  str = "./checkpoints"
    best_model_dir:  str = "./best_model"

    # Evaluation
    eval_freq:       int = 5_000_000
    n_eval_episodes: int = 5

    # WandB -- NEVER hardcode the key. Export WANDB_API_KEY in your shell
    # (or your job launcher's secrets store) before running; `wandb login`
    # also works and persists it outside your source tree.
    wandb_project:   str = "thesis-drl-trxl"


# 
# Running mean/std (used for scaling rewards by the std of discounted returns)
# 

class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x):
        x           = np.asarray(x, dtype=np.float64)
        batch_mean  = float(np.mean(x))
        batch_var   = float(np.var(x))
        batch_count = x.size

        total      = self.count + batch_count
        delta      = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        self.var   = (
            self.count * self.var + batch_count * batch_var
            + delta ** 2 * self.count * batch_count / total
        ) / total
        self.count = total

    def normalise(self, x, clip=10.0):
        normed = (np.asarray(x) - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -clip, clip).astype(np.float32)


# 
# Actor-Critic (env-agnostic: works off observation_space + action_nvec)
# 

class TrXLActorCritic(nn.Module):
    def __init__(self, observation_space, action_nvec, cfg: BaseConfig, extractor_kwargs=None):
        super().__init__()

        self.extractor = TrXLExtractor(
            observation_space,
            features_dim    = cfg.features_dim,
            memory_len      = cfg.memory_len,
            n_layers        = cfg.n_layers,
            n_heads         = cfg.n_heads,
            d_ff_multiplier = cfg.d_ff_multiplier,
            dropout         = cfg.dropout,
            **(extractor_kwargs or {}),
        )

        self.action_nvec = action_nvec
        self.actor_heads = nn.ModuleList([
            nn.Linear(cfg.features_dim, n) for n in action_nvec
        ])
        self.critic_aggregator = nn.Sequential(
            nn.Linear(cfg.features_dim * 2, cfg.features_dim),
            nn.LayerNorm(cfg.features_dim),
            nn.ReLU(),
        )
        self.critic_head = nn.Linear(cfg.features_dim, 1)

        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _get_critic_features(self, features, memory):
        """
        `memory` must be the memory as it was BEFORE the current observation
        was processed (i.e. the same thing the PPO update passes in as
        memory_override). If unavailable, a zero summary is used so the
        aggregator path is always the one taken.
        """
        if memory is None or memory[-1].shape[0] != features.shape[0]:
            mem_summary = torch.zeros_like(features)
        else:
            mem_summary = memory[-1].mean(dim=1)
        combined = torch.cat([features, mem_summary], dim=-1)
        return self.critic_aggregator(combined)

    def get_value(self, obs, memory_override=None):
        # Pass the current memory as an override so this pass does NOT append
        # the observation to memory (it is only a bootstrap estimate).
        memory   = memory_override if memory_override is not None else self.extractor.memory
        features = self.extractor(obs, memory_override=memory)
        critic_features = self._get_critic_features(features, memory)
        return self.critic_head(critic_features)

    def get_action_and_value(self, obs, action=None, memory_override=None):
        # Capture pre-step memory BEFORE the extractor updates it.
        mem_before = memory_override if memory_override is not None else self.extractor.memory

        features    = self.extractor(obs, memory_override=memory_override)
        logits_list = [head(features) for head in self.actor_heads]
        dists       = [Categorical(logits=l) for l in logits_list]

        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=1)

        log_prob        = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy         = sum(d.entropy() for d in dists)
        critic_features = self._get_critic_features(features, mem_before)
        value           = self.critic_head(critic_features)
        return action, log_prob, entropy, value


# 
# Rollout buffer
# 

class TrXLRolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_space, action_nvec,
                 n_layers, memory_len, d_model, device, gamma, gae_lambda):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.n_layers   = n_layers
        self.memory_len = memory_len
        self.d_model    = d_model
        self.device     = device
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

        self.obs_keys = list(obs_space.spaces.keys())

        self.obs_bufs = {
            k: np.zeros((n_steps, n_envs, *obs_space.spaces[k].shape), dtype=np.float32)
            for k in self.obs_keys
        }
        self.actions    = np.zeros((n_steps, n_envs, len(action_nvec)), dtype=np.int64)
        self.rewards    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones      = np.zeros((n_steps, n_envs), dtype=np.float32)  # dones[t]: transition t ended the episode
        self.values     = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs  = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)

        self.memory_snapshots = [
            torch.zeros(n_steps, n_envs, memory_len, d_model, dtype=torch.float32)
            for _ in range(n_layers)
        ]

    def add_step(self, step, obs_dict, actions, rewards, dones, values, log_probs, memory):
        for k in self.obs_keys:
            self.obs_bufs[k][step] = obs_dict[k]

        self.actions[step]   = actions
        self.rewards[step]   = rewards
        self.dones[step]     = dones
        self.values[step]    = values
        self.log_probs[step] = log_probs

        if memory is not None:
            for layer_idx, layer_mem in enumerate(memory):
                self.memory_snapshots[layer_idx][step] = layer_mem.detach().cpu()

    def compute_gae(self, last_values, last_dones=None):
        # `last_dones` is kept for API compatibility; it equals self.dones[-1]
        # and is therefore already covered by the loop below.
        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]

            delta    = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def normalise_advantages(self):
        adv = self.advantages
        self.advantages = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)

    def get_minibatches(self, batch_size):
        total   = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        flat_obs = {
            k: self.obs_bufs[k].reshape(total, *self.obs_bufs[k].shape[2:])
            for k in self.obs_keys
        }
        flat_actions    = self.actions.reshape(total, -1)
        flat_log_probs  = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns    = self.returns.reshape(total)
        flat_values     = self.values.reshape(total)

        flat_memory = [
            self.memory_snapshots[l].reshape(total, self.memory_len, self.d_model)
            for l in range(self.n_layers)
        ]

        for start in range(0, total, batch_size):
            idx = indices[start: start + batch_size]

            obs_batch = {
                k: torch.tensor(flat_obs[k][idx], dtype=torch.float32).to(self.device)
                for k in self.obs_keys
            }
            memory_batch = [flat_memory[l][idx].to(self.device) for l in range(self.n_layers)]

            yield (
                obs_batch,
                torch.tensor(flat_actions[idx],    dtype=torch.long).to(self.device),
                torch.tensor(flat_log_probs[idx],  dtype=torch.float32).to(self.device),
                torch.tensor(flat_advantages[idx], dtype=torch.float32).to(self.device),
                torch.tensor(flat_returns[idx],    dtype=torch.float32).to(self.device),
                torch.tensor(flat_values[idx],     dtype=torch.float32).to(self.device),
                memory_batch,
            )


# 
# Observation helpers
# 

def vec_obs_to_tensor(obs_dict, device):
    return {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in obs_dict.items()}


def single_obs_to_tensor(obs_dict, device):
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in obs_dict.items()}


# 
# Evaluation
# 

def evaluate(agent, make_env_fn, cfg, device, n_episodes=5):
    eval_env   = make_env_fn(cfg, rank=99)()
    ep_rewards = []

    # Preserve training state: mode + the (n_envs-sized) memory buffers.
    was_training   = agent.training
    saved_memory   = agent.extractor.memory
    saved_segments = agent.extractor._segment_hiddens
    agent.eval()

    try:
        for _ in range(n_episodes):
            obs, _    = eval_env.reset()
            done      = False
            ep_reward = 0.0
            agent.extractor.memory           = None   # re-initialised with B=1 on first forward
            agent.extractor._segment_hiddens = None

            while not done:
                obs_t = single_obs_to_tensor(obs, device)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_t)
                obs, reward, terminated, truncated, _ = eval_env.step(
                    action.squeeze(0).cpu().numpy()
                )
                done       = terminated or truncated
                ep_reward += reward

            ep_rewards.append(ep_reward)
    finally:
        eval_env.close()
        agent.extractor.memory           = saved_memory
        agent.extractor._segment_hiddens = saved_segments
        agent.train(was_training)

    return float(np.mean(ep_rewards))


# 
# Main training loop -- shared by every benchmark script.
# 
# make_env_fn(cfg, rank) -> thunk : identical contract to the wildfire
#     script's make_env_fn, so any env can plug in here.
# extractor_kwargs: forwarded to TrXLExtractor (e.g. use_spatial_bias=False
#     for benchmarks with no world-coordinate signal).
# domain_metrics_fn(info) -> dict: optional, for env-specific scalars
#     logged alongside the (identical across benchmarks) episode/* and
#     train/* keys below.
# 

def train(
    cfg: BaseConfig,
    make_env_fn: Callable,
    extractor_kwargs: Optional[dict] = None,
    domain_metrics_fn: Optional[Callable] = None,
    checkpoint_path: Optional[str] = None,
    device_str: Optional[str] = None,
):
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key  # no-op, just documents the expectation
    wandb.init(project=cfg.wandb_project, config=vars(cfg), tags=[cfg.benchmark_name])

    cfg.run_id = wandb.run.name
    logging.info(f"[Train:{cfg.benchmark_name}] Begin training session with ID: {cfg.run_id}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"[INIT] Device: {device} | N envs: {cfg.n_envs}")

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    envs     = SubprocVecEnv([make_env_fn(cfg, rank=i) for i in range(cfg.n_envs)])
    obs_dict = envs.reset()

    obs_space   = envs.observation_space
    action_nvec = list(envs.action_space.nvec) if hasattr(envs.action_space, "nvec") \
        else [envs.action_space.n]

    agent     = TrXLActorCritic(obs_space, action_nvec, cfg, extractor_kwargs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step      = 0
    best_eval_reward = -np.inf

    if checkpoint_path is not None:
        logging.info(f"[INIT] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step      = ckpt.get("global_step", 0)
        best_eval_reward = ckpt.get("best_eval_reward", -np.inf)
        reward_rms_state = ckpt.get("reward_rms", None)
        recent_rewards   = deque(ckpt.get("recent_rewards", []), maxlen=100)
        next_ckpt_step   = global_step + cfg.checkpoint_freq
        next_eval_step   = global_step + cfg.eval_freq
        logging.info(f"[INIT] Resumed from step {global_step}")
    else:
        recent_rewards   = deque(maxlen=100)
        next_ckpt_step   = cfg.checkpoint_freq
        next_eval_step   = cfg.eval_freq
        reward_rms_state = None

    reward_rms = RunningMeanStd()
    if reward_rms_state is not None:
        reward_rms.mean  = reward_rms_state["mean"]
        reward_rms.var   = reward_rms_state["var"]
        reward_rms.count = reward_rms_state["count"]

    if cfg.reward_norm not in ("returns", "none"):
        raise ValueError(f"cfg.reward_norm must be 'returns' or 'none', got {cfg.reward_norm!r}")
    running_ret = np.zeros(cfg.n_envs, dtype=np.float64)  # discounted return per env

    buffer = TrXLRolloutBuffer(
        n_steps     = cfg.n_steps,
        n_envs      = cfg.n_envs,
        obs_space   = obs_space,
        action_nvec = action_nvec,
        n_layers    = cfg.n_layers,
        memory_len  = cfg.memory_len,
        d_model     = cfg.features_dim,
        device      = device,
        gamma       = cfg.gamma,
        gae_lambda  = cfg.gae_lambda,
    )

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.n_envs, dtype=np.int32)

    scatter_ep_data   = []
    scatter_loss_data = []
    scatter_kl_data   = []
    SCATTER_EP_FREQ   = 50
    episode_count     = 0

    agent.extractor.init_memory(batch_size=cfg.n_envs, device=device)

    logging.info(
        f"[TRAIN:{cfg.benchmark_name}] Starting - {cfg.total_timesteps:,} steps | "
        f"rollout size = {cfg.n_steps * cfg.n_envs:,} transitions | reward_norm={cfg.reward_norm}"
    )
    start_time = time.time()

    while global_step < cfg.total_timesteps:
        agent.eval()

        for step in range(cfg.n_steps):
            obs_t = vec_obs_to_tensor(obs_dict, device)

            with torch.no_grad():
                memory_snapshot = (
                    [m.clone() for m in agent.extractor.memory]
                    if agent.extractor.memory is not None else None
                )
                actions, log_probs, _, values = agent.get_action_and_value(obs_t)

            actions_np   = actions.cpu().numpy()
            values_np    = values.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            next_obs_dict, rewards, dones, infos = envs.step(actions_np)

            # ---- reward scaling (no mean subtraction) ----
            if cfg.reward_norm == "returns":
                running_ret = running_ret * cfg.gamma + rewards
                reward_rms.update(running_ret)
                norm_rewards = np.clip(
                    rewards / np.sqrt(reward_rms.var + 1e-8), -10.0, 10.0
                ).astype(np.float32)
                running_ret[dones.astype(bool)] = 0.0
            else:
                norm_rewards = np.asarray(rewards, dtype=np.float32)

            buffer.add_step(
                step      = step,
                obs_dict  = obs_dict,
                actions   = actions_np,
                rewards   = norm_rewards,
                dones     = dones.astype(np.float32),
                values    = values_np,
                log_probs = log_probs_np,
                memory    = memory_snapshot,
            )

            obs_dict     = next_obs_dict
            global_step += cfg.n_envs
            ep_rewards  += rewards
            ep_lengths  += 1

            done_envs = np.where(dones)[0]
            for env_idx in done_envs:
                recent_rewards.append(float(ep_rewards[env_idx]))
                mean_reward     = np.mean(recent_rewards) if recent_rewards else 0.0
                episode_count  += 1

                info      = infos[env_idx]
                extra_log = domain_metrics_fn(info) if domain_metrics_fn else {}

                wandb.log({
                    "episode/reward":      float(ep_rewards[env_idx]),
                    "episode/length":      int(ep_lengths[env_idx]),
                    "episode/mean_reward": mean_reward,
                    "episode/env_idx":     env_idx,
                    "global_step":         global_step,
                    **extra_log,
                })

                scatter_ep_data.append([int(ep_lengths[env_idx]), float(ep_rewards[env_idx]), int(env_idx)])
                if episode_count % SCATTER_EP_FREQ == 0:
                    wandb.log({
                        "scatter/length_vs_reward": wandb.plot.scatter(
                            wandb.Table(columns=["episode_length", "reward", "env_idx"], data=scatter_ep_data),
                            x="episode_length", y="reward", title="Episode Length vs Reward",
                        ),
                        "global_step": global_step,
                    })
                    scatter_ep_data = []

                agent.extractor.reset_memory([env_idx])
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0

        # Bootstrap value: uses current memory as an override so memory is NOT mutated.
        with torch.no_grad():
            obs_t       = vec_obs_to_tensor(obs_dict, device)
            last_values = agent.get_value(
                obs_t, memory_override=agent.extractor.memory
            ).squeeze(-1).cpu().numpy()

        buffer.compute_gae(last_values=last_values, last_dones=dones)
        buffer.normalise_advantages()   # once per rollout, not per minibatch

        agent.train()
        policy_losses, value_losses, entropies, kl_divs, grad_norms = [], [], [], [], []
        n_skipped = 0
        stop_early = False

        for epoch in range(cfg.n_epochs):
            if stop_early:
                break

            epoch_kls = []

            for (obs_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b, old_values_b,
                 memory_b) in buffer.get_minibatches(cfg.batch_size):

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    obs_b, action=actions_b, memory_override=memory_b,
                )
                new_values = new_values.squeeze(-1)

                log_ratio = new_log_probs - old_log_probs_b
                ratio     = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1    = -advantages_b * ratio
                pg_loss2    = -advantages_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped  = old_values_b + torch.clamp(new_values - old_values_b, -cfg.clip_coef, cfg.clip_coef)
                vf_loss1   = (new_values - returns_b).pow(2)
                vf_loss2   = (v_clipped  - returns_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                # ---- non-finite guards: never let a bad batch reach the optimizer ----
                if not torch.isfinite(loss):
                    n_skipped += 1
                    logging.error(f"[PPO] Non-finite loss (pl={policy_loss.item()}, "
                                  f"vl={value_loss.item()}, ent={entropy_loss.item()}); skipping minibatch")
                    if n_skipped > cfg.max_skipped_updates:
                        raise RuntimeError("Too many non-finite updates in one rollout")
                    continue

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)

                if not torch.isfinite(grad_norm):
                    n_skipped += 1
                    bad = [n for n, p in agent.named_parameters()
                           if p.grad is not None and not torch.isfinite(p.grad).all()]
                    logging.error(f"[PPO] Non-finite grad norm; offending params: {bad}")
                    optimizer.zero_grad()
                    if n_skipped > cfg.max_skipped_updates:
                        raise RuntimeError("Too many non-finite updates in one rollout")
                    continue

                optimizer.step()

                pl, vl, ent = policy_loss.item(), value_loss.item(), entropy_loss.item()
                policy_losses.append(pl)
                value_losses.append(vl)
                entropies.append(ent)
                kl_divs.append(approx_kl)
                epoch_kls.append(approx_kl)
                grad_norms.append(float(grad_norm))

                scatter_loss_data.append([pl, vl])
                scatter_kl_data.append([approx_kl, ent])

            if epoch_kls and np.mean(epoch_kls) > cfg.target_kl:
                logging.info(f"[PPO] Early stop at epoch {epoch + 1}, KL={np.mean(epoch_kls):.4f}")
                stop_early = True

        agent.extractor.memory = [m.detach() for m in agent.extractor.memory]
        if agent.extractor._segment_hiddens is not None:
            agent.extractor._segment_hiddens = [h.detach() for h in agent.extractor._segment_hiddens]

        elapsed = time.time() - start_time
        sps     = global_step / elapsed if elapsed > 0 else 0

        mean_pl  = np.mean(policy_losses) if policy_losses else 0.0
        mean_vl  = np.mean(value_losses)  if value_losses  else 0.0
        mean_ent = np.mean(entropies)     if entropies     else 0.0
        mean_kl  = np.mean(kl_divs)       if kl_divs       else 0.0
        mean_gn  = np.mean(grad_norms)    if grad_norms    else 0.0

        wandb.log({
            "train/policy_loss":     mean_pl,
            "train/value_loss":      mean_vl,
            "train/entropy":         mean_ent,
            "train/approx_kl":       mean_kl,
            "train/grad_norm":       mean_gn,
            "train/skipped_updates": n_skipped,
            "train/steps_per_sec":   sps,
            "scatter/policy_loss_vs_value_loss": wandb.plot.scatter(
                wandb.Table(columns=["policy_loss", "value_loss"], data=scatter_loss_data),
                x="policy_loss", y="value_loss", title="Policy Loss vs Value Loss",
            ),
            "scatter/kl_vs_entropy": wandb.plot.scatter(
                wandb.Table(columns=["approx_kl", "entropy"], data=scatter_kl_data),
                x="approx_kl", y="entropy", title="KL Divergence vs Entropy",
            ),
            "global_step": global_step,
        })
        scatter_loss_data = []
        scatter_kl_data   = []

        logging.info(
            f"[{global_step:>8}] pl={mean_pl:.4f} vl={mean_vl:.4f} "
            f"ent={mean_ent:.4f} kl={mean_kl:.4f} gn={mean_gn:.3f} "
            f"skip={n_skipped} sps={sps:.0f}"
        )

        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.benchmark_name}_{global_step}_steps.pt")
            torch.save({
                "agent":            agent.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "global_step":      global_step,
                "best_eval_reward": best_eval_reward,
                "recent_rewards":   list(recent_rewards),
                "reward_rms": {"mean": reward_rms.mean, "var": reward_rms.var, "count": reward_rms.count},
            }, ckpt_path)
            logging.info(f"[CKPT] Saved: {ckpt_path}")
            next_ckpt_step += cfg.checkpoint_freq

        if global_step >= next_eval_step:
            eval_reward = evaluate(agent, make_env_fn, cfg, device, cfg.n_eval_episodes)
            logging.info(f"[EVAL] step={global_step} mean_reward={eval_reward:.3f}")
            wandb.log({"eval/mean_reward": eval_reward, "global_step": global_step})

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                    "global_step": global_step, "best_eval_reward": best_eval_reward,
                }, os.path.join(cfg.best_model_dir, "best_model.pt"))
                logging.info(f"[EVAL] New best: {best_eval_reward:.3f}")

            # evaluate() restores the training memory, so no init_memory() here.
            next_eval_step += cfg.eval_freq

    torch.save({
        "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "global_step": global_step, "best_eval_reward": best_eval_reward,
    }, f"./{cfg.benchmark_name}_final.pt")
    logging.info("[DONE] Training complete.")
    wandb.finish()
    envs.close()