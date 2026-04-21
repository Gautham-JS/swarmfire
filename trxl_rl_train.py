"""
CleanRL-style PPO with TrXL memory snapshot support — 4 parallel environments.

Key changes from single-env version:
  1. SubprocVecEnv runs N envs in separate processes simultaneously.
  2. TrXL memory is now shaped (N, memory_len, d_model) — one slice per env.
  3. Memory snapshots in the buffer store per-env slices at each step.
  4. Episode done handling resets only the finished env's memory slice.
  5. GAE is computed across the (n_steps, N) transition grid.
  6. Minibatch indexing uses flat (step * N + env) indices.
"""

import os
import time
import argparse
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import wandb

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.SingleAgentEnv import SingleAgentEnv
from policies.TrXL import TrXLExtractor


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Environment
    world_size:       tuple = (512, 512)
    n_agents:         int   = 1
    iter_limit:       int   = 500
    seed:             int   = 34
    n_envs:           int   = 4          # ← parallel environments

    # TrXL
    features_dim:     int   = 256
    memory_len:       int   = 128
    n_layers:         int   = 2
    n_heads:          int   = 4
    d_ff_multiplier:  int   = 2
    dropout:          float = 0.1

    # PPO
    total_timesteps:  int   = 5_000_000
    n_steps:          int   = 512        # steps per env per rollout
                                         # total transitions = n_steps * n_envs
    batch_size:       int   = 256        # minibatch size for PPO update
    n_epochs:         int   = 10
    learning_rate:    float = 1e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2
    ent_coef:         float = 0.0001
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.3
    target_kl:        float = 0.03

    # Checkpointing
    checkpoint_freq:  int   = 50_000
    checkpoint_dir:   str   = "./checkpoints"
    best_model_dir:   str   = "./best_model"

    # Evaluation
    eval_freq:        int   = 50_000
    n_eval_episodes:  int   = 5

    # WandB
    wandb_project:    str   = "thesis-drl"
    wandb_api_key:    str   = "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"


# ─────────────────────────────────────────────────────────────────────────────
# Running reward normaliser (one per env, averaged)
# ─────────────────────────────────────────────────────────────────────────────

class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = epsilon

    def update(self, x):
        # x can be a scalar or array of rewards from multiple envs
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


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic (unchanged from single-env version)
# ─────────────────────────────────────────────────────────────────────────────

class TrXLActorCritic(nn.Module):
    def __init__(self, observation_space, action_nvec, cfg: Config):
        super().__init__()

        self.extractor = TrXLExtractor(
            observation_space,
            features_dim    = cfg.features_dim,
            memory_len      = cfg.memory_len,
            n_layers        = cfg.n_layers,
            n_heads         = cfg.n_heads,
            d_ff_multiplier = cfg.d_ff_multiplier,
            dropout         = cfg.dropout,
        )

        self.action_nvec = action_nvec
        self.actor_heads = nn.ModuleList([
            nn.Linear(cfg.features_dim, n) for n in action_nvec
        ])
        self.critic_head = nn.Linear(cfg.features_dim, 1)

        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def get_value(self, obs):
        return self.critic_head(self.extractor(obs))

    def get_action_and_value(self, obs, action=None):
        features    = self.extractor(obs)
        logits_list = [head(features) for head in self.actor_heads]
        dists       = [Categorical(logits=l) for l in logits_list]

        if action is None:
            action = torch.stack([d.sample() for d in dists], dim=1)

        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy  = sum(d.entropy() for d in dists)
        value    = self.critic_head(features)
        return action, log_prob, entropy, value


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer — parallel env version
# ─────────────────────────────────────────────────────────────────────────────

class TrXLRolloutBuffer:
    """
    Stores (n_steps, n_envs) transitions with per-env memory snapshots.

    Layout change from single-env:
      obs_bufs:          (n_steps, n_envs, *obs_shape)
      actions/rewards:   (n_steps, n_envs)
      memory_snapshots:  (n_layers, n_steps, n_envs, memory_len, d_model)

    get_minibatches() flattens the (n_steps * n_envs) dimension and
    shuffles before yielding batches, exactly like CleanRL's reference PPO.
    """

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

        # (n_steps, n_envs, *shape)
        self.obs_bufs = {
            k: np.zeros((n_steps, n_envs, *obs_space.spaces[k].shape), dtype=np.float32)
            for k in self.obs_keys
        }
        self.actions   = np.zeros((n_steps, n_envs, len(action_nvec)), dtype=np.int64)
        self.rewards   = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones     = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)

        # Memory snapshots stored on CPU
        # Shape: (n_layers, n_steps, n_envs, memory_len, d_model)
        self.memory_snapshots = [
            torch.zeros(n_steps, n_envs, memory_len, d_model, dtype=torch.float32)
            for _ in range(n_layers)
        ]

        self.ptr = 0

    def add_step(self, step, obs_dict, actions, rewards, dones, values, log_probs, memory):
        """
        Store one step across all N envs simultaneously.

        obs_dict:  dict of (n_envs, *shape) numpy arrays  — from SubprocVecEnv
        actions:   (n_envs, n_action_dims) numpy array
        rewards:   (n_envs,) numpy array
        dones:     (n_envs,) numpy array
        values:    (n_envs,) numpy array
        log_probs: (n_envs,) numpy array
        memory:    list of n_layers tensors, each (n_envs, memory_len, d_model)
                   — the snapshot BEFORE this step's forward pass
        """
        for k in self.obs_keys:
            self.obs_bufs[k][step] = obs_dict[k]

        self.actions[step]   = actions
        self.rewards[step]   = rewards
        self.dones[step]     = dones
        self.values[step]    = values
        self.log_probs[step] = log_probs

        if memory is not None:
            for layer_idx, layer_mem in enumerate(memory):
                # layer_mem: (n_envs, memory_len, d_model)
                self.memory_snapshots[layer_idx][step] = layer_mem.detach().cpu()

    def compute_gae(self, last_values, last_dones):
        """
        GAE over (n_steps, n_envs) grid.
        last_values: (n_envs,) — value estimates for the state after the rollout
        last_dones:  (n_envs,) — done flags at the rollout boundary
        """
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values       = self.values[t + 1]

            delta              = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae           = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size):
        """
        Flatten (n_steps, n_envs) → (n_steps * n_envs,) then shuffle and yield
        minibatches. Each minibatch includes the memory snapshot from that
        specific (step, env) pair — this is the per-env memory fix.
        """
        total     = self.n_steps * self.n_envs
        indices   = np.random.permutation(total)

        # Pre-flatten everything for fast indexing
        flat_obs = {
            k: self.obs_bufs[k].reshape(total, *self.obs_bufs[k].shape[2:])
            for k in self.obs_keys
        }
        flat_actions   = self.actions.reshape(total, -1)
        flat_log_probs = self.log_probs.reshape(total)
        flat_advantages = self.advantages.reshape(total)
        flat_returns   = self.returns.reshape(total)
        flat_values    = self.values.reshape(total)

        # Memory: (n_layers, n_steps, n_envs, memory_len, d_model)
        #       → (n_layers, n_steps*n_envs, memory_len, d_model)
        flat_memory = [
            self.memory_snapshots[l].reshape(total, self.memory_len, self.d_model)
            for l in range(self.n_layers)
        ]

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]

            obs_batch = {
                k: torch.tensor(flat_obs[k][idx], dtype=torch.float32).to(self.device)
                for k in self.obs_keys
            }
            memory_batch = [
                flat_memory[l][idx].to(self.device)
                for l in range(self.n_layers)
            ]

            yield (
                obs_batch,
                torch.tensor(flat_actions[idx],   dtype=torch.long).to(self.device),
                torch.tensor(flat_log_probs[idx], dtype=torch.float32).to(self.device),
                torch.tensor(flat_advantages[idx],dtype=torch.float32).to(self.device),
                torch.tensor(flat_returns[idx],   dtype=torch.float32).to(self.device),
                torch.tensor(flat_values[idx],    dtype=torch.float32).to(self.device),
                memory_batch,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def vec_obs_to_tensor(obs_dict, device):
    """
    Convert SubprocVecEnv obs dict (n_envs, *shape) numpy → tensors on device.
    No unsqueeze needed — batch dim is already the env dimension.
    """
    return {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in obs_dict.items()
    }


def single_obs_to_tensor(obs_dict, device):
    """Convert single-env obs dict → tensor with batch dim for eval."""
    return {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
        for k, v in obs_dict.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(cfg: Config, rank: int):
    """
    Returns a thunk (zero-argument function) that creates one environment.
    SubprocVecEnv calls these thunks in separate processes.

    Only rank=0 renders and saves videos — all others run silently for speed.
    """
    def _init():
        env = SingleAgentEnv(
            n_agents        = cfg.n_agents,
            world_size      = cfg.world_size,
            start_positions = [(256, 256)],
            render_mode     = "rgb_array" if rank == 0 else "rgb_array",
            sample_interval = 20      if rank == 0 else 999999,
            save_interval   = 20      if rank == 0 else 999999,
            seed            = cfg.seed + rank,   # different seed per env
            fixed_seed      = False,
            is_vid_out      = (rank == 0),
            vid_id          = f"firescout_env{rank}",
            vid_base_path   = "/home/s3400220/swarmfire/vids_parallel/",
            phase_weights   = {
                "exploration":    0.5,
                "exploration_tracking": 0.05,
                "fire_discovery": 16.8,
                "fire_tracking":  1.5,
                "risk":           2.0,
            },
            device=torch.device("cuda:1")
        )
        return TimeLimit(env, max_episode_steps=cfg.iter_limit)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (single env, same as before)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, cfg: Config, device, n_episodes=5):
    eval_env   = make_env_fn(cfg, rank=99)()   # rank=99 → no rendering
    ep_rewards = []

    for _ in range(n_episodes):
        obs, _    = eval_env.reset()
        done      = False
        ep_reward = 0.0
        agent.extractor.memory = None   # fresh memory per episode

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

    eval_env.close()
    agent.extractor.memory = None
    return float(np.mean(ep_rewards))


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config, checkpoint_path=None):

    # ── Setup ─────────────────────────────────────────────────────────────────
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    wandb.init(project=cfg.wandb_project, config=vars(cfg))

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device} | N envs: {cfg.n_envs}")

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Vectorised environment ────────────────────────────────────────────────
    # SubprocVecEnv spawns each env in its own process.
    # All envs step simultaneously — the main process collects results.
    envs = SubprocVecEnv([make_env_fn(cfg, rank=i) for i in range(cfg.n_envs)])
    obs_dict = envs.reset()   # returns dict of (n_envs, *shape) arrays

    obs_space   = envs.observation_space
    action_nvec = envs.action_space.nvec.tolist()   # [3, 3]

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent     = TrXLActorCritic(obs_space, action_nvec, cfg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step      = 0
    best_eval_reward = -np.inf

    # ── Checkpoint loading ────────────────────────────────────────────────────
    if checkpoint_path is not None:
        print(f"[INIT] Loading checkpoint: {checkpoint_path}")
        ckpt             = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step      = ckpt.get("global_step", 0)
        best_eval_reward = ckpt.get("best_eval_reward", -np.inf)
        reward_rms_state = ckpt.get("reward_rms", None)
        recent_rewards   = deque(ckpt.get("recent_rewards", []), maxlen=100)
        next_ckpt_step   = global_step + cfg.checkpoint_freq
        next_eval_step   = global_step + cfg.eval_freq
        print(f"[INIT] Resumed from step {global_step}")
    else:
        recent_rewards = deque(maxlen=100)
        next_ckpt_step = cfg.checkpoint_freq
        next_eval_step = cfg.eval_freq
        reward_rms_state = None

    # ── Reward normaliser ─────────────────────────────────────────────────────
    reward_rms = RunningMeanStd()
    if reward_rms_state is not None:
        reward_rms.mean  = reward_rms_state["mean"]
        reward_rms.var   = reward_rms_state["var"]
        reward_rms.count = reward_rms_state["count"]

    # ── Rollout buffer ────────────────────────────────────────────────────────
    buffer = TrXLRolloutBuffer(
        n_steps    = cfg.n_steps,
        n_envs     = cfg.n_envs,
        obs_space  = obs_space,
        action_nvec = action_nvec,
        n_layers   = cfg.n_layers,
        memory_len = cfg.memory_len,
        d_model    = cfg.features_dim,
        device     = device,
        gamma      = cfg.gamma,
        gae_lambda = cfg.gae_lambda,
    )

    # ── Per-env episode trackers ──────────────────────────────────────────────
    # Each env runs independently — we track reward/length per env separately
    ep_rewards  = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lengths  = np.zeros(cfg.n_envs, dtype=np.int32)

    # Init memory — now batch_size = n_envs, one memory slice per env
    agent.extractor.init_memory(batch_size=cfg.n_envs, device=device)

    print(f"[TRAIN] Starting — {cfg.total_timesteps:,} steps | "
          f"rollout size = {cfg.n_steps * cfg.n_envs:,} transitions")
    start_time = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while global_step < cfg.total_timesteps:

        # ── Rollout collection ────────────────────────────────────────────────
        agent.eval()

        for step in range(cfg.n_steps):
            obs_t = vec_obs_to_tensor(obs_dict, device)

            with torch.no_grad():
                # Snapshot full (n_envs, memory_len, d_model) memory BEFORE forward
                memory_snapshot = (
                    [m.clone() for m in agent.extractor.memory]
                    if agent.extractor.memory is not None else None
                )

                actions, log_probs, _, values = agent.get_action_and_value(obs_t)

            actions_np = actions.cpu().numpy()        # (n_envs, n_dims)
            values_np  = values.squeeze(-1).cpu().numpy()  # (n_envs,)
            log_probs_np = log_probs.cpu().numpy()    # (n_envs,)

            # Step all envs simultaneously — returns (n_envs, *shape) arrays
            next_obs_dict, rewards, dones, infos = envs.step(actions_np)

            # Normalise rewards across all envs together
            reward_rms.update(rewards)
            norm_rewards = reward_rms.normalise(rewards)   # (n_envs,)

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

            obs_dict    = next_obs_dict
            global_step += cfg.n_envs   # n_envs steps happened simultaneously
            ep_rewards  += rewards
            ep_lengths  += 1

            # Handle episode completions per env
            # dones is (n_envs,) bool array from SubprocVecEnv
            done_envs = np.where(dones)[0]
            for env_idx in done_envs:
                recent_rewards.append(float(ep_rewards[env_idx]))
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0

                wandb.log({
                    "episode/reward":      float(ep_rewards[env_idx]),
                    "episode/length":      int(ep_lengths[env_idx]),
                    "episode/mean_reward": mean_reward,
                    "episode/env_idx":     env_idx,
                    "global_step":         global_step,
                })

                # Reset only this env's memory slice — others continue unaffected
                agent.extractor.reset_memory([env_idx])
                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0

        # ── Compute GAE ───────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t      = vec_obs_to_tensor(obs_dict, device)
            last_values = agent.get_value(obs_t).squeeze(-1).cpu().numpy()  # (n_envs,)

        buffer.compute_gae(
            last_values = last_values,
            last_dones  = dones,
        )

        # ── PPO update ────────────────────────────────────────────────────────
        if agent.extractor.memory is not None:
            agent.extractor.memory = [m.detach() for m in agent.extractor.memory]

        agent.train()

        policy_losses, value_losses, entropies, kl_divs = [], [], [], []
        stop_early = False

        for epoch in range(cfg.n_epochs):
            if stop_early:
                break

            for (obs_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b, old_values_b,
                 memory_b) in buffer.get_minibatches(cfg.batch_size):

                # Inject stored per-(step, env) memory snapshot
                # memory_b is a list of (batch_size, memory_len, d_model) tensors
                # — one entry per layer, already on device
                agent.extractor.memory = memory_b

                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    obs_b, action=actions_b
                )
                new_values = new_values.squeeze(-1)

                log_ratio   = new_log_probs - old_log_probs_b
                ratio       = log_ratio.exp()
                approx_kl   = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1    = -advantages_b * ratio
                pg_loss2    = -advantages_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped   = old_values_b + torch.clamp(new_values - old_values_b, -cfg.clip_coef, cfg.clip_coef)
                vf_loss1    = (new_values - returns_b).pow(2)
                vf_loss2    = (v_clipped  - returns_b).pow(2)
                value_loss  = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())
                kl_divs.append(approx_kl)

            # KL early stop checked per epoch
            if kl_divs and np.mean(kl_divs[-cfg.n_steps:]) > cfg.target_kl:
                print(f"[PPO] Early stop at epoch {epoch+1}, KL={np.mean(kl_divs):.4f}")
                stop_early = True

        # Restore live memory for next rollout collection
        # Memory slices that were mid-episode are preserved correctly —
        # only done envs had their slices zeroed during collection
        agent.extractor.memory = None
        agent.extractor.init_memory(batch_size=cfg.n_envs, device=device)

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed  = time.time() - start_time
        sps      = global_step / elapsed if elapsed > 0 else 0

        mean_pl  = np.mean(policy_losses) if policy_losses else 0.0
        mean_vl  = np.mean(value_losses)  if value_losses  else 0.0
        mean_ent = np.mean(entropies)     if entropies      else 0.0
        mean_kl  = np.mean(kl_divs)       if kl_divs        else 0.0

        wandb.log({
            "train/policy_loss":   mean_pl,
            "train/value_loss":    mean_vl,
            "train/entropy":       mean_ent,
            "train/approx_kl":     mean_kl,
            "train/steps_per_sec": sps,
            "global_step":         global_step,
        })

        print(
            f"[{global_step:>8}] "
            f"pl={mean_pl:.4f} vl={mean_vl:.4f} "
            f"ent={mean_ent:.4f} kl={mean_kl:.4f} "
            f"sps={sps:.0f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"firescout_{global_step}_steps.pt"
            )
            torch.save({
                "agent":            agent.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "global_step":      global_step,
                "best_eval_reward": best_eval_reward,
                "recent_rewards":   list(recent_rewards),
                "reward_rms": {
                    "mean":  reward_rms.mean,
                    "var":   reward_rms.var,
                    "count": reward_rms.count,
                },
            }, ckpt_path)
            print(f"[CKPT] Saved → {ckpt_path}")
            next_ckpt_step += cfg.checkpoint_freq

        # ── Evaluation ────────────────────────────────────────────────────────
        if global_step >= next_eval_step:
            eval_reward = evaluate(agent, cfg, device, cfg.n_eval_episodes)
            print(f"[EVAL] step={global_step} mean_reward={eval_reward:.3f}")

            wandb.log({"eval/mean_reward": eval_reward, "global_step": global_step})

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    "agent":            agent.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "global_step":      global_step,
                    "best_eval_reward": best_eval_reward,
                }, os.path.join(cfg.best_model_dir, "best_model.pt"))
                print(f"[EVAL] New best → {best_eval_reward:.3f}")

            # Re-init memory after eval clears it
            agent.extractor.init_memory(batch_size=cfg.n_envs, device=device)
            next_eval_step += cfg.eval_freq

    # ── End ───────────────────────────────────────────────────────────────────
    torch.save({
        "agent":            agent.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "global_step":      global_step,
        "best_eval_reward": best_eval_reward,
    }, "./firescout_final.pt")
    print("[DONE] Training complete.")

    wandb.finish()
    envs.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL TrXL PPO — FireScout (parallel)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    train(cfg, checkpoint_path=args.checkpoint)