"""
CleanRL-style standard PPO — 4 parallel environments.

Drop-in baseline for the TrXL version:
  • Same environment, same SubprocVecEnv setup.
  • Same WandB logging (all episode/train/eval/scatter metrics).
  • Same reward normalisation, GAE, checkpointing, evaluation.
  • NO memory: the actor-critic is a plain MLP over the flattened obs.
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
from envs.IsolatedAgent import IsolatedAgentEnv


# 
# Config
# 

@dataclass
class Config:
    run_id:           str   = None

    # Environment
    world_size:       tuple = (512, 512)
    n_agents:         int   = 1
    iter_limit:       int   = 512
    seed:             int   = None
    n_envs:           int   = 8

    # Network
    features_dim:     int   = 256
    hidden_dim:       int   = 512          # MLP hidden width

    # PPO  (mirrors TrXL config)
    total_timesteps:  int   = 2_000_000
    n_steps:          int   = 512
    batch_size:       int   = 256
    n_epochs:         int   = 10
    learning_rate:    float = 1e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2
    ent_coef:         float = 0.0001
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.3
    target_kl:        float = 0.03

    # Env weights
    phase_weights:    dict  = field(default_factory=lambda: {
        "exploration":          1.0,
        "exploration_tracking": 0.05,
        "fire_discovery":       18.8,
        "fire_tracking":        10.5,
        "risk":                 1.5,
    })

    # Checkpointing
    checkpoint_freq:  int   = 50_000
    checkpoint_dir:   str   = "./checkpoints_ppo"
    best_model_dir:   str   = "./best_model_ppo"

    # Evaluation
    eval_freq:        int   = 50_000
    n_eval_episodes:  int   = 5

    # WandB
    wandb_project:    str   = "thesis-drl-trxl"
    wandb_api_key:    str   = "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"


# 
# Running reward normaliser
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
        total       = self.count + batch_count
        delta       = batch_mean - self.mean
        self.mean   = self.mean + delta * batch_count / total
        self.var    = (
            self.count * self.var + batch_count * batch_var
            + delta ** 2 * self.count * batch_count / total
        ) / total
        self.count  = total

    def normalise(self, x, clip=10.0):
        normed = (np.asarray(x) - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(normed, -clip, clip).astype(np.float32)


# 
# MLP feature extractor
# 

class MLPExtractor(nn.Module):
    """
    Flattens the dict observation and passes it through two FC layers.
    Produces a (B, features_dim) embedding.
    """
    def __init__(self, observation_space, features_dim: int, hidden_dim: int):
        super().__init__()
        total_in = sum(
            int(np.prod(space.shape))
            for space in observation_space.spaces.values()
        )
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )
        self.obs_keys = sorted(observation_space.spaces.keys())

    def forward(self, obs: dict) -> torch.Tensor:
        parts = [obs[k].float().flatten(start_dim=1) for k in self.obs_keys]
        x     = torch.cat(parts, dim=-1)
        return self.net(x)


# 
# Actor-Critic
# 

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_nvec, cfg: Config):
        super().__init__()
        self.extractor  = MLPExtractor(observation_space, cfg.features_dim, cfg.hidden_dim)
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


# 
# Rollout buffer  (no memory snapshots)
# 

class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_space, action_nvec,
                 device, gamma, gae_lambda):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.device     = device
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.obs_keys   = list(obs_space.spaces.keys())

        self.obs_bufs = {
            k: np.zeros((n_steps, n_envs, *obs_space.spaces[k].shape), dtype=np.float32)
            for k in self.obs_keys
        }
        self.actions    = np.zeros((n_steps, n_envs, len(action_nvec)), dtype=np.int64)
        self.rewards    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones      = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values     = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs  = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns    = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add_step(self, step, obs_dict, actions, rewards, dones, values, log_probs):
        for k in self.obs_keys:
            self.obs_bufs[k][step] = obs_dict[k]
        self.actions[step]   = actions
        self.rewards[step]   = rewards
        self.dones[step]     = dones
        self.values[step]    = values
        self.log_probs[step] = log_probs

    def compute_gae(self, last_values, last_dones):
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

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            obs_batch = {
                k: torch.tensor(flat_obs[k][idx], dtype=torch.float32).to(self.device)
                for k in self.obs_keys
            }
            yield (
                obs_batch,
                torch.tensor(flat_actions[idx],    dtype=torch.long).to(self.device),
                torch.tensor(flat_log_probs[idx],  dtype=torch.float32).to(self.device),
                torch.tensor(flat_advantages[idx], dtype=torch.float32).to(self.device),
                torch.tensor(flat_returns[idx],    dtype=torch.float32).to(self.device),
                torch.tensor(flat_values[idx],     dtype=torch.float32).to(self.device),
            )


# 
# Observation helpers
# 

def vec_obs_to_tensor(obs_dict, device):
    return {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in obs_dict.items()}

def single_obs_to_tensor(obs_dict, device):
    return {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device) for k, v in obs_dict.items()}


# 
# Environment factory
# 

def make_env_fn(cfg: Config, rank: int):
    def _init():
        env = IsolatedAgentEnv(
            n_agents        = cfg.n_agents,
            world_size      = cfg.world_size,
            start_positions = [(cfg.world_size[0] // 2, cfg.world_size[1] // 2)],
            render_mode     = "rgb_array",
            sample_interval = 5      if rank == 0 else 999999,
            save_interval   = 5      if rank == 0 else 999999,
            seed            = cfg.seed + rank if cfg.seed is not None else None,
            fixed_seed      = False,
            is_vid_out      = (rank == 0),
            vid_id          = f"firescout_ppo_env{rank}",
            vid_base_path   = "/home/s3400220/swarmfire/vids_ppo/",
            phase_weights   = cfg.phase_weights,
            device          = torch.device("cuda:1"),
        )
        return TimeLimit(env, max_episode_steps=cfg.iter_limit)
    return _init


# 
# Evaluation
# 

def evaluate(agent, cfg, device, n_episodes=5):
    eval_env   = make_env_fn(cfg, rank=99)()
    ep_rewards = []
    for _ in range(n_episodes):
        obs, _    = eval_env.reset()
        done      = False
        ep_reward = 0.0
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
    return float(np.mean(ep_rewards))


# 
# Main training loop
# 

def train(cfg: Config, checkpoint_path=None):
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    wandb.init(project=cfg.wandb_project, config=vars(cfg), name=None)
    cfg.run_id = wandb.run.name
    print(f"[Train] PPO | run: {cfg.run_id}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device} | N envs: {cfg.n_envs}")

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    envs    = SubprocVecEnv([make_env_fn(cfg, rank=i) for i in range(cfg.n_envs)])
    obs_dict = envs.reset()

    obs_space   = envs.observation_space
    action_nvec = envs.action_space.nvec.tolist()

    agent     = MLPActorCritic(obs_space, action_nvec, cfg).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    global_step      = 0
    best_eval_reward = -np.inf

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
        recent_rewards   = deque(maxlen=100)
        next_ckpt_step   = cfg.checkpoint_freq
        next_eval_step   = cfg.eval_freq
        reward_rms_state = None

    reward_rms = RunningMeanStd()
    if reward_rms_state is not None:
        reward_rms.mean  = reward_rms_state["mean"]
        reward_rms.var   = reward_rms_state["var"]
        reward_rms.count = reward_rms_state["count"]

    buffer = RolloutBuffer(
        n_steps     = cfg.n_steps,
        n_envs      = cfg.n_envs,
        obs_space   = obs_space,
        action_nvec = action_nvec,
        device      = device,
        gamma       = cfg.gamma,
        gae_lambda  = cfg.gae_lambda,
    )

    ep_rewards = np.zeros(cfg.n_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.n_envs, dtype=np.int32)

    scatter_ep_data:       list = []
    scatter_coverage_data: list = []
    scatter_loss_data:     list = []
    scatter_kl_data:       list = []
    SCATTER_EP_FREQ = 50
    episode_count   = 0

    print(f"[TRAIN] Starting - {cfg.total_timesteps:,} steps | "
          f"rollout = {cfg.n_steps * cfg.n_envs:,} transitions")
    start_time = time.time()

    while global_step < cfg.total_timesteps:

        # ── Rollout ───────────────────────────────────────────────────────────
        agent.eval()
        for step in range(cfg.n_steps):
            obs_t = vec_obs_to_tensor(obs_dict, device)

            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs_t)

            actions_np   = actions.cpu().numpy()
            values_np    = values.squeeze(-1).cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            next_obs_dict, rewards, dones, infos = envs.step(actions_np)

            reward_rms.update(rewards)
            norm_rewards = reward_rms.normalise(rewards)

            buffer.add_step(
                step      = step,
                obs_dict  = obs_dict,
                actions   = actions_np,
                rewards   = norm_rewards,
                dones     = dones.astype(np.float32),
                values    = values_np,
                log_probs = log_probs_np,
            )

            obs_dict     = next_obs_dict
            global_step += cfg.n_envs
            ep_rewards  += rewards
            ep_lengths  += 1

            done_envs = np.where(dones)[0]
            for env_idx in done_envs:
                recent_rewards.append(float(ep_rewards[env_idx]))
                mean_reward    = np.mean(recent_rewards) if recent_rewards else 0.0
                episode_count += 1

                info             = infos[env_idx]
                domain_metrics   = info.get("domain_metrics", {})
                filtered_domain_metrics = {
                    k: v for k, v in domain_metrics.items() if v != cfg.iter_limit
                }

                wandb.log({
                    "episode/reward":      float(ep_rewards[env_idx]),
                    "episode/length":      int(ep_lengths[env_idx]),
                    "episode/mean_reward": mean_reward,
                    "episode/env_idx":     env_idx,
                    "global_step":         global_step,
                    **filtered_domain_metrics,
                })

                scatter_ep_data.append([
                    int(ep_lengths[env_idx]),
                    float(ep_rewards[env_idx]),
                    int(env_idx),
                ])
                if episode_count % SCATTER_EP_FREQ == 0:
                    wandb.log({
                        "scatter/length_vs_reward": wandb.plot.scatter(
                            wandb.Table(
                                columns=["episode_length", "reward", "env_idx"],
                                data=scatter_ep_data,
                            ),
                            x="episode_length", y="reward",
                            title="Episode Length vs Reward",
                        ),
                        "global_step": global_step,
                    })
                    scatter_ep_data = []

                coverage_thresholds = [25, 50, 75, 90, 99]
                for pct in coverage_thresholds:
                    ts = domain_metrics.get(f"domain/fire_coverage_{pct}", -1)
                    if ts != -1:
                        scatter_coverage_data.append([pct, float(ts), int(env_idx)])

                if episode_count % SCATTER_EP_FREQ == 0 and scatter_coverage_data:
                    wandb.log({
                        "scatter/fire_coverage_progression": wandb.plot.scatter(
                            wandb.Table(
                                columns=["coverage_threshold_%", "steps_to_reach", "env_idx"],
                                data=scatter_coverage_data,
                            ),
                            x="coverage_threshold_%", y="steps_to_reach",
                            title="Steps to Reach Fire Coverage Thresholds",
                        ),
                        "global_step": global_step,
                    })
                    scatter_coverage_data = []

                ep_rewards[env_idx] = 0.0
                ep_lengths[env_idx] = 0

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t       = vec_obs_to_tensor(obs_dict, device)
            last_values = agent.get_value(obs_t).squeeze(-1).cpu().numpy()

        buffer.compute_gae(last_values=last_values, last_dones=dones)

        # ── PPO update ────────────────────────────────────────────────────────
        agent.train()
        policy_losses, value_losses, entropies, kl_divs = [], [], [], []
        stop_early = False

        for epoch in range(cfg.n_epochs):
            if stop_early:
                break

            for (obs_b, actions_b, old_log_probs_b,
                 advantages_b, returns_b, old_values_b) in buffer.get_minibatches(cfg.batch_size):

                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-8)

                _, new_log_probs, entropy, new_values = agent.get_action_and_value(
                    obs_b, action=actions_b
                )
                new_values = new_values.squeeze(-1)

                log_ratio  = new_log_probs - old_log_probs_b
                ratio      = log_ratio.exp()
                approx_kl  = ((ratio - 1) - log_ratio).mean().item()

                pg_loss1    = -advantages_b * ratio
                pg_loss2    = -advantages_b * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_clipped  = old_values_b + torch.clamp(new_values - old_values_b, -cfg.clip_coef, cfg.clip_coef)
                vf_loss1   = (new_values - returns_b).pow(2)
                vf_loss2   = (v_clipped  - returns_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pl  = policy_loss.item()
                vl  = value_loss.item()
                ent = entropy_loss.item()

                policy_losses.append(pl)
                value_losses.append(vl)
                entropies.append(ent)
                kl_divs.append(approx_kl)

                scatter_loss_data.append([pl, vl])
                scatter_kl_data.append([approx_kl, ent])

            if kl_divs and np.mean(kl_divs) > cfg.target_kl:
                print(f"[PPO] Early stop at epoch {epoch+1}, KL={np.mean(kl_divs):.4f}")
                stop_early = True

        # ── Rollout logging ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        sps     = global_step / elapsed if elapsed > 0 else 0

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

        print(
            f"[{global_step:>8}] "
            f"pl={mean_pl:.4f} vl={mean_vl:.4f} "
            f"ent={mean_ent:.4f} kl={mean_kl:.4f} "
            f"sps={sps:.0f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"firescout_ppo_{global_step}_steps.pt")
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
            print(f"[CKPT] Saved: {ckpt_path}")
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
                print(f"[EVAL] New best: {best_eval_reward:.3f}")

            next_eval_step += cfg.eval_freq

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save({
        "agent":            agent.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "global_step":      global_step,
        "best_eval_reward": best_eval_reward,
    }, "./firescout_ppo_final.pt")
    print("[DONE] Training complete.")
    wandb.finish()
    envs.close()


# 
# Entry point
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL PPO - FireScout (parallel)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args()
    train(Config(), checkpoint_path=args.checkpoint)