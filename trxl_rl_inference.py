"""
FireScout TrXL PPO — Inference Script

Loads a checkpoint and runs the agent in a single environment.

Usage:
    python trxl_rl_inference.py --checkpoint ./checkpoints/firescout_500000_steps.pt
    python trxl_rl_inference.py --checkpoint ./best_model/best_model.pt --episodes 10
    python trxl_rl_inference.py --checkpoint ./best_model/best_model.pt --episodes 5 --render
    python trxl_rl_inference.py --checkpoint ./firescout_final.pt --deterministic
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium.wrappers import TimeLimit

from dataclasses import dataclass

from envs.SingleAgentEnv import SingleAgentEnv
from policies.TrXL import TrXLExtractor

"""
    Configuration class, must match training params exactly, preferably load this from a json file and package that with the weights file.
"""
@dataclass
class Config:
    world_size:       tuple = (512, 512)
    n_agents:         int   = 1
    iter_limit:       int   = 1024
    seed                    = None

    # TrXL — must match checkpoint
    features_dim:     int   = 256
    memory_len:       int   = 128
    n_layers:         int   = 2
    n_heads:          int   = 4
    d_ff_multiplier:  int   = 2
    dropout:          float = 0.1


# Actor-Critic (identical to training)
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

    def get_action_and_value(self, obs, action=None, deterministic=False):
        features    = self.extractor(obs)
        logits_list = [head(features) for head in self.actor_heads]
        dists       = [Categorical(logits=l) for l in logits_list]

        if action is None:
            if deterministic:
                action = torch.stack([l.argmax(dim=-1) for l in logits_list], dim=1)
            else:
                action = torch.stack([d.sample() for d in dists], dim=1)

        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        entropy  = sum(d.entropy() for d in dists)
        value    = self.critic_head(features)
        return action, log_prob, entropy, value

    def get_value(self, obs):
        return self.critic_head(self.extractor(obs))


# Helpers
def single_obs_to_tensor(obs_dict, device):
    """Convert a single-env obs dict to tensors with a batch dim."""
    return {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
        for k, v in obs_dict.items()
    }


def make_env(cfg: Config, render: bool = False, seed: int = 0):
    """Create a single evaluation environment."""
    env = SingleAgentEnv(
        n_agents        = cfg.n_agents,
        world_size      = cfg.world_size,
        start_positions = [(cfg.world_size[0] // 2, cfg.world_size[1] // 2)],
        render_mode     = "rgb_array",
        sample_interval = 1 if render else 999999,
        save_interval   = 1 if render else 999999,
        seed            = seed,
        fixed_seed      = False,
        is_vid_out      = render,
        vid_id          = "firescout_inference",
        vid_base_path   = "./inference_vids/",
        phase_weights   = {
            "exploration":          0.5,
            "exploration_tracking": 0.1,
            "fire_discovery":       18.8,
            "fire_tracking":        12.5,
            "risk":                 1.5,
        },
        device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    )
    return TimeLimit(env, max_episode_steps=cfg.iter_limit)


# Main inference loop
def run_inference(checkpoint_path: str, n_episodes: int = 5,
                  deterministic: bool = False, render: bool = False,
                  device_str: str = "auto"):

    cfg = Config()

    if device_str == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[INFERENCE] Device: {device}")

    # Environment (single, for inspection) 
    env = make_env(cfg, render=render, seed=cfg.seed)
    obs, _ = env.reset()

    obs_space   = env.observation_space
    action_nvec = env.action_space.nvec.tolist()

    agent = TrXLActorCritic(obs_space, action_nvec, cfg).to(device)
    agent.eval()

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFERENCE] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    agent.load_state_dict(ckpt["agent"])

    trained_steps = ckpt.get("global_step", "unknown")
    best_reward   = ckpt.get("best_eval_reward", "unknown")
    print(f"[INFERENCE] Checkpoint trained for {trained_steps:,} steps | "
          f"best eval reward: {best_reward}")

    
    # ---- Episodic loop starts here ----
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False

        ep_reward = 0.0
        ep_length = 0

        # Fresh memory for each episode (batch_size=1 for single env inference)
        agent.extractor.init_memory(batch_size=1, device=device)

        while not done:
            
            if ep_length % cfg.memory_len == 0:
                agent.extractor.memory = [m.detach() for m in agent.extractor.memory]

            obs_t = single_obs_to_tensor(obs, device)

            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(
                    obs_t, deterministic=deterministic
                )

            action_np = action.squeeze(0).cpu().numpy()   # (n_action_dims,)

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            ep_reward += reward
            ep_length += 1

            if render:
                print(
                    f"  step={ep_length:4d} | "
                    f"action={action_np.tolist()} | "
                    f"reward={reward:+.4f} | "
                    f"value={value.item():.4f} | "
                    f"log_prob={log_prob.item():.4f}"
                )

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        

        print(
            f"[EP {ep+1:>3}/{n_episodes}] "
            f"reward={ep_reward:+.3f} | "
            f"length={ep_length}"
        )

    print("\n" + "─" * 50)
    print(f"  Episodes      : {n_episodes}")
    print(f"  Mean reward   : {np.mean(episode_rewards):+.3f}")
    print(f"  Std reward    : {np.std(episode_rewards):.3f}")
    print(f"  Min reward    : {np.min(episode_rewards):+.3f}")
    print(f"  Max reward    : {np.max(episode_rewards):+.3f}")
    print(f"  Mean length   : {np.mean(episode_lengths):.1f}")
    print(f"  Deterministic : {deterministic}")
    print("─" * 50)

    env.close()
    return episode_rewards



# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FireScout TrXL PPO — Inference")
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "-n", "--episodes", type=int, default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use argmax instead of sampling from the policy",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering / video output and print per-step info",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to run on: 'auto', 'cpu', 'cuda:0', 'cuda:1', ... (default: auto)",
    )

    args = parser.parse_args()

    run_inference(
        checkpoint_path = args.checkpoint,
        n_episodes      = args.episodes,
        deterministic   = args.deterministic,
        render          = args.render,
        device_str      = args.device,
    )
