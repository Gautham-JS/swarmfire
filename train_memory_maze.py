"""
Memory Maze benchmark for the TrXL agent.

Same wandb metrics (episode/*, train/*, scatter/*, eval/*) as the wildfire
script, via the shared common.ppo_common.train() loop -- set
cfg.wandb_project to the SAME project as your wildfire runs (or leave the
default) and both benchmarks' curves overlay in one dashboard, filterable
by the "benchmark_name" config field / tag.

Usage:
    export WANDB_API_KEY=...          # never hardcode this
    python train_memorymaze.py --maze-size 9x9
    python train_memorymaze.py --checkpoint checkpoints/memory_maze_500000_steps.pt
"""

import argparse
from dataclasses import dataclass

from gymnasium.wrappers import TimeLimit

from train.ppo_common import BaseConfig, train
from envs.MemoryMazeEnv import MemoryMazeEnv


@dataclass
class MemoryMazeConfig(BaseConfig):
    benchmark_name: str   = "memory_maze"
    maze_size:      str   = "9x9"        # 9x9 | 11x11 | 13x13 | 15x15
    iter_limit:     int   = 1000         # TimeLimit steps per episode

    # Memory Maze episodes are shorter-horizon-reward-dense than the
    # wildfire task; a smaller memory window is a reasonable starting
    # point but sweep this per your comparison's needs.
    memory_len:     int   = 128
    n_envs:         int   = 8


def make_env_fn(cfg: MemoryMazeConfig, rank: int):
    """Same contract as the wildfire script's make_env_fn: thunk factory."""
    env_id = f"memory_maze:MemoryMaze-{cfg.maze_size}-v0"

    def _init():
        env = MemoryMazeEnv(env_id=env_id)
        return TimeLimit(env, max_episode_steps=cfg.iter_limit)
    return _init


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrXL PPO - Memory Maze benchmark")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--maze-size", type=str, default="9x9",
                         choices=["9x9", "11x11", "13x13", "15x15"])
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--wandb-project", type=str, default="gtrxlh-mem-maze")
    args = parser.parse_args()

    cfg = MemoryMazeConfig(
        maze_size        = args.maze_size,
        total_timesteps  = args.total_timesteps,
        wandb_project    = args.wandb_project,
    )

    train(
        cfg,
        make_env_fn,
        extractor_kwargs={"use_spatial_bias": False},  # no real position signal here
        checkpoint_path=args.checkpoint,
        device_str="cuda:1"
    )