import os
import time
import argparse
import random
import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import wandb

import time
import datetime

from gymnasium.wrappers import TimeLimit

from envs.WildfireSingleAgentEnv import SingleAgentEnv
from config.Config import VideoWriterConfig, EnvConfig
from policies.TrXL import TrXLExtractor
from comms.web_sockets.server import WSCommsHandler


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# =============================================================================
# Actor-Critic
# =============================================================================

class TrXLActorCritic(nn.Module):

    def __init__(
        self,
        observation_space,
        action_nvec,
        cfg: EnvConfig
    ):

        super().__init__()

        self.extractor = TrXLExtractor(
            observation_space,
            features_dim=cfg.features_dim,
            memory_len=cfg.memory_len,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_ff_multiplier=cfg.d_ff_multiplier,
            dropout=cfg.dropout,
        )

        self.action_nvec = action_nvec

        self.actor_heads = nn.ModuleList([
            nn.Linear(cfg.features_dim, n)
            for n in action_nvec
        ])

        self.critic_aggregator = nn.Sequential(
            nn.Linear(cfg.features_dim * 2, cfg.features_dim),
            nn.LayerNorm(cfg.features_dim),
            nn.ReLU(),
        )

        self.critic_head = nn.Linear(
            cfg.features_dim,
            1
        )

        for head in self.actor_heads:
            nn.init.orthogonal_(
                head.weight,
                gain=0.01
            )
            nn.init.zeros_(
                head.bias
            )

        nn.init.orthogonal_(
            self.critic_head.weight,
            gain=1.0
        )

        nn.init.zeros_(
            self.critic_head.bias
        )

    def _get_critic_features(
        self,
        features,
        memory_override=None
    ):

        memory = (
            memory_override
            if memory_override is not None
            else self.extractor.memory
        )

        if memory is None:
            return features

        # memory[-1]:
        #
        #     (B, memory_len, D)
        #
        # Mean-pool memory:
        #
        #     (B, D)

        mem_summary = memory[-1].mean(dim=1)

        if mem_summary.shape[0] != features.shape[0]:
            return features

        combined = torch.cat(
            [
                features,
                mem_summary
            ],
            dim=-1
        )

        return self.critic_aggregator(
            combined
        )

    def get_value(
        self,
        obs,
        memory_override=None
    ):

        features = self.extractor(
            obs,
            memory_override=memory_override
        )

        critic_features = self._get_critic_features(
            features,
            memory_override=memory_override
        )

        return self.critic_head(
            critic_features
        )

    def get_action_and_value(
        self,
        obs,
        action=None,
        memory_override=None
    ):

        features = self.extractor(
            obs,
            memory_override=memory_override
        )

        logits_list = [
            head(features)
            for head in self.actor_heads
        ]

        # Deterministic evaluation:
        #
        # Instead of sampling from Categorical distributions,
        # select the action with maximum probability.
        #
        # This makes evaluation reproducible.

        if action is None:

            action = torch.stack(
                [
                    torch.argmax(
                        logits,
                        dim=-1
                    )
                    for logits in logits_list
                ],
                dim=1
            )

        critic_features = self._get_critic_features(
            features,
            memory_override=memory_override
        )

        value = self.critic_head(
            critic_features
        )

        return action, value


# =============================================================================
# Observation conversion
# =============================================================================

def single_obs_to_tensor(
    obs_dict,
    device
):

    return {
        k: torch.tensor(
            v,
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        for k, v in obs_dict.items()
    }


# =============================================================================
# Environment creation
# =============================================================================

def make_eval_env(
    cfg: EnvConfig,
    rank: int = 0,
    is_ue5=False,
):

    video_config = VideoWriterConfig(

        # Enable video recording during evaluation
        is_enabled=False if rank == 0 else False,

        sample_interval=1,

        save_interval=1,

        base_path="./eval_videos/"
    )

    env = SingleAgentEnv(

        world_size=cfg.world_size,

        render_mode="rgb_array",

        seed=cfg.seed,

        video_conf=video_config,

        phase_weights=cfg.phase_weights,

        env_id=cfg.run_id,

        is_gt_visible=False,

        is_recency_obs_disabled=True,

        device=torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    env = TimeLimit(
        env,
        max_episode_steps=cfg.iter_limit
    )

    return env


# =============================================================================
# Memory reset
# =============================================================================

def reset_agent_memory(
    agent
):
    agent.extractor.memory = None
    agent.extractor._segment_hiddens = None


# =============================================================================
# Single evaluation episode
# =============================================================================

def run_episode(
    agent,
    env,
    device,
    episode_idx
):

    logging.info("Calling reset")
    obs, info = env.reset()

    # Critical for recurrent / Transformer-XL evaluation:
    #
    # Each episode must begin with an empty memory.
    reset_agent_memory(agent)

    done = False

    episode_reward = 0.0

    episode_length = 0

    episode_domain_metrics = {}

    model_inference_times = []
    episode_times = []

    while not done:
        episode_start = datetime.datetime.now()
        obs_t = single_obs_to_tensor(
            obs,
            device
        )

        with torch.no_grad():

            action, value = agent.get_action_and_value(
                obs_t
            )

        model_inference_time = datetime.datetime.now()

        action_np = (
            action
            .squeeze(0)
            .cpu()
            .numpy()
        )
        obs, reward, terminated, truncated, info = env.step(
            action_np
        )

        done = (
            terminated
            or truncated
        )

        episode_reward += float(
            reward
        )

        episode_length += 1

        # Keep the latest domain metrics.
        #
        # This assumes the environment provides:
        #
        # info["domain_metrics"]
        #
        # in the same way as the training script.

        episode_domain_metrics = info.get(
            "domain_metrics",
            {}
        )
        episode_end_time = datetime.datetime.now()
        model_inference_times.append((model_inference_time - episode_start).total_seconds())
        episode_times.append((episode_end_time - episode_start).total_seconds())

    logging.info("[EVAL] Episode complete, Mean inference time: {:.3f}s, Mean episode time: {:.3f}s".format(
        np.mean(model_inference_times),
        np.mean(episode_times)
    ))

    return {
        "reward": episode_reward,

        "length": episode_length,

        "domain_metrics": episode_domain_metrics
    }


# =============================================================================
# Aggregate evaluation
# =============================================================================

def evaluate(
    agent,
    env,
    cfg,
    device,
    n_episodes
):

    episode_rewards = []

    episode_lengths = []

    all_domain_metrics = []

    scatter_episode_data = []

    coverage_data = []

    for episode_idx in range(
        n_episodes
    ):
        logging.info(f"Episode {episode_idx + 1}/{n_episodes}")
        episode_start = time.time()

        result = run_episode(
            agent=agent,

            env=env,

            device=device,

            episode_idx=episode_idx
        )

        reward = result["reward"]

        length = result["length"]

        domain_metrics = result[
            "domain_metrics"
        ]

        episode_rewards.append(
            reward
        )

        episode_lengths.append(
            length
        )

        all_domain_metrics.append(
            domain_metrics
        )

        # =============================================================
        # Per-episode W&B logging
        # =============================================================

        wandb_metrics = {

            "eval/episode_reward":
                reward,

            "eval/episode_length":
                length,

            "eval/episode":
                episode_idx,

            "eval/episode_time_sec":
                time.time() - episode_start,

            "eval/domain/fire_revisit_count":
                domain_metrics.get(
                    "domain/revisit_count",
                    0
                ),

            "eval/domain/fire_revisit_mean_delta":
                domain_metrics.get(
                    "domain/revisit_delta_mean",
                    0.0
                ),

            "eval/domain/fire_revisit_max_delta":
                domain_metrics.get(
                    "domain/revisit_delta_max",
                    0.0
                ),

            "eval/domain/fire_revisit_min_delta":
                domain_metrics.get(
                    "domain/revisit_delta_min",
                    0.0
                )
        }

        # =============================================================
        # Coverage progression
        # =============================================================

        coverage_thresholds = [
            25,
            50,
            75,
            90,
            99
        ]

        for pct in coverage_thresholds:

            steps_to_reach = domain_metrics.get(
                f"domain/fire_coverage_{pct}",
                -1
            )

            if steps_to_reach != -1:

                coverage_data.append(
                    [
                        pct,

                        float(
                            steps_to_reach
                        ),

                        episode_idx
                    ]
                )

                wandb_metrics[
                    f"eval/domain/fire_coverage_{pct}"
                ] = steps_to_reach

        wandb.log(
            wandb_metrics
        )

        # =============================================================
        # Episode scatter data
        # =============================================================

        scatter_episode_data.append(
            [
                length,

                reward,

                episode_idx
            ]
        )

        logging.info(

            f"[EVAL] "

            f"Episode "
            f"{episode_idx + 1}/"
            f"{n_episodes} | "

            f"Reward: "
            f"{reward:.3f} | "

            f"Length: "
            f"{length}"
        )

    # =========================================================================
    # Aggregate metrics
    # =========================================================================

    mean_reward = float(
        np.mean(
            episode_rewards
        )
    )

    std_reward = float(
        np.std(
            episode_rewards
        )
    )

    min_reward = float(
        np.min(
            episode_rewards
        )
    )

    max_reward = float(
        np.max(
            episode_rewards
        )
    )

    mean_length = float(
        np.mean(
            episode_lengths
        )
    )

    std_length = float(
        np.std(
            episode_lengths
        )
    )

    # =========================================================================
    # Aggregate domain metrics
    # =========================================================================

    revisit_counts = [
        metrics.get(
            "domain/revisit_count",
            0
        )

        for metrics in all_domain_metrics
    ]

    revisit_mean_deltas = [
        metrics.get(
            "domain/revisit_delta_mean",
            0.0
        )

        for metrics in all_domain_metrics
    ]

    revisit_max_deltas = [
        metrics.get(
            "domain/revisit_delta_max",
            0.0
        )

        for metrics in all_domain_metrics
    ]

    revisit_min_deltas = [
        metrics.get(
            "domain/revisit_delta_min",
            0.0
        )

        for metrics in all_domain_metrics
    ]

    aggregate_metrics = {

        "eval/mean_reward":
            mean_reward,

        "eval/std_reward":
            std_reward,

        "eval/min_reward":
            min_reward,

        "eval/max_reward":
            max_reward,

        "eval/mean_episode_length":
            mean_length,

        "eval/std_episode_length":
            std_length,

        "eval/mean_fire_revisit_count":
            float(
                np.mean(
                    revisit_counts
                )
            ),

        "eval/mean_fire_revisit_delta":
            float(
                np.mean(
                    revisit_mean_deltas
                )
            ),

        "eval/mean_fire_revisit_max_delta":
            float(
                np.mean(
                    revisit_max_deltas
                )
            ),

        "eval/mean_fire_revisit_min_delta":
            float(
                np.mean(
                    revisit_min_deltas
                )
            )
    }

    # =========================================================================
    # W&B scatter plots
    # =========================================================================

    if len(
        scatter_episode_data
    ) > 0:

        aggregate_metrics[
            "eval/scatter/length_vs_reward"
        ] = wandb.plot.scatter(

            wandb.Table(

                columns=[
                    "episode_length",
                    "reward",
                    "episode_idx"
                ],

                data=scatter_episode_data
            ),

            x="episode_length",

            y="reward",

            title=(
                "Evaluation: "
                "Episode Length vs Reward"
            )
        )

    if len(
        coverage_data
    ) > 0:

        aggregate_metrics[
            "eval/scatter/fire_coverage_progression"
        ] = wandb.plot.scatter(

            wandb.Table(

                columns=[
                    "coverage_threshold_%",
                    "steps_to_reach",
                    "episode_idx"
                ],

                data=coverage_data
            ),

            x="coverage_threshold_%",

            y="steps_to_reach",

            title=(
                "Evaluation: "
                "Steps to Reach Fire Coverage"
            )
        )

    wandb.log(
        aggregate_metrics
    )

    # =========================================================================
    # Print summary
    # =========================================================================

    logging.info(
        "\n"
        "==============================\n"
        "EVALUATION COMPLETE\n"
        "==============================\n"
        f"Episodes       : {n_episodes}\n"
        f"Mean reward    : {mean_reward:.4f}\n"
        f"Std reward     : {std_reward:.4f}\n"
        f"Min reward     : {min_reward:.4f}\n"
        f"Max reward     : {max_reward:.4f}\n"
        f"Mean length    : {mean_length:.2f}\n"
        f"Std length     : {std_length:.2f}\n"
        "=============================="
    )

    return aggregate_metrics


# =============================================================================
# Main evaluation function
# =============================================================================

def run_evaluation(
    checkpoint_path,
    cfg,
    n_episodes,
    is_ue5=False,
):

    # =========================================================================
    # Reproducibility
    # =========================================================================

    if cfg.seed is not None:

        random.seed(
            cfg.seed
        )

        np.random.seed(
            cfg.seed
        )

        torch.manual_seed(
            cfg.seed
        )

    # =========================================================================
    # Device
    # =========================================================================

    device = torch.device(

        "cuda:1"

        if torch.cuda.is_available()

        else "cpu"
    )

    logging.info(
        f"[INIT] Device: {device}"
    )

    # =========================================================================
    # W&B
    # =========================================================================

    os.environ[
        "WANDB_API_KEY"
    ] = cfg.wandb_api_key

    wandb.init(

        project=cfg.wandb_project,

        name=(
            f"eval_"
            f"{os.path.basename(checkpoint_path)}"
        ),

        config={

            **vars(cfg),

            "checkpoint_path":
                checkpoint_path,

            "n_eval_episodes":
                n_episodes,

            "evaluation":
                True
        },

        tags=[
            "evaluation",
            "checkpoint"
        ]
    )

    cfg.run_id = wandb.run.name

    logging.info(
        f"[INIT] W&B run: "
        f"{wandb.run.name}"
    )

    if is_ue5:
        logging.info("[INIT] UE5 evaluation mode enabled")
        sleep_time = 10
        while sleep_time > 0:
            logging.info(
                f"[INIT] Waiting {sleep_time} seconds for UE5 client to connect..."
            )
            time.sleep(1)
            sleep_time -= 1

    if WSCommsHandler.instance().is_clients_connected():
        logging.info("[INIT] UE5 client connected")
    else:
        logging.warning(
            "[INIT] No UE5 client connected. "
            "Evaluation will proceed without UE5 integration."
        )


    # =========================================================================
    # Environment
    # =========================================================================

    env = make_eval_env(
        cfg
    )

    # =========================================================================
    # Agent
    # =========================================================================

    obs_space = env.observation_space

    action_nvec = (
        env.action_space.nvec.tolist()
    )

    agent = TrXLActorCritic(

        observation_space=obs_space,

        action_nvec=action_nvec,

        cfg=cfg
    ).to(device)

    # =========================================================================
    # Load checkpoint
    # =========================================================================

    logging.info(
        f"[INIT] Loading checkpoint:\n"
        f"{checkpoint_path}"
    )

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device
    )

    # Training checkpoints:
    #
    #     checkpoint["agent"]
    #
    # Best model checkpoints:
    #
    #     checkpoint["agent"]
    #
    # So the same loading code works for both.

    agent.load_state_dict(
        checkpoint["agent"]
    )

    logging.info(
        "Successfully loaded agent weights"
    )

    # =========================================================================
    # Evaluation mode
    # =========================================================================

    agent.eval()

    reset_agent_memory(
        agent
    )

    # =========================================================================
    # Run evaluation
    # =========================================================================
    logging.info(
        "Starting evaluation loop"
    )
    evaluate(

        agent=agent,

        env=env,

        cfg=cfg,

        device=device,

        n_episodes=n_episodes
    )

    # =========================================================================
    # Cleanup
    # =========================================================================

    reset_agent_memory(
        agent
    )

    env.close()

    wandb.finish()

    logging.info(
        "[DONE] Evaluation complete."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--checkpoint",

        type=str,

        required=True,

        help=(
            "Path to the trained "
            "PyTorch checkpoint."
        )
    )

    parser.add_argument(

        "--episodes",

        type=int,

        default=100,

        help=(
            "Number of evaluation "
            "episodes."
        )
    )

    return parser.parse_args()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    args = parse_args()

    # =========================================================================
    # IMPORTANT:
    #
    # This should be the same EnvConfig used during training.
    #
    # Replace this with however your project constructs EnvConfig.
    # =========================================================================

    cfg = EnvConfig()

    run_evaluation(

        checkpoint_path=args.checkpoint,

        cfg=cfg,

        n_episodes=args.episodes
    )