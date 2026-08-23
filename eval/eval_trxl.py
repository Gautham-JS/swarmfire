import os
import time
import copy
import functools
import argparse
import random
import logging
import multiprocessing as mp
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import wandb

import time
import datetime

from gymnasium.wrappers import TimeLimit

from envs.WildfireSingleAgentEnv import SingleAgentEnv
from envs.RedisSingleAgentEnv import RedisRenderedEnv
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
            use_spatial_bias=cfg.is_spatial_bias
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
    rank: int = -1,
    is_ue5=False,
):
    video_config = VideoWriterConfig(
        # Enable video recording during evaluation
        is_enabled=True if rank == 0 else False,
        sample_interval=1,
        save_interval=1,
        base_path="./eval_videos/"
    )
    # env = SingleAgentEnv(
    #     world_size              = cfg.world_size,
    #     render_mode             = "rgb_array",
    #     seed                    = cfg.seed,
    #     video_conf              = video_config,
    #     phase_weights           = cfg.phase_weights,
    #     env_id                  = f"{cfg.run_id}_{cfg.world_size[0]}",
    #     is_gt_visible           = False,
    #     is_recency_obs_disabled = True,
    #     device                  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # )
    env = RedisRenderedEnv(
        world_size              = cfg.world_size,
        render_mode             = "rgb_array",
        seed                    = cfg.seed,
        video_conf              = video_config,
        phase_weights           = cfg.phase_weights,
        env_id                  = f"{cfg.run_id}_{cfg.world_size[0]}",
        is_gt_visible           = False,
        is_recency_obs_disabled = False,
        redis_host               = "localhost",
        redis_port               = 8090,
        redis_channel_prefix      = "eval_run",
        device                  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        model_inference_times.append((model_inference_time - episode_start).total_seconds() * 1000)
        episode_times.append((episode_end_time - episode_start).total_seconds() * 1000)

    logging.info("[EVAL] Episode complete, Mean inference time: {:.3f}ms, Mean episode time: {:.3f}ms".format(
        np.mean(model_inference_times),
        np.mean(episode_times)
    ))

    return {
        "reward": episode_reward,
        "length": episode_length,
        "domain_metrics": episode_domain_metrics,
        "mean_inference_time_ms": np.mean(model_inference_times),
        "mean_episode_time_ms": np.mean(episode_times)
    }


# =============================================================================
# Batched (vectorized) parallel episodes
#
# The neural net forward pass is batched (one call covers all active
# envs), but that alone doesn't parallelize the environment simulation
# itself -- an in-process Python for-loop over env.step() calls still
# runs them one after another. Since the fire simulation is the
# expensive part (CPU-bound, not the ~10ms forward pass), each env's
# step()/reset() is instead executed in its own OS process via
# SubprocVectorEnv, so the simulations genuinely run concurrently
# across CPU cores.
# =============================================================================

def batch_obs_to_tensor(
    obs_list,
    device
):
    """
    Stack a list of per-env observation dicts (each unbatched, as
    returned by env.reset()/env.step()) into a single batched dict of
    tensors with shape (B, ...).
    """

    keys = obs_list[0].keys()

    return {
        k: torch.tensor(
            np.stack(
                [obs[k] for obs in obs_list],
                axis=0
            ),
            dtype=torch.float32
        ).to(device)

        for k in keys
    }


def _vector_env_worker(
    remote,
    parent_remote,
    env_fn,
):
    """
    Runs in a dedicated subprocess: owns exactly one persistent env
    instance and services "step"/"reset"/"close" commands sent over
    the pipe. Kept alive for the whole SubprocVectorEnv lifetime so we
    only pay process-startup cost once, not per batch/chunk.
    """

    parent_remote.close()

    env = env_fn()

    try:
        while True:

            cmd, data = remote.recv()

            if cmd == "step":

                obs, reward, terminated, truncated, info = env.step(data)

                remote.send(
                    (obs, float(reward), bool(terminated), bool(truncated), info)
                )

            elif cmd == "reset":

                obs, info = env.reset(seed=data)

                remote.send((obs, info))

            elif cmd == "close":

                env.close()
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown command: {cmd}")

    except EOFError:
        pass


class SubprocVectorEnv:
    """
    Owns `n_envs` env instances, each running in its own subprocess,
    so step()/reset() calls execute in parallel across CPU cores
    rather than sequentially in the calling process.

    Unlike gymnasium's built-in vector envs, this does NOT auto-reset
    finished envs on the next step() call -- callers explicitly choose
    which env indices to step or reset each call via `indices`, which
    keeps the "stop touching a finished episode's env" logic explicit
    and avoids relying on framework-specific autoreset semantics.
    """

    def __init__(self, env_fns):

        self.n_envs = len(env_fns)

        # "spawn" avoids re-using a CUDA context that may already be
        # initialized in the parent process (the agent lives on its
        # own device) -- forking after CUDA init is unsafe.
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.n_envs)]
        )

        self.processes = []

        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):

            process = ctx.Process(
                target=_vector_env_worker,
                args=(work_remote, remote, env_fn),
                daemon=True,
            )

            process.start()

            self.processes.append(process)

            # The child holds its own copy of work_remote; close the
            # parent's reference to it.
            work_remote.close()

    def reset(self, indices, seeds):

        for idx, seed in zip(indices, seeds):
            self.remotes[idx].send(("reset", seed))

        results = [self.remotes[idx].recv() for idx in indices]

        obs_list, info_list = zip(*results)

        return list(obs_list), list(info_list)

    def step(self, indices, actions):
        """
        Steps only the envs at `indices` (in that order) with the
        corresponding `actions`. All `indices` workers process their
        step() concurrently -- wall time is roughly the slowest single
        step, not the sum of all of them.
        """

        for idx, action in zip(indices, actions):
            self.remotes[idx].send(("step", action))

        return [self.remotes[idx].recv() for idx in indices]

    def close(self):

        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass

        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def run_episode_batch(
    agent,
    vec_env,
    device,
    current_batch_size,
    seeds,
):
    """
    Run `current_batch_size` episodes concurrently using an already-
    running `vec_env` process pool.

    The policy forward pass covers the full `current_batch_size` every
    step (padding finished envs' last observation forward, output
    discarded) so the TrXL memory's batch dimension stays constant
    for the whole rollout. The environment simulation itself, however,
    is only invoked for envs still running, and executes concurrently
    across their subprocesses -- this is the part that was previously
    the sequential bottleneck.

    Returns a list of `current_batch_size` result dicts with the same
    shape as run_episode()'s return value.
    """

    indices = list(range(current_batch_size))

    obs_list, _ = vec_env.reset(indices, seeds)

    # Fresh memory for this whole batch of new episodes.
    reset_agent_memory(agent)

    dones = [False] * current_batch_size

    episode_rewards = [0.0] * current_batch_size
    episode_lengths = [0] * current_batch_size
    episode_domain_metrics = [{} for _ in range(current_batch_size)]

    model_inference_times = [[] for _ in range(current_batch_size)]
    episode_times = [[] for _ in range(current_batch_size)]

    while not all(dones):

        step_start = datetime.datetime.now()

        obs_t = batch_obs_to_tensor(
            obs_list,
            device
        )

        with torch.no_grad():

            actions, values = agent.get_action_and_value(
                obs_t
            )

        model_inference_time = datetime.datetime.now()

        actions_np = actions.cpu().numpy()

        active_indices = [
            i for i in range(current_batch_size) if not dones[i]
        ]

        active_actions = [
            actions_np[i] for i in active_indices
        ]

        # The actual simulation for every still-running episode
        # happens concurrently, one per subprocess.
        step_results = vec_env.step(active_indices, active_actions)

        step_end = datetime.datetime.now()

        for local_i, env_idx in enumerate(active_indices):

            obs_i, reward, terminated, truncated, info = step_results[local_i]

            obs_list[env_idx] = obs_i

            episode_rewards[env_idx] += float(reward)
            episode_lengths[env_idx] += 1
            episode_domain_metrics[env_idx] = info.get("domain_metrics", {})

            model_inference_times[env_idx].append(
                (model_inference_time - step_start).total_seconds() * 1000
            )
            episode_times[env_idx].append(
                (step_end - step_start).total_seconds() * 1000
            )

            if terminated or truncated:
                dones[env_idx] = True

    reset_agent_memory(agent)

    results = []

    for i in range(current_batch_size):

        results.append({
            "reward": episode_rewards[i],
            "length": episode_lengths[i],
            "domain_metrics": episode_domain_metrics[i],
            "mean_inference_time_ms": (
                float(np.mean(model_inference_times[i]))
                if model_inference_times[i] else 0.0
            ),
            "mean_episode_time_ms": (
                float(np.mean(episode_times[i]))
                if episode_times[i] else 0.0
            ),
        })

    logging.info(
        "[EVAL] Batch of {} episodes complete, Mean inference time: {:.3f}ms, Mean episode time: {:.3f}ms".format(
            current_batch_size,
            np.mean([r["mean_inference_time_ms"] for r in results]),
            np.mean([r["mean_episode_time_ms"] for r in results]),
        )
    )

    return results


# =============================================================================
# Aggregate evaluation
# =============================================================================

def evaluate(
    agent,
    env,
    cfg,
    device,
    n_episodes,
    metric_prefix="eval",
    world_size=None,
    velocity=None,
    n_parallel=1,
    is_ue5=False,
):
    """
    Run `n_episodes` evaluation episodes and log results to W&B.

    metric_prefix:
        Namespace used for every per-episode/per-batch W&B key logged
        in this call (e.g. "eval" for a normal run, or
        "eval/world_32" when this call is one point in a world-size
        sweep so each world size gets its own set of time series in
        W&B rather than all sweep points being interleaved on a
        single line). Note this only keeps the *series* visually
        separate -- their X axis is still wandb's internal step
        counter, not world_size itself. See the "world_size-indexed
        logging" block below for the actual world_size-on-X-axis
        charts.

    world_size:
        Optional int, purely for logging/readability. If provided, it
        is included in the printed summary and as a scalar in the
        returned aggregate metrics dict (under "<prefix>/world_size").
        It ALSO triggers a second, additional wandb.log() call that
        writes every scalar aggregate metric (already averaged across
        the batch) under a fixed "sweep_eval/" prefix, together with
        "sweep_eval/world_size". run_world_size_sweep() registers
        "sweep_eval/world_size" as the step_metric for "sweep_eval/*"
        via wandb.define_metric, so those charts plot natively against
        world_size, with exactly one point per evaluated world size
        (batch) -- which is what actually answers "how does
        performance scale with world size".

    n_parallel:
        Number of episodes to run concurrently via batched/vectorized
        stepping. 1 (default) reproduces the original sequential
        behavior using the pre-built `env`. Values > 1 ignore `env`
        and instead build `n_parallel` fresh env instances per batch
        from `cfg`, running `n_episodes` in ceil(n_episodes /
        n_parallel) batches.
    """

    episode_rewards = []

    episode_lengths = []

    all_domain_metrics = []

    scatter_episode_data = []

    coverage_data = []

    if world_size is not None:
        logging.info(
            f"[EVAL] Starting evaluation for world_size={world_size}"
        )
    if velocity is not None:
        logging.info(
            f"[EVAL] Starting evaluation for velocity={velocity}"
        )

    vec_env = None

    if n_parallel > 1:

        env_fns = [
            functools.partial(
                make_eval_env,
                copy.copy(cfg),
                rank=i,
                is_ue5=is_ue5,
            )
            for i in range(n_parallel)
        ]

        vec_env = SubprocVectorEnv(env_fns)

    episode_idx = 0

    try:

        while episode_idx < n_episodes:

            # =====================================================================
            # Produce the next batch of episode results, either sequentially
            # (n_parallel <= 1, using the pre-built `env`) or by running
            # `current_batch_size` episodes concurrently -- one env per
            # subprocess -- with a single batched forward pass per step
            # (n_parallel > 1).
            # =====================================================================

            if n_parallel <= 1:

                logging.info(f"Episode {episode_idx + 1}/{n_episodes}")

                episode_wall_start = time.time()

                result = run_episode(
                    agent=agent,
                    env=env,
                    device=device,
                    episode_idx=episode_idx
                )

                batch_results = [result]
                batch_wall_times = [time.time() - episode_wall_start]

            else:

                current_batch_size = min(
                    n_parallel,
                    n_episodes - episode_idx
                )

                logging.info(
                    f"[EVAL] Running batch of {current_batch_size} parallel "
                    f"episodes ({episode_idx + 1}-"
                    f"{episode_idx + current_batch_size}/{n_episodes})"
                )

                seeds = [
                    (
                        cfg.seed + episode_idx + i
                        if cfg.seed is not None
                        else None
                    )
                    for i in range(current_batch_size)
                ]

                batch_wall_start = time.time()

                batch_results = run_episode_batch(
                    agent=agent,
                    vec_env=vec_env,
                    device=device,
                    current_batch_size=current_batch_size,
                    seeds=seeds,
                )

                batch_wall_elapsed = time.time() - batch_wall_start

                # Individual episode wall-clock time isn't well-defined
                # when episodes run concurrently -- split the batch's
                # total wall time evenly as an approximation.
                batch_wall_times = [
                    batch_wall_elapsed / current_batch_size
                ] * current_batch_size

            # =====================================================================
            # Per-episode processing (identical regardless of how the
            # batch above was produced).
            # =====================================================================

            for result, episode_wall_time in zip(batch_results, batch_wall_times):

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

                    f"{metric_prefix}/episode_reward":
                        reward,

                    f"{metric_prefix}/episode_length":
                        length,

                    f"{metric_prefix}/episode":
                        episode_idx,

                    f"{metric_prefix}/episode_time_sec":
                        episode_wall_time,

                    f"{metric_prefix}/domain/fire_revisit_count":
                        domain_metrics.get(
                            "domain/revisit_count",
                            0
                        ),

                    f"{metric_prefix}/domain/fire_revisit_mean_delta":
                        domain_metrics.get(
                            "domain/revisit_delta_mean",
                            0.0
                        ),

                    f"{metric_prefix}/domain/fire_revisit_max_delta":
                        domain_metrics.get(
                            "domain/revisit_delta_max",
                            0.0
                        ),

                    f"{metric_prefix}/domain/fire_revisit_min_delta":
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
                            f"{metric_prefix}/domain/fire_coverage_{pct}"
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

                episode_idx += 1

    finally:

        if vec_env is not None:
            vec_env.close()

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

        f"{metric_prefix}/mean_reward":
            mean_reward,

        f"{metric_prefix}/std_reward":
            std_reward,

        f"{metric_prefix}/min_reward":
            min_reward,

        f"{metric_prefix}/max_reward":
            max_reward,

        f"{metric_prefix}/mean_episode_length":
            mean_length,

        f"{metric_prefix}/std_episode_length":
            std_length,

        f"{metric_prefix}/mean_fire_revisit_count":
            float(
                np.mean(
                    revisit_counts
                )
            ),

        f"{metric_prefix}/mean_fire_revisit_delta":
            float(
                np.mean(
                    revisit_mean_deltas
                )
            ),

        f"{metric_prefix}/mean_fire_revisit_max_delta":
            float(
                np.mean(
                    revisit_max_deltas
                )
            ),

        f"{metric_prefix}/mean_fire_revisit_min_delta":
            float(
                np.mean(
                    revisit_min_deltas
                )
            )
    }

    if world_size is not None:
        aggregate_metrics[f"{metric_prefix}/world_size"] = world_size
    if velocity is not None:
        aggregate_metrics[f"{metric_prefix}/velocity"] = velocity

    # =========================================================================
    # World-size-indexed logging
    #
    # Everything above (per-episode wandb_metrics, and aggregate_metrics
    # keyed by metric_prefix) is namespaced per world size so the series
    # don't visually collide, but its X axis is still wandb's internal
    # step counter -- not world_size. This block additionally logs each
    # (already-averaged) scalar aggregate metric once per batch under a
    # fixed "sweep_eval/" prefix, together with "sweep_eval/world_size".
    # run_world_size_sweep() registers "sweep_eval/world_size" as the
    # step_metric for "sweep_eval/*", so wandb plots these natively
    # against world_size -- one point per evaluation batch, as requested.
    # =========================================================================

    if world_size is not None:

        sweep_metrics = {
            "sweep_eval/world_size": world_size
        }

        for key, value in aggregate_metrics.items():

            if isinstance(value, (int, float)) and not isinstance(value, bool):

                clean_key = key.split("/")[-1]

                if clean_key == "world_size":
                    continue

                sweep_metrics[f"sweep_eval/{clean_key}"] = value

        wandb.log(
            sweep_metrics
        )

    if velocity is not None:
        sweep_metrics = {
            "sweep_eval/velocity": velocity
        }
        for key, value in aggregate_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                clean_key = key.split("/")[-1]
                if clean_key == "velocity":
                    continue
                sweep_metrics[f"sweep_eval/{clean_key}"] = value

        wandb.log(
            sweep_metrics
        )

    # =========================================================================
    # W&B scatter plots
    # =========================================================================

    if len(
        scatter_episode_data
    ) > 0:

        aggregate_metrics[
            f"{metric_prefix}/scatter/length_vs_reward"
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
            f"{metric_prefix}/scatter/fire_coverage_progression"
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

    world_size_line = (
        f"World size     : {world_size}\n"
        if world_size is not None
        else ""
    )

    logging.info(
        "\n"
        "==============================\n"
        "EVALUATION COMPLETE\n"
        "==============================\n"
        f"{world_size_line}"
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
    n_parallel=1,
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
    #
    # When running batched/parallel evaluation (n_parallel > 1), this
    # env instance is only used to read observation/action space
    # shapes for building the agent -- evaluate() builds its own
    # n_parallel env instances per batch internally. It's closed
    # afterward instead of being reused, to avoid an unused open env
    # (and, in UE5 mode, an unused client connection) sitting around
    # during the run.
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

    if n_parallel > 1:
        env.close()
        env = None

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

        n_episodes=n_episodes,

        n_parallel=n_parallel,

        is_ue5=is_ue5,
    )

    # =========================================================================
    # Cleanup
    # =========================================================================

    reset_agent_memory(
        agent
    )

    if env is not None:
        env.close()

    wandb.finish()

    logging.info(
        "[DONE] Evaluation complete."
    )



def run_velocity_scale_sweep(
    checkpoint_path,
    cfg,
    n_episodes,
    vel_start,
    vel_end,
    step_size=16,
    is_ue5=False,
    n_parallel=1,
):
    """
    Evaluate the same checkpoint across a range of square world sizes,
    from (world_size_start, world_size_start) to
    (world_size_end, world_size_end) inclusive, in increments of
    `step_size`, and log reward / length / domain metrics against
    world size to W&B.

    A single W&B run is used for the whole sweep:
      - Each world size gets its own namespaced time series
        (eval/world_<N>/...) so per-episode curves for different
        world sizes don't get interleaved on the same line. (This was
        previously broken -- the call into evaluate() passed a static
        metric_prefix of "eval/" for every world size instead of
        interpolating world_size_n into it, so every world size's
        per-episode data collided under identical keys.)
      - Every scalar aggregate metric for a given world size is ALSO
        logged once, in a single wandb.log() call, under a fixed
        "sweep_eval/" prefix together with "sweep_eval/world_size".
        wandb.define_metric() below registers "sweep_eval/world_size"
        as the step_metric for "sweep_eval/*", so those charts plot
        natively against world_size on the X axis, with exactly one
        (averaged) point per evaluated world size/batch -- this is
        the direct fix for "log against world_size on the X axis".
      - At the end, a summary wandb.Table plus custom line plots of
        every aggregate metric vs. world_size are additionally logged
        under "sweep_plots/...", as a secondary, one-shot summary view
        of the same data.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================

    if cfg.seed is not None:

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # =========================================================================
    # Device
    # =========================================================================

    device = torch.device(
        "cuda:1"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(f"[INIT] Device: {device}")

    # =========================================================================
    # World sizes to sweep over
    # =========================================================================

    if step_size <= 0:
        raise ValueError("step_size must be a positive integer")

    if vel_end < vel_start:
        raise ValueError(
            "vel_end must be >= vel_start"
        )

    velocities = np.arange(
        vel_start,
        vel_end + step_size,
        step_size
    ).tolist()

    # Make sure the requested endpoint is always evaluated, even if it
    # doesn't fall exactly on a step boundary.
    if velocities[-1] != vel_end:
        velocities.append(vel_end)

    logging.info(
        f"[INIT] Velocity sweep values: {velocities}"
    )

    # =========================================================================
    # W&B (single run for the whole sweep)
    # =========================================================================

    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    wandb.init(
        project=cfg.wandb_project,

        name=(
            f"eval_sweep_"
            f"{os.path.basename(checkpoint_path)}"
        ),

        config={
            **vars(cfg),

            "checkpoint_path": checkpoint_path,
            "n_eval_episodes_per_world_size": n_episodes,
            "vel_start": vel_start,
            "vel_end": vel_end,
            "vel_step": step_size,
            "velocities": velocities,
            "evaluation": True,
            "velocity_sweep": True,
            "n_parallel": n_parallel,
        },

        tags=[
            "evaluation",
            "checkpoint",
            "velocity_sweep",
        ]
    )

    cfg.run_id = wandb.run.name

    logging.info(f"[INIT] W&B run: {wandb.run.name}")

    # =========================================================================
    # World_size as the native X-axis for the "sweep_eval/*" metrics.
    #
    # Without this, every wandb.log() call in the run (including the
    # per-episode ones) shares a single monotonically-increasing
    # internal step counter that has nothing to do with world_size, so
    # a chart for e.g. "sweep_eval/mean_reward" would plot against
    # call-order rather than world_size. This tells wandb to use
    # "sweep_eval/world_size" as the step metric for every
    # "sweep_eval/*" key instead, so those charts get world_size on
    # the X axis automatically -- exactly one point per evaluated
    # world size, since evaluate() logs the sweep_eval/* dict once per
    # world size (see the "World-size-indexed logging" block there).
    # =========================================================================

    wandb.define_metric("sweep_eval/velocity")
    wandb.define_metric("sweep_eval/*", step_metric="sweep_eval/velocity")

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

    probe_env = make_eval_env(cfg)
    obs_space = probe_env.observation_space
    action_nvec = probe_env.action_space.nvec.tolist()
    probe_env.close()

    agent = TrXLActorCritic(
        observation_space=obs_space,
        action_nvec=action_nvec,
        cfg=cfg
    ).to(device)

    logging.info(f"[INIT] Loading checkpoint:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    agent.load_state_dict(checkpoint["agent"])

    logging.info("Successfully loaded agent weights")

    agent.eval()

    # =========================================================================
    # Sweep loop
    # =========================================================================

    summary_rows = []

    for vel_n in velocities:

        logging.info(
            f"[SWEEP] Evaluating velocity={vel_n}"
        )

        cfg.velocity_step_size = vel_n

        # In batched mode, evaluate() builds its own n_parallel env
        # instances per step internally from cfg -- no persistent env
        # is needed here. In sequential mode, build one as before.
        env = None if n_parallel > 1 else make_eval_env(cfg)

        reset_agent_memory(agent)

        aggregate_metrics = evaluate(
            agent=agent,
            env=env,
            cfg=cfg,
            device=device,
            n_episodes=n_episodes,
            # Namespaced per velocity (was previously a static
            # "eval/" for every iteration, which collided all velocities
            # per-episode series onto the same keys).
            metric_prefix=f"eval/velocity_{vel_n}",
            velocity=vel_n,
            n_parallel=n_parallel,
            is_ue5=is_ue5,
        )

        reset_agent_memory(agent)

        if env is not None:
            env.close()

        # Pull out only scalar metrics (skip the wandb.plot.scatter
        # objects) to build the velocity summary table.
        row = {"velocity": vel_n}

        for key, value in aggregate_metrics.items():

            if isinstance(value, (int, float)) and not isinstance(value, bool):

                clean_key = key.split("/")[-1]

                row[clean_key] = value

        summary_rows.append(row)

    # =========================================================================
    # Build velocity vs. metric summary table + line plots
    #
    # This is a secondary, one-shot summary view logged once at the
    # end of the sweep. The live, per-batch "velocity on the X axis"
    # charts are the "sweep_eval/*" ones logged inside evaluate() as
    # the sweep progresses (see wandb.define_metric() above).
    # =========================================================================

    if len(summary_rows) > 0:

        metric_names = sorted(
            {
                key
                for row in summary_rows
                for key in row.keys()
                if key != "velocity"
            }
        )

        columns = ["velocity"] + metric_names

        table_data = [
            [row["velocity"]] + [row.get(m, None) for m in metric_names]
            for row in summary_rows
        ]

        summary_table = wandb.Table(
            columns=columns,
            data=table_data
        )

        sweep_plots = {
            "sweep/summary_table": summary_table
        }

        for metric_name in metric_names:

            sweep_plots[
                f"sweep_plots/{metric_name}_vs_velocity"
            ] = wandb.plot.line(
                summary_table,
                x="velocity",
                y=metric_name,
                title=f"{metric_name} vs Velocity"
            )

        wandb.log(sweep_plots)

        logging.info(
            "\n"
            "==============================\n"
            "VELOCITY SWEEP COMPLETE\n"
            "==============================\n"
            f"Velocities evaluated : {velocities}\n"
            f"Episodes per size     : {n_episodes}\n"
            "=============================="
        )

    wandb.finish()

    logging.info("[DONE] Velocity sweep evaluation complete.")

    return summary_rows


# =============================================================================
# World-size sweep evaluation
# =============================================================================

def run_world_size_sweep(
    checkpoint_path,
    cfg,
    n_episodes,
    world_size_start,
    world_size_end,
    step_size=16,
    is_ue5=False,
    n_parallel=1,
):
    """
    Evaluate the same checkpoint across a range of square world sizes,
    from (world_size_start, world_size_start) to
    (world_size_end, world_size_end) inclusive, in increments of
    `step_size`, and log reward / length / domain metrics against
    world size to W&B.

    A single W&B run is used for the whole sweep:
      - Each world size gets its own namespaced time series
        (eval/world_<N>/...) so per-episode curves for different
        world sizes don't get interleaved on the same line. (This was
        previously broken -- the call into evaluate() passed a static
        metric_prefix of "eval/" for every world size instead of
        interpolating world_size_n into it, so every world size's
        per-episode data collided under identical keys.)
      - Every scalar aggregate metric for a given world size is ALSO
        logged once, in a single wandb.log() call, under a fixed
        "sweep_eval/" prefix together with "sweep_eval/world_size".
        wandb.define_metric() below registers "sweep_eval/world_size"
        as the step_metric for "sweep_eval/*", so those charts plot
        natively against world_size on the X axis, with exactly one
        (averaged) point per evaluated world size/batch -- this is
        the direct fix for "log against world_size on the X axis".
      - At the end, a summary wandb.Table plus custom line plots of
        every aggregate metric vs. world_size are additionally logged
        under "sweep_plots/...", as a secondary, one-shot summary view
        of the same data.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================

    if cfg.seed is not None:

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    # =========================================================================
    # Device
    # =========================================================================

    device = torch.device(
        "cuda:1"
        if torch.cuda.is_available()
        else "cpu"
    )

    logging.info(f"[INIT] Device: {device}")

    # =========================================================================
    # World sizes to sweep over
    # =========================================================================

    if step_size <= 0:
        raise ValueError("step_size must be a positive integer")

    if world_size_end < world_size_start:
        raise ValueError(
            "world_size_end must be >= world_size_start"
        )

    world_sizes = list(
        range(world_size_start, world_size_end + 1, step_size)
    )

    # Make sure the requested endpoint is always evaluated, even if it
    # doesn't fall exactly on a step boundary.
    if world_sizes[-1] != world_size_end:
        world_sizes.append(world_size_end)

    logging.info(
        f"[INIT] World-size sweep values: {world_sizes}"
    )

    # =========================================================================
    # W&B (single run for the whole sweep)
    # =========================================================================

    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key

    wandb.init(
        project=cfg.wandb_project,

        name=(
            f"eval_sweep_"
            f"{os.path.basename(checkpoint_path)}"
        ),

        config={
            **vars(cfg),

            "checkpoint_path": checkpoint_path,
            "n_eval_episodes_per_world_size": n_episodes,
            "world_size_start": world_size_start,
            "world_size_end": world_size_end,
            "world_size_step": step_size,
            "world_sizes": world_sizes,
            "evaluation": True,
            "world_size_sweep": True,
            "n_parallel": n_parallel,
        },

        tags=[
            "evaluation",
            "checkpoint",
            "world_size_sweep",
        ]
    )

    cfg.run_id = wandb.run.name

    logging.info(f"[INIT] W&B run: {wandb.run.name}")

    # =========================================================================
    # World_size as the native X-axis for the "sweep_eval/*" metrics.
    #
    # Without this, every wandb.log() call in the run (including the
    # per-episode ones) shares a single monotonically-increasing
    # internal step counter that has nothing to do with world_size, so
    # a chart for e.g. "sweep_eval/mean_reward" would plot against
    # call-order rather than world_size. This tells wandb to use
    # "sweep_eval/world_size" as the step metric for every
    # "sweep_eval/*" key instead, so those charts get world_size on
    # the X axis automatically -- exactly one point per evaluated
    # world size, since evaluate() logs the sweep_eval/* dict once per
    # world size (see the "World-size-indexed logging" block there).
    # =========================================================================

    wandb.define_metric("sweep_eval/world_size")
    wandb.define_metric("sweep_eval/*", step_metric="sweep_eval/world_size")

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
    # Build the agent once. World size only affects the environment's
    # observation content, not its shape, so the same agent/network
    # can be reused across the whole sweep as long as the observation
    # space is consistent across world sizes. We build the agent using
    # an env constructed at the first (smallest) world size.
    # =========================================================================

    cfg.world_size = (world_sizes[0], world_sizes[0])

    probe_env = make_eval_env(cfg)

    obs_space = probe_env.observation_space
    action_nvec = probe_env.action_space.nvec.tolist()

    probe_env.close()

    agent = TrXLActorCritic(
        observation_space=obs_space,
        action_nvec=action_nvec,
        cfg=cfg
    ).to(device)

    logging.info(f"[INIT] Loading checkpoint:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    agent.load_state_dict(checkpoint["agent"])

    logging.info("Successfully loaded agent weights")

    agent.eval()

    # =========================================================================
    # Sweep loop
    # =========================================================================

    summary_rows = []

    for world_size_n in world_sizes:

        cfg.world_size = (world_size_n, world_size_n)

        logging.info(
            f"[SWEEP] Evaluating world_size={cfg.world_size}"
        )

        # In batched mode, evaluate() builds its own n_parallel env
        # instances per step internally from cfg -- no persistent env
        # is needed here. In sequential mode, build one as before.
        env = None if n_parallel > 1 else make_eval_env(cfg)

        reset_agent_memory(agent)

        aggregate_metrics = evaluate(
            agent=agent,
            env=env,
            cfg=cfg,
            device=device,
            n_episodes=n_episodes,
            # Namespaced per world size (was previously a static
            # "eval/" for every iteration, which collided all world
            # sizes' per-episode series onto the same keys).
            metric_prefix=f"eval/world_{world_size_n}",
            world_size=world_size_n,
            n_parallel=n_parallel,
            is_ue5=is_ue5,
        )

        reset_agent_memory(agent)

        if env is not None:
            env.close()

        # Pull out only scalar metrics (skip the wandb.plot.scatter
        # objects) to build the world-size summary table.
        row = {"world_size": world_size_n}

        for key, value in aggregate_metrics.items():

            if isinstance(value, (int, float)) and not isinstance(value, bool):

                clean_key = key.split("/")[-1]

                row[clean_key] = value

        summary_rows.append(row)

    # =========================================================================
    # Build world-size vs. metric summary table + line plots
    #
    # This is a secondary, one-shot summary view logged once at the
    # end of the sweep. The live, per-batch "world_size on the X axis"
    # charts are the "sweep_eval/*" ones logged inside evaluate() as
    # the sweep progresses (see wandb.define_metric() above).
    # =========================================================================

    if len(summary_rows) > 0:

        metric_names = sorted(
            {
                key
                for row in summary_rows
                for key in row.keys()
                if key != "world_size"
            }
        )

        columns = ["world_size"] + metric_names

        table_data = [
            [row["world_size"]] + [row.get(m, None) for m in metric_names]
            for row in summary_rows
        ]

        summary_table = wandb.Table(
            columns=columns,
            data=table_data
        )

        sweep_plots = {
            "sweep/summary_table": summary_table
        }

        for metric_name in metric_names:

            sweep_plots[
                f"sweep_plots/{metric_name}_vs_world_size"
            ] = wandb.plot.line(
                summary_table,
                x="world_size",
                y=metric_name,
                title=f"{metric_name} vs World Size"
            )

        wandb.log(sweep_plots)

        logging.info(
            "\n"
            "==============================\n"
            "WORLD-SIZE SWEEP COMPLETE\n"
            "==============================\n"
            f"World sizes evaluated : {world_sizes}\n"
            f"Episodes per size     : {n_episodes}\n"
            "=============================="
        )

    wandb.finish()

    logging.info("[DONE] World-size sweep evaluation complete.")

    return summary_rows


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

    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help=(
            "Number of episodes to run concurrently via batched "
            "(vectorized) stepping. Default: 1 (sequential)."
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
        n_episodes=args.episodes,
        n_parallel=args.n_parallel,
    )