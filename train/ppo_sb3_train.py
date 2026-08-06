"""
Plain (memoryless) PPO training script using your custom CNN + position-MLP
feature extractor, structured to mirror train.py's train() and the
RecurrentPPO comparison script as closely as possible, so all three runs
overlay in the same wandb project.

This is the third leg of the comparison: TrXL memory (train.py) vs. LSTM
memory (RecurrentPPO script) vs. no memory at all (this script) -- same
perception frontend, same PPO hyperparameters, same logging keys, the only
thing that changes is the memory mechanism (none, here).

WHAT'S MATCHED vs train.py / the RecurrentPPO script, and how:
  - episode/reward, episode/length, episode/mean_reward, episode/env_idx,
    domain/* metrics, and the length-vs-reward / fire-coverage scatter
    plots are logged with IDENTICAL keys, at the same cadence (every
    SCATTER_EP_FREQ episodes), via EpisodeWandbCallback -- copied verbatim
    from the RecurrentPPO script, since SB3's episode-info mechanism
    (Monitor's "episode" info key) works identically for PPO and
    RecurrentPPO.
  - episode/reward and episode/mean_reward come from Monitor's "r" (RAW,
    un-normalised per-episode return), matching train.py's
    `ep_rewards += rewards` accumulation.
  - Reward normalisation reuses your actual RunningMeanStd class via
    RunningMeanStdRewardWrapper -- copied verbatim, same raw-vs-normalised
    split as the other two scripts.
  - train/policy_loss, train/value_loss, train/entropy, train/approx_kl,
    train/steps_per_sec, eval/mean_reward are forwarded via WandbKVWriter
    -- copied verbatim, hooks SB3's own Logger.
  - SB3 logs entropy_loss = -mean(entropy); WandbKVWriter negates it back
    into train/entropy, same as the RecurrentPPO script.
  - SB3's PPO.train() already breaks out of the minibatch loop the moment
    approx_kl_div > 1.5 * target_kl within an epoch, matching train.py's
    per-minibatch early-stop.

WHAT'S DIFFERENT vs the RecurrentPPO script (memory removed):
  - stable_baselines3.PPO instead of sb3_contrib.RecurrentPPO -- no
    recurrent state, no truncated-BPTT sequence handling.
  - policy = "MultiInputPolicy" instead of "MultiInputLstmPolicy" -- no
    lstm_hidden_size / n_lstm_layers / shared_lstm / enable_critic_lstm
    kwargs, since there's no LSTM.
  - CNNPosFeaturesExtractor is UNCHANGED from the RecurrentPPO script --
    it was already memory-free (CNN + position MLP + fusion only, no
    transformer block, no LSTM). Reusing it here keeps perception
    identical across all three comparison runs, isolating the memory
    mechanism as the only variable.
  - batch_size now means the same thing as train.py's TrXL buffer
    (standard PPO fully shuffles (n_steps, n_envs) into flat minibatches,
    no per-env-contiguous sequence constraint) -- unlike the RecurrentPPO
    script, this one IS directly comparable to train.py on batch_size.
  - value_loss formula difference from train.py still applies here (SB3
    plain MSE vs. train.py's 0.5x + clipped/unclipped max) -- same caveat
    as the RecurrentPPO script, not a bug on either side.

WHAT'S NOT INCLUDED, deliberately:
  - No memory diagnostics (recency profile, attention entropy, CKA, gate
    stats) -- those are all specific to the TrXL memory mechanism and
    don't have an analogue here. This run's job in the comparison is to
    be the memory-free baseline.
"""

import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import wandb

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import KVWriter, Logger
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from envs.WildfireSingleAgentEnv import SingleAgentEnv
from config.Config import VideoWriterConfig, EnvConfig

# Reuse your actual reward normaliser, not a reimplementation, so all
# three runs see reward scale computed by identical code.
from train.trxl_train_single_agent import RunningMeanStd

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


# 
# Features extractor: CNN + position-fusion frontend, identical to the
# RecurrentPPO script's -- no memory mechanism of any kind sits on top of
# this in this script (SB3's default MultiInputPolicy just feeds it
# straight into the actor/critic MLP heads).
# 

class CNNPosFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space["viewport"].shape[0]
        pos_dim    = observation_space["positions"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,         64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,         64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64,        128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(
                torch.zeros(1, *observation_space["viewport"].shape)
            ).shape[1]

        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_dim, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,     128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128,      64),                    nn.ReLU(),
        )

        self.pos_to_cnn_bias = nn.Linear(pos_dim, cnn_out)

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + 64, features_dim * 2),
            nn.LayerNorm(features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        self.token_spatial_encoding = nn.Linear(pos_dim, features_dim)

    def forward(self, observations):
        vp  = observations["viewport"]
        pos = observations["positions"]

        cnn_feat     = self.cnn(vp)
        spatial_bias = self.pos_to_cnn_bias(pos)
        cnn_feat     = cnn_feat * torch.sigmoid(spatial_bias)

        pos_feat = self.pos_mlp(pos)
        current  = self.fusion(torch.cat([cnn_feat, pos_feat], dim=1))
        current  = current + self.token_spatial_encoding(pos)
        return current


# 
# Reward normalisation -- reuses your RunningMeanStd verbatim
# 

class RunningMeanStdRewardWrapper(VecEnvWrapper):
    """
    Applies your existing RunningMeanStd normaliser to rewards AFTER
    SubprocVecEnv but must sit OUTSIDE the per-env Monitor wrappers (i.e.
    Monitor(TimeLimit(env)) -> SubprocVecEnv -> this wrapper), so Monitor's
    "r" (used for episode/reward, episode/mean_reward) reflects the RAW
    reward, exactly like train.py's `ep_rewards += rewards` accumulates
    raw rewards while the normalised reward is what actually goes into
    the buffer for GAE / value targets.

    NOTE: unlike VecNormalize, this does not save/restore its running
    stats on checkpoint reload -- fine for a single side-by-side
    comparison run; extend __getstate__/__setstate__ if you need resume
    support later.
    """
    def __init__(self, venv, clip: float = 10.0):
        super().__init__(venv)
        self.reward_rms = RunningMeanStd()
        self.clip = clip

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.reward_rms.update(rewards)
        norm_rewards = self.reward_rms.normalise(rewards, clip=self.clip)
        return obs, norm_rewards, dones, infos


# 
# wandb logging: SB3 Logger hook + episode-level callback
# (copied verbatim from the RecurrentPPO script -- SB3's episode-info and
# Logger mechanisms work identically for PPO)
# 

class WandbKVWriter(KVWriter):
    """
    Forwards every value SB3's internal Logger records straight to wandb,
    plus derives renamed/sign-corrected keys so this run's charts overlay
    directly on train.py's / the RecurrentPPO script's charts by metric
    name.
    """
    def __init__(self, global_step_ref: dict):
        # global_step_ref is a mutable single-key dict so this writer can
        # read the callback's live timestep count without a circular
        # constructor dependency between writer and callback.
        self._global_step_ref = global_step_ref

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        log_dict = dict(key_values)

        # SB3 logs entropy_loss = -mean(entropy); recover the raw value.
        if "train/entropy_loss" in log_dict:
            log_dict["train/entropy"] = -log_dict["train/entropy_loss"]

        rename_map = {
            "train/policy_gradient_loss": "train/policy_loss",
            "train/value_loss":           "train/value_loss",     # same name, different formula -- see docstring
            "train/approx_kl":            "train/approx_kl",
            "time/fps":                   "train/steps_per_sec",
        }
        for sb3_key, matched_key in rename_map.items():
            if sb3_key in log_dict:
                log_dict[matched_key] = log_dict[sb3_key]

        log_dict["global_step"] = self._global_step_ref["value"]
        wandb.log(log_dict)

    def close(self) -> None:
        pass


class EpisodeWandbCallback(BaseCallback):
    """
    Mirrors the per-episode logging block inside train.py's rollout loop:
    episode/reward, episode/length, episode/mean_reward, episode/env_idx,
    domain/* metrics, and the two scatter plots, at the same
    SCATTER_EP_FREQ cadence. Reads Monitor's "episode" info key (raw
    reward/length) plus your env's own "domain_metrics" info key, which
    Monitor passes through untouched.
    """
    SCATTER_EP_FREQ = 50
    COVERAGE_THRESHOLDS = [25, 50, 75, 90, 99]

    def __init__(self, global_step_ref: dict, recent_rewards_maxlen: int = 100):
        super().__init__()
        self.global_step_ref = global_step_ref
        self.recent_rewards  = deque(maxlen=recent_rewards_maxlen)
        self.episode_count   = 0
        self.scatter_ep_data       = []
        self.scatter_coverage_data = []

    def _on_step(self) -> bool:
        # Keep the shared global_step in sync with SB3's own timestep
        # counter so WandbKVWriter's train/* logs and this callback's
        # episode/* logs land on comparable x-axis values.
        self.global_step_ref["value"] = self.num_timesteps

        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_reward = float(ep_info["r"])
            ep_length = int(ep_info["l"])

            self.recent_rewards.append(ep_reward)
            mean_reward = float(np.mean(self.recent_rewards)) if self.recent_rewards else 0.0
            self.episode_count += 1

            domain_metrics = info.get("domain_metrics", {})

            wandb.log({
                "episode/reward":                float(ep_reward),
                "episode/length":                int(ep_length),
                "episode/mean_reward":           mean_reward,
                "episode/env_idx":               env_idx,
                "domain/fire_revisit_count":      domain_metrics.get("domain/revisit_count", 0),
                "domain/fire_revisit_mean_delta": domain_metrics.get("domain/revisit_delta_mean", 0.0),
                "domain/fire_revisit_max_delta":  domain_metrics.get("domain/revisit_delta_max", 0.0),
                "domain/fire_revisit_min_delta":  domain_metrics.get("domain/revisit_delta_min", 0.0),
                "domain/fire_coverage_mean":      domain_metrics.get("domain/fire_coverage_mean", 0.0),
                "domain/fire_coverage_AUC":       domain_metrics.get("domain/fire_coverage_AUC", 0.0),
                "domain/fire_coverage_final":     domain_metrics.get("domain/fire_coverage_final", 0.0),
                "global_step":                    self.num_timesteps,
            })

            self.scatter_ep_data.append([ep_length, ep_reward, env_idx])
            if self.episode_count % self.SCATTER_EP_FREQ == 0:
                wandb.log({
                    "scatter/length_vs_reward": wandb.plot.scatter(
                        wandb.Table(
                            columns=["episode_length", "reward", "env_idx"],
                            data=self.scatter_ep_data,
                        ),
                        x="episode_length", y="reward",
                        title="Episode Length vs Reward",
                    ),
                    "global_step": self.num_timesteps,
                })
                self.scatter_ep_data = []

            for pct in self.COVERAGE_THRESHOLDS:
                ts = domain_metrics.get(f"domain/fire_coverage_{pct}", -1)
                if ts != -1:
                    self.scatter_coverage_data.append([pct, float(ts), env_idx])

            if self.episode_count % self.SCATTER_EP_FREQ == 0 and self.scatter_coverage_data:
                wandb.log({
                    "scatter/fire_coverage_progression": wandb.plot.scatter(
                        wandb.Table(
                            columns=["coverage_threshold_%", "steps_to_reach", "env_idx"],
                            data=self.scatter_coverage_data,
                        ),
                        x="coverage_threshold_%", y="steps_to_reach",
                        title="Steps to Reach Fire Coverage Thresholds",
                    ),
                    "global_step": self.num_timesteps,
                })
                self.scatter_coverage_data = []

        return True


# 
# Environment factory -- same env/config, Monitor added (required by SB3
# for episode tracking; does not strip your existing info keys)
# 

def make_env_fn(cfg: EnvConfig, rank: int):
    video_config = VideoWriterConfig(
        is_enabled      = rank == 0,
        sample_interval = 5      if rank == 0 else 999999,
        save_interval   = 5      if rank == 0 else 999999,
        base_path       = "./vids_isolated_ppo/"
    )

    def _init():
        env = SingleAgentEnv(
            world_size      = cfg.world_size,
            render_mode     = "rgb_array",
            seed            = cfg.seed + rank if cfg.seed is not None else None,
            video_conf      = video_config,
            phase_weights   = cfg.phase_weights,
            env_id          = cfg.run_id,
            vp_size         = cfg.vp_size,
            is_gt_visible   = True,
            is_recency_obs_disabled=True,
            device          = torch.device("cuda:1"),
        )
        env = TimeLimit(env, max_episode_steps=cfg.iter_limit)
        env = Monitor(env)   # required for episode "r"/"l" info + ep_info_buffer
        return env
    return _init


# 
# Main training entry point -- mirrors train.py's train(cfg, checkpoint_path)
# 

def train(cfg: EnvConfig, checkpoint_path=None):
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    wandb.init(project=cfg.wandb_project, config={**vars(cfg), "algo": "PPO_no_memory"})

    run_id = f"{wandb.run.name}-ppo"
    logging.info(f"[Train] : Begin memoryless PPO comparison run with ID : {run_id}")

    checkpoint_dir = cfg.checkpoint_dir + "_ppo"
    best_model_dir = cfg.best_model_dir + "_ppo"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    if cfg.seed is not None:
        import random
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"[INIT] Device: {device} | N envs: {cfg.n_envs} | Memory: none")

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Vectorised environment ────────────────────────────────────────────
    train_envs = SubprocVecEnv([make_env_fn(cfg, rank=i) for i in range(cfg.n_envs)])
    train_envs = RunningMeanStdRewardWrapper(train_envs, clip=100.0)

    eval_envs = SubprocVecEnv([make_env_fn(cfg, rank=99)])   # single env, like evaluate() in train.py

    # ── Model ──────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = CNNPosFeaturesExtractor,
        features_extractor_kwargs = dict(features_dim=cfg.features_dim),
    )

    model = PPO(
        policy          = "MultiInputPolicy",
        env             = train_envs,
        n_steps         = cfg.n_steps,
        batch_size      = cfg.batch_size,
        n_epochs        = cfg.n_epochs,
        learning_rate   = cfg.learning_rate,
        gamma           = cfg.gamma,
        gae_lambda      = cfg.gae_lambda,
        clip_range      = cfg.clip_coef,
        clip_range_vf   = cfg.clip_coef,   # off by default in SB3 -- set explicitly to at least align this
        ent_coef        = cfg.ent_coef,
        vf_coef         = cfg.vf_coef,
        max_grad_norm   = cfg.max_grad_norm,
        target_kl       = cfg.target_kl,
        policy_kwargs   = policy_kwargs,
        device          = device,
        verbose         = 0,
    )

    if checkpoint_path is not None:
        logging.info(f"[INIT] Loading checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=train_envs, device=device)

    # Shared mutable step counter so WandbKVWriter's train/* logs and
    # EpisodeWandbCallback's episode/* logs use the same x-axis value.
    global_step_ref = {"value": 0}
    model.set_logger(Logger(folder=None, output_formats=[WandbKVWriter(global_step_ref)]))

    # ── Callbacks: checkpoint / eval / episode logging ────────────────────
    # SB3 callback frequencies are in units of "vectorized steps" (one
    # _on_step call per env.step() across all n_envs), while train.py's
    # global_step already counts n_envs per call -- divide by n_envs so
    # the wall-clock cadence matches train.py's checkpoint_freq/eval_freq.
    checkpoint_callback = CheckpointCallback(
        save_freq    = max(cfg.checkpoint_freq // cfg.n_envs, 1),
        save_path    = checkpoint_dir,
        name_prefix  = "firescout_ppo",
    )
    eval_callback = EvalCallback(
        eval_envs,
        best_model_save_path = best_model_dir,
        log_path              = None,
        eval_freq             = max(cfg.eval_freq // cfg.n_envs, 1),
        n_eval_episodes        = cfg.n_eval_episodes,
        deterministic          = True,
    )
    episode_callback = EpisodeWandbCallback(global_step_ref)

    logging.info(
        f"[TRAIN] Starting - {cfg.total_timesteps:,} steps | "
        f"rollout size = {cfg.n_steps * cfg.n_envs:,} transitions"
    )
    start_time = time.time()

    model.learn(
        total_timesteps = cfg.total_timesteps,
        callback         = [checkpoint_callback, eval_callback, episode_callback],
        progress_bar     = False,
    )

    elapsed = time.time() - start_time
    logging.info(f"[DONE] Memoryless PPO training complete in {elapsed:.0f}s.")

    model.save("./firescout_ppo_final.zip")
    wandb.finish()
    train_envs.close()
    eval_envs.close()


if __name__ == "__main__":
    train(EnvConfig())