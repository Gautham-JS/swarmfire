"""
RecurrentPPO (sb3-contrib) baseline on Memory Maze, logging the SAME wandb
keys as the shared TrXL train() so the two runs overlay in one project:

    episode/reward, episode/length, episode/mean_reward, episode/env_idx
    train/policy_loss, train/value_loss, train/entropy, train/approx_kl,
    train/grad_norm, train/skipped_updates, train/steps_per_sec
    scatter/length_vs_reward, scatter/policy_loss_vs_value_loss,
    scatter/kl_vs_entropy, eval/mean_reward, global_step

Why not just use SB3's own logger?
  * SB3 logs `train/entropy_loss` = -mean(entropy)  -> that is why you saw
    negative values. Here we log +mean(entropy) as `train/entropy`.
  * SB3 logs value_loss = plain MSE; the TrXL loop uses 0.5 * max(clipped,
    unclipped) MSE. Different scale -> curves would not overlap.
  * SB3 has no grad-norm metric and does not expose per-minibatch values
    (needed for the scatter plots).
So LoggedRecurrentPPO overrides train() with a loss that mirrors the TrXL
loop (see the list of matched behaviours below) and logs to wandb itself.

Matched to the TrXL loop:
  - PPO loss: max-form clipped policy loss, 0.5*max-form clipped value loss
    (clip = cfg.clip_coef), loss = pl + vf_coef*vl - ent_coef*entropy
  - advantages normalised ONCE per rollout (not per minibatch)
  - KL early stop on the epoch-mean approx_kl > cfg.target_kl
  - grad norm = pre-clip norm returned by clip_grad_norm_, averaged
  - non-finite loss / grad -> minibatch skipped, counted, abort past limit
  - reward scaling: VecNormalize(norm_obs=False, norm_reward=True) divides
    by the running std of discounted returns, no mean subtraction, clip 10
    (cfg.reward_norm == "none" disables it)
  - Adam eps=1e-5, same lr / gamma / lambda / n_steps / batch_size / epochs
  - episode/* logged from Monitor => always RAW (unnormalised) returns

NOT matched (inherent to the architectures / SB3):
  - Network: CNN -> LSTM here vs CNN -> Gated TrXL there.
  - SB3 bootstraps the value on time-limit truncation; the TrXL loop doesn't.
  - Minibatches: SB3 splits by padded sequences and does BPTT through them;
    the TrXL loop samples individual transitions with stored memory.

Adjust the two imports below to your module names.
"""

import os
import time
import random
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn
import wandb
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

from train.ppo_common import BaseConfig            # <- your shared PPO/TrXL module
from envs.MemoryMazeEnv import MemoryMazeEnv      # <- the adapter from above

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MemoryMazeConfig(BaseConfig):
    benchmark_name:    str = "memory_maze"
    algo:              str = "recurrent_ppo"     # logged to wandb config + tag
    env_id:            str = "memory_maze:MemoryMaze-9x9-v0"
    max_episode_steps: int = 1000
    lstm_hidden_size:  int = 256


# ---------------------------------------------------------------------------
# Env factory (Monitor sits INSIDE the subprocess => raw episode stats)
# ---------------------------------------------------------------------------

def make_env_fn(cfg, rank):
    def _thunk():
        env = MemoryMazeEnv(cfg.env_id)
        env = TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
        env = Monitor(env)
        return env
    return _thunk


# ---------------------------------------------------------------------------
# Feature extractor. SB3's default CombinedExtractor would FLATTEN the float
# "viewport" (it only treats uint8 Boxes as images), so use a CNN explicitly.
# "positions" is a constant zero vector for Memory Maze and is ignored.
# ---------------------------------------------------------------------------

class MazeCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.spaces["viewport"].shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs["viewport"]))


# ---------------------------------------------------------------------------
# RecurrentPPO with a train() that mirrors the TrXL loop's loss + logging
# ---------------------------------------------------------------------------

class LoggedRecurrentPPO(RecurrentPPO):
    max_skipped_updates: int = 20

    def learn(self, *args, **kwargs):
        self._wb_start = time.time()
        return super().learn(*args, **kwargs)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # Normalise advantages once per rollout (buffer is still (T, N) here).
        adv = self.rollout_buffer.advantages
        self.rollout_buffer.advantages = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)

        pls, vls, ents, kls, gns = [], [], [], [], []
        scatter_loss, scatter_kl = [], []
        n_skipped = 0

        for epoch in range(self.n_epochs):
            epoch_kls = []

            for rd in self.rollout_buffer.get(self.batch_size):
                actions = rd.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                mask = rd.mask > 1e-8   # ignore padded timesteps

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rd.observations, actions, rd.lstm_states, rd.episode_starts,
                )
                values = values.flatten()

                log_ratio = log_prob - rd.old_log_prob
                ratio = log_ratio.exp()
                approx_kl = ((ratio - 1) - log_ratio)[mask].mean().item()

                pg1 = -rd.advantages * ratio
                pg2 = -rd.advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = th.max(pg1, pg2)[mask].mean()

                v_clipped = rd.old_values + th.clamp(values - rd.old_values, -clip_range, clip_range)
                vf1 = (values - rd.returns) ** 2
                vf2 = (v_clipped - rd.returns) ** 2
                value_loss = 0.5 * th.max(vf1, vf2)[mask].mean()

                entropy_mean = entropy[mask].mean()     # POSITIVE entropy
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean

                if not th.isfinite(loss):
                    n_skipped += 1
                    logging.error(f"[PPO] Non-finite loss (pl={policy_loss.item()}, "
                                  f"vl={value_loss.item()}, ent={entropy_mean.item()}); skipping minibatch")
                    if n_skipped > self.max_skipped_updates:
                        raise RuntimeError("Too many non-finite updates in one rollout")
                    continue

                self.policy.optimizer.zero_grad()
                loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                if not th.isfinite(grad_norm):
                    n_skipped += 1
                    bad = [n for n, p in self.policy.named_parameters()
                           if p.grad is not None and not th.isfinite(p.grad).all()]
                    logging.error(f"[PPO] Non-finite grad norm; offending params: {bad}")
                    self.policy.optimizer.zero_grad()
                    if n_skipped > self.max_skipped_updates:
                        raise RuntimeError("Too many non-finite updates in one rollout")
                    continue

                self.policy.optimizer.step()

                pl, vl, ent = policy_loss.item(), value_loss.item(), entropy_mean.item()
                pls.append(pl); vls.append(vl); ents.append(ent)
                kls.append(approx_kl); epoch_kls.append(approx_kl); gns.append(float(grad_norm))
                scatter_loss.append([pl, vl])
                scatter_kl.append([approx_kl, ent])

            self._n_updates += 1
            if self.target_kl is not None and epoch_kls and np.mean(epoch_kls) > self.target_kl:
                logging.info(f"[PPO] Early stop at epoch {epoch + 1}, KL={np.mean(epoch_kls):.4f}")
                break

        step = self.num_timesteps
        elapsed = time.time() - self._wb_start
        sps = step / elapsed if elapsed > 0 else 0.0
        m = lambda x: float(np.mean(x)) if len(x) else 0.0

        wandb.log({
            "train/policy_loss":     m(pls),
            "train/value_loss":      m(vls),
            "train/entropy":         m(ents),
            "train/approx_kl":       m(kls),
            "train/grad_norm":       m(gns),
            "train/skipped_updates": n_skipped,
            "train/steps_per_sec":   sps,
            "scatter/policy_loss_vs_value_loss": wandb.plot.scatter(
                wandb.Table(columns=["policy_loss", "value_loss"], data=scatter_loss),
                x="policy_loss", y="value_loss", title="Policy Loss vs Value Loss",
            ),
            "scatter/kl_vs_entropy": wandb.plot.scatter(
                wandb.Table(columns=["approx_kl", "entropy"], data=scatter_kl),
                x="approx_kl", y="entropy", title="KL Divergence vs Entropy",
            ),
            "global_step": step,
        })
        logging.info(
            f"[{step:>8}] pl={m(pls):.4f} vl={m(vls):.4f} ent={m(ents):.4f} "
            f"kl={m(kls):.4f} gn={m(gns):.3f} skip={n_skipped} sps={sps:.0f}"
        )


# ---------------------------------------------------------------------------
# Callback: per-episode logs, scatter, checkpoints, eval
# ---------------------------------------------------------------------------

class WandbTrainCallback(BaseCallback):
    SCATTER_EP_FREQ = 50

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.recent = deque(maxlen=100)
        self.scatter_ep = []
        self.episode_count = 0
        self.next_ckpt = cfg.checkpoint_freq
        self.next_eval = cfg.eval_freq
        self.best_eval = -np.inf

    def _on_step(self) -> bool:
        step = self.num_timesteps

        for env_idx, info in enumerate(self.locals["infos"]):
            ep = info.get("episode")           # set by Monitor => raw reward/length
            if ep is None:
                continue
            reward, length = float(ep["r"]), int(ep["l"])
            self.recent.append(reward)
            self.episode_count += 1

            wandb.log({
                "episode/reward":      reward,
                "episode/length":      length,
                "episode/mean_reward": float(np.mean(self.recent)),
                "episode/env_idx":     env_idx,
                "global_step":         step,
            })

            self.scatter_ep.append([length, reward, env_idx])
            if self.episode_count % self.SCATTER_EP_FREQ == 0:
                wandb.log({
                    "scatter/length_vs_reward": wandb.plot.scatter(
                        wandb.Table(columns=["episode_length", "reward", "env_idx"], data=self.scatter_ep),
                        x="episode_length", y="reward", title="Episode Length vs Reward",
                    ),
                    "global_step": step,
                })
                self.scatter_ep = []

        if step >= self.next_ckpt:
            path = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.benchmark_name}_recppo_{step}_steps")
            self.model.save(path)
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(path + "_vecnormalize.pkl")
            logging.info(f"[CKPT] Saved: {path}.zip")
            self.next_ckpt += self.cfg.checkpoint_freq

        if step >= self.next_eval:
            eval_env = DummyVecEnv([make_env_fn(self.cfg, rank=99)])
            try:
                # Sampled (stochastic) actions, same as the TrXL evaluate().
                eval_reward, _ = evaluate_policy(
                    self.model, eval_env,
                    n_eval_episodes=self.cfg.n_eval_episodes, deterministic=False,
                )
            finally:
                eval_env.close()
            eval_reward = float(eval_reward)
            logging.info(f"[EVAL] step={step} mean_reward={eval_reward:.3f}")
            wandb.log({"eval/mean_reward": eval_reward, "global_step": step})

            if eval_reward > self.best_eval:
                self.best_eval = eval_reward
                self.model.save(os.path.join(self.cfg.best_model_dir, "best_model"))
                logging.info(f"[EVAL] New best: {self.best_eval:.3f}")
            self.next_eval += self.cfg.eval_freq

        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = MemoryMazeConfig()

    wandb.init(
        project="gtrxlh-mem-maze",
        config=vars(cfg),
        tags=[cfg.benchmark_name, cfg.algo],
    )
    cfg.run_id = wandb.run.name
    logging.info(f"[Train:{cfg.benchmark_name}/{cfg.algo}] Begin session: {cfg.run_id}")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.best_model_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        th.manual_seed(cfg.seed)

    if cfg.reward_norm not in ("returns", "none"):
        raise ValueError(f"cfg.reward_norm must be 'returns' or 'none', got {cfg.reward_norm!r}")

    venv = SubprocVecEnv([make_env_fn(cfg, rank=i) for i in range(cfg.n_envs)])
    if cfg.reward_norm == "returns":
        # Scale by std of discounted returns, no centering, clip at 10.
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True,
                            clip_reward=10.0, gamma=cfg.gamma)

    policy_kwargs = dict(
        features_extractor_class=MazeCNNExtractor,
        features_extractor_kwargs=dict(features_dim=cfg.features_dim),
        lstm_hidden_size=cfg.lstm_hidden_size,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        optimizer_kwargs=dict(eps=1e-5),
    )

    model = LoggedRecurrentPPO(
        "MultiInputLstmPolicy",
        venv,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_coef,
        normalize_advantage=False,     # done once per rollout in train()
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,       # our train() applies it to the epoch-mean KL
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        device="cuda:0",
        verbose=0,
    )
    model.max_skipped_updates = cfg.max_skipped_updates

    logging.info(
        f"[TRAIN:{cfg.benchmark_name}/{cfg.algo}] {cfg.total_timesteps:,} steps | "
        f"rollout = {cfg.n_steps * cfg.n_envs:,} transitions | reward_norm={cfg.reward_norm}"
    )

    model.learn(total_timesteps=cfg.total_timesteps, callback=WandbTrainCallback(cfg))

    model.save(f"./{cfg.benchmark_name}_recppo_final")
    if cfg.reward_norm == "returns":
        venv.save(f"./{cfg.benchmark_name}_recppo_final_vecnormalize.pkl")
    logging.info("[DONE] Training complete.")
    wandb.finish()
    venv.close()


if __name__ == "__main__":
    main()
