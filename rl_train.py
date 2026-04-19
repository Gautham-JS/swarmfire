from envs.MultiAgentEnv import MultiAgentEnv
from envs.SingleAgentEnv import SingleAgentEnv
from policies import TemporalTransformerModel, CnnModel 

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from gymnasium.wrappers import TimeLimit
from wandb.integration.sb3 import WandbCallback

from collections import deque

import os
import argparse
import wandb
WANDB_API_KEY = "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"
os.environ['WANDB_API_KEY'] = WANDB_API_KEY


import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# Curriculum schedule definitions
# ─────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _smooth_weight(t, ramp_start, ramp_end, w_start, w_end):
    """Smoothly interpolate a weight between two timestep boundaries."""
    if t <= ramp_start:
        return w_start
    if t >= ramp_end:
        return w_end
    progress = (t - ramp_start) / (ramp_end - ramp_start)
    # Smooth via sigmoid centred at 0.5
    smooth = _sigmoid((progress - 0.5) * 8)
    smooth = (smooth - _sigmoid(-4)) / (_sigmoid(4) - _sigmoid(-4))  # renorm to [0,1]
    return w_start + (w_end - w_start) * smooth


PHASE_HARD = [
    # (timestep_start, weights_dict)
    (0,         {"exploration": 3.0, "fire_discovery": 0.0,
                 "fire_tracking": 0.0, "risk": 0.0}),
    (300_000,   {"exploration": 1.5, "fire_discovery": 3.0,
                 "fire_tracking": 0.0, "risk": 0.0}),
    (700_000,   {"exploration": 0.8, "fire_discovery": 2.0,
                 "fire_tracking": 2.0, "risk": 1.0}),
    (1_200_000, {"exploration": 0.5, "fire_discovery": 1.5,
                 "fire_tracking": 3.0, "risk": 2.5}),
]

# For smooth annealing each entry is:
# weight_name -> list of (timestep, target_value) breakpoints
SMOOTH_SCHEDULE = {
    "exploration":    [(0, 3.0), (400_000, 1.5), (800_000, 0.8), (1_500_000, 0.4)],
    "fire_discovery": [(0, 0.0), (200_000, 1.0), (600_000, 3.0), (1_000_000, 2.0)],
    "fire_tracking":  [(0, 0.0), (500_000, 0.0), (800_000, 1.5), (1_500_000, 3.5)],
    "risk":           [(0, 0.0), (600_000, 0.0), (900_000, 1.0), (1_500_000, 2.5)],
}

PERFORMANCE_GATES = [
    # (min_mean_reward_to_advance, weights_on_entry)
    (0.2,  {"exploration": 3.0, "fire_discovery": 0.0,
            "fire_tracking": 0.0, "risk": 0.0}),
    (0.5,  {"exploration": 1.5, "fire_discovery": 3.0,
            "fire_tracking": 0.0, "risk": 0.0}),
    (0.9,  {"exploration": 0.8, "fire_discovery": 2.0,
            "fire_tracking": 2.0, "risk": 1.0}),
    (1.0,  {"exploration": 0.5, "fire_discovery": 1.5,
            "fire_tracking": 3.0, "risk": 2.5}),
]


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

class MemoryResetCallback(BaseCallback):
    def _on_step(self):
        done_indices = [i for i, d in enumerate(self.locals["dones"]) if d]
        if done_indices:
            self.model.policy.features_extractor.reset_memory(done_indices)
        return True


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self._timestep_at_last_ep = 0

    def _on_step(self):
        for done in self.locals["dones"]:
            if done:
                self.episode_count += 1
                steps = self.num_timesteps - self._timestep_at_last_ep
                self._timestep_at_last_ep = self.num_timesteps
                if self.verbose and self.episode_count % 50 == 0:
                    print(
                        f"Episode {self.episode_count:>5} | "
                        f"Timestep {self.num_timesteps:>8} | "
                        f"Steps/ep {steps:>5}"
                    )
        return True

    def _on_training_end(self):
        print(f"\nTraining ended: {self.episode_count} episodes, "
              f"{self.num_timesteps} timesteps")


class HardPhaseCurriculumCallback(BaseCallback):
    """Jumps weights at fixed timestep thresholds."""

    def __init__(self, phases=PHASE_HARD, verbose=1):
        super().__init__(verbose)
        self.phases = phases          # list of (timestep, weights)
        self._current_phase = -1

    def _get_phase(self, t):
        phase = 0
        for i, (threshold, _) in enumerate(self.phases):
            if t >= threshold:
                phase = i
        return phase

    def _apply_weights(self, weights):
        for env in self.training_env.envs:
            # unwrap TimeLimit if present
            inner = env.env if hasattr(env, 'env') else env
            inner.set_reward_weights(weights)

    def _on_step(self):
        phase = self._get_phase(self.num_timesteps)
        if phase != self._current_phase:
            self._current_phase = phase
            _, weights = self.phases[phase]
            self._apply_weights(weights)
            if self.verbose:
                print(f"\n[Curriculum] Phase {phase} at step "
                      f"{self.num_timesteps}: {weights}")
        return True


class SmoothCurriculumCallback(BaseCallback):
    """Smoothly interpolates weights according to a per-weight schedule."""

    def __init__(self, schedule=SMOOTH_SCHEDULE, update_freq=500, verbose=1):
        super().__init__(verbose)
        self.schedule  = schedule
        self.update_freq = update_freq
        self._last_update = -1

    def _interpolate(self, t, breakpoints):
        """Linear interpolation between breakpoints, smooth at transitions."""
        if t <= breakpoints[0][0]:
            return breakpoints[0][1]
        if t >= breakpoints[-1][0]:
            return breakpoints[-1][1]
        for i in range(len(breakpoints) - 1):
            t0, v0 = breakpoints[i]
            t1, v1 = breakpoints[i + 1]
            if t0 <= t <= t1:
                return _smooth_weight(t, t0, t1, v0, v1)
        return breakpoints[-1][1]

    def _apply_weights(self, weights):
        for env in self.training_env.envs:
            inner = env.env if hasattr(env, 'env') else env
            inner.set_reward_weights(weights)

    def _on_step(self):
        t = self.num_timesteps
        if t - self._last_update < self.update_freq:
            return True
        self._last_update = t

        weights = {
            key: self._interpolate(t, bp)
            for key, bp in self.schedule.items()
        }
        self._apply_weights(weights)

        if self.verbose and t % 50_000 < self.update_freq:
            w_str = " | ".join(f"{k}={v:.2f}" for k, v in weights.items())
            print(f"\n[Curriculum] t={t}: {w_str}")
        return True


class PerformanceGatedCurriculumCallback(BaseCallback):
    """
    Advances through phases only when rolling mean reward exceeds a threshold.
    Prevents moving on before the agent has actually learned the current skill.
    """

    def __init__(self, gates=PERFORMANCE_GATES, window=200, verbose=1):
        super().__init__(verbose)
        self.gates   = gates    # list of (reward_threshold, weights)
        self.window  = window   # rolling window size in steps
        self._phase  = 0
        self._reward_buffer = []

    def _apply_weights(self, weights):
        print(f"Advancing performance gate phase to {self._phase}")
        for env in self.training_env.envs:
            inner = env.env if hasattr(env, 'env') else env
            inner.set_reward_weights(weights)

    def _on_step(self):
        # Accumulate rewards
        rewards = self.locals.get("rewards", [])
        self._reward_buffer.extend(rewards.tolist()
                                   if hasattr(rewards, 'tolist') else list(rewards))
        if len(self._reward_buffer) > self.window:
            self._reward_buffer = self._reward_buffer[-self.window:]

        # Apply current phase weights on first step
        if self.num_timesteps == 1:
            _, weights = self.gates[self._phase]
            self._apply_weights(weights)

        # Check if ready to advance
        if (self._phase < len(self.gates) - 1
                and len(self._reward_buffer) >= self.window):
            mean_reward = np.mean(self._reward_buffer)
            next_threshold, _ = self.gates[self._phase + 1]
            print(f"[ONSTEP] : Mean reward : {mean_reward}")
            # next_threshold here is the reward needed to advance
            # (reusing the tuple slot — see PERFORMANCE_GATES definition)
            current_gate_reward = self.gates[self._phase][0]
            if mean_reward >= current_gate_reward:
                self._phase += 1
                _, weights = self.gates[self._phase]
                self._apply_weights(weights)
                self._reward_buffer.clear()
                if self.verbose:
                    print(f"\n[Curriculum] Advanced to phase {self._phase} "
                          f"(mean reward {mean_reward:.2f} >= {current_gate_reward:.2f})"
                          f" at step {self.num_timesteps}")
        return True






import threading
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque


class LiveTrainingPlot(BaseCallback):
    """
    Plots policy loss, value loss, entropy, and mean reward in real time.
    Runs the matplotlib figure in a background thread so training is never blocked.
    """

    def __init__(
        self,
        window: int = 200,        # rolling window for smoothing
        update_freq: int = 100,   # update plot every N steps
        save_path: str = None,    # optional: save figure to disk periodically
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window      = window
        self.update_freq = update_freq
        self.save_path   = save_path

        # Data buffers — deque auto-drops old entries
        self.policy_losses  = deque(maxlen=5000)
        self.value_losses   = deque(maxlen=5000)
        self.entropies      = deque(maxlen=5000)
        self.approx_kls     = deque(maxlen=5000)
        self.ep_rewards     = deque(maxlen=500)
        self.ep_lengths     = deque(maxlen=500)
        self.timesteps      = deque(maxlen=5000)
        self.ep_timesteps   = deque(maxlen=500)

        self._current_ep_reward = 0.0
        self._current_ep_len    = 0
        self._last_plot_step    = 0

        # Thread coordination
        self._plot_data   = {}
        self._data_lock   = threading.Lock()
        self._stop_event  = threading.Event()
        self._plot_thread = None

    # ── Callback hooks ────────────────────────────────────────────────

    def _on_training_start(self):
        self._plot_thread = threading.Thread(
            target=self._plot_worker, daemon=True
        )
        self._plot_thread.start()

    def _on_step(self):
        # Accumulate episode reward
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])

        for r, d in zip(rewards, dones):
            self._current_ep_reward += float(r)
            self._current_ep_len    += 1
            if d:
                with self._data_lock:
                    self.ep_rewards.append(self._current_ep_reward)
                    self.ep_lengths.append(self._current_ep_len)
                    self.ep_timesteps.append(self.num_timesteps)
                self._current_ep_reward = 0.0
                self._current_ep_len    = 0

        return True

    def _on_rollout_end(self):
        """Called after each PPO update — pull losses from the logger."""
        logger = self.model.logger
        if logger is None:
            return

        # SB3 stores these in the logger's name_to_value dict
        kv = logger.name_to_value

        t = self.num_timesteps
        with self._data_lock:
            self.timesteps.append(t)

            pg  = kv.get("train/policy_gradient_loss", None)
            vf  = kv.get("train/value_loss",           None)
            ent = kv.get("train/entropy_loss",         None)
            kl  = kv.get("train/approx_kl",            None)

            if pg  is not None: self.policy_losses.append(float(pg))
            if vf  is not None: self.value_losses.append(float(vf))
            if ent is not None: self.entropies.append(float(ent))
            if kl  is not None: self.approx_kls.append(float(kl))

    def _on_training_end(self):
        self._stop_event.set()
        if self._plot_thread is not None:
            self._plot_thread.join(timeout=5.0)
        if self.save_path:
            self._save_figure()

    # ── Background plot worker ────────────────────────────────────────

    def _plot_worker(self):
        plt.ion()
        fig = plt.figure(figsize=(14, 8), facecolor="#1a1a2e")
        fig.canvas.manager.set_window_title("Training Monitor")

        gs   = gridspec.GridSpec(
            2, 3, figure=fig,
            hspace=0.45, wspace=0.35,
            left=0.07, right=0.97, top=0.92, bottom=0.08
        )
        ax_pg  = fig.add_subplot(gs[0, 0])
        ax_vf  = fig.add_subplot(gs[0, 1])
        ax_ent = fig.add_subplot(gs[0, 2])
        ax_kl  = fig.add_subplot(gs[1, 0])
        ax_rew = fig.add_subplot(gs[1, 1])
        ax_len = fig.add_subplot(gs[1, 2])

        axes_cfg = [
            (ax_pg,  "Policy gradient loss",  "#5DCAA5"),
            (ax_vf,  "Value loss",             "#7F77DD"),
            (ax_ent, "Entropy loss",           "#EF9F27"),
            (ax_kl,  "Approx KL",              "#F0997B"),
            (ax_rew, "Episode reward",         "#85B7EB"),
            (ax_len, "Episode length",         "#C0DD97"),
        ]

        for ax, title, _ in axes_cfg:
            ax.set_facecolor("#0d0d1a")
            ax.set_title(title, color="white", fontsize=9, pad=4)
            ax.tick_params(colors="gray", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        fig.suptitle("Training monitor", color="white",
                     fontsize=11, fontweight="500", y=0.97)
        plt.show()

        while not self._stop_event.is_set():
            with self._data_lock:
                pg_data  = list(self.policy_losses)
                vf_data  = list(self.value_losses)
                ent_data = list(self.entropies)
                kl_data  = list(self.approx_kls)
                rew_data = list(self.ep_rewards)
                len_data = list(self.ep_lengths)
                rew_t    = list(self.ep_timesteps)

            datasets = [pg_data, vf_data, ent_data, kl_data, rew_data, len_data]

            for (ax, title, color), data in zip(axes_cfg, datasets):
                if len(data) < 2:
                    continue
                ax.cla()
                ax.set_facecolor("#0d0d1a")
                ax.set_title(title, color="white", fontsize=9, pad=4)
                ax.tick_params(colors="gray", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333")

                x = np.arange(len(data))
                y = np.array(data)

                # Raw values in faded color
                ax.plot(x, y, color=color, alpha=0.25, linewidth=0.8)

                # Rolling mean
                if len(y) >= self.window:
                    kernel = np.ones(self.window) / self.window
                    smooth = np.convolve(y, kernel, mode="valid")
                    x_s    = x[self.window - 1:]
                    ax.plot(x_s, smooth, color=color,
                            linewidth=1.5, label=f"mean({self.window})")

                # Latest value annotation
                ax.text(
                    0.98, 0.95, f"{y[-1]:.4f}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    color=color, fontsize=8,
                    bbox=dict(facecolor="#00000088",
                              edgecolor="none", pad=2)
                )

            fig.suptitle(
                f"Training monitor  —  {self.num_timesteps:,} steps",
                color="white", fontsize=11, fontweight="500", y=0.97
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            # Save periodically if path given
            if self.save_path and self.num_timesteps % 50_000 < 500:
                self._fig_ref = fig
                self._save_figure()

            self._stop_event.wait(timeout=2.0)   # update every 2 seconds

        plt.close(fig)

    def _save_figure(self):
        if hasattr(self, "_fig_ref") and self.save_path:
            try:
                self._fig_ref.savefig(
                    self.save_path,
                    facecolor="#1a1a2e",
                    dpi=120,
                    bbox_inches="tight"
                )
            except Exception:
                pass


class WandbLoggingCallback(BaseCallback):
    """
    Clean SB3 → Weights & Biass logger.

    Logs:
    - policy loss
    - value loss
    - entropy
    - KL divergence
    - episode reward
    - episode length
    - timesteps

    No plotting, no threads, minimal overhead.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        self._current_ep_reward = 0.0
        self._current_ep_len = 0
        self.episode_rewards = deque(maxlen=100)

    # ─────────────────────────────────────────────
    # STEP: track episode stats
    # ─────────────────────────────────────────────
    def _on_step(self):
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for r, d in zip(rewards, dones):
            self._current_ep_reward += float(r)
            self._current_ep_len += 1

            if d:
                self.episode_rewards.append(self._current_ep_reward)
                mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)


                wandb.log({
                    "episode/reward": self._current_ep_reward,
                    "episode/length": self._current_ep_len,
                    "episode/mean_reward_100_steps": mean_reward,
                    "timesteps": self.num_timesteps,
                })

                self._current_ep_reward = 0.0
                self._current_ep_len = 0

        return True

    # ─────────────────────────────────────────────
    # ROLLOUT END: grab training losses
    # ─────────────────────────────────────────────
    def _on_rollout_end(self):
        logger = self.model.logger
        if logger is None:
            return

        kv = logger.name_to_value

        log_dict = {
            "timesteps": self.num_timesteps
        }

        # Pull SB3 internal metrics
        mapping = {
            "train/policy_gradient_loss": "loss/policy",
            "train/value_loss": "loss/value",
            "train/entropy_loss": "loss/entropy",
            "train/approx_kl": "loss/approx_kl",
            "train/clip_fraction": "train/clip_fraction",
            "train/explained_variance": "train/explained_variance",
        }

        for sb3_key, wandb_key in mapping.items():
            val = kv.get(sb3_key)
            if val is not None:
                log_dict[wandb_key] = float(val)

        wandb.log(log_dict)

    # ─────────────────────────────────────────────
    # TRAINING END
    # ─────────────────────────────────────────────
    def _on_training_end(self):
        wandb.finish()



# ─────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────

N_ENVS       = 1
ITER_LIMIT   = 500
WORLD_SIZE   = (512, 512)
N_AGENTS     = 1
CURRICULUM   = "performance"   # "hard" | "smooth" | "performance"


def make_env(rank, is_eval=False):
    def _init():
        env = SingleAgentEnv(
            n_agents=N_AGENTS,
            world_size=WORLD_SIZE,
            start_positions=[(256, 256)],
            render_mode="human" if (rank == 0 and not is_eval) else "rgb_array",
            sample_interval=50  if (rank == 0 and not is_eval) else 999999,
            save_interval=50   if (rank == 0 and not is_eval) else 999999,
            seed=34,
            fixed_seed=False,
            is_vid_out=(rank == 0 and not is_eval),
            vid_id="firescout_smooth",
            vid_base_path="/home/s3400220/swarmfire/vids",
            # Start with pure exploration weights — curriculum callback takes over
            phase_weights={"exploration": 5.0, "fire_discovery": 9.8,
                           "fire_tracking": 6.0, "risk": 5.0},
        )
        return TimeLimit(env, max_episode_steps=ITER_LIMIT)
    return _init


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='rl_train.py',
        description='Trains the DRL model and saves/loads from checkpoint',
        epilog=''
    )

    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-r', '--render-mode')
    parser.add_argument('-s', '--sample-interval')


    args = parser.parse_args()

    reset_timesteps = True
    f_checkpoint = None
    f_vecnormalize = None
    is_ckpt_load = False
    if args.checkpoint is not None:
        print(f"Using checkpoint {args.checkpoint}")
        f_checkpoint = f"firescout_{args.checkpoint}_steps.zip"
        f_vecnormalize = f"firescout_vecnormalize_{args.checkpoint}_steps.pkl"
        reset_timesteps = False
        is_ckpt_load = True

    


    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    wandb.init(project="thesis-drl")

    # Training env
    train_env = DummyVecEnv([make_env(0)])
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True,
                             clip_reward=10.0)

    # Eval env — same config, no rendering, no curriculum interference
    eval_env = DummyVecEnv([make_env(0, is_eval=True)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False,
                            training=False)

    # Pick curriculum strategy
    if CURRICULUM == "hard":
        curriculum_cb = HardPhaseCurriculumCallback(phases=PHASE_HARD, verbose=1)
    elif CURRICULUM == "smooth":
        curriculum_cb = SmoothCurriculumCallback(
            schedule=SMOOTH_SCHEDULE, update_freq=500, verbose=1
        )
    elif CURRICULUM == "performance":
        curriculum_cb = PerformanceGatedCurriculumCallback(
            gates=PERFORMANCE_GATES, window=200, verbose=1
        )
    else:
        raise ValueError(f"Unknown curriculum: {CURRICULUM}")



    # Model
    if is_ckpt_load:
        train_env = VecNormalize.load(f_vecnormalize, train_env)
        train_env.training = True       # re-enable stat updates for continued training
        train_env.norm_reward = True    # make sure this matches your original training config
        print(f"[MAIN] : Loading checkpoint file {f_checkpoint}")
        model = RecurrentPPO.load(f_checkpoint, env=train_env)
    else:
        temporal_xformer_policy_kwargs = dict(
            features_extractor_class=TemporalTransformerModel.TemporalTransformerExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                n_heads=4,
                n_layers=3,
                memory_len=8,   # how many past frames to attend over
            ),
        )
        
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            train_env,
            policy_kwargs=dict(
                features_extractor_class=CnnModel.PlainCNNExtractor,
                features_extractor_kwargs=dict(features_dim=256),
                lstm_hidden_size=512,
                n_lstm_layers=4,
                shared_lstm=False,        # separate LSTM for actor and critic
                enable_critic_lstm=True,
                normalize_images=False,
            ),
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,              # steps per env per rollout
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            max_grad_norm=0.5,
        )

    callbacks = [
        TrainingMonitorCallback(verbose=1),
        # curriculum_cb,
        CheckpointCallback(
            save_freq=50_000,
            save_path="./checkpoints/",
            name_prefix="firescout",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="./best_model/",                                                                       
            log_path="./logs/",
            eval_freq=50_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        WandbLoggingCallback()
    ]

    model.learn(
        total_timesteps=5_000_000,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    model.save("./firescout_final.zip")
    train_env.save("./firescout_vecnorm.pkl")
    train_env.close()
    eval_env.close()
