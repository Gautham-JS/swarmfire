from envs.MultiAgentEnv import MultiAgentEnv
from policies import TemporalTransformerModel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from gymnasium.wrappers import TimeLimit

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
        env = MultiAgentEnv(
            n_agents=N_AGENTS,
            world_size=WORLD_SIZE,
            start_positions=[(256, 256)],
            render_mode="human" if (rank == 0 and not is_eval) else "rgb_array",
            sample_interval=100  if (rank == 0 and not is_eval) else 999999,
            save_interval=100   if (rank == 0 and not is_eval) else 999999,
            seed=34,
            fixed_seed=False,
            is_vid_out=(rank == 0 and not is_eval),
            vid_id="firescout_smooth",
            vid_base_path="./vids/",
            # Start with pure exploration weights — curriculum callback takes over
            phase_weights={"exploration": 3.0, "fire_discovery": 0.0,
                           "fire_tracking": 0.0, "risk": 0.0},
        )
        return TimeLimit(env, max_episode_steps=ITER_LIMIT)
    return _init


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

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
    model = PPO(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=dict(
            features_extractor_class=TemporalTransformerModel.TemporalTransformerExtractor,
            features_extractor_kwargs=dict(
                features_dim=256,
                n_heads=4,
                n_layers=3,
                memory_len=8,
                n_envs=N_ENVS,
            ),
        ),
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.10,       # high entropy early — curriculum will tighten task
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
    )

    callbacks = [
        MemoryResetCallback(),
        TrainingMonitorCallback(verbose=1),
        curriculum_cb,
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