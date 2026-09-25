"""
Adapter so DeepMind/jurgisp's Memory Maze (a legacy-gym env with a
Dict(image=...) observation and a Discrete action space) can be dropped
into the existing TrXLExtractor / SubprocVecEnv pipeline unchanged.

Two gaps to bridge:
  1. API: memory_maze registers under old `gym` (reset() -> obs,
     step() -> obs, reward, done, info), but the rest of this codebase is
     gymnasium (reset() -> obs, info; step() -> ..., terminated, truncated,
     info). We translate by hand below (no shimmy dependency required).
  2. Observation shape: TrXLExtractor expects a Dict with "viewport"
     (C,H,W image) AND "positions" (world x,y). Memory Maze exposes only
     an image -- there is no ground-truth coordinate readout in the base
     task. "positions" is filled with a constant zero vector instead of
     dropped, so the extractor's Dict observation_space contract doesn't
     change across benchmarks.

IMPORTANT: pair this wrapper with `extractor_kwargs={"use_spatial_bias": False}`
when building the agent for this benchmark. use_spatial_bias multiplies the
CNN features by sigmoid(Linear(positions)); with positions frozen at zero
that's just a fixed per-channel scalar gate learned from a constant input,
i.e. dead weight -- turning it off removes those unused parameters instead
of leaving them to do nothing.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import gym as gym_legacy          # the legacy `gym` package memory_maze registers into
import memory_maze                # noqa: F401 -- import registers the MemoryMaze-* env ids


class MemoryMazeEnv(gym.Env):
    """
    env_id examples: "memory_maze:MemoryMaze-9x9-v0", "...-11x11-v0",
    "...-13x13-v0", "...-15x15-v0". See the memory-maze README for the
    full list (including -HD- and -ExtraObs- variants).
    """

    metadata = {"render_modes": []}

    def __init__(self, env_id: str = "memory_maze:MemoryMaze-9x9-v0", pos_dim: int = 2):
        super().__init__()
        self._env = gym_legacy.make(env_id)
        self._pos_dim  = pos_dim
        self._zero_pos = np.zeros(pos_dim, dtype=np.float32)

        # Depending on the installed memory_maze version / env id, the base
        # env's observation_space is either Dict(image=Box(...), ...) or,
        # for pure-image variants (and apparently this version's default),
        # a bare Box(0,255,(H,W,3),uint8). Handle both so this wrapper
        # doesn't silently break on a version bump.
        raw_obs_space = self._env.observation_space
        self._obs_is_dict = hasattr(raw_obs_space, "spaces")
        img_space = raw_obs_space.spaces["image"] if self._obs_is_dict else raw_obs_space
        h, w, c = img_space.shape
        # Memory Maze's image comes back as raw uint8 pixels (0-255). The
        # rest of this pipeline (CNN -> spectral_norm ff -> LayerNorm) was
        # built assuming roughly unit-scale input, since the wildfire env
        # already normalizes its own "viewport" internally. Feeding 0-255
        # values straight through blows up activations and eventually
        # shows up as NaN a few PPO updates in -- so normalize to [0,1]
        # here, once, at the source.
        self.observation_space = spaces.Dict({
            "viewport":  spaces.Box(low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32),
            "positions": spaces.Box(low=-1.0, high=1.0, shape=(pos_dim,), dtype=np.float32),
        })

        # Present as length-1 MultiDiscrete so it plugs into the existing
        # per-dimension actor_heads / action_nvec machinery unchanged, even
        # though there's only one discrete action dimension here.
        n = self._env.action_space.n
        self.action_space = spaces.MultiDiscrete([n])

    def _convert_obs(self, obs):
        img = obs["image"] if self._obs_is_dict else obs  # (H,W,3) uint8, 0-255
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0  # -> (3,H,W), [0,1]
        return {
            "viewport":  img,
            "positions": self._zero_pos,
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
        obs = self._env.reset()
        return self._convert_obs(obs), {}

    def step(self, action):
        a = int(np.asarray(action).reshape(-1)[0])
        obs, reward, done, info = self._env.step(a)
        # Memory Maze has no separate truncation signal of its own; the
        # outer TimeLimit wrapper applied in make_env_fn supplies truncated.
        return self._convert_obs(obs), float(reward), bool(done), False, info

    def close(self):
        self._env.close()

    def render(self):
        return None