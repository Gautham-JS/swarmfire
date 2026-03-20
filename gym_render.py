import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec

import functools

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import stable_baselines3 as sb3

from utils import Generators, Viewpoint
from agents.Drone import Drone


class MultiAgentEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}

    def __init__(self, size: tuple, start_pos: tuple, render_mode: str = None, n_agents=5):
        super().__init__()
        self.size = size
        self.vp_size = 64
        self.step_size = 10
        self.render_mode = render_mode
        self.n_agents = n_agents
        self.start_pos = start_pos

        # PettingZoo requires these two attributes
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        self.agent_position = np.array(start_pos, dtype=np.float32)
        self.map_generator = Generators.FuelMapGenerator(self.size)
        self.map = self.create_random_map()
        self.visited_map = np.zeros(size, dtype=np.bool)

        self.agent_map = {}

        # Rendering state
        self._fig = None
        self._axes = None
        self._last_obs = {}
        self._last_rewards = {}
        self._last_views = {}
        self._step_count = 0

        self._init_agents()

    # PettingZoo requires these as methods, not a single shared space
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return gym.spaces.Dict({
            "viewport": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.vp_size, self.vp_size),
                dtype=np.float32
            ),
            "position": gym.spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([self.size[1] - 1, self.size[0] - 1], dtype=np.int32),
                dtype=np.int32
            ),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.MultiDiscrete([3, 3])

    def _init_agents(self):
        self.agent_map = {}
        for i in range(self.n_agents):
            agent_id = f"agent_{i}"
            self.agent_map[agent_id] = Drone(
                id=agent_id,
                pos={'x': self.start_pos[0], 'y': self.start_pos[1], 'z': 0},
            )
            self._last_rewards[agent_id] = 0.0
            self._last_views[agent_id] = np.zeros((self.vp_size, self.vp_size), dtype=np.float32)



    def create_random_map(self, seed=None):
        return self.map_generator.create_mask(0.001, 0.003, seed=seed)

    def extract_viewpoint(self, x, y, z):
        view, recently_visited, delta_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self.map, self.visited_map, (x, y), size=self.vp_size)
        self.visited_map = recently_visited
        return view, delta_mask

    def get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size

    # --- step and reset return flat dicts keyed by agent_id, same as before ---

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        for agent_id, action in actions.items():
            ax, ay = action[0], action[1]
            dx, dy = self.get_position_delta_from_action(ax), self.get_position_delta_from_action(ay)

            agent_pos = self.agent_map[agent_id].get_position_array()

            nx = int(np.clip(agent_pos[0] + dx, 0, self.size[1] - 1))
            ny = int(np.clip(agent_pos[1] + dy, 0, self.size[0] - 1))

            view, delta_mask = self.extract_viewpoint(nx, ny, 0)

            obs[agent_id] = {
                "viewport": delta_mask,
                "position": np.array([nx, ny], dtype=np.int32),
            }

            self.agent_map[agent_id].set_position({"x": nx, "y": ny, "z": 0})

            rewards[agent_id]    = float(np.sum(view) / (self.vp_size*self.vp_size)) + self._last_rewards[agent_id]

            terminated[agent_id] = False
            if agent_pos[0] <= 0 or agent_pos[1] <=0 or agent_pos[0] >= self.size[0] - 1 or agent_pos[1] >= self.size[1] - 1:
                terminated[agent_id] = True
            if self._step_count > 50:
                terminated[agent_id] = True
                
            truncated[agent_id]  = False
            infos[agent_id]      = {}

        self._last_obs     = obs
        self._last_rewards = rewards
        self._step_count  += 1

        # PettingZoo: update self.agents to only living agents
        self.agents = [a for a in self.agents if not terminated.get(a) and not truncated.get(a)]

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminated, truncated, infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.map = self.create_random_map(seed=seed)
        self._step_count = 0

        obs, infos = {}, {}
        for aid, a in self.agent_map.items():
            a.set_position({'x': self.start_pos[0], 'y': self.start_pos[1], 'z': 0})
            v, _ = self.extract_viewpoint(self.start_pos[0], self.start_pos[1], 0)
            obs[aid] = {
                "viewport": v,
                "position": np.array(self.start_pos[:2], dtype=np.int32),
            }
            infos[aid] = {}

        self._last_obs     = obs
        self._last_rewards = {aid: 0.0 for aid in self.agent_map}

        if self.render_mode == "human":
            self._init_figure()
            self.render()

        return obs, infos
    
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _init_figure(self):
        """Create the figure layout once and reuse it across frames."""
        n_agents = max(len(self.agent_map), 1)

        # Layout: left column = full map, right columns = per-agent viewports
        self._fig = plt.figure(
            figsize=(4 + 3 * n_agents, 5),
            facecolor="#1a1a2e"
        )
        gs = GridSpec(
            2, 1 + n_agents,
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08
        )

        self._ax_map = self._fig.add_subplot(gs[:, 0])          # full map spans both rows
        self._ax_viewports = [
            (self._fig.add_subplot(gs[0, i + 1]),               # viewport image
             self._fig.add_subplot(gs[1, i + 1]))               # reward bar
            for i in range(n_agents)
        ]

        self._fig.canvas.manager.set_window_title("SingleAgentEnv")
        plt.ion()
        plt.show()

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._fig is None:
            self._init_figure()

        agents  = list(self.agent_map.keys())
        n_agents = len(agents)

        # ── Full map ──────────────────────────────────────────────────
        ax = self._ax_map
        ax.cla()
        ax.imshow(
            self.map, cmap="YlOrRd", origin="upper",
            vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        ax.set_title("Map", color="white", fontsize=9, pad=4)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Draw each agent's position and viewport footprint on the map
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_agents, 1)))
        for i, aid in enumerate(agents):
            pos = self._last_obs.get(aid, {}).get("position")
            if pos is None:
                continue
            x, y = int(pos[0]), int(pos[1])
            half = self.vp_size // 2

            # Viewport rectangle
            rect = patches.Rectangle(
                (x - half, y - half), self.vp_size, self.vp_size,
                linewidth=1.2, edgecolor=colors[i], facecolor="none",
                linestyle="--", alpha=0.7
            )
            ax.add_patch(rect)

            # Agent marker
            ax.scatter(x, y, s=60, color=colors[i], zorder=5,
                       edgecolors="white", linewidths=0.6)
            ax.annotate(
                aid, (x, y), textcoords="offset points", xytext=(4, 4),
                color=colors[i], fontsize=6, fontweight="bold"
            )

        # ── Per-agent viewports ───────────────────────────────────────
        max_reward = max((v for v in self._last_rewards.values()), default=1) or 1

        for i, aid in enumerate(agents):
            if i >= len(self._ax_viewports):
                break
            ax_vp, ax_bar = self._ax_viewports[i]

            # Viewport image
            ax_vp.cla()
            obs = self._last_obs.get(aid, {})
            vp  = obs.get("viewport")
            if vp is not None:
                ax_vp.imshow(
                    vp, cmap="YlOrRd", origin="upper",
                    vmin=0.0, vmax=1.0, interpolation="nearest"
                )
            ax_vp.set_title(aid, color=colors[i], fontsize=8, pad=3)
            ax_vp.set_facecolor("#0d0d1a")
            ax_vp.tick_params(left=False, bottom=False,
                              labelleft=False, labelbottom=False)
            for spine in ax_vp.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(1.5)

            # Reward bar
            ax_bar.cla()
            reward = self._last_rewards.get(aid, 0.0)
            ax_bar.barh(
                0, reward, color=colors[i], alpha=0.8,
                height=0.5, zorder=3
            )
            ax_bar.barh(
                0, max_reward, color="#333", alpha=0.4,
                height=0.5, zorder=2
            )
            ax_bar.set_xlim(0, max_reward * 1.05)
            ax_bar.set_ylim(-0.5, 0.5)
            ax_bar.set_yticks([])
            ax_bar.set_facecolor("#0d0d1a")
            ax_bar.tick_params(colors="gray", labelsize=6)
            ax_bar.set_xlabel("reward", color="gray", fontsize=6)
            ax_bar.text(
                max_reward * 1.02, 0, f"{reward:.1f}",
                va="center", ha="left", color="white", fontsize=7
            )
            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#333")

        # ── Figure title ──────────────────────────────────────────────
        self._fig.suptitle(
            f"Step {self._step_count}",
            color="white", fontsize=11, fontweight="bold", y=0.97
        )

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            return None

        # rgb_array mode — render to numpy array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3]   # drop alpha → H×W×3 uint8
        return img

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig  = None
            self._axes = None






import cv2
from pettingzoo.test import parallel_api_test
import stable_baselines3 as sb3
import supersuit as ss



env = MultiAgentEnv(size=(512, 512), start_pos=(256, 256), render_mode="rgb_array")

# Replaces check_env
parallel_api_test(env, num_cycles=1000)



# import supersuit as ss
# # Vectorise all agents as independent learners
# vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
# vec_env = ss.concat_vec_envs_v1(vec_env, 1, base_class="stable_baselines3")
# model = sb3.PPO("MultiInputPolicy", vec_env)
# model.learn(total_timesteps=100_000)




from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
import supersuit as ss
import numpy as np

class PettingZooVecEnvWrapper(VecEnv):
    """Thin wrapper that makes a SuperSuit VecEnv SB3-compatible."""
    
    def __init__(self, env):
        self.env = env
        n_envs = env.num_envs
        observation_space = env.observation_space
        action_space = env.action_space
        super().__init__(n_envs, observation_space, action_space)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, dones, infos = self.env.step(self._actions)
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        pass

class Sb3ShimWrapper(PettingZooVecEnvWrapper):
    metadata = {'render_modes': ['human', 'files', 'none'], "name": "Sb3ShimWrapper-v0"}

    def __init__(self, env):
        super().__init__(env)

    def reset(self, seed=None, options=None):
        return self.env.reset()[0]

    def step_async(self, actions):
        self.env.step_async(actions)

    def step_wait(self):
        result = self.env.step_wait()
        
        if len(result) == 5:
            # New gymnasium API: obs, rewards, terminated, truncated, infos
            obs, rewards, terminated, truncated, infos = result
            dones = np.logical_or(terminated, truncated)
            return obs, rewards, dones, infos
        else:
            # Old gym API: obs, rewards, dones, infos
            return result


from stable_baselines3.common.callbacks import BaseCallback
import cv2

class RenderCallback(BaseCallback):
    """Renders a separate env every `render_every_n_steps` steps."""

    def __init__(self, render_every_n_steps: int = 1000, n_render_steps: int = 100, verbose=0):
        super().__init__(verbose)
        self.render_every_n_steps = render_every_n_steps
        self.n_render_steps = n_render_steps
        self._render_env = None

    def _init_render_env(self):
        """Create a separate env for rendering — never used for training."""
        env = MultiAgentEnv(
            size=(512, 512),
            start_pos=(256, 256),
            render_mode="human",  # ← human mode for live display
            n_agents=5
        )
        return env

    def _on_step(self) -> bool:
        if self.num_timesteps % self.render_every_n_steps == 0:
            if self._render_env is None:
                self._render_env = self._init_render_env()

            print(f"\n[RenderCallback] Rendering at step {self.num_timesteps}...")
            obs, _ = self._render_env.reset()

            for _ in range(self.n_render_steps):
                # Use the trained model to pick actions
                actions = {}
                for aid in self._render_env.agents:
                    obs_tensor = {
                        k: v[np.newaxis, :]  # add batch dim
                        for k, v in obs[aid].items()
                    }
                    action, _ = self.model.predict(obs_tensor, deterministic=True)
                    actions[aid] = action.squeeze()

                obs, _, terminated, truncated, _ = self._render_env.step(actions)

                # Remove done agents
                obs = {aid: o for aid, o in obs.items()
                       if not terminated.get(aid) and not truncated.get(aid)}

                if not self._render_env.agents:
                    break

        return True  # returning False would stop training

    def _on_training_end(self):
        if self._render_env is not None:
            self._render_env.close()




# Usage
raw_env = MultiAgentEnv(size=(512, 512), start_pos=(256, 256), render_mode="rgb_array", n_agents=5)
vec_env = ss.pettingzoo_env_to_vec_env_v1(raw_env)
env = Sb3ShimWrapper(vec_env)

# model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-3, batch_size=128, n_steps=128)
# model.learn(total_timesteps=10000)
# env.close()



env = raw_env

# Training loop
for episode in range(10):
    obs, _ = env.reset()
    episode_reward = {aid: 0.0 for aid in env.agents}
    is_exit = False

    while env.agents:  # PettingZoo: loop until no agents remain
        actions = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rewards, terminated, truncated, infos = env.step(actions)
        for aid, r in rewards.items():
            episode_reward[aid] += r
        
        done = all(terminated[aid] or truncated[aid] for aid in env.agent_map)
        frame = env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("SingleAgentEnv", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            is_exit = True
            break
    
    if is_exit:
        break

    print(f"Episode {episode} | rewards: {episode_reward}")



cv2.destroyAllWindows()
env.close()