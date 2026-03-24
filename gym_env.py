import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import cv2

from utils import Generators, Viewpoint
from agents import Drone



class MultiAgentEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}

    def __init__(self, n_agents, world_size, start_positions:list=None, iter_limit=100, seed = None, env_id="MultiAgentEnv", render_mode="human"):
        super().__init__()

        self.n_agents = n_agents
        self.world_size = world_size
        self.iter_limit = iter_limit
        self.seed = seed
        self.render_mode = render_mode
        self.start_poss = start_positions
        self.env_id = env_id


        self.vp_size = 64
        self.step_size = 4

        # sort of flattening the multi agent space into a 1D single agent space.
        # TODO : Encoder can make it input shape agnostic
        self.actions_per_agent = 2
        self.n_actions = self.n_agents * self.actions_per_agent
        self.action_space = gym.spaces.Box(
            low=0,
            high=2,
            shape=(self.n_agents * 2,),
            dtype=np.uint8
        )

        # once again, flattening the multi agent observations into a single world level observation.
        # TODO: Encoder should once again make it shape agnostic
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255,
            shape=(2, self.world_size[0], self.world_size[1]), 
            dtype=np.uint8
        )

        self.world_gen = Generators.FuelMapGenerator(self.world_size)
        self.view_acc = Viewpoint.IncrementalViewAccumulator(self.world_size, 2)


        # vars initialized by reset fn
        self.map = None             # global map HxWx1
        self.visited_map = None
        self.agents = None          # indexed list of N agents
        self._obs_hsitory = None    # history of all rewards n_agents x n_iters
        self._reward_history = None 
        self._view_history = None
        self._pos_history = None
        self._agent_positions = None
        self._step_count = 0

        # render states
        self._fig = None
        self._axes = None

    def get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size
        
    def extract_viewpoint(self, x, y):
        fuel_view, recently_visited_fuel, delta_fuel_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self.map[:, :, 0], self.visited_map, (x, y), size=self.vp_size)
        fire_view, recently_visited_fire, delta_fire_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self.map[:, :, 1], self.visited_map, (x, y), size=self.vp_size)
        self.visited_map = recently_visited_fuel

        view, deltas = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32), np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        view[:, :, 0], view[:, :, 1] = fuel_view, fire_view

        #print(f"Delta fuel shape : {delta_fuel_mask.shape} | View shape : {fuel_view.shape}")

        deltas[:, :, 0], deltas[:, :, 1] = delta_fuel_mask, delta_fire_mask
        return view, deltas

    def reset(self, seed = None, options = None):
        self.close()
        self._obs_hsitory, self._agent_positions, self._reward_history, self._view_history, self._pos_history = [], [], [], [], []
        self._step_count = 0
        
        self.map = self.world_gen.create_map(0.001, 0.003, seed=seed)
        self.visited_map = np.zeros(self.world_size, dtype=np.bool)
        self.view_acc.reset()
        
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        if self.start_poss is None:
            self._agent_positions = [(np.random.randint(0, self.world_size[0]), np.random.randint(0, self.world_size[1])) for _ in range(self.n_agents)]
        else:
            self._agent_positions = self.start_poss

        obs, infos = np.zeros((2, *self.world_size), dtype=np.uint8), {}
        return obs, infos
    
    def evaluate_risk_map(self, map):
        """
            Sort of the heart of the logic

            2 key layers:
                fuel_map: map[:, :, 0]
                fire_map: map[:, :, 1]
            
            key functions needed:
                sample_fire_idxs(fire) -> np.ndarray of shape (Nx2), coordinates of each fire sample. Resolution could be chosen too ig.
                    -> sample fire map as grids
                    -> for each patch:
                        -> if mask activation exists:
                            -> calculate centroid of the detection in patch
                            -> transform centroid to world map coordinates\
                            -> store centroid coord as fire sample

                sort_fuels_by_risk(fire_samples, fuels_map, eval_radius) -> np.ndarray of same size as map, with pixel values having range 0-1 representing risk of that fuel.:
                    -> for each fire sample,
                        -> extract a radius around the point.
                        -> for each fuel coordinate in that radius:
                            -> if nonzero value:
                                -> d = eucledian_dist(fuel_coord, fire_coord)
                                -> fuel_val, fire_val = fuel_map[fuel_coord], fire_map[fire_coord]
                                -> risk = (1/d+1) * (fuel_val + fire_val)



        """
    
    def step(self, action):
        reward, terminated, truncated, infos = 0.0, False, False, {}
        loc_pos_history = []
        loc_view_history = []
        penality = 0.1
        for i in range(self.n_agents):
            agent_actions = action[self.actions_per_agent*i : self.actions_per_agent*(i+1)]   # for i-th agent, the corresponding action would be [2*i:2*1+1]
            agent_pos:tuple = self._agent_positions[i]
            
            dx, dy = self.get_position_delta_from_action(agent_actions[0]), self.get_position_delta_from_action(agent_actions[1])
            px, py = int(agent_pos[0] + dx), int(agent_pos[1] + dy)
            
            if px >= self.world_size[0] or px < 0 or py >= self.world_size[1] or py <= 0:
                #print(f"Agent {self.agents[i]} hit bounds")
                self._agent_positions[i] = (px, py)
                
                loc_pos_history.append((px - dx, py - dx))
                loc_view_history.append(np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32))
                penality+=1
                continue

            view, delta_view = self.extract_viewpoint(px, py)
            self.view_acc.accumulate(view, (px, py), self.vp_size)

            self._agent_positions[i] = (px, py)
            
            loc_pos_history.append((px, py))
            loc_view_history.append(view)

        if penality == self.n_agents:
            terminated=True

        obs = self.view_acc.get_scene()
        fuel_obs, fire_obs = obs[:, :, 0], obs[:, :, 1] 
        
        fuel_reward = (np.sum(fuel_obs) / (self.vp_size * self.vp_size) )
        fire_reward = (np.sum(fire_obs) / (self.vp_size * self.vp_size) )

        penality_scale = 5
        fire_scale = 10
        total_reward = fuel_reward + (fire_scale*fire_reward)
        total_reward = total_reward - (penality_scale * penality)

        self._reward_history.append(total_reward)
        self._obs_hsitory.append(obs)
        self._view_history.append(loc_view_history)
        self._pos_history.append(loc_pos_history)

        self._step_count+=1

        if self.render_mode == "human":
            self.render()

        if self._step_count > self.iter_limit:
            terminated = True

        obs = np.transpose(obs, (2, 0, 1))      # → (2, H, W)
        obs = (obs * 255).clip(0, 255).astype(np.uint8)  # if scene is float [0,1]

        return obs, total_reward, terminated, truncated, infos
    
    def _init_figure(self):
        """Create the figure layout once and reuse it across frames."""
        n_agents = max(self.n_agents, 1)

        # Layout: left column = full map, right columns = per-agent viewports
        # self._fig = plt.figure(
        #     figsize=(4 + 3 * n_agents, 5),
        #     facecolor="#1a1a2e"
        # )
        # gs = GridSpec(
        #     2, 1 + n_agents,
        #     figure=self._fig,
        #     hspace=0.4, wspace=0.35,
        #     left=0.06, right=0.97, top=0.88, bottom=0.08
        # )

        self._fig = plt.figure(
            figsize=(4 + 3 + 3 * n_agents, 5),  # extra 3 for obs column
            facecolor="#1a1a2e"
        )
        gs = GridSpec(
            2, 2 + n_agents,          # 2 fixed cols (map + obs) + n_agents cols
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08
        )


        self._ax_map = self._fig.add_subplot(gs[:, 0])          # full map spans both rows
        self._ax_obs = self._fig.add_subplot(gs[:, 1])
        self._ax_viewports = [
            (self._fig.add_subplot(gs[0, i + 2]),               # viewport image
             self._fig.add_subplot(gs[1, i + 2]))               # reward bar
            for i in range(n_agents)
        ]

        self._fig.canvas.manager.set_window_title(self.env_id)
        plt.ion()
        plt.show()
    
    def _composite_rgb_map(self, shape, r_channel, g_channel, b_channel):
        if r_channel is None:
            r_channel = np.zeros(shape, dtype=np.float32)
        if g_channel is None:
            g_channel = np.zeros(shape, dtype=np.float32)
        if b_channel is None:
            b_channel = np.zeros(shape, dtype=np.float32)
        
        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
        canvas[:, :, 0] = r_channel
        canvas[:, :, 1] = g_channel
        canvas[:, :, 2] = b_channel
        return canvas

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._fig is None:
            print(f"Initializing figures")
            self._init_figure()

        agents  = list(self.agents)
        n_agents = len(agents)

        # ── Full map ──────────────────────────────────────────────────
        map_image = self._composite_rgb_map(self.map.shape[:2] ,self.map[:, :, 1], self.map[:, :, 0], None)
        ax = self._ax_map
        ax.cla()
        ax.imshow(
            map_image, cmap="YlOrRd", origin="upper",
            vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        ax.set_title("Map", color="white", fontsize=9, pad=4)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        if len(self._obs_hsitory) > 0:
            last_obs = self._obs_hsitory[len(self._obs_hsitory) - 1]
            obs_image = self._composite_rgb_map(last_obs.shape[:2] , last_obs[:, :, 1], last_obs[:, :, 0], None)
            ax = self._ax_obs
            ax.cla()
            ax.imshow(
                obs_image, cmap="YlOrRd", origin="upper",
                vmin=0.0, vmax=1.0, interpolation="nearest"
            )
            ax.set_title("Observation", color="white", fontsize=9, pad=4)
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(colors="gray", labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        # Draw each agent's position and viewport footprint on the map
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_agents, 1)))
        for i, aid in enumerate(agents):
            pos = self._agent_positions[i]
            if pos is None:
                continue
            x, y = int(pos[1]), int(pos[0])
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
        max_reward = max((v for v in self._reward_history), default=1) or 1

        for i, aid in enumerate(agents):
            if i >= len(self._ax_viewports):
                break
            ax_vp, ax_bar = self._ax_viewports[i]

            # Viewport image
            ax_vp.cla()
            vp  = self._view_history[len(self._view_history) - 1][i]
            if vp is not None:
                vp_image = self._composite_rgb_map(vp.shape[:2], vp[:, :, 1], vp[:, :, 0], None)
                ax_vp.imshow(
                    vp_image, cmap="YlOrRd", origin="upper",
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
            reward = self._reward_history[len(self._reward_history) - 1]
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



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

env = MultiAgentEnv(n_agents=3, world_size=(512, 512), render_mode="human")
model = PPO("CnnPolicy", env, verbose=1, learning_rate=1e-3, batch_size=128, n_steps=128)
model.learn(total_timesteps=10000)

# for episode in range(10):
#     obs, _ = env.reset()
#     episode_reward = 0.0
#     is_exit = False
#     terminated = False

#     while not terminated or not is_exit:  # PettingZoo: loop until no agents remain
#         actions = env.action_space.sample()
#         obs, rewards, terminated, truncated, infos = env.step(actions)
#         episode_reward += rewards
#         frame = env.render()
    
#     if is_exit:
#         break

#     print(f"Episode {episode} | rewards: {episode_reward}")






        
        




        
