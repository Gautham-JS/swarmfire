import gymnasium as gym
import numpy as np
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import cv2
import os

from utils import Generators, Viewpoint
from agents import Drone


class MultiAgentEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}

    def __init__(
            self, 
            n_agents, 
            world_size, 
            start_positions:list=None, 
            iter_limit=1500, 
            seed = None, 
            env_id="MultiAgentEnv", 
            render_mode="human", 
            sample_interval=100, 
            save_interval=500, 
            is_vid_out=False, 
            vid_base_path = "./vids/", 
            vid_id="test_"
        ):
        super().__init__()

        self.n_agents = n_agents
        self.world_size = world_size
        self.iter_limit = iter_limit
        self.seed = seed
        self.render_mode = render_mode
        self.start_poss = start_positions
        self.env_id = env_id
        self.sample_int = sample_interval
        self.save_int = save_interval

        self._episode_count = 0

        # reward fn weights
        self._r_fire_wt = 5.0
        self._r_fuel_wt = 2.0
        self._r_oob_penality = -1.5
        self._r_revisit_penality = -0.5


        self.vp_size = 64
        self.step_size = 1

        # sort of flattening the multi agent space into a 1D single agent space.
        # TODO : Encoder can make it input shape agnostic
        self.actions_per_agent = 2
        self.n_actions = self.n_agents * self.actions_per_agent
        # self.action_space = gym.spaces.Box(
        #     low=0,
        #     high=2,
        #     shape=(self.n_agents * 2,),
        #     dtype=np.uint8
        # )

        self.action_space = gym.spaces.MultiDiscrete([3] * (self.n_agents * 2))


        # once again, flattening the multi agent observations into a single world level observation.
        # TODO: Encoder should once again make it shape agnostic

        # self.observation_space = gym.spaces.Dict({
        #     "viewport": gym.spaces.Box(
        #         low=0.0, high=1.0,
        #         shape=(2, self.world_size[0], self.world_size[1]),  # CHW
        #         dtype=np.float32
        #     ),
        #     "positions": gym.spaces.Box(
        #         low=0,
        #         high=max(self.world_size),
        #         shape=(self.n_agents * 2,),
        #         dtype=np.int32
        #     )
        # })

        self.observation_space = gym.spaces.Dict({
            "viewport": gym.spaces.Box(low=0.0, high=1.0, shape=(2, 84, 84), dtype=np.float32),
            "positions": gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_agents * 2,), dtype=np.float32)
        })

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
        
        self.is_vid_out = is_vid_out
        self.out = None
        if is_vid_out:
            self.vid_base_path = vid_base_path
            self.vid_id = vid_id
            self.fps = 30


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
        self._episode_count += 1
        self._visited_frac = 0.0
        if self.seed is not None:
            if self._episode_count < 1000:
                self.map = self.world_gen.create_map(0.001, 0.003, seed=self.seed)
            elif self._episode_count < 2000:
                if self._episode_count % 100 == 0:
                    self.seed+=1
                self.map = self.world_gen.create_map(0.001, 0.003, seed = self.seed)
            else:
                self.map = self.world_gen.create_map(0.001, 0.003, seed = None)
        else:
            self.map = self.world_gen.create_map(0.001, 0.003, seed=self.seed)
        self.visited_map = np.zeros(self.world_size, dtype=np.bool)
        self.view_acc.reset()
        
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_instances = [Drone.Drone(f"agent_{i}") for i in range(self.n_agents)]

        if self.start_poss is None:
            self._agent_positions = [(np.random.randint(0, self.world_size[0]), np.random.randint(0, self.world_size[1])) for _ in range(self.n_agents)]
        else:
            self._agent_positions = [(128, 128), (256, 128), (128, 256), (256, 256)][:self.n_agents]

        for p, a in zip(self._agent_positions, self.agent_instances):
            a.set_position({'x':p[0], 'y':p[1], 'z':0})

        obs = {}
        obs["viewport"] = np.zeros((2, 84, 84), dtype=np.float32)
        
        pos = []
        for p in self._agent_positions:
            pos.append(p[0])
            pos.append(p[1])
        
        obs["positions"] = np.asarray(pos, dtype=np.float32) / max(self.world_size)

        infos = {}
        print(f"[RESET] : Starting episode {self._episode_count}")
        return obs, infos

    def evaluate_risk_map_2(self, scene_obs, eval_radius=50):
        fuel_map = scene_obs[:, :, 0]
        fire_map = scene_obs[:, :, 1]

        risk_map = np.zeros(fuel_map.shape, dtype=np.float32)

        fire_coords = np.argwhere(fire_map > 0)  # (N, 2)
        if len(fire_coords) == 0:
            return risk_map

        fuel_coords = np.argwhere(fuel_map > 0)  # (M, 2)
        if len(fuel_coords) == 0:
            return risk_map

        # build KD-tree on fire coords, query all fuel coords at once
        tree = cKDTree(fire_coords)
        nearest_dists, nearest_idxs = tree.query(fuel_coords, k=1, workers=-1)

        # mask out fuel pixels beyond eval_radius
        in_radius = nearest_dists <= eval_radius
        fuel_coords  = fuel_coords[in_radius]
        nearest_dists = nearest_dists[in_radius]
        nearest_idxs  = nearest_idxs[in_radius]

        if len(fuel_coords) == 0:
            return risk_map

        fuel_vals = fuel_map[fuel_coords[:, 0], fuel_coords[:, 1]]
        fire_vals = fire_map[fire_coords[nearest_idxs, 0], fire_coords[nearest_idxs, 1]]

        risks = np.clip((fuel_vals + fire_vals) / (nearest_dists*2 + 1), 0.0, 1.0)
        risk_map[fuel_coords[:, 0], fuel_coords[:, 1]] = risks

        return risk_map

    def calculate_reward(self, scene_obs, risk_map, prev_visited_frac):
        """
        Reward strategy:
        - Always reward exploration (new area coverage).
        - If no fire detected yet: pure exploration reward.
        - If fire detected:
            - Reward coverage of fire pixels (finding more fire).
            - Reward coverage of high-risk fuel pixels.
            - Small continued exploration bonus to keep agents moving.
        - Penalty for boundary violations (handled in step() via penalty arg).

        Returns (total_reward, new_visited_frac)
        """
        fuel_map  = scene_obs[:, :, 0]
        fire_map  = scene_obs[:, :, 1]

        total_pixels   = scene_obs.shape[0] * scene_obs.shape[1]
        visited_mask   = (fuel_map > 0) | (fire_map > 0)   # any nonzero = visited
        visited_frac   = np.sum(visited_mask) / total_pixels

        # --- exploration reward: delta coverage since last step ---
        exploration_reward = (visited_frac - prev_visited_frac) * 10

        fire_detected = np.any(fire_map > 0)

        if not fire_detected:
            # pure exploration phase
            total_reward = exploration_reward
            return total_reward, visited_frac

        # --- fire coverage reward: fraction of fire pixels in view ---
        fire_pixels_visible = np.sum(fire_map > 0)
        fire_coverage_reward = (fire_pixels_visible / (self.vp_size * self.vp_size)) * 10.0

        # --- risk reward: mean risk score over visible high-risk fuel ---
        high_risk_mask  = risk_map > 0.5
        if np.any(high_risk_mask):
            risk_reward = np.mean(risk_map[high_risk_mask]) * 1.50
        else:
            risk_reward = 0.0

        # exploration bonus persists so agents keep searching for more fire
        total_reward = exploration_reward + fire_coverage_reward + risk_reward
        return total_reward, visited_frac
    
    def calculate_reward_2(self, scene_obs, risk_map, prev_visited_frac, delta_views):
        # delta_views already captures new-pixel info — use that exclusively for exploration
        new_fuel_pixels = sum(np.sum(d[:, :, 0] > 0) for d in delta_views)
        new_fire_pixels = sum(np.sum(d[:, :, 1] > 0) for d in delta_views)

        exploration_reward = (new_fuel_pixels / (self.vp_size * self.vp_size * self.n_agents)) * 2.0
        fire_discovery_reward = (new_fire_pixels / (self.vp_size * self.vp_size * self.n_agents)) * 5.0 * 10.0

        # fuel-risk reward: only on NEWLY seen fuel this step
        fuel_risk_reward = 0.0
        for delta, pos in zip(delta_views, self._agent_positions):
            px, py = pos
            half = self.vp_size // 2
            x0, x1 = np.clip(px - half, 0, self.world_size[0]), np.clip(px + half, 0, self.world_size[0])
            y0, y1 = np.clip(py - half, 0, self.world_size[1]), np.clip(py + half, 0, self.world_size[1])
            risk_patch = risk_map[x0:x1, y0:y1]
            h, w = risk_patch.shape
            delta_fuel = delta[:h, :w, 0]
            fuel_risk_reward += float(np.sum(delta_fuel * risk_patch)) / (self.vp_size * self.vp_size) * self._r_fire_wt

        return exploration_reward + fire_discovery_reward + fuel_risk_reward, prev_visited_frac
    
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
    
    def calculate_per_agent_reward(self, delta_views, risk_map):
        rewards = []
        for delta, pos in zip(delta_views, self._agent_positions):
            px, py = pos
            new_fuel = np.sum(delta[:, :, 0] > 0) / (self.vp_size ** 2)
            new_fire = np.sum(delta[:, :, 1] > 0) / (self.vp_size ** 2)
            half = self.vp_size // 2
            x0, x1 = np.clip(px - half, 0, self.world_size[0]), np.clip(px + half, 0, self.world_size[0])
            y0, y1 = np.clip(py - half, 0, self.world_size[1]), np.clip(py + half, 0, self.world_size[1])
            rp = risk_map[x0:x1, y0:y1]
            h, w = rp.shape
            risk_score = float(np.sum(delta[:h, :w, 0] * rp)) / (self.vp_size ** 2)
            rewards.append(2.0 * new_fuel + 5.0 * new_fire + 4.0 * risk_score)
        return rewards 
    
    def calculate_near_boundary_penality(self, x, y):
        near_bound_penality = 0
        if x + self.vp_size >= self.world_size[0]:
            near_bound_penality += abs(self.world_size[0] - (x + self.vp_size))
        if x - self.vp_size <= 0:
            near_bound_penality += abs(x - self.vp_size)
        if y + self.vp_size >= self.world_size[1]:
            near_bound_penality += abs(self.world_size[1] - (y + self.vp_size))
        if y - self.vp_size <= 0:
            near_bound_penality += abs(y - self.vp_size)

        return 0.05 * near_bound_penality

    def step(self, action):
        reward, terminated, truncated, infos, obs = 0.0, False, False, {}, {}
        loc_pos_history = []
        loc_view_history = []
        penality = 0
        poss = []
        deltas = []
        per_agent_views = []

        nb_penality = 0

        for i in range(self.n_agents):
            agent:Drone.Drone = self.agent_instances[i]
            agent_actions = action[self.actions_per_agent*i : self.actions_per_agent*(i+1)]   # for i-th agent, the corresponding action would be [2*i:2*1+1]
            agent_pos:tuple = self._agent_positions[i]
            
            dx, dy = self.get_position_delta_from_action(agent_actions[0]), self.get_position_delta_from_action(agent_actions[1])
            agent.inject_velocity({"x" : dx, "y":dy, "z": 0})

            px, py = int(agent.get_position_array()[0]), int(agent.get_position_array()[1])

            self._agent_positions[i] = (px, py)
            poss.append(px)
            poss.append(py)

            nb_penality += self.calculate_near_boundary_penality(px, py)

            # if px >= self.world_size[0] or px < 0 or py >= self.world_size[1] or py <= 0:
            #     #print(f"Agent {self.agents[i]} hit bounds")
            #     self._agent_positions[i] = (px, py)
            #     loc_pos_history.append((px - dx, py - dx))
            #     loc_view_history.append(np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32))
            #     penality+=1

            #     continue
            if px >= self.world_size[0] or px < 0 or py >= self.world_size[1] or py < 0:
                px = np.clip(px, 0, self.world_size[0] - 1)
                py = np.clip(py, 0, self.world_size[1] - 1)
                self._agent_positions[i] = (px, py)   # clamp, don't go OOB
                penality += 1

                loc_pos_history.append((px - dx, py - dx))
                loc_view_history.append(np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32))
                print("SHI OOB BRUH")
                terminated=True
                continue
            

            view, delta_view = self.extract_viewpoint(px, py)

            view_fuel = Viewpoint.get_square_viewpoint(self.map[:, :, 0], (px, py), self.vp_size)
            view_fire = Viewpoint.get_square_viewpoint(self.map[:, :, 1], (px, py), self.vp_size)
            view_agent = np.stack([view_fuel, view_fire], axis=0)  # (2, vp_size, vp_size)
            small = np.stack([
                cv2.resize(view_agent[c], (84, 84), interpolation=cv2.INTER_AREA)
                for c in range(2)
            ])
            per_agent_views.append(small)

            deltas.append(delta_view)
            self.view_acc.accumulate(view, (px, py), self.vp_size)
            
            # view = np.transpose(view, (2, 0, 1))      # → (2, H, W)
            # view = (view * 255).clip(0, 255).astype(np.uint8)

            loc_pos_history.append((px, py))
            loc_view_history.append(delta_view)

        if penality == self.n_agents:
            terminated=True

        scene_obs = self.view_acc.get_scene()

        penality_scale = 3

        prev_visited_frac = getattr(self, "_visited_frac", 0.0)
        risk_map = self.evaluate_risk_map_2(scene_obs, eval_radius=100)
        total_reward, new_visited_frac = self.calculate_reward_2(scene_obs, risk_map, prev_visited_frac, deltas)
        self._visited_frac = new_visited_frac

        total_reward -= penality_scale * penality
        total_reward -= nb_penality

        self._reward_history.append(total_reward)
        show_risk_map = False
        if show_risk_map and risk_map is not None:
            new_composite = np.zeros((self.world_size[0], self.world_size[1], 2), dtype=np.float32)
            new_composite[:, :, 0] = risk_map
            new_composite[:, :, 1] = scene_obs[:, :, 1]
            self._obs_hsitory.append(new_composite)
        else:
            self._obs_hsitory.append(scene_obs)
        self._view_history.append(loc_view_history)
        self._pos_history.append(loc_pos_history)

        self._step_count+=1

        if self.render_mode == "human" and (self._episode_count % self.sample_int) == 0:
            self.render()
        
        obs["viewport"] = np.mean(per_agent_views, axis=0).astype(np.float32)
        obs["positions"] = np.asarray(poss, dtype=np.float32) / max(self.world_size)

        if self._step_count > self.iter_limit:
            terminated = True

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
        # Initialize video writer now that the figure exists and has a size
        if self.is_vid_out and self.out is None and (self._episode_count % self.save_int == 0):
            self._fig.canvas.draw()  # force a draw so buffer_rgba() is valid
            buf = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            h, w = buf.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'H264')

            os.makedirs(self.vid_base_path, exist_ok=True)
            self.out = cv2.VideoWriter(
                f"{self.vid_base_path}/{self.vid_id}_{self._episode_count}.mp4",  # MJPG works best with .avi
                fourcc, self.fps, (w, h), isColor=True
            )
        plt.ion()
        plt.show()
    
    def get_render_as_img(self):
        b = self._fig.axes[0].get_window_extent()
        img = np.array(self._fig.canvas.buffer_rgba())
        img = img[int(b.y0):int(b.y1),int(b.x0):int(b.x1),:]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img

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

        # drawing agents historic positions:
        pos_array = np.asarray(self._pos_history, dtype=np.float32)
        for i, aid in enumerate(agents):
            pos_hist = pos_array[:, i]
            #print(f"Agent : {aid}, pos : {pos_hist}")
            ax.plot(pos_hist[:, 1], pos_hist[:, 0], color=colors[i])

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
            f"Episode {self._episode_count} | Step {self._step_count}",
            color="white", fontsize=11, fontweight="bold", y=0.97
        )

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)

            if self.is_vid_out and self.out is not None:
                frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                self.out.write(frame)
            return None

        # rgb_array mode — render to numpy array
        self._fig.canvas.draw()
        buf = self._fig.canvas.buffer_rgba()
        img = np.asarray(buf)[..., :3]   # drop alpha → H×W×3 uint8

        

        return img
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

        if self.out is not None:
            self.out.release()
            self.out = None




from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

env = DummyVecEnv(
    [lambda: MultiAgentEnv(
            n_agents=1,
            world_size=(512, 512),
            start_positions=[(256, 256), (256, 128), (128, 256), (256, 256)],  # fixed grid
            render_mode="human",
            sample_interval=5,
            save_interval=100,
            seed=None,
            is_vid_out=True,
            vid_id="no_swarming_global_reward",
            vid_base_path="/home/gjs/software/thesis/swarmfire/vids/"
        )
    ]
)
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

model = PPO(
    "MultiInputPolicy", env,
    verbose=1,
    learning_rate=1e-4,          # lower — reward scale is high
    batch_size=256,
    n_steps=2048,                # ~20 full episodes per rollout
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.05,               # higher entropy to fight premature convergence
    vf_coef=0.5,
    max_grad_norm=0.5,
    clip_range=0.2,
)

model.learn(total_timesteps=500_000)

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






        
        




        
