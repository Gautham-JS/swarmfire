import gymnasium as gym
import numpy as np
from scipy.spatial import cKDTree
import torch
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import cv2
import os

from utils import Generators, Viewpoint, GenericUtils
from agents import Drone

import wandb

class IsolatedAgentEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}

    def __init__(
            self, 
            n_agents, 
            world_size,
            is_recency_learned = True, 
            start_positions:list=None, 
            iter_limit=4500, 
            vp_size=64,
            seed = None, 
            fixed_seed=False,
            env_id="MultiAgentEnv", 
            render_mode="human", 
            sample_interval=100, 
            save_interval=500, 
            is_vid_out=False, 
            vid_base_path = "./vids/", 
            vid_id="test_",
            phase_weights:dict = None,
            device=None
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
        self.fixed_seed = fixed_seed
        self.is_recency_learned = is_recency_learned

        if device is not None:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self._episode_count = 0

        self._recency_decay = 0.9995   # per-step decay factor; tune this
        self._recency_visit_bump = 0.03   # value stamped on visit

        # reward fn weights
        self.set_reward_weights(phase_weights)


        self.vp_size = vp_size
        self.step_size = 1
        self.map_update_interval = 20
        self.stepped_map_update = False
        self.reduction_factor = 2

        self.actions_per_agent = 2
        self.n_actions = self.n_agents * self.actions_per_agent

        self.action_space = gym.spaces.MultiDiscrete([3] * (self.n_agents * 2))

        self.n_obs_channels = 3 if self.is_recency_learned else 2
        self.observation_space = gym.spaces.Dict({
            "viewport": gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_obs_channels, 84, 84), dtype=np.float32),
            "positions": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents * 4,), dtype=np.float32)
        })

        self.world_gen = Generators.FuelMapGenerator(self.world_size)
        self.view_acc = Viewpoint.IncrementalViewAccumulator(self.world_size, self.n_obs_channels)


        # vars initialized by reset fn
        self.map = None             # global map HxWx1 [TODO: Remove]
        self.visited_map = None
        self.agents = None          # indexed list of N agents
        self._obs_hsitory = None    # history of all rewards n_agents x n_iters
        self._reward_history = None 
        self._view_history = None
        self._pos_history = None
        self._agent_positions = None
        self._step_count = 0

        self.recency_map = None
        self.fire_disc_map = None

        # render states
        self._fig = None
        self._axes = None

        self.init_domain_logs()

        self.env_state_dict = {
            "map_size"                  : [self.world_size[0], self.world_size[1]],
            "vp_size"                   : self.vp_size,
            "recency_decay_factor"      : self._recency_decay,
            "recency_visit_bump"        : self._recency_visit_bump,
            "start_pos"                 : self.start_poss
        }
        
        self.is_vid_out = is_vid_out
        self.out = None
        if is_vid_out:
            self.vid_base_path = vid_base_path
            self.vid_id = vid_id
            self.fps = 30
    
    def init_domain_logs(self):
        # domain specific logging
        self.fire_coverage_timesteps_dict = {
            "overall": [], "25": self.iter_limit, "50": self.iter_limit, "75": self.iter_limit, "90": self.iter_limit, "99": self.iter_limit
        }
        self.fuel_coverage_timesteps_dict = {
            "overall": [], "25": self.iter_limit, "50": self.iter_limit, "75":self.iter_limit, "90": self.iter_limit, "99":self.iter_limit
        }
        self.fire_coverage_percentages = []
        self.fuel_coverage_percentages = []
    
    def _init_revisit_tracker(self):
        """
        Per-fire-cell tracking for revisit deltas.
        State per cell: -1 = unseen, 0 = currently visited, N > 0 = last step seen (absent)
        """
        self._revisit_last_seen  = np.full(self.world_size, -1, dtype=np.int32)   # step when cell left viewport
        self._revisit_deltas     = []                                               # all gap lengths this episode
        self._revisit_state      = np.zeros(self.world_size, dtype=np.uint8)       # 0=unseen, 1=visited, 2=absent
        self._revisit_gap_threshold = max(1, self.vp_size*2)       # must leave for at least this many steps
    
    def set_reward_weights(self, weights: dict):
        if weights is None:
            weights = {}
        self._w_exploration    = weights.get("exploration",    1.0)
        self._w_exploration_track    = weights.get("exploration_tracking",    1.0)
        self._w_fire_discovery = weights.get("fire_discovery", 1.0)
        self._w_fire_tracking  = weights.get("fire_tracking",  1.0)
        self._w_risk           = weights.get("risk",           1.0)

        # self.env_state_dict["_w_exploration"] = self._w_exploration
        # self.env_state_dict["_w_exploration_track"] = self._w_exploration_track
        # self.env_state_dict["_w_fire_discovery"] = self._w_fire_discovery
        # self.env_state_dict["_w_risk"] = self._w_risk

        print(f"[REWARD_WEIGHTS_UPDATE] -> exploration : {self._w_exploration} | fire discovery : {self._w_fire_discovery} | fire tracking : {self._w_fire_tracking} | risk : {self._w_risk}")

    def get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size
        
    def extract_viewpoint(self, x, y):
        # [TODO: dont use the map directly, grab from UE5 transport layer]
        fuel_view, recently_visited_fuel, delta_fuel_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self.map[:, :, 0], self.visited_map, (x, y), size=self.vp_size)
        fire_view, recently_visited_fire, delta_fire_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self.map[:, :, 1], self.visited_map, (x, y), size=self.vp_size)
        self.visited_map = recently_visited_fuel
        view, deltas = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32), np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        view[:, :, 0], view[:, :, 1] = fuel_view, fire_view
        deltas[:, :, 0], deltas[:, :, 1] = delta_fuel_mask, delta_fire_mask
        return view, deltas
    
    def check_and_update_seed(self):
        # check if the seed updates in stages or in one go.
        if not self.fixed_seed and self.seed is not None and self._episode_count > 10:
            if ( (self._episode_count - 1) % self.map_update_interval == 0):
                self.seed+=1
                if self.stepped_map_update:
                    self.map_update_interval = self.map_update_interval // 2
                    if self.map_update_interval < 1:
                        self.map_update_interval = 1

    def init_agent_positions(self):
        if self.start_poss is not None:
            self._agent_positions = self.start_poss[:self.n_agents]
        else:
            self._agent_positions = [
                (np.random.randint(0, self.world_size[0]),
                np.random.randint(0, self.world_size[1]))
                for _ in range(self.n_agents)
            ]        
        for p, a in zip(self._agent_positions, self.agent_instances):
            a.set_position({'x': p[0], 'y': p[1], 'z': 0})

        # Seed pos_history so _build_positions_obs has something to diff on step 1
        self._pos_history = [list(self._agent_positions)]

    def reset(self, seed=None, options=None):
        if self._episode_count!=0 and self._episode_count % self.sample_int == 0:
            if len(self._reward_history) > 10:
                print(f"[RESET] Reward history - MAX : {np.max(self._reward_history[10:])} | MEAN : {np.mean(self._reward_history[10:])} | MIN : {np.min(self._reward_history[10:])}")


        # Clear episode state without destroying the video writer or figure
        self._obs_hsitory    = []
        self._agent_positions = []
        self._reward_history  = []
        self._view_history    = []
        self._pos_history     = []
        self._step_count      = 0
        self._episode_count  += 1
        self._visited_frac    = 0.0
        self._last_global_vp  = None

        self.init_domain_logs()
        self._init_revisit_tracker()

        # Reset maps
        self.visited_map   = np.zeros(self.world_size, dtype=np.bool_)
        self.recency_map   = np.zeros(self.world_size, dtype=np.float32)
        self.fire_disc_map = np.full(self.world_size, -1, dtype=np.int32)
        self.view_acc.reset()

        self.check_and_update_seed()
        
        # [TODO: remove]
        self.map = self.world_gen.create_map(0.001, 0.003, seed=self.seed)

        # Spawn agents [future multi agency support]
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_instances = [Drone.Drone(f"agent_{i}") for i in range(self.n_agents)]

        self.init_agent_positions()

        obs = {
            "viewport":  np.zeros((self.n_obs_channels, 84, 84), dtype=np.float32),
            "positions": self._build_positions_obs(),
        }
        
        if self.out is not None:
            self.out.release()
            self.out = None
        
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

        return obs, {}

    def exploration_reward(self, delta, local_view, c_delta=1, c_view=1):
        return (c_delta * (np.sum(delta[:, :, 0]) / self.vp_size**2)) + (c_view * (np.sum(local_view[:, :, 0]) / self.vp_size**2))
    
    def fire_reward(self, delta, local_view, c_fd = 1, c_ft=1):
        return (c_fd * (np.sum(delta[:, :, 1]) / self.vp_size**2)) + (c_ft * (np.sum(local_view[:, :, 1]) / self.vp_size**2))
    
    """
        [REWARD COMPONENT]
    """
    def recency_penality(self, px, py, c_rp=10):
        recency_patch = self.extract_recency_map(px, py)
        return float(np.mean(recency_patch)) * c_rp


    def mark_all_recency(self):
        for pos in self._agent_positions:
            px, py = pos
            self.mark_recency_map(px, py)

    """
        [REWARD COMPONENT]
        Exponential penalty that grows sharply as agent approaches the wall.
        Increased scale so it decisively dominates any wall-camping reward.
    """
    def calculate_near_boundary_penalty(self, x, y):
        margin = self.vp_size // 2
        penalty = 0.0
        for coord, limit in [(x, self.world_size[0]), (y, self.world_size[1])]:
            dist_low  = coord
            dist_high = limit - coord
            if dist_low < margin:
                penalty += np.exp((margin - dist_low) / (margin / 4)) - 1
            if dist_high < margin:
                penalty += np.exp((margin - dist_high) / (margin / 4)) - 1
        # Removed the 0.5 discount — boundary must hurt more than staying pays
        return penalty

    """
        Returns the fraction of the viewport that lies within world bounds [0, 1].
        Used to scale down rewards when the agent is near a wall and half its
        viewport is zero-padded — prevents the agent from getting free reward
        for looking at nothing.
    """
    def _viewport_coverage_fraction(self, px, py) -> float:
        half = self.vp_size // 2
        x0 = max(px - half, 0);  x1 = min(px + half, self.world_size[0])
        y0 = max(py - half, 0);  y1 = min(py + half, self.world_size[1])
        valid_pixels = (x1 - x0) * (y1 - y0)
        total_pixels = self.vp_size * self.vp_size
        return valid_pixels / total_pixels


    """
        [REWARD COMPONENT]
        These functions added for promoting fire crossing if significant area is still unexplored behind the fire:
    """
    def calculate_fire_crossing_penalty(self, px, py) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene      = self.view_acc.get_scene()
        fire_value = float(scene[cx, cy, 1])

        if fire_value <= 0.0:
            return 0.0

        base_penalty = 80.0
        return base_penalty * (0.5 + 0.5 * fire_value)

    """
        [REWARD COMPONENT]
        Rewards moving tangentially along the fire boundary as seen in the
        accumulated observation, not the ground truth map.
    """
    def fire_perimeter_alignment_reward(self, delta_views) -> float:
        from scipy.ndimage import sobel

        scene = self.view_acc.get_scene()
        total = 0.0

        for i, (delta, pos) in enumerate(zip(delta_views, self._agent_positions)):
            px, py = pos
            half = self.vp_size // 2
            x0 = int(np.clip(px - half, 0, self.world_size[0]))
            x1 = int(np.clip(px + half, 0, self.world_size[0]))
            y0 = int(np.clip(py - half, 0, self.world_size[1]))
            y1 = int(np.clip(py + half, 0, self.world_size[1]))

            fire_patch = scene[x0:x1, y0:y1, 1]
            if not np.any(fire_patch > 0):
                continue

            gx  = sobel(fire_patch.astype(np.float32), axis=1)
            gy  = sobel(fire_patch.astype(np.float32), axis=0)
            mag = np.sqrt(gx ** 2 + gy ** 2)

            total_mag = mag.sum()
            if total_mag < 1e-6:
                continue

            boundary_normal = np.array([
                np.sum(gy * mag) / total_mag,
                np.sum(gx * mag) / total_mag,
            ], dtype=np.float64)
            bn_norm = np.linalg.norm(boundary_normal)
            if bn_norm < 1e-8:
                continue
            boundary_normal  /= bn_norm
            boundary_tangent  = np.array([-boundary_normal[1], boundary_normal[0]])

            if len(self._pos_history) < 2:
                continue
            prev     = self._pos_history[-2][i]
            curr     = self._pos_history[-1][i]
            vel      = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 1e-6:
                continue
            vel_dir   = vel / vel_norm
            alignment = float(np.dot(vel_dir, boundary_tangent))
            total    += max(0.0, alignment) * 12.0
        return total

    """
        [REWARD COMPONENT]
        Gaussian bonus for being near but not inside known fire,
        derived from the accumulated observation.
    """
    def fire_proximity_bonus(self, px, py, ideal_dist_cells: float = 2.0) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene = self.view_acc.get_scene()
        # No bonus if standing in known fire
        if scene[cx, cy, 1] > 0:
            return 0.0

        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))

        fire_patch  = scene[x0:x1, y0:y1, 1]
        fire_coords = np.argwhere(fire_patch > 0)
        if len(fire_coords) == 0:
            return 0.0

        local_cx = cx - x0
        local_cy = cy - y0
        diffs    = fire_coords - np.array([local_cx, local_cy])
        dist     = float(np.min(np.linalg.norm(diffs, axis=1)))

        sigma = ideal_dist_cells * 0.8
        return float(3.0 * np.exp(-0.5 * ((dist - ideal_dist_cells) / sigma) ** 2))


    """
        [REWARD COMPONENT]
        Estimates unseen area occluded by *known* fire in the accumulated scene.

        Rays are cast outward. Once a ray crosses a known-fire cell, everything
        beyond that is still unvisited counts as occluded potential.
    
        The key difference from the ground-truth version: we treat cells that
        are both unvisited AND beyond known fire as occluded. Cells that are
        simply unvisited but not beyond fire are just unexplored — the agent
        should explore those via normal exploration reward, not fire-crossing.

        [TODO: Modularize params as env params]
    """
    def _estimate_occluded_area(self, px, py) -> float:
        scene      = self.view_acc.get_scene()
        n_rays     = 8
        max_range  = int(max(self.world_size) * 0.75)
        step_size  = 2
        occluded   = 0
        total_rays = 0

        for angle_idx in range(n_rays):
            angle = 2 * np.pi * angle_idx / n_rays
            dx    = np.cos(angle)
            dy    = np.sin(angle)

            in_fire      = False
            ray_occluded = 0

            for dist in range(step_size, max_range, step_size):
                rx = int(round(px + dx * dist))
                ry = int(round(py + dy * dist))

                if rx < 0 or rx >= self.world_size[0] or ry < 0 or ry >= self.world_size[1]:
                    break

                # Fire determined from accumulated observation only
                cell_is_known_fire = scene[rx, ry, 1] > 0
                cell_is_unseen     = not self.visited_map[rx, ry]

                if cell_is_known_fire:
                    in_fire = True

                # Only count as occluded if behind known fire AND unvisited
                if in_fire and cell_is_unseen:
                    ray_occluded += 1

            occluded   += ray_occluded
            total_rays += max_range // step_size

        if total_rays == 0:
            return 0.0

        return float(occluded) / float(total_rays)

    """
        [REWARD COMPONENT]
        Counteracts the crossing penalty when crossing known fire gives access
        to a large unseen area beyond it, and the local fire patch is thin.
        
        Both fire presence and fire density measured from accumulated scene.
    """
    def _fire_crossing_opportunity_reward(self, px, py) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene = self.view_acc.get_scene()

        # Only activates when agent is inside known fire
        if scene[cx, cy, 1] <= 0:
            return 0.0

        occluded_fraction = self._estimate_occluded_area(px, py)

        thin_radius = self.vp_size // 4
        x0 = int(np.clip(px - thin_radius, 0, self.world_size[0]))
        x1 = int(np.clip(px + thin_radius, 0, self.world_size[0]))
        y0 = int(np.clip(py - thin_radius, 0, self.world_size[1]))
        y1 = int(np.clip(py + thin_radius, 0, self.world_size[1]))

        # Local fire density from accumulated scene
        local_fire_patch   = scene[x0:x1, y0:y1, 1]
        local_fire_density = float(np.mean(local_fire_patch > 0))
        thinness           = 1.0 - local_fire_density

        opportunity = occluded_fraction * thinness
        return opportunity * 150.0
    """
        [REWARD COMPONENT]
        Returns a positive bonus when the agent moves away from the corner
        it is closest to, measured as Euclidean distance to the nearest corner.
        This fires even when moving_away from the nearest *wall* is ambiguous
        (e.g. diagonal corners where both walls are equidistant).
    """
    def _corner_escape_bonus(self, px, py, prev_px, prev_py) -> float:
        corners = [
            (0, 0),
            (0, self.world_size[1]),
            (self.world_size[0], 0),
            (self.world_size[0], self.world_size[1]),
        ]
        def min_corner_dist(x, y):
            return min(np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners)

        prev_dist = min_corner_dist(prev_px, prev_py)
        curr_dist = min_corner_dist(px, py)
        return max(0.0, curr_dist - prev_dist) * 3.5   # scale, [TODO: Convert to env param]
    

    def calculate_reward(self, delta_views, recency_map):
        reward = 0.0
        step_advantage = 2.0 * (1.0 + (
            1.0 - GenericUtils.normalize_data(self._step_count, 0, self.iter_limit)
        ))

        for i, (delta, pos) in enumerate(zip(delta_views, self._agent_positions)):
            px, py = pos
            half = self.vp_size // 2
            x0 = int(np.clip(px - half, 0, self.world_size[0]))
            x1 = int(np.clip(px + half, 0, self.world_size[0]))
            y0 = int(np.clip(py - half, 0, self.world_size[1]))
            y1 = int(np.clip(py + half, 0, self.world_size[1]))

            local_view = self.view_acc.get_scene()[x0:x1, y0:y1, :]

            vp_scale = self._viewport_coverage_fraction(px, py) ** 2

            new_fuel_pixels  = float(np.sum(delta[:, :, 0] > 0))
            new_fire_pixels  = float(np.sum(delta[:, :, 1] > 0))
            seen_fuel_pixels = float(np.sum(local_view[:, :, 0] > 0))

            exploration      = self._w_exploration * 0.03 * new_fuel_pixels * vp_scale
            fire_disc        = self._w_fire_discovery * 0.09 * new_fire_pixels * step_advantage * vp_scale
            fuel_track       = self._w_exploration_track * 0.001 * seen_fuel_pixels * vp_scale
            perimeter_reward = self._w_fire_tracking * self.fire_perimeter_alignment_reward([delta]) * vp_scale
            proximity        = self.fire_proximity_bonus(px, py) * vp_scale

            # Crossing opportunity reward - net out the penalty when crossing is smart
            crossing_opportunity = self._fire_crossing_opportunity_reward(px, py)
            fire_cross           = self._w_risk * self.calculate_fire_crossing_penalty(px, py)
            # Net fire cost: penalty minus opportunity. Can go positive (net reward)
            # when crossing is strategically valuable, negative (net penalty) otherwise.
            net_fire_cost = fire_cross - crossing_opportunity

            # Movement reward
            if len(self._pos_history) >= 2:
                prev  = self._pos_history[-2][i]
                curr  = self._pos_history[-1][i]
                moved = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) > 0.2

                dist_to_nearest_wall = min(px, self.world_size[0] - px,
                                        py, self.world_size[1] - py)
                in_margin = dist_to_nearest_wall < self.vp_size // 2

                if moved:
                    if in_margin:
                        prev_dist = min(prev[0], self.world_size[0] - prev[0],
                                        prev[1], self.world_size[1] - prev[1])
                        curr_dist = min(curr[0], self.world_size[0] - curr[0],
                                        curr[1], self.world_size[1] - curr[1])
                        moving_away = curr_dist > prev_dist
                        movement = 2.0 if moving_away else -0.3

                        # Corner escape bonus on top of wall escape
                        movement += self._corner_escape_bonus(px, py, prev[0], prev[1])
                    else:
                        movement = 0.5
                else:
                    movement = -1.5
            else:
                movement = 0.0
            recency     = float(np.mean(self.extract_recency_map(px, py)))
            recency_pen = (math.exp(recency) - 1) / (math.exp(1) - 1) * 140
            boundary    = self.calculate_near_boundary_penalty(px, py)

            reward += (
                exploration
                + fire_disc
                + fuel_track
                + perimeter_reward
                + proximity
                + movement
                - recency_pen
                - boundary
                - net_fire_cost     # replaces the old "fire_cross" component
            )

            return reward

    def mark_recency_map(self, px, py):
        half = self.vp_size // 2
        x0 = np.clip(px - half, 0, self.world_size[0])
        x1 = np.clip(px + half, 0, self.world_size[0])
        y0 = np.clip(py - half, 0, self.world_size[1])
        y1 = np.clip(py + half, 0, self.world_size[1])

        self.recency_map[x0:x1, y0:y1] += self._recency_visit_bump
        self.recency_map[x0:x1, y0:y1] = np.minimum(self.recency_map[x0:x1, y0:y1], np.ones((x1 - x0, y1 - y0), dtype=np.float32))

        return self.recency_map[x0:x1, y0:y1]
    
    

    def extract_recency_map(self, px, py):
        half = self.vp_size // 2
        x0 = np.clip(px - half, 0, self.world_size[0])
        x1 = np.clip(px + half, 0, self.world_size[0])
        y0 = np.clip(py - half, 0, self.world_size[1])
        y1 = np.clip(py + half, 0, self.world_size[1])
        return self.recency_map[x0:x1, y0:y1].copy()

    def create_global_crop_viewport_obs(self, sz=84):
        scene = self.view_acc.get_scene()   # (H, W, 2) — fuel + fire only
        cx = int(np.mean([p[0] for p in self._agent_positions]))
        cy = int(np.mean([p[1] for p in self._agent_positions]))

        loc_view = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        loc_fuel = Viewpoint.get_square_viewpoint(scene[:, :, 0], (cx, cy), self.vp_size)
        loc_fire = Viewpoint.get_square_viewpoint(scene[:, :, 1], (cx, cy), self.vp_size)
        loc_view[:, :, 0], loc_view[:, :, 1] = loc_fuel, loc_fire

        scene_crop = loc_view                                            # (H', W', 2)
        scene_crop_resized = cv2.resize(scene_crop, (84, 84), interpolation=cv2.INTER_AREA)
        scene_chw = np.transpose(scene_crop_resized, (2, 0, 1))                         # (2, 84, 84)

        # recency channel from the same spatial crop
        recency_crop = Viewpoint.get_square_viewpoint(self.recency_map, (cx, cy), self.vp_size)                                  # (H', W')
        recency_resized = cv2.resize(recency_crop, (84, 84), interpolation=cv2.INTER_AREA)[None]  # (1, 84, 84)

        return scene_chw.astype(np.float32)  # (2, 84, 84)

    def _build_positions_obs(self) -> np.ndarray:
        obs = []
        scene = self.view_acc.get_scene()
        fire_coords = np.argwhere(scene[:, :, 1] > 0)

        for i, pos in enumerate(self._agent_positions):
            px, py = pos
            # Normalised position
            obs.append(px / self.world_size[0])
            obs.append(py / self.world_size[1])

            # Velocity (from last step)
            if len(self._pos_history) >= 2:
                prev = self._pos_history[-2][i]
                vx = (px - prev[0]) / (self.step_size + 1e-8)
                vy = (py - prev[1]) / (self.step_size + 1e-8)
            else:
                vx, vy = 0.0, 0.0
            obs.append(np.clip(vx, -1, 1))
            obs.append(np.clip(vy, -1, 1))

        return np.asarray(obs, dtype=np.float32)

    def step(self, action):
        reward, terminated, truncated, infos, obs = 0.0, False, False, {}, {}
        loc_pos_history = []
        loc_view_history = []
        penality = 0
        poss = []
        deltas = []
        per_agent_views = []

        nb_penality = 0
        self.recency_map *= self._recency_decay
        is_oob = False


        for i in range(self.n_agents):
            agent:Drone.Drone = self.agent_instances[i]
            agent_actions = action[self.actions_per_agent*i : self.actions_per_agent*(i+1)]   # for i-th agent, the corresponding action would be [2*i:2*1+1]
            
            dx, dy = self.get_position_delta_from_action(agent_actions[0]), self.get_position_delta_from_action(agent_actions[1])
            agent.inject_velocity({"x" : dx, "y":dy, "z": 0})

            px, py = int(agent.get_position_array()[0]), int(agent.get_position_array()[1])

            self._agent_positions[i] = (px, py)
            poss.append(px)
            poss.append(py)

            if px >= self.world_size[0] or px < 0 or py >= self.world_size[1] or py < 0:
                px = np.clip(px, 0, self.world_size[0] - 1)
                py = np.clip(py, 0, self.world_size[1] - 1)
                self._agent_positions[i] = (px, py)
                agent.set_position({'x': px, 'y': py, 'z': 0})  # zero out accumulated velocity
                penality += 55 
                is_oob = True
            

            view, delta_view = self.extract_viewpoint(px, py)
            view_recency = self.extract_recency_map(px, py)
            
            # [TODO: dont use the map directly, grab from UE5 transport layer]
            view_fuel = Viewpoint.get_square_viewpoint(self.map[:, :, 0], (px, py), self.vp_size)
            view_fire = Viewpoint.get_square_viewpoint(self.map[:, :, 1], (px, py), self.vp_size)
            view_agent = np.stack([view_fuel, view_fire], axis=0)  # (2, vp_size, vp_size)
            
            small = np.stack([
                cv2.resize(view_agent[0], (84, 84), interpolation=cv2.INTER_AREA),
                cv2.resize(view_agent[1], (84, 84), interpolation=cv2.INTER_AREA),
                cv2.resize(view_recency, (84, 84), interpolation=cv2.INTER_AREA)
            ])
            
            per_agent_views.append(small)

            deltas.append(delta_view)
            self.view_acc.accumulate(view, (px, py), self.vp_size)

            loc_pos_history.append((px, py))
            loc_view_history.append(delta_view)

        
        for delta, pos in zip(deltas, self._agent_positions):
            px, py = pos
            half = self.vp_size // 2
            x0 = np.clip(px - half, 0, self.world_size[0])
            x1 = np.clip(px + half, 0, self.world_size[0])
            y0 = np.clip(py - half, 0, self.world_size[1])
            y1 = np.clip(py + half, 0, self.world_size[1])

            h = x1 - x0
            w = y1 - y0

            # delta[:, :, 1] is the fire delta — newly seen fire pixels this step
            new_fire_mask = delta[:h, :w, 1] > 0
            # only stamp pixels not yet discovered
            undiscovered = self.fire_disc_map[x0:x1, y0:y1] == -1
            stamp_mask = new_fire_mask & undiscovered
            self.fire_disc_map[x0:x1, y0:y1][stamp_mask] = self._step_count


        scene_obs = self.view_acc.get_scene()

        penality_scale = 1

        prev_visited_frac = getattr(self, "_visited_frac", 0.0)
        total_reward = self.calculate_reward(
            deltas, self.recency_map
        )

        self.mark_all_recency()
        self._update_revisit_tracker()


        total_reward -= penality_scale * penality

        if self._episode_count !=0 and (self._episode_count % self.sample_int == 0) and (self._step_count % 50 == 0):
            print(f"[CALC REWARD] : total reward value = {total_reward} | penality scale : {penality}")

        self._reward_history.append(total_reward)
        self._obs_hsitory.append(scene_obs)
        self._view_history.append(loc_view_history)
        self._pos_history.append(loc_pos_history)

        self._step_count+=1

        self.log_coverage_percentiles()

        if (self._episode_count % self.sample_int) == 0:
            self.render()

        
        infos["domain_metrics"] = self.log_metrics()

        obs["viewport"] = self.create_global_crop_viewport_obs()
        obs["positions"] = self._build_positions_obs()

        #print(f"[STEP] POSITION : {obs['positions']}")

        if self._step_count > self.iter_limit:
            truncated = True
        self._last_global_vp = obs["viewport"].copy()

        return obs, total_reward, terminated, truncated, infos
    





    def _channels_to_rgb(self, chw: np.ndarray) -> np.ndarray:
        """
        Convert a (C, H, W) float32 array to an (H, W, 3) RGB image.
        - C == 1 : greyscale repeated to 3 channels
        - C == 2 : fuel=G, fire=R, B=0
        - C == 3 : fuel=G, fire=R, recency=B
        - C  > 3 : PCA projected to 3 components, then normalised to [0,1]
        """
        c, h, w = chw.shape

        if c == 1:
            grey = chw[0]
            return np.stack([grey, grey, grey], axis=-1)

        if c == 2:
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            canvas[:, :, 1] = chw[0]   # fuel → green
            canvas[:, :, 0] = chw[1]   # fire → red
            return canvas

        if c == 3:
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            canvas[:, :, 1] = chw[0]   # fuel    → green
            canvas[:, :, 0] = chw[1]   # fire    → red
            canvas[:, :, 2] = chw[2]   # recency → blue
            return canvas

        # C > 3 — PCA onto 3 components
        flat = chw.reshape(c, -1).T          # (H*W, C)
        flat -= flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        projected = flat @ Vt[:3].T          # (H*W, 3)
        projected -= projected.min(axis=0)
        denom = projected.max(axis=0)
        denom[denom == 0] = 1
        projected /= denom
        return projected.reshape(h, w, 3).astype(np.float32)

    
    def _init_figure(self):
        """Create the figure layout once and reuse it across frames."""
        n_agents = max(self.n_agents, 1)

        self._fig = plt.figure(
            figsize=(4 + 3 + 3 + 3 * n_agents, 5),   # extra 3 for global crop panel
            facecolor="#1a1a2e"
        )
        gs = GridSpec(
            2, 3 + n_agents,          # map | obs | global_crop | agent cols
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08
        )

        self._ax_map        = self._fig.add_subplot(gs[:, 0])
        self._ax_obs        = self._fig.add_subplot(gs[:, 1])
        self._ax_global_vp  = self._fig.add_subplot(gs[:, 2])   # ← new
        self._ax_viewports  = [
            (self._fig.add_subplot(gs[0, i + 3]),
            self._fig.add_subplot(gs[1, i + 3]))
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
        if self.render_mode == "human":
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
            ax.set_title("Accumulated Observation", color="white", fontsize=9, pad=4)
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
            # ax_bar.barh(
            #     0, reward, color=colors[i], alpha=0.8,
            #     height=0.5, zorder=3
            # )
            # ax_bar.barh(
            #     0, max_reward, color="#333", alpha=0.4,
            #     height=0.5, zorder=2
            # )
            data = self._reward_history
            if len(self._reward_history) >= 1:
                data = data[1:]                  

            ax_bar.plot(range(len(data)), data, color=colors[i])

            #ax_bar.set_xlim(0, max_reward * 1.05)
            #ax_bar.set_ylim(-0.5, 0.5)
            ax_bar.set_yticks([])
            ax_bar.set_xticks([])
            ax_bar.set_facecolor("#0d0d1a")
            ax_bar.tick_params(colors="gray", labelsize=6)
            ax_bar.set_xlabel("reward", color="gray", fontsize=6)
            ax_bar.set_ylabel("step", color="gray", fontsize=6)
            ax_bar.text(
                0, max_reward * 1.02, f"{reward:.1f}",
                va="center", ha="left", color="white", fontsize=7
            )
            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#333")


        if self._last_global_vp is not None:
            rgb = self._channels_to_rgb(self._last_global_vp)   # (84, 84, 3)
            ax = self._ax_global_vp
            ax.cla()
            ax.imshow(rgb, origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
            c = self._last_global_vp.shape[0]
            label = "Current Observation" if c <= 3 else f"Current Observation (PCA {c}ch)"
            ax.set_title(label, color="white", fontsize=9, pad=4)
            ax.set_facecolor("#0d0d1a")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # channel legend in the corner
            legend_lines = {
                1: ["grey=fuel/fire"],
                2: ["G=fuel", "R=fire"],
                3: ["G=fuel", "R=fire", "B=recency"],
            }
            for line_i, txt in enumerate(legend_lines.get(c, ["PCA projected"])):
                ax.text(
                    2, 4 + line_i * 9, txt,
                    color="white", fontsize=5,
                    bbox=dict(facecolor="#00000088", edgecolor="none", pad=1)
                )
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

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

        if self.is_vid_out and self.out is not None:
            frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            self.out.write(frame)

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

    
    # Domain specific metrics for evaluation of performance

    # TODO: Cannot use global self.map to measure this. Need to detach generated map and live feed map.
    def _update_revisit_tracker(self):
        """
        Call once per step, AFTER agent positions are updated and viewports extracted.
        Builds a boolean mask of which fire cells are currently in any viewport,
        then runs the state machine transitions.
        """
        in_viewport = np.zeros(self.world_size, dtype=bool)
        for pos in self._agent_positions:
            px, py = pos
            half = self.vp_size // 2
            x0 = int(np.clip(px - half, 0, self.world_size[0]))
            x1 = int(np.clip(px + half, 0, self.world_size[0]))
            y0 = int(np.clip(py - half, 0, self.world_size[1]))
            y1 = int(np.clip(py + half, 0, self.world_size[1]))
            in_viewport[x0:x1, y0:y1] = True

        # Only care about cells that are actually fire
        is_fire = self.map[:, :, 1] > 0

        # Transition: visited to absent 
        leaving = (self._revisit_state == 1) & ~in_viewport & is_fire
        self._revisit_last_seen[leaving] = self._step_count
        self._revisit_state[leaving] = 2

        # Transition: absent -> visited (= a revisit!) 
        revisiting = (self._revisit_state == 2) & in_viewport & is_fire
        if np.any(revisiting):
            last_seen_steps = self._revisit_last_seen[revisiting]
            gaps = self._step_count - last_seen_steps
            # Only count gaps that exceed the threshold (genuine re-visits, not edge shimmer)
            valid_gaps = gaps[gaps >= self._revisit_gap_threshold]
            self._revisit_deltas.extend(valid_gaps.tolist())
        self._revisit_state[revisiting] = 1

        # Transition: unseen -> visited
        first_visit = (self._revisit_state == 0) & in_viewport & is_fire
        self._revisit_state[first_visit] = 1

    def log_coverage_percentiles(self):
        viewed = self.view_acc.get_scene()
        map = self.map

        percent_fire_seen = Viewpoint.get_coverage_percentage(1, map, viewed)
        percent_fuel_seen = Viewpoint.get_coverage_percentage(0, map, viewed)

        self.fire_coverage_percentages.append((self._step_count, percent_fire_seen))
        self.fuel_coverage_percentages.append((self._step_count, percent_fuel_seen))


        if (percent_fire_seen >= 99 and self.fire_coverage_timesteps_dict["99"] == self.iter_limit):
            self.fire_coverage_timesteps_dict["99"] = self._step_count
        elif (percent_fire_seen >= 90 and self.fire_coverage_timesteps_dict["90"] == self.iter_limit):
            self.fire_coverage_timesteps_dict["90"] = self._step_count
        elif (percent_fire_seen >= 75 and self.fire_coverage_timesteps_dict["75"] == self.iter_limit):
            self.fire_coverage_timesteps_dict["75"] = self._step_count
        elif (percent_fire_seen >= 50 and self.fire_coverage_timesteps_dict["50"] == self.iter_limit):
            self.fire_coverage_timesteps_dict["50"] = self._step_count
        elif (percent_fire_seen >= 25 and self.fire_coverage_timesteps_dict["25"] == self.iter_limit):
            self.fire_coverage_timesteps_dict["25"] = self._step_count

        if (percent_fuel_seen >= 99 and self.fuel_coverage_timesteps_dict["99"] == self.iter_limit):
            self.fuel_coverage_timesteps_dict["99"] = self._step_count
        elif (percent_fuel_seen >= 90 and self.fuel_coverage_timesteps_dict["90"] == self.iter_limit):
            self.fuel_coverage_timesteps_dict["90"] = self._step_count
        elif (percent_fuel_seen >= 75 and self.fuel_coverage_timesteps_dict["75"] == self.iter_limit):
            self.fuel_coverage_timesteps_dict["75"] = self._step_count
        elif (percent_fuel_seen >= 50 and self.fuel_coverage_timesteps_dict["50"] == self.iter_limit):
            self.fuel_coverage_timesteps_dict["50"] = self._step_count
        elif (percent_fuel_seen >= 25 and self.fuel_coverage_timesteps_dict["25"] == self.iter_limit):
            self.fuel_coverage_timesteps_dict["25"] = self._step_count

    def get_revisit_deltas_minmeanmax(self):
        return np.min(self.revisit_deltas), np.mean(self.revisit_deltas), np.max(self.revisit_deltas)

    def get_fire_coverage_timestamps_dict(self):
        return self.fire_coverage_timesteps_dict
    
    def get_fuel_coverage_timestamps_dict(self):
        return self.fuel_coverage_timesteps_dict
    
    def get_revisit_delta_stats(self) -> dict:
        """Returns revisit gap stats for WandB logging. Safe to call even with no revisits yet."""
        if not self._revisit_deltas:
            return {
                "domain/revisit_delta_mean": float(self.iter_limit),
                "domain/revisit_delta_min":  float(self.iter_limit),
                "domain/revisit_delta_max":  float(self.iter_limit),
                "domain/revisit_count":      0,
            }
        arr = np.array(self._revisit_deltas, dtype=np.float32)
        return {
            "domain/revisit_delta_mean": float(np.mean(arr)),
            "domain/revisit_delta_min":  float(np.min(arr)),
            "domain/revisit_delta_max":  float(np.max(arr)),
            "domain/revisit_count":      len(arr),
        }

    def compute_fire_coverage_auc(self) -> float:
        """
        Trapezoidal AUC of the fire coverage curve.
        Points are (timestep, coverage_fraction). Normalised by iter_limit
        so the result is in [0, 1] regardless of episode length.
        Lower = faster coverage.
        """

        # Build (time, coverage) pairs from your existing percentile dict.
        # Use iter_limit as the timestep for any percentile not yet reached.
        points = self.fire_coverage_percentages

        # Trapezoidal integration, then normalise axes to [0,1]
        auc = 0.0
        for i in range(1, len(points)):
            t0, c0 = points[i - 1]
            t1, c1 = points[i]
            dt = (t1 - t0) / self.iter_limit   # normalised time width
            auc += 0.5 * (c0 + c1) * dt

        return float(auc)
    
    def log_metrics(self):
        #min_revisit_d, mean_revisit_d, max_revisit_d = self.get_revisit_deltas_minmeanmax()
        
        metrics = {
            "domain/fire_coverage_25" : self.get_fire_coverage_timestamps_dict()["25"],
            "domain/fire_coverage_50" : self.get_fire_coverage_timestamps_dict()["50"],
            "domain/fire_coverage_75" : self.get_fire_coverage_timestamps_dict()["75"],
            "domain/fire_coverage_90" : self.get_fire_coverage_timestamps_dict()["90"],
            "domain/fire_coverage_99" : self.get_fire_coverage_timestamps_dict()["99"],
            "domain/fire_coverage_AUC" : self.compute_fire_coverage_auc()
        }
        metrics.update(self.get_revisit_delta_stats())
        return metrics
    
        

            

