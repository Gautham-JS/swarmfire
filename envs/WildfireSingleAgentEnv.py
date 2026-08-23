import os
import gymnasium as gym
import numpy as np
from scipy.spatial import cKDTree
import torch
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import cv2
from scipy.ndimage import sobel, label

from utils import Viewpoint, GenericUtils
from utils.MapManager import BaseAgent, InProcessAgent, UE5Agent, AgentState
from config.Config import VideoWriterConfig, EnvConfig

import logging
log = logging.getLogger(__name__)


# ====================================================== SINGLE AGENT ENV REFAC ===================================

class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}
    
    def __init__(
            self, 
            world_size, 
            start_positions:list=None,
            is_gt_visible=True, 
            iter_limit=4500, 
            vp_size=64,
            seed = None, 
            fixed_seed=False,
            is_recency_obs_disabled = False,
            env_id="MultiAgentEnv", 
            render_mode="human", 
            video_conf: VideoWriterConfig = None,
            phase_weights:dict = None,
            is_eval_mode:bool=False,
            is_ue5_mode:bool=False,
            device=None
        ):
        super().__init__()

        self.world_size = world_size
        self.iter_limit = iter_limit
        self.seed = seed
        self.render_mode = render_mode
        self.start_poss = start_positions
        self.env_id = env_id
        self.is_gt_visible = is_gt_visible
        self.fixed_seed = fixed_seed
        self.video_config = video_conf
        self._disable_recency_obs = is_recency_obs_disabled
        self.is_ue5_mode = is_ue5_mode
        self.is_eval_mode = is_eval_mode

        if device is not None:
            self.device=device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._episode_count = 0

        self._ideal_revisit_delta = 200
        self._recency_reward_scale = 440
        self._fire_revisit_count = 0
        self._fire_revisit_deltas = []

        self.fire_threshold = 0.5

        # reward fn weights
        self.set_reward_weights(phase_weights)


        self.vp_size = vp_size
        self.step_size = 1
        self.map_update_interval = 1
        self.stepped_map_update = False
        self.reduction_factor = 2

        self.action_space = gym.spaces.MultiDiscrete([3] * (2))

        self._n_obs_channels = 2 if self._disable_recency_obs else 3

        # (x, y, vx, vy) = n_agents * 4
        self.observation_space = gym.spaces.Dict({
            "viewport": gym.spaces.Box(low=0.0, high=1.0, shape=(self._n_obs_channels, 84, 84), dtype=np.float32),
            "positions": gym.spaces.Box(low=-1.0, high=1.0, shape=(4, ), dtype=np.float32)
        })

        self.view_acc = Viewpoint.IncrementalViewAccumulator(self.world_size, 2)

        # new local variables
        self._positions_history     = []
        self._observation_history   = []
        self._reward_history         = []
        self._fire_coverage_history = []
        self._reward_components = dict()

        # vars initialized by reset fn
        self._step_count = 0


        # render states
        self._fig = None
        self._axes = None    
        self.out = None
        
    def set_reward_weights(self, weights: dict):
        if weights is None:
            weights = {}
        self._w_exploration    = weights.get("exploration",    1.0)
        self._w_exploration_track    = weights.get("exploration_tracking",    1.0)
        self._w_fire_discovery = weights.get("fire_discovery", 1.0)
        self._w_fire_tracking  = weights.get("fire_tracking",  1.0)
        self._w_risk           = weights.get("risk",           1.0)
        print(f"[REWARD_WEIGHTS_UPDATE] -> exploration : {self._w_exploration} | fire discovery : {self._w_fire_discovery} | fire tracking : {self._w_fire_tracking} | risk : {self._w_risk}")

    def rebuild_reward_components_dict(self):
        self._reward_components.clear()
        self._reward_components["p_recency"] = list()
        self._reward_components["p_bounds"] = list()
        self._reward_components["r_fire"] = list()
        self._reward_components["r_fuel"] = list()
        self._reward_components["r_movement"] = list()
        self._reward_components["r_fire_proximity"] = list()
        self._reward_components["r_fire_tangent"] = list()
        self._reward_components["r_fire_revisit"] = list()
    
    def log_reward_components(
            self,
            r_fire:float,
            r_fuel:float,
            r_movement:float,
            r_fire_proximity:float,
            r_fire_tangent:float,
            p_bounds:float,
            p_recency:float,
            r_fire_revisit:float,
        ):
        self._reward_components["p_recency"].append(p_recency)
        self._reward_components["p_bounds"].append(p_bounds)
        self._reward_components["r_fire"].append(r_fire)
        self._reward_components["r_fuel"].append(r_fuel)
        self._reward_components["r_movement"].append(r_movement)
        self._reward_components["r_fire_proximity"].append(r_fire_proximity)
        self._reward_components["r_fire_tangent"].append(r_fire_tangent)
        self._reward_components["r_fire_revisit"].append(r_fire_revisit)

    def reset(self, seed=None, options=None):
        # Clear episode state without destroying the video writer or figure
        self._positions_history.clear()
        self._reward_history.clear()
        self._observation_history.clear()
        self.rebuild_reward_components_dict()
        self._fire_revisit_count = 0
        self._fire_revisit_deltas.clear()
        self._fire_coverage_history.clear()


        self._step_count      = 0
        self._episode_count  += 1

        # Reset maps
        self.view_acc.reset()

        # if not self.fixed_seed and self.seed is not None and self._episode_count > 10:
        #     if ( (self._episode_count - 1) % self.map_update_interval == 0):
        #         self.seed += 1
        #         if self.stepped_map_update:
        #             self.map_update_interval = self.map_update_interval // 2
        #             if self.map_update_interval < 1:
        #                 self.map_update_interval = 1
        if self.seed is not None:
            if self.fixed_seed and ( (self._episode_count - 1) % self.map_update_interval == 0):
                self.seed += 1
            else:
                self.seed += 1
        np.random.seed(self.seed)

        if self.start_poss is None:
            self.start_poss = (
                (np.random.randint(0, self.world_size[0] - 1,), np.random.randint(0, self.world_size[1] - 1))
            )

        # Spawn agents
        if not self.is_ue5_mode:
            self.agent_instance = InProcessAgent("agent_0", self.world_size, start_pos=self.start_poss, seed=self.seed, vp_size=self.vp_size)
        else:
            logging.info("Running in UE5 mode, creating UE5 websocket client")
            self.agent_instance = UE5Agent("agent_0", self.world_size, start_pos=self.start_poss, seed=self.seed, vp_size=self.vp_size)
        state:AgentState = self.agent_instance.step([1, 1], step_id=-1)

        # Seed pos_history so _build_positions_obs has something to diff on step 1
        if self.is_ue5_mode:
            self._positions_history.append((state.pos_x, state.pos_y))
        else:
            self._positions_history.append(self.start_poss)

        obs = {
            "viewport":  self.create_global_crop_viewport_obs(state),
            "positions": self._build_positions_obs(),
        }

        if not self.is_ue5_mode:
            self._init_revisit_tracker()
        
        if self.video_config.is_enabled and self.out is not None:
            self.out.release()
            self.out = None
        
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

        return obs, {}

    def _init_revisit_tracker(self):
        """
        Per-fire-cell tracking for revisit deltas.
        State per cell: -1 = unseen, 0 = currently visited, N > 0 = last step seen (absent)
        """
        self._revisit_last_seen  = np.full(self.world_size, -1, dtype=np.int32)   # step when cell left viewport
        self._revisit_deltas     = []                                               # all gap lengths this episode
        self._revisit_state      = np.zeros(self.world_size, dtype=np.uint8)       # 0=unseen, 1=visited, 2=absent
        self._revisit_gap_threshold = max(1, self.vp_size*2)       # must leave for at least this many steps


    def _init_revisit_tracker(self):
        """
        Region-level (connected-component) revisit tracking.
        Each spatially distinct fire blob gets a persistent id. A "revisit" is
        counted ONCE per blob, when it re-enters the viewport after being absent
        for at least `_revisit_gap_threshold` steps — not once per pixel.
        """
        self._region_id_map        = np.zeros(self.world_size, dtype=np.int32)  # 0 = background
        self._next_region_id       = 1
        self._region_last_left     = {}    # region_id -> step it left the viewport
        self._region_visible       = set() # region_ids currently inside the viewport
        self._revisit_count        = 0
        self._revisit_deltas       = []
        self._revisit_gap_threshold = max(1, self.vp_size * 2)

        scene = self.agent_instance.get_GT_map()
        fire_mask = scene[:, :, 1] > 0
        if not np.any(fire_mask):
            return
        self._id_map = self._relabel_fire_regions(fire_mask)

    def _relabel_fire_regions(self, fire_mask: np.ndarray) -> np.ndarray:
        """
        Connected-component labeling with persistent IDs across steps.
        A blob that overlaps a previously-seen blob inherits its id (matched by
        pixel-majority overlap), so a fire that grows in place isn't treated as
        a brand-new region.
        """
        new_labels, n_new = label(fire_mask, structure=np.ones((3, 3)))  # 8-connectivity
        new_id_map = np.zeros_like(self._region_id_map)

        for i in range(1, n_new + 1):
            blob_mask = new_labels == i
            overlap_ids = self._region_id_map[blob_mask]
            overlap_ids = overlap_ids[overlap_ids > 0]

            if overlap_ids.size > 0:
                vals, counts = np.unique(overlap_ids, return_counts=True)
                matched_id = int(vals[np.argmax(counts)])   # id covering most of this blob
            else:
                matched_id = self._next_region_id
                self._next_region_id += 1

            new_id_map[blob_mask] = matched_id

        self._region_id_map = new_id_map
        return new_id_map

    def _update_revisit_tracker(self, px, py):
        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))

        scene = self.agent_instance.get_GT_map()
        fire_mask = scene[:, :, 1] > 0
        if not np.any(fire_mask):
            return

        region_map = self._id_map

        viewport_region_ids = set(np.unique(region_map[x0:x1, y0:y1]))
        viewport_region_ids.discard(0)

        # Regions that just left the viewport
        for rid in (self._region_visible - viewport_region_ids):
            self._region_last_left[rid] = self._step_count

        # Regions that just entered/re-entered the viewport
        for rid in (viewport_region_ids - self._region_visible):
            last_left = self._region_last_left.get(rid)
            if last_left is not None:
                gap = self._step_count - last_left
                if gap >= self._revisit_gap_threshold:
                    self._revisit_count += 1
                    self._revisit_deltas.append(gap)
            # else: first-ever sighting of this region - not a revisit

        self._region_visible = viewport_region_ids

    def _track_fire_revisit(self, px, py, state: AgentState) -> None:
        """
        Tracks genuine revisits to previously-seen fire cells at the agent's
        current position. A "revisit" is counted only when:
        1. The current cell is known fire (from the accumulated view_acc scene)
        2. The cell was visited before (revisit_ts_map has a valid prior timestamp)
        3. At least `_ideal_revisit_delta` steps have passed since that visit —
            matching the recency window used by fire_revisit_reward.

        Used purely for evaluation/logging (memory ablation), not reward shaping.
        """
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene = self.agent_instance.get_GT_map()
        if scene[cx, cy, 1] <= 0.0:
            return  # not a known fire cell

        ts_patch = state.revisit_ts_map
        if ts_patch is None or ts_patch.size == 0:
            return

        x0, x1, y0, y1 = self._u_get_view_bounds_from_position(px, py, self.vp_size)
        local_x, local_y = cx - x0, cy - y0
        if not (0 <= local_x < ts_patch.shape[0] and 0 <= local_y < ts_patch.shape[1]):
            return

        last_visit_ts = ts_patch[local_x, local_y]
        if last_visit_ts < 0:
            return  # never visited before -- first visit, not a revisit

        delta = self._step_count - int(last_visit_ts)
        if delta >= self._ideal_revisit_delta:
            self._fire_revisit_count += 1
            self._fire_revisit_deltas.append(delta)

    def calculate_near_boundary_penalty(self, x, y):
        """
        Exponential penalty that grows sharply as agent approaches the wall.
        Increased scale so it decisively dominates any wall-camping reward.
        """
        margin = self.vp_size // 2
        penalty = 0.0
        for coord, limit in [(x, self.world_size[0]), (y, self.world_size[1])]:
            dist_low  = coord
            dist_high = limit - coord
            if dist_low < margin:
                penalty += np.exp((margin - dist_low) / (margin / 4)) - 1
            if dist_high < margin:
                penalty += np.exp((margin - dist_high) / (margin / 4)) - 1
        # Removed the 0.5 discount - boundary must hurt more than staying pays
        return penalty


    def _viewport_coverage_fraction(self, px, py) -> float:
        """
        Returns the fraction of the viewport that lies within world bounds [0, 1].
        Used to scale down rewards when the agent is near a wall and half its
        viewport is zero-padded — prevents the agent from getting free reward
        for looking at nothing.
        """
        half = self.vp_size // 2
        x0 = max(px - half, 0);  x1 = min(px + half, self.world_size[0])
        y0 = max(py - half, 0);  y1 = min(py + half, self.world_size[1])
        valid_pixels = (x1 - x0) * (y1 - y0)
        total_pixels = self.vp_size * self.vp_size
        return valid_pixels / total_pixels


    """
    These functions added for promoting fire crossing if significant area is still unexplored behind the fire:
    """

    def calculate_fire_crossing_penalty(self, px, py) -> float:
        """
        Penalty when the agent is at a position it has previously seen as fire
        in its accumulated observation. Uses view_acc instead of ground truth map.
        
        Note: this will only penalize crossings of *known* fire — fire the agent
        has never seen won't trigger this, which is realistic (the drone doesn't
        know it's flying into unknown fire until it sees it).
        """
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene      = self.view_acc.get_scene()
        fire_value = float(scene[cx, cy, 1])

        if fire_value <= 0.0:
            return 0.0

        base_penalty = 80.0
        return base_penalty * (0.5 + 0.5 * fire_value)

    def fire_perimeter_alignment_reward(self, pos:tuple, fire_patch = None) -> float:
        """
        Rewards moving tangentially along the fire boundary as seen in the
        accumulated observation, not the ground truth map. The tangent
        direction is chosen to match the agent's current heading, so
        orbiting a fire either clockwise or counter-clockwise is rewarded
        equally instead of favoring one fixed rotational sense.
        """
        if fire_patch is None:
            scene = self.view_acc.get_scene()
        total = 0.0

        # --- need velocity first: no velocity, nothing to align ---
        if len(self._positions_history) < 2:
            return total
        prev = self._positions_history[-2]
        curr = self._positions_history[-1]
        vel = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            return total
        vel_dir = vel / vel_norm

        px, py = pos
        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))
        if fire_patch is None:
            fire_patch = scene[x0:x1, y0:y1, 1]
        # if not np.any(fire_patch > self.fire_threshold):
        #     log.info(f"[PERIMETER REW NF] No fire, returning zero")
        #     return total
        # else:
        #     log.info(f"[PERIMETER REW] Valid fire : Min : {fire_patch.min()} | Max : {fire_patch.max()} | Mean : {fire_patch.mean()}")

        if self.is_ue5_mode:
            fire_patch = (fire_patch > self.fire_threshold).astype(np.float32)

        gx = sobel(fire_patch.astype(np.float32), axis=1)
        gy = sobel(fire_patch.astype(np.float32), axis=0)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        total_mag = mag.sum()
        if total_mag < 1e-6:
            return total

        boundary_normal = np.array([
            np.sum(gy * mag) / total_mag,
            np.sum(gx * mag) / total_mag,
        ], dtype=np.float64)
        bn_norm = np.linalg.norm(boundary_normal)
        if bn_norm < 1e-8:
            return total
        boundary_normal /= bn_norm

        # Two candidate tangent directions (perimeter can be walked either way)
        tangent_a = np.array([-boundary_normal[1], boundary_normal[0]])
        tangent_b = -tangent_a

        # Pick whichever tangent direction is closer to how the agent is
        # actually moving, rather than always rewarding one fixed rotation.
        align_a = np.dot(vel_dir, tangent_a)
        align_b = np.dot(vel_dir, tangent_b)
        alignment = align_a if align_a >= align_b else align_b

        total += max(0.0, alignment) * 12.0
        return total


    def fire_proximity_bonus(self, px, py, ideal_dist_cells: float = 2.0) -> float:
        """
        Gaussian bonus for being near but not inside known fire,
        derived from the accumulated observation.
        """
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))

        scene = self.view_acc.get_scene()

        # No bonus if standing in known fire
        if scene[cx, cy, 1] > self.fire_threshold:
            return 0.0

        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))

        fire_patch  = scene[x0:x1, y0:y1, 1]
        fire_coords = np.argwhere(fire_patch > self.fire_threshold)
        if len(fire_coords) == 0:
            return 0.0

        local_cx = cx - x0
        local_cy = cy - y0
        diffs    = fire_coords - np.array([local_cx, local_cy])
        dist     = float(np.min(np.linalg.norm(diffs, axis=1)))

        sigma = ideal_dist_cells * 0.8
        return float(3.0 * np.exp(-0.5 * ((dist - ideal_dist_cells) / sigma) ** 2))

    def _corner_escape_bonus(self, px, py, prev_px, prev_py) -> float:
        """
        Returns a positive bonus when the agent moves away from the corner
        it is closest to, measured as Euclidean distance to the nearest corner.
        This fires even when moving_away from the nearest *wall* is ambiguous
        (e.g. diagonal corners where both walls are equidistant).
        """
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
        return max(0.0, curr_dist - prev_dist) * 3.5   # scale as needed

    
    def movement_direction_bonus(self, px, py):
        movement = 0.0
        if len(self._positions_history) >= 2:
            prev  = self._positions_history[-2]
            curr  = self._positions_history[-1]
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
        return movement
    
    def fire_revisit_reward(self, px, py, state: AgentState) -> float:
        x0, x1, y0, y1 = self._u_get_view_bounds_from_position(px, py, self.vp_size)

        revisit_falloff_sigma = 80

        fire_patch = self.view_acc.get_scene()[x0:x1, y0:y1, 1]
        fire_mask = fire_patch > self.fire_threshold
        if not np.any(fire_mask):
            return 0.0

        ts_patch = state.revisit_ts_map
        h = min(fire_patch.shape[0], ts_patch.shape[0])
        w = min(fire_patch.shape[1], ts_patch.shape[1])
        if h == 0 or w == 0:
            return 0.0
        fire_patch = fire_patch[:h, :w]
        ts_patch   = ts_patch[:h, :w]

        fire_mask = fire_patch > self.fire_threshold
        if not np.any(fire_mask):
            return 0.0

        valid = fire_mask & (ts_patch >= 0)
        if not np.any(valid):
            return 0.0

        delta = np.clip(self._step_count - ts_patch[valid], 0, None)
        gauss = np.exp(-0.5 * ((delta - self._ideal_revisit_delta) / revisit_falloff_sigma) ** 2)
        return float(np.mean(gauss))

    def _u_get_view_bounds_from_position(self, px, py, vp_size):
        half = vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))
        return x0, x1, y0, y1

    def _calculate_reward(self, state:AgentState):
        reward = 0.0
        step_advantage = 2.0 * (1.0 + (
            1.0 - GenericUtils.normalize_data(self._step_count, 0, self.iter_limit)
        ))
        px, py = state.pos_x, state.pos_y
        x0, x1, y0, y1 = self._u_get_view_bounds_from_position(px, py, state.vp_size)

        #local_view = self.view_acc.get_scene()[x0:x1, y0:y1, :]

        local_view = state.vp_image

        vp_scaling_fac = self._viewport_coverage_fraction(px, py) ** 2
        vp_norm_fac = self.vp_size * self.vp_size

        new_fuel_pixels  = float(np.sum(local_view[:, :, 0]))
        new_fire_pixels  = float(np.sum(local_view[:, :, 1]))
        seen_fuel_pixels = float(np.sum(local_view[:, :, 0]))

        rc_exploration      = (self._w_exploration * new_fuel_pixels * vp_scaling_fac) / vp_norm_fac
        rc_fire_disc        = (self._w_fire_discovery * new_fire_pixels * step_advantage * vp_scaling_fac) / vp_norm_fac
        rc_fuel_track       = (self._w_exploration_track * seen_fuel_pixels * vp_scaling_fac) / vp_norm_fac
        rc_perimeter_reward = (self._w_fire_tracking * self.fire_perimeter_alignment_reward((px, py), fire_patch=local_view[:, :, 1]) * vp_scaling_fac) ** 2 
        rc_proximity        = (self.fire_proximity_bonus(px, py) * vp_scaling_fac ) / 3
        rc_fire_revisit     = self.fire_revisit_reward(px, py, state) * 160
        #rc_fire_revisit = 0.0
        b_movement = self.movement_direction_bonus(px, py) / 3

        recency     = float(np.mean(state.recency_image))
        pc_recency_pen = ((math.exp(recency) - 1) * 15)**2
        
        pc_boundary    = self.calculate_near_boundary_penalty(px, py)


        self.log_reward_components(
            r_fire = rc_fire_disc,
            r_fuel = rc_exploration,
            r_movement = b_movement,
            r_fire_proximity = rc_proximity,
            r_fire_tangent = rc_perimeter_reward,
            p_bounds = pc_boundary,
            p_recency = pc_recency_pen,
            r_fire_revisit=rc_fire_revisit
        )

        reward += (
            rc_exploration
            + rc_fire_disc
            + rc_perimeter_reward
            + rc_proximity
            + b_movement
            + rc_fire_revisit
            - pc_boundary
            - pc_recency_pen
        )
        return reward
    
    

    def create_global_crop_viewport_obs(self, state:AgentState, sz=84):
        scene = self.view_acc.get_scene()   # (H, W, 2) — fuel + fire only
        agent_pos = (state.pos_x, state.pos_y)
        cx = int(agent_pos[0])
        cy = int(agent_pos[1])

        # loc_view = np.zeros((state.vp_size, state.vp_size, 2), dtype=np.float32)
        # loc_fuel = Viewpoint.get_square_viewpoint(scene[:, :, 0], (cx, cy), state.vp_size)
        # loc_fire = Viewpoint.get_square_viewpoint(scene[:, :, 1], (cx, cy), state.vp_size)
        # loc_view[:, :, 0], loc_view[:, :, 1] = loc_fuel, loc_fire

        loc_view = state.vp_image

        scene_crop = loc_view                                            # (H', W', 2)
        scene_crop_resized = cv2.resize(scene_crop, (84, 84), interpolation=cv2.INTER_AREA)
        scene_chw = np.transpose(scene_crop_resized, (2, 0, 1))                         # (2, 84, 84)

        if self._disable_recency_obs:
            return scene_chw.astype(np.float32)  # (2, 84, 84)

        # recency channel from the same spatial crop
        recency_crop = state.recency_image
        recency_resized = cv2.resize(recency_crop, (84, 84), interpolation=cv2.INTER_AREA)[None]  # (1, 84, 84)

        return np.concatenate([scene_chw, recency_resized], axis=0).astype(np.float32)  # (3, 84, 84)

    def _build_positions_obs(self) -> np.ndarray:
        obs = []
        last_pos = self._positions_history[-1] if len(self._positions_history) > 0 else self.start_poss
        px, py = last_pos[0], last_pos[1]
        # Normalised position
        obs.append(px / self.world_size[0])
        obs.append(py / self.world_size[1])

        # Velocity (from last step)
        if len(self._positions_history) >= 2:
            prev = self._positions_history[-2]
            vx = (px - prev[0]) / (self.step_size + 1e-8)
            vy = (py - prev[1]) / (self.step_size + 1e-8)
        else:
            vx, vy = 0.0, 0.0
        obs.append(np.clip(vx, -1, 1))
        obs.append(np.clip(vy, -1, 1))
        return np.asarray(obs, dtype=np.float32)

    def _is_sample_episode(self):
        return (self._episode_count % self.video_config.sample_interval == 0)

    def _step(self, action):
        reward, terminated, truncated, infos, obs = 0.0, False, False, {}, {}
        penality_scale = 1

        state:AgentState = self.agent_instance.step(action, step_id=self._step_count)
        self.view_acc.accumulate(state.vp_image, (state.pos_x, state.pos_y), state.vp_size)

        if not self.is_ue5_mode:
            self._track_fire_revisit(state.pos_x, state.pos_y, state)
            self._update_revisit_tracker(state.pos_x, state.pos_y)

        reward = self._calculate_reward(state)
        if state.is_oob:
            reward -= penality_scale * 55

        obs["viewport"] = self.create_global_crop_viewport_obs(state)
        obs["positions"] = self._build_positions_obs()

        # logging.info(f"[STEP] -> obs_positions = {obs['positions']}")

        self._positions_history.append((state.pos_x, state.pos_y))
        self._reward_history.append(reward)
        self._observation_history.append(obs)
        if not self.is_ue5_mode:
            self._fire_coverage_history.append(self.get_fire_coverage_percentage())
        self._step_count+=1

        infos["domain_metrics"] = self.log_metrics()

        if (self._episode_count % self.video_config.sample_interval) == 0:
            self.render()

        if self._step_count > self.iter_limit:
            truncated = True

        return obs, reward, terminated, truncated, infos



    def step(self, action):
        assert(len(action) == 2, f"Action shape mismatch, expected length 2, got length {len(action)}")
        return self._step(action)

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
        self._fig = plt.figure(
            figsize = (4 + 3 + 3 + 3, 5),
            facecolor="#1a1a2e"
        )
        gs = GridSpec(
            6, 4,
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08
        )

        self._ax_map                = self._fig.add_subplot(gs[:2, 0])   # GT map
        self._ax_obs                = self._fig.add_subplot(gs[:2, 1])   # Obs as passed to the model
        self._ax_global_vp          = self._fig.add_subplot(gs[:2, 2])   # Accumulated maps
        self._ax_global_recency     = self._fig.add_subplot(gs[:2, 3])   # Accumulated maps with global recency map overlaid

        self._ax_reward_plot        = self._fig.add_subplot(gs[2:4, :])
        self._ax_reward_comps_plot  = self._fig.add_subplot(gs[4:, :])

        # from the map manager, need to grab GT map and complete recency map
        # GT map is tricky for UE5 but okay for in memory map

        self._fig.canvas.manager.set_window_title(f"{self.env_id}")
        if self.video_config.is_enabled:
            self._fig.canvas.draw()
            buffer = cv2.cvtColor(
                np.asarray(self._fig.canvas.buffer_rgba()),
                cv2.COLOR_RGBA2BGR
            )
            h, w = buffer.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            os.makedirs(f"{self.video_config.base_path}", exist_ok=True)
            self.out = cv2.VideoWriter(
                f"{self.video_config.base_path}/{self.env_id}_{self._episode_count}.mp4",
                fourcc,
                self.video_config.fps,
                (w, h),
                isColor=True
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
    
    def _draw_agent_view_rectangle(self, ax:plt.Axes, pos:tuple):
        pos_x, pos_y = pos[0], pos[1]
        half = self.vp_size // 2
        rect = patches.Rectangle(
            (pos_y - half, pos_x - half),
            self.vp_size,
            self.vp_size,
            linewidth=1.2,
            edgecolor="green",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.scatter(pos_y, pos_x, s=60, color="green", linewidths=0.6)
    
    def _plot_reward_component(self, data_key:str, ax:plt.Axes, label:str, color:str):
        data = self._reward_components[f"{data_key}"]
        ax.plot(range(len(data)), data, color=color, label=f"{label}")

    def _draw_agent_trajectory(self, ax:plt.Axes):
        if len(self._positions_history) < 2:
            return
        xs, ys = zip(*self._positions_history)
        ax.plot(ys, xs, color="cyan", linewidth=3.0, alpha=1)

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
        
        if self._fig is None:
            self._init_figure()
        
        last_pos = self._positions_history[-1] if len(self._positions_history) > 0 else self.start_poss
        #wind_vec:tuple[float, float, float] = self.agent.get_wind_vector()

        # render GT map to its axis only if GT view is permitted
        if self.is_gt_visible:    
            gt_map = self.agent_instance.get_GT_map()
            gt_map_canvas = self._composite_rgb_map(self.world_size, gt_map[:, :, 1], gt_map[:, :, 0], None)
        else:
            gt_map_canvas = self._composite_rgb_map(self.world_size, None, None, None)
            
        self._ax_map.cla()
        self._ax_map.imshow(
            gt_map_canvas,
            cmap="YlOrRd",
            origin="upper",
            vmin = 0.0,
            vmax = 1.0,
            interpolation="nearest"
        )
        if not self.is_gt_visible: self._ax_map.text(100, 100, "[NO DATA]", size=15, color="green")
        self._ax_map.set_title("GT Map", color="white", fontsize=9, pad=4)
        self._ax_map.set_facecolor("#0d0d1a")
        self._ax_map.tick_params(colors="gray", labelsize=6)
        #self._draw_wind_arrow(self._ax_map, (wind_vec[0], wind_vec[1]), self.world_size)
        for s in self._ax_map.spines.values(): s.set_edgecolor("#444")

        # draw accumulated view
        scene = self.view_acc.get_scene()
        scene = self._composite_rgb_map(self.world_size, scene[:, :, 1], scene[:, :, 0], None)
        self._ax_global_vp.cla()
        self._ax_global_vp.imshow(
            scene,
            cmap="YlOrRd",
            origin="upper",
            vmin = 0.0,
            vmax = 1.0,
            interpolation="nearest"
        )
        self._draw_agent_view_rectangle(self._ax_global_vp, last_pos)
        self._draw_agent_trajectory(self._ax_global_vp)
        self._ax_global_vp.set_title("Accumulated Observations")
        self._ax_global_vp.set_facecolor("#0d0d1a")
        self._ax_global_vp.tick_params(colors="gray", labelsize=6)

        # draw agents observation
        latest_obs = self._observation_history[-1]["viewport"] if len(self._observation_history) > 0 else np.zeros((84, 84), dtype=np.float32)
        latest_obs = np.transpose(latest_obs, (1, 2, 0))
        view = self._composite_rgb_map(latest_obs.shape[:2], latest_obs[:, :, 1], latest_obs[:, :, 0], None)
        self._ax_obs.cla()
        self._ax_obs.imshow(
            view,
            cmap="YlOrRd",
            origin="upper",
            vmin = 0.0,
            vmax = 1.0,
            interpolation="nearest"
        )
        self._ax_obs.set_title("Observation")
        self._ax_obs.set_facecolor("#0d0d1a")
        self._ax_obs.tick_params(colors="gray", labelsize=6)

        # Draw recency map (Global)
        g_recency = self.agent_instance.get_recency_map()
        self._ax_global_recency.cla()
        self._ax_global_recency.imshow(
            g_recency,
            cmap="jet",
            origin="upper",
            vmin = 0.0,
            vmax = 1.0,
            interpolation="nearest"
        )
        self._draw_agent_view_rectangle(self._ax_global_recency, last_pos)
        self._draw_agent_trajectory(self._ax_global_recency)
        self._ax_global_recency.set_title("Global Recency Map")
        self._ax_global_recency.set_facecolor("#0d0d1a")
        self._ax_global_recency.tick_params(colors="gray", labelsize=6)

        # Plot reward history
        data = self._reward_history
        self._ax_reward_plot.cla()
        self._ax_reward_plot.plot(range(len(data)), data, color="red")
        # self._ax_reward_plot.set_xticks([])
        # self._ax_reward_plot.set_yticks([])
        self._ax_reward_plot.tick_params(colors="gray", labelsize=6)
        self._ax_reward_plot.set_ylabel("Reward", color="gray", fontsize=6)
        self._ax_reward_plot.set_xlabel("Steps", color="gray", fontsize=6)
        self._ax_reward_plot.set_title("Rewards over Steps", color="gray", fontsize=6)
        self._ax_reward_plot.grid(True)
        for s in self._ax_reward_plot.spines.values(): s.set_edgecolor("#333")

        # Plot reward components:
        self._ax_reward_comps_plot.cla()

        self._plot_reward_component("p_recency", self._ax_reward_comps_plot, "Recency P", "red")
        self._plot_reward_component("p_bounds", self._ax_reward_comps_plot, "Bounds P", "orange")
        self._plot_reward_component("r_fire", self._ax_reward_comps_plot, "Fire R", "green")
        self._plot_reward_component("r_fuel", self._ax_reward_comps_plot, "Fuel R", "yellow")
        self._plot_reward_component("r_movement", self._ax_reward_comps_plot, "Movement R", "blue")
        self._plot_reward_component("r_fire_proximity", self._ax_reward_comps_plot, "Proximity R", "purple")
        self._plot_reward_component("r_fire_tangent", self._ax_reward_comps_plot, "Perimeter R", "deepskyblue")
        self._plot_reward_component("r_fire_revisit", self._ax_reward_comps_plot, "Fire Revisit R", "magenta")
        
        # self._ax_reward_plot.set_xticks([])
        # self._ax_reward_plot.set_yticks([])
        self._ax_reward_comps_plot.tick_params(colors="gray", labelsize=6)
        self._ax_reward_comps_plot.set_ylabel("Reward Components", color="gray", fontsize=6)
        self._ax_reward_comps_plot.set_xlabel("Steps", color="gray", fontsize=6)
        self._ax_reward_comps_plot.set_title("Reward Comps over Steps", color="gray", fontsize=6)
        self._ax_reward_comps_plot.grid(True)
        self._ax_reward_comps_plot.legend()
        for s in self._ax_reward_comps_plot.spines.values(): s.set_edgecolor("#333")

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            if self.video_config.is_enabled and self.out is not None:
                frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                self.out.write(frame)
            return None

        self._fig.canvas.draw()
        if self.video_config.is_enabled and self.out is not None:
            frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            self.out.write(frame)
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3]
        
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

        if self.out is not None:
            self.out.release()
            self.out = None

    def get_revisit_deltas_minmeanmax(self):
        return np.min(self.revisit_deltas), np.mean(self.revisit_deltas), np.max(self.revisit_deltas)

    def get_revisit_delta_stats(self) -> dict:
        if self.is_ue5_mode:
            return {}
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

    def get_fire_coverage_percentage(self) -> float:
        # ------------------------------------------------------------
        # Ground truth may be hidden
        # ------------------------------------------------------------
        gt_map = self.agent_instance.get_GT_map()
        if gt_map is None:
            return -1.0

        accumulated_scene = self.view_acc.get_scene()

        gt_fire = gt_map[:, :, 1] > 0
        observed_fire = accumulated_scene[:, :, 1] > 0

        total_fire_pixels = np.sum(gt_fire)
        if total_fire_pixels == 0:
            return 0.0

        # ------------------------------------------------------------
        # Fire pixels that have been discovered
        # ------------------------------------------------------------
        discovered_fire_pixels = np.sum(gt_fire & observed_fire)

        coverage_percentage = (
            discovered_fire_pixels
            /
            total_fire_pixels
            *
            100.0
        )

        return float(
            coverage_percentage
        )
    
    def get_mean_fire_coverage(self) -> float:
        coverage = np.asarray(self._fire_coverage_history, dtype=np.float32)
        valid = coverage >= 0
        if not np.any(valid):
            return -1.0
        return float(
            np.mean(
                coverage[valid]
            )
        )
    
    def get_fire_coverage_auc(self, normalize=True) -> float:
        coverage = np.asarray(self._fire_coverage_history, dtype=np.float32)
        valid = coverage >= 0
        if not np.any(valid):
            return -1.0
        coverage = coverage[valid]
        if len(coverage) == 1:
            return float(coverage[0])

        timesteps = np.arange(len(coverage), dtype=np.float32)

        auc = np.trapezoid(coverage, timesteps)
        if normalize:
            auc = auc / (len(coverage) - 1)

        return float(auc)


    def log_metrics(self):
        #min_revisit_d, mean_revisit_d, max_revisit_d = self.get_revisit_deltas_minmeanmax()
        
        metrics = {}
        metrics.update(self.get_revisit_delta_stats())


        metrics["domain/fire_coverage_mean"] = self.get_mean_fire_coverage()
        metrics["domain/fire_coverage_AUC"] = self.get_fire_coverage_auc()
        metrics["domain/fire_coverage_final"] = self._fire_coverage_history[-1] if len(self._fire_coverage_history) > 0 else -1.0

        return metrics
    