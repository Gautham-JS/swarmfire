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

import logging

log = logging.getLogger(__name__)


class SingleAgentEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "multi_drone_v0"}

    def __init__(
        self,
        n_agents,
        world_size,
        start_positions: list = None,
        iter_limit=4500,
        vp_size=64,
        seed=None,
        fixed_seed=False,
        env_id="MultiAgentEnv",
        render_mode="human",
        sample_interval=100,
        save_interval=500,
        is_vid_out=False,
        vid_base_path="./vids/",
        vid_id="test_",
        phase_weights: dict = None,
        device=None,
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

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._episode_count = 0

        self._recency_decay = 0.9995
        self._recency_visit_bump = 0.03

        self.set_reward_weights(phase_weights)

        self.vp_size = vp_size
        self.step_size = 1
        self.map_update_interval = 20
        self.stepped_map_update = False
        self.reduction_factor = 2

        self.actions_per_agent = 2
        self.n_actions = self.n_agents * self.actions_per_agent

        self.action_space = gym.spaces.MultiDiscrete([3] * (self.n_agents * 2))

        self.observation_space = gym.spaces.Dict(
            {
                "viewport": gym.spaces.Box(low=0.0, high=1.0, shape=(3, 84, 84), dtype=np.float32),
                "positions": gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents * 6,), dtype=np.float32),
            }
        )

        self.world_gen = Generators.FuelMapGenerator(self.world_size)
        self.view_acc = Viewpoint.IncrementalViewAccumulator(self.world_size, 2)

        # Wind - set in reset(), used by reward and render
        self.wind_vector = (1.0, 0.0)   # (wx, wy) unit vector; default east
        self.wind_mag = 1.0

        # vars initialized by reset fn
        self.map = None
        self.visited_map = None
        self.agents = None
        self._obs_hsitory = None
        self._reward_history = None
        self._view_history = None
        self._pos_history = None
        self._agent_positions = None
        self._step_count = 0

        self.recency_map = None
        self.fire_disc_map = None

        self._fig = None
        self._axes = None

        self.init_domain_logs()

        self.env_state_dict = {
            "map_size": [self.world_size[0], self.world_size[1]],
            "vp_size": self.vp_size,
            "recency_decay_factor": self._recency_decay,
            "recency_visit_bump": self._recency_visit_bump,
            "start_pos": self.start_poss,
        }

        self.is_vid_out = is_vid_out
        self.out = None
        if is_vid_out:
            self.vid_base_path = vid_base_path
            self.vid_id = vid_id
            self.fps = 30

    # ─────────────────────────────────────────────────────────────────────────
    # Domain logging
    # ─────────────────────────────────────────────────────────────────────────

    def init_domain_logs(self):
        self.fire_coverage_timesteps_dict = {
            "overall": [], "25": self.iter_limit, "50": self.iter_limit,
            "75": self.iter_limit, "90": self.iter_limit, "99": self.iter_limit,
        }
        self.fuel_coverage_timesteps_dict = {
            "overall": [], "25": self.iter_limit, "50": self.iter_limit,
            "75": self.iter_limit, "90": self.iter_limit, "99": self.iter_limit,
        }
        self.fire_coverage_percentages = []
        self.fuel_coverage_percentages = []

    def _init_revisit_tracker(self):
        self._revisit_last_seen = np.full(self.world_size, -1, dtype=np.int32)
        self._revisit_deltas = []
        self._revisit_state = np.zeros(self.world_size, dtype=np.uint8)
        self._revisit_gap_threshold = max(1, self.vp_size * 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Reward weights
    # ─────────────────────────────────────────────────────────────────────────

    def set_reward_weights(self, weights: dict):
        if weights is None:
            weights = {}
        self._w_exploration = weights.get("exploration", 1.0)
        self._w_exploration_track = weights.get("exploration_tracking", 1.0)
        self._w_fire_discovery = weights.get("fire_discovery", 1.0)
        self._w_fire_tracking = weights.get("fire_tracking", 1.0)
        self._w_risk = weights.get("risk", 1.0)
        self._w_upwind = weights.get("upwind_observation", 10.0)   # NEW
        log.debug(
            f"[REWARD_WEIGHTS_UPDATE] -> exploration : {self._w_exploration} "
            f"| fire discovery : {self._w_fire_discovery} "
            f"| fire tracking : {self._w_fire_tracking} "
            f"| risk : {self._w_risk} "
            f"| upwind_observation : {self._w_upwind}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Action / viewpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size

    def extract_viewpoint(self, x, y):
        fuel_view, recently_visited_fuel, delta_fuel_mask = Viewpoint.get_square_viewpoint_and_mark_visited(
            self.map[:, :, 0], self.visited_map, (x, y), size=self.vp_size
        )
        fire_view, recently_visited_fire, delta_fire_mask = Viewpoint.get_square_viewpoint_and_mark_visited(
            self.map[:, :, 1], self.visited_map, (x, y), size=self.vp_size
        )
        self.visited_map = recently_visited_fuel
        view = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        deltas = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        view[:, :, 0], view[:, :, 1] = fuel_view, fire_view
        deltas[:, :, 0], deltas[:, :, 1] = delta_fuel_mask, delta_fire_mask
        return view, deltas

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if self._episode_count != 0 and self._episode_count % self.sample_int == 0:
            if len(self._reward_history) > 10:
                log.debug(
                    f"[RESET] Reward history - MAX : {np.max(self._reward_history[10:])} "
                    f"| MEAN : {np.mean(self._reward_history[10:])} "
                    f"| MIN : {np.min(self._reward_history[10:])}"
                )

        self._obs_hsitory = []
        self._agent_positions = []
        self._reward_history = []
        self._view_history = []
        self._pos_history = []
        self._step_count = 0
        self._episode_count += 1
        self._visited_frac = 0.0
        self._last_global_vp = None

        self.init_domain_logs()
        self._init_revisit_tracker()

        self.visited_map = np.zeros(self.world_size, dtype=np.bool_)
        self.recency_map = np.zeros(self.world_size, dtype=np.float32)
        self.fire_disc_map = np.full(self.world_size, -1, dtype=np.int32)
        self.view_acc.reset()

        if not self.fixed_seed and self.seed is not None and self._episode_count > 10:
            if (self._episode_count - 1) % self.map_update_interval == 0:
                self.seed += 1
                if self.stepped_map_update:
                    self.map_update_interval = max(1, self.map_update_interval // 2)

        # ── Generate map + wind ───────────────────────────────────────────────
        self.map, wind_components = self.world_gen.create_map(0.001, 0.003, seed=self.seed)
        self.wind_vector = (wind_components[0], wind_components[1])
        self.wind_mag = wind_components[2]

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_instances = [Drone.Drone(f"agent_{i}") for i in range(self.n_agents)]

        fire_coords = np.argwhere(self.map[:, :, 1] > 0)
        if len(fire_coords) > 0 and self._episode_count < -1:
            scatter = self.vp_size * 3
            self._agent_positions = []
            for _ in range(self.n_agents):
                centre = fire_coords[np.random.randint(len(fire_coords))]
                px = int(np.clip(centre[0] + np.random.randint(-scatter, scatter), 0, self.world_size[0] - 1))
                py = int(np.clip(centre[1] + np.random.randint(-scatter, scatter), 0, self.world_size[1] - 1))
                self._agent_positions.append((px, py))
        elif self.start_poss is not None:
            self._agent_positions = self.start_poss[: self.n_agents]
        else:
            self._agent_positions = [
                (np.random.randint(0, self.world_size[0]), np.random.randint(0, self.world_size[1]))
                for _ in range(self.n_agents)
            ]

        for p, a in zip(self._agent_positions, self.agent_instances):
            a.set_position({"x": p[0], "y": p[1], "z": 0})

        self._pos_history = [list(self._agent_positions)]

        obs = {
            "viewport": np.zeros((3, 84, 84), dtype=np.float32),
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

    # ─────────────────────────────────────────────────────────────────────────
    # Reward helpers
    # ─────────────────────────────────────────────────────────────────────────

    def exploration_reward(self, delta, local_view, c_delta=1, c_view=1):
        return (c_delta * (np.sum(delta[:, :, 0]) / self.vp_size ** 2)) + (
            c_view * (np.sum(local_view[:, :, 0]) / self.vp_size ** 2)
        )

    def fire_reward(self, delta, local_view, c_fd=1, c_ft=1):
        return (c_fd * (np.sum(delta[:, :, 1]) / self.vp_size ** 2)) + (
            c_ft * (np.sum(local_view[:, :, 1]) / self.vp_size ** 2)
        )

    def recency_penality(self, px, py, c_rp=10):
        recency_patch = self.extract_recency_map(px, py)
        return float(np.mean(recency_patch)) * c_rp

    def novelty_reward(self, px, py, c_rp=5):
        recency_patch = self.extract_recency_map(px, py)
        return (1 - float(np.mean(recency_patch))) * c_rp

    def mark_all_recency(self):
        for pos in self._agent_positions:
            px, py = pos
            self.mark_recency_map(px, py)

    # ── NEW: upwind observation reward ────────────────────────────────────────

    def upwind_observation_reward(self, px: int, py: int, scale: float = 8.0) -> float:
        """
        Reward for observing fire from an upwind position.

        The agent is rewarded when fire pixels within its viewport lie
        *in the direction the wind is blowing* relative to the agent —
        i.e. the agent is upwind (behind the wind) and looking toward
        the fire front.

        Geometry:
            wind_unit_vector  ŵ = (wx, wy)
            agent→fire vector f̂ = normalise(fire_centroid − agent_pos)
            score = dot(f̂, ŵ)  ∈ [−1, 1]

        Score is positive only when the fire centroid is downwind of the
        agent, matching the physical scenario where a drone observes a
        fire from behind the wind line.  The result is clamped to [0, 1]
        so there is no penalty for being downwind (other rewards handle
        positioning).

        Args:
            px, py   : agent row / col position (integer pixels)
            scale    : maximum reward magnitude (before vp_scale)
        """
        wx, wy = self.wind_vector

        # Sample fire pixels from the accumulated observation (not ground truth)
        scene = self.view_acc.get_scene()
        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))

        fire_patch = scene[x0:x1, y0:y1, 1]
        fire_coords_local = np.argwhere(fire_patch > 0)   # (N, 2) row/col in patch

        if len(fire_coords_local) == 0:
            return 0.0

        # Convert patch-local coords back to world coords
        fire_rows = fire_coords_local[:, 0] + x0
        fire_cols = fire_coords_local[:, 1] + y0

        # Centroid of visible fire in world coords
        fire_cx = float(np.mean(fire_rows))
        fire_cy = float(np.mean(fire_cols))

        # Vector from agent to fire centroid (row = x-axis, col = y-axis)
        dx = fire_cx - px
        dy = fire_cy - py
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-6:
            return 0.0

        # Unit vector pointing from agent toward fire
        agent_to_fire_x = dx / dist
        agent_to_fire_y = dy / dist

        # Dot product with wind direction:
        #   positive → fire is downwind of agent  → agent is upwind  ✓
        #   negative → fire is upwind of agent    → agent is downwind ✗
        alignment = agent_to_fire_x * wx + agent_to_fire_y * wy

        alignment = -1 * alignment * self.wind_mag

        # Only reward upwind positioning; no penalty for the opposite
        return float(np.clip(alignment, 0.0, 100.0)) * scale

    # ─────────────────────────────────────────────────────────────────────────
    # Boundary / viewport helpers
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_near_boundary_penalty(self, x, y):
        margin = self.vp_size // 2
        penalty = 0.0
        for coord, limit in [(x, self.world_size[0]), (y, self.world_size[1])]:
            dist_low = coord
            dist_high = limit - coord
            if dist_low < margin:
                penalty += np.exp((margin - dist_low) / (margin / 4)) - 1
            if dist_high < margin:
                penalty += np.exp((margin - dist_high) / (margin / 4)) - 1
        return penalty

    def _viewport_coverage_fraction(self, px, py) -> float:
        half = self.vp_size // 2
        x0 = max(px - half, 0)
        x1 = min(px + half, self.world_size[0])
        y0 = max(py - half, 0)
        y1 = min(py + half, self.world_size[1])
        valid_pixels = (x1 - x0) * (y1 - y0)
        return valid_pixels / (self.vp_size * self.vp_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Fire crossing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_fire_crossing_penalty(self, px, py) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))
        scene = self.view_acc.get_scene()
        fire_value = float(scene[cx, cy, 1])
        if fire_value <= 0.0:
            return 0.0
        return 80.0 * (0.5 + 0.5 * fire_value)

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

            gx = sobel(fire_patch.astype(np.float32), axis=1)
            gy = sobel(fire_patch.astype(np.float32), axis=0)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            total_mag = mag.sum()
            if total_mag < 1e-6:
                continue

            boundary_normal = np.array(
                [np.sum(gy * mag) / total_mag, np.sum(gx * mag) / total_mag], dtype=np.float64
            )
            bn_norm = np.linalg.norm(boundary_normal)
            if bn_norm < 1e-8:
                continue
            boundary_normal /= bn_norm
            boundary_tangent = np.array([-boundary_normal[1], boundary_normal[0]])

            if len(self._pos_history) < 2:
                continue
            prev = self._pos_history[-2][i]
            curr = self._pos_history[-1][i]
            vel = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)
            vel_norm = np.linalg.norm(vel)
            if vel_norm < 1e-6:
                continue
            vel_dir = vel / vel_norm
            alignment = float(np.dot(vel_dir, boundary_tangent))
            total += max(0.0, alignment) * 12.0

        return total

    def fire_proximity_bonus(self, px, py, ideal_dist_cells: float = 2.0) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))
        scene = self.view_acc.get_scene()
        if scene[cx, cy, 1] > 0:
            return 0.0
        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))
        fire_patch = scene[x0:x1, y0:y1, 1]
        fire_coords = np.argwhere(fire_patch > 0)
        if len(fire_coords) == 0:
            return 0.0
        local_cx, local_cy = cx - x0, cy - y0
        diffs = fire_coords - np.array([local_cx, local_cy])
        dist = float(np.min(np.linalg.norm(diffs, axis=1)))
        sigma = ideal_dist_cells * 0.8
        return float(3.0 * np.exp(-0.5 * ((dist - ideal_dist_cells) / sigma) ** 2))

    def _estimate_occluded_area(self, px, py) -> float:
        scene = self.view_acc.get_scene()
        n_rays = 16
        max_range = int(max(self.world_size) * 0.75)
        step_size = 2
        occluded = 0
        total_rays = 0

        for angle_idx in range(n_rays):
            angle = 2 * np.pi * angle_idx / n_rays
            dx, dy = np.cos(angle), np.sin(angle)
            in_fire = False
            ray_occluded = 0
            for dist in range(step_size, max_range, step_size):
                rx = int(round(px + dx * dist))
                ry = int(round(py + dy * dist))
                if rx < 0 or rx >= self.world_size[0] or ry < 0 or ry >= self.world_size[1]:
                    break
                if scene[rx, ry, 1] > 0:
                    in_fire = True
                if in_fire and not self.visited_map[rx, ry]:
                    ray_occluded += 1
            occluded += ray_occluded
            total_rays += max_range // step_size

        return float(occluded) / float(total_rays) if total_rays else 0.0

    def _fire_crossing_opportunity_reward(self, px, py) -> float:
        cx = int(np.clip(px, 0, self.world_size[0] - 1))
        cy = int(np.clip(py, 0, self.world_size[1] - 1))
        scene = self.view_acc.get_scene()
        if scene[cx, cy, 1] <= 0:
            return 0.0
        occluded_fraction = self._estimate_occluded_area(px, py)
        thin_radius = self.vp_size // 4
        x0 = int(np.clip(px - thin_radius, 0, self.world_size[0]))
        x1 = int(np.clip(px + thin_radius, 0, self.world_size[0]))
        y0 = int(np.clip(py - thin_radius, 0, self.world_size[1]))
        y1 = int(np.clip(py + thin_radius, 0, self.world_size[1]))
        local_fire_patch = scene[x0:x1, y0:y1, 1]
        local_fire_density = float(np.mean(local_fire_patch > 0))
        thinness = 1.0 - local_fire_density
        return occluded_fraction * thinness * 150.0

    def _corner_escape_bonus(self, px, py, prev_px, prev_py) -> float:
        corners = [
            (0, 0), (0, self.world_size[1]),
            (self.world_size[0], 0), (self.world_size[0], self.world_size[1]),
        ]

        def min_corner_dist(x, y):
            return min(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for cx, cy in corners)

        prev_dist = min_corner_dist(prev_px, prev_py)
        curr_dist = min_corner_dist(px, py)
        return max(0.0, curr_dist - prev_dist) * 3.5

    # ─────────────────────────────────────────────────────────────────────────
    # Main reward function
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_reward(self, delta_views, recency_map):
        reward = 0.0

        step_advantage = 2.0 * (
            1.0 + (1.0 - GenericUtils.normalize_data(self._step_count, 0, self.iter_limit))
        )

        for i, (delta, pos) in enumerate(zip(delta_views, self._agent_positions)):
            px, py = pos
            half = self.vp_size // 2
            x0 = int(np.clip(px - half, 0, self.world_size[0]))
            x1 = int(np.clip(px + half, 0, self.world_size[0]))
            y0 = int(np.clip(py - half, 0, self.world_size[1]))
            y1 = int(np.clip(py + half, 0, self.world_size[1]))

            local_view = self.view_acc.get_scene()[x0:x1, y0:y1, :]
            vp_scale = self._viewport_coverage_fraction(px, py) ** 2

            new_fuel_pixels = float(np.sum(delta[:, :, 0]))
            new_fire_pixels = float(np.sum(delta[:, :, 1]))
            seen_fuel_pixels = float(np.sum(local_view[:, :, 0] > 0))

            exploration = self._w_exploration * 0.03 * new_fuel_pixels * vp_scale
            fire_disc = self._w_fire_discovery * 0.09 * new_fire_pixels * step_advantage * vp_scale
            fuel_track = self._w_exploration_track * 0.001 * seen_fuel_pixels * vp_scale
            perimeter_reward = self._w_fire_tracking * self.fire_perimeter_alignment_reward([delta]) * vp_scale
            proximity = self.fire_proximity_bonus(px, py) * vp_scale

            # ── Upwind observation reward (NEW) ───────────────────────────────
            upwind_reward = self._w_upwind * self.upwind_observation_reward(px, py) * vp_scale

            # ── Fire crossing ─────────────────────────────────────────────────
            crossing_opportunity = self._fire_crossing_opportunity_reward(px, py)
            fire_cross = self._w_risk * self.calculate_fire_crossing_penalty(px, py)
            net_fire_cost = fire_cross - crossing_opportunity

            # ── Movement ──────────────────────────────────────────────────────
            if len(self._pos_history) >= 2:
                prev = self._pos_history[-2][i]
                curr = self._pos_history[-1][i]
                moved = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) > 0.2

                dist_to_nearest_wall = min(px, self.world_size[0] - px, py, self.world_size[1] - py)
                in_margin = dist_to_nearest_wall < self.vp_size // 2

                if moved:
                    if in_margin:
                        prev_dist = min(
                            prev[0], self.world_size[0] - prev[0],
                            prev[1], self.world_size[1] - prev[1],
                        )
                        curr_dist = min(
                            curr[0], self.world_size[0] - curr[0],
                            curr[1], self.world_size[1] - curr[1],
                        )
                        moving_away = curr_dist > prev_dist
                        movement = 2.0 if moving_away else -0.3
                        movement += self._corner_escape_bonus(px, py, prev[0], prev[1])
                    else:
                        movement = 0.5
                else:
                    movement = -1.5
            else:
                movement = 0.0

            recency = float(np.mean(self.extract_recency_map(px, py)))
            recency_pen = (math.exp(recency) - 1) / (math.exp(1) - 1) * 140
            boundary = self.calculate_near_boundary_penalty(px, py)

            reward += (
                exploration
                + fire_disc
                + fuel_track
                + perimeter_reward
                + proximity
                + upwind_reward      
                + movement
                - recency_pen
                - boundary
                - net_fire_cost
            )

            return reward

    # ─────────────────────────────────────────────────────────────────────────
    # Recency map
    # ─────────────────────────────────────────────────────────────────────────

    def mark_recency_map(self, px, py):
        half = self.vp_size // 2
        x0 = np.clip(px - half, 0, self.world_size[0])
        x1 = np.clip(px + half, 0, self.world_size[0])
        y0 = np.clip(py - half, 0, self.world_size[1])
        y1 = np.clip(py + half, 0, self.world_size[1])
        self.recency_map[x0:x1, y0:y1] += self._recency_visit_bump
        self.recency_map[x0:x1, y0:y1] = np.minimum(
            self.recency_map[x0:x1, y0:y1],
            np.ones((x1 - x0, y1 - y0), dtype=np.float32),
        )
        return self.recency_map[x0:x1, y0:y1]

    def extract_recency_map(self, px, py):
        half = self.vp_size // 2
        x0 = np.clip(px - half, 0, self.world_size[0])
        x1 = np.clip(px + half, 0, self.world_size[0])
        y0 = np.clip(py - half, 0, self.world_size[1])
        y1 = np.clip(py + half, 0, self.world_size[1])
        return self.recency_map[x0:x1, y0:y1].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────────

    def create_global_crop_viewport_obs(self, sz=84):
        scene = self.view_acc.get_scene()
        cx = int(np.mean([p[0] for p in self._agent_positions]))
        cy = int(np.mean([p[1] for p in self._agent_positions]))

        loc_view = np.zeros((self.vp_size, self.vp_size, 2), dtype=np.float32)
        loc_fuel = Viewpoint.get_square_viewpoint(scene[:, :, 0], (cx, cy), self.vp_size)
        loc_fire = Viewpoint.get_square_viewpoint(scene[:, :, 1], (cx, cy), self.vp_size)
        loc_view[:, :, 0], loc_view[:, :, 1] = loc_fuel, loc_fire

        scene_crop_resized = cv2.resize(loc_view, (84, 84), interpolation=cv2.INTER_AREA)
        scene_chw = np.transpose(scene_crop_resized, (2, 0, 1))

        wind_vector_x = np.ones((84, 84, 1), dtype=np.float32)
        wind_vector_y = np.ones((84, 84, 1), dtype=np.float32)
        wind_vector_x[:, :, 0], wind_vector_y[:, :, 0]  = self.wind_vector[0] * self.wind_mag, self.wind_vector[1] * self.wind_mag
        wind_vector_x_chw, wind_vector_y_chw = np.transpose(wind_vector_x, (2, 0, 1)), np.transpose(wind_vector_y, (2, 0, 1))

        recency_crop = Viewpoint.get_square_viewpoint(self.recency_map, (cx, cy), self.vp_size)
        recency_resized = cv2.resize(recency_crop, (84, 84), interpolation=cv2.INTER_AREA)[None]

        vobs = np.concatenate([scene_chw, recency_resized], axis=0).astype(np.float32)
        return vobs

    def _build_positions_obs(self) -> np.ndarray:
        obs = []
        scene = self.view_acc.get_scene()
        fire_coords = np.argwhere(scene[:, :, 1] > 0)

        for i, pos in enumerate(self._agent_positions):
            px, py = pos
            obs.append(px / self.world_size[0])
            obs.append(py / self.world_size[1])
            if len(self._pos_history) >= 2:
                prev = self._pos_history[-2][i]
                vx = (px - prev[0]) / (self.step_size + 1e-8)
                vy = (py - prev[1]) / (self.step_size + 1e-8)
            else:
                vx, vy = 0.0, 0.0
            obs.append(np.clip(vx, -1, 1))
            obs.append(np.clip(vy, -1, 1))
            obs.append(self.wind_vector[0] * self.wind_mag)
            obs.append(self.wind_vector[1] * self.wind_mag)

        return np.asarray(obs, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action):
        reward, terminated, truncated, infos, obs = 0.0, False, False, {}, {}
        loc_pos_history = []
        loc_view_history = []
        penality = 0
        poss = []
        deltas = []
        per_agent_views = []

        self.recency_map *= self._recency_decay

        for i in range(self.n_agents):
            agent: Drone.Drone = self.agent_instances[i]
            agent_actions = action[self.actions_per_agent * i: self.actions_per_agent * (i + 1)]

            dx = self.get_position_delta_from_action(agent_actions[0])
            dy = self.get_position_delta_from_action(agent_actions[1])
            agent.inject_velocity({"x": dx, "y": dy, "z": 0})

            px = int(agent.get_position_array()[0])
            py = int(agent.get_position_array()[1])

            self._agent_positions[i] = (px, py)

            if px >= self.world_size[0] or px < 0 or py >= self.world_size[1] or py < 0:
                px = np.clip(px, 0, self.world_size[0] - 1)
                py = np.clip(py, 0, self.world_size[1] - 1)
                self._agent_positions[i] = (px, py)
                agent.set_position({"x": px, "y": py, "z": 0})
                penality += 55

            view, delta_view = self.extract_viewpoint(px, py)
            view_recency = self.extract_recency_map(px, py)

            view_fuel = Viewpoint.get_square_viewpoint(self.map[:, :, 0], (px, py), self.vp_size)
            view_fire = Viewpoint.get_square_viewpoint(self.map[:, :, 1], (px, py), self.vp_size)
            view_agent = np.stack([view_fuel, view_fire], axis=0)

            small = np.stack([
                cv2.resize(view_agent[0], (84, 84), interpolation=cv2.INTER_AREA),
                cv2.resize(view_agent[1], (84, 84), interpolation=cv2.INTER_AREA),
                cv2.resize(view_recency, (84, 84), interpolation=cv2.INTER_AREA),
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
            h, w = x1 - x0, y1 - y0
            new_fire_mask = delta[:h, :w, 1] > 0
            undiscovered = self.fire_disc_map[x0:x1, y0:y1] == -1
            stamp_mask = new_fire_mask & undiscovered
            self.fire_disc_map[x0:x1, y0:y1][stamp_mask] = self._step_count

        scene_obs = self.view_acc.get_scene()

        total_reward = self.calculate_reward(deltas, self.recency_map)
        self.mark_all_recency()
        self._update_revisit_tracker()
        total_reward -= penality

        if (
            self._episode_count != 0
            and (self._episode_count % self.sample_int == 0)
            and (self._step_count % 50 == 0)
        ):
            log.debug(f"[CALC REWARD] : total reward value = {total_reward} | penality scale : {penality}")

        self._reward_history.append(total_reward)
        self._obs_hsitory.append(scene_obs)
        self._view_history.append(loc_view_history)
        self._pos_history.append(loc_pos_history)

        self._step_count += 1
        self.log_coverage_percentiles()

        if (self._episode_count % self.sample_int) == 0:
            self.render()

        infos["domain_metrics"] = self.log_metrics()

        obs["viewport"] = self.create_global_crop_viewport_obs()
        obs["positions"] = self._build_positions_obs()

        if self._step_count > self.iter_limit:
            truncated = True
        self._last_global_vp = obs["viewport"].copy()

        return obs, total_reward, terminated, truncated, infos

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _channels_to_rgb(self, chw: np.ndarray) -> np.ndarray:
        c, h, w = chw.shape
        if c == 1:
            grey = chw[0]
            return np.stack([grey, grey, grey], axis=-1)
        if c == 2:
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            canvas[:, :, 1] = chw[0]
            canvas[:, :, 0] = chw[1]
            return canvas
        if c == 3:
            canvas = np.zeros((h, w, 3), dtype=np.float32)
            canvas[:, :, 1] = chw[0]
            canvas[:, :, 0] = chw[1]
            canvas[:, :, 2] = chw[2]
            return canvas
        flat = chw.reshape(c, -1).T
        flat -= flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        projected = flat @ Vt[:3].T
        projected -= projected.min(axis=0)
        denom = projected.max(axis=0)
        denom[denom == 0] = 1
        projected /= denom
        return projected.reshape(h, w, 3).astype(np.float32)

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

    def _draw_wind_arrow(self, ax, wind_vector, map_shape):
        """
        Overlay a wind direction arrow on a map axes.

        The arrow is anchored 10 % in from the top-right corner and
        points in the wind direction.  A short compass-style label
        (e.g. "W →") is placed beside it.

        Args:
            ax          : matplotlib Axes showing the map
            wind_vector : (wx, wy) unit vector.
                          wx > 0 = east (rightward in image x/col axis)
                          wy > 0 = south (downward in image y/row axis)
            map_shape   : (H, W) of the displayed map
        """
        H, W = map_shape

        # Arrow anchor (image coords: x = col, y = row)
        anchor_x = W * 0.88
        anchor_y = H * 0.10

        # Wind vector: wx maps to image +x (col), wy maps to image +y (row)
        wx, wy = wind_vector
        arrow_length = min(H, W) * 0.10   # 10 % of the shorter map dimension

        dx_img = wx * arrow_length   # positive → rightward
        dy_img = wy * arrow_length   # positive → downward (image y)

        ax.annotate(
            "",
            xy=(anchor_x + dx_img, anchor_y + dy_img),
            xytext=(anchor_x, anchor_y),
            arrowprops=dict(
                arrowstyle="->,head_width=0.4,head_length=0.3",
                color="deepskyblue",
                lw=2.0,
            ),
        )

        # Cardinal label derived from wind direction
        angle_deg = math.degrees(math.atan2(wy, wx)) % 360
        if angle_deg < 22.5 or angle_deg >= 337.5:
            compass = "E"
        elif angle_deg < 67.5:
            compass = "SE"
        elif angle_deg < 112.5:
            compass = "S"
        elif angle_deg < 157.5:
            compass = "SW"
        elif angle_deg < 202.5:
            compass = "W"
        elif angle_deg < 247.5:
            compass = "NW"
        elif angle_deg < 292.5:
            compass = "N"
        else:
            compass = "NE"

        label_x = anchor_x + dx_img * 1.35
        label_y = anchor_y + dy_img * 1.35
        ax.text(
            label_x, label_y,
            f"wind\n{compass}",
            color="deepskyblue",
            fontsize=6,
            ha="center", va="center",
            bbox=dict(facecolor="#00000088", edgecolor="none", pad=1.5),
        )

    def _init_figure(self):
        n_agents = max(self.n_agents, 1)
        self._fig = plt.figure(
            figsize=(4 + 3 + 3 + 3 * n_agents, 5),
            facecolor="#1a1a2e",
        )
        gs = GridSpec(
            2, 3 + n_agents,
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08,
        )
        self._ax_map = self._fig.add_subplot(gs[:, 0])
        self._ax_obs = self._fig.add_subplot(gs[:, 1])
        self._ax_global_vp = self._fig.add_subplot(gs[:, 2])
        self._ax_viewports = [
            (self._fig.add_subplot(gs[0, i + 3]), self._fig.add_subplot(gs[1, i + 3]))
            for i in range(n_agents)
        ]
        self._fig.canvas.manager.set_window_title(self.env_id)

        if self.is_vid_out and self.out is None and (self._episode_count % self.save_int == 0):
            self._fig.canvas.draw()
            buf = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            h, w = buf.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            os.makedirs(self.vid_base_path, exist_ok=True)
            self.out = cv2.VideoWriter(
                f"{self.vid_base_path}/{self.vid_id}_{self._episode_count}.mp4",
                fourcc, self.fps, (w, h), isColor=True,
            )
        if self.render_mode == "human":
            plt.ion()
            plt.show()

    def get_render_as_img(self):
        b = self._fig.axes[0].get_window_extent()
        img = np.array(self._fig.canvas.buffer_rgba())
        img = img[int(b.y0): int(b.y1), int(b.x0): int(b.x1), :]
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        return img

    # ─────────────────────────────────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._fig is None:
            log.info("Initializing figures")
            self._init_figure()

        agents = list(self.agents)
        n_agents = len(agents)

        # ── Full map ──────────────────────────────────────────────────────────
        map_image = self._composite_rgb_map(
            self.map.shape[:2], self.map[:, :, 1], self.map[:, :, 0], None
        )
        ax = self._ax_map
        ax.cla()
        ax.imshow(map_image, cmap="YlOrRd", origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title("Map", color="white", fontsize=9, pad=4)
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # ── Wind arrow on the map panel ───────────────────────────────────────
        self._draw_wind_arrow(ax, self.wind_vector, self.map.shape[:2])

        # ── Accumulated observation ───────────────────────────────────────────
        if len(self._obs_hsitory) > 0:
            last_obs = self._obs_hsitory[-1]
            obs_image = self._composite_rgb_map(last_obs.shape[:2], last_obs[:, :, 1], last_obs[:, :, 0], None)
            ax_obs = self._ax_obs
            ax_obs.cla()
            ax_obs.imshow(obs_image, cmap="YlOrRd", origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_obs.set_title("Accumulated Observation", color="white", fontsize=9, pad=4)
            ax_obs.set_facecolor("#0d0d1a")
            ax_obs.tick_params(colors="gray", labelsize=6)
            for spine in ax_obs.spines.values():
                spine.set_edgecolor("#444")

            # ── Wind arrow on accumulated obs panel too ───────────────────────
            self._draw_wind_arrow(ax_obs, self.wind_vector, last_obs.shape[:2])

        # ── Agent positions & viewport footprints on map ──────────────────────
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_agents, 1)))
        for i, aid in enumerate(agents):
            pos = self._agent_positions[i]
            if pos is None:
                continue
            x, y = int(pos[1]), int(pos[0])
            half = self.vp_size // 2

            rect = patches.Rectangle(
                (x - half, y - half), self.vp_size, self.vp_size,
                linewidth=1.2, edgecolor=colors[i], facecolor="none",
                linestyle="--", alpha=0.7,
            )
            ax.add_patch(rect)
            ax.scatter(x, y, s=60, color=colors[i], zorder=5, edgecolors="white", linewidths=0.6)
            ax.annotate(aid, (x, y), textcoords="offset points", xytext=(4, 4),
                        color=colors[i], fontsize=6, fontweight="bold")

        # ── Agent trajectory lines ────────────────────────────────────────────
        pos_array = np.asarray(self._pos_history, dtype=np.float32)
        for i, aid in enumerate(agents):
            pos_hist = pos_array[:, i]
            ax.plot(pos_hist[:, 1], pos_hist[:, 0], color=colors[i])

        # ── Per-agent viewport panels ─────────────────────────────────────────
        max_reward = max((v for v in self._reward_history), default=1) or 1

        for i, aid in enumerate(agents):
            if i >= len(self._ax_viewports):
                break
            ax_vp, ax_bar = self._ax_viewports[i]

            ax_vp.cla()
            vp = self._view_history[-1][i]
            if vp is not None:
                vp_image = self._composite_rgb_map(vp.shape[:2], vp[:, :, 1], vp[:, :, 0], None)
                ax_vp.imshow(vp_image, cmap="YlOrRd", origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax_vp.set_title(aid, color=colors[i], fontsize=8, pad=3)
            ax_vp.set_facecolor("#0d0d1a")
            ax_vp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax_vp.spines.values():
                spine.set_edgecolor(colors[i])
                spine.set_linewidth(1.5)

            ax_bar.cla()
            reward = self._reward_history[-1]
            data = self._reward_history[1:] if len(self._reward_history) >= 1 else self._reward_history
            ax_bar.plot(range(len(data)), data, color=colors[i])
            ax_bar.set_yticks([])
            ax_bar.set_xticks([])
            ax_bar.set_facecolor("#0d0d1a")
            ax_bar.tick_params(colors="gray", labelsize=6)
            ax_bar.set_xlabel("reward", color="gray", fontsize=6)
            ax_bar.set_ylabel("step", color="gray", fontsize=6)
            ax_bar.text(0, max_reward * 1.02, f"{reward:.1f}", va="center", ha="left",
                        color="white", fontsize=7)
            for spine in ax_bar.spines.values():
                spine.set_edgecolor("#333")

        # ── Global viewport panel ─────────────────────────────────────────────
        if self._last_global_vp is not None:
            rgb = self._channels_to_rgb(self._last_global_vp)
            ax_gvp = self._ax_global_vp
            ax_gvp.cla()
            ax_gvp.imshow(rgb, origin="upper", vmin=0.0, vmax=1.0, interpolation="nearest")
            c = self._last_global_vp.shape[0]
            label = "Current Observation" if c <= 3 else f"Current Observation (PCA {c}ch)"
            ax_gvp.set_title(label, color="white", fontsize=9, pad=4)
            ax_gvp.set_facecolor("#0d0d1a")
            ax_gvp.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            legend_lines = {
                1: ["grey=fuel/fire"],
                2: ["G=fuel", "R=fire"],
                3: ["G=fuel", "R=fire", "B=recency"],
            }
            for line_i, txt in enumerate(legend_lines.get(c, ["PCA projected"])):
                ax_gvp.text(2, 4 + line_i * 9, txt, color="white", fontsize=5,
                            bbox=dict(facecolor="#00000088", edgecolor="none", pad=1))
            for spine in ax_gvp.spines.values():
                spine.set_edgecolor("#444")

        # ── Wind info text below figure ───────────────────────────────────────
        wx, wy = self.wind_vector
        angle_deg = math.degrees(math.atan2(wy, wx)) % 360
        wind_label = f"Wind  wx={wx:+.2f}  wy={wy:+.2f}  ({angle_deg:.0f}°)"
        self._fig.text(
            0.5, 0.01, wind_label,
            ha="center", va="bottom",
            color="deepskyblue", fontsize=7,
        )

        # ── Figure title ──────────────────────────────────────────────────────
        self._fig.suptitle(
            f"Episode {self._episode_count} | Step {self._step_count}",
            color="white", fontsize=11, fontweight="bold", y=0.97,
        )

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            if self.is_vid_out and self.out is not None:
                frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                self.out.write(frame)
            return None

        self._fig.canvas.draw()
        if self.is_vid_out and self.out is not None:
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

    # ─────────────────────────────────────────────────────────────────────────
    # Domain metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _update_revisit_tracker(self):
        in_viewport = np.zeros(self.world_size, dtype=bool)
        for pos in self._agent_positions:
            px, py = pos
            half = self.vp_size // 2
            x0 = int(np.clip(px - half, 0, self.world_size[0]))
            x1 = int(np.clip(px + half, 0, self.world_size[0]))
            y0 = int(np.clip(py - half, 0, self.world_size[1]))
            y1 = int(np.clip(py + half, 0, self.world_size[1]))
            in_viewport[x0:x1, y0:y1] = True

        is_fire = self.map[:, :, 1] > 0

        leaving = (self._revisit_state == 1) & ~in_viewport & is_fire
        self._revisit_last_seen[leaving] = self._step_count
        self._revisit_state[leaving] = 2

        revisiting = (self._revisit_state == 2) & in_viewport & is_fire
        if np.any(revisiting):
            last_seen_steps = self._revisit_last_seen[revisiting]
            gaps = self._step_count - last_seen_steps
            valid_gaps = gaps[gaps >= self._revisit_gap_threshold]
            self._revisit_deltas.extend(valid_gaps.tolist())
        self._revisit_state[revisiting] = 1

        first_visit = (self._revisit_state == 0) & in_viewport & is_fire
        self._revisit_state[first_visit] = 1

    def log_coverage_percentiles(self):
        viewed = self.view_acc.get_scene()
        map_ = self.map

        percent_fire_seen = Viewpoint.get_coverage_percentage(1, map_, viewed)
        percent_fuel_seen = Viewpoint.get_coverage_percentage(0, map_, viewed)

        self.fire_coverage_percentages.append((self._step_count, percent_fire_seen))
        self.fuel_coverage_percentages.append((self._step_count, percent_fuel_seen))

        for threshold, key in [(99, "99"), (90, "90"), (75, "75"), (50, "50"), (25, "25")]:
            if (
                percent_fire_seen >= threshold
                and self.fire_coverage_timesteps_dict[key] == self.iter_limit
            ):
                self.fire_coverage_timesteps_dict[key] = self._step_count
                break

        for threshold, key in [(99, "99"), (90, "90"), (75, "75"), (50, "50"), (25, "25")]:
            if (
                percent_fuel_seen >= threshold
                and self.fuel_coverage_timesteps_dict[key] == self.iter_limit
            ):
                self.fuel_coverage_timesteps_dict[key] = self._step_count
                break

    def get_revisit_delta_stats(self) -> dict:
        if not self._revisit_deltas:
            return {
                "domain/revisit_delta_mean": float(self.iter_limit),
                "domain/revisit_delta_min": float(self.iter_limit),
                "domain/revisit_delta_max": float(self.iter_limit),
                "domain/revisit_count": 0,
            }
        arr = np.array(self._revisit_deltas, dtype=np.float32)
        return {
            "domain/revisit_delta_mean": float(np.mean(arr)),
            "domain/revisit_delta_min": float(np.min(arr)),
            "domain/revisit_delta_max": float(np.max(arr)),
            "domain/revisit_count": len(arr),
        }

    def compute_fire_coverage_auc(self) -> float:
        points = self.fire_coverage_percentages
        auc = 0.0
        for i in range(1, len(points)):
            t0, c0 = points[i - 1]
            t1, c1 = points[i]
            dt = (t1 - t0) / self.iter_limit
            auc += 0.5 * (c0 + c1) * dt
        return float(auc)

    def log_metrics(self):
        metrics = {
            "domain/fire_coverage_25": self.get_fire_coverage_timestamps_dict()["25"],
            "domain/fire_coverage_50": self.get_fire_coverage_timestamps_dict()["50"],
            "domain/fire_coverage_75": self.get_fire_coverage_timestamps_dict()["75"],
            "domain/fire_coverage_90": self.get_fire_coverage_timestamps_dict()["90"],
            "domain/fire_coverage_99": self.get_fire_coverage_timestamps_dict()["99"],
            "domain/fire_coverage_AUC": self.compute_fire_coverage_auc(),
        }
        metrics.update(self.get_revisit_delta_stats())
        return metrics

    def get_fire_coverage_timestamps_dict(self):
        return self.fire_coverage_timesteps_dict

    def get_fuel_coverage_timestamps_dict(self):
        return self.fuel_coverage_timesteps_dict