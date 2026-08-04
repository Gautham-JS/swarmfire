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


class WildfireSingleAgentEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 30, 
        "name": "wildfire_agent_v0"
    }

    def __init__(
            self,
            world_size              : tuple[int, int],
            obs_size                : tuple[int, int]   = (84, 84),
            vp_size                 : int               = 64, 
            is_recency_learned      : bool              = True,
            iter_limit              : int               = 4500,
            seed                    : int               = -1,
            seed_increment_interval : int               = 10,
            disable_recency_obs     : bool              = False,
            env_id                  : str               = "SingleAgentEnv",
            render_mode             : str               = "human",
            video_config            : VideoWriterConfig = None,
            phase_weights           : dict              = None,
            device                  : str               = None,
            is_domain_logs_enabled  : bool              = False,
            is_gt_visible           : bool              = False,
            is_seed_incremental     : bool              = False
        ):
        super().__init__()

        self.world_size = world_size
        self.iter_limit = iter_limit
        self.seed = seed
        self.render_mode = render_mode
        self.env_id = env_id
        self.video_config = video_config
        self.device = device
        self.phase_weights = phase_weights
        self.vp_size = vp_size
        self.is_recency_learned = is_recency_learned
        self.is_domain_logs_enabled = is_domain_logs_enabled
        self.is_gt_visible = is_gt_visible
        self.obs_size = obs_size
        self.seed_increment_interval = seed_increment_interval
        self.is_seed_incremental = is_seed_incremental
        self._disable_recency_obs = disable_recency_obs

        if self.is_recency_learned:
            self.n_obs_img_channels = 3
        else:
            self.n_obs_img_channels = 2
        

        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            log.info(f"Auto assigning compute device. Assigned device : {self.device}")
        
        # env state inits
        self._episode_count = 0
        self._step_size = 1
        self._step_count = 0

        self._fire_confidence = 0.0          # EMA of fire fraction in viewport
        self._fire_confidence_alpha = 0.05   # EMA smoothness — tune this
        self._fire_found_threshold = 0.01    # fraction of viewport that counts as "found"

        self._obs_history           :list                       = []
        self._agent_pos_history     :list[tuple[int, int]]      = []
        self._reward_history        :list[float]                = []
        self._view_history          :list[np.ndarray]           = []

        # action and obs space declaration
        self.action_space = gym.spaces.MultiDiscrete([3] * 2)
        self._n_obs_channels = 3 if not self._disable_recency_obs else 2
        self.observation_space = gym.spaces.Dict({
            "viewport": gym.spaces.Box(
                0.0, 
                1.0, 
                shape=(self._n_obs_channels, self.obs_size[0], self.obs_size[1]), 
                dtype=np.float32
            ),
            "positions": gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        })

        self.view_accumulator = Viewpoint.IncrementalViewAccumulator(self.world_size, 2)

        # agent initialized in reset
        self.agent : BaseAgent = None

        # render states
        self._fig = None
        self._axes = None
        self._vid_out_stream = None

        self.set_phase_weights(phase_weights)
    

    def set_phase_weights(self, weights:dict) -> None:
        if weights is None:
            weights = {}
        self._w_exploration             = weights.get("exploration",    1.0)
        self._w_exploration_track       = weights.get("exploration_tracking",    1.0)
        self._w_fire_discovery          = weights.get("fire_discovery", 1.0)
        self._w_fire_tracking           = weights.get("fire_tracking",  1.0)
        self._w_risk                    = weights.get("risk",           1.0)

    def get_position_observation(self, agent: BaseAgent):
        pos_obs = []
        scene = self.view_accumulator.get_scene()
        px, py = agent.get_agent_state().pos_x, agent.get_agent_state().pos_y

        if not agent.get_agent_state().is_pos_normed:
            px, py = px / self.world_size[0], py / self.world_size[1]
        
        pos_obs.extend([px, py])


    def _build_positions_obs(self, agent:BaseAgent) -> np.ndarray:
        state = agent.get_state()
        npx, npy = agent.get_norm_position()
        vx, vy = state.vel_x, state.vel_y
        return np.asarray([npx, npy, vx, vy], dtype=np.float32)
    
    def _update_fire_confidence(self, state: AgentState) -> float:
        """
        EMA of (fire pixels in current viewport / total viewport pixels).
        Rises quickly when fire is visible, decays slowly when it isn't.
        This gives a smooth phase signal rather than a binary switch.
        """
        fire_channel = state.vp_image[:, :, 1]
        fire_fraction = float(np.mean(fire_channel > 0))
        # Use asymmetric alpha: fast rise (found fire), slow decay (lost fire)
        alpha = self._fire_confidence_alpha if fire_fraction > self._fire_confidence else self._fire_confidence_alpha * 0.1
        self._fire_confidence = alpha * fire_fraction + (1 - alpha) * self._fire_confidence
        return self._fire_confidence
    
    def _update_seed(self):
        if self.is_seed_incremental and self.seed is not None:
            if self._episode_count % self.seed_increment_interval == 0:
                log.info("[RESET] : Incrementing generation seed")
                self.seed += 1


    def reset(self, seed=None, options=None):
        # states to maintain history
        self._obs_history.clear()
        self._agent_pos_history.clear()
        self._reward_history.clear()
        self._view_history.clear()

        self.view_accumulator.reset()
        

        # reset global states
        self._step_count = 0
        self._episode_count += 1

        # update seed
        self._update_seed()

        #spawning agent instance
        start_position = (self.world_size[0] // 2, self.world_size[1] // 2)
        self.agent = InProcessAgent("agent_0", self.world_size, start_pos=start_position, seed=self.seed)

        self._agent_pos_history.append(start_position)

        if self.video_config.is_enabled and self._vid_out_stream is not None:
            self._vid_out_stream.release()
            self._vid_out_stream = None
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None


        #build an empty observation | TODO: Actually grab the 0-th valid data for more consistency
        obs = self.agent.reset()
        return obs, {}

    def check_agent_state_object(self, state:AgentState):
        print_var_err = lambda var : log.error(f"[ENV::check_agent_state_object] {var} returned from agent is None! Fatal Error.")

        if state.delta_vp_image is None:
            print_var_err("Delta Image")
        if state.recency_image is None:
            print_var_err("Recency Image")
        if state.vp_image is None:
            print_var_err("Viewport Image")

    
    def _r_viewport_coverage_fraction(self, state:AgentState) -> float:
        """
        Returns the fraction of the viewport that lies within world bounds [0, 1].
        Used to scale down rewards when the agent is near a wall and half its
        viewport is zero-padded — prevents the agent from getting free reward
        for looking at nothing.
        """
        px, py = state.pos_x, state.pos_y
        half = state.vp_size // 2
        total_pixels = state.vp_size * state.vp_size

        x0 = max(px - half, 0);  x1 = min(px + half, self.world_size[0])
        y0 = max(py - half, 0);  y1 = min(py + half, self.world_size[1])
        valid_pixels = (x1 - x0) * (y1 - y0)
        return valid_pixels / total_pixels

    # Rewards motion alignment with a fire perimeter edge. Promotes exploration
    def _r_fire_perimeter_alignment(self, state:AgentState) -> float:
        px, py = state.pos_x, state.pos_y
        half = state.vp_size // 2
        scene = self.view_accumulator.get_scene() # TODO: What if we do not use accumulator at all and rely just on the view

        x0, x1 = int(np.clip(px - half, 0, self.world_size[0])), int(np.clip(px + half, 0, self.world_size[0]))
        y0, y1 = int(np.clip(py - half, 0, self.world_size[1])), int(np.clip(py + half, 0, self.world_size[1]))

        fire_patch = scene[x0:x1, y0:y1, 1]
        if not np.any(fire_patch > 0):
            return 0.0
        
        gx, gy = sobel(fire_patch.astype(np.float32), axis=1), sobel(fire_patch.astype(np.float32), axis=0)
        mag = np.sqrt(gx**2 + gy**2)
        total_mag = mag.sum()
        if total_mag < 1e-6:
            return 0.0
        
        boundary_normal = np.array([
            np.sum(gy * mag) / total_mag,
            np.sum(gx * mag) / total_mag
        ], dtype=np.float64)
        n_boundary_normal = np.linalg.norm(boundary_normal)
        if n_boundary_normal < 1e-6:
            return 0.0

        boundary_normal = boundary_normal / n_boundary_normal
        boundary_tangent = np.array([-boundary_normal[1], boundary_normal[0]])

        velocity_vector = np.array([state.vel_x, state.vel_y], dtype=np.float32)
        velocity_vector_norm = np.linalg.norm(velocity_vector)
        if velocity_vector_norm < 1e-6:
            return 0.0
        
        velocity_dir = velocity_vector / velocity_vector_norm
        alignment = float(np.dot(velocity_dir, boundary_tangent))

        return max(0.0, alignment) * 12.0
    
    def _r_fire_perimeter_alignment_fix(self, state: AgentState) -> float:
        """
        FIXED version:
        1. Uses state.vp_image directly (not the accumulator — that's lagged)
        2. Uses np.hypot instead of np.sqrt(gx**2, gy**2) (that was a bug)
        3. Computes tangent direction from the LOCAL gradient, not global scene
        4. Returns 0 cleanly when no fire edge is visible
        """
        fire_patch = state.vp_image[:, :, 1].astype(np.float32)
        if not np.any(fire_patch > 0):
            return 0.0

        gx = sobel(fire_patch, axis=1)   # horizontal gradient
        gy = sobel(fire_patch, axis=0)   # vertical gradient
        mag = np.hypot(gx, gy)           # BUG FIX: was np.sqrt(gx**2, gy**2)

        total_mag = mag.sum()
        if total_mag < 1e-6:
            return 0.0

        # Gradient points INTO fire — tangent is perpendicular to it
        # Weighted average of gradient direction across the viewport
        boundary_normal = np.array([
            np.sum(gy * mag) / total_mag,
            np.sum(gx * mag) / total_mag,
        ], dtype=np.float32)

        n = np.linalg.norm(boundary_normal)
        if n < 1e-6:
            return 0.0
        boundary_normal /= n

        # Tangent = rotate normal 90 degrees
        boundary_tangent = np.array([-boundary_normal[1], boundary_normal[0]])

        velocity_vector = np.array([state.vel_x, state.vel_y], dtype=np.float32)
        v_norm = np.linalg.norm(velocity_vector)
        if v_norm < 1e-6:
            return 0.0

        velocity_dir = velocity_vector / v_norm
        alignment = float(np.dot(velocity_dir, boundary_tangent))

        # Both tangent directions are valid — abs() rewards tracking either way
        # around the perimeter (clockwise or counterclockwise)
        return abs(alignment) * 15.0
    
    # Gaussian reward for being an 'ideal cells' near the fire
    def _r_fire_proximity(self, state:AgentState, ideal_cell_dist:float = 2.0) -> float:
        scene = self.view_accumulator.get_scene()
        center_x = int(np.clip(state.pos_x, 0, self.world_size[0] - 1))
        center_y = int(np.clip(state.pos_y, 0, self.world_size[1] - 1))

        # no reward if standing over fire. Bad drone!
        if scene[center_x, center_y, 1] > 0:
            return 0.0
        
        half = state.vp_size // 2
        x0 = int(np.clip(state.pos_x - half, 0, self.world_size[0]))
        x1 = int(np.clip(state.pos_x + half, 0, self.world_size[0]))
        y0 = int(np.clip(state.pos_y - half, 0, self.world_size[1]))
        y1 = int(np.clip(state.pos_y + half, 0, self.world_size[1]))

        fire_patch  = scene[x0:x1, y0:y1, 1]
        fire_coords = np.argwhere(fire_patch > 0)
        if len(fire_coords) == 0:
            return 0.0

        local_cx = center_x - x0
        local_cy = center_y - y0
        diffs = fire_coords - np.array([local_cx, local_cy])
        dist = float(np.min(np.linalg.norm(diffs, axis=1)))

        sigma = ideal_cell_dist * 0.8
        return float(3.0 * np.exp(-0.5 * ((dist - ideal_cell_dist) / sigma) ** 2))


    def _r_corner_escape(self, state:AgentState, prev_pos_x, prev_pos_y) -> float:
        min_corner_dist = lambda x, y, corner_list: min(np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corner_list)
        corners = [
            (0, 0),
            (0, self.world_size[1]),
            (self.world_size[0], 0),
            (self.world_size[0], self.world_size[1])
        ]

        prev_dist = min_corner_dist(prev_pos_x, prev_pos_y, corners)
        curr_dist = min_corner_dist(state.pos_x, state.pos_y, corners)
        return max(0.0, curr_dist - prev_dist) * 3.5
    
    # NOTE: INOP | Needs visited map, hidden from env
    def _rh_estimate_occluded_area(self, state:AgentState):
        scene = self.view_accumulator.get_scene()
        n_rays = 16
        max_range = int(max(self.world_size) * 0.75)
        step_size = 2
        occluded = 0
        total_rays = 0

        for angle_idx in range(n_rays):
            angle = 2 * np.pi * angle_idx / n_rays
            dx, dy = np.cos(angle), np.sin(angle)

            is_in_fire = False
            ray_occluded = 0

            for dist in range(step_size, max_range, step_size):
                rx = int(round(state.pos_x + dx * dist))
                ry = int(round(state.pos_y + dy * dist))
                if rx < 0 or rx >= self.world_size[0] or ry < 0 or ry >= self.world_size[1]:
                    break
                cell_is_fire_known = scene[rx, ry, 1] > 0

    # NOTE: INOP | _rh_estimate_occluded_area(...) Needs visited map, hidden from env
    def _r_fire_crossing_opportunity(self, state:AgentState) -> float:
        scene = self.view_accumulator.get_scene()
        center_x = int(np.clip(state.pos_x, 0, self.world_size[0] - 1))
        center_y = int(np.clip(state.pos_y, 0, self.world_size[1] - 1))

        # only requires when in a fire area
        if scene[center_x, center_y, 1] <= 0:
            return 0.0
    
    def _p_fire_crossing(self, state:AgentState) -> float:
        center_x = int(np.clip(state.pos_x, 0, self.world_size[0] - 1))
        center_y = int(np.clip(state.pos_y, 0, self.world_size[1] - 1))

        scene = self.view_accumulator.get_scene()
        fire_intensity = float(scene[center_x, center_y, 1])
        if fire_intensity <= 0:
            return 0.0
        return 10 * (0.5 + 0.5 * fire_intensity)
    
    def _r_movement_bonus(self, state:AgentState) -> float:
        px, py = state.pos_x, state.pos_y
        if len(self._agent_pos_history) >= 2:
            prev  = self._agent_pos_history[-2]
            curr  = self._agent_pos_history[-1]
            moved = abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) > 0.2

            dist_to_nearest_wall = min(px, self.world_size[0] - px, py, self.world_size[1] - py)
            in_margin = dist_to_nearest_wall < self.vp_size // 2

            if moved:
                if in_margin:
                    prev_dist = min(prev[0], self.world_size[0] - prev[0], prev[1], self.world_size[1] - prev[1])
                    curr_dist = min(curr[0], self.world_size[0] - curr[0], curr[1], self.world_size[1] - curr[1])
                    moving_away = curr_dist > prev_dist
                    movement = 2.0 if moving_away else -0.3

                    # Corner escape bonus on top of wall escape
                    movement += self._r_corner_escape(state, prev[0], prev[1])
                else:
                    movement = 0.5
            else:
                movement = -1.5
        else:
            movement = 0.0
        return movement
    
    def _p_near_boundary(self, state:AgentState) -> float:
        margin = state.vp_size // 2
        penality = 0.0
        for coord, lim in [(state.pos_x, self.world_size[0]), (state.pos_y, self.world_size[1])]:
            dist_low = coord
            dist_high = lim - coord
            if dist_low < margin:
                penality += np.exp((margin - dist_low) / (margin / 4)) - 1
            if dist_high < margin:
                penality += np.exp((margin - dist_high) / (margin / 4)) - 1
        
        if penality != 0:
            penality += 55 
        return penality

    def _p_recency(self, state:AgentState) -> float:
        val = float(np.mean(state.recency_image))
        return (math.exp(val) - 1) / (math.exp(1) - 1) * 140
    
    def _p_recency_update(self, state: AgentState, phase_track: float = 0.0) -> float:
        val = float(np.mean(state.recency_image))
        base = (math.exp(val) - 1) / (math.exp(1) - 1) * 140
        # During tracking, the agent *should* stay near the perimeter —
        # reduce recency penalty so it can orbit without being punished
        return base * (1.0 - 0.6 * phase_track)

    def compute_reward(self, state:AgentState) -> float:
        reward = 0.0
        prev_pos_x, prev_pos_y = self._agent_pos_history[-1] if len(self._agent_pos_history) != 0 else (state.pos_x, state.pos_y)

        step_advantage_factor = 2.0 * (1.0 + (
            1.0 - GenericUtils.normalize_data(self._step_count, 0, self.iter_limit)
        ))


        coverage_factor             = self._r_viewport_coverage_fraction(state) ** 2
        fire_perimeter_align_factor = self._r_fire_perimeter_alignment(state)
        fire_proximity_factor       = self._r_fire_proximity(state)
        movement_bonus_factor       = self._r_movement_bonus(state)

        crossing_pen_factor         = self._p_fire_crossing(state)
        recency_pen_factor          = self._p_recency(state)
        near_bound_pen_factor       = self._p_near_boundary(state)

        # TODO: The fires should not be using new fires to measure reward, it should maintain coverage.
        new_fuel_pixels = float(np.sum(state.delta_vp_image[:, :, 0] > 0))
        new_fire_pixels = float(np.sum(state.delta_vp_image[:, :, 1] > 0))
        seen_fuel_pixels = float(np.sum(state.vp_image[:, :, 0] > 0))

        # Core compnents
        exploration_comp            = self._w_exploration * 0.003 * new_fuel_pixels * coverage_factor
        fire_discovery_comp         = self._w_fire_discovery * 0.009 * new_fire_pixels * step_advantage_factor * coverage_factor
        fuel_tracking_comp          = self._w_exploration_track * 0.001 * seen_fuel_pixels * coverage_factor
        perimeter_comp              = self._w_fire_tracking * fire_perimeter_align_factor * coverage_factor
        proximity_comp              = fire_proximity_factor * coverage_factor

        # Penality components
        crossing_pen_comp           = self._w_risk * crossing_pen_factor
        recency_pen_comp            = recency_pen_factor
        near_bound_pen_comp         = near_bound_pen_factor

        # Bonus components
        movement_bonus_comp = movement_bonus_factor

        if self.is_render_step():
            log.info(f"[Calculate reward] : Components at step {self._step_count} : \n\t\
                    EXPLORATION : {exploration_comp} \n\t\
                    FIRE DISC : {fire_discovery_comp} \n\t\
                    FIRE PIXELS : {new_fire_pixels} \n\t\
                    FUEL TRACK : {fuel_tracking_comp} \n\t\
                    PERIMETER : {perimeter_comp} \n\t\
                    PROXIMITY : {proximity_comp} \n\t\
                    RECENCY PEN : {recency_pen_comp} \n\t\
                    NEAR BOUND PEN : {near_bound_pen_comp} \n\t\
                    \n\t\
                    Velocity components : [vx : {state.vel_x} , vy : {state.vel_y}]"
            )

        # accumulate everything
        reward += (
            exploration_comp
            + fire_discovery_comp
            + fuel_tracking_comp
            + perimeter_comp
            + proximity_comp
            + movement_bonus_comp
            - recency_pen_comp
            - near_bound_pen_comp
        )
        return reward
    
    def compute_reward_update(self, state: AgentState) -> float:
        reward = 0.0

        # ── Phase signal ─────────────────────────────────────────────────────
        fire_conf = self._update_fire_confidence(state)
        # Smooth gate: 0.0 = pure exploration, 1.0 = pure tracking
        # Sigmoid-shaped transition around the threshold
        phase_track = float(1 / (1 + np.exp(-40 * (fire_conf - self._fire_found_threshold))))
        phase_explore = 1.0 - phase_track

        coverage_factor = self._r_viewport_coverage_fraction(state) ** 2

        # ── Phase 1: Exploration rewards (suppressed once fire found) ────────
        new_fuel_pixels  = float(np.sum(state.delta_vp_image[:, :, 0]))
        new_fire_pixels  = float(np.sum(state.delta_vp_image[:, :, 1] > 0))
        seen_fuel_pixels = float(np.sum(state.vp_image[:, :, 0]))

        step_advantage = 2.0 * (1.0 + (1.0 - GenericUtils.normalize_data(self._step_count, 0, self.iter_limit)))

        exploration_comp    = self._w_exploration * phase_explore * 0.03 * new_fuel_pixels * coverage_factor
        fire_discovery_comp = self._w_fire_discovery * 0.09 * new_fire_pixels * step_advantage * coverage_factor
        fuel_tracking_comp  = self._w_exploration_track * phase_explore * 0.001 * seen_fuel_pixels * coverage_factor

        # ── Phase 2: Fire tracking rewards (only active once fire found) ─────
        # Note: these use the FIXED local-viewport versions above
        perimeter_align = self._r_fire_perimeter_alignment(state)
        proximity       = self._r_fire_proximity(state)

        perimeter_comp  = self._w_fire_tracking * phase_track * perimeter_align * coverage_factor
        proximity_comp  = phase_track * proximity * coverage_factor

        # ── Movement bonus — phase-aware ─────────────────────────────────────
        # During tracking phase, penalize staying still MORE (agent should orbit)
        movement_bonus = self._r_movement_bonus(state)
        # Small extra reward for actually moving when fire is visible
        if phase_track > 0.5 and (abs(state.vel_x) + abs(state.vel_y)) > 0.1:
            movement_bonus += phase_track * 1.5

        # ── Penalties (always active) ────────────────────────────────────────
        crossing_pen   = self._w_risk * self._p_fire_crossing(state)
        recency_pen    = self._p_recency(state)
        near_bound_pen = self._p_near_boundary(state)

        reward = (
            exploration_comp
            + fire_discovery_comp
            + fuel_tracking_comp
            + perimeter_comp
            + proximity_comp
            + movement_bonus
            - crossing_pen
            - recency_pen
            - near_bound_pen
        )
        return reward

    def _update_env_states(self, state:AgentState, reward:float, observation:dict):
        self._agent_pos_history.append((state.pos_x, state.pos_y))
        self._obs_history.append(observation)
        self._view_history.append(state.vp_image)
        self._reward_history.append(reward)


    def is_truncated(self):
        if self._step_count >= 4000:
            return True
        return False
    
    def is_terminated(self, state:AgentState):
        if state.pos_x <= 0 or state.pos_x > self.world_size[0] - 1:
            log.debug("[ENV] [TERMINATION] : X position exceeded bounds, terminating.")
            return True
        if state.pos_y <= 0 or state.pos_y > self.world_size[1] - 1:
            log.debug("[ENV] [TERMINATION] : Y position exceeded bounds, terminating.")
            return True
        return False
    
    def is_render_step(self):
        return (self._episode_count % self.video_config.sample_interval) == 0

    def _update_and_exec_counter_ops(self):
        self._step_count += 1
        if self.is_render_step():
            self.render()



    def step(self, action):
        reward, terminated, truncated, infos, obs = 0.0, False, False, {}, {}
        if len(action)  != 2:
            log.warning(f"[ENV::STEP] : Action vector's dimensions do not match internally computed values! action length : {len(action)}, expected length : 2")

        
        state:AgentState = self.agent.step(action, self._step_count)              # S2 : Instruct the agent to step once and update internal states
        self.check_agent_state_object(state)                    # S3 : Quick sanity checks on the states

        self.view_accumulator.accumulate(                       # S4 : Accumulate views
            state.vp_image, 
            (state.pos_x, state.pos_y), 
            state.vp_size
        )
        obs = self.agent.get_obs()
        reward = self.compute_reward(state)                     # S5 : use the state to compute rewards
        self._update_env_states(state, reward, obs)                  # S6 : store states for plotting
        truncated = self.is_truncated()                         # S7 : check truncation/termination

        self._update_and_exec_counter_ops()
        return obs, reward, terminated, truncated, infos


    
    
    def _init_figures(self):
        self._fig = plt.figure(
            figsize = (4 + 3 + 3 + 3, 5),
            facecolor="#1a1a2e"
        )
        gs = GridSpec(
            4, 4,
            figure=self._fig,
            hspace=0.4, wspace=0.35,
            left=0.06, right=0.97, top=0.88, bottom=0.08
        )

        self._ax_map                = self._fig.add_subplot(gs[:2, 0])   # GT map
        self._ax_obs                = self._fig.add_subplot(gs[:2, 1])   # Obs as passed to the model
        self._ax_global_vp          = self._fig.add_subplot(gs[:2, 2])   # Accumulated maps
        self._ax_global_recency     = self._fig.add_subplot(gs[:2, 3])   # Accumulated maps with global recency map overlaid

        self._ax_reward_plot        = self._fig.add_subplot(gs[2:, :])

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
            #log.info(f"[RENDER] : Creating video writer object of shape {h}x{w}")
            self._vid_out_stream = cv2.VideoWriter(
                f"{self.video_config.base_path}/{self.env_id}_{self._episode_count}.mp4",
                fourcc,
                self.video_config.fps,
                (w, h),
                isColor=True
            )
        if self.render_mode == "human":
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

    def _draw_wind_arrow(self, ax, wind_vector, map_shape):
        H, W = map_shape

        # Arrow anchor (image coords: x = col, y = row)
        anchor_x = W * 0.88
        anchor_y = H * 0.10

        # Wind vector: wx maps to image +x (col), wy maps to image +y (row)
        wx, wy = wind_vector
        arrow_length = min(H, W) * 0.10   # 10 % of the shorter map dimension

        dx_img = wx * arrow_length   # positive = rightward
        dy_img = wy * arrow_length   # positive = downward (image y)

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
    
    def _draw_agent_view_rectangle(self, ax:plt.Axes, state:AgentState):
        half = state.vp_size // 2
        rect = patches.Rectangle(
            (state.pos_y - half, state.pos_x - half),
            state.vp_size,
            state.vp_size,
            linewidth=1.2,
            edgecolor="green",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.scatter(state.pos_y, state.pos_x, s=60, color="green", linewidths=0.6)
    
    def _create_latest_reward_map(self, state:AgentState):
        recent_reward = self._reward_history[-1] if len(self._reward_history) > 0 else 0.0
        half = state.vp_size // 2
        reward_map = np.zeros()


    
    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
        
        if self._fig is None:
            log.info("[Render] : Initializing figures")
            self._init_figures()
        

        state:AgentState                    = self.agent.get_state()
        wind_vec:tuple[float, float, float] = self.agent.get_wind_vector()

        # render GT map to its axis only if GT view is permitted
        if self.is_gt_visible:    
            gt_map = self.agent.get_GT_map()
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
        self._draw_wind_arrow(self._ax_map, (wind_vec[0], wind_vec[1]), self.world_size)
        for s in self._ax_map.spines.values(): s.set_edgecolor("#444")

        # draw accumulated view
        scene = self.view_accumulator.get_scene()
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
        self._draw_agent_view_rectangle(self._ax_global_vp, state)
        self._draw_agent_trajectory(self._ax_global_vp)
        self._ax_global_vp.set_title("Accumulated Observations")
        self._ax_global_vp.set_facecolor("#0d0d1a")
        self._ax_global_vp.tick_params(colors="gray", labelsize=6)

        # draw agents observation
        # latest_obs:np.ndarray = self._obs_history[-1]["viewport"] if len(self._obs_history) > 0 else {"viewport": np.zeros((84, 84), dtype=np.float32)}
        # latest_obs = np.transpose(latest_obs, (1, 2, 0))
        # view = self._composite_rgb_map(latest_obs.shape[:2], latest_obs[:, :, 1], latest_obs[:, :, 0], latest_obs[:, :, 2])
        log.info(f"[RENDER] : Delta viewpoint shape : {state.delta_vp_image.shape}")
        view = self._composite_rgb_map(state.delta_vp_image.shape[:2], state.delta_vp_image[:, :, 1], state.delta_vp_image[:, :, 0], None)
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
        g_recency = self.agent.get_recency_map()
        self._ax_global_recency.cla()
        self._ax_global_recency.imshow(
            g_recency,
            cmap="jet",
            origin="upper",
            vmin = 0.0,
            vmax = 1.0,
            interpolation="nearest"
        )
        self._draw_agent_view_rectangle(self._ax_global_recency, state)
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

        if self.render_mode == "human":
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            if self.video_config.is_enabled and self._vid_out_stream is not None:
                frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
                #log.info(f"[RENDER]: Frame shape: {frame.shape}")
                self._vid_out_stream.write(frame)
            return None

        self._fig.canvas.draw()
        if self.video_config.is_enabled and self._vid_out_stream is not None:
            frame = cv2.cvtColor(np.asarray(self._fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
            #log.info(f"[RENDER]: Frame shape: {frame.shape}")
            self._vid_out_stream.write(frame)
        return np.asarray(self._fig.canvas.buffer_rgba())[..., :3]


    def close(self):
        if self.video_config.is_enabled and self._vid_out_stream is not None:
            self._vid_out_stream.release()
            self._vid_out_stream = None
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
        return super().close()





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
        state:AgentState = self.agent_instance.step([1, 1], step_id=0)


        # Seed pos_history so _build_positions_obs has something to diff on step 1
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

    # TODO : Recomputing view bounds and velocities here, should be done once and stored in state obj
    def fire_perimeter_alignment_reward(self, pos:tuple) -> float:
        """
        Rewards moving tangentially along the fire boundary as seen in the
        accumulated observation, not the ground truth map.
        """

        scene = self.view_acc.get_scene()
        total = 0.0

        px, py = pos
        half = self.vp_size // 2
        x0 = int(np.clip(px - half, 0, self.world_size[0]))
        x1 = int(np.clip(px + half, 0, self.world_size[0]))
        y0 = int(np.clip(py - half, 0, self.world_size[1]))
        y1 = int(np.clip(py + half, 0, self.world_size[1]))

        # Use accumulated scene instead of self.map
        fire_patch = scene[x0:x1, y0:y1, 1]
        if not np.any(fire_patch > 0):
            return total

        gx  = sobel(fire_patch.astype(np.float32), axis=1)
        gy  = sobel(fire_patch.astype(np.float32), axis=0)
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
        boundary_normal  /= bn_norm
        boundary_tangent  = np.array([-boundary_normal[1], boundary_normal[0]])

        if len(self._positions_history) < 2:
            return total
        prev     = self._positions_history[-2]
        curr     = self._positions_history[-1]
        vel      = np.array([curr[0] - prev[0], curr[1] - prev[1]], dtype=np.float64)
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            return total
        vel_dir   = vel / vel_norm
        alignment = float(np.dot(vel_dir, boundary_tangent))
        total    += max(0.0, alignment) * 12.0

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
        fire_mask = fire_patch > 0
        if not np.any(fire_mask):
            return 0.0

        ts_patch = state.revisit_ts_map
        h = min(fire_patch.shape[0], ts_patch.shape[0])
        w = min(fire_patch.shape[1], ts_patch.shape[1])
        if h == 0 or w == 0:
            return 0.0
        fire_patch = fire_patch[:h, :w]
        ts_patch   = ts_patch[:h, :w]

        fire_mask = fire_patch > 0
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

        local_view = self.view_acc.get_scene()[x0:x1, y0:y1, :]

        vp_scaling_fac = self._viewport_coverage_fraction(px, py) ** 2
        vp_norm_fac = self.vp_size * self.vp_size

        new_fuel_pixels  = float(np.sum(local_view[:, :, 0]))
        new_fire_pixels  = float(np.sum(local_view[:, :, 1]))
        seen_fuel_pixels = float(np.sum(local_view[:, :, 0]))

        rc_exploration      = (self._w_exploration * new_fuel_pixels * vp_scaling_fac) / vp_norm_fac
        rc_fire_disc        = (self._w_fire_discovery * new_fire_pixels * step_advantage * vp_scaling_fac) / vp_norm_fac
        rc_fuel_track       = (self._w_exploration_track * seen_fuel_pixels * vp_scaling_fac) / vp_norm_fac
        rc_perimeter_reward = (self._w_fire_tracking * self.fire_perimeter_alignment_reward((px, py)) * vp_scaling_fac) ** 2 
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

        loc_view = np.zeros((state.vp_size, state.vp_size, 2), dtype=np.float32)
        loc_fuel = Viewpoint.get_square_viewpoint(scene[:, :, 0], (cx, cy), state.vp_size)
        loc_fire = Viewpoint.get_square_viewpoint(scene[:, :, 1], (cx, cy), state.vp_size)
        loc_view[:, :, 0], loc_view[:, :, 1] = loc_fuel, loc_fire

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
    