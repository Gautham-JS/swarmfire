from dataclasses import dataclass
from copy import copy

import numpy as np
import cv2

from agents import Drone
from utils import Viewpoint, Generators
from comms.web_sockets.server import WSCommsHandler

import logging
log = logging.getLogger(__name__)

@dataclass
class AgentState:
    pos_x           : float         = 0.0
    pos_y           : float         = 0.0
    vel_x           : float         = 0.0
    vel_y           : float         = 0.0

    step_id         : int           = -1
    world_id        : int           = -1
    vp_size         : int           = 64
    
    vp_image        : np.ndarray    = None
    delta_vp_image  : np.ndarray    = None
    recency_image   : np.ndarray    = None
    revisit_ts_map  : np.ndarray    = None

    is_oob          : bool          = False
    is_pos_normed   : bool          = False

@dataclass
class Action:
    dx              : int           = 0
    dy              : int           = 0
    step_id         : int           = -1
    ep_id           : int           = -1


class BaseAgent:
    def __init__(self, agent_id: str, world_size:tuple, start_pos:tuple = (0, 0), seed=None):
        self._agent_id = agent_id
        self._world_size = world_size
        self._start_pos = start_pos
        self.seed = seed

        self._state          : AgentState = AgentState()
        self._prev_state     : AgentState = None

        self._state.pos_x = start_pos[0]
        self._state.pos_y = start_pos[1]

        self.step_size = 1
    
    def get_state(self):
        return self._state
    
    def get_norm_position(self):
        npx, npy  = self._state.pos_x, self._state.pos_y
        if not self._state.is_pos_normed:
            npx = npx / self._world_size[0]
            npy = npy / self._world_size[1]
        
        return npx, npy

    def _get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size
    
    def step(self, action, step_id: int):
        raise NotImplementedError("[ERROR] : AGENT::step() is an abstract method to be implemented by a child class.")
    
    def get_obs(self):
        raise NotImplementedError("[ERROR] : AGENT::get_obs() is an abstract method to be implemented by a child class.")

    def get_wind_vector(self):
        raise NotImplementedError("[ERROR] : AGENT::get_wind_vector() is an abstract method to be implemented by a child class.")

    def update_agent_postion_state(self):
        raise NotImplementedError("[ERROR] : AGENT::update_agent_postion_state() is an abstract method to be implemented by a child class.")
    
    def update_agent_view_state(self, recency_obs:np.ndarray):
        raise NotImplementedError("[ERROR] : AGENT::update_agent_view_state() is an abstract method to be implemented by a child class.")


    def step(action: list):
        raise NotImplementedError("[ERROR] : AGENT::step() is an abstract method to be implemented by a child class.")
    
    def get_GT_map(self):
        raise NotImplementedError("[ERROR] : AGENT::get_GT_map() is an abstract method to be implemented by a child class.")

    def get_recency_map(self):
        raise NotImplementedError("[ERROR] : AGENT::get_recency_map() is an abstract method to be implemented by a child class.")
    

class InProcessAgent(BaseAgent):
    def __init__(self, agent_id, world_size, start_pos = (0, 0), seed=None, vp_size=64):
        super().__init__(agent_id, world_size, start_pos, seed)
        self.map_manager: SimulatedMapManager = SimulatedMapManager(
            self._world_size,
            vp_size,
            self.seed
        )
        self.map_manager.reset(self.seed)

        self.drone : Drone.Drone = Drone.Drone(agent_id, max_speed=50)
        self.drone.set_position({
            'x': self._state.pos_x, 
            'y': self._state.pos_y, 
            'z': 0
        })
    def reset(self) -> dict:
        recency = self.map_manager.get_recency_observation(self._state.pos_x, self._state.pos_y)
        self.update_agent_postion_state()
        self.update_agent_view_state(recency)
        return self.get_obs()


    def update_agent_postion_state(self):
        px, py = int(self.drone.get_position_array()[0]), int(self.drone.get_position_array()[1])

        if (px >= self._world_size[0]) or (px < 0) or (py >= self._world_size[1]) or (py < 0):
            px = np.clip(px, 0, self._world_size[0] - 1)
            py = np.clip(py, 0, self._world_size[1] - 1)
            self.drone.set_position({'x': px, 'y': py, 'z': 0})  # zero out accumulated velocity if oob
            self._state.is_oob = True

        self._state.pos_x = px
        self._state.pos_y = py

        if self._prev_state is not None:
            self._state.vel_x = (self._state.pos_x - self._prev_state.pos_x) / (self.step_size + 1e-8)
            self._state.vel_y = (self._state.pos_y - self._prev_state.pos_y) / (self.step_size + 1e-8)
        else:
            self._state.vel_x = 0.0
            self._state.vel_y = 0.0

        # Always update _prev_state to current AFTER computing velocity
        self._prev_state = copy(self._state)
    
    def update_agent_view_state(self, recency_obs:np.ndarray, revisit_view: np.ndarray):
        view, deltas = self.map_manager.extract_view_and_deltas_update(self._state)
        self._state.delta_vp_image = deltas
        self._state.vp_image = view
        self._state.recency_image = recency_obs
        self._state.revisit_ts_map = revisit_view


    def step(self, action, step_id: int) -> AgentState:
        self._state.step_id = step_id
        # S1 : Apply action
        dx = self._get_position_delta_from_action(action[0])
        dy = self._get_position_delta_from_action(action[1])
        self.drone.inject_velocity({
            "x" : dx, 
            "y" : dy, 
            "z" : 0
        })

        # S2 : Obtain global states first
        recency_obs = self.map_manager.get_recency_observation(self._state.pos_x, self._state.pos_y)
        revisit_view = self.map_manager.get_revisit_timestep_view(self._state.pos_x, self._state.pos_y)

        # S3 : Update local states
        self.update_agent_postion_state()
        self.update_agent_view_state(recency_obs, revisit_view)
        # S4 : Finally mark recency
        self.map_manager.decay_recency_map()
        self.map_manager.mark_recency_map(self._state.pos_x, self._state.pos_y)
        self.map_manager.track_revisit_timestep(self._state.pos_x, self._state.pos_y, step_count=step_id)

        # S5 : Return state, env buils observation from it
        return copy(self._state)
        
    def get_GT_map(self) -> np.ndarray:
        return self.map_manager.get_GT_map()
    
    def get_wind_vector(self) -> tuple[float, float, float]:
        return self.map_manager.get_wind_vector()
    
    def get_recency_map(self):
        return self.map_manager.get_recency_map()
    
    def _get_obs_viewport_component(self, sz = (84, 84)) -> np.ndarray:
        loc_view = np.zeros((self._state.vp_size, self._state.vp_size, 2), dtype=np.float32)
        loc_view[:, :, 0] = self._state.vp_image[:, :, 0].copy()
        loc_view[:, :, 1] = self._state.vp_image[:, :, 1].copy()
        loc_view_chw = np.transpose(loc_view, (2, 0, 1))

        if self.map_manager.is_recency_enabled():
            recency_obs = self.map_manager.get_recency_observation(self._state.pos_x, self._state.pos_y)
        else:
            recency_obs = np.zeros((self._state.vp_size, self._state.vp_size), dtype=np.float32)
        recency_view = recency_obs.reshape(*recency_obs.shape, 1)
        recency_view_chw = np.transpose(recency_view, (2, 0, 1))
        loc_view_chw = np.concatenate([loc_view_chw, recency_view_chw], axis=0).astype(np.float32)
        
        return np.stack([
            cv2.resize(loc_view_chw[0], (84, 84), interpolation=cv2.INTER_AREA),
            cv2.resize(loc_view_chw[1], (84, 84), interpolation=cv2.INTER_AREA),
            cv2.resize(loc_view_chw[2], (84, 84), interpolation=cv2.INTER_AREA)
        ])
    
    def _get_obs_spatial_component(self):
        return np.asarray(
            [self._state.pos_x, self._state.pos_y, self._state.vel_x, self._state.vel_y],
            dtype=np.float32
        )

    def get_obs(self) -> dict:
        return {
            "viewport": self._get_obs_viewport_component(),
            "positions": self._get_obs_spatial_component()
        }
        

"""
- Map Managers:
    - Maintains map related interfaces.
    - Handles regeneration/distribution of map data
    - Global state hidden unless implementation permits it.
    - Maintains, steps and resets recency map.
"""

class MapManagerSingleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwds) 
        return cls._instances[cls] 

class BaseMapManager:
    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True):
        self._world_size = world_size
        self._seed = seed
        self._is_recency_enabled = is_recency_enabled
        self._vp_size = vp_size
        self._wind_vector: tuple[float, float, float] = None    # [x comp, y comp, magnitude]

        self._step_id = 0

        # init states
        self._recency_map   = None
        self._visited_map   = None
        self._revisit_timestep_map = None

        self._recency_decay = 0.9995   # per-step decay factor; tunable
        self._recency_visit_bump = 0.03   # value stamped on visit
        pass

    def get_recency_map(self):
        return self._recency_map
    
    def is_recency_enabled(self) -> bool:
        return self.is_recency_enabled

    def get_seed(self):
        return self._seed

    def set_vp_size(self, vp_size):
        self._vp_size = vp_size

    def get_vp_size(self):
        return self._vp_size

    def reset(self, seed):
        self._step_id = 0
        self._seed = seed

        self._visited_map   = np.zeros(self._world_size, dtype=np.bool_)
        self._recency_map   = np.zeros(self._world_size, dtype=np.float32)
        self._revisit_timestep_map = np.full(self._world_size, -1, dtype=np.int32)

    def get_map_observation(self, state:AgentState):
        raise NotImplementedError("[ERROR] [BaseMapManager::get_viewport is an abstract method that must be implemented by a child class!]")
    
    def mark_recency(self, norm_pos_x, norm_pos_y, vp_size):
        # un-norm x, y pos
        # mark vp_size X vp_size patch in recency_map
        # returned marked map
        pass

    def mark_recency_map(self, px, py):
        half = self._vp_size // 2
        x0, x1 = np.clip(px - half, 0, self._world_size[0]), np.clip(px + half, 0, self._world_size[0])
        y0, y1 = np.clip(py - half, 0, self._world_size[1]), np.clip(py + half, 0, self._world_size[1])

        self._recency_map[x0:x1, y0:y1] += self._recency_visit_bump
        self._recency_map[x0:x1, y0:y1] = np.minimum(
            self._recency_map[x0:x1, y0:y1], 
            np.ones((x1 - x0, y1 - y0), dtype=np.float32)
        )
        return self._recency_map[x0:x1, y0:y1]
    
    def track_revisit_timestep(self, px, py, step_count:int):
        half = self._vp_size // 2
        x0, x1 = np.clip(px - half, 0, self._world_size[0]), np.clip(px + half, 0, self._world_size[0])
        y0, y1 = np.clip(py - half, 0, self._world_size[1]), np.clip(py + half, 0, self._world_size[1])

        self._revisit_timestep_map[x0:x1, y0:y1] = step_count
        return self._revisit_timestep_map[x0:x1, y0:y1]
    
    def get_recency_view(self, px, py):
        half = self._vp_size // 2
        x0, x1 = np.clip(px - half, 0, self._world_size[0]), np.clip(px + half, 0, self._world_size[0])
        y0, y1 = np.clip(py - half, 0, self._world_size[1]), np.clip(py + half, 0, self._world_size[1])
        
        return self._recency_map[x0:x1, y0:y1].copy()
    
    def get_revisit_timestep_view(self, px, py):
        half = self._vp_size // 2
        x0, x1 = np.clip(px - half, 0, self._world_size[0]), np.clip(px + half, 0, self._world_size[0])
        y0, y1 = np.clip(py - half, 0, self._world_size[1]), np.clip(py + half, 0, self._world_size[1])
        
        return self._revisit_timestep_map[x0:x1, y0:y1].copy()
    
    def get_recency_observation(self, px, py):
        recency_crop = Viewpoint.get_square_viewpoint(self._recency_map, (px, py), self._vp_size)                                  # (H', W')
        return recency_crop

    def get_recency_map(self):
        return self._recency_map
    

    def get_visited_map(self):
        return self._visited_map
    
    def get_revisit_timestep_map(self):
        return self._revisit_timestep_map
    
    def decay_recency_map(self):
        self._recency_map *= self._recency_decay

    
    def extract_view_and_deltas(self, agent_state:AgentState):
        raise NotImplementedError("[ERROR] [BaseMapManager::get_viewport is an abstract method that must be implemented by a child class!]")
    
    def get_wind_vector(self) -> tuple[float, float, float]:
        raise NotImplementedError("[ERROR] [BaseMapManager::get_wind_vector is an abstract method that must be implemented by a child class!]")



class SimulatedMapManager(BaseMapManager, metaclass=MapManagerSingleton):

    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True):
        super().__init__(world_size, vp_size, seed, is_recency_enabled)

        self._generator = Generators.FuelMapGenerator(world_size)

        self._map = None
        self._obs_map = dict()
        
    
    def reset(self, seed):
        super().reset(seed)
        self._map = None
        self._obs_map = dict()

        self._map, self._wind_vector = self._generator.create_map(0.001, 0.003, seed=self._seed)

    def extract_view_and_deltas_update(self, agent_state: AgentState):
        x, y = agent_state.pos_x, agent_state.pos_y
        size = self._vp_size
        H, W = self._map.shape[:2]
        row, col = x, y
        half = size // 2

        src_r0 = max(0, row - half);  src_r1 = min(H, row - half + size)
        src_c0 = max(0, col - half);  src_c1 = min(W, col - half + size)
        dst_r0 = src_r0 - (row - half)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = src_c0 - (col - half)
        dst_c1 = dst_c0 + (src_c1 - src_c0)

        # Compute the delta ONCE before marking — valid pixels not yet visited
        valid_mask = np.zeros((size, size), dtype=np.bool_)
        valid_mask[dst_r0:dst_r1, dst_c0:dst_c1] = True

        already_visited = np.zeros((size, size), dtype=np.bool_)
        already_visited[dst_r0:dst_r1, dst_c0:dst_c1] = \
            self._visited_map[src_r0:src_r1, src_c0:src_c1]

        new_cell_mask = valid_mask & ~already_visited  # shape (size, size)

        # NOW mark visited (once, not twice)
        self._visited_map[src_r0:src_r1, src_c0:src_c1] = True

        # Extract both channels using the same geometry
        fuel_vp = np.zeros((size, size), dtype=np.float32)
        fire_vp = np.zeros((size, size), dtype=np.float32)
        fuel_vp[dst_r0:dst_r1, dst_c0:dst_c1] = self._map[src_r0:src_r1, src_c0:src_c1, 0]
        fire_vp[dst_r0:dst_r1, dst_c0:dst_c1] = self._map[src_r0:src_r1, src_c0:src_c1, 1]

        view = np.zeros((size, size, 2), dtype=np.float32)
        view[:, :, 0] = fuel_vp
        view[:, :, 1] = fire_vp

        # Delta: newly seen pixels for each channel independently
        deltas = np.zeros((size, size, 2), dtype=np.float32)
        deltas[:, :, 0] = fuel_vp * new_cell_mask   # fuel value at newly seen cells
        deltas[:, :, 1] = fire_vp * new_cell_mask   # fire value at newly seen cells
        return view, deltas

    def extract_view_and_deltas(self, agent_state:AgentState):
        x, y = agent_state.pos_x, agent_state.pos_y
        fuel_view, recently_visited_fuel, delta_fuel_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self._map[:, :, 0], self._visited_map, (x, y), size=self._vp_size)
        fire_view, recently_visited_fire , delta_fire_mask = Viewpoint.get_square_viewpoint_and_mark_visited(self._map[:, :, 1], self._visited_map, (x, y), size=self._vp_size)
        
        self._visited_map = recently_visited_fuel.copy()
        
        view, deltas = np.zeros((self._vp_size, self._vp_size, 2), dtype=np.float32), np.zeros((self._vp_size, self._vp_size, 2), dtype=np.float32)
        view[:, :, 0], view[:, :, 1] = fuel_view, fire_view
        deltas[:, :, 0], deltas[:, :, 1] = delta_fuel_mask, delta_fire_mask

        return view, deltas

    def get_map_observation(self, state:AgentState):
        # extract obs around state.agent_pos_x, state.agent_pos_y
        # update recency and return
        px, py = state.pos_x, state.pos_y

        view_fuel = Viewpoint.get_square_viewpoint(self._map[:, :, 0], (px, py), self._vp_size)
        view_fire = Viewpoint.get_square_viewpoint(self._map[:, :, 1], (px, py), self._vp_size)
        view_agent = np.stack([view_fuel, view_fire], axis=0)  # (2, vp_size, vp_size)

        recency_view = self.get_recency_view(state.pos_x, state.pos_y)
    
        observation = np.stack([
            cv2.resize(view_agent[0], (84, 84), interpolation=cv2.INTER_AREA),
            cv2.resize(view_agent[1], (84, 84), interpolation=cv2.INTER_AREA),
            recency_view
        ])
        return observation

    def get_GT_map(self) -> np.ndarray:
        return self._map
    
    def get_wind_vector(self):
        return self._wind_vector
    


class UE5MapManager(BaseMapManager, metaclass=MapManagerSingleton):
    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True):
        super().__init__(world_size, vp_size, seed, is_recency_enabled)
        self.comms_handler : WSCommsHandler = WSCommsHandler.instance()

    def get_map_observation(self, state:AgentState):
        response = self.comms_handler.get_response_with_retries(state.step_id)
        return {
            "step_id" : {state.step_id}
        }
    
    def send_map_action(self, step_idx, action):
        self.comms_handler.send_msg(action)
    
