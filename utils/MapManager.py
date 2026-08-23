from dataclasses import dataclass
from copy import copy

import numpy as np
import cv2
import math

from agents import Drone
from utils import Viewpoint, Generators
from comms.web_sockets.server import WSCommsHandler, decode_observation_image

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
    world_size_x    : int           = -1
    world_size_y    : int           = -1
    
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
    def __init__(self, agent_id: str, world_size:tuple, start_pos:tuple = (0, 0), seed=None, is_eval_mode=False, step_size=1):
        self._agent_id = agent_id
        self._world_size = world_size
        self._start_pos = start_pos
        self.seed = seed
        self._is_eval_mode = is_eval_mode
        self.step_size = step_size

        self._state          : AgentState = AgentState()
        self._prev_state     : AgentState = None

        self._state.pos_x = start_pos[0]
        self._state.pos_y = start_pos[1]
    
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
    def __init__(self, agent_id, world_size, start_pos = (0, 0), seed=None, vp_size=64, is_eval_mode=False, step_size=1):
        super().__init__(agent_id, world_size, start_pos, seed, is_eval_mode=is_eval_mode, step_size=step_size)
        self.map_manager: SimulatedMapManager = SimulatedMapManager(
            self._world_size,
            vp_size,
            self.seed,
            is_eval_mode=is_eval_mode
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
        raw_x, raw_y = self.drone.get_position_array()[0], self.drone.get_position_array()[1]

        is_oob = (raw_x >= self._world_size[0]) or (raw_x < 0) or (raw_y >= self._world_size[1]) or (raw_y < 0)

        px = int(np.clip(raw_x, 0, self._world_size[0] - 1))
        py = int(np.clip(raw_y, 0, self._world_size[1] - 1))

        self._state.vp_size = self.map_manager.get_vp_size()

        if is_oob:
            self.drone.set_position({'x': px, 'y': py, 'z': 0})
        self._state.is_oob = is_oob   # <-- always assign, both True and False

        self._state.pos_x = px
        self._state.pos_y = py

        if self._prev_state is not None:
            self._state.vel_x = (self._state.pos_x - self._prev_state.pos_x) / (self.step_size + 1e-8)
            self._state.vel_y = (self._state.pos_y - self._prev_state.pos_y) / (self.step_size + 1e-8)
        else:
            self._state.vel_x = 0.0
            self._state.vel_y = 0.0

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

    def close(self):
        self.map_manager.close()
        self.map_manager = None


class UE5Agent(BaseAgent):
    """
    NOTE on world-size handling
    ----------------------------
    `self._world_size` (inherited from BaseAgent) is the CANONICAL Python-side
    world size passed in at construction time (i.e. what the rest of the
    training stack — reward shaping, viewport math, recency maps — is built
    around). It must never be overwritten with whatever UE5 reports for its
    own world dimensions. UE5's world size is only ever used as a source
    coordinate space to scale FROM, inside UE5MapManager. See
    UE5MapManager.compute_scaling_fac / get_state_from_response.
    """
    def __init__(self, agent_id, world_size, start_pos = (0, 0), seed=None, vp_size=64, is_eval_mode=False):
        super().__init__(agent_id, world_size, start_pos, seed, is_eval_mode=is_eval_mode)
        self.map_manager: UE5MapManager = UE5MapManager(
            self._world_size,
            vp_size,
            self.seed,
            is_eval_mode=is_eval_mode
        )
        self.map_manager.reset(self.seed)
        self.last_response = None
        self.last_step_id = -1

    def update_agent_postion_state(self):
        response = self.map_manager.get_response(self._state.step_id)
        self._state.pos_x = response["x_pos"]
        self._state.pos_y = response["y_pos"]
        self._state.vel_x = response["x_vel"]
        self._state.vel_y = response["y_vel"]

        self._state.is_oob = False
        self._state.vp_size = self.map_manager.get_vp_size()

    def update_agent_view_state(self, recency_obs:np.ndarray):
        raw_x, raw_y = self._state.pos_x, self._state.pos_y
        is_oob = (raw_x >= self._world_size[0]) or (raw_x < 0) or (raw_y >= self._world_size[1]) or (raw_y < 0)
        self._state.is_oob = is_oob

    def update_state_from_response(self, step_id):
        logging.info("Updating state from response...")
        response = self.map_manager.get_response(step_id)
        logging.info(f"Response : X POS : {response['x_pos']} | Y POS : {response['y_pos']} | W SIZE X : {response['w_shape_x']} | W SIZE Y : {response['w_shape_y']}")
        # NOTE: map_manager scales UE5's raw response into canonical
        # Python world_size coordinates internally. self._world_size is
        # NOT reassigned here — it stays fixed to what this agent was
        # constructed with.
        self._state = self.map_manager.get_state_from_response(response)

    

    def reset(self) -> dict:
        # self.map_manager.reset(self.seed)
        self.map_manager.send_map_action(-1, [1, 1])
        return self.get_obs()

    def get_obs(self) -> dict:
        return {
            "viewport": self._get_obs_viewport_component(),
            "positions": self._get_obs_spatial_component()
        }

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

    def _cv_action_to_ue5_action(self, action):
        """
        Converts a raw discrete MultiDiscrete action [ax, ay] (values in
        {0, 1, 2}, using the same convention as
        BaseAgent._get_position_delta_from_action: 0 -> negative delta,
        1 -> stop, 2 -> positive delta) into the action UE5 expects.

        IMPORTANT: ADroneParent::Step() in the UE5 C++ code switches on
        action.dx / action.dy as LITERAL integers 0/1/2 (dx: 0=MoveBack,
        1=StopX, 2=MoveAhead; dy: 0=MoveLeft, 1=StopY, 2=MoveRight) — it
        does NOT expect signed deltas like -1/0/1. This must forward
        literal {0,1,2} indices, never converted deltas.

        x: UE5's +X ("ahead") lines up directly with Python's +dx (both
           left-origin, no flip established on the position side either)
           -> pass the index through unchanged.

        y: UE5's world Y is NOT flipped on the UE5 side itself (BuildObservation
           reports a raw, unflipped drone_loc - vol_min offset) — the flip is
           applied entirely on the Python inbound side, in
           UE5MapManager.convert_ue5_to_cv_frame. To stay consistent, a
           Python action of "+dy" (move toward larger row index / "down" in
           CV-space) must produce UE5's *negative* y motion, since inbound
           conversion maps larger python_y <-> smaller ue5_y. Because the
           enum is symmetric around 1 (stop), negating an index in {0,1,2}
           is `2 - index` (0<->2 swap, 1 stays 1).
        """
        ax, ay = int(action[0]), int(action[1])
        ue5_ax = ax
        ue5_ay = 2 - ay
        return [ue5_ax, ue5_ay]

    def step(self, action, step_id):
        # Forward the discrete action to UE5 using ITS convention (literal
        # 0/1/2 indices dispatched via ADroneParent::Step()'s switch
        # statement) rather than converting to a signed float delta first —
        # UE5's action handler is itself already discrete, just like
        # Python's, so no delta conversion belongs here at all. Only the y
        # index needs flipping to stay consistent with the position-frame
        # conversion applied on the inbound side.
        ue5_action = self._cv_action_to_ue5_action(action)

        logging.info("Sending map action to UE5")
        self.map_manager.send_map_action(step_id, ue5_action)
        self.update_state_from_response(step_id)
        recency_obs = self.map_manager.get_recency_observation(self._state.pos_x, self._state.pos_y)
        revisit_view = self.map_manager.get_revisit_timestep_view(self._state.pos_x, self._state.pos_y)

        self._state.revisit_ts_map = revisit_view

        self.map_manager.decay_recency_map()
        self.map_manager.mark_recency_map(self._state.pos_x, self._state.pos_y)
        self.map_manager.track_revisit_timestep(self._state.pos_x, self._state.pos_y, step_id)

        return copy(self._state)

    def get_GT_map(self):
        raise NotImplementedError("[ERROR] : GT map hidden when using an external simulator")

    def get_recency_map(self):
        return self.map_manager.get_recency_map()

    def close(self):
        self.map_manager.close()
        self.map_manager = None


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

    def remove_instance(cls, target_class): # <--- Update to accept TWO arguments
        # arg0 (cls) is the Metaclass (MapManagerSingleton)
        # arg1 (target_class) is the class you want to delete (SimulatedMapManager)
        if target_class in cls._instances:
            del cls._instances[target_class]

class BaseMapManager:
    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True, is_eval_mode=False, randomized_scale=True):
        self._world_size = world_size
        self._seed = seed
        self._is_recency_enabled = is_recency_enabled
        self._is_eval_mode = is_eval_mode
        self._vp_size = vp_size
        self._wind_vector: tuple[float, float, float] = None    # [x comp, y comp, magnitude]
        self._is_randomized_scale = randomized_scale

        self._step_id = 0

        # init states
        self._recency_map   = None
        self._visited_map   = None
        self._revisit_timestep_map = None

        self._recency_decay = 0.9995   # per-step decay factor; tunable
        self._recency_visit_bump = 0.01   # value stamped on visit
        pass

    def close(self):
        type(self).remove_instance(type(self))


    def get_recency_map(self):
        return self._recency_map
    
    def is_recency_enabled(self) -> bool:
        return self._is_recency_enabled

    def get_seed(self):
        return self._seed

    def set_vp_size(self, vp_size):
        self._vp_size = vp_size

    def get_vp_size(self):
        return self._vp_size

    def get_world_size(self):
        return self._world_size

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
        x0, x1 = int(np.clip(px - half, 0, self._world_size[0])), int(np.clip(px + half, 0, self._world_size[0]))
        y0, y1 = int(np.clip(py - half, 0, self._world_size[1])), int(np.clip(py + half, 0, self._world_size[1]))
        
        return self._revisit_timestep_map[x0:x1, y0:y1].copy()
    
    def get_recency_observation(self, px, py):
        recency_crop = Viewpoint.get_square_viewpoint(self._recency_map, (px, py), self._vp_size)                                  # (H', W')
        return recency_crop

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

    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True, is_eval_mode=False, randomized_scale=True):
        super().__init__(world_size, vp_size, seed, is_recency_enabled, is_eval_mode=is_eval_mode, randomized_scale=randomized_scale)
        
        if self._is_eval_mode:
            self._generator = Generators.FuelMapGeneratorEval(world_size)
        else:
            self._generator = Generators.FuelMapGenerator(world_size)

        self._map = None
        self._obs_map = dict()
        
    
    def reset(self, seed):
        super().reset(seed)
        self._map = None
        self._obs_map = dict()

        if self._is_eval_mode:
            self._map, self._wind_vector = self._generator.create_eval_map(2, 0.001, 0.003, seed=self._seed)
        else:
            self._map, self._wind_vector = self._generator.create_map(0.001, 0.003, seed=self._seed, selection_frac=0.5)

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
    """
    Bridges UE5's coordinate/world space to the CANONICAL Python-side
    world_size that the rest of the training stack (reward shaping,
    recency maps, viewport extraction, SingleAgentEnv.world_size, etc.)
    is built around.

    `self._world_size` is fixed at construction time and is NEVER
    reassigned from UE5 response data. UE5's own reported world shape
    (`w_shape_x` / `w_shape_y`) is only ever used, per-response, as the
    source space to scale FROM via `compute_scaling_fac`.

    Coordinate frame: UE5 uses an origin-bottom-left, y-up frame; the rest
    of this codebase (Viewpoint crops, recency maps, reward shaping, image
    rendering) assumes an OpenCV-style origin-top-left, y-down frame.

    Inbound (UE5 -> Python): every position/velocity pulled from a UE5
    response is converted via `convert_ue5_to_cv_frame` before it touches
    any Python-side state — x passes through unchanged (both frames are
    left-origin), y is flipped about the UE5 world height and its velocity
    sign inverted.

    Outbound (Python -> UE5): the model was trained purely against the
    Python sim's discrete action semantics (0/1/2 -> negative/stop/positive,
    see BaseAgent._get_position_delta_from_action). ADroneParent::Step() on
    the UE5 side is ALSO discrete and switches on the literal 0/1/2 index
    directly (dx: 0=MoveBack, 1=StopX, 2=MoveAhead; dy: 0=MoveLeft, 1=StopY,
    2=MoveRight) — so no delta conversion happens on the outbound path at
    all, only an index re-mapping. UE5Agent._cv_action_to_ue5_action passes
    the x index through unchanged and flips the y index via `2 - ay`
    (0<->2 swap) to stay consistent with the position-frame flip applied on
    the inbound side, since the UE5 C++ side itself reports position as a
    raw, unflipped offset from the volume's min corner.
    """
    def __init__(self, world_size, vp_size, seed=None, is_recency_enabled=True, is_eval_mode=False, randomized_scale=True):
        super().__init__(world_size, vp_size, seed, is_recency_enabled, is_eval_mode=is_eval_mode, randomized_scale=randomized_scale)
        self.comms_handler : WSCommsHandler = WSCommsHandler.instance()
        self.last_step_id = -1
        self.last_response = None
        self._re_reset = False

    def reset(self, seed):
        # self.comms_handler.send_msg(
        #     {"type": "reset", "seed": seed}
        # )
        # response = self.get_response(-1)
        # self.parse_response(response)
        return super().reset(seed)

    def is_response_ready(self):
        return (self.last_response != None)

    def get_world_size(self):
        return self._world_size

    def compute_scaling_fac(self, ue5_w_size) -> tuple[float, float]:
        """
        Returns (scale_x, scale_y) that map a coordinate/velocity expressed
        in UE5's world space into the canonical Python world_size space:

            python_coord = ue5_coord * scale

        ue5_w_size: (w_shape_x, w_shape_y) as reported by the UE5 response.
        """
        ue5_w, ue5_h = ue5_w_size
        if ue5_w <= 0 or ue5_h <= 0:
            raise ValueError(f"[ERROR] : Invalid UE5 world shape received: {ue5_w_size}")

        scale_x = self._world_size[0] / ue5_w
        scale_y = self._world_size[1] / ue5_h
        return scale_x, scale_y

    def _compute_capture_footprint_canonical(self, ue5_w, ue5_h, fov_deg=60.0, elevation=2800.0):
        """
        Ground footprint of a nadir (straight-down) camera capture, converted
        into canonical world-cell units.
        Assumes:
        - Camera points straight down (no tilt)
        - Render target capture is square, so horizontal FOV == vertical FOV
            (per SceneCaptureComponent2D behavior: horizontal FOV is fixed,
            vertical derives from the render target aspect ratio, which is 1
            for a square capture)
        - elevation is in the same UE units as world_size / scale computation
        """
        footprint_ue_units = 2.0 * elevation * math.tan(math.radians(fov_deg) / 2.0)
        scale_x, _ = self.compute_scaling_fac((ue5_w, ue5_h))
        return footprint_ue_units * scale_x

    def convert_ue5_to_cv_frame(self, x_ue5, y_ue5, vx_ue5, vy_ue5, ue5_w, ue5_h):
        """
        Converts a position + velocity from UE5's coordinate frame into the
        OpenCV-style frame the rest of the Python stack (Viewpoint, recency
        maps, reward shaping, etc.) is built around, and scales it into
        canonical world_size units in the same pass.

        UE5 frame here : origin bottom-left, x right, y UP.
        Python/CV frame : origin top-left,    x right, y DOWN.

        x has the same handedness in both frames (left origin either way),
        so only x is scaled. y must be flipped about the UE5 world height
        *before* scaling, since flip and scale don't commute if you did it
        the other way around with a different world_size). Velocity y must
        flip sign for the same reason — "moving up" in UE5 is "moving up
        the image" i.e. decreasing row index in CV space.

        Returns: (px, py, vx, vy) in canonical, unscaled-to-int python units.
        """
        scale_x, scale_y = self.compute_scaling_fac((ue5_w, ue5_h))

        flipped_y_ue5 = ue5_h - y_ue5

        px = x_ue5 * scale_x
        py = flipped_y_ue5 * scale_y

        vx = vx_ue5 * scale_x
        vy = -vy_ue5 * scale_y

        return px, py, vx, vy

    def get_state_from_response(self, response):
        state = AgentState()

        ue5_w = float(response["w_shape_x"])
        ue5_h = float(response["w_shape_y"])

        footprint_canonical = self._compute_capture_footprint_canonical(ue5_w, ue5_h)
        # Converts UE5's bottom-left-origin, y-up frame into the OpenCV-style
        # top-left-origin, y-down frame the rest of the stack expects, and
        # scales into canonical world_size units in the same step.
        scaled_x, scaled_y, scaled_vx, scaled_vy = self.convert_ue5_to_cv_frame(
            float(response["x_pos"]), float(response["y_pos"]),
            float(response["x_vel"]), float(response["y_vel"]),
            ue5_w, ue5_h
        )

        # Position is rounded (not truncated) before casting to int, since
        # downstream utils (Viewpoint, recency/visited maps, etc.) all index
        # with integer cell coordinates.
        state.pos_x = int(np.clip(round(scaled_x), 0, self._world_size[0] - 1))
        state.pos_y = int(np.clip(round(scaled_y), 0, self._world_size[1] - 1))

        state.vel_x = scaled_vx
        state.vel_y = scaled_vy

        state.step_id = response["step_id"]

        # vp_size stays the canonical int this manager was constructed with —
        # never derived from UE5 data / image shape.
        state.vp_size = self._vp_size

        # Report the CANONICAL world size back on the state, not UE5's raw
        # w_shape_x/w_shape_y, so anything reading state.world_size_x/y
        # downstream stays consistent with self._world_size.
        state.world_size_x = self._world_size[0]
        state.world_size_y = self._world_size[1]

        # OOB check against canonical bounds, x paired with world_size[0]
        # and y paired with world_size[1] (consistent with the rest of the
        # codebase, e.g. InProcessAgent.update_agent_postion_state).
        state.is_oob = (
            state.pos_x < 0 or state.pos_x >= self._world_size[0] or
            state.pos_y < 0 or state.pos_y >= self._world_size[1]
        )

        image_data = decode_observation_image(response)
        view_agent = self.compose_response_image_layers(image_data)

        raw_px = view_agent.shape[0]  # assumes square raw capture
        corrected_px = int(round(raw_px * (self._vp_size / footprint_canonical)))

        if corrected_px <= raw_px:
            # Raw capture covers MORE canonical cells than vp_size needs (camera
            # footprint > vp_size) — center-crop down to just the corrected_px
            # region that corresponds to exactly vp_size canonical cells.
            offset = (raw_px - corrected_px) // 2
            view_agent = view_agent[offset:offset + corrected_px, offset:offset + corrected_px, :]
        else:
            # Raw capture covers FEWER canonical cells than vp_size needs
            # (footprint_canonical < vp_size) — pad with zeros so the result still
            # represents exactly vp_size canonical cells, with the physically
            # uncaptured border reported as empty (no fuel/fire data).
            pad = corrected_px - raw_px
            pad_before = pad // 2
            pad_after = pad - pad_before
            view_agent = np.pad(
                view_agent,
                ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
                mode="constant", constant_values=0.0
            )

        # ONLY remaining resize: corrected_px (== exactly vp_size canonical cells,
        # at whatever raw pixel density) -> vp_size pixels. This changes pixel
        # count only, not the represented footprint, since the crop/pad above
        # already fixed that.
        state.vp_image = cv2.resize(view_agent, (state.vp_size, state.vp_size), interpolation=cv2.INTER_NEAREST)
        state.recency_image = self.get_recency_view(state.pos_x, state.pos_y)
        state.revisit_ts_map = self.get_revisit_timestep_view(state.pos_x, state.pos_y)

        # IMPORTANT: self._world_size and self._vp_size are NOT mutated here.
        # They remain fixed to the canonical values this manager was built
        # with, so _recency_map / _visited_map / _revisit_timestep_map
        # (sized in reset()) stay consistent with every index used above.

        if not self._re_reset:
            self.reset(self._seed)
            self._re_reset = True

        return state
     

    def get_response(self, step_id):
        if step_id == self.last_step_id and self.last_response is not None:
            return self.last_response
        response = self.comms_handler.get_response_with_retries(step_id)
        self.last_step_id = step_id
        self.last_response = response
        return response

    def fast_min_max_inplace(self, arr: np.ndarray) -> None:
        """
        Normalizes the array in-place to save memory and improve speed.
        Modifies 'arr' directly.
        """
        a_min = arr.min()
        a_max = arr.max()
        diff = a_max - a_min

        if diff == 0:
            arr.fill(0)
            return

        # Step 1: Subtract min from every element (in-place)
        arr -= a_min

        # Step 2: Divide by the range (in-place)
        arr /= diff

    def compose_response_image_layers(self, decoded_image):
        """
        Empirically determined via 90°-rotation testing: the raw UE5 camera
        framebuffer is related to our world-array convention (axis0=pos_x,
        axis1=pos_y) by a full 180° rotation — i.e. BOTH raw axes need to be
        reversed, and no transpose/axis-swap is needed at all. (A transpose
        would only be required if the raw image's row/col axes were swapped
        relative to world x/y; the fact that a pure 90°+90°=180° rotation
        resolves it confirms axis0/axis1 already correspond correctly to
        pos_x/pos_y — they're just both reversed.)
        """
        w = 0.7
        high_risk_fuels   = decoded_image[:, :, 2] # B
        low_risk_fuels    = decoded_image[:, :, 1] # G
        fires             = decoded_image[:, :, 0] # R

        # 180° rotation = flip both axes, no transpose
        high_risk_fuels = high_risk_fuels[::-1, ::-1]
        low_risk_fuels  = low_risk_fuels[::-1, ::-1]
        fires           = fires[::-1, ::-1]

        alive = low_risk_fuels.astype(np.float32) / 255.0
        dead  = high_risk_fuels.astype(np.float32) / 255.0

        fires = fires.astype(np.float32) / 255.0
        if fires.max() < 0.05:
            fires = np.zeros_like(fires, dtype=np.float32)
        else:
            self.fast_min_max_inplace(fires)
            
        fuel = w * alive + (1.0 - w) * dead
        return np.stack([fuel, fires], axis=-1)  # (H, W, 2)


    def get_map_observation(self, state:AgentState):
        response = self.get_response(state.step_id)
        image_data = decode_observation_image(response["image_b64"])
        view_agent = self.compose_response_image_layers(image_data)  # (H, W, 2): [fuel, fire]

        recency_view = self.get_recency_view(state.pos_x, state.pos_y)

        # NOTE: view_agent is (H, W, 2) -- channel-last -- so slicing must
        # be view_agent[:, :, k], not view_agent[k] (which would index rows,
        # not channels).
        observation = np.stack([
            cv2.resize(view_agent[:, :, 0], (84, 84), interpolation=cv2.INTER_AREA),  # fuel
            cv2.resize(view_agent[:, :, 1], (84, 84), interpolation=cv2.INTER_AREA),  # fire
            recency_view
        ])
        return observation

    def dump_debug_observation_frame(self, view_agent: np.ndarray, step_id: int, out_dir: str):
        """
        Writes view_agent (H, W, 2) as a viewable 3-channel PNG for debugging.
        cv2.imwrite silently fails (no exception, empty/corrupt file) on a
        raw 2-channel array, so pad to 3 channels first. fuel -> green,
        fire -> red (OpenCV is BGR, so fire goes in index 2).
        """
        debug_img = np.zeros((*view_agent.shape[:2], 3), dtype=np.float32)
        debug_img[:, :, 1] = view_agent[:, :, 0]   # fuel -> green
        debug_img[:, :, 2] = view_agent[:, :, 1]   # fire -> red

        path = f"{out_dir}/debug_obs_{step_id}.png"
        ok = cv2.imwrite(path, (debug_img * 255).astype(np.uint8))
        if not ok:
            logging.warning(f"[DEBUG DUMP FAILED] : {path}")

    def send_map_action(self, step_id, action):
        """
        action: [ax, ay] LITERAL discrete indices in {0, 1, 2}, already
        y-flipped into UE5's convention by
        UE5Agent._cv_action_to_ue5_action. ADroneParent::Step() on the UE5
        side switches on these as literal ints (0=negative/back/left,
        1=stop, 2=positive/ahead/right) — do not send signed float deltas
        here, UE5 does not interpret them that way.
        """
        logging.info("Send map action : Enter")
        msg = {
            "type": "action",
            "step_id": f"{step_id}",
            "dx": f"{action[0]}",
            "dy": f"{action[1]}",
            "step_idx": 0
        }
        self.comms_handler.send_msg(msg)
        logging.info("Send map action : Exit")

    def get_vp_size(self):
        # Canonical vp_size — do not derive this from the UE5 response.
        return self._vp_size
