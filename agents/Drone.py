import math
import numpy as np

from comms.web_sockets.server import WSCommsHandler

# Single agent controller code. Steps position state by specified velocty.
# message definitions: position:dict, velocity:dict
# Keeping orientation optional for now, shall be incorporated into the final controller.

# 1 should be dimensionality independent, states could be 1d/2d/3d
# 2 


class BaseDrone:
    def __init__(self, id:str, pos:dict = None, neighbours:list = None,
                 p_const:float=0.1, i_const:float=0.1, d_const:float=0.,
                 max_speed:float=50.0, damping:float=0.9):
        self.id = id
        self.neighbours = neighbours
        self.p_const = p_const
        self.i_const = i_const
        self.d_const = d_const
        self.max_speed = max_speed  # hard clamp on speed magnitude
        self.damping = damping      # velocity decay per step (0=instant stop, 1=no decay)

        self.step_size = 1

        if pos is None:
            pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.set_position(pos)
        self.set_velocity({"x": 0.0, "y": 0.0, "z": 0.0})

    def set_position(self, position:dict):
        self.pos = position

    def set_velocity(self, velocity:dict):
        self.vel = velocity

    def get_position(self):
        return self.pos
    
    def get_position_delta_from_action(self, action):
        if action == 0:
            return -1 * self.step_size
        elif action == 1:
            return 0
        else:
            return 1 * self.step_size
    
    def get_velocity_from_action(self, action:list[int, int]) -> dict:
        return {
            "x": self.get_position_delta_from_action(action[0]),
            "y": self.get_position_delta_from_action(action[1]),
            "z": 0
        }

    def get_position_array(self):
        return np.array([self.pos["x"], self.pos["y"], self.pos["z"]], dtype=np.float32)

    def get_id(self):
        return self.id

    def step(self, timer=None):
        return NotImplementedError("[ERROR] : BaseDrone's step method should be implemented by a child class")
    
    def initialize(self, timer=None):
        return NotImplementedError("[ERROR] : BaseDrone's initialize method should be implemented by a child class")

    def inject_velocity(self, velocity:dict, accumulate=True) -> dict:
        return NotImplementedError("[ERROR] : BaseDrone's inject_velocity method should be implemented by a child class")

    def inject_action(self, action:list, accumulate=True, metadata:dict = None):
        return NotImplementedError("[ERROR] : BaseDrone's inject_action method should be implemented by a child class")

class Drone(BaseDrone):
    def __init__(self, id:str, pos:dict = None, neighbours:list = None,
                 p_const:float=0.1, i_const:float=0.1, d_const:float=0.,
                 max_speed:float=50.0, damping:float=0.9):
        super().__init__(id, pos, neighbours, p_const, i_const, d_const, max_speed, damping)

    def step(self, timer=None):
        # Apply damping before position update so velocity naturally decays
        self.vel["x"] *= self.damping
        self.vel["y"] *= self.damping
        self.vel["z"] *= self.damping

        self.pos["x"] += self.vel["x"]
        self.pos["y"] += self.vel["y"]
        self.pos["z"] += self.vel["z"]

    def _clamp_velocity(self):
        """Clamp velocity vector to max_speed magnitude, preserving direction."""
        speed = math.sqrt(self.vel["x"]**2 + self.vel["y"]**2 + self.vel["z"]**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vel["x"] *= scale
            self.vel["y"] *= scale
            self.vel["z"] *= scale

    def inject_velocity(self, velocity:dict, accumulate=True):
        if not accumulate:
            self.set_velocity(velocity)
        else:
            self.vel["x"] += velocity["x"]
            self.vel["y"] += velocity["y"]
            self.vel["z"] += velocity["z"]

        self._clamp_velocity()  # clamp after accumulation, before stepping
        self.step()
        return self.get_position()
    
    def inject_action(self, action, accumulate=True, metadata:dict = None):
        velocities = self.get_velocity_from_action(action)
        return self.inject_velocity(velocities, accumulate=accumulate)

    def initialize(self):
        return

class UE5Drone(BaseDrone):
    def __init__(self, id, pos = None, neighbours = None, p_const = 0.1, i_const = 0.1, d_const = 0, max_speed = 50, damping = 0.9):
        super().__init__(id, pos, neighbours, p_const, i_const, d_const, max_speed, damping)
        self._action_msg = None

    def _construct_ws_action_msg(self, dx: int, dy:int, step_id:int, step_idx:int):
        return {
            "type": "action",
            "dx": f"{dx}",
            "dy": f"{dy}",
            "step_id": f"{step_id}",
            "step_idx": f"{step_idx}",
        }

    # dispatch action via websockets
    def step(self, timer=None):
        pass

    def inject_action(self, action: list, accumulate=True, metadata:dict = None):
        msg = self._construct_ws_action_msg(
            dx = action[0],
            dy = action[1],
            step_id = metadata["step_id"],
            step_idx = metadata["step_idx"]
        )
        WSCommsHandler.instance().send_msg(msg)
        res = WSCommsHandler.instance().get_response_blocking(metadata["step_id"])
        



    # construct velocity message and send it out through step
    def inject_velocity(self, velocity: dict, accumulate=True):
        pass



def inject_input(agent:Drone, velocity:dict):
    agent.set_velocity(velocity)
    agent.step()
    return agent.get_position()










