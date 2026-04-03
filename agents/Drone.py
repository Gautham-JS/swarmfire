import math
import numpy as np


# Single agent controller code. Steps position state by specified velocty.
# message definitions: position:dict, velocity:dict
# Keeping orientation optional for now, shall be incorporated into the final controller.

# 1 should be dimensionality independent, states could be 1d/2d/3d
# 2 



class Drone:
    def __init__(self, id:str, pos:dict = None, neighbours:list = None,
                 p_const:float=0.1, i_const:float=0.1, d_const:float=0.,
                 max_speed:float=20.0, damping:float=0.85):
        self.id = id
        self.neighbours = neighbours
        self.p_const = p_const
        self.i_const = i_const
        self.d_const = d_const
        self.max_speed = max_speed  # hard clamp on speed magnitude
        self.damping = damping      # velocity decay per step (0=instant stop, 1=no decay)

        if pos is None:
            pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.set_position(pos)
        self.set_velocity({"x": 0.0, "y": 0.0, "z": 0.0})

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
    
    def initialize(self):
        return

    def set_position(self, position:dict):
        self.pos = position

    def set_velocity(self, velocity:dict):
        self.vel = velocity

    def get_position(self):
        return self.pos

    def get_position_array(self):
        return np.array([self.pos["x"], self.pos["y"], self.pos["z"]], dtype=np.float32)

    def get_id(self):
        return self.id

class Drone2:
    def __init__(self, id:str, pos:dict = None, neighbours:list = None, p_const:float=0.1, i_const:float=0.1, d_const:float=0.):
        self.id = id
        self.neighbours = neighbours

        self.p_const = p_const
        self.i_const = i_const
        self.d_const = d_const
        
        if pos is None:
            pos = {}
            pos["x"] = 0.0
            pos["y"] = 0.0
            pos["z"] = 0.0

        self.set_position(pos)

        v = {}
        v["x"] = 0.0
        v["y"] = 0.0
        v["z"] = 0.0

        self.set_velocity(v)

    

    # executes and steps states, 
    #   timer:  optional arg that ensures 
    #           one click is the same across agents
    #           independent of processing 
    #           & network overhead.
    def step(self, timer=None):
        self.pos["x"] += self.vel["x"]
        self.pos["y"] += self.vel["y"]
        self.pos["z"] += self.vel["z"]
        return
    
    def inject_velocity(self, velocity:dict, acummulate=True):
        if not acummulate:
            self.set_velocity(velocity)
        else:
            self.vel['x'] += velocity['x']
            self.vel['y'] += velocity['y']
            self.vel['z'] += velocity['z']
        
        self.step()
        return self.get_position()


def inject_input(agent:Drone, velocity:dict):
    agent.set_velocity(velocity)
    agent.step()
    return agent.get_position()










