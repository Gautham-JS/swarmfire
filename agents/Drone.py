import math
import numpy as np


# Single agent controller code. Steps position state by specified velocty.
# message definitions: position:dict, velocity:dict
# Keeping orientation optional for now, shall be incorporated into the final controller.

# 1 should be dimensionality independent, states could be 1d/2d/3d
# 2 



class Drone:
    def __init__(self, id:str, pos:dict = None, neighbours:list = None, p_const:float=1.0, i_const:float=1.0, d_const:float=1.0):
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


def inject_input(agent:Drone, velocity:dict):
    agent.set_velocity(velocity)
    agent.step()
    return agent.get_position()










