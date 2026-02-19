import numpy as np

from agents.Drone import Drone, inject_input


class DroneController:
    def __init__(self, p:float=1.0, i:float=1.0, d:float=1.0):
        self.p = p
        self.i = i
        self.d = d
        
        self.setpoint = np.zeros((1, 3), dtype=np.float32)

    def set_positions(self, positions):
        # Nx3
        self.setpoint = np.array(positions, dtype=np.float32)

    def get_positions(self, ):
        return self.setpoint

    def control(self, agents:list):
        # Nx3
        pos_list = np.array([a.get_position_array() for a in agents], dtype=np.float32)
        deltas = self.p * (self.setpoint - pos_list)

        for a, dv in zip(agents, deltas):
            vel = {
                'x': dv[0], 
                'y':dv[1], 
                'z':dv[2]
            }
            inject_input(a, vel)

        return agents



