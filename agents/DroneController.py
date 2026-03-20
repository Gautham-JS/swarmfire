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
    

class AgentController:
    def __init__(self, agent_id, p:float=1.0, i:float=1.0, d:float=1.0):
        self.p = p
        self.i = i
        self.d = d
        self.agent_id = agent_id
        self.threshold = 0.005

        self.setpoint = np.zeros((1, 3), dtype=np.float32)
        self.is_velocity_mode = False


    def setpoint_position(self, position):
        # Nx3
        self.setpoint = np.array(position, dtype=np.float32)

    def setpoint_velocity(self, velocity):
        self.is_velocity_mode = True
        self.setpoint = velocity

    def get_setpoint(self, ):
        return self.setpoint

    def control(self, agent):
        # Nx3
        
        curr_pos = agent.get_position_array()
        delta = self.p * (self.setpoint - curr_pos)

        velocities = {
            'x': delta[0],
            'y': delta[1],
            'z': delta[2]
        }
        
        converged = delta[0] < self.threshold and delta[1] < self.threshold and delta[2] < self.threshold

        return velocities, converged
        

class AgentVelocityController:
    def __init__(self, agent_id, p:float=0.1, i:float=0.1, d:float=0.1):
        self.p = p
        self.i = i
        self.d = d
        self.agent_id = agent_id
        self.threshold = 0.005

        self.setpoint = np.zeros((1, 3), dtype=np.float32)
        self.curr_vel = np.zeros((1, 3), dtype=np.float32)

    def setpoint_velocity(self, velocity):
        self.is_velocity_mode = True
        self.setpoint = velocity

    def get_setpoint(self, ):
        return self.setpoint

    def control(self, agent):
        # Nx3

        next_vel = self.curr_vel +  (self.p * (self.setpoint - self.curr_vel))

        velocities = {
            'x': next_vel[0],
            'y': next_vel[1],
            'z': next_vel[2]
        }
        self.curr_vel = next_vel
        converged = velocities['x'] < self.threshold and velocities['y'] < self.threshold and velocities['z'] < self.threshold
        return velocities, converged
        


    
class DroneActionController:
    def __init__(self, p=0.1, i=0.1, d=0.1):
        self.p = p
        self.i = i
        self.d = d

        self.prev_pos = None

    def execute_action_vector(pos, action_vector):
        pass





