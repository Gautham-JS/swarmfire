import random
import numpy as np

from agents.Drone import Drone


# Add poisson disk sampling, grid partitioning and rejection approach
class PointGenerators:
    def __init__(self):
        return
    
    def random_3d_points(self, N, x_range, y_range, z_range):
        return self.random_3d_point_sets(1, N, x_range, y_range, z_range)[0]
    
    def random_3d_point_sets(self, n_sets, points_per_set, x_range, y_range, z_range):
        """
        Generate N sets of random 3D points.

        Parameters:
            n_sets (int): Number of sets to generate
            points_per_set (int): Number of points in each set
            x_range (tuple): (min_x, max_x)
            y_range (tuple): (min_y, max_y)
            z_range (tuple): (min_z, max_z)

        Returns:
            list: A list containing N sets of 3D points
        """
        all_sets = []

        for s in range(n_sets):
            point_set = []
            for _ in range(points_per_set):
                x = random.uniform(*x_range)
                y = random.uniform(*y_range)
                z = random.uniform(*z_range)
                point_set.append((x, y, z))
            all_sets.append(point_set)

        return np.array(all_sets, dtype=np.float32)


class AgentGenerators:
    def __init__(self):
        return
    
    def create_agents(self, N:int, formation_controller = None) -> list:
        agents = []
        seed_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        pos_steps = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        for i in range(N):
            seed_pos = seed_pos + pos_steps
            pos = {}
            pos["x"] = seed_pos[0]
            pos["y"] = seed_pos[1]
            pos["z"] = seed_pos[2]

            agents.append(Drone(i, pos))
        return agents