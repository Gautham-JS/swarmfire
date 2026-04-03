import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from utils import Generators, Viewpoint
from agents import Drone, DroneController





class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, size:tuple, start_pos:tuple):
        super().__init__()
        self.size = size
        self.vp_size = 64
        
        self.agent_position = np.array(start_pos, dtype=np.float32)
        self.map_generator = Generators.FuelMapGenerator(self.size)

        self.map = self.create_random_map()
        
        self.action_space = gym.spaces.Dict({
            "move":     gym.spaces.MultiDiscrete([3, 3]),  # delta X, delta Y
        })

        self.observation_space = gym.spaces.Dict({
            "viewport": gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.vp_size, self.vp_size),
                dtype=np.float32
            ),
            "position": gym.spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array([self.size[1] - 1, self.size[0] - 1], dtype=np.int32),
                dtype=np.int32
            ),
        })

        self.agent_map = {}  # agent_id : Drone


    def create_random_map(self, seed=None):
        return self.map_generator.create_mask(0.001, 0.003, seed=seed)
    
    def extract_viewpoint(self, x, y, z):
        view = Viewpoint.get_square_viewpoint(self.map, (x, y), size=self.vp_size)
        return view
    

    def step(self, actions):
        """
        actions: {agent_id: np.array([dx_code, dy_code])}
        Codes: 
            0 :-1 
            1 : 0
            2 : +1
        """

        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        for agent_id, action in actions.items():
            dx, dy = action[0], action[1]
            agent_pos = self.agent_map[agent_id].get_position_array()

            nx = int(np.clip(agent_pos[0] + dx, 0, self.size[1] - 1))
            ny = int(np.clip(agent_pos[1] + dy, 0, self.size[0] - 1))

            view = self.extract_viewpoint(nx, ny, 0)

            obs[agent_id] = {
                "viewport": view,
                "position": np.array([nx, ny], dtype=np.int32),
            }

            rewards[agent_id] = float(np.count_nonzero(view))
            terminated[agent_id] = False
            truncated[agent_id]  = False

        return obs, rewards, terminated, truncated, infos
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.map = self.create_random_map(seed=seed)

        obs = {}
        for aid, a in self.agent_map.items():
            position = [0, 0, 0]
            obs[aid] = {
                "viewport": self.extract_viewpoint(*position),
                "position": np.asarray(position[:2], np.int32)
            }
        return obs, {}





    




