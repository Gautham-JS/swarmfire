import numpy as np

from utils import Viewpoint

"""
- Map Managers:
    - Maintains map related interfaces.
    - Handles regeneration/distribution of map data
    - Global state hidden unless implementation permits it.
    - Maintains, steps and resets recency map.
"""
class BaseMapManager:
    def __init__(self, world_size, seed=None, is_recency_mode=True):
        self.world_size = world_size
        self.seed = seed
        self.is_recency_mode = is_recency_mode

        # init states
        self._recency_map = None
        self._visited_map = None
        self._fire_disc_map = None
        pass

    def reset(self):
        self._visited_map   = np.zeros(self.world_size, dtype=np.bool_)
        self._recency_map   = np.zeros(self.world_size, dtype=np.float32)
        self._fire_disc_map = np.full(self.world_size, -1, dtype=np.int32)

    def get_viewport(self, agent_position):
        raise NotImplementedError("[ERROR] [BaseMapManager::get_viewport is an abstract method that must be implemented by a child class!]")
    

    def get_recency_map(self):
        return self._recency_map
    
    def get_visited_map(self):
        return self._visited_map
    
    def get_fire_disc_map(self):
        return self._fire_disc_map
    

    
