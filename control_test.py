from agents.Drone import Drone, inject_input
from agents.DroneController import DroneController

from utils.Generators import PointGenerators, AgentGenerators
from clients.RedisClient import RedisClient

import time
import numpy as np


rclient = RedisClient()

point_gen = PointGenerators()
agent_gen = AgentGenerators()
controller = DroneController(p=0.6)






N = 30
min_lim = -200
max_lim = 200
clear = False



agents = agent_gen.create_agents(N)
set_pts = point_gen.random_3d_points(N, (min_lim, max_lim), (min_lim, max_lim), (min_lim, max_lim))
print(set_pts)


def agents_position_to_array(agents):
    poss = []
    for agent in agents:
        poss.append(agent.get_position_array())
    return np.array(poss, dtype=np.float32)


controller.set_positions(set_pts)
for i in range(50):
    controller.control(agents)
    positions = agents_position_to_array(agents)
    rclient.set_numpy("positions", positions)
    rclient.set_numpy("targets", set_pts)
    
    print(f"Iteration [{i}]")
    # for agent, spt in zip(agents, set_pts):
    #     print(f"\tAgent {agent.get_id()} position : [{agent.get_position()['x']:.2f}, {agent.get_position()['y']:.2f}, {agent.get_position()['z']:.2f}] | Target : [{spt[0]:.2f}, {spt[1]:.2f}, {spt[2]:.2f}]")

    time.sleep(0.5)

if clear:
    rclient.clear_keys(["positions", "targets"])


