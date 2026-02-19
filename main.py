from agents.Drone import Drone, inject_input
import time
import numpy as np



pos = {}
pos["x"] = 1.0
pos["y"] = 1.0
pos["z"] = 1.0


v = {}
v["x"] = 0.0
v["y"] = 0.0
v["z"] = 0.0


agent = Drone(1, pos)
setpt = np.array([43.0, 21.0, 12.0], dtype=np.float32)
p = 0.4

for i in range(100):
    #if (i+1) % 10 == 0:
    
    curr_pos = agent.get_position_array()
    delta = p*(setpt - curr_pos)

    v["x"] = delta[0]
    v["y"] = delta[1]
    v["z"] = delta[2]

    print(f"[{i}] Delta input : {delta}")

    inject_input(agent, v)
    print(f"[{i}] Agent {agent.get_id()} position : {agent.get_position()['x']}, {agent.get_position()['y']}, {agent.get_position()['z']}")
    time.sleep(0.5)









