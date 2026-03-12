from utils import Generators, Viewpoint

import matplotlib.pyplot as plt
import numpy as np
import cv2
import time





size = (512, 512)

fm_gen = Generators.FuelMapGenerator(size)
vp_acc = Viewpoint.IncrementalViewAccumulator(size)
path_gen = Generators.PathGenerator()

points = np.array([(0, 0), (10, 10), (30, 20), (178, 198), (156, 398), (398, 200), (400, 400)])


fm = fm_gen.create_mask(0.001, 0.003)

cv2.imshow("Fuel map", fm)
is_exit = False
view_size = 64


bez_path = path_gen.generate_bezier(fm, points)


for p in bez_path:
    view = Viewpoint.get_square_viewpoint(fm, (p[0], p[1]), size=view_size)
    vp_acc.accumulate(view, (p[0], p[1]), view_size=view_size)

    percieved_scene = vp_acc.get_scene()

    percieved_scene_np = (percieved_scene * 255).astype(np.uint8)

    scene_bgr = np.stack([percieved_scene_np, percieved_scene_np, percieved_scene_np], axis=-1)
    cv2.drawMarker(scene_bgr, (p[1], p[0]), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow("window", (view * 255).astype(np.uint8))
    cv2.imshow("Perception Accumulator", scene_bgr)
    time.sleep(0.08)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        is_exit = True
        break

cv2.destroyAllWindows()
