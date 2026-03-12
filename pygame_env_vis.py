import pygame
import numpy as np
import matplotlib.cm as cm
import cv2

from utils import Generators, Viewpoint


size = (1024, 1024)
vp_size = 64

fm_gen = Generators.FuelMapGenerator(size)
point_gen = Generators.PointGenerators()
vp_acc = Viewpoint.IncrementalViewAccumulator(size)
path_gen = Generators.PathGenerator()

points = np.array([(0, 0), (178, 198), (156, 398), (398, 200), (400, 400)])

fm = fm_gen.create_mask(0.001, 0.003)
bez_path = path_gen.generate_bezier(fm, points)

class GlobalViewer:
    def __init__(self, update_obj, sz):
        self.update_obj = update_obj
        self.sz = sz
        pygame.init()
        self.display = pygame.display.set_mode(self.sz)
    
    def set_title(self, title):
        pygame.display.set_caption(title)
    
    def start(self):
        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
            
            z = self.update_obj.update_fn()
            surf = pygame.surfarray.make_surface(z)
            self.display.blit(surf, (0, 0))
            
            pygame.display.update()
        pygame.quit()


class LocalFetcher:
    def __init__(self, point_map:dict):
        self.point_map = point_map
        self.counter_map = {}
        for a, p in point_map.items():
            self.counter_map[a] = 0
    
    def fetch(self, agent_id):
        return self.point_map[agent_id]
    
    def get_agents(self):
        return self.point_map.keys()



class SwarmUpdater:
    def __init__(self, sz, fetcher):
        self.size = sz
        self.accumulator = Viewpoint.IncrementalViewAccumulator(self.size)
        self.fetcher = fetcher

        self.local_counter_map = {}

        self.counter = 0
        self.last = pygame.time.get_ticks()
        self.vp_size = 64
        self.cooldown = 100

        self.check_and_reset_all_counters()

    def fetch_agent_points(self, agent_id):
        return self.fetcher.fetch(agent_id)
    
    def update_local_view(self, agent_id):
        points = self.fetch_agent_points(agent_id)
        
        idx = self.local_counter_map[agent_id]
        pos = (int(points[idx][0]), int(points[idx][1]))
        view = Viewpoint.get_square_viewpoint(fm, pos, size=vp_size)
        
        return view, pos
    
    def check_and_reset_all_counters(self):
        for agent_id in self.fetcher.get_agents():
            if agent_id not in self.local_counter_map:
                self.local_counter_map[agent_id] = 0
                continue
            points = self.fetch_agent_points(agent_id)
            if self.local_counter_map[agent_id] >= len(points):
                self.local_counter_map[agent_id] = 0
                self.accumulator.reset()
    
    def increment_all_counters(self):
        for agent_id in self.fetcher.get_agents():
            self.local_counter_map[agent_id] += 1
    
    def draw_all_agent_information(self):
        view_acc = self.accumulator.get_scene()
        lfm = cv2.applyColorMap((view_acc * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        for agent_id in self.fetcher.get_agents():
            points = self.fetch_agent_points(agent_id)
            idx = self.local_counter_map[agent_id]
            pos = (int(points[idx][0]), int(points[idx][1]))
            hist_pos = points[:idx]

            view_bound_coords = Viewpoint.get_view_bound_coords(fm, pos[::-1], vp_size)    
            cv2.rectangle(lfm, view_bound_coords[1], view_bound_coords[0], color=(0, 255, 0), thickness=1)
            cv2.drawMarker(lfm, pos[::-1], color=(0, 255, 0), thickness=1)
            cv2.putText(lfm, agent_id, (pos[1] - 20, pos[0]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.5)
            if idx > 1:
                hist = np.array(hist_pos, dtype=np.int32)[:, ::-1].reshape((-1, 1, 2))
                cv2.polylines(lfm, [hist], isClosed=False, color=(0, 255, 255), thickness=2)
        return lfm

    def update_fn(self):
        self.check_and_reset_all_counters()
        agents = self.fetcher.get_agents()
        for agent_id in agents:
            view, pos = self.update_local_view(agent_id)
            self.accumulator.accumulate(view, pos, self.vp_size)

        global_scene = self.draw_all_agent_information()

        now = pygame.time.get_ticks()
        if now - self.last > self.cooldown:
            self.last = now
            self.increment_all_counters()
        return global_scene.astype('uint8').transpose(1, 0, 2)


class LocalUpdater:
    def __init__(self, sz):
        self.size = sz
        self.accumulator = Viewpoint.IncrementalViewAccumulator(self.size)

        self.counter = 0
        self.last = pygame.time.get_ticks()
        self.vp_size = 64
        self.points = ([0, 0])
        self.cooldown = 100
        self.npts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))

    def set_points(self, points):
        self.points = points
        self.npts = np.array([(p[1], p[0]) for p in points], dtype=np.int32).reshape((-1, 1, 2))

    def set_vp_szie(self, vp_size):
        self.vp_size = vp_size

    def update_fn(self):
        if self.counter >= len(self.points):
            self.counter = 0
            self.accumulator.reset()

        idx = self.counter
        pos = (int(self.points[idx][0]), int(self.points[idx][1]))

        now = pygame.time.get_ticks()
        if now - self.last > self.cooldown:
            self.last = now
            self.counter += 1
            
            # Only accumulate on counter advance, not every frame
            view = Viewpoint.get_square_viewpoint(fm, pos, size=vp_size)
            self.accumulator.accumulate(view, pos, self.vp_size)
        
        view_acc = self.accumulator.get_scene()
        
        lfm = cv2.applyColorMap((view_acc * 255).astype(np.uint8), cv2.COLORMAP_BONE)

        view_bound_coords = Viewpoint.get_view_bound_coords(fm, pos[::-1], vp_size)    
        cv2.rectangle(lfm, view_bound_coords[1], view_bound_coords[0], color=(0, 255, 0), thickness=1)
        cv2.drawMarker(lfm, pos[::-1], color=(0, 255, 0), thickness=1)
        cv2.polylines(lfm, [self.npts], isClosed=False, color=(200, 0, 0), thickness=1)
        cv2.polylines(lfm, [self.npts[:idx]], isClosed=False, color=(255, 255, 0), thickness=2)

        return lfm.astype('uint8')
    

updater = LocalUpdater(size)
updater.set_points(bez_path)


N = 5
point_list = point_gen.random_3d_point_sets(N, 5, (0, size[0]), (0, size[1]), (0, 0))
point_list_2d = point_list[:, :, :2]



agent_point_map = {}
for i, p in enumerate(point_list_2d):
    bp = Generators.bezier_curve(p, 100)
    agent_point_map[f"a_{i}"] = bp

local_fetcher = LocalFetcher(agent_point_map)
swarm_updater = SwarmUpdater(size, local_fetcher)


viewer = GlobalViewer(swarm_updater, size)
viewer.set_title("Fuel Detection Simulation")
viewer.start()
