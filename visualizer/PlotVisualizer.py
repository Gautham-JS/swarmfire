import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import redis

min_lim = -200
max_lim = 200

class Animator3D:
    def __init__(
        self,
        redis_host='localhost',
        redis_port=6379,
        pos_key='positions',
        target_key='targets',
        history_len=20
    ):
        # --- Redis ---
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.pos_key = pos_key
        self.target_key = target_key

        self.history_len = history_len

        # --- Figure ---
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        self.ax.set_xlim(min_lim, max_lim)
        self.ax.set_ylim(min_lim, max_lim)
        self.ax.set_zlim(min_lim, max_lim)

        # --- Data ---
        self.points = None
        self.targets = None

        self.history = []
        self.lines = []

        # --- Plots ---
        self.point_scat = self.ax.scatter([], [], [], c='blue', marker='o', label='Positions')
        self.target_scat = self.ax.scatter([], [], [], c='red', marker='^', s=80, label='Targets')

        self.ax.legend()

    # --------------------------
    # Redis fetching
    # --------------------------
    def fetch_array(self, key):
        data = self.r.get(key)
        if data is None:
            print("Data is none")
            return None

        try:
            arr = np.array(json.loads(data))
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
        except Exception:
            pass

        return None

    # --------------------------
    # Initialize plots when data arrives
    # --------------------------
    def initialize_if_needed(self, points, targets):
        if self.points is not None:
            return  # already initialized

        if points is None or targets is None:
            return

        if len(points) != len(targets):
            print("Points and targets size mismatch")
            return

        self.points = points
        self.targets = targets

        n = len(points)

        # Initialize history
        self.history = [[] for _ in range(n)]

        # Initialize path lines
        self.lines = [
            self.ax.plot([], [], [], c='gray', alpha=0.6)[0]
            for _ in range(n)
        ]

        # Set initial scatter
        self.point_scat._offsets3d = (
            points[:, 0], points[:, 1], points[:, 2]
        )

        self.target_scat._offsets3d = (
            targets[:, 0], targets[:, 1], targets[:, 2]
        )

    # --------------------------
    # Update animation
    # --------------------------
    def update(self, frame):
        new_points = self.fetch_array(self.pos_key)
        new_targets = self.fetch_array(self.target_key)

        # Initialize once data is available
        self.initialize_if_needed(new_points, new_targets)

        if self.points is None:
            return self.point_scat,

        # Update data if new data available
        if new_points is not None:
            self.points = new_points

        if new_targets is not None:
            self.targets = new_targets

        # --- Update scatter ---
        self.point_scat._offsets3d = (
            self.points[:, 0],
            self.points[:, 1],
            self.points[:, 2]
        )

        self.target_scat._offsets3d = (
            self.targets[:, 0],
            self.targets[:, 1],
            self.targets[:, 2]
        )

        # --- Update history ---
        for i in range(len(self.points)):
            self.history[i].append(self.points[i].copy())

            if len(self.history[i]) > self.history_len:
                self.history[i].pop(0)

        # --- Update paths ---
        for i, line in enumerate(self.lines):
            hist = np.array(self.history[i])
            if len(hist) > 1:
                line.set_data(hist[:, 0], hist[:, 1])
                line.set_3d_properties(hist[:, 2])

        return [self.point_scat, self.target_scat, *self.lines]

    # --------------------------
    # Run animation
    # --------------------------
    def run(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=50,
            blit=False  # important for 3D
        )
        plt.show()


# Run
Animator3D(history_len=200).run()
