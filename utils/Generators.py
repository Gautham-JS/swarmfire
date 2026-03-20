import random
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import dilation, erosion, disk

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
    

class FuelMapGenerator:
    def __init__(self, size):
        self.size = size

    def generate_tree_mask_fastest(
            self,
            shape,
            canopy_density=0.2,
            canopy_size_mean=5,
            canopy_size_std=2,
            edge_noise_strength=0.3,
            edge_noise_scale=3,   # gotta keep small, blows up 
            merge_radius=3,
            seed=None
        ):
        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        # --- Step 1: generate centers ---
        centers = np.random.rand(H, W) < canopy_density

        if not centers.any():
            return np.zeros((H, W), dtype=np.uint8)

        # --- Step 2: distance field ---
        dist = distance_transform_edt(~centers).astype(np.float32)

        # --- Step 3: spatially varying canopy size ---
        # small smoothing only
        size_noise = gaussian_filter(
            np.random.randn(H, W).astype(np.float32),
            sigma=5
        )

        size_noise = (size_noise - size_noise.min()) / (size_noise.max() - size_noise.min())

        radius_map = canopy_size_mean + canopy_size_std * (size_noise - 0.5)
        radius_map = np.clip(radius_map, 2, None)

        # --- Step 4: field ---
        field = radius_map - dist

        # --- Step 5: edge irregularity ---
        edge_noise = gaussian_filter(
            np.random.randn(H, W).astype(np.float32),
            sigma=edge_noise_scale
        )

        edge_noise = (edge_noise - edge_noise.min()) / (edge_noise.max() - edge_noise.min())

        field += edge_noise_strength * (edge_noise - 0.5) * canopy_size_mean

        # --- Step 6: threshold ---
        mask = field > 0

        # --- Step 7: merging (replaces large sigma smoothing!) ---
        if merge_radius > 0:
            footprint = disk(merge_radius)
            mask = dilation(mask, footprint=footprint)
            mask = erosion(mask, footprint=footprint)

        return mask.astype(np.uint8)
    
    def create_mask(self, canopy_density_alive, canopy_density_dead, canopy_size_mean=8, merge_radius=3, seed=None):
        tree_mask_base = self.generate_tree_mask_fastest(
            self.size,
            canopy_density=canopy_density_alive,
            canopy_size_mean=8,
            merge_radius=3,      
            edge_noise_scale=2,
            seed=seed
        )


        tree_mask_dead = self.generate_tree_mask_fastest(
            self.size,
            canopy_density=canopy_density_dead,
            canopy_size_mean=8,
            merge_radius=3,      
            edge_noise_scale=2,
            seed=10
        )
        
        w = 0.7
        forest_fuel_map = (w * tree_mask_base) + ((1-w) * tree_mask_dead)
        return forest_fuel_map
    

from scipy.special import comb

def bezier_curve(control_points: np.ndarray, num_samples: int = 500) -> np.ndarray:
    """Evaluate a Bezier curve given control points."""
    n = len(control_points) - 1
    t = np.linspace(0, 1, num_samples)
    curve = np.zeros((num_samples, 2))
    for i, cp in enumerate(control_points):
        bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        curve += bernstein[:, None] * cp
    return curve


def generate_bezier_path(
    array: np.ndarray,
    points: np.ndarray,
    samples_per_segment: int = 100,
    control_offset_scale: float = 0.1
) -> np.ndarray:
    """
    Generate a smooth random Bezier path through a set of 2D points.

    Args:
        array:                2D numpy array of shape (H, W), used for bounds
        points:               Array of shape (N, 2) with (row, col) waypoints
        samples_per_segment:  Number of sampled points between each pair of waypoints
        control_offset_scale: Scale of random control point offsets, as fraction of
                              the larger dimension (higher = more curved)

    Returns:
        Array of shape (M, 2) of (row, col) points forming a continuous path
    """
    H, W = array.shape
    N = len(points)
    assert N >= 2, "Need at least 2 points"

    max_offset = control_offset_scale * max(H, W)
    path_segments = []

    for i in range(N - 1):
        p0 = points[i].astype(float)
        p3 = points[i + 1].astype(float)

        # Random cubic bezier control points between p0 and p3
        def random_ctrl(a, b):
            mid = (a + b) / 2
            offset = np.random.uniform(-max_offset, max_offset, size=2)
            pt = mid + offset
            # Clamp to array bounds
            pt[0] = np.clip(pt[0], 0, H - 1)
            pt[1] = np.clip(pt[1], 0, W - 1)
            return pt

        p1 = random_ctrl(p0, p3)
        p2 = random_ctrl(p0, p3)

        segment = bezier_curve([p0, p1, p2, p3], num_samples=samples_per_segment)
        # Avoid duplicating the junction point between segments
        if i > 0:
            segment = segment[1:]
        path_segments.append(segment)

    path = np.concatenate(path_segments, axis=0)
    return np.round(path).astype(int)

class PathGenerator:
    def __init__(self):
        pass

    def generate_bezier(self, layer, points):
        return generate_bezier_path(layer, points)