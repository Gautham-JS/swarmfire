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
    
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import dilation, disk


    def generate_wind_field(self, shape, magnitude_mean=1.0, magnitude_std=0.3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        angle = np.random.uniform(0, 2*np.pi)
        magnitude = max(0.1, np.random.normal(magnitude_mean, magnitude_std))

        wx = magnitude * np.cos(angle)
        wy = magnitude * np.sin(angle)

        return wx, wy


    def generate_fire_field_clustered(
        self,
        shape,
        num_regions=3,
        region_scale=150,
        scale_min=30,
        scale_max=120,
        seed=None
    ):
        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        # region clustering
        region_field = gaussian_filter(np.random.rand(H, W), sigma=region_scale)
        region_field = (region_field - region_field.min()) / (region_field.max() - region_field.min())

        thresholds = np.linspace(0.6, 0.9, num_regions)

        field = np.zeros((H, W), dtype=np.float32)

        for t in thresholds:
            region = region_field > t

            if not region.any():
                continue

            noise = np.random.randn(H, W).astype(np.float32)
            sigma = np.random.uniform(scale_min, scale_max)

            local = gaussian_filter(noise, sigma=sigma)
            local = (local - local.min()) / (local.max() - local.min())

            field += local * region

        # normalize
        field = (field - field.min()) / (field.max() - field.min())

        return field


    def generate_fire_perimeter_timeseries(
            self,
            shape,
            timesteps=5,
            fronts_per_step=2,
            width_mean=3,
            width_std=1,
            growth_rate=0.02,
            wind_strength=0.5,
            edge_sigma=1.0,
            seed=None,
            num_regions=3
        ):

        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        # base field
        field = self.generate_fire_field_clustered(shape, seed=seed, num_regions=num_regions)

        # wind
        wx, wy = self.generate_wind_field(shape, seed=seed)

        # coordinate grid (for wind bias)
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        x_norm = (x - W/2) / W
        y_norm = (y - H/2) / H

        wind_bias = wx * x_norm + wy * y_norm

        # store results
        masks = []

        # initial levels (fire fronts)
        base_levels = np.random.uniform(0.3, 0.7, size=fronts_per_step)

        for t in range(timesteps):

            mask_lines = np.zeros((H, W), dtype=bool)

            # time-dependent level shift (expansion)
            time_shift = t * growth_rate

            # wind pushes fire faster in direction
            effective_field = field + wind_strength * wind_bias

            for lvl in base_levels:

                # expansion outward/inward
                level = lvl + time_shift

                band = np.abs(effective_field - level) < 0.01

                # thickness
                width = max(1, int(np.random.normal(width_mean, width_std)))
                if width > 1:
                    band = dilation(band, footprint=disk(width))

                mask_lines |= band

            # smooth edges slightly
            if edge_sigma > 0:
                smooth = gaussian_filter(mask_lines.astype(float), sigma=edge_sigma)
                mask_lines = smooth > 0.3

            masks.append(mask_lines.astype(np.uint8))

        return masks, (wx, wy)

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
    
    def create_map(self, canopy_density_alive, canopy_density_dead, canopy_size_mean=8, merge_radius=3, seed=None):
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
            seed=seed
        )

        world_map = np.zeros((self.size[0], self.size[0], 2), dtype=np.float32)

        fire_masks, wind_vectors = self.generate_fire_perimeter_timeseries(self.size, 1, width_mean=2, fronts_per_step=10, edge_sigma=0.5, growth_rate=0.03, wind_strength=1.0, seed=seed, num_regions=3)
        fire_mask = fire_masks[0]
        
        w = 0.7
        forest_fuel_map = (w * tree_mask_base) + ((1-w) * tree_mask_dead)
        world_map[:, :, 0] = forest_fuel_map
        world_map[:, :, 1] = fire_mask


        return world_map
    

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