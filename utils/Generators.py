import random
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import dilation, erosion, disk

from agents.Drone import Drone

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import dilation, erosion, disk
from matplotlib.colors import Normalize


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
    




import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, disk, erosion # Added erosion for consistency in tree mask methods


class FuelMapGenerator:
    def __init__(self, size):
        self.size = size
        # Assuming FireClusterMapGenerator is defined elsewhere and used here
        # self.fire_gen = FireClusterMapGenerator(self.size) 

    from scipy.ndimage import gaussian_filter
    from skimage.morphology import dilation, disk


    def generate_wind_field(self, shape, magnitude_mean=1.0, magnitude_std=0.3, seed=None):
        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        angle = np.random.uniform(0, 2*np.pi)
        magnitude = max(0.1, np.random.normal(magnitude_mean, magnitude_std))

        wx = np.cos(angle)
        wy = np.sin(angle)

        return wx, wy, magnitude


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
        wx, wy, _ = self.generate_wind_field(shape, seed=seed)

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


    # --- NEW FUNCTION FOR DATA AUGMENTATION ---
    def generate_multi_blob_fire_field(
        self,
        shape,
        num_blobs=5,
        avg_radius=20,
        radius_std=10,
        seed=None
    ):
        """
        Generates a single static fire map composed of multiple randomized, overlapping blobs.
        This serves as the alternative (data augmentation) method for fire generation.
        """
        if seed is not None:
            np.random.seed(seed)

        H, W = shape

        # Start with an empty mask
        fire_mask = np.zeros((H, W), dtype=np.uint8)

        for _ in range(num_blobs):
            # Random center coordinates
            y_center = np.random.randint(0, H)
            x_center = np.random.randint(0, W)

            # Random radius/size for this blob
            radius = max(5, np.random.normal(avg_radius, radius_std))
            
            # Create a circular Gaussian hot spot around the center
            y_coords, x_coords = np.ogrid[:H, :W]
            dist_squared = (x_coords - x_center)**2 + (y_coords - y_center)**2
            
            # Use a normalized Gaussian falloff for the intensity
            blob_intensity = np.exp(-np.sqrt(dist_squared) / radius) 

            # Normalize and scale the blob to create a visible hot zone
            blob = (blob_intensity - blob_intensity.min()) / (blob_intensity.max() - blob_intensity.min())
            
            # Apply an initial threshold/smoothing before combining
            if blob.max() > 0:
                blob = gaussian_filter(blob, sigma=2)

            # Accumulate the blobs (using OR operation or summation)
            fire_mask = np.maximum(fire_mask, blob * 1.0)


        # Final thresholding to get a binary mask and normalization if necessary
        final_mask = fire_mask > 0.4 # Threshold determines how faint a spot counts as "on fire"

        return final_mask.astype(np.uint8)
    # --- END NEW FUNCTION ---

    

    def generate_tree_mask_fastest_viz(
        self,
        shape,
        canopy_density=0.2,
        canopy_size_mean=5,
        canopy_size_std=2,
        edge_noise_strength=0.3,
        edge_noise_scale=3,
        merge_radius=3,
        seed=None,
        visualize=True
    ):
        """
        Generate a tree canopy mask with optional visualization of each step.

        Parameters:
            visualize (bool): Whether to generate visualizations (images/heatmaps).
        """
        if seed is not None:
            np.random.seed(seed)
        H, W = shape

        # --- Step 1: Generate tree centers (binary mask)
        centers = np.random.rand(H, W) < canopy_density
        if not centers.any() and visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(centers, cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title("Step 1: Tree Centers (Binary Mask)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 2: Distance field (Euclidean distance to nearest center)
        dist = distance_transform_edt(~centers).astype(np.float32)

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(dist, cmap='viridis', aspect='auto')
            plt.title("Step 2: Distance Field (to Tree Centers)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 3: Spatially varying canopy size (radius map)
        size_noise = gaussian_filter(
            np.random.randn(H, W).astype(np.float32),
            sigma=5
        )
        size_noise = (size_noise - size_noise.min()) / (size_noise.max() - size_noise.min())

        radius_map = canopy_size_mean + canopy_size_std * (size_noise - 0.5)
        radius_map = np.clip(radius_map, 2, None)

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(radius_map, cmap='viridis', aspect='auto')
            plt.title("Step 3: Canopy Size Map (Spatially Varying)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 4: Field = radius_map - distance
        field = radius_map - dist

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(field, cmap='coolwarm', aspect='auto')
            plt.title("Step 4: Canopy Field (Radius - Distance)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 5: Edge irregularity (add noise)
        edge_noise = gaussian_filter(
            np.random.randn(H, W).astype(np.float32),
            sigma=edge_noise_scale
        )
        edge_noise = (edge_noise - edge_noise.min()) / (edge_noise.max() - edge_noise.min())

        field += edge_noise_strength * (edge_noise - 0.5) * canopy_size_mean

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(field, cmap='inferno', aspect='auto')
            plt.title("Step 5: Field with Edge Noise")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 6: Threshold to binary mask
        mask = field > 0

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title("Step 6: Binary Tree Mask (Thresholded Field)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        # --- Step 7: Morphological smoothing (dilation/erosion)
        if merge_radius > 0:
            footprint = disk(merge_radius)
            mask = dilation(mask, footprint=footprint)
            mask = erosion(mask, footprint=footprint)

        if visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1, aspect='auto')
            plt.title("Step 7: Final Tree Mask (Morphological Smoothing)")
            plt.colorbar()
            plt.tight_layout()
            plt.show()

        return mask.astype(np.uint8)

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
    
    def create_map(self, canopy_density_alive, canopy_density_dead, canopy_size_mean=8, merge_radius=3, seed=None, selection_frac=0.5):
        # Determine which fire generation method to use (random switch for augmentation)
        use_blob_generation = np.random.rand() < selection_frac # 50% chance of using the new blob generator

        if use_blob_generation:
            #print("--- Using Multi-Blob Fire Generation (Augmentation Mode) ---")
            fire_mask = self.generate_multi_blob_fire_field(self.size, num_blobs=8, avg_radius=25, radius_std=15, seed=seed)
        else:
            #print("--- Using Time-Series Fire Generation (Original Mode) ---")
            # Original fire generation method
            fire_masks, _ = self.generate_fire_perimeter_timeseries(self.size, 
                                                                    timesteps=1, fronts_per_step=30, edge_sigma=0.5, growth_rate=0.03, wind_strength=1.0, seed=seed, num_regions=3)
            fire_mask = fire_masks[0]


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

        # wind field generation remains the same
        wind_field = self.generate_wind_field(self.size, seed=seed)

        w = 0.7
        forest_fuel_map = (w * tree_mask_base) + ((1-w) * tree_mask_dead)
        world_map[:, :, 0] = forest_fuel_map
        world_map[:, :, 1] = fire_mask


        return world_map, wind_field
    

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
    



import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from noise import pnoise2   # pip install noise


class FireClusterMapGenerator:
    """
    Generates maps with spread-out fire clusters in varied shapes:
      - Noisy rings / annuli
      - Irregular blobs (Perlin-masked ellipses)
      - Arc segments (partial rings)
      - Streak clusters (wind-driven appearance)
    
    Each map has 2 channels: [:,:,0] = fuel, [:,:,1] = fire.
    Fuel is generated independently via Perlin noise.
    """

    def __init__(self, world_size: tuple[int, int]):
        self.H, self.W = world_size

    def create_map(self, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # map_out = np.zeros((self.H, self.W, 2), dtype=np.float32)
        # # map_out[:, :, 0] = self._generate_fuel(seed)
        # map_out[:, :, 1] = self._generate_fire_clusters(rng)

        # # Fire only burns where there is fuel
        # map_out[:, :, 1] *= (map_out[:, :, 0] > 0.2).astype(np.float32)

        return self._generate_fire_clusters(rng)

    # ── Fuel ──────────────────────────────────────────────────────────────────

    def _generate_fuel(self, seed) -> np.ndarray:
        """Smooth Perlin fuel map, values in [0, 1]."""
        seed = seed or 0
        scale = 0.003
        fuel = np.array([
            [pnoise2(x * scale + seed, y * scale + seed, octaves=4)
             for y in range(self.W)]
            for x in range(self.H)
        ], dtype=np.float32)
        # Normalise to [0, 1]
        fuel = (fuel - fuel.min()) / (fuel.max() - fuel.min() + 1e-8)
        return fuel

    # ── Fire cluster orchestration ────────────────────────────────────────────

    def _generate_fire_clusters(self, rng: np.random.Generator) -> np.ndarray:
        fire = np.zeros((self.H, self.W), dtype=np.float32)

        n_clusters = int(rng.integers(8, 16))

        # Spread cluster centres across the map using a simple repulsion grid
        centres = self._sample_spread_centres(rng, n_clusters, min_sep=180)

        for cx, cy in centres:
            style = rng.choice(["ring", "arc", "blob", "streak"], p=[0.35, 0.25, 0.25, 0.15])

            if style == "ring":
                patch = self._make_noisy_ring(rng, cx, cy)
            elif style == "arc":
                patch = self._make_arc(rng, cx, cy)
            elif style == "blob":
                patch = self._make_blob(rng, cx, cy)
            else:
                patch = self._make_streak(rng, cx, cy)

            fire = np.maximum(fire, patch)

        # Light global Perlin mask so fire intensity varies spatially
        perlin_mask = self._perlin_mask(rng)
        fire = fire * (0.6 + 0.4 * perlin_mask)

        # Threshold and smooth edges
        fire = (fire > 0.3).astype(np.float32)
        fire = gaussian_filter(fire, sigma=1.5)
        fire = np.clip(fire / (fire.max() + 1e-8), 0, 1)
        fire = (fire > 0.4).astype(np.float32)

        return fire

    # ── Centre sampling ───────────────────────────────────────────────────────

    def _sample_spread_centres(self, rng, n, min_sep):
        """
        Sample n centres with a minimum separation using rejection sampling.
        Falls back gracefully if it can't place all n points.
        """
        margin  = 150
        centres = []
        max_attempts = 500

        for _ in range(n):
            for _ in range(max_attempts):
                x = int(rng.integers(margin, self.H - margin))
                y = int(rng.integers(margin, self.W - margin))
                if all(np.sqrt((x - cx)**2 + (y - cy)**2) >= min_sep
                       for cx, cy in centres):
                    centres.append((x, y))
                    break

        return centres

    # ── Ring ─────────────────────────────────────────────────────────────────

    def _make_noisy_ring(self, rng, cx, cy) -> np.ndarray:
        """
        Annulus with Perlin-noise radius perturbation — looks like a
        fire front that has propagated outward from an ignition point.
        """
        radius     = float(rng.integers(20, 80))
        thickness  = float(rng.integers(6, 35))
        noise_amp  = radius * rng.uniform(0.15, 0.45)  # how ragged the ring is
        noise_freq = rng.uniform(0.03, 0.08)
        seed_off   = float(rng.integers(0, 10000))

        ys, xs = np.ogrid[:self.H, :self.W]
        dx = xs - cy
        dy = ys - cx
        angles = np.arctan2(dy, dx)           # (H, W)
        dist   = np.sqrt(dx**2 + dy**2)      # (H, W)

        # Perturb the target radius by a smooth noise function of angle
        # Use vectorised sin/cos series as a cheap angle-domain noise proxy
        n_harmonics = 6
        perturb = np.zeros_like(angles)
        for k in range(1, n_harmonics + 1):
            phase = float(rng.uniform(0, 2 * np.pi))
            amp   = noise_amp / k
            perturb += amp * np.sin(k * angles + phase)

        target_radius = radius + perturb       # (H, W) — varies per angle
        ring = np.abs(dist - target_radius) < (thickness / 2)

        out = np.zeros((self.H, self.W), dtype=np.float32)
        out[ring] = 1.0
        return out

    # ── Arc ───────────────────────────────────────────────────────────────────

    def _make_arc(self, rng, cx, cy) -> np.ndarray:
        """Partial ring — like a fire that has spread in one wind direction."""
        radius     = float(rng.integers(50, 140))
        thickness  = float(rng.integers(10, 28))
        arc_start  = float(rng.uniform(0, 2 * np.pi))
        arc_span   = float(rng.uniform(np.pi * 0.4, np.pi * 1.4))
        noise_amp  = radius * rng.uniform(0.1, 0.3)

        ys, xs     = np.ogrid[:self.H, :self.W]
        dx = xs - cy
        dy = ys - cx
        angles = np.arctan2(dy, dx) % (2 * np.pi)
        dist   = np.sqrt(dx**2 + dy**2)

        # Noisy radius, same harmonic approach as ring
        n_harmonics = 4
        perturb = np.zeros_like(angles)
        for k in range(1, n_harmonics + 1):
            phase    = float(rng.uniform(0, 2 * np.pi))
            perturb += (noise_amp / k) * np.sin(k * angles + phase)

        target_radius = radius + perturb
        in_band       = np.abs(dist - target_radius) < (thickness / 2)

        # Angular mask for the arc segment
        arc_end = (arc_start + arc_span) % (2 * np.pi)
        if arc_end > arc_start:
            in_arc = (angles >= arc_start) & (angles <= arc_end)
        else:
            in_arc = (angles >= arc_start) | (angles <= arc_end)

        out = np.zeros((self.H, self.W), dtype=np.float32)
        out[in_band & in_arc] = 1.0
        return out

    # ── Blob ─────────────────────────────────────────────────────────────────

    def _make_blob(self, rng, cx, cy) -> np.ndarray:
        """
        Irregular elliptical blob masked by Perlin noise —
        looks like a ground fire spreading through patchy fuel.
        """
        rx = float(rng.integers(30, 100))
        ry = float(rng.integers(30, 100))
        angle = float(rng.uniform(0, np.pi))

        ys, xs = np.ogrid[:self.H, :self.W]
        dx = xs - cy
        dy = ys - cx

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rx_rot =  cos_a * dx + sin_a * dy
        ry_rot = -sin_a * dx + cos_a * dy
        ellipse = ((rx_rot / rx)**2 + (ry_rot / ry)**2) <= 1.0

        # Perlin mask to break up the ellipse interior
        perlin = self._perlin_mask(rng, freq=0.015)
        threshold = rng.uniform(0.3, 0.55)

        out = np.zeros((self.H, self.W), dtype=np.float32)
        out[ellipse & (perlin > threshold)] = 1.0
        return out

    # ── Streak ────────────────────────────────────────────────────────────────

    def _make_streak(self, rng, cx, cy) -> np.ndarray:
        """
        Wind-driven elongated fire — a rotated, tapered ellipse
        with noisy edges, wider at the origin and narrowing downwind.
        """
        length    = float(rng.integers(80, 220))
        width     = float(rng.integers(15,  45))
        direction = float(rng.uniform(0, 2 * np.pi))

        ys, xs = np.ogrid[:self.H, :self.W]
        dx = xs - cy
        dy = ys - cx

        cos_d, sin_d = np.cos(direction), np.sin(direction)
        along  =  cos_d * dx + sin_d * dy   # distance along wind direction
        across = -sin_d * dx + cos_d * dy   # distance across wind direction

        # Taper: width narrows linearly from base to tip
        taper_ratio = np.clip(1.0 - along / (length + 1e-8), 0.1, 1.0)
        in_streak   = (
            (along >= 0) & (along <= length) &
            (np.abs(across) <= width * taper_ratio)
        )

        # Noisy boundary
        noise = self._perlin_mask(rng, freq=0.02)
        out   = np.zeros((self.H, self.W), dtype=np.float32)
        out[in_streak] = 1.0
        out = out * (0.5 + 0.5 * noise)
        out = (out > 0.4).astype(np.float32)
        return out

    # ── Perlin mask helper ────────────────────────────────────────────────────

    def _perlin_mask(self, rng, freq=0.008) -> np.ndarray:
        """Returns a (H, W) Perlin noise array normalised to [0, 1]."""
        seed_off = float(rng.integers(0, 100000))
        arr = np.array([
            [pnoise2((x + seed_off) * freq, (y + seed_off) * freq, octaves=3)
             for y in range(self.W)]
            for x in range(self.H)
        ], dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        return arr






from scipy.ndimage import gaussian_filter


class FuelMapGeneratorEval:

    def __init__(self, size):
        self.size = size

    # -------------------------------------------------------------------------
    # Deterministic fire blob
    # -------------------------------------------------------------------------

    def generate_fixed_blob(
        self,
        shape,
        center,
        radius=25,
        edge_sigma=2.0,
        threshold=0.4
    ):
        """
        Generate a single deterministic circular fire blob.

        Parameters
        ----------
        shape : tuple
            (H, W)

        center : tuple
            (y, x)

        radius : float
            Radius of the fire blob.

        edge_sigma : float
            Gaussian smoothing applied to the blob.

        threshold : float
            Threshold used to obtain the final binary fire mask.
        """

        H, W = shape
        cy, cx = center

        y, x = np.ogrid[:H, :W]

        distance = np.sqrt(
            (x - cx) ** 2 +
            (y - cy) ** 2
        )

        # Smooth radial blob
        blob = np.exp(
            -(distance ** 2) /
            (2 * radius ** 2)
        )

        if edge_sigma > 0:
            blob = gaussian_filter(blob, sigma=edge_sigma)

        blob = blob / (blob.max() + 1e-8)

        return (blob > threshold).astype(np.uint8)

    # -------------------------------------------------------------------------
    # Generate polygon vertices
    # -------------------------------------------------------------------------

    def generate_polygon_centers(
        self,
        shape,
        num_vertices,
        side_length,
        center=None,
        rotation=0.0
    ):
        """
        Generate equally spaced vertices of a regular polygon.

        Important:
        ----------
        side_length is the distance between adjacent polygon vertices.

        For a regular polygon with N vertices:

            circumradius =
                side_length /
                (2 * sin(pi / N))

        Returns
        -------
        centers : list of (y, x)
        """

        H, W = shape

        if center is None:
            center_y = H / 2
            center_x = W / 2
        else:
            center_y, center_x = center

        # Convert polygon side length into circumradius
        circumradius = side_length / (
            2 * np.sin(np.pi / num_vertices)
        )

        centers = []

        for i in range(num_vertices):

            angle = (
                rotation +
                2 * np.pi * i / num_vertices
            )

            y = center_y + circumradius * np.sin(angle)
            x = center_x + circumradius * np.cos(angle)

            # Keep vertices inside map
            y = int(np.clip(y, 0, H - 1))
            x = int(np.clip(x, 0, W - 1))

            centers.append((y, x))

        return centers

    # -------------------------------------------------------------------------
    # Fire blobs arranged as a regular polygon
    # -------------------------------------------------------------------------

    def generate_polygon_fire(
        self,
        shape,
        num_vertices,
        side_length,
        blob_radius=25,
        center=None,
        rotation=0.0,
        edge_sigma=2.0,
        threshold=0.4
    ):
        """
        Generate fire blobs positioned at the vertices of a regular polygon.

        Examples
        --------
        num_vertices=3 -> triangle
        num_vertices=4 -> square
        num_vertices=5 -> pentagon
        """

        centers = self.generate_polygon_centers(
            shape=shape,
            num_vertices=num_vertices,
            side_length=side_length,
            center=center,
            rotation=rotation
        )

        fire = np.zeros(shape, dtype=np.float32)

        for center in centers:

            blob = self.generate_fixed_blob(
                shape=shape,
                center=center,
                radius=blob_radius,
                edge_sigma=edge_sigma,
                threshold=threshold
            )

            fire = np.maximum(fire, blob)

        return fire.astype(np.uint8), centers

    # -------------------------------------------------------------------------
    # Two separated fire blobs
    # -------------------------------------------------------------------------

    def generate_two_blob_fire(
        self,
        shape,
        spacing,
        blob_radius=25,
        center=None,
        angle=0.0,
        edge_sigma=2.0,
        threshold=0.4
    ):
        """
        Generate exactly two fire blobs separated by a fixed center-to-center
        distance.

        Parameters
        ----------
        spacing : float
            Distance between the centers of the two blobs.

        angle : float
            Orientation of the pair in radians.
        """

        H, W = shape

        if center is None:
            center_y = H / 2
            center_x = W / 2
        else:
            center_y, center_x = center

        half_spacing = spacing / 2

        dy = half_spacing * np.sin(angle)
        dx = half_spacing * np.cos(angle)

        center_1 = (
            int(np.clip(center_y - dy, 0, H - 1)),
            int(np.clip(center_x - dx, 0, W - 1))
        )

        center_2 = (
            int(np.clip(center_y + dy, 0, H - 1)),
            int(np.clip(center_x + dx, 0, W - 1))
        )

        fire_1 = self.generate_fixed_blob(
            shape=shape,
            center=center_1,
            radius=blob_radius,
            edge_sigma=edge_sigma,
            threshold=threshold
        )

        fire_2 = self.generate_fixed_blob(
            shape=shape,
            center=center_2,
            radius=blob_radius,
            edge_sigma=edge_sigma,
            threshold=threshold
        )

        fire = np.maximum(fire_1, fire_2)

        return fire.astype(np.uint8), [center_1, center_2]

    # -------------------------------------------------------------------------
    # Main deterministic evaluation map generator
    # -------------------------------------------------------------------------

    def create_eval_map(
        self,
        map_id,
        canopy_density_alive=0.2,
        canopy_density_dead=0.1,
        canopy_size_mean=8,
        merge_radius=3,

        # Fire parameters
        blob_radius=25,
        side_length=150,
        blob_spacing=200,

        # Polygon parameters
        polygon_center=None,
        polygon_rotation=0.0,

        # Two-blob parameters
        two_blob_center=None,
        two_blob_angle=0.0,

        seed=1234
    ):
        """
        Generate deterministic evaluation maps.

        map_id
        ------
        0 : Triangle
        1 : Square
        2 : Pentagon
        3 : Two separated blobs

        Returns
        -------
        world_map : np.ndarray
            Shape: (H, W, 2)

            world_map[:, :, 0] = fuel
            world_map[:, :, 1] = fire

        metadata : dict
            Contains fire blob centers and map information.
        """

        H, W = self.size

        # ================================================================
        # Deterministic fuel map
        # ================================================================

        # Use separate deterministic seeds for alive and dead vegetation.
        #
        # This avoids both masks being identical when the same seed is used.
        tree_mask_alive = self.generate_tree_mask_fastest(
            shape=self.size,
            canopy_density=canopy_density_alive,
            canopy_size_mean=canopy_size_mean,
            merge_radius=merge_radius,
            edge_noise_scale=2,
            seed=seed
        )

        tree_mask_dead = self.generate_tree_mask_fastest(
            shape=self.size,
            canopy_density=canopy_density_dead,
            canopy_size_mean=canopy_size_mean,
            merge_radius=merge_radius,
            edge_noise_scale=2,
            seed=seed + 1
        )

        w = 0.7

        forest_fuel_map = (
            w * tree_mask_alive +
            (1 - w) * tree_mask_dead
        )

        # ================================================================
        # Deterministic fire map
        # ================================================================

        if map_id == 0:

            # Triangle
            fire_mask, fire_centers = self.generate_polygon_fire(
                shape=self.size,
                num_vertices=3,
                side_length=side_length,
                blob_radius=blob_radius,
                center=polygon_center,
                rotation=polygon_rotation
            )

            map_name = "triangle"

        elif map_id == 1:

            # Square
            fire_mask, fire_centers = self.generate_polygon_fire(
                shape=self.size,
                num_vertices=4,
                side_length=side_length,
                blob_radius=blob_radius,
                center=polygon_center,
                rotation=polygon_rotation
            )

            map_name = "square"

        elif map_id == 2:

            # Pentagon
            fire_mask, fire_centers = self.generate_polygon_fire(
                shape=self.size,
                num_vertices=5,
                side_length=side_length,
                blob_radius=blob_radius,
                center=polygon_center,
                rotation=polygon_rotation
            )

            map_name = "pentagon"

        elif map_id == 3:

            # Two separated blobs
            fire_mask, fire_centers = self.generate_two_blob_fire(
                shape=self.size,
                spacing=blob_spacing,
                blob_radius=blob_radius,
                center=two_blob_center,
                angle=two_blob_angle
            )

            map_name = "two_blobs"

        else:
            raise ValueError(
                f"Unknown map_id={map_id}. "
                f"Available map IDs: 0, 1, 2, 3."
            )

        # ================================================================
        # Construct world map
        # ================================================================

        world_map = np.zeros(
            (H, W, 2),
            dtype=np.float32
        )

        world_map[:, :, 0] = forest_fuel_map
        world_map[:, :, 1] = fire_mask

        # Fixed wind for evaluation
        # You can change this depending on whether wind is part of the
        # observation or dynamics.
        wind_field = (
            1.0,
            0.0,
            1.0
        )

        metadata = {
            "map_id": map_id,
            "map_name": map_name,
            "fire_centers": fire_centers,
            "blob_radius": blob_radius,
            "side_length": side_length,
            "blob_spacing": blob_spacing
        }

        return world_map, wind_field, metadata