import numpy as np
from scipy.ndimage import zoom


def get_view_bound_coords_3d(array, point: tuple[int, int, int], size: int):
    H, W = array.shape
    row, col, z = point
    actual_size = size + z  # z expands the viewpoint size
    half = actual_size // 2

    src_row_start = max(0, row - half)
    src_row_end   = min(H, row + half)
    src_col_start = max(0, col - half)
    src_col_end   = min(W, col + half)

    return ((src_row_start, src_col_start), (src_row_end, src_col_end))


def get_square_viewpoint_3d(
    array: np.ndarray,
    point: tuple[int, int, int],
    size: int = 16,
    z_scale: float = 1.0,
    confidence_base: float = 2.0
) -> np.ndarray:
    """
    Extract a scaled viewpoint from array centered on point.

    Args:
        array:           2D input array to sample from.
        point:           (row, col, z) — z expands view size and reduces pixel confidence.
        size:            Base viewpoint size at z=0.
        z_scale:         Controls how aggressively z grows the viewpoint window.
        confidence_base: Base of the exponential decay for pixel scaling.
                         Higher = faster decay as z increases.
    """
    H, W = array.shape
    row, col, z = point

    # --- Size scaling: z expands the sampled window ---
    actual_size = size + int(z * z_scale)
    actual_size = max(1, actual_size)  # guard against degenerate sizes
    half = actual_size // 2

    # Source bounds (clamped to array edges)
    src_row_start = max(0, row - half)
    src_row_end   = min(H, row + half)
    src_col_start = max(0, col - half)
    src_col_end   = min(W, col + half)

    # Destination bounds in output (always fixed output size)
    dst_row_start = src_row_start - (row - half)
    dst_row_end   = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - (col - half)
    dst_col_end   = dst_col_start + (src_col_end - src_col_start)

    # Sample the source patch and resize it down to the fixed output size
    src_patch = array[src_row_start:src_row_end, src_col_start:src_col_end]
    viewpoint = np.zeros((actual_size, actual_size), dtype=np.float64)
    viewpoint[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = src_patch

    # Resize the larger sampled window back to the base output size
    viewpoint = zoom(viewpoint, size / actual_size)

    # --- Confidence scaling: higher z attenuates pixel values ---
    # Decays as: 1 / (confidence_base ^ z), so z=0 → scale=1.0
    confidence_scale = 1.0 / (confidence_base ** z)
    viewpoint = viewpoint * confidence_scale

    return viewpoint.astype(array.dtype)


def get_view_bound_coords(array, point:tuple[int, int], size:int):
    H, W = array.shape
    row, col = point
    half = size // 2

    src_row_start = max(0, row - half)
    src_row_end   = min(H, row - half + size)
    src_col_start = max(0, col - half)
    src_col_end   = min(W, col - half + size)

    return ((src_row_start, src_col_start), (src_row_end, src_col_end))

   
def get_square_viewpoint(array: np.ndarray, point: tuple[int, int], size: int = 16) -> np.ndarray:
    H, W = array.shape
    row, col = point
    half = size // 2

    # Source bounds (clamped to array edges)
    src_row_start = max(0, row - half)
    src_row_end   = min(H, row - half + size)
    src_col_start = max(0, col - half)
    src_col_end   = min(W, col - half + size)

    # Destination bounds (where to place in output)
    dst_row_start = src_row_start - (row - half)
    dst_row_end   = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - (col - half)
    dst_col_end   = dst_col_start + (src_col_end - src_col_start)

    viewpoint = np.zeros((size, size), dtype=array.dtype)
    viewpoint[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
        array[src_row_start:src_row_end, src_col_start:src_col_end]

    return viewpoint


def get_square_viewpoint_and_mark_visited(array: np.ndarray, visited: np.ndarray, point: tuple[int, int], size: int = 16) -> np.ndarray:
    H, W = array.shape
    row, col = point
    half = size // 2

    # Source bounds (clamped to array edges)
    src_row_start = max(0, row - half)
    src_row_end   = min(H, row - half + size)
    src_col_start = max(0, col - half)
    src_col_end   = min(W, col - half + size)

    # Destination bounds (where to place in output)
    dst_row_start = src_row_start - (row - half)
    dst_row_end   = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - (col - half)
    dst_col_end   = dst_col_start + (src_col_end - src_col_start)

    viewpoint = np.zeros((size, size), dtype=array.dtype)
    viewpoint[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
        array[src_row_start:src_row_end, src_col_start:src_col_end]
    
    old_visited_mask = np.zeros((size, size), dtype=np.bool)
    old_visited_mask[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = visited[src_row_start:src_row_end, src_col_start:src_col_end] > 0
    delta_mask = np.logical_not(old_visited_mask).copy()

    visited[src_row_start:src_row_end, src_col_start:src_col_end] = True

    return viewpoint, visited, delta_mask


class IncrementalViewAccumulator:
    def __init__(self, size, n_channels):
        self.size = size
        self.scene = np.zeros((size[0], size[1], n_channels), dtype=np.float32)
        self.visited = np.zeros((size[0], size[1]), dtype=bool)

    def accumulate(self, view: np.ndarray, center_position: tuple, view_size=32):
        H, W = self.scene.shape[:2]
        row, col = center_position[:2]
        half = view_size // 2

        src_row_start = max(0, row - half)
        src_row_end   = min(H, row - half + view_size)
        src_col_start = max(0, col - half)
        src_col_end   = min(W, col - half + view_size)

        dst_row_start = src_row_start - (row - half)
        dst_row_end   = dst_row_start + (src_row_end - src_row_start)
        dst_col_start = src_col_start - (col - half)
        dst_col_end   = dst_col_start + (src_col_end - src_col_start)

        visited_slice = self.visited[src_row_start:src_row_end, src_col_start:src_col_end]

        self.scene[src_row_start:src_row_end, src_col_start:src_col_end, :] = view[dst_row_start:dst_row_end, dst_col_start:dst_col_end, :]
        self.visited[src_row_start:src_row_end, src_col_start:src_col_end] = True

        transformed_bounds = ((src_row_start, src_col_start), (src_row_end, src_col_end))

        return self.get_scene(), transformed_bounds

    def get_scene(self):
        return np.where(self.visited[:, :, np.newaxis], self.scene, 0.0)

    def reset(self):
        self.scene[:] = 0.0
        self.visited[:] = False