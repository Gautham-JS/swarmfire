import numpy as np


   
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



class IncrementalViewAccumulator:
    def __init__(self, size):
        self.size = size
        self.scene = np.zeros(size, dtype=np.float32)
        self.counts = np.zeros(size, dtype=np.float32)

    def accumulate(self, view:np.ndarray, center_position:tuple, view_size=32):
        vh, vw = view.shape[:2]
        vh_half, vw_half = vh/2, vw/2

        H, W = self.scene.shape
        row, col = center_position
        half = view_size // 2

        src_row_start = max(0, row - half)
        src_row_end   = min(H, row - half + view_size)
        src_col_start = max(0, col - half)
        src_col_end   = min(W, col - half + view_size)

        dst_row_start = src_row_start - (row - half)
        dst_row_end   = dst_row_start + (src_row_end - src_row_start)
        dst_col_start = src_col_start - (col - half)
        dst_col_end   = dst_col_start + (src_col_end - src_col_start)

        self.scene[
            src_row_start : src_row_end, 
            src_col_start : src_col_end
        ] += view [
            dst_row_start:dst_row_end, 
            dst_col_start:dst_col_end
        ]

        self.counts[
            src_row_start : src_row_end, 
            src_col_start : src_col_end
        ] += 1


    def get_scene(self):
        local_scene = np.divide(self.scene, self.counts, where=self.counts>0)
        return local_scene
