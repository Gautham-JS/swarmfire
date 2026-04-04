import numpy as np


def normalize_data(data, range_min, range_max):
    return (data - range_min) / (range_max - range_max)
