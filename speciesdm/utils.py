import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import random


def extract_subarrays_from_raster(raster, radius):
    """
    Extract 3D subarrays around each cell in a raster.

    Parameters:
    - raster: 3D NumPy array representing the raster
    - radius: Integer representing the radius around each cell

    Returns:
    - subarrays: List of 3D NumPy arrays containing the extracted subarrays
    """

    x_max, y_max, _ = raster.shape
    subarrays = []

    for x in range(x_max):
        x_min = max(0, x - radius)
        x_max_slice = min(x_max, x + radius)

        for y in range(y_max):
            print(x, y)

            y_min = max(0, y - radius)
            y_max_slice = min(y_max, y + radius)

            subarray = raster[x_min:x_max_slice, y_min:y_max_slice, :]
            subarrays.append(subarray)

    return subarrays
