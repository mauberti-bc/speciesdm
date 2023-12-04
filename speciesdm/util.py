import xarray
from typing import Optional
import numpy as np
from geopandas import GeoDataFrame


# resolution: int, bbox: tuple, lat: List[float], lon: List[float], data: np.array = None, description: Optional[str] = None, units: Optional[str] = None):
def make_grid(resolution: int, gdf_template: Optional[GeoDataFrame] = None, raster_bounds=None, bbox: Optional[tuple] = None):
    """
    Create a DataArray raster

    parameters
    ----------
    gdf_template:
        A GeoDataFrame from which to determine the bounding box of the grid
    rioxarray_template
        A rioxarray object from which to determine the bounding box of the grid
    resolution:
        The resolution of the grid
    bbox:
        The bounding box (xmin, ymin, xmax, ymax) of the grid
    data:
        The values to assign to the output raster
    lat:


    """
    # Create empty grid to project onto the species occurrence records
    # Get bounding box of the input GeoDataFrame
    if (bbox is None) and (gdf_template is None):
        return "Must include either gdf_template or bbox to determine bounds of the grid"
    elif (bbox is not None) and (gdf_template is not None):
        return "Can only accept gdf_template or bbox, not both"
    # elif (gdf_template is not None):
    #     xmin, ymin, xmax, ymax = gdf_template.total_bounds

    # if (rioxarray_template is not None):
    xmin, ymin, xmax, ymax = raster_bounds
    # else:
    #     xmin, ymin, xmax, ymax = bbox

    # Create array of coordinate values along x and y axes, using the resolution
    # Calculate the number of required grid cells in the x (xsize) and y (ysize) directions given the desired grid cell resolution
    # xsize = math.ceil((xmax-xmin) / resolution)
    # ysize = math.ceil((ymax-ymin) / resolution)

    # Create numpy arrays of the raster cell centroids in the x (longitudes) and y (latitudes) directions
    latitudes = np.arange(ymin, ymax, resolution)
    longitudes = np.arange(xmin, xmax, resolution)
    lat, lon = np.meshgrid(latitudes, longitudes)
    ysize = len(latitudes)
    xsize = len(longitudes)

    # # Create xarray DataArray
    grid = xarray.DataArray(
        data=np.zeros((len(longitudes), len(latitudes)), dtype=int),
        dims=["x", "y"],
        # coords = dict(
        #     x = (["x","y"], lon),
        #     y = (["x","y"], lat)
        # )
        coords={
            "x": longitudes,
            "y": latitudes,
            "longitude": (["x", "y"], lon),
            "latitude": (["x", "y"], lat)
        },
        name='Occurrences'
    )

    return grid