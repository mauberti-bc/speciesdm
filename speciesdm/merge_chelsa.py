import os
import dask
import dask.array as da
import rioxarray
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import numpy as np
from rioxarray.merge import merge_arrays
import threading
import xarray as xr

# Define the output projection and resolution
CRS = "EPSG:3005"
RESOLUTION = 500


def load_tiff(tiff):
    with rasterio.open(tiff) as src:
        vrt_options = {
            "resampling": Resampling.average,
            "nodata": np.nan,
        }

        # Importing with dask (chunks=True) causes gap error
        with WarpedVRT(src, **vrt_options) as vrt:
            src_file = rioxarray.open_rasterio(vrt, cache=False, chunks=True)
            return src_file


# Set the path to your folder containing .tif files
input_folder = "../data/bio"

# Set the path for the output merged, reprojected, and downsampled raster file
output_file = "../data/chelsa_tif_bcalbers.tif"

# List all the .tif files in the input folder
tif_files = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.endswith(".tif") and "bio" in f.split("_")[1]
]

# Ensure the list is sorted to maintain order
tif_files.sort()

minx, miny, maxx, maxy = gpd.read_file("../data/bec").total_bounds

tifs = []
for idx, tif in enumerate(tif_files):
    print(idx)
    t = load_tiff(tif).rio.clip_box(minx, miny, maxx, maxy)
    tifs.append(t)

print("Merging!")

# merged_ds = merge_arrays(tifs).rio.reproject(CRS)
merged_ds = xr.concat(tifs, dim="band", join="override").rio.reproject(
    dst_crs=CRS,
    resampling=Resampling.average,
    resolution=(RESOLUTION, -RESOLUTION),
)

merged_ds["band"] = [tif.split("/")[-1] for tif in tif_files]
merged_ds.attrs["band_names"] = [tif.split("/")[-1] for tif in tif_files]

print("Saving!")

# Save the clipped dataset to a new GeoTIFF file
merged_ds.rio.to_raster(
    output_file
)  # , driver="GTiff", lock=threading.Lock(), num_threads=8)
