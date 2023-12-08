import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
from itertools import repeat
from multiprocessing import Pool
# import biogeodataframe
import pandas as pd
from rasterio.enums import Resampling
import math
import rasterio
from rasterio.vrt import WarpedVRT

def sample_pseudo_zeros(amount, region_poly, constrain_by=None):

    minx, miny, maxx, maxy = region_poly.total_bounds

    random_lat = np.random.uniform(low=miny, high=maxy, size=(amount,))
    random_lon = np.random.uniform(low=minx, high=maxx, size=(amount,))

    geodf_tmp = gpd.GeoDataFrame(
        {'decimalLatitude': random_lat, 'decimalLongitude': random_lon})

    geodf_tmp = geodf_tmp.set_geometry(gpd.points_from_xy(
        geodf_tmp.decimalLongitude, geodf_tmp.decimalLatitude)).set_crs(region_poly.crs)

    geodf = geodf_tmp.sjoin(region_poly[['geometry']], how='inner')

    geodf = geodf[~geodf.intersects(constrain_by)]

    points_remaining = amount - geodf.shape[0]

    if points_remaining > 0:

        print(f'{points_remaining} pseudo-absence points remaining.')

        extras = sample_pseudo_zeros(
            region_poly=region_poly,  amount=points_remaining, constrain_by=constrain_by)
        
        geodf = pd.concat([geodf, extras])

    return geodf