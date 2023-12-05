from __future__ import annotations
from typing import Optional, List, Union
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from util import make_grid
from pseudo_zeros import sample_pseudo_zeros_alt
from multiprocessing import Pool
from itertools import repeat

def _extract_values(self, raster, bands=None):
        print(raster)
        if bands is None:
            bands = list(raster.keys())

        outer_list = []

        for idx, point in self.iterrows():
            inner_array = None
            try:
                rast = raster.rio.clip(
                    geometries=[point['buffered_geometry']], all_touched=True)
                
                for idx, band in enumerate(bands):
                    values = rast[band].values
                    print(idx, values)
                    if None in values:
                        continue
                    if inner_array is None:
                        inner_array = values
                    else:
                        inner_array = np.stack((inner_array, values))
            except Exception as e:
                print(e)
            outer_list.append({'arr': inner_array, 'presence': point.presence})

        return outer_list