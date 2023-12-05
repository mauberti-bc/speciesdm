from __future__ import annotations
from typing import Optional, List, Union
import os
# os.environ['USE_PYGEOS'] = '0'
import rioxarray
import xarray
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from util import make_grid
import math
from pseudo_zeros import sample_pseudo_zeros_alt


class BioSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return BioSeries

    @property
    def _constructor_expanddim(self):
        return BioGeoDataFrame


class BioGeoSeries(gpd.GeoSeries):

    _metadata = ['name']

    def __init__(self, data: gpd.GeoDataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        # self._obj: gpd.GeoSeries = data
        # self.geometry = data.geometry

    @property
    def _constructor(self):
        return BioGeoSeries

    @property
    def _constructor_expanddim(self):
        return BioGeoDataFrame


class BioGeoDataFrame(gpd.GeoDataFrame):

    _metadata = ['name']

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self._obj: gpd.GeoDataFrame = data

    @property
    def _constructor(self):
        return BioGeoDataFrame

    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """
            A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries:
            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """

            srs = BioSeries(*args, **kwargs)
            is_row_proxy = srs.index is self.columns
            if (
                isinstance(getattr(srs, "dtype", None), gpd.base.BaseGeometry) |
                isinstance(getattr(srs, "dtype", None), gpd.base.GeometryDtype)
            ) and not is_row_proxy:
                srs = BioGeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

    # # @classmethod
    # def from_txt(
    #         cls,
    #         file_path: str,
    #         lat: str,
    #         lon: str,
    #         crs: Union[Optional[str], Optional[int]] = 4326):

    #     data = pd.read_csv(file_path, sep="\t", low_memory=False)
    #     geometry = [Point(xy) for xy in zip(data[lon], data[lat])]
    #     # og_data = gpd.GeoDataFrame(data, crs=f"EPSG:{crs}", geometry=geometry)
    #     og_data = gpd.GeoDataFrame(data, crs=f"EPSG:{crs}", geometry=geometry)

    #     return cls(og_data)

    def is_within(
            self,
            raster: xarray.DataArray) -> BioGeoDataFrame:
        """
        Check if geometries from a GeoDataFrame are within the bounds of a DataArray

        Parameters
        ----------
        raster: xarray.DataArray
            A DataArray with which to intersect the geometries

        Returns
        -------
        :obj:`pandas.Series`
            Boolean Series describing whether each row intersects with the raster
        """
        return self.intersects(box(*raster.rio.bounds()))

    # def thin(
    #         self,
    #         resolution: float = 5,
    #         thin_proportion: Union[Optional[float], Optional[str]] = 'auto',
    #         raster_bounds=None,
    #         minimum: Optional[int] = None) -> BioGeoDataFrame:
    #     """
    #     Remove some species occurrence records from areas with a high density of species occurrences.
    #     Works by projecting a grid with resolution `resolution` onto the occurrence records, then removing a
    #     proportion (equal to `thin_proportion`) of occurrences in each grid cell. The occurrences to remove are randomly chosen.

    #     Parameters
    #     -----------
    #     resolution: float
    #         The size of each grid cell in which to downsample records by the `thin_proportion`.
    #         Defaults to 5 km (25km^2)
    #     thin_proportion: float
    #         The proportion of species occurrence records to remove within each grid cell. Defaults to 'auto', which is \
    #             the the number of records in the cell divided by the resolution^2.
    #     minimum:
    #         The minimum number of species occurrence records that should remain in each grid cell. If `thin_proportion`
    #         yields fewer species occurrence records than `minimum`, thinning will stop when the minimum number
    #         of occurrence records is reached. If a cell has fewer records than `minimum`, none of will be removed. Defaults to None.

    #     Returns
    #     -------
    #     :obj:`GeoDataFrame`:
    #         A thinned set of the original species occurrence records.

    #     """
    #     if (type(thin_proportion) == str) and thin_proportion not in ['auto', 'binary']:
    #         return "`thin_proportion` must be a value between 0 and 1, 'auto', or 'binary'"
    #     if set(self._obj.geom_type) != {'Point'}:
    #         return "All geometries must be of geom type 'Point'"

    #     # Create empty grid to overlay on species occurrence records
    #     grid = make_grid(gdf_template=self._obj, resolution=resolution,
    #                      raster_bounds=raster_bounds)

    #     # For each grid cell, randomly select (1-thin_proportion) of them
    #     new_points_list = []
    #     for x in range(len(grid.coords['x']) - 1):
    #         for y in range(len(grid.coords['y']) - 1):

    #             # print(grid.coords['x'], 'grid coords x: ', grid.coords['x'][x])
    #             bbox = grid.coords['x'][x].values, grid.coords['y'][y].values, \
    #                 grid.coords['x'][x+1].values, grid.coords['y'][y+1].values
    #             bbox = [int(p) for p in bbox]

    #             w = self.is_within(grid[x:x+2, y:y+2])
    #             points = self._obj[w]

    #             nrows = points.shape[0]
    #             if nrows == 0:
    #                 continue

    #             if thin_proportion == 'auto':
    #                 # Must change to vary with resolution and nrows, like 0.75 * (nrows / resolution ** 2) or something
    #                 thin_proportion = 0.75

    #             # If the minimum value is None, thin the records in every cell by the thin_proportion
    #             if minimum is None or (nrows * thin_proportion > minimum):
    #                 new_points = points.sample(
    #                     frac=1 - thin_proportion, replace=False)
    #             # # If the number of records after thinning by thin_proportion is above the minimum, thin the records by thin_proportion
    #             # elif nrows * thin_proportion > minimum:
    #             #     new_points = points.sample(
    #             #         frac=1 - thin_proportion, replace=False)
    #             # If the number of records after thinning by thin_proportion is equal to or less than the minimum,
    #             # thin until the minimum is reached
    #             elif nrows * thin_proportion <= minimum:
    #                 # If there are fewer records than the minimum, use the number of records
    #                 if nrows < minimum:
    #                     new_points = points.sample(n=nrows, replace=False)
    #                 # If there are more records then minimum, use the minimum
    #                 else:
    #                     new_points = points.sample(n=minimum, replace=False)

    #             new_points_list.append(new_points)

    #     self._obj = pd.concat(new_points_list, ignore_index=True)

    #     return self._obj

    # def rasterize(
    #         self,
    #         rioxarray_template,
    #         # resolution: int == None,
    #         threshold: Optional[int] = 1,
    #         type: Optional[str] = 'presence') -> xarray.DataArray:
    #     """
    #     Create a raster to represent the number of species occurrences or the presence-absence of species occurrences.
    #     Defaults to a binary raster where 1 represents presence and 0 represents absence.

    #     Parameters
    #     -----------
    #     bounds:
    #         The bounding box (xmin, ymin, xmax, ymax) of the output raster
    #     resolution:
    #         The resolution of each cell in the output raster
    #     threshold:
    #         The minimum number of occurrences for a species to be considered present in a raster cell. Defaults to 1.
    #     type:
    #         Whether the output raster should represent presence-absence values or a count of the number of species occurrence records.
    #         Use 'Presence' for presence-absence and 'Count' to count the number of occurrence records.

    #     """
    #     if type not in ['presence', 'count']:
    #         return "Type must be equal to 'presence' or 'pount'"

    #     resolution = math.ceil(set([abs(x)
    #                                 for x in rioxarray_template.rio.resolution()]).pop())*1

    #     print(resolution)

    #     # Create empty grid to overlay on species occurrence records
    #     grid = make_grid(gdf_template=self._obj,
    #                      rioxarray_template=rioxarray_template,
    #                      resolution=resolution)

    #     for x in range(len(grid.coords['x']) - 1):
    #         for y in range(len(grid.coords['y']) - 1):

    #             # print(grid.coords['x'], 'grid coords x: ', grid.coords['x'][x])
    #             bbox = grid.coords['x'][x].values, grid.coords['y'][y].values, \
    #                 grid.coords['x'][x+1].values, grid.coords['y'][y+1].values
    #             bbox = [int(p) for p in bbox]

    #             w = self.is_within(grid[x:x+2, y:y+2])
    #             points = self._obj[w]

    #             if type == 'presence':
    #                 if not points.empty:
    #                     grid[x, y] = 1
    #             elif type == 'count':
    #                 grid[x, y] = points.shape[0]

    #             grid.rio.write_crs(rioxarray_template.rio.crs, inplace=True)

    #     return grid

    def which_rasters(
            self,
            distance: int,
            rasters: Union[List[xarray.DataArray], str]) -> xarray.DataArray:

        rasters_list = []

        # Given a list of raster tiles, find which ones are within x distance of a coord
        gdf_copy = self.copy()
        gdf_copy['geometry'] = gdf_copy.buffer(distance/2, cap_style=3)

        # For each raster tile, check if any of the coordinates are within it
        for raster in rasters:
            if type(raster) == str:
                r = rioxarray.open_rasterio(raster)
            else:
                r = raster

            w = gdf_copy[gdf_copy.is_within(r)]

            # dict = {
            #     "raster": raster,
            #     "records": buffered_gdf[w].index.values.tolist()
            # }

            # rasters_list.append(dict)
            if not w.empty:
                rasters_list.append(raster)

            try:
                del r
            except:
                continue

        return rasters_list

    def load_rasters(
            self,
            distance: int,
            rasters: Union[List[xarray.DataArray], str],
            chunks: str = 'auto') -> xarray.DataArray:
        """
        Load all rasters within `distance` of any species occurrence record.
        """
        to_load = self.which_rasters(distance=distance, rasters=rasters)

        return [rioxarray.open_rasterio(file, chunks='auto') for file in to_load]

    def add_pseudo_absences(
        self,
        amount,
        region_poly,
        # constrain_by=None,
        shuffle=True
    ) -> BioGeoDataFrame:
        self['presence'] = 1

        zeros = sample_pseudo_zeros_alt(amount=amount, region_poly=region_poly) #constrain_by=constrain_by,
        zeros['presence'] = 0

        if shuffle:
            zeros = BioGeoDataFrame(pd.concat([self, zeros])).sample(frac=1).drop(columns=['index_right'])

        return zeros
    
    # def extract_values(self, raster, bands=None):
    #     # master_raster = rioxarray.open_rasterio(
    #     #     raster, cache=False, chunks=True, mask_and_scale=True).squeeze()

    #     # vars = master_raster.attrs['long_name']
    #     # vars = [var for var in vars if 'max' not in var.split(
    #     #     "_") and 'min' not in var.split("_")]

    #     # print(master_raster.rio.crs, master_raster.rio.crs == geodataframe.crs)
    #     if bands is None:
    #         bands = list(raster.keys())

    #     # print(raster)
    #     # print(bands)

    #     outer_array = None

    #     for idx, point in self.iterrows():
    #         inner_array = None
    #         print(f"Extracting values: iteration {idx}")
    #         # try:
    #         rast = raster.rio.clip(
    #             geometries=[point['buffered_geometry']], all_touched=True)
            
    #         for idx, band in enumerate(bands):
    #             # for idx, band in enumerate(files):
    #             values = rast[band].values
    #             # mean = np.nanmean(values)
    #             # point[f'{band}'] = mean
    #             if inner_array is None:
    #                 inner_array = values
    #             else:
    #                 inner_array = np.stack((inner_array, values))
    #             # print(f"{band}: {values}")
    #         # except Exception as e:
    #         #     print(e)
    #         print(inner_array, inner_array.shape)
    #         if outer_array is None:
    #             outer_array = inner_array
    #         else:
    #             outer_array = np.dstack(outer_array, inner_array)

    #     return outer_array
    def extract_values(self, raster, bands=None):
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

