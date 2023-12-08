from __future__ import annotations
from typing import List, Union
import rioxarray
import xarray
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from multiprocessing import Pool
from itertools import repeat


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

    def _is_within(
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

    def list_rasters(
            self,
            distance: int,
            rasters: Union[List[xarray.DataArray], str]) -> xarray.DataArray:

        rasters_list = []

        # Given a list of raster tiles, find which ones are within x distance of a coord
        self_copy = self.copy()
        self_copy['geometry'] = self_copy.buffer(distance/2, cap_style=3)

        # For each raster tile, check if any of the coordinates are within it
        for raster in rasters:
            if type(raster) == str:
                r = rioxarray.open_rasterio(raster)
            else:
                r = raster

            w = self_copy[self_copy._is_within(r)]

            # rasters_list.append(dict)
            if not w.empty:
                rasters_list.append(raster)

            try:
                del r
            except:
                continue

        return rasters_list

    def add_pseudo_absences(
        self,
        amount,
        region_poly,
        not_within=5000,
        constrain_by=None,
        shuffle=True
    ) -> BioGeoDataFrame:
        
        # Pseudo-absences cannot be within {not_within} units (metres) of a presence

        self['presence'] = 1

        constrain_by = self.buffer(not_within).unary_union

        zeros = self._sample_pseudo_absences(amount=amount, region_poly=region_poly, constrain_by=constrain_by)
        zeros['presence'] = 0

        if shuffle:
            zeros = BioGeoDataFrame(pd.concat([self, zeros])).sample(frac=1).drop(columns=['index_right'])

        return zeros

    def _sample_pseudo_absences(self, amount, region_poly, constrain_by=None):

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

            extras = self._sample_pseudo_absences(
                region_poly=region_poly,  amount=points_remaining, constrain_by=constrain_by)
            
            geodf = pd.concat([geodf, extras])

            return geodf
        
    def _extract_values(self, chunk, raster, distance, bands=None):
        nrow = chunk.shape[0]

        if bands is None:
            bands = list(raster.keys())

        outer_list = []

        for idx, point in chunk.iterrows():
            g = point['geometry']
            minx, miny, maxx, maxy = (g.x - distance, g.y - distance, g.x + distance, g.y + distance)

            inner_array = None
            try:
                # rast = raster.rio.clip(
                #     geometries=[point['buffered_geometry']], all_touched=True)
                rast = raster.rio.clip_box(
                    minx, miny, maxx, maxy
                )
                
                for idx, band in enumerate(bands):
                    values = rast[band].values
                    print(idx, nrow, values.shape)
                    if None in values:
                        continue
                    if inner_array is None:
                        inner_array = values
                    else:
                        inner_array = np.stack((inner_array, values))
            except Exception as e:
                print(e)

            outer_list.append({'arr': inner_array, 'presence': point['presence']})

        return outer_list

    def extract_values(self, raster, distance, n_cores=8):
            chunks = np.array_split(self, n_cores)

            with Pool(n_cores) as pool:
                data = pool.starmap(self._extract_values, zip(
                    chunks, repeat(raster), repeat(distance)))
                pool.close()

            return data

    

