import numpy as np
import concurrent.futures
from shapely.geometry import box
from functools import partial
import pandas as pd
import geopandas as gpd
from typing import List, Union
import xarray
import rioxarray


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
    def __init__(self, data: gpd.GeoDataFrame, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return BioGeoSeries

    @property
    def _constructor_expanddim(self):
        return BioGeoDataFrame


class BioGeoDataFrame(gpd.GeoDataFrame):
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

    @property
    def _constructor(self):
        return BioGeoDataFrame

    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            srs = BioSeries(*args, **kwargs)
            is_row_proxy = srs.index is self.columns
            if (
                isinstance(getattr(srs, "dtype", None), gpd.base.BaseGeometry)
                and not is_row_proxy
            ):
                srs = BioGeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

    def _is_within(self, raster):
        return self.intersects(box(*raster.rio.bounds()))

    def list_rasters(
        self, distance: int, rasters: Union[List[xarray.DataArray], str]
    ) -> List[xarray.DataArray]:
        rasters_list = []

        self_copy = self.copy()
        self_copy["geometry"] = self_copy.buffer(distance / 2, cap_style=3)

        for raster in rasters:
            r = rioxarray.open_rasterio(raster) if isinstance(raster, str) else raster
            w = self_copy[self_copy._is_within(r)]
            if not w.empty:
                rasters_list.append(r)

        return rasters_list

    def add_pseudo_absences(
        self, amount, region_poly=None, not_within=5000, shuffle=True
    ):
        self["presence"] = 1

        constrain_by = self.buffer(not_within).unary_union
        zeros = self._sample_pseudo_absences(amount, region_poly, constrain_by)

        zeros["presence"] = 0

        if shuffle:
            zeros = (
                pd.concat([self, zeros]).sample(frac=1).drop(columns=["index_right"])
            )

        return zeros

    def _sample_pseudo_absences(self, amount, region_poly, constrain_by=None):
        minx, miny, maxx, maxy = region_poly.total_bounds

        random_lat = np.random.uniform(low=miny, high=maxy, size=(amount,))
        random_lon = np.random.uniform(low=minx, high=maxx, size=(amount,))

        geodf_tmp = gpd.GeoDataFrame(
            {"decimalLatitude": random_lat, "decimalLongitude": random_lon}
        )

        geodf_tmp = geodf_tmp.set_geometry(
            gpd.points_from_xy(geodf_tmp.decimalLongitude, geodf_tmp.decimalLatitude)
        ).set_crs(region_poly.crs)

        geodf = geodf_tmp.sjoin(region_poly[["geometry"]], how="inner")

        geodf = geodf[~geodf.intersects(constrain_by)]

        points_remaining = amount - geodf.shape[0]

        if points_remaining > 0:
            print(f"{points_remaining} pseudo-absence points remaining.")

            extras = self._sample_pseudo_absences(
                region_poly=region_poly,
                amount=points_remaining,
                constrain_by=constrain_by,
            )

            geodf = pd.concat([geodf, extras])

            return geodf

    def _extract_values(self, chunk, raster, distance):
        nrow = chunk.shape[0]

        # if bands is None:
        #     bands = list(raster.keys())

        res = raster.rio.resolution()[1]

        outer_list = []

        for idx, point in chunk.iterrows():
            g = point["geometry"]
            minx, miny, maxx, maxy = (
                g.x - (distance - res / 2),
                g.y - (distance - res / 2),
                g.x + (distance - res / 2),
                g.y + (distance - res / 2),
            )

            inner_array = None
            inner_list = []

            try:
                # rast = raster.rio.clip(
                #     geometries=[point['buffered_geometry']], all_touched=True)
                rast = raster.rio.clip_box(minx, miny, maxx, maxy)

                for band in rast:
                    values = rast[band].values

                    if None in values or values.shape != (64, 64):
                        continue

                    inner_list.append(values)

            except Exception as e:
                print(e)

            if len(inner_list) < 2:
                continue

            inner_array = np.stack(inner_list)

            outer_list.append({"arr": inner_array, "presence": point["presence"]})

        return outer_list

    def extract_values(self, raster, distance, n_cores=1):
        # bands = list(raster.keys())

        if n_cores > 1:
            chunks = np.array_split(self, n_cores)
            extract_func = partial(
                self._extract_values, raster=raster, distance=distance
            )

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_cores
            ) as executor:
                data = list(executor.map(extract_func, chunks))

        else:
            data = self._extract_values(self, raster, distance)

        return data
