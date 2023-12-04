from biogeodataframe import BioGeoDataFrame
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from geocube.api.core import make_geocube
from rioxarray.merge import merge_arrays

# Set the CRS to BC Albers
CRS = 'EPSG:3005'
BUFFER_DISTANCE = 5000 # in units of CRS

# Read in species occurrence data as a geodataframe and remove non-georeferenced rows
species_tmp = gpd.read_file('../data/black_bear_occurrences.csv')
species_tmp = species_tmp[(species_tmp['decimalLatitude'] != '') & (species_tmp['decimalLongitude'] != '')]

# Convert the geopandas to a BioGeoDataFrame, giving access to useful methods
species = BioGeoDataFrame(species_tmp).sample(500)
species = species.set_geometry(gpd.points_from_xy(
        species['decimalLongitude'], species['decimalLatitude'])).set_crs(4326)
species = species.to_crs(CRS)

# Load in biogeoclimatic zones and reproject
bec_tmp = gpd.read_file('../data/bec').to_crs(CRS)
bec_tmp = bec_tmp[['ZONE', 'AREA_SQM', 'geometry']]

# Categorical variables must be made numeric to be transformed into a raster, so must convert numbers back to strings
# To do this, create list of all strings
bec_zones = bec_tmp.ZONE.drop_duplicates().values.tolist()
categorical_enums = {'ZONE': bec_zones}

# Convert bec geodataframe to rioxarray raster
# Resolution is in the units of target CRS
bec = make_geocube(vector_data = bec_tmp, resolution=(1000, -1000), categorical_enums=categorical_enums)

# Convert numeric back to categorical string
zone_string = bec['ZONE_categories'][bec['ZONE'].astype(int)].drop('ZONE_categories')
bec['ZONE'] = zone_string

# Given a list of raster tiles, find which ones intersect the species occurrence points and are therefore required
# Using a single raster, bec, for simplicity
rasters = species.which_rasters(BUFFER_DISTANCE, [bec])

# Load the list of raster tiles into memory
# Would load the rasters here, but bec is already loaded for simplicity. Something like:
# rasters = [rioxarray.open_rasterio(x) for x in raster]
# merged_raster = merge_arrays(rasters)
merged_raster = bec

# # Buffer each point so it intersects adjacent raster cells
species['buffered_geometry'] = species['geometry'].buffer(BUFFER_DISTANCE, cap_style=3)

# For each occurrence point, build a 3D tensor
vals = species.extract_values(merged_raster)


