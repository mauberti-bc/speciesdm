from biogeodataframe import BioGeoDataFrame
from osgeo import gdal
import geopandas as gpd
from rioxarray.merge import merge_arrays
from geocube.api.core import make_geocube
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


# Set the CRS to BC Albers
CRS = "EPSG:3005"
GEOCUBE_RES = 100
N_SAMPLES = 5000


# Read in species occurrence data as a geodataframe and remove non-georeferenced rows
species_tmp = gpd.read_file("../data/black_bear_occurrences.csv")
species_tmp = species_tmp[
    (species_tmp["decimalLatitude"] != "") & (species_tmp["decimalLongitude"] != "")
]


# Convert the geopandas to a BioGeoDataFrame, giving access to useful methods
N = np.nanmin((N_SAMPLES, species_tmp.shape[0]))
species_tmp = species_tmp.sample(N)

species = BioGeoDataFrame(species_tmp)
species = species.set_geometry(
    gpd.points_from_xy(species["decimalLongitude"], species["decimalLatitude"])
).set_crs(4326)
species = species.to_crs(CRS)


# Load in biogeoclimatic zones and reproject to desired CRS
# Use only the ZONE and geometry fields, the former of which is what we will predict species' distributions with
bec_tmp = gpd.read_file("../data/bec").to_crs(CRS)
bec_tmp = bec_tmp[["ZONE", "SUBZONE", "geometry"]]


# Categorical variables must be made numeric to be transformed into a raster, so must convert numbers back to strings
# To do this, create list of all strings
bec_zones = bec_tmp.ZONE.drop_duplicates().values.tolist()
bec_subzones = bec_tmp.SUBZONE.drop_duplicates().values.tolist()
categorical_enums = {"ZONE": bec_zones, "SUBZONE": bec_subzones}


# Convert bec geodataframe to rioxarray raster
# Resolution is in the units of target CRS
bec = make_geocube(
    vector_data=bec_tmp,
    resolution=(GEOCUBE_RES, -GEOCUBE_RES),
    categorical_enums=categorical_enums,
)


# Convert numeric back to categorical string
######################################### DO NOT DELETE #########################################
# zone_string = bec['ZONE_categories'][bec['ZONE'].astype(int)].drop('ZONE_categories')
# bec['ZONE'] = zone_string


# Create pseudo-absences
pres_abs = species.add_pseudo_absences(amount=species.shape[0], region_poly=bec_tmp)


BUFFER_DISTANCE = bec.rio.resolution()[1] * 31.5  # in units of CRS


# Given a list of raster tiles, find which ones intersect the species occurrence points and are therefore required
# Using a single raster, bec, for simplicity
rasters = pres_abs.list_rasters(BUFFER_DISTANCE, [bec])


# Load the list of raster tiles into memory
# Would load the rasters here, but bec is already loaded for simplicity. Something like:
# rasters = [rioxarray.open_rasterio(x) for x in raster]
# merged_raster = merge_arrays(rasters)
merged_raster = bec


# For each occurrence point, build a 3D tensor
vals = pres_abs.extract_values(raster=merged_raster, distance=BUFFER_DISTANCE)
vals = np.concatenate(vals)

x_train = np.stack(
    [x["arr"].transpose() for x in vals]
)  # if None not in x['arr'] is not None and 'nodata' not in x['arr']], axis=0)
y_train = np.stack(
    [x["presence"] for x in vals]
)  # if x['arr'] is not None and 'nodata' not in x['arr']])

# Create keras Sequential model
model = tf.keras.models.Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(64, 64, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downsample each dimension by a factor of 2

model.add(Conv2D(32, (2, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(2))  # This should be the number of layers
model.add(Activation("softmax"))
# len(model.weights)


model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, batch_size=128, epochs=100)
