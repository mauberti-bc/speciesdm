import geopandas as gpd
from rioxarray.merge import merge_arrays
from geocube.api.core import make_geocube
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers
from osgeo import gdal
from biogeodataframe import BioGeoDataFrame


def read_and_shuffle_data(species_file_path, N_SAMPLES):
    # Read species occurrence data and shuffle
    species_tmp = gpd.read_file(species_file_path)
    N = min(N_SAMPLES, species_tmp.shape[0])
    species_tmp = species_tmp.sample(N)
    return BioGeoDataFrame(species_tmp)


def clean_species_data(species):
    return species[(species["decimalLongitude"] != "")]


def convert_and_reproject(species, CRS):
    # Convert to BioGeoDataFrame, set geometry, and reproject
    species = species.set_geometry(
        gpd.points_from_xy(species["decimalLongitude"], species["decimalLatitude"]),
        crs=4326,
    ).to_crs(CRS)
    return BioGeoDataFrame(species)


def read_and_preprocess_bec(bec_file_path, CRS, RESOLUTION):
    # Read and preprocess biogeoclimatic zones
    bec_shp = gpd.read_file(bec_file_path).to_crs(CRS)[["ZONE", "SUBZONE", "geometry"]]
    categorical_enums = {
        "ZONE": bec_shp.ZONE.drop_duplicates().values.tolist(),
        "SUBZONE": bec_shp.SUBZONE.drop_duplicates().values.tolist(),
    }
    print("Making geocube!")
    bec = make_geocube(
        vector_data=bec_shp,
        resolution=(RESOLUTION, -RESOLUTION),
        categorical_enums=categorical_enums,
    )
    return bec, bec_shp


def create_pseudo_absences(species, bec_shp, MIN_DISTANCE):
    # Create pseudo-absences
    pres_abs = species.add_pseudo_absences(
        amount=species.shape[0], region_poly=bec_shp, not_within=MIN_DISTANCE
    )
    return pres_abs


def find_and_merge_rasters(pres_abs, bec):
    # Find intersecting rasters and merge
    # rasters_list = pres_abs.list_rasters(buffer_distance, rasters=[bec])
    # rasters = [rioxarray.open_rasterio(x) for x in rasters_list]
    # merged_raster = merge_arrays(rasters)
    merged_raster = bec
    return merged_raster


def extract_and_format_data(pres_abs, merged_raster, buffer_distance, n_cores):
    # For each occurrence point, build a 3D tensor by extracting values for each of the bands in the neighbourhood
    # Increase n_cores for parallel processing
    vals = pres_abs.extract_values(
        raster=merged_raster, distance=buffer_distance, n_cores=n_cores
    )

    # Concatenate only if vals was produced with n_cores > 1
    try:
        vals = np.concatenate(vals)
    except:
        pass

    # Reformat the extracted values for tensorflow to be able to interpret
    x_data = np.stack([x["arr"].transpose() for x in vals])
    y_data = np.stack([x["presence"] for x in vals])

    return x_data, y_data


def split_and_randomize_data(x_data, y_data):
    # Split the data into training and test categories with randomization
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


def build_and_train_model(x_train, y_train, x_test, y_test):
    model = models.Sequential()

    # Adjust input shape based on the dimensions of your satellite data
    input_shape = (64, 64, 2)

    # Increase model depth
    model.add(layers.Conv2D(32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer to transition from convolutional layers to dense layers
    model.add(layers.Flatten())

    # Use more units in dense layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))  # Adjust dropout rate

    # Adjust the number of output units based on your task
    model.add(layers.Dense(2, activation="softmax"))

    # Experiment with different optimizers
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model with data augmentation
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
    )

    return model


# Set parameters
CRS = "EPSG:3005"
RESOLUTION = 1000
N_SAMPLES = 5000
NEIGHBOURHOOD_SIZE = 64
MIN_DISTANCE = 5000  # Minimum distance for pseudo-absences


# Define file paths
species_file_path = "../data/black_bear_observations.csv"
bec_file_path = "../data/bec"


if __name__ == "__main__":
    species = read_and_shuffle_data(species_file_path, N_SAMPLES)
    species = clean_species_data(species)
    species = convert_and_reproject(species, CRS)

    bec, bec_shp = read_and_preprocess_bec(bec_file_path, CRS, RESOLUTION)

    pres_abs = create_pseudo_absences(species, bec_shp, MIN_DISTANCE)

    # Set the buffer distance required for the specified NEIGHBOURHOOD_SIZE
    buffer_distance = (NEIGHBOURHOOD_SIZE / 2) * bec.rio.resolution()[1]

    merged_raster = find_and_merge_rasters(pres_abs, bec)

    x_data, y_data = extract_and_format_data(
        pres_abs, merged_raster, buffer_distance, n_cores=8
    )

    x_train, x_test, y_train, y_test = split_and_randomize_data(x_data, y_data)

    model = build_and_train_model(x_train, y_train, x_test, y_test)

    model.save("../data/black_bear_cnn.keras")
