import geopandas as gpd
from rioxarray.merge import merge_arrays
from geocube.api.core import make_geocube
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers, optimizers
from osgeo import gdal
from biogeodataframe import BioGeoDataFrame
import rioxarray
import zarr


# Function to read and shuffle species occurrence data
def read_and_shuffle_data(species_file_path, N_SAMPLES):
    species_tmp = gpd.read_file(species_file_path)
    N = min(N_SAMPLES, species_tmp.shape[0])
    species_tmp = species_tmp.sample(N)
    return BioGeoDataFrame(species_tmp)


# Function to clean species data by removing rows with empty longitude
def clean_species_data(species, within):
    return species[(species["decimalLongitude"] != "")]


# Function to convert and reproject species data
def convert_and_reproject(species, CRS):
    species = species.set_geometry(
        gpd.points_from_xy(species["decimalLongitude"], species["decimalLatitude"]),
        crs=4326,
    ).to_crs(CRS)
    return BioGeoDataFrame(species)


def filter_within_bc(species, filter_within):
    return BioGeoDataFrame(species.sjoin(filter_within[["geometry"]], how="inner"))


# Function to read and preprocess biogeoclimatic zones
def read_and_preprocess_bec(raster_file_path, CRS, RESOLUTION):
    bec_shp = gpd.read_file(raster_file_path).to_crs(CRS)[
        ["ZONE", "SUBZONE", "geometry"]
    ]
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


# Function to create pseudo-absences
def create_pseudo_absences(species, bec_shp, MIN_DISTANCE):
    pres_abs = species.add_pseudo_absences(
        amount=species.shape[0], region_poly=bec_shp, not_within=MIN_DISTANCE
    )
    return pres_abs


# Function to load raster data
def load_rasters(raster):
    return rioxarray.open_rasterio(
        raster, chunks=True, cache=False, mask_and_scale=True
    )


# Function to extract and format data
def extract_and_format_data(
    pres_abs, merged_raster, buffer_distance, neighbourhood_size, n_cores
):
    vals = pres_abs.extract_values(
        raster=merged_raster,
        distance=buffer_distance,
        neighbourhood_size=neighbourhood_size,
        n_cores=n_cores,
    )

    print(vals)

    try:
        vals = np.concatenate(vals)
    except:
        pass

    x_data = np.stack([x["arr"].transpose() for x in vals])
    y_data = np.stack([x["presence"] for x in vals])

    return x_data, y_data


# Function to split and randomize data
def split_and_randomize_data(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )
    return x_train, x_test, y_train, y_test


# Function to standardize x_data
def standardize_x_data(data):
    mean_values = np.nanmean(data, axis=(0, 1, 2), keepdims=True)
    std_values = np.nanstd(data, axis=(0, 1, 2), keepdims=True)
    return (data - mean_values) / std_values


# Function to remove NaN values
def remove_nan_values(x_data, y_data):
    nan_rows = np.any(np.isnan(x_data), axis=(1, 2, 3))
    x_clean = x_data[~nan_rows]
    y_clean = y_data[~nan_rows]
    return x_clean, y_clean


# Function to build and train the model
def build_and_train_model(
    x_train,
    y_train,
    x_test,
    y_test,
    neighbourhood_size,
    n_layers,  # , learning_rate=0.01
):
    model = models.Sequential()

    input_shape = (neighbourhood_size, neighbourhood_size, n_layers)

    model.add(layers.Conv2D(128, (3, 3), activation="tanh", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="tanh"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="tanh"))
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer="adam",  # optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=16,
        validation_data=(x_test, y_test),
    )
    return model


# Set parameters
CRS = "EPSG:3005"
N_SAMPLES = 20000
NEIGHBOURHOOD_SIZE = 16
MIN_DISTANCE = 8000
N_LAYERS = 19

# Define file paths
species_file_path = "../data/strigidae.csv"  # "../data/black_bear_observations.csv"
raster_file_path = "../data/chelsa_tif_bcalbers.tif"
bcbounds_file_path = "../data/bec"

# Main execution block
if __name__ == "__main__":
    bc_shp = gpd.read_file(bcbounds_file_path).to_crs(CRS)
    species = read_and_shuffle_data(species_file_path, N_SAMPLES)
    species = clean_species_data(species, bc_shp)
    species = convert_and_reproject(species, CRS)
    species = filter_within_bc(species, bc_shp)
    pres_abs = create_pseudo_absences(species, bc_shp, MIN_DISTANCE)
    merged_raster = load_rasters(raster_file_path)
    buffer_distance = (NEIGHBOURHOOD_SIZE / 2) * abs(merged_raster.rio.resolution()[0])
    x_data, y_data = extract_and_format_data(
        pres_abs, merged_raster, buffer_distance, NEIGHBOURHOOD_SIZE, n_cores=8
    )
    x_data, y_data = remove_nan_values(x_data, y_data)
    x_data_st = standardize_x_data(x_data)
    x_train, x_test, y_train, y_test = split_and_randomize_data(x_data_st, y_data)
    model = build_and_train_model(
        x_train, y_train, x_test, y_test, NEIGHBOURHOOD_SIZE, N_LAYERS
    )
    # Evaluate the model on the test set
    results = model.evaluate(x_test, y_test, verbose=2)

    # Print the evaluation results
    print("Test Loss:", results[0])
    print("Test Accuracy:", results[1])
    model.save("../data/black_bear_cnn.keras")
