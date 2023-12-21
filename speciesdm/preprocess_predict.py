from utils import extract_subarrays_from_raster
from keras.models import load_model
import numpy as np
import xarray
import zarr
from train import (
    raster_file_path,
    CRS,
    NEIGHBOURHOOD_SIZE,
    raster_file_path,
    N_LAYERS,
    load_rasters,
)


def load_and_preprocess_model(
    model_path, raster_file_path, CRS, NEIGHBOURHOOD_SIZE, n_layers
):
    # Load model
    model = load_model(model_path)

    # # Read and preprocess raster data
    # raster, raster_shp = read_and_preprocess_raster(
    #     raster_file_path=raster_file_path, CRS=CRS, RESOLUTION=RESOLUTION
    # )
    raster = load_rasters(raster_file_path)

    # Get shape of the original array
    reshape_size = (len(raster.x.values), len(raster.y.values))

    # Transpose raster data
    # transposed_raster = xarray.Dataset.to_array(raster).transpose()
    transposed_raster = raster.transpose()

    print(transposed_raster)

    # Extract subarrays from raster
    input_data = extract_subarrays_from_raster(
        raster=transposed_raster, radius=int(NEIGHBOURHOOD_SIZE / 2)
    )

    # Create mask for valid input data
    mask = np.array(
        [
            True
            if x.shape == (NEIGHBOURHOOD_SIZE, NEIGHBOURHOOD_SIZE, n_layers)
            else False
            for x in input_data
        ]
    )

    # Filter input data based on the mask
    input_data_list = [x for x, m in zip(input_data, mask) if m]

    return model, input_data_list, mask, reshape_size


def batchwise_predict(model, input_data_list, batch_size):
    predictions = []

    # Iterate through the input data in chunks
    for i in range(0, len(input_data_list), batch_size):
        print(f"Batch {i//batch_size + 1}/{len(input_data_list)//batch_size + 1}")
        batch_input = np.array(input_data_list[i : i + batch_size])
        batch_predictions = model.predict(batch_input)
        predictions.extend(batch_predictions)

    # Convert the predictions list to a NumPy array
    predictions_array = np.array(predictions)

    return predictions_array


def postprocess_predictions(predictions_array, mask):
    nl = []
    arr = predictions_array[:, 0]
    counter = 0

    for idx in range(len(mask)):
        if mask[idx]:
            val = arr[counter]
            counter += 1
            nl.append(val)
        else:
            nl.append(np.nan)

    pred_array = np.array(nl)
    return pred_array


def reshape_predictions(predictions_array, reshape_to):
    return predictions_array.reshape((reshape_to))


def save_predictions_to_zarr(predictions, output_path):
    zarr.save(output_path, predictions)


model_path = "../data/black_bear_cnn.keras"
output_zarr_path = "../data/prediction.zarr"
BATCH_SIZE = 1000


if __name__ == "__main__":
    model, input_data_list, mask, reshape_size = load_and_preprocess_model(
        model_path, raster_file_path, CRS, NEIGHBOURHOOD_SIZE, N_LAYERS
    )

    predictions_array = batchwise_predict(model, input_data_list, BATCH_SIZE)

    postprocessed_predictions = postprocess_predictions(predictions_array, mask)

    reshaped_predictions = reshape_predictions(postprocessed_predictions, reshape_size)

    save_predictions_to_zarr(reshaped_predictions, output_zarr_path)
