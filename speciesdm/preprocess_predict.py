# from utils import extract_subarrays_from_raster
# from keras.models import load_model
# import numpy as np
# import xarray
# import zarr
# from train import (
#     bec_file_path,
#     CRS,
#     RESOLUTION,
#     NEIGHBOURHOOD_SIZE,
#     read_and_preprocess_bec,
# )

# RESOLUTION = 1000

# model = load_model("../data/black_bear_cnn.keras")

# bec, bec_shp = read_and_preprocess_bec(
#     bec_file_path=bec_file_path, CRS=CRS, RESOLUTION=RESOLUTION
# )

# transposed_bec = xarray.Dataset.to_array(bec).transpose()

# input_data = extract_subarrays_from_raster(
#     raster=transposed_bec, radius=int(NEIGHBOURHOOD_SIZE / 2)
# )

# mask = [True if x.shape == (64, 64, 2) else False for x in input_data]

# input_data_list = [x for x in input_data if x.shape == (64, 64, 2)]

# batch_size = 10000  # Adjust the batch size based on your requirements
# predictions = []
# n_iterations = len(input_data_list) / batch_size

# # Iterate through the input data in chunks
# for i in range(0, len(input_data_list), batch_size):
#     print(f"{i/batch_size}/{n_iterations}")
#     batch_input = np.array(input_data_list[i : i + batch_size])
#     batch_predictions = model.predict(batch_input)
#     predictions.extend(batch_predictions)

# # Convert the predictions list to a NumPy array if needed
# predictions_array = np.array(predictions)


# nl = []
# arr = predictions_array[:, 0]
# counter = 0

# for idx in range(len(mask)):
#     if mask[idx] == True:
#         val = arr[counter]
#         counter += 1
#         nl.append(val)
#         print(arr)
#     elif mask[idx] == False:
#         nl.append(np.nan)

# pred_array = np.array(nl)

# zarr.save("../data/prediction.zarr", pred_array)

from utils import extract_subarrays_from_raster
from keras.models import load_model
import numpy as np
import xarray
import zarr
from train import (
    bec_file_path,
    CRS,
    RESOLUTION,
    NEIGHBOURHOOD_SIZE,
    read_and_preprocess_bec,
)


def load_and_preprocess_model(
    model_path, bec_file_path, CRS, RESOLUTION, NEIGHBOURHOOD_SIZE, batch_size=10000
):
    # Load model
    model = load_model(model_path)

    # Read and preprocess BEC data
    bec, bec_shp = read_and_preprocess_bec(
        bec_file_path=bec_file_path, CRS=CRS, RESOLUTION=RESOLUTION
    )

    # Transpose BEC data
    transposed_bec = xarray.Dataset.to_array(bec).transpose()

    # Extract subarrays from raster
    input_data = extract_subarrays_from_raster(
        raster=transposed_bec, radius=int(NEIGHBOURHOOD_SIZE / 2)
    )

    # Create mask for valid input data
    mask = np.array([True if x.shape == (64, 64, 2) else False for x in input_data])

    # Filter input data based on the mask
    input_data_list = [x for x, m in zip(input_data, mask) if m]

    return model, input_data_list, mask, batch_size


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


def save_predictions_to_zarr(predictions, output_path):
    zarr.save(output_path, predictions)


model_path = "../data/black_bear_cnn.keras"
output_zarr_path = "../data/prediction.zarr"


if __name__ == "__main__":
    model, input_data_list, mask, batch_size = load_and_preprocess_model(
        model_path, bec_file_path, CRS, RESOLUTION, NEIGHBOURHOOD_SIZE
    )

    predictions_array = batchwise_predict(model, input_data_list, batch_size)

    postprocessed_predictions = postprocess_predictions(predictions_array, mask)

    save_predictions_to_zarr(postprocessed_predictions, output_zarr_path)
