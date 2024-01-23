import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from os.path import join as pathjoin
from os.path import exists

import libs.foxutils.utils.data_generators as dgen
import libs.foxutils.utils.train_functionalities as trainfunc
import libs.foxutils.utils.core_utils as core_utils
import libs.foxutils.utils.keras_models as km
from utils.fetch_drive import fetch_h5_file_from_drive
import utils.google_drive_links as gdl
from urllib.request import urlopen

import logging
logger = logging.getLogger("utils.vehicle_forecasting")
MODELS_DIR = core_utils.models_dir


def load_vehicle_forecasting_model():
    vehicle_prediction_folder = core_utils.settings["VEHICLE_FORECASTING"]["vehicle_prediction_folder"]
    total_vehicles_prediction_model = core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model"]
    total_vehicles_prediction_model_type = core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_type"]
    total_vehicles_prediction_model_time_step = core_utils.settings["VEHICLE_FORECASTING"][
        "total_vehicles_prediction_model_time_step"]
    vehicle_pred_model_filepath = pathjoin(MODELS_DIR, vehicle_prediction_folder, total_vehicles_prediction_model)
    vehicle_pred_scaler = joblib.load(urlopen(gdl.links["vehicle_prediction"]["cnn_weather_historystep5_v1.scaler.pkl"]))

    vehicle_pred_model, descr = km.make_single_step_model(total_vehicles_prediction_model_type,
                                                          conv_width=int(total_vehicles_prediction_model_time_step))
    vehicle_pred_model.build(input_shape=(None, 5, 7))
    vehicle_pred_model.compile()
    if not exists(vehicle_pred_model_filepath + ".weights.h5"):
        fetch_h5_file_from_drive(gdl.links["vehicle_prediction"]["cnn_weather_historystep5_v1.weights.h5"],
                                 vehicle_pred_model_filepath + ".weights.h5")
    vehicle_pred_model.load_weights(vehicle_pred_model_filepath + ".weights.h5")
    logger.info(f"New vehicle prediction model loaded from {total_vehicles_prediction_model} on device {core_utils.device}. "
          f"Model type {type(vehicle_pred_model)}.")

    return vehicle_pred_model, vehicle_pred_scaler


BASE_COLUMNS = ["weather", "description", "weather_id", "temp", "feels_like", "pressure", "humidity", "wind_speed",
                "wind_deg", "clouds_all", "visibility", "bicycle", "bus", "car", "motorcycle", "person", "truck",
                "total_pedestrians", "total_vehicles"]

def forecast_vehicles(vehicle_pred_model, vehicle_pred_scaler, in_df, history_length):
    weather_columns = ["temp", "humidity", "wind_speed", "clouds_all", "visibility", "weather_id"]
    target_column = "total_vehicles"

    out_df = in_df.copy()
    out_df = out_df[BASE_COLUMNS]
    _, out_df = trainfunc.apply_scaling(out_df, vehicle_pred_scaler, has_fit=False)
    out_df = out_df[weather_columns + [target_column]]

    train_df = pd.DataFrame(columns=out_df.columns)
    val_df = pd.DataFrame(columns=out_df.columns)
    test_df = out_df

    batch_size = len(out_df) - history_length + 1
    conv_window_gen = dgen.conv_window_generator(train_df, val_df, test_df, target_column, history_length, batch_size)
    # print(f"\nWindow generator\n{conv_window_gen} \nfor historylength = {history_length}\n")

    conv_window_gen._example = (tf.convert_to_tensor([test_df.values]),
                                tf.convert_to_tensor([np.zeros((batch_size, 1))]))
    inputs, labels = conv_window_gen.example
    predictions = vehicle_pred_model(inputs)
    predictions = predictions[0, 0, :].numpy()
    predictions_df = pd.DataFrame({target_column: predictions})
    predictions = trainfunc.inverse_scaling(predictions_df, vehicle_pred_scaler)
    predictions = predictions[target_column].values

    return predictions


def set_results(df, predictions):
    return {"dataframe": df, "predictions": predictions}
