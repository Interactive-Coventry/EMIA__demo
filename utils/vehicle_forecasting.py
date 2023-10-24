import pandas as pd
import tensorflow as tf
import numpy as np
import joblib
from os.path import join as pathjoin

import libs.foxutils.utils.data_generators as dgen
import libs.foxutils.utils.train_functionalities as trainfunc
import libs.foxutils.utils.core_utils as core_utils
import libs.foxutils.utils.keras_models as km

MODELS_DIR = core_utils.models_dir


def load_vehicle_forecasting_model():
    vehicle_prediction_folder = core_utils.settings['MODELS']['vehicle_prediction_folder']
    total_vehicles_prediction_model = core_utils.settings['MODELS']['total_vehicles_prediction_model']
    total_vehicles_prediction_model_type = core_utils.settings['MODELS']['total_vehicles_prediction_model_type']
    total_vehicles_prediction_model_time_step = core_utils.settings['MODELS'][
        'total_vehicles_prediction_model_time_step']
    vehicle_pred_model_filepath = pathjoin(MODELS_DIR, vehicle_prediction_folder, total_vehicles_prediction_model)
    vehicle_pred_scaler = joblib.load(vehicle_pred_model_filepath + '.scaler.pkl')

    vehicle_pred_model, descr = km.make_single_step_model(total_vehicles_prediction_model_type,
                                                          conv_width=int(total_vehicles_prediction_model_time_step))
    vehicle_pred_model.build(input_shape=(None, 5, 7))
    vehicle_pred_model.compile()
    vehicle_pred_model.load_weights(vehicle_pred_model_filepath + '.weights.h5')
    print(f'New vehicle prediction model loaded from {total_vehicles_prediction_model}.')

    return vehicle_pred_model, vehicle_pred_scaler


def forecast_vehicles(vehicle_pred_model, vehicle_pred_scaler, in_df, history_length):
    weather_columns = ['temp', 'humidity', 'wind_speed', 'clouds_all', 'visibility', 'weather_id']
    target_column = 'total_vehicles'

    out_df = in_df.copy()
    _, out_df = trainfunc.apply_scaling(out_df, vehicle_pred_scaler, has_fit=False)
    out_df = out_df[weather_columns + [target_column]]

    train_df = pd.DataFrame(columns=out_df.columns)
    val_df = pd.DataFrame(columns=out_df.columns)
    test_df = out_df

    batch_size = len(out_df) - history_length + 1
    conv_window_gen = dgen.conv_window_generator(train_df, val_df, test_df, target_column, history_length, batch_size)
    print(f'\nWindow generator\n{conv_window_gen} \nfor historylength = {history_length}\n')

    conv_window_gen._example = (tf.convert_to_tensor([test_df.values]),
                                tf.convert_to_tensor([np.zeros((batch_size, 1))]))
    inputs, labels = conv_window_gen.example
    predictions = vehicle_pred_model(inputs)
    predictions = predictions[0, 0, :].numpy()
    predictions_df = pd.DataFrame({target_column: predictions})
    predictions = trainfunc.inverse_scaling(predictions_df, vehicle_pred_scaler)
    predictions = predictions[target_column].values

    return predictions
