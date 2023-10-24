from os.path import join as pathjoin
import os
from os import sep
import shutil
from os.path import exists
import json

import pandas as pd

import libs.foxutils.utils.core_utils as core_utils
import utils.object_detection as od
import utils.weather_detection_utils as wd
import utils.anomaly_detection as ad
import utils.vehicle_forecasting as vf

from PIL import Image
from utils.settings import DEFAULT_FILEPATH, OBJECT_DETECTION_DIR, DEFAULT_VEHICLE_FORECAST_FEATURES_DF

DEVICE = core_utils.device
print(f'Running on {DEVICE}')
MODELS_DIR = core_utils.models_dir
results_dir = 'runs'



#############Load models#####
print('\n\n------------------Load Models------------------')
vf_model, vf_scaler = vf.load_vehicle_forecasting_model()
weather_class_model, weather_class_model_name = wd.load_weather_detection_model()
ad_model, ad_trainer, ad_config = ad.load_anomaly_detection_model(device=DEVICE)
od_model, od_opt = od.load_object_detection_model(save_img=True, save_txt=True, device=DEVICE)
print('\n\n------------------Finished Loading Models------------------')


def split_filename_folder(filename):
    image_file = filename.split(sep)[-1]
    folder = filename.split(sep)[-2]
    filepath = sep.join(filename.split(sep)[:-1])
    filepath = pathjoin(filepath, '')
    return image_file, folder, filepath


def get_target_datetime(filename):
    current_datetime_str = filename.split('_')[-1].replace('.jpg', '')
    current_datetime = core_utils.convert_fully_connected_string_to_datetime(current_datetime_str)
    return current_datetime


def apply_weather_detection(image_path, image_filename):
    label, prob = wd.predict_weather_class(image_path, image_filename, weather_class_model, wd.weather_classes,
                                           weather_class_model_name)

    return label, prob


def set_results_input(image_file, filepath, camera_id):
    return {'filename': image_file, 'filepath': filepath, 'camera_id': camera_id}


def set_results_object_detection(vehicle_img, object_df):
    return {'image': vehicle_img, 'dataframe': object_df}


def set_results_weather_detection(label, prob):
    weather_label = '{label:<75} ({p:.2f}%)'.format(label=label, p=prob)
    return {'label': label, 'prob': prob / 100, 'label_string': weather_label}



def set_results_vehicle_forecasting(df, predictions):
    print(f'Predicted num of vehicles in the next time step: {predictions[0]:.2f}')
    return {'dataframe': df, 'predictions': predictions}


def get_relevant_data(full_filename, delete_previous_results=False, history_length=5):
    print('\n\n------------------Start processing------------------')

    image_file, folder, filepath = split_filename_folder(full_filename)
    if delete_previous_results and exists(results_dir):
        shutil.rmtree(results_dir)

    print(f'Reading image from {full_filename}.')
    print(f'Path: {filepath}\nFolder: {folder}')

    file_list = os.listdir(filepath)
    file_list = [x for x in file_list if '.jpg' in x]
    file_list.sort(reverse=True)
    target_id = file_list.index(image_file)
    start_id = target_id + 1
    end_id = start_id + history_length
    if len(file_list) >= end_id:
        target_files = file_list[start_id:end_id]
        print(f'For target file {image_file}, the following files are recovered:\n{target_files}')
    else:
        raise ValueError(f"Not enough images are available for prediction with lookback {history_length}")

    current_datetime = get_target_datetime(image_file)
    print(f'Current datetime {current_datetime}')

    return image_file, folder, filepath, target_files


def object_detection(image_file, filepath, target_files):
    target_filepaths = [pathjoin(filepath, x) for x in target_files] + [pathjoin(filepath, image_file)]
    object_df = od.detect_objects(od_model, od_opt, filepath, keep_all_detected_classes=True,
                                  file_list=target_filepaths)
    current_datetime = get_target_datetime(image_file)
    object_df = object_df[object_df['datetime'] == current_datetime]
    return object_df


def analyze(full_filename, delete_previous_results, history_length):
    image_file, folder, filepath, target_files = get_relevant_data(full_filename, delete_previous_results,
                                                                   history_length)

    img = Image.open(full_filename)
    orig_dim = img.size

    results_dict = {'input': set_results_input(image_file, filepath, folder)}

    print('\n\n------------------Run object detection------------------')
    object_df = object_detection(image_file, filepath, target_files)
    vehicle_img = Image.open(pathjoin(OBJECT_DETECTION_DIR, image_file))
    results_dict['object_detection'] = set_results_object_detection(vehicle_img, object_df)

    print('\n\n------------------Run weather detection------------------')
    label, prob = apply_weather_detection(filepath, image_file)
    results_dict['weather_detection'] = set_results_weather_detection(label, prob)

    print('\n\n------------------Run anomaly detection------------------')
    results = ad.infer(ad_model, ad_trainer, ad_config.dataset.image_size, full_filename)
    anomaly_img = Image.open(pathjoin(pathjoin(ad_config.project.path, ad.INFER_FOLDER_NAME), folder, image_file))
    results_dict['anomaly_detection'] = ad.set_results_anomaly_detection(full_filename, anomaly_img, results[0],
                                                                         ad_config.project.path, orig_dim)

    print('\n\n------------------Run vehicle flow prediction------------------')
    target_files = [x.replace('.jpg', '.csv') for x in target_files]
    #vf_feature_df = process_utils.fetch_features_for_vehicle_counts(OBJECT_DETECTION_DIR, include_weather=True,
    #                                                                explore_data=False,
    #                                                                keep_all_detected_classes=False, dropna=True,
    #                                                                include_weather_description=True,
    #                                                                target_files=target_files)
    #vf_feature_df.to_csv('vf_feature_df.csv')
    vf_feature_df = pd.read_csv(DEFAULT_VEHICLE_FORECAST_FEATURES_DF)
    vf_predictions = vf.forecast_vehicles(vf_model, vf_scaler, vf_feature_df, history_length)
    results_dict['vehicle_forecasting'] = set_results_vehicle_forecasting(vf_feature_df, vf_predictions)

    return results_dict


############################## Test ######################################
def test_analyze():
    target_dir = DEFAULT_FILEPATH
    print(f'Test for {target_dir}')
    results_dict = analyze(target_dir, delete_previous_results=False, history_length=5)
    print_results(results_dict)
    return results_dict


def print_results(results_dict):
    print('Input:')
    print(json.dumps(results_dict['input'], indent=4, sort_keys=False))
    print('Object Detection')
    print(results_dict['object_detection']['dataframe'])
    print('Weather Detection')
    print(json.dumps(results_dict['weather_detection'], indent=4, sort_keys=False))
    print('Anomaly Detection')
    an_dict = {key: results_dict['anomaly_detection'][key] for key in ['label', 'prob']}
    print(json.dumps(an_dict, indent=4, sort_keys=False))
    print('Vehicle Flow Prediction')
    print(results_dict['vehicle_forecasting']['predictions'])
