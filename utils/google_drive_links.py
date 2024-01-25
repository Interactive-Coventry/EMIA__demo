from os import mkdir

from libs.foxutils.utils import core_utils
from libs.foxutils.utils.fetch_from_google_drive import load_weights_from_google_drive, fetch_h5_file_from_drive
from os.path import join as pathjoin, exists
import logging

logger = logging.getLogger("utils.google_drive_links")

export_link_parts = {
    "prefix": "https://drive.google.com/uc?id=",
    "suffix": "&export=download"
}

shared_folder_link = "https://drive.google.com/drive/folders/1DcHp05K-qK92GpM2788MqeV8-SQhyVxs?usp=drive_link"

ids = {
    "vehicle_prediction": {
    "cnn_weather_historystep5_v1.scaler.pkl": "1_jxsTwjlzHBvWt3ugeC-7YYnNHOw8z7Z",
    "cnn_weather_historystep5_v1.weights.h5": "1jAjh30HMXJqlh3amtxrEcWmXqV2P18tM"
},
    "anomaly_detection": {
        "ad_reverse_distillation.yaml": "1mPo0RsATQ8ebPT0r2buC0vnAQ4aeYAJh",
        "reverse_distillation_model-v3.ckpt": "1YVD5gFW_J1YmQCc_4YICWANrPV4an2us",
    },
    "anonymization": {"model":
        {
            "car_person":
                {
                    "ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03":
                        {
                            "saved_model": "1vpsH_H_u7_UDJeKqU79sYQly0ICvMOhb"
                        }
                },
            "face_license":
                {
                    "saved_model": "1oFmMZe5D2wIYAF5iNmJil0SfGV3FHtRx"
                }
        }

    },
    "weather_detection": {"resnet-18-v1_TL_pl.pts": "1MDkq9Ufhfu3C7mEpLvvoltcw8XTep7fY"},

    "yolov7\\weights": {
        "yolov7_training.pt": "1boQeWLjOknYC2XMpTjhAPt-Zgslc0VtX",
        "yolov7.pt": "1tU62PA2FdAQtN7rv_EmodvD7tWxodPqd"
    },
}

MODELS_DIR = core_utils.models_dir


def check_and_download_file(target_file, target_folder):
    checkpoint_path = pathjoin(MODELS_DIR, target_folder, target_file)
    if not exists(checkpoint_path):
        file_id = ids[target_folder][target_file]
        load_weights_from_google_drive(file_id, checkpoint_path)


def download_shared_folder():
    from gdown import download_folder
    if not exists(MODELS_DIR):
        mkdir(MODELS_DIR)
        download_folder(url=shared_folder_link, output=MODELS_DIR, quiet=False)
    else:
        logger.info(f"Folder {MODELS_DIR} already exists. Skipping download.")


def download_files():
    # Needs to setup credentials from Google Drive API
    # https://developers.google.com/drive/api/v3/quickstart/python
    # Use config_files/pydrive_settings.yaml
    # Download the file credentials.json.

    # Vehicle prediction
    vehicle_prediction_folder = core_utils.settings["VEHICLE_FORECASTING"]["vehicle_prediction_folder"]
    vehicle_pred_model_filepath = core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model"]
    check_and_download_file(vehicle_pred_model_filepath + ".scaler.pkl", vehicle_prediction_folder)

    target_file = vehicle_pred_model_filepath + ".weights.h5"
    checkpoint_path = pathjoin(MODELS_DIR, vehicle_prediction_folder, target_file)
    if not exists(checkpoint_path):
        file_id = ids[vehicle_prediction_folder][target_file]
        fetch_h5_file_from_drive(export_link_parts["prefix"] + file_id, checkpoint_path)

    # Weather detection
    weather_detection_folder = core_utils.settings["WEATHER_DETECTION"]["weather_detection_folder"]
    weather_detection_checkpoint_file = core_utils.settings["WEATHER_DETECTION"]["weather_detection_checkpoint_file"]
    check_and_download_file(weather_detection_checkpoint_file + ".pts", weather_detection_folder)


    # Anomaly detection
    anomaly_detection_folder = core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_folder"]
    anomaly_detection_checkpoint_file = core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_checkpoint_file"]
    anomaly_detection_config_file = core_utils.settings["ANOMALY_DETECTION"]["anomaly_detection_config_file"]

    check_and_download_file(anomaly_detection_config_file, anomaly_detection_folder)
    check_and_download_file(anomaly_detection_checkpoint_file, anomaly_detection_folder)


    # Object detection
    object_detection_folder = pathjoin(core_utils.settings["OBJECT_DETECTION"]["object_detection_folder"], "weights")
    object_detection_checkpoint_file = core_utils.settings["OBJECT_DETECTION"]["object_detection_model_weights"]
    object_detection_checkpoint_training_file = core_utils.settings["OBJECT_DETECTION"]["object_detection_model_training_weights"]
    check_and_download_file(object_detection_checkpoint_file, object_detection_folder)
    check_and_download_file(object_detection_checkpoint_training_file, object_detection_folder)

