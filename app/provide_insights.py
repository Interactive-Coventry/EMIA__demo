import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import libs.foxutils.utils.core_utils as core_utils
logger = core_utils.get_logger("app.provide_insights")

from os.path import join as pathjoin
from os import sep
import json
import cv2
import pytz
import streamlit as st
import pandas as pd
from PIL import Image
import gc
from emia_utils.process_utils import make_vehicle_counts_df
from app.common import read_vehicle_forecast_data_from_database, append_vehicle_counts_data_to_database
from libs.foxutils.utils.train_functionalities import get_label_and_prob_string
from libs.foxutils.streams.stream_utils import LoadStreams
from libs.tools.object_detection import load_object_detection_model, detect_from_image
import utils.object_detection as od
import utils.weather_detection_utils as wd
import utils.anomaly_detection as ad
import utils.vehicle_forecasting as vf
from utils.configuration import DEFAULT_FILEPATH


DEVICE = core_utils.device
logger.info(f"Running on {DEVICE}")
MODELS_DIR = core_utils.models_dir
target_tz = pytz.timezone(core_utils.settings["RUN"]["timezone"])
HISTORY_LENGTH = int(core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_history_length"])
HISTORY_STEP = int(core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_time_step"])
HISTORY_STEP_UNIT = core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_time_step_unit"]
HAS_IMAGE_HISTORY = bool(eval(core_utils.settings["HISTORY"]["has_image_history"]))
RUN_PER_FRAME = True
IS_TEST = bool(eval(core_utils.settings["RUN"]["is_test"]))
logger.debug(f"Is test: {IS_TEST}")
logger.debug(f"Has image history: {HAS_IMAGE_HISTORY}")

#############Load models#####
if IS_TEST:
    vf_model, vf_scaler = None, None
    weather_class_model, weather_class_model_name = None, None
    ad_model, ad_tfms, ad_config = None, None, None
    ad_trainer = None
    od_model, od_opt = None, None
else:
    logger.info("\n\n------------------Load Models------------------")
    vf_model, vf_scaler = vf.load_vehicle_forecasting_model()
    weather_class_model, weather_class_model_name = wd.load_weather_detection_model()
    ad_model, ad_tfms, ad_config = ad.load_anomaly_detection_model(
        device=DEVICE, set_up_trainer=not RUN_PER_FRAME)
    if not RUN_PER_FRAME:
        ad_trainer = ad_tfms
    od_model, od_opt = load_object_detection_model(save_img=True, save_txt=True,
                                                                                        device=DEVICE)
    od_model.eval()

logger.info("\n\n------------------Finished Loading Models------------------")


def split_filename_folder(filename):
    image_file = filename.split(sep)[-1]
    folder = filename.split(sep)[-2]
    filepath = sep.join(filename.split(sep)[:-1])
    filepath = pathjoin(filepath, "")
    return image_file, folder, filepath


def get_target_datetime(filename):
    current_datetime_str = filename.split("_")[-1].replace(".jpg", "")
    current_datetime = core_utils.convert_fully_connected_string_to_datetime(current_datetime_str)
    return current_datetime


def get_relevant_data(full_filename, delete_previous_results=False, history_length=HISTORY_LENGTH):
    logger.info("\n\n------------------Start processing------------------")

    image_file, folder, filepath = split_filename_folder(full_filename)
    logger.debug(f"Reading image from {full_filename}.")
    logger.debug(f"Path: {filepath}\nFolder: {folder}")

    file_list = core_utils.find_files_by_extension(filepath, ".jpg", ascending=False)
    target_id = file_list.index(image_file)
    start_id = target_id + 1
    end_id = start_id + history_length
    if len(file_list) >= end_id:
        target_files = file_list[start_id:end_id]
        logger.debug(f"For target file {image_file}, the following files are recovered:\n{target_files}")
    else:
        target_files = []
        logger.warning(f"Not enough images are available for prediction with lookback {history_length}")

    current_datetime = get_target_datetime(image_file)
    logger.debug(f"Current datetime {current_datetime}")

    return image_file, folder, filepath, target_files


def process_frame(img_dict, device, camera_id=None):
    if isinstance(img_dict, dict):
        target_datetime = img_dict["datetime"]
        cv2_img = img_dict["image"]
    else:
        target_datetime = core_utils.get_current_datetime(tz=target_tz)
        cv2_img = img_dict

    target_datetime = target_datetime.replace(tzinfo=None)
    logger.info(f"Target datetime: {target_datetime}")
    results_dict = {"input": {"datetime": target_datetime, "timezone": target_tz}}
    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    orig_dim = pil_img.size  # (width, height)
    logger.debug(f"Original image size: {orig_dim}")

    if IS_TEST:
        logger.info("Test mode is on. Skipping model inference.")
    else:
        logger.debug("------------------Run object detection------------------")
        od_img, od_dict = detect_from_image(cv2_img, od_model, od_opt, device)
        od_dfs = od.post_process_detect_vehicles(class_dict_list=[od_dict])
        od_row = pd.DataFrame(od_dfs.iloc[-1]).T
        od_row["datetime"] = [target_datetime]
        od_row["camera_id"] = [camera_id]
        vc_df = pd.DataFrame(make_vehicle_counts_df(od_row.iloc[0].to_dict()), index=[0])
        vc_df.set_index("datetime", inplace=True, drop=True)
        append_vehicle_counts_data_to_database(vc_df)
        results_dict["object_detection"] = od.set_results(od_img, od_row)

        logger.debug("------------------Run weather detection------------------")
        wd_label, wd_prob, _ = wd.predict(pil_img, weather_class_model, wd.weather_classes, weather_class_model_name)

        logger.debug(f"Weather detection predictions: {wd_label}, {wd_prob}")
        results_dict["weather_detection"] = wd.set_results(wd_label, wd_prob)

        logger.debug("------------------Run anomaly detection------------------")
        ad_result = ad.infer_from_image(cv2_img, ad_model, device, ad_tfms)
        results_dict["anomaly_detection"] = ad.set_results(cv2_img, ad_result, orig_dim)
        # image_utils.write_image("heatmap.jpg", ad_config.project.path, results_dict["anomaly_detection"]["heat_map_image"],
        #                        ad.HEATMAP_FOLDER_NAME)
        logger.debug(f"Anomaly detection predictions: {ad_result['pred_labels']}, {ad_result['pred_scores']}")

        logger.debug("------------------Run vehicle flow prediction------------------")
        vf_feature_df, weather_info = read_vehicle_forecast_data_from_database(target_datetime, camera_id, HISTORY_LENGTH)
        if len(vf_feature_df) < HISTORY_LENGTH:
            logger.debug(f"Not enough history values for calculation")
            results_dict["vehicle_forecasting"] = vf.set_results(pd.DataFrame({"total_vehicles": [0]}), [0])
        else:
            vf_predictions = vf.forecast_vehicles(vf_model, vf_scaler, vf_feature_df,
                                                  HISTORY_LENGTH)
            results_dict["vehicle_forecasting"] = vf.set_results(vf_feature_df, vf_predictions)
            logger.debug(f"Predicted num of vehicles in the next time step: {vf_predictions[0]:.2f}")

        results_dict["weather_info"] = weather_info

    return results_dict


def get_processing_results(img, camera_id=None):
    if camera_id is None:
        camera_id = 'NA'

    results = process_frame(img, device=DEVICE, camera_id=camera_id)

    if IS_TEST:
        return None
    else:
        vf_df = results["vehicle_forecasting"]["dataframe"]
        vf_predictions = results["vehicle_forecasting"]["predictions"]

        wd_label_str = get_label_and_prob_string(results["weather_detection"]["label"],
                                                 results["weather_detection"]["prob"])
        ad_label_str = get_label_and_prob_string(results["anomaly_detection"]["label"],
                                                 results["anomaly_detection"]["prob"])
        weather_info = results["weather_info"]
        weather_info = weather_info[["weather", "description", "temp", "feels_like", "pressure", "humidity", "wind_speed",
                                     "clouds_all"]]
        weather_info.rename({"weather": "Weather", "description": "Description", "temp": "Temp",
                             "feels_like": "FeelsLike", "pressure": "Pressure", "humidity": "Humidity",
                             "wind_speed": "WindSpeed", "clouds_all": "CloudCoverage"}, inplace=True)
        weather_info = weather_info.to_frame()

        outputs = {
            "target_datetime": results["input"]["datetime"],
            "vehicle_detection_img": results["object_detection"]["image"],
            "vehicle_detection_df": results["object_detection"]["dataframe"],
            "weather_detection_label": {wd_label_str: results["weather_detection"]["prob"]},
            "anomaly_detection_label": {ad_label_str: results["anomaly_detection"]["prob"]},
            "anomaly_detection_img": results["anomaly_detection"]["heat_map"],
            "vehicle_forecast": {"previous_counts": vf_df, "predictions": vf_predictions},
            "weather_info": weather_info,
        }

        return outputs


def get_insights(mode="files", **kwargs):
    if mode == "files":
        full_filename = kwargs["full_filename"]
        image_file, folder, _ = split_filename_folder(full_filename)

        current_datetime = get_target_datetime(image_file)
        img = {"image": cv2.imread(full_filename), "datetime": current_datetime}
        image_file, folder, filepath, target_files = get_relevant_data(full_filename,
                                                                       delete_previous_results=False,
                                                                       history_length=HISTORY_LENGTH)

        return get_processing_results(img, camera_id=folder)

    elif mode == "stream":
        stream_url = kwargs["stream_url"]
        stream_name = kwargs["stream_name"]
        present_results_func = kwargs["present_results_func"]
        update_every_n_frames = kwargs["update_every_n_frames"]

        dataset = LoadStreams(stream_url, custom_fps=None, use_oauth=False, allow_oauth_cache=False)

        st_frame = st.empty()
        container_placeholder = st.empty()
        count = 0
        for image in dataset:
            count = count + 1
            if image is not None:
                st_frame.image(image,
                               caption="Detected Video",
                               channels="BGR",
                               use_column_width=True
                               )
                if count == update_every_n_frames:
                    count = 0
                    current_time = core_utils.get_current_datetime(tz=target_tz)
                    img = {"image": image, "datetime": current_time}

                    results = get_processing_results(img, stream_name)
                    present_results_func(container_placeholder, results)
                    gc.collect()

            if not st.session_state.is_running:
                logger.info(f"Stop button pressed.")
                break
            if dataset.terminated:
                logger.info(f"Stream terminated.")
                break
        container_placeholder.empty()
        st_frame.caption("Finished processing stream.")
        logger.info(f"Finished processing stream.")
        return


############################## Test ######################################
def test_analyze():
    target_dir = DEFAULT_FILEPATH
    print(f"Test for {target_dir}")
    raise NotImplementedError


def print_results(results_dict):
    print("Input:")
    print(json.dumps(results_dict["input"], indent=4, sort_keys=False))
    print("Object Detection")
    print(results_dict["object_detection"]["dataframe"])
    print("Weather Detection")
    print(json.dumps(results_dict["weather_detection"], indent=4, sort_keys=False))
    print("Anomaly Detection")
    an_dict = {key: results_dict["anomaly_detection"][key] for key in ["label", "prob"]}
    print(json.dumps(an_dict, indent=4, sort_keys=False))
    print("Vehicle Flow Prediction")
    print(results_dict["vehicle_forecasting"]["predictions"])
