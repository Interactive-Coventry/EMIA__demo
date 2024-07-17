import warnings

import numpy as np
from emia_utils import download_utils, database_utils
warnings.simplefilter(action='ignore', category=FutureWarning)
import libs.foxutils.utils.core_utils as core_utils

logger = core_utils.get_logger("emia.provide_insights")
DISABLE_PROSSECING = bool(eval(core_utils.settings["RUN"]["disable_processing"]))

from os.path import join as pathjoin
from os import sep
import json
import cv2
import pytz
import streamlit as st
import pandas as pd
from PIL import Image
import gc
import requests
from emia_utils.process_utils import make_vehicle_counts_df, make_weather_df
from utils.common import read_vehicle_forecast_data_from_database, append_vehicle_counts_data_to_database, \
    append_camera_location_data_to_database, append_weather_data_to_database, get_target_image, present_results, \
    get_target_camera_info
from libs.foxutils.utils.train_functionalities import get_label_and_prob_string
from libs.foxutils.streams.stream_utils import LoadStreams
from libs.tools.object_detection import load_object_detection_model, detect_from_image
import utils.object_detection as od
import utils.weather_detection_utils as wd
import utils.anomaly_detection as ad
import utils.vehicle_forecasting as vf
from utils.configuration import DEFAULT_FILEPATH, TRAFFIC_IMAGES_PATH, camera_id_key_name, latitude_key_name, \
    longitude_key_name

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

WEBSOCKET_SERVER_URL = core_utils.settings["SERVER"]["url"]
DATA_SERVER_PORT = int(core_utils.settings["SERVER"]["data_port"])
DATA_SERVER_FULL_URL = f"https://{WEBSOCKET_SERVER_URL}:{DATA_SERVER_PORT}"


def get_auth_token():
    uri = DATA_SERVER_FULL_URL + "/auth"
    st.session_state["dashcam_bearer_token"] = st.secrets["database"]["bearer_token"]
    headers = {
        'email': st.secrets["streaming"]["server"]["username"],
        'password': st.secrets["streaming"]["server"]["password"]
    }
    try:
        response = requests.get(uri, headers=headers)

        if response.status_code == 200:
            r = response.json()
            if "token" in r:
                st.session_state["dashcam_bearer_token"] = r["token"]
                logger.debug(f"Token for dashcam servers is {r['token']}")
        else:
            # Request failed
            logger.info(f"Dashcam authentication request failed with status code {response.status_code}")
            logger.info(response.text)  # Print the error message if any
            return None

    except requests.exceptions.SSLError as e:
        logger.info("Please check the SSL certificate of the server.")
        logger.error(f"SSL Error: {e}")
        return None


#############Load models#####
if IS_TEST or DISABLE_PROSSECING:
    logger.info("\n\n------------------Test Mode------------------")
    vf_model, vf_scaler = None, None
    weather_class_model, weather_class_model_name = None, None
    ad_model, ad_tfms, ad_config = None, None, None
    ad_trainer = None
    od_model, od_opt = None, None
else:
    get_auth_token()
    logger.info("\n\n------------------Load Models------------------")
    vf_model, vf_scaler = vf.load_vehicle_forecasting_model()
    weather_class_model, weather_class_model_name = wd.load_weather_detection_model()
    ad_model, ad_tfms, ad_config = ad.load_anomaly_detection_model(
        device=DEVICE, set_up_trainer=not RUN_PER_FRAME)
    if not RUN_PER_FRAME:
        ad_trainer = ad_tfms
    od_model, od_opt = load_object_detection_model(save_img=True, save_txt=True, device=DEVICE)
    od_model.eval()

logger.info("\n\n------------------Finished Loading Models------------------")


def split_filename_folder(filename):
    image_file = filename.split(sep)[-1]
    folder = filename.split(sep)[-2]
    filepath = sep.join(filename.split(sep)[:-1])
    filepath = pathjoin(filepath, "")
    return image_file, folder, filepath


def get_target_datetime(filename):
    filename = filename.replace(".png", "").replace(".jpg", "")
    current_datetime_str = filename.split("_")[-1]
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


def process_batch(img_list, device):
    db_results = pd.DataFrame()
    for img_dict in img_list:
        if isinstance(img_dict, dict):
            target_datetime = img_dict["datetime"]
            cv2_img = img_dict["image"]
            camera_id = img_dict[camera_id_key_name]
        else:
            raise NotImplementedError

        target_datetime = target_datetime.replace(tzinfo=None)
        logger.info(f"Target datetime: {target_datetime}")
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

        if IS_TEST or DISABLE_PROSSECING:
            logger.info("Test mode is on. Skipping model inference.")
        else:
            logger.debug("------------------Run object detection------------------")
            od_img, od_dict = detect_from_image(cv2_img.copy(), od_model, od_opt)
            od_dfs = od.post_process_detect_vehicles(class_dict_list=[od_dict])
            od_row = pd.DataFrame(od_dfs.iloc[-1]).T
            od_row["datetime"] = [target_datetime]
            od_row[camera_id_key_name] = [camera_id]
            vc_df = pd.DataFrame(make_vehicle_counts_df(od_row.iloc[0].to_dict()), index=[0])
            vc_df.set_index("datetime", inplace=True, drop=True)
            db_results = pd.concat([db_results, vc_df], axis=0)

            logger.debug("------------------Run weather detection------------------")
            wd_label, wd_prob, _ = wd.predict(pil_img, weather_class_model, wd.weather_classes, weather_class_model_name)

            logger.debug("------------------Run anomaly detection------------------")
            ad_result = ad.infer_from_image(cv2_img, ad_model, device, ad_tfms)

    append_vehicle_counts_data_to_database(db_results)


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

    if IS_TEST or DISABLE_PROSSECING:
        logger.info("Test mode is on. Skipping model inference.")
    else:
        logger.debug("------------------Run object detection------------------")
        od_img, od_dict = detect_from_image(cv2_img.copy(), od_model, od_opt)
        od_dfs = od.post_process_detect_vehicles(class_dict_list=[od_dict])
        od_row = pd.DataFrame(od_dfs.iloc[-1]).T
        od_row["datetime"] = [target_datetime]
        od_row[camera_id_key_name] = [camera_id]
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
        vf_feature_df, weather_info = read_vehicle_forecast_data_from_database(target_datetime, camera_id,
                                                                               HISTORY_LENGTH)
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


def get_dashcam_location(dashcam_id, current_datetime):
    uri = DATA_SERVER_FULL_URL + "/cars"
    if "dashcam_bearer_token" not in st.session_state:
        get_auth_token()
    bearer_token = st.session_state["dashcam_bearer_token"]
    # bearer_token = st.secrets["database"]["bearer_token"]
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(uri, headers=headers)

    if response.status_code == 200:
        cam_list = response.json()
        cam_info = [x for x in cam_list if x["_id"] == dashcam_id]
        if len(cam_info) == 0:
            logger.warning(f"Dashcam {dashcam_id} not found in the database.")
            return None
        location = cam_info[0]["location"]
        logger.debug(f"Location of dashcam {dashcam_id}: {location}")
        df_coord = pd.DataFrame({camera_id_key_name: [dashcam_id],
                                 latitude_key_name: [location["lat"]],
                                 longitude_key_name: [location["lng"]],
                                 "datetime": [current_datetime]})
        df_coord.set_index("datetime", inplace=True, drop=True)
        return df_coord
    else:
        # Request failed
        logger.debug(f"Dashcam location request failed with status code {response.status_code}")
        logger.debug(response.text)  # Print the error message if any
        return None


def get_processing_results(img, camera_id=None, get_location=False):
    if camera_id is None:
        camera_id = 'NA'

    results = process_frame(img, device=DEVICE, camera_id=camera_id)
    if isinstance(img, dict):
        target_datetime = img["datetime"]
    else:
        target_datetime = core_utils.get_current_datetime(tz=target_tz)

    if isinstance(get_location, pd.DataFrame):
        location = get_location

    elif isinstance(get_location, float):
        target_datetime = target_datetime.replace(tzinfo=None)
        location = get_dashcam_location(camera_id, current_datetime=target_datetime)
        if location is not None:
            append_camera_location_data_to_database(location)
        else:
            location = pd.DataFrame({camera_id_key_name: [camera_id], latitude_key_name: [37], longitude_key_name: [22],
                                     "datetime": [target_datetime]})
    else:
        location = None

    if IS_TEST or DISABLE_PROSSECING:
        return None
    else:
        vf_df = results["vehicle_forecasting"]["dataframe"]
        vf_predictions = results["vehicle_forecasting"]["predictions"]

        wd_label_str = get_label_and_prob_string(results["weather_detection"]["label"],
                                                 results["weather_detection"]["prob"])
        ad_label_str = get_label_and_prob_string(results["anomaly_detection"]["label"],
                                                 results["anomaly_detection"]["prob"])
        weather_info = results["weather_info"]
        weather_info = weather_info[
            ["weather", "description", "temp", "feels_like", "pressure", "humidity", "wind_speed",
             "clouds_all"]]
        weather_info.rename({"weather": "Weather", "description": "Description", "temp": "Temperature",
                             "feels_like": "Feels Like", "pressure": "Pressure", "humidity": "Humidity",
                             "wind_speed": "Wind Speed", "clouds_all": "Cloudiness"}, inplace=True)
        weather_info = weather_info.to_frame()

        weather_info = weather_info.transpose()
        weather_info.drop(columns=["Feels Like", "Weather"], inplace=True)
        weather_info["Temperature"] = np.trunc(weather_info["Temperature"] - 273.15).astype(str) + "°C"
        weather_info["Pressure"] = weather_info["Pressure"].astype(str) + " hPa"
        weather_info["Humidity"] = weather_info["Humidity"].astype(str) + "%"
        weather_info["Wind Speed"] = weather_info["Wind Speed"].astype(str) + " m/s"
        weather_info["Cloudiness"] = weather_info["Cloudiness"].astype(str) + "%"

        weather_info.index = ["Station 1" for x in weather_info.index]
        weather_info.index.name = "Location"

        vehicles_df = results["object_detection"]["dataframe"].copy()
        vehicles_df.drop(columns=["datetime", camera_id_key_name], inplace=True)
        if "person" in vehicles_df.columns:
            vehicles_df.drop(columns=["person"], inplace=True)
        vehicles_df.rename(columns={"total_vehicles": "Vehicles", "total_pedestrians": "Pedestrians",
                                    "car": "Car", "bus": "Bus", "truck": "Truck", "motorcycle": "Motorcycle",
                                    }, inplace=True)
        vehicles_df.set_index("Vehicles", inplace=True, drop=True)

        outputs = {
            "target_datetime": results["input"]["datetime"],
            "vehicle_detection_img": results["object_detection"]["image"],
            "vehicle_detection_df": vehicles_df,
            "weather_detection_label": {wd_label_str: results["weather_detection"]["prob"]},
            "anomaly_detection_label": {ad_label_str: results["anomaly_detection"]["prob"]},
            "anomaly_detection_img": results["anomaly_detection"]["heat_map"],
            "vehicle_forecast": {"previous_counts": vf_df, "predictions": vf_predictions},
            "weather_info": weather_info,
            "location": location,
        }

        return outputs


def get_insights(mode="files", **kwargs):
    if mode == "files":
        full_filename = kwargs["full_filename"]
        camera_id = kwargs[camera_id_key_name]
        get_location = kwargs["get_location"]
        image_file, folder, _ = split_filename_folder(full_filename)
        current_datetime = get_target_datetime(image_file)
        if camera_id is not None:
            folder = camera_id
        img = {"image": cv2.imread(full_filename), "datetime": current_datetime}
        return get_processing_results(img, camera_id=folder, get_location=get_location)

    elif mode == "image":
        image = kwargs["image"]
        current_datetime = kwargs["current_datetime"]
        camera_id = kwargs[camera_id_key_name]
        get_location = kwargs["get_location"]
        img = {"image": image, "datetime": current_datetime}
        return get_processing_results(img, camera_id=camera_id, get_location=get_location)

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

                    results = get_processing_results(img, camera_id=stream_name)
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


def process_dashcam_frame(img, dashcam_id, datadir, count):
    # logger.debug(f"Fetching img from dashcam {dashcam_id} at {datadir} at count {count}")
    current_datetime = core_utils.get_current_datetime(tz=target_tz)
    current_datetime = core_utils.convert_datetime_to_fully_connected_string(current_datetime)
    save_path = pathjoin(datadir, dashcam_id + "_" + current_datetime + ".png")
    img.save(save_path)
    logger.info(f"Saved video frame {count} at {save_path}")


def get_insights_and_present_results(target_camera_id, savedir, preview_container_placeholder,
                                     results_container_placeholder, target_camera_info):
    file_list = core_utils.find_files_by_extension(savedir, ".jpg", ascending=False)
    target_file = pathjoin(savedir, file_list[0])
    preview_img, map_fig = get_target_image(target_camera_info, target_camera_id, target_file)

    if IS_TEST:
        with preview_container_placeholder.container():
            st.markdown("### Preview")
            capture_time = get_target_datetime(file_list[0])
            st.markdown(f"Last capture time: {capture_time}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Target image preview")
                st.image(preview_img, use_column_width=True)
            with col2:
                st.markdown("##### Expressway CCTV locations")
                st.pyplot(map_fig)

    location = get_target_camera_info(target_camera_id)
    outputs = get_insights(mode="files", full_filename=target_file,
                                            camera_id=target_camera_id,
                                            get_location=location)
    present_results(results_container_placeholder, outputs)


def fetch_current_data(target_camera_id):
    download_utils.fetch_traffic_images_from_link(TRAFFIC_IMAGES_PATH, target_camera_id=[target_camera_id])
    weather_metrics_dict = download_utils.download_weather_info_from_openweather(download_utils.TARGET_CITY)
    weather_df = pd.DataFrame(make_weather_df(weather_metrics_dict), index=[0])
    weather_df.set_index(["datetime"], inplace=True, drop=True)
    append_weather_data_to_database(weather_df)


def fetch_current_camera_data(target_camera_id=None):
    download_utils.fetch_traffic_images_from_link(TRAFFIC_IMAGES_PATH, target_camera_id=target_camera_id)


def fetch_current_weather_data():
    weather_metrics_dict = download_utils.download_weather_info_from_openweather(download_utils.TARGET_CITY)
    weather_df = pd.DataFrame(make_weather_df(weather_metrics_dict), index=[0])
    weather_df.set_index(["datetime"], inplace=True, drop=True)
    append_weather_data_to_database(weather_df)


def update_current_camera_state(target_camera_id, run_info_text_placeholder, preview_container_placeholder,
                                results_container_placeholder, target_camera_info):
    run_info_text_placeholder.text("Processing...")

    fetch_current_weather_data()
    logger.debug(f"Updated weather data.")

    fetch_current_camera_data(target_camera_id)
    logger.debug(f"Updated data for camera {target_camera_id}")

    savedir = pathjoin(core_utils.datasets_dir, download_utils.DATAMALL_FOLDER,
                       TRAFFIC_IMAGES_PATH.replace("/", sep).replace("?", ""), target_camera_id, "")
    get_insights_and_present_results(target_camera_id, savedir, preview_container_placeholder,
                                     results_container_placeholder, target_camera_info)
    run_info_text_placeholder.text("")


def update_global_state(target_cameras):
    fetch_current_weather_data()
    logger.info(f"Updated weather data.")

    fetch_current_camera_data(target_cameras)
    logger.info(f"Updated data for all cameras.")

    target_images = []
    for target_camera_id in target_cameras:
        savedir = pathjoin(core_utils.datasets_dir, download_utils.DATAMALL_FOLDER,
                           TRAFFIC_IMAGES_PATH.replace("/", sep).replace("?", ""), target_camera_id, "")
        file_list = core_utils.find_files_by_extension(savedir, ".jpg", ascending=False)
        target_file = pathjoin(savedir, file_list[0])

        image_file, folder, _ = split_filename_folder(target_file)
        current_datetime = get_target_datetime(image_file)

        target_images.append({"image": cv2.imread(target_file), "datetime": current_datetime,
                              camera_id_key_name: target_camera_id})

    process_batch(target_images, DEVICE)


def update_traffic_stats(target_cameras):
    update_global_state(target_cameras)

    current_date = core_utils.get_current_datetime(tz=target_tz)
    batch_size = len(target_cameras)
    camera_str = ", ".join([database_utils.enclose_in_quotes(str(x)) for x in target_cameras])
    fetch_top = "\nORDER BY datetime DESC\nFETCH FIRST " + str(batch_size) + " ROWS ONLY"
    params = [["datetime", "<=", database_utils.enclose_in_quotes(current_date), "AND"],
              [camera_id_key_name, "in", " (" + camera_str + ")"]]
    params[-1].append(fetch_top)
    df_vehicles = database_utils.read_table_with_select('vehicle_counts', params, conn=st.session_state.conn)
    return df_vehicles


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
