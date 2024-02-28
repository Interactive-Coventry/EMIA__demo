from datetime import timedelta
from os.path import join as pathjoin
import streamlit as st
from PIL import Image
import libs.foxutils.utils.core_utils as core_utils
from emia_utils.database_utils import check_connection, connect, USES_FIREBASE
from emia_utils.process_utils import prepare_features_for_vehicle_counts
from emia_utils import database_utils
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from libs.foxutils.utils.display_and_plot import plot_markers_on_map
from streamlit_folium import folium_static

from utils.configuration import DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE
from utils.map_utils import print_expressway_camera_locations

logger = core_utils.get_logger("app.common")
HISTORY_STEP = int(core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_time_step"])
HISTORY_STEP_UNIT = core_utils.settings["VEHICLE_FORECASTING"]["total_vehicles_prediction_model_time_step_unit"]

SHOW_ANOMALY_LABEL = False
SHOW_WEATHER_LABEL = False
SHOW_VEHICLE_FORECAST_GRAPH = True


def set_value(key, value, reset=False):
    st.session_state[key] = value
    if reset:
        logger.info(f'Reset so that {key}={st.session_state[key]}')
    else:
        logger.info(f'Set so that {key}={st.session_state[key]}')


def initialize_value(key, value):
    if key not in st.session_state:
        set_value(key, value, reset=False)


def initialize_session_state():
    initialize_value("is_running", False)
    initialize_value("conn", None)
    initialize_value("firebase_db", None)
    initialize_value("target_dashcam", None)
    initialize_value("target_expressway_camera", None)
    initialize_value("loop", None)


@st.cache_resource
def init_connection():
    import socket
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    logger.info(f"Your Computer Name is: {hostname}")
    logger.info(f"Your Computer IP Address is:{IPAddr}")

    conn_ = connect()
    logger.info(f"Streamlit connect: DB Experimental connection is {conn_}")
    check_connection(conn_)
    return conn_


if USES_FIREBASE:
    st.session_state.firebase_db = database_utils.init_firebase()
    logger.info(f"Firebase connect: Connecting to {st.session_state.firebase_db}")
else:
    st.session_state.conn = init_connection()


def append_weather_data_to_database(weather_df):
    if USES_FIREBASE:
        weather_df.reset_index(inplace=True, drop=False)
        row_dict = weather_df.iloc[0].to_dict()
        row_dict["id"] = core_utils.convert_datetime_to_string(row_dict["datetime"])
        row_dict["datetime"] = row_dict["id"]
        database_utils.insert_row_to_firebase(st.session_state.firebase_db, row_dict, "weather", "id")
    else:
        database_utils.append_df_to_table(weather_df, "weather", append_only_new=True, conn=st.session_state.conn)


def append_camera_location_data_to_database(location_df):
    if USES_FIREBASE:
        location_df.reset_index(inplace=True, drop=False)
        row_dict = location_df.iloc[0].to_dict()
        row_dict["datetime"] = core_utils.convert_datetime_to_string(row_dict["datetime"])
        database_utils.insert_row_to_firebase(st.session_state.firebase_db, row_dict, "dashcams", "id")
    else:
        database_utils.append_df_to_table(location_df, "dashcams", append_only_new=True,
                                          conn=st.session_state.conn, append_index=True)


def append_vehicle_counts_data_to_database(vehicle_counts_df):
    if USES_FIREBASE:
        vehicle_counts_df.reset_index(inplace=True, drop=False)
        row_dict = vehicle_counts_df.iloc[0].to_dict()
        row_dict["id"] = core_utils.convert_datetime_to_string(row_dict["datetime"])
        row_dict["datetime"] = row_dict["id"]
        try:
            database_utils.insert_row_to_firebase(st.session_state.firebase_db, row_dict, "vehicle_counts", "id")
        except AttributeError as e:
            logger.debug(f"Table vehicle_counts , row {row_dict}.")
            logger.error(f"AttributeError: {e}")
    else:
        database_utils.append_df_to_table(vehicle_counts_df, "vehicle_counts", append_only_new=True, conn=st.session_state.conn)


def read_vehicle_forecast_data_from_database(current_date, camera_id, history_length):
    batch_size = 32

    if USES_FIREBASE:
        current_date = core_utils.convert_datetime_to_string(current_date)
        params = [["datetime", "<=", current_date]]
        weather_data = st.session_state.firebase_db.collection("weather") \
            .where(filter=FieldFilter(params[0][0], params[0][1], params[0][2])) \
            .order_by("datetime", direction=firestore.Query.ASCENDING) \
            .limit_to_last(batch_size).get()
        df_weather = database_utils.collection_reference_to_dataframe(weather_data, is_list=True)
        df_weather.drop(columns=["id"], inplace=True)
        df_weather["datetime"] = [core_utils.convert_string_to_date(x) for x in df_weather["datetime"]]

        params = [["datetime", "<=", current_date],
                  ["camera_id", "==", str(camera_id)]]
        vehicle_counts_data = st.session_state.firebase_db.collection("vehicle_counts") \
            .where(filter=FieldFilter(params[0][0], params[0][1], params[0][2])) \
            .where(filter=FieldFilter(params[1][0], params[1][1], params[1][2])) \
            .order_by("datetime", direction=firestore.Query.ASCENDING) \
            .limit_to_last(batch_size).get()
        df_vehicles = database_utils.collection_reference_to_dataframe(vehicle_counts_data, is_list=True)
        df_vehicles.drop(columns=["id"], inplace=True)
        df_vehicles["datetime"] = [core_utils.convert_string_to_date(x) for x in df_vehicles["datetime"]]
    else:
        params = [["datetime", "<=", database_utils.enclose_in_quotes(current_date)]]
        fetch_top = "\nORDER BY datetime DESC\nFETCH FIRST " + str(batch_size) + " ROWS ONLY"
        params[-1].append(fetch_top)
        df_weather = database_utils.read_table_with_select("weather", params, conn=st.session_state.conn)

        params = [["datetime", "<=", database_utils.enclose_in_quotes(current_date), "AND"],
                  ["camera_id", "=", database_utils.enclose_in_quotes(str(camera_id))]]
        params[-1].append(fetch_top)
        df_vehicles = database_utils.read_table_with_select('vehicle_counts', params, conn=st.session_state.conn)

    latest_weather_info = df_weather.iloc[0].copy()
    df_features = prepare_features_for_vehicle_counts(df_vehicles, df_weather, dropna=True,
                                                      include_weather_description=True)
    df_features = df_features.iloc[-history_length:]
    #logger.debug(f"Recovered features for vehicle forecasting: {df_features}")
    return df_features, latest_weather_info


def get_target_image(camera_selection, image_file=None):
    if image_file is None:
        image_file = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)

    logger.debug(f"Reading image from {image_file}")
    img = Image.open(image_file)
    fig = print_expressway_camera_locations([camera_selection])
    logger.debug(f"Finished preparing preview.")
    return img, fig


def present_results(container_placeholder, outputs, forecast_step=HISTORY_STEP):
    if outputs is not None:
        with container_placeholder.container():
            st.markdown("### Results")
            st.markdown(f"##### Current Datetime: {outputs['target_datetime'].strftime('%Y/%m/%d, %H:%M:%S')} "
                        f"({core_utils.settings['RUN']['timezone']})")

            if outputs["location"] is not None:
                location = outputs["location"]
                col1, col2 = st.columns(2)
                with col1:
                    m = plot_markers_on_map(None, location, label_column="camera_id")
                    folium_static(m, height=200, width=200)
                    logger.debug(f"Finished plotting markers on map.")
                    logger.debug(f"Location: {outputs['location']}")
                with col2:
                    lat = location.iloc[0]["lat"]
                    lng = location.iloc[0]["lng"]
                    st.markdown(f"#### Camera Location:  \n\nlatitude: {lat}  \n\nlongitude: {lng}")

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("##### Object Detection:")
                st.image(outputs["vehicle_detection_img"], use_column_width=True)

                if SHOW_WEATHER_LABEL:
                    st.markdown("##### Weather Label:")
                    for k, v in outputs["weather_detection_label"].items():
                        st.progress(v, text=k)

            with col4:
                st.markdown("##### Anomaly Heatmap:")
                st.image(outputs["anomaly_detection_img"], use_column_width=True)

                if SHOW_ANOMALY_LABEL:
                    st.markdown("##### Anomaly Label:")
                    show_general_label = True
                    if show_general_label:
                        for k, v in outputs["anomaly_detection_label"].items():
                            tt = k.split("(")[0]
                            break
                        st.markdown(tt)
                    else:
                        for k, v in outputs["anomaly_detection_label"].items():
                            st.progress(v, text=k)

            st.markdown("##### Weather Information:")
            weather_info = outputs["weather_info"].transpose()
            st.dataframe(weather_info, use_container_width=True)

            st.markdown("##### Detected Vehicles:")
            vehicles_df = outputs["vehicle_detection_df"].copy()
            vehicles_df.drop(columns=["datetime"], inplace=True)
            st.dataframe(vehicles_df, use_container_width=True)

            st.markdown(f"##### Vehicle Forecasting (in the next {forecast_step} {HISTORY_STEP_UNIT}):")
            if SHOW_VEHICLE_FORECAST_GRAPH:
                try:
                    vf_df = outputs["vehicle_forecast"]["previous_counts"].copy()
                    vf_predictions = outputs["vehicle_forecast"]["predictions"]
                    vf_df = vf_df[["total_vehicles"]]
                    vf_df.insert(loc=len(vf_df.columns), column="Predicted Total Vehicles", value=[None] * len(vf_df))
                    vf_df.rename(columns={"total_vehicles": "Measured Total Vehicles"}, inplace=True)
                    pred_datetime = vf_df.index[-1] + timedelta(minutes=forecast_step)
                    if HISTORY_STEP_UNIT != "minutes":
                        raise NotImplementedError("Only minutes are supported for now.")
                    pred_value = round(vf_predictions[0])
                    target_val = vf_df.iloc[len(vf_df) - 1]
                    vf_df.loc[vf_df.index[len(vf_df) - 1]] = [target_val[0], target_val[0]]
                    vf_df.loc[pred_datetime, :] = [None, pred_value]
                    st.line_chart(vf_df, use_container_width=True)
                    #logger.debug(vf_df)
                except ValueError as e:
                    pass
                except TypeError as e:
                    logger.debug(f"TypeError: {e}")
                    logger.debug("No vehicle forecasting results available")
                    st.markdown(f"No vehicle forecasting results available, because no previous values are available. "
                                f"Wait so that at least {HISTORY_STEP} images are captured.")
            else:
                vf_df = outputs["vehicle_forecast"]["previous_counts"]
                vf_predictions = outputs["vehicle_forecast"]["predictions"]
                vehicle_prediction_str = "-Previous Counts: [" \
                                         + ", ".join([str(round(x)) for x in vf_df["total_vehicles"].values]) \
                                         + "]\n\n-Predicted Count (next 5min): " + str(round(vf_predictions[0]))

                st.markdown(vehicle_prediction_str)
    else:
        with container_placeholder.container():
            logger.info("No output results during testing.")
            st.markdown("No results available during testing.")
