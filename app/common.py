from os.path import join as pathjoin

import streamlit as st
from PIL import Image
import libs.foxutils.utils.core_utils as core_utils
from emia_utils.process_utils import prepare_features_for_vehicle_counts, minute_rounder
from emia_utils import database_utils

from utils.map_utils import print_expressway_camera_locations
from utils.settings import DEFAULT_IMAGE_FILE, DEFAULT_DATASET_DIR

import logging
logger = logging.getLogger("app.common")

conn = st.experimental_connection("postgresql", type="sql")

def append_weather_data_to_database(weather_df):
    if conn is not None:
        database_utils.append_df_to_table(weather_df, "weather", append_only_new=True)
    else:
        raise ValueError("No connection to database.")

def append_vehicle_counts_data_to_database(vehicle_counts_df):
    if conn is not None:
        database_utils.append_df_to_table(vehicle_counts_df, "vehicle_counts", append_only_new=True)
    else:
        raise ValueError("No connection to database.")

def read_vehicle_forecast_data_from_database(current_date, camera_id, history_length):
    batch_size = 32
    if conn is not None:
        params = [["datetime", "<=", database_utils.enclose_in_quotes(current_date)]]
        fetch_top = "\nORDER BY datetime DESC\nFETCH FIRST " + str(batch_size) + " ROWS ONLY"
        params[-1].append(fetch_top)
        df_weather = database_utils.read_table_with_select("weather", params, conn)

        params = [["datetime", "<=", database_utils.enclose_in_quotes(current_date), "AND"],
                  ["camera_id", "=", database_utils.enclose_in_quotes(str(camera_id))]]
        params[-1].append(fetch_top)
        df_vehicles = database_utils.read_table_with_select('vehicle_counts', params, conn)

        df_features = prepare_features_for_vehicle_counts(df_vehicles, df_weather, dropna=True,
                                                          include_weather_description=True)
        df_features = df_features.iloc[-history_length:]
        logger.debug(f"Recovered features for vehicle forecasting: {df_features}")
        return df_features

    else:
        raise ValueError("No connection to database.")


def get_target_image(camera_selection, image_file=None):
    if image_file is None:
        image_file = pathjoin(DEFAULT_DATASET_DIR, DEFAULT_IMAGE_FILE)

    logger.debug(f"Reading image from {image_file}")
    img = Image.open(image_file)
    fig = print_expressway_camera_locations([camera_selection])
    logger.debug(f"Finished preparing preview.")
    return img, fig


def present_results(container_placeholder, outputs):
    with container_placeholder.container():
        st.markdown("### Results")
        st.markdown(f"##### Current Datetime: {outputs['target_datetime'].strftime('%Y/%m/%d, %H:%M:%S')} "
                    f"({core_utils.settings['RUN']['timezone']})")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("##### Object Detection:")
            st.image(outputs["vehicle_detection_img"], use_column_width=True)

            st.markdown("##### Weather Label:")
            for k, v in outputs["weather_detection_label"].items():
                st.progress(v, text=k)

        with col4:
            st.markdown("##### Anomaly Heatmap:")
            st.image(outputs["anomaly_detection_img"], use_column_width=True)

            st.markdown("##### Anomaly Label:")
            for k, v in outputs["anomaly_detection_label"].items():
                st.progress(v, text=k)

        st.markdown("##### Detected Vehicles:")
        st.dataframe(outputs["vehicle_detection_df"], use_container_width=True)
        st.markdown("##### Vehicle Forecasting:")
        st.markdown(outputs["vehicle_forecast"])
