from os import sep
from os.path import join as pathjoin

import pandas as pd
import streamlit as st
from schedule import every, repeat, run_pending
from schedule import clear as clear_all_jobs
import time

from emia_utils import download_utils
from libs.foxutils.utils import core_utils
from emia_utils.process_utils import make_weather_df, minute_rounder
from utils.map_utils import get_expressway_camera_info
from . import provide_insights
from .common import get_target_image, present_results, append_weather_data_to_database
from .provide_insights import HISTORY_STEP, get_target_datetime

import logging
logger = logging.getLogger("app.static_camera_view")

TRAFFIC_IMAGES_PATH = "ltaodataservice/Traffic-Imagesv2"


def fetch_current_data(target_camera_id):
    download_utils.fetch_traffic_images_from_link(TRAFFIC_IMAGES_PATH, target_camera_id=target_camera_id)
    weather_metrics_dict = download_utils.download_weather_info_from_openweather(download_utils.TARGET_CITY)
    weather_df = pd.DataFrame(make_weather_df(weather_metrics_dict), index=[0])
    weather_df.set_index(["datetime"], inplace=True, drop=True)
    append_weather_data_to_database(weather_df)


def run_process(target_camera_id, savedir, preview_container_placeholder, results_container_placeholder):
    file_list = core_utils.find_files_by_extension(savedir, ".jpg", ascending=False)
    target_file = pathjoin(savedir, file_list[0])
    preview_img, map_fig = get_target_image(target_camera_id, target_file)

    with preview_container_placeholder.container():
        st.markdown("### Preview")
        capture_time = get_target_datetime(file_list[0])
        st.markdown(f"Last capture time: {capture_time}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Target image preview:")
            st.image(preview_img, use_column_width=True)
        with col2:
            st.markdown("##### Expressway camera locations:")
            st.pyplot(map_fig)

    outputs = provide_insights.get_insights(mode="files", full_filename=target_file)
    present_results(results_container_placeholder, outputs)

def clear_jobs():
    clear_all_jobs()
    st.session_state.is_running = False
    logger.info(f"Terminated all schedulers.")


def setup_expressway_camera_view():
    st.markdown("### Input from Expressway Camera")
    available_cameras = [str(x) for x in get_expressway_camera_info()["CameraID"].values]
    default_index = available_cameras.index("1703")
    dashcam_source_btn = st.selectbox(label="Select input source", options=available_cameras,
                                      index=default_index, key="dashcam_source")

    if "is_running" not in st.session_state:
        clear_jobs()

    st.session_state.is_running = False

    exec_btn_placeholder = st.empty()

    if not st.session_state.is_running:
        if exec_btn_placeholder.button("Fetch latest", key="start_btn"):
            st.session_state.is_running = True
            if exec_btn_placeholder.button("Stop", key="stop_btn"):
                clear_jobs()

            target_camera_id = dashcam_source_btn
            savedir = pathjoin(core_utils.datasets_dir, download_utils.DATAMALL_FOLDER,
                               TRAFFIC_IMAGES_PATH.replace("/", sep).replace("?", ""), target_camera_id, "")
            core_utils.mkdir_if_not_exist(savedir)

            preview_container_placeholder = st.empty()
            run_info_text_placeholder = st.empty()
            results_container_placeholder = st.empty()

            @repeat(every(HISTORY_STEP).minutes)
            def job():
                fetch_current_data(target_camera_id)
                run_info_text_placeholder.text("Processing...")
                run_process(target_camera_id, savedir, preview_container_placeholder, results_container_placeholder)
                run_info_text_placeholder.text("")

            try:
                job()
                while st.session_state.is_running:
                    run_pending()
                    time.sleep(1)
            except AttributeError as e:# st.session_state has no attribute "is_running". Did you forget to initialize it?
                print(e)


