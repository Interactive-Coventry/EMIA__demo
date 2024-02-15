from os import sep
from os.path import join as pathjoin
import pandas as pd
import streamlit as st
from schedule import every, repeat, run_pending
from schedule import clear as clear_all_jobs
import time
from emia_utils import download_utils
from libs.foxutils.utils import core_utils
from emia_utils.process_utils import make_weather_df
from utils.configuration import TRAFFIC_IMAGES_PATH
from utils.map_utils import get_expressway_camera_info
from utils import provide_insights
from utils.common import get_target_image, present_results, append_weather_data_to_database, set_value
from utils.provide_insights import HISTORY_STEP, get_target_datetime

logger = core_utils.get_logger("app.static_camera_view")


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
    set_value("is_running", False, reset=True)
    logger.info(f"Terminated all schedulers.")


def setup_expressway_camera_view():
    st.markdown("### Input from Expressway Camera")
    available_cameras = [str(x) for x in get_expressway_camera_info()["CameraID"].values]
    default_index = available_cameras.index("1703")
    dashcam_source_btn = st.selectbox(label="Select input source", options=available_cameras,
                                      index=default_index, key="dashcam_source")
    exec_btn_placeholder = st.empty()

    if st.session_state.is_running:
        clear_jobs()

    if not st.session_state.is_running:
        if exec_btn_placeholder.button("Fetch latest", key="start_btn_static"):
            logger.debug("Start button clicked.")
            set_value("is_running", True)
            exec_btn_placeholder.button("Stop", key="stop_btn_static")

            st.session_state.target_expressway_camera = dashcam_source_btn
            target_camera_id = st.session_state.target_expressway_camera
            savedir = pathjoin(core_utils.datasets_dir, download_utils.DATAMALL_FOLDER,
                               TRAFFIC_IMAGES_PATH.replace("/", sep).replace("?", ""), target_camera_id, "")
            core_utils.mkdir_if_not_exist(savedir)

            preview_container_placeholder = st.empty()
            run_info_text_placeholder = st.empty()
            results_container_placeholder = st.empty()

            @repeat(every(HISTORY_STEP).minutes)
            def job():
                logger.debug(f"Running job for camera {target_camera_id}.")
                fetch_current_data(target_camera_id)
                run_info_text_placeholder.text("Processing...")
                run_process(target_camera_id, savedir, preview_container_placeholder, results_container_placeholder)
                run_info_text_placeholder.text("")

            try:
                job()
                while "is_running" in st.session_state and st.session_state.is_running:
                    run_pending()
                    time.sleep(1)
            except AttributeError as e:
                logger.info(f"AttributeError: {e}")


if __name__ == "__main__":
    setup_expressway_camera_view()
