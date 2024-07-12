from os import sep
from os.path import join as pathjoin

import streamlit as st

st.set_page_config(page_title="Expressway Cameras", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

from schedule import every, repeat, run_pending
from schedule import clear as clear_all_jobs
import time
from emia_utils import download_utils
from libs.foxutils.utils import core_utils
from utils.configuration import TRAFFIC_IMAGES_PATH
from utils.map_utils import get_expressway_camera_info
from utils.common import set_value, \
    setup_sidebar_info
from utils.provide_insights import HISTORY_STEP, fetch_current_data, get_insights_and_present_results

logger = core_utils.get_logger("page.static_camera_view")


def clear_jobs():
    clear_all_jobs()
    set_value("is_running", False, reset=True)
    logger.info(f"Terminated all schedulers.")


def setup_expressway_camera_view():
    st.markdown("### Input from Expressway CCTV Camera")
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
                get_insights_and_present_results(target_camera_id, savedir, preview_container_placeholder,
                                                 results_container_placeholder)
                run_info_text_placeholder.text("")

            job()
            while "is_running" in st.session_state and st.session_state.is_running:
                run_pending()
                time.sleep(1)


if __name__ == "__main__":
    setup_expressway_camera_view()
    setup_sidebar_info()
