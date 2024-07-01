from os import sep
from os.path import join as pathjoin
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Expressway Cameras", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

from schedule import every, repeat, run_pending
from schedule import clear as clear_all_jobs
import time
from emia_utils import download_utils
from libs.foxutils.utils import core_utils
from emia_utils.process_utils import make_weather_df
from utils.configuration import TRAFFIC_IMAGES_PATH
from utils.map_utils import get_expressway_camera_info, print_expressway_camera_locations
from utils import provide_insights
from utils.common import get_target_image, present_results, append_weather_data_to_database, set_value, \
    setup_sidebar_info
from utils.provide_insights import HISTORY_STEP, get_target_datetime

logger = core_utils.get_logger("page.static_camera_view")


def clear_jobs():
    clear_all_jobs()
    set_value("is_running", False, reset=True)
    logger.info(f"Terminated all schedulers.")


def setup_overview():
    st.markdown("### Available Data Sources")
    available_cameras = [str(x) for x in get_expressway_camera_info()["CameraID"].values]

    map_fig = print_expressway_camera_locations(available_cameras)
    st.markdown("##### Camera locations:")
    st.pyplot(map_fig)

    default_index = available_cameras.index("1703")
    dashcam_source_btn = st.selectbox(label="Select input source", options=available_cameras,
                                      index=default_index, key="dashcam_source")
    exec_btn_placeholder = st.empty()

    if st.session_state.is_running:
        clear_jobs()




if __name__ == "__main__":
    setup_overview()
    setup_sidebar_info()
