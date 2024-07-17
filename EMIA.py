import time

from utils.configuration import DEFAULT_CAMERA_ID, camera_id_key_name

start_time = time.time()
import streamlit as st
from os.path import join as pathjoin

st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
from libs.foxutils.utils.core_utils import logger
from utils.common import initialize_session_state
from utils.common import get_expressway_camera_info_from_db

initialize_session_state()

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import utils.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()

end_time = time.time()
elapsed_time = end_time - start_time
logger.debug(f"Initialization elapsed time: {elapsed_time}seconds")

from libs.foxutils.utils import core_utils
from utils.map_utils import print_expressway_camera_locations
from utils.common import setup_sidebar_info
from utils.provide_insights import update_current_camera_state

logger = core_utils.get_logger("page.main")
camera_info = get_expressway_camera_info_from_db()
available_cameras = [str(x) for x in camera_info[camera_id_key_name].values]
map_fig = print_expressway_camera_locations(camera_info, available_cameras)
default_index = available_cameras.index(DEFAULT_CAMERA_ID)


def setup_overview():
    st.markdown("### Available Data Sources")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Camera locations")
        st.pyplot(map_fig)

    with col2:
        st.markdown("##### Target camera")
        dashcam_source_btn = st.selectbox(label="Select input source", options=available_cameras,
                                          index=default_index, key="dashcam_source")

    preview_container_placeholder = st.empty()
    run_info_text_placeholder = st.empty()
    results_container_placeholder = st.empty()

    target_camera_id = dashcam_source_btn
    target_camera_info = camera_info[camera_info[camera_id_key_name] == target_camera_id]
    update_current_camera_state(target_camera_id, run_info_text_placeholder, preview_container_placeholder,
                                results_container_placeholder, target_camera_info)


if __name__ == "__main__":
    setup_sidebar_info()
    setup_overview()
