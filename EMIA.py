import time

start_time = time.time()
import streamlit as st
from os.path import join as pathjoin
from utils.configuration import DEFAULT_CAMERA_ID, camera_id_key_name

st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
from libs.foxutils.utils import core_utils
import utils.common
time.sleep(0.1)
from utils.common import initialize_session_state, get_camera_info_from_db
from utils.map_utils import print_camera_locations

initialize_session_state()

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import utils.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()

end_time = time.time()
elapsed_time = end_time - start_time
logger = core_utils.get_logger("page.main")
logger.debug(f"Initialization elapsed time: {elapsed_time}seconds")

from utils.common import setup_sidebar_info
from utils.provide_insights import update_current_camera_state

if not st.session_state.active_connection:
    utils.common.run_init_conenction()

camera_info = get_camera_info_from_db()
available_cameras = camera_info[camera_id_key_name]
default_index = int(available_cameras[available_cameras == DEFAULT_CAMERA_ID].index[0])
camera_info["is_selected"] = [1 if x == DEFAULT_CAMERA_ID else 0 for x in camera_info[camera_id_key_name]]
map_fig = print_camera_locations(camera_info, available_cameras.values)



def setup_overview():
    st.markdown("### Available Data Sources")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Camera locations")
        if map_fig is not None:
            st.pyplot(map_fig)

    with col2:
        st.markdown("##### Target camera")
        source_btn = st.selectbox(label="Select input source", options=available_cameras, index=default_index,
                                  key="camera_source")

    preview_container_placeholder = st.empty()
    run_info_text_placeholder = st.empty()
    results_container_placeholder = st.empty()

    target_camera_id = source_btn
    target_camera_info = camera_info[camera_info[camera_id_key_name] == target_camera_id]
    if st.session_state.active_connection:
        update_current_camera_state(target_camera_id, run_info_text_placeholder, preview_container_placeholder,
                                    results_container_placeholder, target_camera_info)


if __name__ == "__main__":
    setup_sidebar_info()
    setup_overview()
