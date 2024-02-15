import time
start_time = time.time()
import streamlit as st
from app.common import reset_values

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
from libs.foxutils.utils.core_utils import logger

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import app.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()

from app.dashcam_view import setup_dashcam_view
from app.cctv_view import setup_cctv_view
from app.test_view import setup_test_view
from app.static_camera_view import setup_expressway_camera_view
from app.video_view import setup_video_view

end_time = time.time()
elapsed_time = end_time - start_time
logger.debug(f"Initialization elapsed time: {elapsed_time}seconds")

reset_values()


def main():

    # Create the Streamlit app
    st.sidebar.title("Digital Twin of Singapore")
    st.sidebar.write("Describe the current situation on the road.")

    menu_items = {
        "Expressway Camera": "Fetch images from expressway cameras API",
        "Dashcam": "Fetch video stream from dashcam",
        "Video": "Fetch video stream from URL",
        "CCTV": "Fetch video stream from CCTV",
    }

    main_menu_radio_btn = st.sidebar.radio("Select input source", menu_items.keys(), index=0, key="input_source", )
    # captions=menu_items.values())

    if main_menu_radio_btn == "Test":
        setup_test_view()

    elif main_menu_radio_btn == "Expressway Camera":
        setup_expressway_camera_view()

    elif main_menu_radio_btn == "Dashcam":
        setup_dashcam_view()

    elif main_menu_radio_btn == "Video":
        setup_video_view()

    elif main_menu_radio_btn == "CCTV":
        setup_cctv_view()


if __name__ == "__main__":
    main()