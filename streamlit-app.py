import streamlit as st

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
import libs.foxutils.utils.core_utils as core_utils

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import app.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()

from app.dashcam_view import setup_dashcam_view
from app.stream_view import setup_stream_view
from app.cctv_view import setup_cctv_view
from app.test_view import setup_test_view
from app.static_camera_view import setup_expressway_camera_view
from app.provide_insights import IS_TEST

# Create the Streamlit app
st.sidebar.title("Digital Twin of Singapore")
st.sidebar.write("Describe the current situation on the road.")

if IS_TEST:
    menu_items = {"Test": "Test",
                  "Expressway Camera": "NotImplemented",
                  "Dashcam": "NotImplemented",
                  "Livestream": "Youtube, Livecam, etc.",
                  }
else:
    menu_items = {
        "Expressway Camera": "Fetch images from expressway cameras API",
        "Dashcam": "Fetch video stream from dashcam",
        "CCTV": "Fetch video stream from CCTV",
    }

main_menu_radio_btn = st.sidebar.radio("Select input source", menu_items.keys(), index=0, key="input_source", )


# captions=menu_items.values())

def reset_values():
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    st.session_state.is_running = False


if "is_running" not in st.session_state:
    st.session_state.is_running = False

if main_menu_radio_btn == "Test":
    reset_values()
    setup_test_view()

elif main_menu_radio_btn == "Expressway Camera":
    reset_values()
    setup_expressway_camera_view()

elif main_menu_radio_btn == "Dashcam":
    reset_values()
    setup_dashcam_view()

elif main_menu_radio_btn == "Livestream":
    reset_values()
    setup_stream_view()

elif main_menu_radio_btn == "CCTV":
    reset_values()
    setup_cctv_view()