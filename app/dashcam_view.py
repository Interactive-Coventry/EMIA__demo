from os.path import join as pathjoin
import streamlit as st

from . import provide_insights
from .provide_insights import IS_TEST
from .common import present_results


def setup_dashcam_view():
    st.markdown("### Input from Dashcam")

    if IS_TEST:
        camera_choices = ["Camera 1", "Camera 2", "Camera 3"]
    else:
        camera_choices = ["Camera 1 (SG)", "Camera 2 (GR)", "Camera 3 (UK)"]

    dashcam_source_btn = st.radio("Select input source", camera_choices, index=0, key="dashcam_source")

    if "stream_url" not in st.session_state:
        st.session_state.stream_url = ""

    if "update_every_n_frames" not in st.session_state:
        st.session_state.update_every_n_frames = None

    if "stream_name" not in st.session_state:
        st.session_state.stream_name = ""

    if dashcam_source_btn == "Camera 1 (SG)": # Youtube link
        youtube_url = "https://www.youtube.com/watch?v=8mqSJLPvLWg"
        st.session_state.stream_name = "test_SG"
        st.session_state.stream_url = youtube_url
        st.session_state.update_every_n_frames = 60

    if dashcam_source_btn == "Camera 2 (GR)": # Dashcam from greece
        dashcam_url = pathjoin("data", "test", "dashcam", "sources.txt")
        st.session_state.stream_name = "test_GR"
        st.session_state.stream_url = dashcam_url
        st.session_state.update_every_n_frames = 5

    if dashcam_source_btn == "Camera 3 (UK)": # Youtube link
        youtube_url = "https://www.youtube.com/watch?v=CZuZ4RXl4Wg"
        st.session_state.stream_name = "test_UK"
        st.session_state.stream_url = youtube_url
        st.session_state.update_every_n_frames = 20

    exec_btn_placeholder = st.empty()

    if st.session_state.stream_url is not None and exec_btn_placeholder.button("Start", key="start_btn"):
        if exec_btn_placeholder.button("Stop", key="stop_btn"):
            st.stop()

        provide_insights.get_insights(mode="stream",
                                      stream_url=st.session_state.stream_url,
                                      stream_name=st.session_state.stream_name,
                                      present_results_func=present_results,
                                      update_every_n_frames=st.session_state.update_every_n_frames)
