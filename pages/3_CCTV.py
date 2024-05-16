import streamlit as st
from os.path import join as pathjoin
st.set_page_config(page_title="CCTV", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")
from utils import provide_insights
from utils.common import present_results, set_value, setup_sidebar_info


def setup_cctv_view():
    st.markdown("### Input from CCTV livestream")

    camera_choices = ["Camera 1 (JP)"]

    cctv_source_btn = st.radio("Select input source", camera_choices, index=0, key="cctv_source")

    if "stream_url" not in st.session_state:
        st.session_state.stream_url = ""

    if "update_every_n_frames" not in st.session_state:
        st.session_state.update_every_n_frames = None

    if "stream_name" not in st.session_state:
        st.session_state.stream_name = ""

    if cctv_source_btn == "Camera 1 (JP)": # Livestream cam from japan
        livecam_url = "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard"
        st.session_state.stream_name = "livecam"
        st.session_state.stream_url = livecam_url
        st.session_state.update_every_n_frames = 20

    exec_btn_placeholder = st.empty()

    if not st.session_state.is_running:
        if exec_btn_placeholder.button("Fetch latest", key="start_btn_cctv"):
            set_value("is_running", True)
            if exec_btn_placeholder.button("Stop", key="stop_btn_cctv"):
                set_value("is_running", False, reset=True)
                exec_btn_placeholder.empty()

            provide_insights.get_insights(mode="stream",
                                          stream_url=st.session_state.stream_url,
                                          stream_name=st.session_state.stream_name,
                                          present_results_func=present_results,
                                          update_every_n_frames=st.session_state.update_every_n_frames)


if __name__ == "__main__":
    setup_cctv_view()
    setup_sidebar_info()

