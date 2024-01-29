from os.path import join as pathjoin

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from libs.foxutils.utils import core_utils
from libs.foxutils.utils.display_and_plot import plot_markers_on_map

from . import provide_insights
from .provide_insights import IS_TEST
from .common import present_results, on_start_button_click, reset_values


def setup_dashcam_view():
    reset_values()

    st.markdown("### Input from Dashcam")

    if IS_TEST:
        camera_choices = ["Camera 1", "Camera 2", "Camera 3"]
    else:
        camera_choices = ["Camera 1 (SG)", "Camera 2 (GR)", "Camera 3 (UK)", "Camera 4 (UK)"]

    if "stream_url" not in st.session_state:
        st.session_state.stream_url = ""

    if "update_every_n_frames" not in st.session_state:
        st.session_state.update_every_n_frames = None

    if "stream_name" not in st.session_state:
        st.session_state.stream_name = ""

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            dashcam_source_btn = st.radio("Select input source", camera_choices, index=0, key="dashcam_source")
            df_coord = pd.DataFrame({"ID": ["Camera 1 (SG)"], "Longitude": [103.8198], "Latitude": [1.300190]})

            if dashcam_source_btn == "Camera 1 (SG)":  # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=8mqSJLPvLWg"
                st.session_state.stream_name = "test_SG"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 60
                df_coord = pd.DataFrame({"ID": ["Camera 1 (SG)"], "Longitude": [103.8198], "Latitude": [1.300190]})

            if dashcam_source_btn == "Camera 2 (GR)":  # Dashcam from greece
                dashcam_url = pathjoin(core_utils.datasets_dir, "test", "dashcam", "sources.txt")
                st.session_state.stream_name = "test_GR"
                st.session_state.stream_url = dashcam_url
                st.session_state.update_every_n_frames = 5
                df_coord = pd.DataFrame({"ID": ["Camera 2 (GR)"], "Longitude": [23.405626], "Latitude": [38.037139]})

            if dashcam_source_btn == "Camera 3 (UK)":  # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=CZuZ4RXl4Wg"
                st.session_state.stream_name = "test_UK"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 20
                df_coord = pd.DataFrame({"ID": ["Camera  3 (UK)"], "Longitude": [-1.141297], "Latitude": [52.601009]})

            if dashcam_source_btn == "Camera 4 (UK)": # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=QI4_dGvZ5yE"
                st.session_state.stream_name = "test_UK2"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 60
                df_coord = pd.DataFrame({"ID": ["Camera  4 (UK)"], "Longitude": [-0.075278], "Latitude": [51.505554]})

        with col2:
            if st.checkbox("Show camera locations on map", value=False, key="plot_markers_on_map"):
                center_coords = [df_coord.iloc[0]["Latitude"], df_coord.iloc[0]["Longitude"]]
                m = plot_markers_on_map(center_coords, df_coord)
                st_data = st_folium(m, height=300)

        exec_btn_placeholder = st.empty()

        if not st.session_state.is_running:
            if exec_btn_placeholder.button("Fetch latest", key="start_btn_dashcam"):
                on_start_button_click(True)
                if exec_btn_placeholder.button("Stop", key="stop_btn_dashcam"):
                    reset_values()
                    exec_btn_placeholder.empty()

                provide_insights.get_insights(mode="stream",
                                              stream_url=st.session_state.stream_url,
                                              stream_name=st.session_state.stream_name,
                                              present_results_func=lambda x, y: present_results(x, y, forecast_step=1),
                                              update_every_n_frames=st.session_state.update_every_n_frames)


