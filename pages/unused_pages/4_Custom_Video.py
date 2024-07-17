from os.path import join as pathjoin
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from libs.foxutils.utils import core_utils
from libs.foxutils.utils.display_and_plot import plot_markers_on_map
from utils import provide_insights
from utils.configuration import camera_id_key_name, longitude_key_name, latitude_key_name
from utils.provide_insights import IS_TEST
from utils.common import present_results, set_value, setup_sidebar_info

st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

def setup_video_view():
    st.markdown("### Input from Video")

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
            df_coord = pd.DataFrame({camera_id_key_name: ["Camera 1 (SG)"], longitude_key_name: [103.8198], latitude_key_name: [1.300190]})

            if dashcam_source_btn == "Camera 1 (SG)":  # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=8mqSJLPvLWg"
                st.session_state.stream_name = "test_SG"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 60
                df_coord = pd.DataFrame({camera_id_key_name: ["Camera 1 (SG)"], longitude_key_name: [103.8198], latitude_key_name: [1.300190]})

            if dashcam_source_btn == "Camera 2 (GR)":  # Dashcam from greece
                dashcam_url = pathjoin(core_utils.datasets_dir, "test", "dashcam", "sources.txt")
                st.session_state.stream_name = "test_GR"
                st.session_state.stream_url = dashcam_url
                st.session_state.update_every_n_frames = 5
                df_coord = pd.DataFrame({camera_id_key_name: ["Camera 2 (GR)"], longitude_key_name: [23.405626], latitude_key_name: [38.037139]})

            if dashcam_source_btn == "Camera 3 (UK)":  # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=CZuZ4RXl4Wg"
                st.session_state.stream_name = "test_UK"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 20
                df_coord = pd.DataFrame({camera_id_key_name: ["Camera  3 (UK)"], longitude_key_name: [-1.141297], latitude_key_name: [52.601009]})

            if dashcam_source_btn == "Camera 4 (UK)": # Youtube link
                youtube_url = "https://www.youtube.com/watch?v=QI4_dGvZ5yE"
                st.session_state.stream_name = "test_UK2"
                st.session_state.stream_url = youtube_url
                st.session_state.update_every_n_frames = 60
                df_coord = pd.DataFrame({camera_id_key_name: ["Camera  4 (UK)"], longitude_key_name: [-0.075278], latitude_key_name: [51.505554]})

        with col2:
            if st.checkbox("Show camera locations on map", value=False, key="plot_markers_on_map"):
                center_coords = [df_coord.iloc[0][latitude_key_name], df_coord.iloc[0][longitude_key_name]]
                m = plot_markers_on_map(center_coords, df_coord, label_column=camera_id_key_name)
                st_data = st_folium(m, height=300)

        exec_btn_placeholder = st.empty()

        if not st.session_state.is_running:
            if exec_btn_placeholder.button("Fetch latest", key="start_btn_video"):
                set_value("is_running", True)
                if exec_btn_placeholder.button("Stop", key="stop_btn_video"):
                    set_value("is_running", False, reset=True)
                    exec_btn_placeholder.empty()

                provide_insights.get_insights(mode="stream",
                                              stream_url=st.session_state.stream_url,
                                              stream_name=st.session_state.stream_name,
                                              present_results_func=lambda x, y: present_results(x, y, forecast_step=1),
                                              update_every_n_frames=st.session_state.update_every_n_frames)


if __name__ == "__main__":
    setup_video_view()
    setup_sidebar_info()