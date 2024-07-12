from os.path import join as pathjoin

import streamlit as st
from matplotlib import pyplot as plt

st.set_page_config(page_title="Expressway Cameras", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

from emia_utils import database_utils
from libs.foxutils.utils import core_utils
from utils.configuration import EXPRESSWAY_CAMERA_IDS
from utils.map_utils import get_expressway_camera_info, print_expressway_camera_locations, assign_heatmap_colors
from utils.common import setup_sidebar_info
from utils.provide_insights import update_global_state, target_tz, update_traffic_stats

logger = core_utils.get_logger("page.overview_view")

available_cameras = [str(x) for x in get_expressway_camera_info()["CameraID"].values]
available_cameras = EXPRESSWAY_CAMERA_IDS

def setup_overview_view():
    st.markdown("### Overview")

    if st.button("Update traffic stats"):
        st.markdown("Updating traffic stats...")
        df_vehicles = update_traffic_stats(available_cameras)
        colors_vehicles = assign_heatmap_colors(df_vehicles["total_vehicles"].values)
        colors_pedestrians = assign_heatmap_colors(df_vehicles["total_pedestrians"].values)

        st.image("assets/maps/colormap.png", width=500)

        st.markdown("#### Vehicle Density")
        map_fig = print_expressway_camera_locations(available_cameras, colors_vehicles)
        st.pyplot(map_fig)

        st.markdown("#### Pedestrian Density")
        map_fig = print_expressway_camera_locations(available_cameras, colors_pedestrians)
        st.pyplot(map_fig)


if __name__ == "__main__":
    setup_overview_view()
    setup_sidebar_info()
