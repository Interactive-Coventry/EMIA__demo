from os.path import join as pathjoin

import streamlit as st

from utils.configuration import camera_id_key_name

st.set_page_config(page_title="Expressway Cameras", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

from libs.foxutils.utils import core_utils
from utils.map_utils import print_expressway_camera_locations, assign_heatmap_colors
from utils.common import setup_sidebar_info, get_expressway_camera_info_from_db
from utils.provide_insights import update_traffic_stats

logger = core_utils.get_logger("page.overview_view")
camera_info = get_expressway_camera_info_from_db()
available_cameras = [str(x) for x in camera_info[camera_id_key_name].values]
available_cameras = available_cameras[0:5]


def setup_overview_view():
    st.markdown("### Overview")
    message_placeholder = st.empty()

    if st.button("Update traffic stats"):
        message_placeholder.markdown("Updating traffic stats...")
        print(available_cameras)
        df_vehicles = update_traffic_stats(available_cameras)
        colors_vehicles = assign_heatmap_colors(df_vehicles["total_vehicles"].values)
        colors_pedestrians = assign_heatmap_colors(df_vehicles["total_pedestrians"].values)

        st.image("assets/maps/colormap.png", width=500)

        st.markdown("#### Vehicle Density")
        map_fig = print_expressway_camera_locations(camera_info, available_cameras, colors_vehicles)
        st.pyplot(map_fig)

        st.markdown("#### Pedestrian Density")
        map_fig = print_expressway_camera_locations(camera_info, available_cameras, colors_pedestrians)
        st.pyplot(map_fig)

        message_placeholder.markdown("Finished update!")

if __name__ == "__main__":
    setup_overview_view()
    setup_sidebar_info()
