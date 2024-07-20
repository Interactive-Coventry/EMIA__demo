from os.path import join as pathjoin

import pandas as pd
import streamlit as st

from utils.configuration import camera_id_key_name

st.set_page_config(page_title="Expressway Cameras", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

from libs.foxutils.utils import core_utils
from utils.map_utils import assign_heatmap_colors, print_camera_locations
from utils.common import setup_sidebar_info, get_camera_info_from_db
from utils.provide_insights import update_traffic_stats

logger = core_utils.get_logger("page.overview_view")
camera_info = get_camera_info_from_db()
camera_info.set_index(camera_id_key_name, inplace=True, drop=False)
camera_info["colors"] = [(0, 0, 0, 0) for x in range(len(camera_info))]
available_cameras = camera_info[camera_id_key_name].iloc[0:4]


def setup_overview_view():
    st.markdown("### Overview")
    message_placeholder = st.empty()

    if st.button("Update traffic stats"):
        message_placeholder.markdown("Updating traffic stats...")
        logger.debug(f"Available cameras: {available_cameras}")
        df_vehicles = update_traffic_stats(available_cameras)
        colors_vehicles = assign_heatmap_colors(df_vehicles["total_vehicles"].values)
        colors_pedestrians = assign_heatmap_colors(df_vehicles["total_pedestrians"].values)

        st.image("assets/maps/colormap.png", width=500)

        st.markdown("#### Vehicle Density")
        camera_info.loc[available_cameras, "colors"] = pd.Series(colors_vehicles, index=available_cameras.index)
        map_fig = print_camera_locations(camera_info, available_cameras.values, show_legend=False)
        st.pyplot(map_fig)

        st.markdown("#### Pedestrian Density")
        camera_info.loc[available_cameras, "colors"] = pd.Series(colors_pedestrians, index=available_cameras.index)
        map_fig = print_camera_locations(camera_info, available_cameras.values, show_legend=False)
        st.pyplot(map_fig)

        message_placeholder.markdown("Finished update!")


if __name__ == "__main__":
    setup_overview_view()
    setup_sidebar_info()
