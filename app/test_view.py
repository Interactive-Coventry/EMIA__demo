import streamlit as st

from utils.configuration import EXPRESSWAY_CAMERA_IDS, DEFAULT_FILEPATH, DEFAULT_CAMERA_ID
from . import provide_insights
from .common import present_results, get_target_image

def setup_test_view():
    with st.form("test"):
        st.markdown("### Input from File")
        camera_selection = st.selectbox("Camera ID", EXPRESSWAY_CAMERA_IDS, disabled=True,
                                        index=EXPRESSWAY_CAMERA_IDS.index(DEFAULT_CAMERA_ID))
        st.text(f"Auto-select file {DEFAULT_FILEPATH}.")
        submit_btn = st.form_submit_button('Submit', type='secondary')

    preview_container_placeholder = st.empty()
    run_info_text_placeholder = st.empty()
    results_container_placeholder = st.empty()

    if submit_btn:
        preview_img, map_fig = get_target_image("Test", camera_selection)

        with preview_container_placeholder.container():
            st.markdown("### Preview")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Target image preview:")
                st.image(preview_img, use_column_width=True)
            with col2:
                st.markdown("##### Expressway camera locations:")
                st.pyplot(map_fig)

        run_info_text_placeholder.text("Processing...")
        outputs = provide_insights.get_insights(mode="files", full_filename=DEFAULT_FILEPATH)
        present_results(results_container_placeholder, outputs)
        run_info_text_placeholder.text("")
