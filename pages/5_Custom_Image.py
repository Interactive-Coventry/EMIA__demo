import time
from os.path import join as pathjoin
import streamlit as st
from libs.foxutils.utils import core_utils
from utils import provide_insights
from utils.common import present_results, setup_sidebar_info, set_value
from PIL import Image

st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

TARGET_FILE_EXTENSION = ".png"
DEFAULT_IMAGE_PATH = "D:\\data\\emia\\dashcam\\singapore\\2024-03-10_04-34-58\\"
CAMERA_ID = "custom"
logger = core_utils.get_logger("page.custom_image_view")


def setup_image_series_view():
    st.markdown("### Input from Image Series")

    image_series_path = st.text_input("Enter the path to the image series", key="image_series_path",
                                      value=DEFAULT_IMAGE_PATH)
    exec_btn_placeholder = st.empty()

    if st.session_state.is_running:
        set_value("is_running", False, reset=True)

    if not st.session_state.is_running:
        if exec_btn_placeholder.button("Submit", key="submit_image_series_btn"):
            logger.debug("Submit button clicked.")
            set_value("is_running", True)
            exec_btn_placeholder.button("Stop", key="stop_btn_static")

            files = core_utils.find_files_by_extension(image_series_path, TARGET_FILE_EXTENSION, ascending=True)
            if len(files) == 0:
                st.error("No images found in the directory.")
                return

            preview_placeholder = st.empty()
            results_container_placeholder = st.empty()

            for file in files:
                target_file = pathjoin(image_series_path, file)
                name = file
                img = Image.open(target_file)
                outputs = provide_insights.get_insights(mode="files", full_filename=target_file, camera_id=CAMERA_ID,
                                                        get_location=False)
                preview_placeholder.markdown("### Preview")
                preview_placeholder.image(img, width=500, caption=name.replace(TARGET_FILE_EXTENSION, ""))
                present_results(results_container_placeholder, outputs)
                time.sleep(2)


if __name__ == "__main__":
    setup_image_series_view()
    setup_sidebar_info()
