import time

start_time = time.time()
import streamlit as st
from os.path import join as pathjoin

st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
from libs.foxutils.utils.core_utils import logger
from utils.common import initialize_session_state, setup_sidebar_info

initialize_session_state()


init_bar.progress(30, text="Initialization in progress. Downloading models. This will take some time.")
import utils.google_drive_links as gdl
#gdl.download_files()
gdl.download_shared_folder()

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import utils.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()

end_time = time.time()
elapsed_time = end_time - start_time
logger.debug(f"Initialization elapsed time: {elapsed_time}seconds")

st.markdown("# Digital Twin of Singapore")
st.markdown("## EMIA Dashboard")
st.markdown(
    """
    The EMIA PROJECT Epi-terrestrial Multi-modals Input Assimilation is a digital twin framework built specifically for
    Machine Learning and Data Science projects. It can describe the current situation on the road.
    """
)
st.success("**👈 Select a module from the sidebar** to see some examples of what it can do!")
st.markdown(
    """
    ### Available Demos
    - Expressway Cameras: Fetch images from expressway cameras API \n\n  
    - Dashcams: Fetch video stream from dashcam\n\n  
    - Custom Video: Fetch video stream from URL\n\n  
    - CCTV: Fetch video stream from CCTV\n\n  
    - Custom Image: Fetch image series from folder\n\n  

    """
)

st.markdown(
    """
    ### Want to learn more?
    - Check out [InteractiveCoventry.com/EMIA](https://www.interactivecoventry.com/emia/#main)
    """
)

st.image("assets/qr.png", width=200)

setup_sidebar_info()