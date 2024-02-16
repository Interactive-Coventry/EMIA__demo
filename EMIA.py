import time
start_time = time.time()
import streamlit as st
from os.path import join as pathjoin
st.set_page_config(page_title="EMIA Dashboard", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")

init_bar = st.progress(0, text="Initialization in progress. Please wait.")
from libs.foxutils.utils.core_utils import logger
from utils.common import initialize_session_state
initialize_session_state()

init_bar.progress(30, text="Initialization in progress. Loading models. This will take some time.")
import utils.provide_insights

init_bar.progress(100, text="Finished initialization.")
init_bar.empty()


end_time = time.time()
elapsed_time = end_time - start_time
logger.debug(f"Initialization elapsed time: {elapsed_time}seconds")


st.markdown("# Digital Twin of Singapore")
st.markdown("Describe the current situation on the road.")

st.markdown(
    """
    EMIA is a digital twin framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a module from the sidebar** to see some examples
    of what it can do!
    ### Want to learn more?
    - Check out [InteractiveCoventry.com/EMIA](https://www.interactivecoventry.com/emia/#main)

    ### Available Demos
    - Expressway Cameras: Fetch images from expressway cameras API \n\n  
    - Dashcams: Fetch video stream from dashcam\n\n  
    - Custom Video: Fetch video stream from URL\n\n  
    - CCTV: Fetch video stream from CCTV\n\n  
"""
)

st.sidebar.markdown("""
    <style>
    [data-testid='stSidebarNav'] > ul {
        min-height: 60vh;
    } 
    </style>
    """, unsafe_allow_html=True)

st.sidebar.success("Select a module above.")


