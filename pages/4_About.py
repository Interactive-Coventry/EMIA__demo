import streamlit as st
from utils.common import setup_sidebar_info

setup_sidebar_info()

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

