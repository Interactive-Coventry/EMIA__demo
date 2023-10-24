import streamlit as st

from . import provide_insights
from .common import present_results

def setup_stream_view():
    st.markdown("### Input from Youtube")
    youtube_url = "https://www.youtube.com/watch?v=8mqSJLPvLWg"
    livecam_url = "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard"

    multi = f'''Example links:

     - From Livecam (Japan): {livecam_url}

    - From Youtube (Singapore): {youtube_url}
    '''
    st.markdown(multi)
    livecam_source_btn = st.radio("Select input source", ["Youtube", "Livecam", "Custom"], index=0,
                                  key="livecam_source")

    if livecam_source_btn == "Youtube":
        st.session_state.stream_url = st.text_input("Input custom url", key="youtube_url", disabled=True,
                                                    value=youtube_url)
    if livecam_source_btn == "Livecam":
        st.session_state.stream_url = st.text_input("Input custom url", key="youtube_url", disabled=True,
                                                    value=livecam_url)
    if livecam_source_btn == "Custom":
        st.session_state.stream_url = st.text_input("Input custom url", key="youtube_url", disabled=False)

    if st.session_state.stream_url is not None and st.button("Start", key="start"):
        provide_insights.get_insights(mode="stream", stream_url=st.session_state.stream_url,
                                      present_results_func=present_results)