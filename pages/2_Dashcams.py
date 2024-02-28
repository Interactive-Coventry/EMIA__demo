import asyncio
import concurrent.futures
import os
import time
from datetime import datetime
from os.path import join as pathjoin
import streamlit as st

from utils.configuration import DASHCAM_NAMES

st.set_page_config(page_title="Dashcams", page_icon=pathjoin('assets', 'favicon.ico'), layout="centered",
                   initial_sidebar_state="expanded")
import websockets
from PIL import UnidentifiedImageError, ImageFile
from libs.foxutils.utils.core_utils import get_logger, settings
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from utils import configuration
from utils.common import set_value, present_results
from utils.provide_insights import get_insights, process_dashcam_frame
from utils.streaming import video_call, WEBSOCKET_SERVER_FULL_URL, send_disconnect_message

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = get_logger("dashcam-view")
DATA_DIR = settings["DIRECTORY"]["datasets_dir"]

def make_new_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
    logger.info(f"Made new Event loop: {id(loop)}")
    return loop


def cancel_all_tasks(pending):
    for task in pending:
        try:
            task.cancel()
        except asyncio.CancelledError as e:
            logger.debug(f"asyncio.CancelledError: {e}")
            logger.info(f"Task {task} is cancelled.")
    logger.info("All pending tasks were cancelled.")


async def cancel_all_tasks2(pending):
    for task in pending:
        try:
            task.cancel()
            await task
        except asyncio.CancelledError:
            logger.info(f"Task {task} is cancelled.")
    logger.info("All pending tasks were cancelled.")


def is_target_task(x):
    details = x.get_coro()
    if details:
        if "WebSocketCommonProtocol" not in str(details):
            return True
    return False


def ask_exit():
    if "loop" in st.session_state and st.session_state.loop is not None:
        loop = st.session_state.loop
        pending = asyncio.all_tasks(loop)
        logger.info(f"Pending tasks: {pending}")
        pending = [x for x in pending if is_target_task(x)]
        logger.debug(f"Pending tasks: {pending}")

        if pending:
            logger.info(f"Pending tasks: len={len(pending)}")
            asyncio.set_event_loop(loop)
            try:
                #asyncio.ensure_future(cancel_all_tasks2(pending))
                cancel_all_tasks(pending)
            except RuntimeError as e:
                logger.error(f"Runtime Error: {e}")


async def make_call(target_dashcam, datadir):
    delete_empty_folders(DATA_DIR)
    os.makedirs(datadir, exist_ok=True)

    logger.debug("Starting a call to camera {}".format(target_dashcam))

    uri = WEBSOCKET_SERVER_FULL_URL
    async with websockets.connect(uri) as ws:
        logger.debug(f"Running loop {id(asyncio.get_running_loop())}")

        task = asyncio.create_task(video_call(ws, target_dashcam, datadir, processing_func_=process_dashcam_frame),
                                   name="video_call")
        await task
        await send_disconnect_message(ws, target_dashcam)

    logger.info(f"Websocket connection {id(ws)} is closing. Call has ended.")


def run_async_task(loop, target_dashcam, datadir):
    loop.run_until_complete(make_call(target_dashcam, datadir))
    loop.stop()
    logger.info(f"Event loop {id(loop)} is closed.")


def display_fetched_image(container_placeholder, datadir, previous_files):
    files = [x for x in os.listdir(datadir) if ".png" in x]
    if len(files) > previous_files:
        previous_files = len(files)

        try:
            file = pathjoin(datadir, files[-1])
            time.sleep(0.1)
            if os.path.exists(file):
                camera_id = st.session_state.target_dashcam
                outputs = get_insights(mode="files", full_filename=file, camera_id=camera_id, get_location=True)
                if outputs is not None:
                    #with container_placeholder:
                    #    st.image(outputs["vehicle_detection_img"], width=500, caption=str(len(files)))
                    present_results(container_placeholder, outputs)

        except UnidentifiedImageError as e:
            logger.error(f"UnidentifiedImageError: {e}")

    time.sleep(0.5)
    return previous_files


def delete_empty_folders(target_dir):
    logger.info(f"Deleting empty folders in {target_dir}")
    deleted = set()

    for current_dir, subdirs, files in os.walk(target_dir, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break

        try:
            if not any(files) and not still_has_subdirs:
                os.rmdir(current_dir)
                deleted.add(current_dir)
        except FileNotFoundError as e:
            #logger.debug(f"FileNotFoundError: {e}")
            pass

    logger.debug(f"Deleted folders: {deleted}")
    return deleted


def clear_jobs():
    logger.debug("Stop button clicked.")
    ask_exit()
    set_value("is_running", False, reset=True)


def setup_dashcam_view():
    st.markdown("### Input from Dashcam")
    st.markdown(configuration.DEMO_INSTRUCTIONS)

    col1, col2 = st.columns([0.5, 0.5], gap="small")
    with col1:
        radio_btn = st.selectbox("Select a camera and click the start button:", configuration.DASHCAM_IDS.keys(),
                                 index=0)
        st.session_state.target_dashcam = configuration.DASHCAM_IDS[radio_btn]

    col3, col4, col5 = st.columns([0.3, 0.3, 0.4])
    with col3:
        exec_btn_placeholder = st.empty()
    with col4:
        refresh_btn = st.button("Refresh", key="refresh_btn_dashcam")

    container_placeholder = st.empty()

    if st.session_state.is_running:
        clear_jobs()

    if not st.session_state.is_running:
        if exec_btn_placeholder.button("Fetch latest", key="start_btn_dashcam"):
            logger.debug("Start button clicked.")
            set_value("is_running", True)
            exec_btn_placeholder.button("Stop", key="stop_btn_dashcam")

            loop = make_new_loop()

            with concurrent.futures.ThreadPoolExecutor() as executor:
                ctx = get_script_run_ctx()

                datadir = os.path.join(DATA_DIR, "dashcam", st.session_state.target_dashcam,
                                       datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                future = executor.submit(run_async_task, loop, st.session_state.target_dashcam, datadir)
                for t in executor._threads:
                    logger.debug(f"add_script_run_ctx Thread: {t}")
                    add_script_run_ctx(t, ctx)

                previous_files = 0
                while future.running():
                    try:
                        previous_files = display_fetched_image(container_placeholder, datadir, previous_files)
                    except FileNotFoundError as e:
                        with container_placeholder.container():
                            st.info("Waiting for the first image to be fetched.")
                        logger.debug(f"FileNotFoundError: {e}")



if __name__ == "__main__":
    setup_dashcam_view()