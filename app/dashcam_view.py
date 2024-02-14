import asyncio
import concurrent.futures
import os
import time
from datetime import datetime
from os.path import join as pathjoin
import streamlit as st
import websockets
from PIL import Image, UnidentifiedImageError, ImageFile
from libs.foxutils.utils.core_utils import get_logger, settings
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from utils import configuration
from utils.streaming import video_call, WEBSOCKET_SERVER_FULL_URL, send_disconnect_message
from . import provide_insights
from .common import present_results, reset_values

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = get_logger("dashcam-view")
DATA_DIR = settings["DIRECTORY"]["datasets_dir"]

if "is_calling" not in st.session_state:
    st.session_state.is_calling = False

if "has_pending_tasks" not in st.session_state:
    st.session_state.has_pending_tasks = False

if "target_device" not in st.session_state:
    st.session_state.target_device = None

if "loop" not in st.session_state:
    st.session_state.loop = None

if "first_run" not in st.session_state:
    st.session_state.first_run = True


def is_first_run():
    return st.session_state.first_run


def make_new_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    st.session_state.loop = loop
    logger.info(f"Made new Event loop: {id(loop)}")
    return loop


def on_button_click(is_calling):
    st.session_state.is_calling = is_calling
    logger.debug(f"Changed is_calling to {is_calling}")
    if is_calling:
        set_has_pending_tasks(True)


def set_has_pending_tasks(value):
    st.session_state.has_pending_tasks = value
    logger.info(f"Changed has_pending_tasks to {value}")


def cancel_all_tasks(pending):
    for task in pending:
        try:
            task.cancel()
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
                cancel_all_tasks(pending)
            except RuntimeError as e:
                logger.error(f"Runtime Error: {e}")


async def make_call(target_device, datadir):
    delete_empty_folders(DATA_DIR)
    os.makedirs(datadir, exist_ok=True)

    logger.debug("Starting a call to camera {}".format(target_device))

    uri = WEBSOCKET_SERVER_FULL_URL
    async with websockets.connect(uri) as ws:
        logger.debug(f"Running loop {id(asyncio.get_running_loop())}")

        task = asyncio.create_task(video_call(ws, target_device, datadir), name="video_call")
        await task
        await send_disconnect_message(ws, target_device)

    logger.info(f"Websocket connection {id(ws)} is closing. Call has ended.")


def run_async_task(loop, target_device, datadir):
    loop.run_until_complete(make_call(target_device, datadir))
    loop.stop()
    logger.info(f"Event loop {id(loop)} is closed.")
    set_has_pending_tasks(False)
    st.rerun()


def display_fetched_image(container_placeholder, datadir):
    with container_placeholder.container():
        st.markdown(f"Current time is {datetime.now().strftime('%Y-%m-%d %H-%M')}")
        try:
            file = pathjoin(datadir, "test.png")
            # file = pathjoin('images', "test.png")
            if os.path.exists(file):
                img = Image.open(file)
                st.image(img, width=500)
        except UnidentifiedImageError as e:
            logger.error(f"Error: {e}")
        time.sleep(0.5)

        #provide_insights.get_insights(mode="stream",
        #                              stream_url=st.session_state.stream_url,
        #                              stream_name=st.session_state.stream_name,
        #                              present_results_func=lambda x, y: present_results(x, y, forecast_step=1),
        #                              update_every_n_frames=st.session_state.update_every_n_frames)


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
            logger.error(f"FileNotFoundError: {e}")

    logger.debug(f"Deleted folders: {deleted}")
    return deleted


def setup_dashcam_view():
    reset_values()

    st.markdown("### Input from Dashcam")
    st.markdown(configuration.DEMO_INSTRUCTIONS)

    col1, col2 = st.columns([0.5, 0.5], gap="small")
    with col1:
        radio_btn = st.selectbox("Select a camera and click the start button:", configuration.DASHCAM_IDS.keys(),
                                 index=0)
        st.session_state.target_device = configuration.DASHCAM_IDS[radio_btn]

    col3, col4, col5 = st.columns([0.3, 0.3, 0.4], gap="small")
    with col3:
        start_button = st.button("Start Call", key="start_call", on_click=on_button_click, args=(True,),
                                 disabled=st.session_state.has_pending_tasks)
    with col4:
        stop_button = st.button("Stop", key="stop_call", on_click=on_button_click, args=(False,),
                                disabled=not st.session_state.has_pending_tasks)
    container_placeholder = st.empty()

    if st.session_state.is_calling:
        logger.info("Pressed start button...")
        loop = make_new_loop()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            ctx = get_script_run_ctx()

            datadir = os.path.join(DATA_DIR, datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            future = executor.submit(run_async_task, loop, st.session_state.target_device, datadir)
            for t in executor._threads:
                add_script_run_ctx(t, ctx)

            while future.running():
                display_fetched_image(container_placeholder, datadir)

    else:
        if not is_first_run():
            logger.info("Pressed stop button...")
            with container_placeholder.container():
                st.info("Wait until streaming has ended.")
            ask_exit()

        else:
            st.session_state.first_run = False



