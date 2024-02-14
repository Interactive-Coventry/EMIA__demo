import asyncio
import json
import os
import re
import warnings
import streamlit as st
import websockets
from aioice.stun import TransactionFailed
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCIceServer, RTCConfiguration
from aiortc.mediastreams import VideoStreamTrack, MediaStreamError, MediaStreamTrack
from aiortc.sdp import candidate_from_sdp
from libs.foxutils.utils.core_utils import get_logger, settings
from websockets.exceptions import ConnectionClosedError

logger = get_logger("streaming")

QUEUE_SIZE = 16

DEFAULT_ICE_SERVER = settings["ICE_SERVERS"]["url1"]
TARGET_STREAM = settings["STREAM_INFO"]["target_stream"]
CONNECTED_USER = settings["STREAM_INFO"]["connected_user"]
WEBSOCKET_SERVER_URL = settings["SERVER"]["url"]
WEBSOCKET_SERVER_PORT = int(settings["SERVER"]["port"])
WEBSOCKET_SERVER_FULL_URL = f"wss://{WEBSOCKET_SERVER_URL}:{WEBSOCKET_SERVER_PORT}"
ENCODING = settings["SIGNALING"]["encoding"]
TURN_SERVER_URL = settings["TURN_SERVER"]["url"]
TURN_SERVER_USERNAME = settings["TURN_SERVER"]["username"]
TURNS_SERVER_CREDENTIAL = settings["TURN_SERVER"]["credential"]
SAVE_EVERY_N_FRAMES = int(settings["STREAM"]["save_every_n_frames"])
TIMEOUT_AT_N_FRAMES = int(settings["STREAM"]["timeout_at_n_frames"])

def get_ice_servers():
    try:
        turn_server = RTCIceServer(urls=TURN_SERVER_URL,
                                   username=TURN_SERVER_USERNAME,
                                   credential=TURNS_SERVER_CREDENTIAL)
        ice_servers = [turn_server, ]
        return ice_servers

    except KeyError as e:
        logger.warning("Credentials are not set. Fallback to a free STUN server from Google.")
        turn_server = {"urls": DEFAULT_ICE_SERVER}
        turn_server = RTCIceServer(**turn_server)
        ice_servers = [turn_server, ]
        return ice_servers


async def send_disconnect_message(ws, target_device):
    try:
        data = {"type": "disconnect", "name": target_device}
        logger.info(f"Sent: {data}")
        await ws.send(json.dumps(data), )

        data = {"type": "leave", "name": target_device}
        logger.info(f"Sent: {data}")
        await ws.send(json.dumps(data), )
        logger.info("Finished send disconnect message.")

    except ConnectionClosedError as e:
        logger.info(f"Can't send disconnect because connection is already closed: {e}")


async def video_call(ws_, target_device_, datadir_):
    candidate_queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    pcs = set()
    pc = RTCPeerConnection(configuration=RTCConfiguration(get_ice_servers()))
    pcs.add(pc)
    answered = False

    pc.addTrack(VideoStreamTrack())
    pc.addTransceiver("video", direction="recvonly")

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.debug(f"ICE Connection State: {pc.iceConnectionState}")

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.debug(f"ICE Gathering State: {pc.iceGatheringState}")

    @pc.on("datachannel")
    async def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            logger.debug(f"Data channel message: {message}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.debug(f"Connection state is %s" % pc.connectionState)

        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("icecandidate")
    async def on_ice_candidate(candidate):
        logger.debug("CANDIDATE: " + str(candidate))

    async def pc_has_answered():
        while not answered:
            await asyncio.sleep(1)

    async def consume_audio(track):
        while True:
            await track.recv()
            await asyncio.sleep(1)

    class Consumer(MediaStreamTrack):
        kind = "video"

        def __init__(self, track, datadir):
            super().__init__()
            self.track = track
            self.count = 0
            self.datadir = datadir

        async def recv(self):
            frame = await self.track.recv()
            self.count += 1

            if (self.count == 1) or (self.count % SAVE_EVERY_N_FRAMES == 0):
                img = frame.to_image()
                #img.save(os.path.join(self.datadir, "test" + str(self.count) + ".png"))
                save_path = os.path.join(self.datadir, "test" + ".png")
                img.save(save_path)
                logger.info(f"Saved video frame {self.count} at {save_path}")
                await asyncio.sleep(0.5)

            logger.debug(f"Retrieved video frame {self.count} {frame}")
            if 0 < TIMEOUT_AT_N_FRAMES < self.count:
                logger.info(f"Timeout at {self.count} frames")
                raise MediaStreamError()

            return frame

    async def consume_video(track, datadir):
        cons = Consumer(track, datadir)

        while True:
            try:
                frame = await cons.recv()

            except RuntimeError as e:
                logger.info(f"Runtime Error: {e}")
                track.stop()
                break

            except MediaStreamError as e:
                track.stop()
                break

    @pc.on('track')
    async def on_track(track):
        logger.info(f"On Track is on")

        datadir = datadir_

        if track.kind == 'audio':
            asyncio.ensure_future(consume_audio(track))

        elif track.kind == 'video':
            asyncio.ensure_future(consume_video(track, datadir))

        @track.on("ended")
        async def on_ended():
            global answered
            await pc.close()
            logger.info("Peer connection closed")
            answered = False
            logger.info("Set answered to False")
            await send_disconnect_message(ws_, target_device_)

    class Inline(object):
        pass

        def parse_ice_candidate(self: str) -> RTCIceCandidate:
            parts = self.split()

            return RTCIceCandidate(
                component=int(parts[1]),
                foundation=parts[0].split(":")[1],
                protocol=parts[2],
                priority=int(parts[3]),
                ip=parts[4],
                port=int(parts[5]),
                type=parts[7],
            )

    async def handle_message(message):
        global answered

        try:
            answered
        except NameError:
            answered = False

        try:
            # traceback.print_stack()

            data = json.loads(message)
            message_type = data.get("type")
            logger.info(f"Received message of type: {message_type}")

            if message_type == "offer":
                pass

            elif message_type == "answer":
                logger.info("Start: In handle answer")

                a = Inline()
                a.type = data['answer']['type']
                a.sdp = data['answer']['sdp']

                await pc.setRemoteDescription(a)

                logger.info("Received response from the server:")

                while not candidate_queue.empty():
                    logger.debug("Processing candidate in queue")
                    candidate_data = await candidate_queue.get()

                    await process_candidate(candidate_data)

                answered = True
                logger.debug(f"Check answered: {answered}")
                logger.info("End: In handle answer")

            elif message_type == "candidate":
                logger.debug(f"Check answered: {answered}")

                if not answered:
                    logger.debug(f"Add to queue: {data}")
                    loop = st.session_state.loop
                    #asyncio.run_coroutine_threadsafe(candidate_queue.put(data), loop)
                    await candidate_queue.put(data)

                else:
                    logger.debug(f"Process candidate: {data}")
                    # If an answer has arrived, process the candidate immediately
                    await process_candidate(data)

            elif message_type == "login":
                if data.get("success"):
                    logger.info("Login successful")
                else:
                    logger.info("Ooops...try a different username")
                    raise ConnectionError("Login failed")

            elif message_type == "getConfigReq":
                logger.info(
                    f"Received response to Config Request from {data.get('reqUser')} for {data.get('configParm')}")

            elif message_type == "connectFrom":
                if data.get("success"):
                    logger.info(f"Received response to ConnectTo as {data.get('name')}")
                else:
                    warnings.warn(f"Connection refused by remote user for ConnectTo")

            elif message_type == "disconnectFrom":
                logger.info(f"Received disconnect message. Disconnecting from {data.get('name')}")

            elif message_type == "GPS_OUT":
                logger.info(f"Received GPS data: {data}")

            elif message_type == "error":
                logger.info(f"Received an error message: {data.get('message')}")

            else:
                logger.info(f"Received a message of an unexpected type: {data}")

        except json.JSONDecodeError:
            logger.info("Received an invalid JSON message:", message)

        except TransactionFailed as e:
            logger.info(f"STUN Transaction failed: {e}")

    async def send_offer_to_server(ws, target_device):
        sdp_offer = await create_offer()

        data = {
            "type": "offer",
            "offer": {
                "sdp": sdp_offer
            },
            "name": target_device,
        }
        await ws.send(json.dumps(data), )

        logger.debug("Data sent to the server")

        message = await ws.recv()

        await handle_message(message)

        return sdp_offer

    async def process_candidate(candidate_data):
        candidate = candidate_from_sdp(candidate_data["candidate"]["candidate"].split(':', 1)[1])

        candidate.sdpMid = candidate_data["candidate"]["sdpMid"]

        candidate.sdpMLineIndex = candidate_data["candidate"]["sdpMLineIndex"]
        if not candidate.relatedAddress:
            logger.debug(f"Process candidate: {candidate}")
        await pc.addIceCandidate(candidate)

    async def login_to_websocket_server(ws):
        # Construct the login message
        login_message = {"type": "login", "name": CONNECTED_USER}
        await ws.send(json.dumps(login_message), )
        logger.info(f"Sent: {json.dumps(login_message)}")

        message = await ws.recv()
        await handle_message(message)

    async def send_connect_to_message(ws, target_device):
        message = {"type": "connectTo", "name": target_device}

        message_str = json.dumps(message)

        await ws.send(message_str, )
        logger.info(f"Sent: {message_str}")

        message = await ws.recv()
        await handle_message(message)

    async def send_candidate(ws, candidate, target_device):
        candidate_lines = re.findall(r'a=candidate:.*', candidate)
        ufrag_lines = re.findall(r'a=ice-ufrag:.*', candidate)
        ufrag = ufrag_lines[0].split(':')[1].replace('\r', '')
        for s in candidate_lines:
            logger.debug(f"Line {s}")

            message = {
                "type": "candidate",
                "candidate": {
                    "candidate": s.replace("a=candidate", "candidate").replace('\r',
                                                                               '') + " generation 0 ufrag " + ufrag + " network-id 0 network-cost 50",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0,
                    "usernameFragment": ufrag

                },
                "name": target_device,
            }

            logger.info(f"Sent: {message}")
            await ws.send(json.dumps(message), )

            logger.debug(f"Wait response for send:")

            message = await ws.recv()

            await handle_message(message)

    async def create_offer():
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        sdp_offer = pc.localDescription.sdp
        return sdp_offer

    async def run_video_call(ws, target_device):
        try:
            while True:
                await login_to_websocket_server(ws)
                await send_connect_to_message(ws, target_device)
                # await send_gps_request(ws, target_device)
                sdp_offer = await send_offer_to_server(ws, target_device)

                await asyncio.sleep(3)

                await send_candidate(ws, sdp_offer, target_device)

                logger.info("Finished sending candidates")

                while True:
                    await asyncio.sleep(1)

        except ConnectionError as e:
            logger.info(f"Connection failed. Exiting... with {e}")

        except websockets.ConnectionClosedOK as e:
            logger.info(f"Connection closed OK... with message {e}")

        logger.info("Finished video call")

    try:
        await run_video_call(ws_, target_device_)
    except asyncio.CancelledError:
        await asyncio.sleep(1)
        logger.info("Stop button was pressed in the main program.")
