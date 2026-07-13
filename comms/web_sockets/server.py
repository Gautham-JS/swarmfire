import asyncio
from janus import Queue
import json
import websockets
from websockets.server import WebSocketServerProtocol
import uuid
import time

import base64
import numpy as np
import cv2
import threading

import logging

# Connected clients
clients: dict[str, WebSocketServerProtocol] = {}


# message queues for incoming and outgoing messages.
mq_incoming: Queue = Queue()
mq_outgoing: Queue = Queue()

# Shared slot for the latest frame - thread-safe single-value exchange
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()

def push_frame(frame: np.ndarray):
    """Called from the test/response thread to update the display frame."""
    global _latest_frame
    with _frame_lock:
        _latest_frame = frame

def display_loop():
    """
    Runs in its own daemon thread.
    Polls _latest_frame and renders it via OpenCV.
    Press 'q' in the OpenCV window to close it.
    """
    logging.info("[display_loop] : Starting OpenCV display thread.")
    cv2.namedWindow("UE5 Observation", cv2.WINDOW_NORMAL)

    while True:
        frame = None
        with _frame_lock:
            if _latest_frame is not None:
                frame = _latest_frame.copy()

        if frame is not None:
            cv2.imshow("UE5 Observation", frame)

        # waitKey drives the OpenCV event loop — must be called even with no frame
        key = cv2.waitKey(30)  # ~33fps polling rate
        if key == ord('q'):
            logging.info("[display_loop] : 'q' pressed, closing display window.")
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

def decode_observation_image(msg: dict) -> np.ndarray | None:
    """
    Decodes the base64 PNG image from an observation message.
    Returns a BGR numpy array ready for OpenCV, or None on failure.
    """
    b64_str = msg.get("image_b64", "")
    if not b64_str:
        logging.info("[decode_observation_image] : No image_b64 field in message.")
        return None

    try:
        png_bytes = base64.b64decode(b64_str)
        np_arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # decodes PNG → BGR
        if img is None:
            logging.info("[decode_observation_image] : cv2.imdecode returned None, likely corrupt data.")
        return img
    except Exception as e:
        logging.info(f"[decode_observation_image] : Failed to decode image: {e}")
        return None





class WSCommsHandler:
    _instance = None

    def __init__(self):
        raise RuntimeError('__init__ not supported for singleton class, call instance() instead')
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            logging.info(f"[WSCommsHandler::instance] - Creating WS Communications Handler instance in memory.")
            cls._instance = cls.__new__(cls)
        return cls._instance
    
    def get_response_with_retries(self, step_id, n_retries=5, wait_delay=0.5):
        for attempt in range(n_retries):
            res = self.get_response(step_id)  # sync_q.get() also safe here
            if res is not None:
                break
            logging.info(f"[WAIT] Did not receive response for step_id {step_id}, re-attempted {attempt + 1}/{n_retries} times...")
            time.sleep(0.5)
        return res

    def get_response(self, step_id):
        msg:dict = mq_incoming.sync_q.get()
        if msg is not None:
            if msg.get("step_id", "") == step_id:
                return msg
            else:
                logging.info(f"[GET RESPONSE] : Found message but a different step ID, query : {step_id}, found : {msg.get('step_id', '')}")
            ## Putting it back wouldnt really work as it is a queue afterall.
            #mq_incoming.sync_q.put(msg)
        return None
    
    def get_response_blocking(self, step_id, n_retries: int = 10, retry_window_secs:float = 1.0):
        retry_window_ms = retry_window_secs * 1000
        retry_interval_ms:float = retry_window_ms / n_retries
        
        i:float = 0.0
        cnt = 0
        res = None
        while (i < retry_window_ms):
            res = self.get_response(step_id)
            if res is not None:
                return res
            i += retry_interval_ms
            cnt += 1
        if res is None:
            logging.error(f"[get_response_blocking] : Failed to get a response for step id {step_id} after {cnt} retries")
        return res

        
    
    def send_msg(self, msg):
        logging.info("[WSCommsHandler::send_msg] enter")
        mq_outgoing.sync_q.put(msg)
        logging.info("[WSCommsHandler::send_msg] exit")
    
    def is_clients_connected(self) -> bool:
        if clients is None:
            return False
        return (len(clients) != 0)
    




async def handle_client(websocket: WebSocketServerProtocol):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = websocket
    logging.info(f"[handle_client +] Client connected: {client_id} from {websocket.remote_address}")

    try:
        async for raw in websocket:
            await handle_message(websocket, client_id, raw)

    except websockets.exceptions.ConnectionClosedOK:
        logging.info(f"[handle_client -] Client disconnected cleanly: {client_id}")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.info(f"[handle_client -] Client disconnected with error: {e}")
    finally:
        del clients[client_id]


async def handle_message(
    websocket: WebSocketServerProtocol,
    client_id: str,
    raw: str
):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await send(websocket, {"type": "error", "message": "invalid json"})
        return

    msg_type = msg.get("type", "")
    logging.info(f"[handle_message {client_id}] Received: {msg_type}")

    if msg_type == "hello":
        logging.info(f"[handle_message {client_id}] Client identified as: {msg.get('client')}")
        await send(websocket, {
            "type":    "ack",
            "message": "Hello from Python server"
        })

    elif msg_type == "ping":
        await send(websocket, {"type": "pong"})

    elif msg_type == "pong":
        logging.info(f"[{client_id}] Pong received")

    elif msg_type == "ack":
        logging.info(f"[handle_message {client_id}] Ack for command: {msg.get('command')}")

    elif msg_type == "status":
        logging.info(f"[handle_message {client_id}] Status: {msg.get('value')}")
        await send(websocket, {
            "type":    "status_received",
            "message": f"Got status: {msg.get('value')}"
        })

    elif msg_type == "observation":
        logging.info(f"[handle_message {client_id}] Received observation!")
        await mq_incoming.async_q.put(msg)
        await send(websocket, {
            "type":    "obs_received",
            "code": 0,
            "message": f"OK"
        })

    else:
        logging.info(f"[handle_message {client_id}] Unknown type: {msg_type}")
        await send(websocket, {
            "type":    "error",
            "message": f"Unknown message type: {msg_type}"
        })


async def send(websocket: WebSocketServerProtocol, data: dict):
    """Send JSON to a single client."""
    try:
        await websocket.send(json.dumps(data))
    except websockets.exceptions.ConnectionClosed:
        pass


async def broadcast(data: dict):
    """Send JSON to all connected clients."""
    if not clients:
        return
    message = json.dumps(data)
    await asyncio.gather(*[
        client.send(message)
        for client in clients.values()
    ])



def test_random_actions_sync():
    logging.info("[test_random_actions : Enter]")
    ws_handler = WSCommsHandler.instance()
    next_msg = 1
    i = 0
    while (i < 100):
        if clients is None or len(clients) == 0:
            logging.info("[TEST :: SEND] :: No client connected, waiting")
            time.sleep(1)
            continue

        logging.info(f"[TEST :: SEND] : Sending action with step_id {i}")
        ws_handler.send_msg({
            "type": "action",
            "dx": next_msg,
            "dy": 0,
            "step_id": i,
            "step_idx": 0,
        })
        next_msg *= -1
        time.sleep(1)  # safe to block here - we're in a thread


        res = None
        for _ in range(5):
            res = ws_handler.get_response(i)  # sync_q.get() also safe here
            if res is not None:
                break
            logging.info(f"[WAIT] Did not receive response for step_id {i}, re-attempting...")
            time.sleep(0.5)

        i += 1

        if res is None:
            logging.info(f"[TEST :: Receive] : Failed after 5 attempts, giving up.")
        else:
            logging.info(f"[OK] [TEST :: Receive] : Received response for step ID {i}")
            frame = decode_observation_image(res)
            if frame is not None:
                push_frame(frame)  # hand off to display thread





# handles dispatching outgoing msgs
async def dispatch_loop():
    logging.info("[dispatch_loop : Enter]")
    while True:
        msg = await mq_outgoing.async_q.get()
        if msg is None:
            continue
        logging.info(f"[OUTGOING MSG] : Sending msg {msg}")
        
        await broadcast(msg)



async def command_loop():
    """
    Read commands from stdin and broadcast to all UE5 clients.
    Run this in parallel with the server.

    Type commands like:
        spawn_fire
        stop_patrol
        start_patrol
        reset
    """
    loop = asyncio.get_event_loop()

    while True:
        # Read input without blocking event loop
        cmd = await loop.run_in_executor(None, input, "Command > ")
        cmd = cmd.strip()
        tokens = cmd.split()

        if len(tokens) == 0:
            logging.info("ERROR : Invalid cmd, failed to tokenize")

        if not cmd:
            continue
        
        if cmd == "quit":
            logging.info("[Server] Shutting down...")
            break
        elif tokens[0] == "action":
            logging.info("[Server] Sending action")
            await broadcast({
                "type":    "action",
                "dx": tokens[1],
                "dy": tokens[2],
                "step_idx": tokens[3] 
            })
        else:
            logging.info(f"[Server] Broadcasting command: {cmd}")
            await broadcast({
                "type":    "action",
                "command": cmd
            })



async def main():
    host = "0.0.0.0"
    port = 8090

    logging.info(f"[Server] Starting on ws://{host}:{port}")
    logging.info(f"[Server] Connect UE5 to: ws://YOUR_IP:{port}")
    logging.info(f"[Server] Type commands to send to UE5 clients")
    logging.info(f"[Server] Commands: spawn_fire, stop_patrol, start_patrol, reset, quit")

    t = threading.Thread(target=test_random_actions_sync, daemon=True)
    t.start()

    #display_t = threading.Thread(target=display_loop, daemon=True)
    #display_t.start()

    async with websockets.serve(
        handle_client,
        host,
        port,
        max_size=50 * 1024 * 1024
    ):
        await asyncio.gather(
            asyncio.Future(),   # keep server running
            dispatch_loop(),      # read stdin commands
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    asyncio.run(main())
