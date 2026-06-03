import asyncio
from asyncio import Queue
import json
import websockets
from websockets.server import WebSocketServerProtocol
import uuid

# Connected clients
clients: dict[str, WebSocketServerProtocol] = {}


# message queues for incoming and outgoing messages.
mq_incoming: Queue = Queue()
mq_outgoing: Queue = Queue()


class IndexedQueue:
    def __init__(self):
        self._queue: Queue = Queue()
        self._index: dict[str, asyncio.Event] = {}
        self._store: dict[str, dict] = {}

    async def put(self, message: dict):
        msg_id = message["id"]
        self._store[msg_id] = message
        event = asyncio.Event()
        self._index[msg_id] = event
        await self._queue.put(message)
        event.set()

    async def get(self) -> dict:
        """FIFO pop, like a normal queue."""
        message = await self._queue.get()
        self._store.pop(message["id"], None)
        self._index.pop(message["id"], None)
        return message

    async def get_by_id(self, msg_id: str, timeout: float = 5.0) -> dict | None:
        """Wait for a specific message ID and pop it."""
        if msg_id not in self._index:
            self._index[msg_id] = asyncio.Event()

        try:
            await asyncio.wait_for(self._index[msg_id].wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        return self._store.pop(msg_id, None)




class WSCommsHandler:
    def __init__(self):
        pass

    def get_response(self, step_idx:int):
        pass





async def handle_client(websocket: WebSocketServerProtocol):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = websocket
    print(f"[handle_client +] Client connected: {client_id} from {websocket.remote_address}")

    try:
        async for raw in websocket:
            await handle_message(websocket, client_id, raw)

    except websockets.exceptions.ConnectionClosedOK:
        print(f"[handle_client -] Client disconnected cleanly: {client_id}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[handle_client -] Client disconnected with error: {e}")
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
    print(f"[handle_message {client_id}] Received: {msg_type}")

    if msg_type == "hello":
        print(f"[handle_message {client_id}] Client identified as: {msg.get('client')}")
        await send(websocket, {
            "type":    "ack",
            "message": "Hello from Python server"
        })

    elif msg_type == "ping":
        await send(websocket, {"type": "pong"})

    elif msg_type == "pong":
        print(f"[{client_id}] Pong received")

    elif msg_type == "ack":
        print(f"[handle_message {client_id}] Ack for command: {msg.get('command')}")

    elif msg_type == "status":
        print(f"[handle_message {client_id}] Status: {msg.get('value')}")
        await send(websocket, {
            "type":    "status_received",
            "message": f"Got status: {msg.get('value')}"
        })

    elif msg_type == "obs":
        print(f"[handle_message {client_id}] Received observation!")
        mq_incoming.put(msg)
        await send(websocket, {
            "type":    "obs_received",
            "code": 0,
            "message": f"OK"
        })

    else:
        print(f"[handle_message {client_id}] Unknown type: {msg_type}")
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
            print("ERROR : Invalid cmd, failed to tokenize")

        if not cmd:
            continue
        
        if cmd == "quit":
            print("[Server] Shutting down...")
            break
        elif tokens[0] == "action":
            print("[Server] Sending action")
            await broadcast({
                "type":    "action",
                "dx": tokens[1],
                "dy": tokens[2],
                "step_idx": tokens[3] 
            })
        else:
            print(f"[Server] Broadcasting command: {cmd}")
            await broadcast({
                "type":    "action",
                "command": cmd
            })


async def main():
    host = "localhost"
    port = 8080

    print(f"[Server] Starting on ws://{host}:{port}")
    print(f"[Server] Connect UE5 to: ws://YOUR_IP:{port}")
    print(f"[Server] Type commands to send to UE5 clients")
    print(f"[Server] Commands: spawn_fire, stop_patrol, start_patrol, reset, quit")

    async with websockets.serve(handle_client, host, port, max_size=50 * 1024 * 1024):
        await asyncio.gather(
            asyncio.Future(),   # keep server running
            command_loop()      # read stdin commands
        )


if __name__ == "__main__":
    asyncio.run(main())

