# test_client.py
import asyncio
import json
import websockets

async def test():
    url = "ws://localhost:8080"
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        print("Connected")

        # Test 1 — hello
        await ws.send(json.dumps({
            "type":   "hello",
            "client": "TestClient"
        }))
        response = await ws.recv()
        print(f"Hello response: {response}")

        # Test 2 — status
        await ws.send(json.dumps({
            "type":  "status",
            "value": "flying"
        }))
        response = await ws.recv()
        print(f"Status response: {response}")

        # Test 3 — listen for commands
        # (type something in the server terminal)
        print("Waiting for commands from server (type in server terminal)...")
        print("Press Ctrl+C to stop")

        async for message in ws:
            msg = json.loads(message)
            print(f"Received: {msg}")

            # Ack any command
            if msg.get("type") == "command":
                await ws.send(json.dumps({
                    "type":    "ack",
                    "command": msg.get("command")
                }))
                print(f"Acked command: {msg.get('command')}")

asyncio.run(test())
