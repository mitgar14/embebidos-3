"""Probe WS roundtrip contra la Nano. Uso interno de diagnostico."""
import asyncio
import json
import websockets


async def main() -> None:
    url = "ws://100.100.166.120:8000/ws"
    print(f"[ws] dial {url}")
    async with websockets.connect(url) as ws:
        print("[ws] connected")
        await ws.send(json.dumps({"type": "conf", "value": 0.5}))
        print("[ws] sent conf")
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
            print(f"[ws] reply: {msg[:300]}")
        except asyncio.TimeoutError:
            print("[ws] no reply 3s (server quiza solo replica a frames binarios)")


if __name__ == "__main__":
    asyncio.run(main())
