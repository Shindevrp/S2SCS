#!/usr/bin/env python3
"""WebSocket client for real-time conversation."""

import asyncio
import argparse
import base64
import json
import wave
from pathlib import Path

import numpy as np


class ConversationClient:
    def __init__(self, url: str = "ws://localhost:8000/ws/conversation") -> None:
        self.url = url
        self.websocket = None
        self.sample_rate = 16000

    async def connect(self) -> None:
        try:
            import websockets
        except ImportError:
            raise RuntimeError("websockets required. Install: uv pip install websockets")

        self.websocket = await websockets.connect(self.url)
        print(f"Connected to {self.url}")

    async def send_audio_file(self, path: Path) -> None:
        """Send audio file to server."""
        with wave.open(str(path), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError("Audio must be mono")
            if wf.getframerate() != self.sample_rate:
                raise ValueError(f"Audio must be {self.sample_rate} Hz")

            frames = wf.readframes(wf.getnframes())
            audio_bytes = wf.getparams().sampwidth * wf.getnframes()

        audio_b64 = base64.b64encode(frames).decode("ascii")
        await self.websocket.send(json.dumps({"type": "audio", "data": audio_b64}))

    async def send_text(self, text: str) -> None:
        """Send text to server."""
        await self.websocket.send(json.dumps({"type": "text", "text": text}))

    async def receive_loop(self) -> None:
        """Receive and print messages from server."""
        async for message in self.websocket:
            data = json.loads(message)
            event = data.get("event")
            payload = data.get("data", {})

            if event == "transcript":
                print(f"\n[User] {payload.get('text')}")
            elif event == "response_text":
                print(f"[Assistant] {payload.get('text')}")
            elif event == "audio_chunk":
                is_first = payload.get("is_first", False)
                is_last = payload.get("is_last", False)
                print(f"[Audio] chunk {'first' if is_first else ''} {'last' if is_last else ''}")
            elif event == "processing":
                print("[Processing...]")
            elif event == "complete":
                print("\n[Done]")
                break
            elif event == "error":
                print(f"[Error] {payload.get('message')}")
                break
            elif event == "warmed_up":
                print("[Warmed up]")

    async def chat(self) -> None:
        """Interactive chat loop."""
        print("Type 'quit' to exit")
        await self.connect()
        await self._send_warmup()

        while True:
            text = input("\nYou: ").strip()
            if not text:
                continue
            if text.lower() == "quit":
                break

            await self.send_text(text)
            await self.receive_loop()

    async def _send_warmup(self) -> None:
        """Send warmup request."""
        await self.websocket.send(json.dumps({"type": "warmup"}))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Conversation WebSocket client")
    parser.add_argument("--url", default="ws://localhost:8000/ws/conversation")
    parser.add_argument("--text", help="Send text and exit")
    parser.add_argument("--audio", type=Path, help="Send audio file and exit")
    args = parser.parse_args()

    client = ConversationClient(args.url)

    if args.text:
        await client.connect()
        await client.send_text(args.text)
        await client.receive_loop()
    elif args.audio:
        await client.connect()
        await client.send_audio_file(args.audio)
        await client.receive_loop()
    else:
        await client.chat()


if __name__ == "__main__":
    asyncio.run(main())