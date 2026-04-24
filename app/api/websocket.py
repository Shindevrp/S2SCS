from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import torch

from app.config import AppConfig
from app.monitoring.metrics import MetricsRegistry
from app.pipeline.e2e_pipeline import (
    EndToEndSpeechPipeline,
    TextResponseArtifacts,
    TranscriptionArtifacts,
)
from app.streaming.streamer import StreamingAudioChunk
from app.utils.logger import get_logger


@dataclass
class WSMessage:
    event: str
    data: dict[str, Any]


class AsyncConversationHandler:
    """Async WebSocket handler for low-latency conversation."""

    def __init__(
        self,
        pipeline: EndToEndSpeechPipeline,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics or MetricsRegistry()
        self.logger = get_logger(self.__class__.__name__)

    async def handle_client(
        self,
        websocket: Any,
    ) -> None:
        """Handle WebSocket client connection."""
        await websocket.accept()

        try:
            async for message in self._iter_messages(websocket):
                await self._process_message(websocket, message)
        except Exception as exc:
            self.logger.error("WebSocket error: %s", exc)
            await self._send_event(websocket, "error", {"message": str(exc)})

    async def _iter_messages(
        self,
        websocket: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Iterate over WebSocket messages."""
        while True:
            try:
                data = await websocket.receive_text()
                yield json.loads(data)
            except Exception:
                break

    async def _process_message(
        self,
        websocket: Any,
        message: dict[str, Any],
    ) -> None:
        """Process a client message."""
        msg_type = message.get("type")

        if msg_type == "audio":
            await self._handle_audio(websocket, message)
        elif msg_type == "text":
            await self._handle_text(websocket, message)
        elif msg_type == "warmup":
            await self._handle_warmup(websocket)
        elif msg_type == "stop":
            await self._send_event(websocket, "stopped", {})
        else:
            await self._send_event(websocket, "error", {"message": f"Unknown type: {msg_type}"})

    async def _handle_audio(
        self,
        websocket: Any,
        message: dict[str, Any],
    ) -> None:
        """Handle incoming audio."""
        audio_data = message.get("data")
        if not audio_data:
            await self._send_event(websocket, "error", {"message": "No audio data"})
            return

        try:
            audio_bytes = base64.b64decode(audio_data)
            import numpy as np
            import io
            import wave

            with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(samples)
                sample_rate = wf.getframerate()
        except Exception as exc:
            await self._send_event(websocket, "error", {"message": str(exc)})
            return

        await self._send_event(websocket, "processing", {})

        result = await asyncio.to_thread(
            self.pipeline.run_audio_turn,
            audio_tensor,
            sample_rate,
            apply_vad=True,
            synthesize_audio=False,
        )

        await self._send_event(
            websocket,
            "transcript",
            {"text": result.transcription.transcript_text},
        )

        if result.text_response.response_text:
            await self._send_event(
                websocket,
                "response_text",
                {"text": result.text_response.response_text},
            )

        if result.speech_result:
            speech_result = await asyncio.to_thread(
                self.pipeline.synthesize_response,
                result.text_response,
            )

            async for chunk in self._stream_chunks(websocket, speech_result.waveform, speech_result.sample_rate):
                pass

        await self._send_event(websocket, "complete", {})

    async def _handle_text(
        self,
        websocket: Any,
        message: dict[str, Any],
    ) -> None:
        """Handle incoming text."""
        text = message.get("text", "")
        if not text:
            await self._send_event(websocket, "error", {"message": "Empty text"})
            return

        result = await asyncio.to_thread(
            self.pipeline.respond_to_text,
            text,
        )

        await self._send_event(
            websocket,
            "response_text",
            {"text": result.response_text},
        )

        speech_result = await asyncio.to_thread(
            self.pipeline.synthesize_response,
            result,
        )

        async for chunk in self._stream_chunks(websocket, speech_result.waveform, speech_result.sample_rate):
            pass

        await self._send_event(websocket, "complete", {})

    async def _handle_warmup(
        self,
        websocket: Any,
    ) -> None:
        """Handle warmup request."""
        await asyncio.to_thread(lambda: self.pipeline)
        await self._send_event(websocket, "warmed_up", {})

    async def _stream_chunks(
        self,
        websocket: Any,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> AsyncIterator[None]:
        """Stream audio chunks to client."""
        from app.audio.playback import waveform_to_pcm16_bytes

        chunk_samples = sample_rate // 10
        for i in range(0, waveform.numel(), chunk_samples):
            chunk = waveform[i : i + chunk_samples]
            pcm_bytes = waveform_to_pcm16_bytes(chunk)
            await self._send_event(
                websocket,
                "audio_chunk",
                {
                    "data": base64.b64encode(pcm_bytes).decode("ascii"),
                    "is_first": i == 0,
                    "is_last": i + chunk_samples >= waveform.numel(),
                },
            )
            yield

    async def _send_event(
        self,
        websocket: Any,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Send event to client."""
        message = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        await websocket.send_text(message)


class LowLatencyPipeline:
    """Async wrapper for the pipeline for low-latency processing."""

    def __init__(
        self,
        config: AppConfig,
        *,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or MetricsRegistry()
        self.logger = get_logger(self.__class__.__name__)

        self._pipeline: EndToEndSpeechPipeline | None = None
        self._handler: AsyncConversationHandler | None = None

    async def get_handler(self) -> AsyncConversationHandler:
        """Get or create the async handler."""
        if self._handler is None:
            pipeline = await self._get_pipeline()
            self._handler = AsyncConversationHandler(pipeline, self.metrics)
        return self._handler

    async def _get_pipeline(self) -> EndToEndSpeechPipeline:
        """Get pipeline (lazy load)."""
        if self._pipeline is None:
            self._pipeline = await asyncio.to_thread(
                EndToEndSpeechPipeline.from_config,
                self.config,
                metrics=self.metrics,
            )
        return self._pipeline