from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from app.streaming.streamer import StreamingAudioChunk


def waveform_to_pcm16_bytes(waveform: torch.Tensor) -> bytes:
    normalized = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu().reshape(-1)
    clipped = normalized.clamp(-1.0, 1.0)
    pcm = (clipped * 32767.0).round().to(torch.int16)
    return pcm.numpy().tobytes()


class AudioPlayback:
    """Thin playback wrapper used by local live-conversation clients."""

    def play_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        blocking: bool = True,
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for audio playback. Install it with `uv pip install sounddevice`."
            ) from exc

        audio = (
            torch.as_tensor(waveform, dtype=torch.float32)
            .detach()
            .cpu()
            .reshape(-1)
            .numpy()
            .astype(np.float32, copy=False)
        )
        if audio.size == 0:
            return

        sd.play(audio, samplerate=int(sample_rate), blocking=blocking)

    def play_stream(
        self,
        chunks: Iterable[StreamingAudioChunk],
        *,
        blocking: bool = True,
    ) -> None:
        for chunk in chunks:
            self.play_waveform(chunk.waveform, chunk.sample_rate, blocking=blocking)
