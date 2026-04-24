from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torchaudio

from app.vad.silero_vad import AudioChunk, SileroVAD


@dataclass
class CapturedAudioTurn:
    samples: torch.Tensor
    sample_rate: int
    start_sec: float
    end_sec: float
    speech_frames: int
    total_frames: int


def decode_audio_bytes(payload: bytes) -> tuple[torch.Tensor, int]:
    if not payload:
        raise ValueError("audio payload must not be empty")

    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
    except Exception:
        waveform, sample_rate = _decode_wave_bytes(payload)

    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        waveform = waveform.reshape(-1)

    return waveform.contiguous(), int(sample_rate)


class MicrophoneTurnCapture:
    """Blocking microphone capture that yields speech turns using VAD."""

    def __init__(
        self,
        vad: SileroVAD,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        end_silence_ms: int = 700,
        min_speech_ms: int = 350,
        max_utterance_s: float = 12.0,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000.0))
        self.end_silence_frames = max(1, int(end_silence_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_utterance_samples = max(1, int(max_utterance_s * sample_rate))

    def iter_turns(self) -> Iterator[CapturedAudioTurn]:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone capture. Install it with `uv pip install sounddevice`."
            ) from exc

        current_frames: list[np.ndarray] = []
        speech_frames = 0
        silence_frames = 0
        turn_start_frame_index = 0
        frame_index = 0
        in_turn = False

        while True:
            frame = sd.rec(
                self.frame_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            ).reshape(-1)

            frame_tensor = torch.from_numpy(frame)
            vad_result = self.vad.process_chunk(
                AudioChunk(samples=frame_tensor, sample_rate=self.sample_rate)
            )

            if not in_turn:
                if not vad_result.is_speech:
                    frame_index += 1
                    continue

                in_turn = True
                current_frames = [frame]
                speech_frames = 1
                silence_frames = 0
                turn_start_frame_index = frame_index
                frame_index += 1
                continue

            current_frames.append(frame)
            if vad_result.is_speech:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            total_samples = sum(piece.shape[0] for piece in current_frames)
            should_end = (
                silence_frames >= self.end_silence_frames
                or total_samples >= self.max_utterance_samples
            )

            if should_end:
                if speech_frames >= self.min_speech_frames:
                    waveform = np.concatenate(current_frames, axis=0).astype(
                        np.float32, copy=False
                    )
                    start_sec = turn_start_frame_index * self.frame_ms / 1000.0
                    end_sec = start_sec + waveform.shape[0] / float(self.sample_rate)
                    yield CapturedAudioTurn(
                        samples=torch.from_numpy(waveform).contiguous(),
                        sample_rate=self.sample_rate,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        speech_frames=speech_frames,
                        total_frames=len(current_frames),
                    )

                current_frames = []
                speech_frames = 0
                silence_frames = 0
                in_turn = False

            frame_index += 1


def _decode_wave_bytes(payload: bytes) -> tuple[torch.Tensor, int]:
    with wave.open(io.BytesIO(payload), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_data = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = np.frombuffer(frame_data, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frame_data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frame_data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError("unsupported WAV sample width")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return torch.from_numpy(audio.copy()), int(sample_rate)
