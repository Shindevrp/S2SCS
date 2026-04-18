from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from app.utils.logger import get_logger


SUPPORTED_SAMPLE_RATES = {8000, 16000}
SILERO_WINDOW_SAMPLES = {
    8000: 256,
    16000: 512,
}


@dataclass
class AudioChunk:
    samples: torch.Tensor
    sample_rate: int
    chunk_id: Optional[str] = None


@dataclass
class VADResult:
    is_speech: bool
    speech_probability: float
    chunk_id: Optional[str]
    sample_rate: int
    num_samples: int


class SileroVAD:
    """Streaming-friendly Silero VAD wrapper for chunk-wise classification."""

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "cpu",
        model: Optional[Any] = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.device = torch.device(device)
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or self._load_model()

    def process_chunk(self, chunk: AudioChunk) -> VADResult:
        """Classify one audio chunk as speech or non-speech."""
        self._validate_chunk(chunk)
        audio = self._prepare_audio(chunk.samples)

        try:
            probability = self._infer_probability(audio, chunk.sample_rate)
        except Exception as exc:
            self.logger.exception("Silero VAD inference failed for chunk %s", chunk.chunk_id)
            raise RuntimeError("Silero VAD inference failed") from exc

        is_speech = probability >= self.threshold
        self.logger.debug(
            "chunk_id=%s sample_rate=%s samples=%s speech_probability=%.4f is_speech=%s",
            chunk.chunk_id,
            chunk.sample_rate,
            audio.numel(),
            probability,
            is_speech,
        )

        return VADResult(
            is_speech=is_speech,
            speech_probability=probability,
            chunk_id=chunk.chunk_id,
            sample_rate=chunk.sample_rate,
            num_samples=audio.numel(),
        )

    def _infer_probability(self, audio: torch.Tensor, sample_rate: int) -> float:
        window = SILERO_WINDOW_SAMPLES[sample_rate]
        probabilities: list[float] = []

        with torch.inference_mode():
            if audio.numel() <= window:
                windowed = self._pad_to_window(audio, window)
                return float(self.model(windowed, sample_rate).item())

            for start in range(0, audio.numel(), window):
                end = min(start + window, audio.numel())
                windowed = self._pad_to_window(audio[start:end], window)
                prob = float(self.model(windowed, sample_rate).item())
                probabilities.append(prob)

        if not probabilities:
            return 0.0

        # A chunk is considered speech if any internal window is strongly speech-like.
        return max(probabilities)

    def _pad_to_window(self, audio: torch.Tensor, window: int) -> torch.Tensor:
        if audio.numel() == window:
            return audio

        if audio.numel() > window:
            return audio[:window].contiguous()

        padded = torch.zeros(window, dtype=audio.dtype, device=audio.device)
        padded[: audio.numel()] = audio
        return padded.contiguous()

    def _load_model(self) -> Any:
        try:
            from silero_vad import load_silero_vad
        except ImportError as exc:
            self.logger.exception("silero-vad is not installed")
            raise RuntimeError(
                "silero-vad is required. Install it with `pip install silero-vad`."
            ) from exc

        try:
            model = load_silero_vad(onnx=False)
            if hasattr(model, "to"):
                model = model.to(self.device)
            return model
        except Exception as exc:
            self.logger.exception("Failed to load Silero VAD model")
            raise RuntimeError("Failed to load Silero VAD model") from exc

    def _prepare_audio(self, samples: torch.Tensor) -> torch.Tensor:
        audio = torch.as_tensor(samples, dtype=torch.float32, device=self.device)

        if audio.ndim == 2:
            # Convert stereo or multi-channel chunks to mono for VAD.
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            raise ValueError("AudioChunk.samples must be 1D or 2D")

        return audio.contiguous()

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        if chunk.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"Unsupported sample rate {chunk.sample_rate}. "
                f"Expected one of {sorted(SUPPORTED_SAMPLE_RATES)}."
            )
        if torch.as_tensor(chunk.samples).numel() == 0:
            raise ValueError("AudioChunk.samples must not be empty")
