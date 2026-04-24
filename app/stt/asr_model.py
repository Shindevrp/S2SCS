from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from app.utils.logger import get_logger


DEFAULT_MODEL_ID = "Nabeh/QwBaseerSTT"
DEFAULT_LOCAL_MODEL_DIR = Path("models/Nabeh_QwBaseerSTT")
TIMESTAMP_PATTERN = re.compile(
    r"\[\s*(?P<start>\d+(?:\.\d+)?)\s*[-,]\s*(?P<end>\d+(?:\.\d+)?)\s*\]\s*(?P<text>[^\[]+)"
)


@dataclass
class ASRAudioChunk:
    samples: torch.Tensor
    sample_rate: int
    chunk_id: Optional[str] = None
    chunk_start_sec: float = 0.0


@dataclass
class TranscriptionSegment:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class ASRResult:
    text: str
    segments: list[TranscriptionSegment]
    language_hint: Optional[str]
    chunk_id: Optional[str]
    duration_sec: float


class QwBaseerASR:
    """Offline-capable ASR wrapper for Nabeh/QwBaseerSTT using qwen-asr library."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_LOCAL_MODEL_DIR,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 256,
        local_files_only: bool = True,
        processor: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = dtype or (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.logger = get_logger(self.__class__.__name__)

        self.processor = processor
        self.model = model

        if self.model is None:
            self.model = self._load_components()

    def transcribe_chunk(
        self,
        chunk: ASRAudioChunk,
        language_hint: Optional[str] = None,
        return_timestamps: bool = True,
    ) -> ASRResult:
        self._validate_chunk(chunk)
        audio = self._prepare_audio(chunk.samples)
        duration_sec = audio.numel() / float(chunk.sample_rate)
        prompt = self._build_prompt(
            language_hint=language_hint, return_timestamps=return_timestamps
        )

        try:
            decoded_text = self._run_inference(
                audio=audio,
                sample_rate=chunk.sample_rate,
                prompt=prompt,
                language_hint=language_hint,
                return_timestamps=return_timestamps,
            )
        except Exception as exc:
            self.logger.exception("ASR inference failed for chunk %s", chunk.chunk_id)
            raise RuntimeError("QwBaseer ASR inference failed") from exc

        segments = self._build_segments(
            decoded_text=decoded_text,
            chunk_start_sec=chunk.chunk_start_sec,
            chunk_duration_sec=duration_sec,
        )

        self.logger.debug(
            "chunk_id=%s duration_sec=%.3f transcript_chars=%s segments=%s",
            chunk.chunk_id,
            duration_sec,
            len(decoded_text),
            len(segments),
        )

        return ASRResult(
            text=" ".join(segment.text for segment in segments).strip() or decoded_text,
            segments=segments,
            language_hint=language_hint,
            chunk_id=chunk.chunk_id,
            duration_sec=duration_sec,
        )

    def _load_components(self) -> Any:
        model_source = self._resolve_model_source()

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            self.logger.exception("qwen-asr is not installed")
            raise RuntimeError(
                "qwen-asr is required for QwBaseer STT. Install with: uv pip install qwen-asr"
            ) from exc

        try:
            model = Qwen3ASRModel.from_pretrained(
                model_source,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                max_new_tokens=self.max_new_tokens,
                trust_remote_code=True,
            )
            return model
        except Exception as exc:
            self.logger.exception("Failed to load QwBaseer STT from %s", model_source)
            raise RuntimeError(
                f"Failed to load QwBaseer STT from {model_source}"
            ) from exc

    def _resolve_model_source(self) -> str:
        local_path = Path(self.model_name_or_path)
        if local_path.exists():
            return str(local_path)
        return DEFAULT_MODEL_ID

    def _prepare_audio(self, samples: torch.Tensor) -> torch.Tensor:
        audio = torch.as_tensor(samples, dtype=torch.float32)

        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            raise ValueError("ASRAudioChunk.samples must be 1D or 2D")

        return audio.contiguous()

    def _move_to_device(self, model_inputs: Any) -> Any:
        if hasattr(model_inputs, "to"):
            return model_inputs.to(self.device)

        if isinstance(model_inputs, dict):
            moved = {}
            for key, value in model_inputs.items():
                moved[key] = value.to(self.device) if hasattr(value, "to") else value
            return moved

        return model_inputs

    def _build_prompt(
        self, language_hint: Optional[str], return_timestamps: bool
    ) -> str:
        prompt_parts = [
            "Transcribe the provided Arabic-English speech faithfully.",
            "Keep code-switching as spoken.",
        ]

        if language_hint:
            prompt_parts.append(f"Language hint: {language_hint}.")

        if return_timestamps:
            prompt_parts.append(
                "If supported, include timestamps for each spoken segment."
            )

        return " ".join(prompt_parts)

    def _build_segments(
        self,
        decoded_text: str,
        chunk_start_sec: float,
        chunk_duration_sec: float,
    ) -> list[TranscriptionSegment]:
        parsed_segments = self._parse_timestamped_segments(decoded_text)
        if parsed_segments:
            return [
                TranscriptionSegment(
                    start_sec=chunk_start_sec + segment.start_sec,
                    end_sec=chunk_start_sec + segment.end_sec,
                    text=segment.text,
                )
                for segment in parsed_segments
            ]

        cleaned_text = decoded_text.strip()
        if not cleaned_text:
            return []

        return [
            TranscriptionSegment(
                start_sec=chunk_start_sec,
                end_sec=chunk_start_sec + chunk_duration_sec,
                text=cleaned_text,
            )
        ]

    def _parse_timestamped_segments(
        self, decoded_text: str
    ) -> list[TranscriptionSegment]:
        json_segments = self._parse_json_segments(decoded_text)
        if json_segments:
            return json_segments

        matches = TIMESTAMP_PATTERN.finditer(decoded_text)
        segments: list[TranscriptionSegment] = []
        for match in matches:
            text = match.group("text").strip()
            if not text:
                continue
            segments.append(
                TranscriptionSegment(
                    start_sec=float(match.group("start")),
                    end_sec=float(match.group("end")),
                    text=text,
                )
            )
        return segments

    def _parse_json_segments(self, decoded_text: str) -> list[TranscriptionSegment]:
        try:
            payload = json.loads(decoded_text)
        except json.JSONDecodeError:
            return []

        if isinstance(payload, dict):
            payload = payload.get("segments", [])

        if not isinstance(payload, list):
            return []

        segments: list[TranscriptionSegment] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = item.get("start", item.get("start_sec"))
            end = item.get("end", item.get("end_sec"))
            if start is None or end is None:
                continue
            segments.append(
                TranscriptionSegment(
                    start_sec=float(start),
                    end_sec=float(end),
                    text=text,
                )
            )
        return segments

    def _run_inference(
        self,
        *,
        audio: torch.Tensor,
        sample_rate: int,
        prompt: str,
        language_hint: Optional[str],
        return_timestamps: bool,
    ) -> str:
        audio_input = (audio.detach().cpu().numpy(), sample_rate)

        if hasattr(self.model, "transcribe"):
            result = self.model.transcribe(
                audio_input,
                context=prompt,
                language=language_hint,
                return_time_stamps=return_timestamps,
            )
            return str(result.text)

        if self.processor is not None and hasattr(self.model, "generate"):
            model_inputs = self.processor(
                text=prompt,
                audio=audio_input[0],
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            model_inputs = self._move_to_device(model_inputs)

            with torch.inference_mode():
                generated_ids = self.model.generate(**model_inputs)

            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            return str(decoded[0] if decoded else "")

        raise RuntimeError("ASR model must expose either `transcribe` or `generate` support")

    def _validate_chunk(self, chunk: ASRAudioChunk) -> None:
        if chunk.sample_rate != 16000:
            raise ValueError("QwBaseer STT expects 16 kHz audio input")
        if torch.as_tensor(chunk.samples).numel() == 0:
            raise ValueError("ASRAudioChunk.samples must not be empty")
