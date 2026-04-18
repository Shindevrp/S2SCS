from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from app.tts.tts_model import QwenOmniTTSSynthesizer, SynthesizedSegment, TTSSynthesisRequest
from app.tts.tts_router import ResponseRoutingResult
from app.utils.logger import get_logger


@dataclass
class StreamingAudioChunk:
    waveform: torch.Tensor
    sample_rate: int
    chunk_index: int
    segment_index: Optional[int]
    stream_start_sample: int
    stream_end_sample: int
    segment_start_sample: int
    segment_end_sample: int
    text: str
    voice_id: Optional[str]
    is_first_chunk: bool
    is_last_chunk: bool
    is_stream_end: bool


class LowLatencyAudioStreamer:
    """Streams synthesized speech as low-latency audio chunks."""

    def __init__(
        self,
        synthesizer: Optional[QwenOmniTTSSynthesizer] = None,
        chunk_duration_ms: int = 120,
        first_chunk_duration_ms: int = 40,
        inter_segment_pause_ms: int = 0,
    ) -> None:
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")
        if first_chunk_duration_ms <= 0:
            raise ValueError("first_chunk_duration_ms must be positive")
        if inter_segment_pause_ms < 0:
            raise ValueError("inter_segment_pause_ms must be non-negative")

        self.synthesizer = synthesizer
        self.chunk_duration_ms = chunk_duration_ms
        self.first_chunk_duration_ms = min(first_chunk_duration_ms, chunk_duration_ms)
        self.inter_segment_pause_ms = inter_segment_pause_ms
        self.logger = get_logger(self.__class__.__name__)

    def stream_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        segment_index: Optional[int] = None,
        text: str = "",
        voice_id: Optional[str] = None,
        stream_offset_samples: int = 0,
        is_final_segment: bool = True,
    ) -> Iterator[StreamingAudioChunk]:
        normalized_waveform = self._normalize_waveform(waveform)
        if normalized_waveform.numel() == 0:
            return

        chunk_samples = self._duration_to_samples(sample_rate, self.chunk_duration_ms)
        first_chunk_samples = self._duration_to_samples(sample_rate, self.first_chunk_duration_ms)

        local_start = 0
        chunk_index = 0
        while local_start < normalized_waveform.numel():
            current_chunk_size = first_chunk_samples if chunk_index == 0 else chunk_samples
            local_end = min(local_start + current_chunk_size, normalized_waveform.numel())

            yield StreamingAudioChunk(
                waveform=normalized_waveform[local_start:local_end].clone(),
                sample_rate=sample_rate,
                chunk_index=chunk_index,
                segment_index=segment_index,
                stream_start_sample=stream_offset_samples + local_start,
                stream_end_sample=stream_offset_samples + local_end,
                segment_start_sample=local_start,
                segment_end_sample=local_end,
                text=text,
                voice_id=voice_id,
                is_first_chunk=chunk_index == 0,
                is_last_chunk=local_end >= normalized_waveform.numel(),
                is_stream_end=is_final_segment and local_end >= normalized_waveform.numel(),
            )

            local_start = local_end
            chunk_index += 1

    def stream_synthesized_segment(
        self,
        segment: SynthesizedSegment,
        *,
        stream_offset_samples: int = 0,
        is_final_segment: bool = True,
    ) -> Iterator[StreamingAudioChunk]:
        yield from self.stream_waveform(
            waveform=segment.waveform,
            sample_rate=segment.sample_rate,
            segment_index=segment.segment_index,
            text=segment.text,
            voice_id=segment.voice.voice_id,
            stream_offset_samples=stream_offset_samples,
            is_final_segment=is_final_segment,
        )

    def stream_routed_response(
        self,
        routing_result: ResponseRoutingResult,
    ) -> Iterator[StreamingAudioChunk]:
        if self.synthesizer is None:
            raise RuntimeError(
                "No synthesizer was provided. Initialize LowLatencyAudioStreamer with "
                "a QwenOmniTTSSynthesizer to stream routed responses."
            )

        stream_offset_samples = 0
        for segment_index, segment in enumerate(routing_result.segments):
            synthesized = self.synthesizer.synthesize_segment(
                TTSSynthesisRequest(
                    text=segment.text,
                    voice=segment.voice,
                    segment_index=segment_index,
                    start_char=segment.start_char,
                    end_char=segment.end_char,
                )
            )

            is_final_segment = segment_index == len(routing_result.segments) - 1
            yield from self.stream_synthesized_segment(
                synthesized,
                stream_offset_samples=stream_offset_samples,
                is_final_segment=is_final_segment,
            )
            stream_offset_samples += synthesized.waveform.numel()

            if segment_index < len(routing_result.segments) - 1 and self.inter_segment_pause_ms > 0:
                pause_waveform = self._build_pause(
                    sample_rate=synthesized.sample_rate,
                    duration_ms=self.inter_segment_pause_ms,
                )
                yield from self.stream_waveform(
                    waveform=pause_waveform,
                    sample_rate=synthesized.sample_rate,
                    segment_index=None,
                    text="",
                    voice_id=None,
                    stream_offset_samples=stream_offset_samples,
                    is_final_segment=False,
                )
                stream_offset_samples += pause_waveform.numel()

        self.logger.debug(
            "streamed_segments=%s total_stream_samples=%s",
            len(routing_result.segments),
            stream_offset_samples,
        )

    def _duration_to_samples(self, sample_rate: int, duration_ms: int) -> int:
        return max(1, int(sample_rate * duration_ms / 1000.0))

    def _build_pause(self, sample_rate: int, duration_ms: int) -> torch.Tensor:
        return torch.zeros(self._duration_to_samples(sample_rate, duration_ms), dtype=torch.float32)

    def _normalize_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        normalized = torch.as_tensor(waveform, dtype=torch.float32).detach().cpu()
        if normalized.ndim != 1:
            normalized = normalized.reshape(-1)
        return normalized.contiguous()
