from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from app.cs_detection.xlmr_model import XLMRCodeSwitchDetector
from app.dialect.camel_dialect import CamelDialectIdentifier
from app.llm.qwen_model import QwenResponseGenerator
from app.normalization.arabic_normalizer import ArabicTextNormalizer
from app.pipeline.target_streaming_pipeline import StreamingReasoningOrchestrator
from app.streaming.streamer import LowLatencyAudioStreamer
from app.stt.asr_model import ASRAudioChunk
from app.stt.asr_model import QwBaseerASR
from app.tts.tts_model import QwenOmniTTSSynthesizer
from app.tts.tts_router import ResponseTTSRouter
from app.utils.logger import get_logger
from app.vad.silero_vad import AudioChunk
from app.vad.silero_vad import SileroVAD


@dataclass
class LiveState:
    in_utterance: bool = False
    silence_frames: int = 0
    speech_frames: int = 0


@dataclass
class StreamingTurnBuffers:
    frames: list[np.ndarray]
    asr_buffer: list[np.ndarray]
    transcript_segments: list[str]
    asr_consumed_samples: int

    @classmethod
    def empty(cls) -> "StreamingTurnBuffers":
        return cls(frames=[], asr_buffer=[], transcript_segments=[], asr_consumed_samples=0)


class LiveVoiceConverter:
    """Streaming-first live voice conversion using mic input and chunked playback."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 32,
        end_silence_ms: int = 700,
        min_speech_ms: int = 350,
        max_utterance_s: float = 12.0,
        asr_chunk_ms: int = 1200,
        sliding_window_tokens: int = 48,
        max_context_updates: int = 8,
        llm_streaming: bool = True,
        task_instruction: str = "Generate a natural conversational response.",
        language_hint: Optional[str] = None,
    ) -> None:
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = max(1, int(sample_rate * frame_ms / 1000.0))
        self.end_silence_frames = max(1, int(end_silence_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_utterance_samples = int(max_utterance_s * sample_rate)
        self.asr_chunk_ms = asr_chunk_ms
        self.asr_chunk_samples = max(1, int(sample_rate * asr_chunk_ms / 1000.0))
        self.llm_streaming = llm_streaming
        self.task_instruction = task_instruction
        self.language_hint = language_hint

        self.logger = get_logger(self.__class__.__name__)
        self.vad = SileroVAD(threshold=0.5)
        self.asr = QwBaseerASR(local_files_only=True)
        self.dialect_identifier = CamelDialectIdentifier()
        self.text_normalizer = ArabicTextNormalizer()
        self.code_switch_detector = XLMRCodeSwitchDetector()
        self.response_generator = QwenResponseGenerator()
        self.tts_router = ResponseTTSRouter(code_switch_detector=self.code_switch_detector)
        self.tts_synthesizer = QwenOmniTTSSynthesizer()
        self.streamer = LowLatencyAudioStreamer(
            synthesizer=self.tts_synthesizer,
            chunk_duration_ms=120,
            first_chunk_duration_ms=40,
            inter_segment_pause_ms=0,
        )

        self.streaming_reasoning = StreamingReasoningOrchestrator(
            dialect_identifier=self.dialect_identifier,
            text_normalizer=self.text_normalizer,
            code_switch_detector=self.code_switch_detector,
            response_generator=self.response_generator,
            sliding_window_tokens=sliding_window_tokens,
            max_context_updates=max_context_updates,
        )

    def run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for live conversion. Install with `uv pip install sounddevice`."
            ) from exc

        self.logger.info("Starting live voice conversion at %s Hz", self.sample_rate)
        self.logger.info("Speak naturally. Press Ctrl+C to stop.")

        state = LiveState()
        buffers = StreamingTurnBuffers.empty()

        while True:
            frame = sd.rec(
                self.frame_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            ).reshape(-1)

            vad_result = self.vad.process_chunk(
                AudioChunk(
                    samples=torch.from_numpy(frame),
                    sample_rate=self.sample_rate,
                )
            )

            if not state.in_utterance:
                if vad_result.is_speech:
                    state.in_utterance = True
                    state.silence_frames = 0
                    state.speech_frames = 1
                    buffers = StreamingTurnBuffers.empty()
                    buffers.frames.append(frame)
                continue

            buffers.frames.append(frame)
            buffers.asr_buffer.append(frame)
            if vad_result.is_speech:
                state.speech_frames += 1
                state.silence_frames = 0
            else:
                state.silence_frames += 1

            if self._ready_for_partial_asr(buffers):
                self._process_partial_transcript_update(buffers)

            utterance_samples = sum(piece.shape[0] for piece in buffers.frames)
            reached_end_silence = state.silence_frames >= self.end_silence_frames
            reached_max_len = utterance_samples >= self.max_utterance_samples

            if not reached_end_silence and not reached_max_len:
                continue

            if state.speech_frames < self.min_speech_frames:
                self.streaming_reasoning.reset_turn()
                buffers = StreamingTurnBuffers.empty()
                state = LiveState()
                continue

            self._process_partial_transcript_update(buffers)
            transcript_text = " ".join(buffers.transcript_segments).strip()
            buffers = StreamingTurnBuffers.empty()
            state = LiveState()

            self._finalize_turn(transcript_text=transcript_text, sd=sd)

    def _finalize_turn(self, transcript_text: str, sd) -> None:
        started = time.perf_counter()

        if not transcript_text:
            self.logger.warning("No partial transcript captured for this turn")
            self.streaming_reasoning.reset_turn()
            return

        try:
            prompt_input = self.streaming_reasoning.build_turn_prompt_input(
                task_instruction=self.task_instruction,
            )
            dialect_signal = prompt_input.dialect_signal

            if self.llm_streaming and hasattr(self.response_generator, "stream_response"):
                response_text = self._play_streaming_llm_response(
                    prompt_input=prompt_input,
                    dialect_signal=dialect_signal,
                    sd=sd,
                )
            else:
                turn_response = self.response_generator.generate_response(prompt_input)
                response_text = turn_response.response_text.strip()
                self._play_full_text_response(
                    response_text=response_text,
                    dialect_signal=dialect_signal,
                    sd=sd,
                )

            if not response_text:
                raise RuntimeError("LLM returned empty response text")
        except Exception as exc:
            self.logger.warning("Utterance processing failed: %s", exc)
            self.streaming_reasoning.reset_turn()
            return

        elapsed = time.perf_counter() - started
        self.logger.info("User: %s", transcript_text)
        self.logger.info("Assistant: %s", response_text)
        self.logger.info("Turn latency: %.2fs", elapsed)

        self.streaming_reasoning.commit_assistant_response(response_text)
        self.streaming_reasoning.reset_turn()

    def _play_streaming_llm_response(self, prompt_input, dialect_signal, sd) -> str:
        response_text, chunks = self.response_generator.stream_response(prompt_input)

        pending = ""
        for chunk in chunks:
            pending += chunk.text
            completed_units, pending = self._extract_speakable_units(pending)
            for unit in completed_units:
                self._speak_text_unit(unit, dialect_signal=dialect_signal, sd=sd)

        if pending.strip():
            self._speak_text_unit(pending.strip(), dialect_signal=dialect_signal, sd=sd)

        return response_text

    def _play_full_text_response(self, response_text: str, dialect_signal, sd) -> None:
        self._speak_text_unit(response_text, dialect_signal=dialect_signal, sd=sd)

    def _speak_text_unit(self, text: str, dialect_signal, sd) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        routing_result = self.tts_router.route_response(
            text=cleaned,
            dialect_signal=dialect_signal,
        )
        if not routing_result.segments:
            return

        for chunk in self.streamer.stream_routed_response(routing_result):
            audio = chunk.waveform.detach().cpu().numpy().astype(np.float32, copy=False)
            if audio.size == 0:
                continue
            sd.play(audio, samplerate=chunk.sample_rate, blocking=True)

    def _extract_speakable_units(self, text: str) -> tuple[list[str], str]:
        stripped = text.strip()
        if not stripped:
            return [], ""

        # Emit sentence-like units to overlap TTS with ongoing token generation.
        chunks = re.split(r"([.!?\n،؛]+)", stripped)
        completed_units: list[str] = []
        current = ""
        for part in chunks:
            if not part:
                continue
            current += part
            if re.fullmatch(r"[.!?\n،؛]+", part):
                if current.strip():
                    completed_units.append(current.strip())
                current = ""

        return completed_units, current

    def _ready_for_partial_asr(self, buffers: StreamingTurnBuffers) -> bool:
        if not buffers.asr_buffer:
            return False

        buffered_samples = sum(piece.shape[0] for piece in buffers.asr_buffer)
        return buffered_samples >= self.asr_chunk_samples

    def _process_partial_transcript_update(self, buffers: StreamingTurnBuffers) -> None:
        if not buffers.asr_buffer:
            return

        chunk_audio = np.concatenate(buffers.asr_buffer, axis=0)
        buffers.asr_buffer = []
        chunk_waveform = torch.from_numpy(chunk_audio.astype(np.float32, copy=False)).contiguous()
        chunk_start_sec = buffers.asr_consumed_samples / float(self.sample_rate)
        buffers.asr_consumed_samples += int(chunk_waveform.numel())

        try:
            asr_result = self.asr.transcribe_chunk(
                ASRAudioChunk(
                    samples=chunk_waveform,
                    sample_rate=self.sample_rate,
                    chunk_start_sec=chunk_start_sec,
                ),
                language_hint=self.language_hint,
                return_timestamps=False,
            )
        except Exception as exc:
            self.logger.warning("Partial ASR failed: %s", exc)
            return

        if not asr_result.text.strip():
            return

        buffers.transcript_segments.append(asr_result.text.strip())
        partial_transcript = " ".join(buffers.transcript_segments).strip()
        update = self.streaming_reasoning.process_partial_transcript(partial_transcript)
        self.logger.info(
            "Streaming update=%s csi=%.3f matrix=%s partial_chars=%s",
            update.update_index,
            update.code_switch_metrics.cs_index,
            update.code_switch_metrics.matrix_language,
            len(update.partial_transcript),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live microphone speech-to-speech conversion")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--frame-ms", type=int, default=32, help="Mic frame duration in ms")
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=700,
        help="Silence duration that ends a speaking turn",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=350,
        help="Minimum speech duration to accept a turn",
    )
    parser.add_argument(
        "--max-utterance-s",
        type=float,
        default=12.0,
        help="Maximum single utterance length in seconds",
    )
    parser.add_argument(
        "--asr-chunk-ms",
        type=int,
        default=1200,
        help="Partial ASR chunk duration in milliseconds",
    )
    parser.add_argument(
        "--window-tokens",
        type=int,
        default=48,
        help="Sliding-window token count for incremental XLM-R labeling",
    )
    parser.add_argument(
        "--max-context-updates",
        type=int,
        default=8,
        help="Maximum incremental updates stored in context buffer",
    )
    parser.add_argument(
        "--no-llm-streaming",
        action="store_true",
        help="Disable token-streaming LLM output and use full-response playback",
    )
    parser.add_argument(
        "--task",
        default="Generate a natural conversational response.",
        help="Task instruction for Stage 7 response generation",
    )
    parser.add_argument(
        "--language-hint",
        default=None,
        help="Optional ASR language hint (example: ar-en)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = LiveVoiceConverter(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        end_silence_ms=args.end_silence_ms,
        min_speech_ms=args.min_speech_ms,
        max_utterance_s=args.max_utterance_s,
        asr_chunk_ms=args.asr_chunk_ms,
        sliding_window_tokens=args.window_tokens,
        max_context_updates=args.max_context_updates,
        llm_streaming=not args.no_llm_streaming,
        task_instruction=args.task,
        language_hint=args.language_hint,
    )
    app.run()


if __name__ == "__main__":
    main()
