from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Iterator

import torch
import torchaudio.functional as audio_functional

from app.config import AppConfig
from app.dialect.camel_dialect import CamelDialectIdentifier
from app.llm.gemma_model import GemmaResponseGenerator
from app.llm.qwen_model import QwenResponseGenerator
from app.monitoring.metrics import MetricsRegistry
from app.normalization.arabic_normalizer import ArabicTextNormalizer
from app.pipeline.main_pipeline import Stage3ConditionedTextPipeline, Stage3ConditionedTextResult
from app.stt.asr_model import ASRAudioChunk, ASRResult, QwBaseerASR, TranscriptionSegment
from app.streaming.streamer import LowLatencyAudioStreamer, StreamingAudioChunk
from app.tts.tts_model import BilingualSpeechResult, QwenOmniTTSSynthesizer
from app.tts.tts_router import ResponseRoutingResult, ResponseTTSRouter
from app.utils.logger import get_logger
from app.cs_detection.xlmr_model import XLMRCodeSwitchDetector
from app.vad.silero_vad import AudioChunk, SileroVAD


@dataclass
class FrameDecision:
    frame_index: int
    start_sec: float
    end_sec: float
    is_speech: bool
    speech_probability: float


@dataclass
class DetectedSpeechTurn:
    turn_index: int
    start_sec: float
    end_sec: float
    sample_rate: int
    samples: torch.Tensor
    speech_frames: int
    total_frames: int


@dataclass
class VoiceActivityTrace:
    sample_rate: int
    frame_ms: int
    total_frames: int
    speech_frames: int
    turns: list[DetectedSpeechTurn] = field(default_factory=list)
    frame_decisions: list[FrameDecision] = field(default_factory=list)


@dataclass
class TranscriptionArtifacts:
    transcript_text: str
    sample_rate: int
    duration_sec: float
    segments: list[TranscriptionSegment]
    vad_trace: VoiceActivityTrace | None
    raw_results: list[ASRResult] = field(default_factory=list)


@dataclass
class TextResponseArtifacts:
    input_text: str
    stage_result: Stage3ConditionedTextResult
    response_text: str
    routing_result: ResponseRoutingResult


@dataclass
class EndToEndTurnResult:
    transcription: TranscriptionArtifacts
    text_response: TextResponseArtifacts
    speech_result: BilingualSpeechResult | None


class LazyPipelineProvider:
    """Thread-safe lazy initializer used by the API app."""

    def __init__(self, factory: Callable[[], "EndToEndSpeechPipeline"]) -> None:
        self.factory = factory
        self._pipeline: EndToEndSpeechPipeline | None = None
        self._lock = Lock()

    @property
    def is_initialized(self) -> bool:
        return self._pipeline is not None

    def get_pipeline(self) -> "EndToEndSpeechPipeline":
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self._pipeline = self.factory()
        return self._pipeline


class EndToEndSpeechPipeline:
    """Config-driven orchestration for VAD -> ASR -> text reasoning -> TTS -> stream."""

    def __init__(
        self,
        *,
        config: AppConfig,
        vad: SileroVAD,
        asr: QwBaseerASR,
        dialect_identifier: CamelDialectIdentifier,
        text_normalizer: ArabicTextNormalizer,
        code_switch_detector: XLMRCodeSwitchDetector,
        response_generator: Any,
        tts_router: ResponseTTSRouter,
        tts_synthesizer: QwenOmniTTSSynthesizer,
        streamer: LowLatencyAudioStreamer,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.config = config
        self.vad = vad
        self.asr = asr
        self.dialect_identifier = dialect_identifier
        self.text_normalizer = text_normalizer
        self.code_switch_detector = code_switch_detector
        self.response_generator = response_generator
        self.tts_router = tts_router
        self.tts_synthesizer = tts_synthesizer
        self.streamer = streamer
        self.metrics = metrics
        self.logger = get_logger(self.__class__.__name__)
        self.text_pipeline = Stage3ConditionedTextPipeline(
            dialect_identifier=self.dialect_identifier,
            text_normalizer=self.text_normalizer,
            code_switch_detector=self.code_switch_detector,
            response_generator=self.response_generator,
        )

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        metrics: MetricsRegistry | None = None,
    ) -> "EndToEndSpeechPipeline":
        llm_provider = (config.models.llm.provider or "qwen").strip().lower()
        if llm_provider == "gemma":
            response_generator = GemmaResponseGenerator(
                model_name_or_path=config.resolve_reference(
                    config.models.llm.model_name_or_path,
                    local_only=config.models.llm.local_files_only,
                ),
                device=config.models.llm.device,
                max_new_tokens=config.models.llm.max_new_tokens,
                temperature=config.models.llm.temperature,
                top_p=config.models.llm.top_p,
                local_files_only=config.models.llm.local_files_only,
            )
        else:
            response_generator = QwenResponseGenerator(
                model_name_or_path=config.resolve_reference(
                    config.models.llm.model_name_or_path,
                    local_only=config.models.llm.local_files_only,
                ),
                device=config.models.llm.device,
                max_new_tokens=config.models.llm.max_new_tokens,
                temperature=config.models.llm.temperature,
                top_p=config.models.llm.top_p,
                local_files_only=config.models.llm.local_files_only,
            )

        vad = SileroVAD(
            threshold=config.models.vad.threshold,
            device=config.models.vad.device,
        )
        asr = QwBaseerASR(
            model_name_or_path=config.resolve_reference(
                config.models.asr.model_name_or_path,
                local_only=config.models.asr.local_files_only,
            ),
            device=config.models.asr.device,
            max_new_tokens=config.models.asr.max_new_tokens,
            local_files_only=config.models.asr.local_files_only,
        )
        dialect_identifier = CamelDialectIdentifier(
            confidence_threshold=config.models.dialect.confidence_threshold,
            minimum_arabic_chars=config.models.dialect.minimum_arabic_chars,
        )
        text_normalizer = ArabicTextNormalizer()
        code_switch_detector = XLMRCodeSwitchDetector(
            model_name_or_path=config.resolve_reference(
                config.models.code_switch.model_name_or_path,
                local_only=False,
            ),
            device=config.models.code_switch.device,
            max_length=config.models.code_switch.max_length,
        )
        tts_synthesizer = QwenOmniTTSSynthesizer(
            model_name_or_path=config.resolve_reference(
                config.models.tts.model_name_or_path,
                local_only=config.models.tts.local_files_only,
            ),
            device=config.models.tts.device,
            local_files_only=config.models.tts.local_files_only,
            pause_duration_ms=config.models.tts.pause_duration_ms,
        )
        tts_router = ResponseTTSRouter(code_switch_detector=code_switch_detector)
        streamer = LowLatencyAudioStreamer(
            synthesizer=tts_synthesizer,
            chunk_duration_ms=config.pipeline.stream_chunk_duration_ms,
            first_chunk_duration_ms=config.pipeline.first_chunk_duration_ms,
            inter_segment_pause_ms=config.pipeline.inter_segment_pause_ms,
        )

        return cls(
            config=config,
            vad=vad,
            asr=asr,
            dialect_identifier=dialect_identifier,
            text_normalizer=text_normalizer,
            code_switch_detector=code_switch_detector,
            response_generator=response_generator,
            tts_router=tts_router,
            tts_synthesizer=tts_synthesizer,
            streamer=streamer,
            metrics=metrics,
        )

    def transcribe_audio(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        *,
        language_hint: str | None = None,
        apply_vad: bool = True,
    ) -> TranscriptionArtifacts:
        prepared_audio = self._prepare_audio(samples, sample_rate)
        prepared_sample_rate = self.config.audio.input_sample_rate
        vad_trace: VoiceActivityTrace | None = None
        turns: list[DetectedSpeechTurn]

        if apply_vad:
            vad_trace = self._timed(
                "vad",
                lambda: self._detect_speech_turns(
                    prepared_audio,
                    sample_rate=prepared_sample_rate,
                ),
            )
            turns = vad_trace.turns
        else:
            turns = [
                DetectedSpeechTurn(
                    turn_index=1,
                    start_sec=0.0,
                    end_sec=prepared_audio.numel() / float(prepared_sample_rate),
                    sample_rate=prepared_sample_rate,
                    samples=prepared_audio,
                    speech_frames=1,
                    total_frames=1,
                )
            ]

        if not turns:
            return TranscriptionArtifacts(
                transcript_text="",
                sample_rate=prepared_sample_rate,
                duration_sec=prepared_audio.numel() / float(prepared_sample_rate),
                segments=[],
                vad_trace=vad_trace,
                raw_results=[],
            )

        def _run_asr() -> tuple[list[ASRResult], list[TranscriptionSegment], str]:
            raw_results: list[ASRResult] = []
            combined_segments: list[TranscriptionSegment] = []
            transcript_parts: list[str] = []

            for turn in turns:
                result = self.asr.transcribe_chunk(
                    ASRAudioChunk(
                        samples=turn.samples,
                        sample_rate=turn.sample_rate,
                        chunk_start_sec=turn.start_sec,
                        chunk_id=f"turn-{turn.turn_index}",
                    ),
                    language_hint=language_hint or self.config.pipeline.language_hint,
                    return_timestamps=True,
                )
                raw_results.append(result)
                combined_segments.extend(result.segments)
                if result.text.strip():
                    transcript_parts.append(result.text.strip())

            return raw_results, combined_segments, " ".join(transcript_parts).strip()

        raw_results, segments, transcript_text = self._timed("asr", _run_asr)

        return TranscriptionArtifacts(
            transcript_text=transcript_text,
            sample_rate=prepared_sample_rate,
            duration_sec=prepared_audio.numel() / float(prepared_sample_rate),
            segments=segments,
            vad_trace=vad_trace,
            raw_results=raw_results,
        )

    def respond_to_text(
        self,
        text: str,
        *,
        task_instruction: str | None = None,
    ) -> TextResponseArtifacts:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        stage_result = self._timed(
            "text_pipeline",
            lambda: self.text_pipeline.generate_response(
                text,
                task_instruction=task_instruction or self.config.pipeline.task_instruction,
            ),
        )
        response_text = self._extract_response_text(stage_result.llm_response)
        routing_result = self._timed(
            "tts_routing",
            lambda: self.tts_router.route_response(
                text=response_text,
                dialect_signal=stage_result.dialect_signal,
            ),
        )

        return TextResponseArtifacts(
            input_text=text,
            stage_result=stage_result,
            response_text=response_text,
            routing_result=routing_result,
        )

    def synthesize_response(
        self,
        response: TextResponseArtifacts,
    ) -> BilingualSpeechResult:
        return self._timed(
            "tts_synthesis",
            lambda: self.tts_synthesizer.synthesize_routed_response(
                response.routing_result
            ),
        )

    def run_audio_turn(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        *,
        language_hint: str | None = None,
        task_instruction: str | None = None,
        apply_vad: bool = True,
        synthesize_audio: bool = True,
    ) -> EndToEndTurnResult:
        transcription = self.transcribe_audio(
            samples,
            sample_rate,
            language_hint=language_hint,
            apply_vad=apply_vad,
        )
        if not transcription.transcript_text:
            raise ValueError("No speech was detected in the provided audio")

        text_response = self.respond_to_text(
            transcription.transcript_text,
            task_instruction=task_instruction,
        )
        speech_result = self.synthesize_response(text_response) if synthesize_audio else None
        return EndToEndTurnResult(
            transcription=transcription,
            text_response=text_response,
            speech_result=speech_result,
        )

    def iter_audio_chunks(
        self,
        response: TextResponseArtifacts,
    ) -> Iterator[StreamingAudioChunk]:
        started = perf_counter()
        success = False
        try:
            for chunk in self.streamer.stream_routed_response(response.routing_result):
                yield chunk
            success = True
        finally:
            self._record_metric(
                "tts_stream",
                duration_ms=(perf_counter() - started) * 1000.0,
                success=success,
            )

    def _prepare_audio(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = torch.as_tensor(samples, dtype=torch.float32).detach().cpu()
        if waveform.ndim == 2:
            if waveform.shape[0] <= waveform.shape[1]:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.mean(dim=1)
        elif waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        if sample_rate != self.config.audio.input_sample_rate:
            waveform = audio_functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.config.audio.input_sample_rate,
            )

        return waveform.contiguous()

    def _detect_speech_turns(
        self,
        samples: torch.Tensor,
        *,
        sample_rate: int,
    ) -> VoiceActivityTrace:
        frame_samples = max(
            1, int(sample_rate * self.config.audio.frame_ms / 1000.0)
        )
        end_silence_frames = max(
            1, int(self.config.audio.end_silence_ms / self.config.audio.frame_ms)
        )
        min_speech_frames = max(
            1, int(self.config.audio.min_speech_ms / self.config.audio.frame_ms)
        )
        max_turn_samples = max(1, int(self.config.audio.max_utterance_s * sample_rate))

        frame_decisions: list[FrameDecision] = []
        turns: list[DetectedSpeechTurn] = []
        current_frames: list[torch.Tensor] = []
        speech_frames = 0
        silence_frames = 0
        total_speech_frames = 0
        turn_start_frame = 0
        in_turn = False

        for frame_index, frame_start in enumerate(range(0, samples.numel(), frame_samples), start=1):
            frame_end = min(frame_start + frame_samples, samples.numel())
            frame = samples[frame_start:frame_end]
            vad_result = self.vad.process_chunk(
                AudioChunk(samples=frame, sample_rate=sample_rate, chunk_id=str(frame_index))
            )
            frame_decisions.append(
                FrameDecision(
                    frame_index=frame_index,
                    start_sec=frame_start / float(sample_rate),
                    end_sec=frame_end / float(sample_rate),
                    is_speech=vad_result.is_speech,
                    speech_probability=vad_result.speech_probability,
                )
            )

            if not in_turn:
                if not vad_result.is_speech:
                    continue

                in_turn = True
                current_frames = [frame]
                speech_frames = 1
                silence_frames = 0
                turn_start_frame = frame_index
                total_speech_frames += 1
                continue

            current_frames.append(frame)
            if vad_result.is_speech:
                speech_frames += 1
                total_speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            turn_sample_count = sum(piece.numel() for piece in current_frames)
            should_end = (
                silence_frames >= end_silence_frames
                or turn_sample_count >= max_turn_samples
            )
            if not should_end:
                continue

            maybe_turn = self._finalize_detected_turn(
                turns=turns,
                current_frames=current_frames,
                speech_frames=speech_frames,
                min_speech_frames=min_speech_frames,
                sample_rate=sample_rate,
                turn_start_frame=turn_start_frame,
            )
            if maybe_turn is not None:
                turns.append(maybe_turn)

            current_frames = []
            speech_frames = 0
            silence_frames = 0
            in_turn = False

        if in_turn and current_frames:
            maybe_turn = self._finalize_detected_turn(
                turns=turns,
                current_frames=current_frames,
                speech_frames=speech_frames,
                min_speech_frames=min_speech_frames,
                sample_rate=sample_rate,
                turn_start_frame=turn_start_frame,
            )
            if maybe_turn is not None:
                turns.append(maybe_turn)

        return VoiceActivityTrace(
            sample_rate=sample_rate,
            frame_ms=self.config.audio.frame_ms,
            total_frames=len(frame_decisions),
            speech_frames=total_speech_frames,
            turns=turns,
            frame_decisions=frame_decisions,
        )

    def _finalize_detected_turn(
        self,
        *,
        turns: list[DetectedSpeechTurn],
        current_frames: list[torch.Tensor],
        speech_frames: int,
        min_speech_frames: int,
        sample_rate: int,
        turn_start_frame: int,
    ) -> DetectedSpeechTurn | None:
        if speech_frames < min_speech_frames:
            return None

        waveform = torch.cat(current_frames, dim=0).contiguous()
        start_sec = (turn_start_frame - 1) * self.config.audio.frame_ms / 1000.0
        end_sec = start_sec + waveform.numel() / float(sample_rate)
        return DetectedSpeechTurn(
            turn_index=len(turns) + 1,
            start_sec=start_sec,
            end_sec=end_sec,
            sample_rate=sample_rate,
            samples=waveform,
            speech_frames=speech_frames,
            total_frames=len(current_frames),
        )

    def _extract_response_text(self, llm_response: Any) -> str:
        response_text = getattr(llm_response, "response_text", "")
        if not response_text or not str(response_text).strip():
            raise RuntimeError("LLM returned an empty response")
        return str(response_text).strip()

    def _timed(self, stage: str, operation: Callable[[], Any]) -> Any:
        started = perf_counter()
        success = False
        try:
            result = operation()
            success = True
            return result
        finally:
            self._record_metric(
                stage,
                duration_ms=(perf_counter() - started) * 1000.0,
                success=success,
            )

    def _record_metric(self, stage: str, *, duration_ms: float, success: bool) -> None:
        if self.metrics is not None:
            self.metrics.record_stage(stage, duration_ms=duration_ms, success=success)
        if not success:
            self.logger.warning("Stage %s failed", stage)
