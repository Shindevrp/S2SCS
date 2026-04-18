from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torchaudio

from app.cs_detection.xlmr_model import XLMRCodeSwitchDetector
from app.dialect.camel_dialect import CamelDialectIdentifier
from app.llm.qwen_model import QwenResponseGenerator
from app.normalization.arabic_normalizer import ArabicTextNormalizer
from app.pipeline.main_pipeline import Stage3ConditionedTextPipeline
from app.streaming.streamer import LowLatencyAudioStreamer, StreamingAudioChunk
from app.stt.asr_model import ASRAudioChunk, QwBaseerASR
from app.tts.tts_model import BilingualSpeechResult, QwenOmniTTSSynthesizer
from app.tts.tts_router import ResponseRoutingResult, ResponseTTSRouter
from app.utils.logger import get_logger
from app.vad.silero_vad import AudioChunk, SileroVAD


TARGET_STT_SAMPLE_RATE = 16000


@dataclass
class STSRunResult:
    transcript_text: str
    response_text: str
    routing_result: ResponseRoutingResult
    speech_result: BilingualSpeechResult
    streamed_chunks: list[StreamingAudioChunk]


class EndToEndSTSPipeline:
    """Audio-in to streamed-audio-out STS orchestration pipeline."""

    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        asr: Optional[QwBaseerASR] = None,
        dialect_identifier: Optional[CamelDialectIdentifier] = None,
        text_normalizer: Optional[ArabicTextNormalizer] = None,
        code_switch_detector: Optional[XLMRCodeSwitchDetector] = None,
        response_generator: Optional[QwenResponseGenerator] = None,
        tts_router: Optional[ResponseTTSRouter] = None,
        tts_synthesizer: Optional[QwenOmniTTSSynthesizer] = None,
        streamer: Optional[LowLatencyAudioStreamer] = None,
        chunk_duration_ms: int = 1200,
    ) -> None:
        if chunk_duration_ms <= 0:
            raise ValueError("chunk_duration_ms must be positive")

        self.logger = get_logger(self.__class__.__name__)
        self.chunk_duration_ms = chunk_duration_ms

        self.vad = vad or SileroVAD(threshold=0.5)
        self.asr = asr or QwBaseerASR(local_files_only=True)
        self.dialect_identifier = dialect_identifier or CamelDialectIdentifier()
        self.text_normalizer = text_normalizer or ArabicTextNormalizer()
        self.code_switch_detector = code_switch_detector or XLMRCodeSwitchDetector()
        self.response_generator = response_generator or QwenResponseGenerator()

        self.text_pipeline = Stage3ConditionedTextPipeline(
            dialect_identifier=self.dialect_identifier,
            text_normalizer=self.text_normalizer,
            code_switch_detector=self.code_switch_detector,
            response_generator=self.response_generator,
        )

        self.tts_router = tts_router or ResponseTTSRouter(
            code_switch_detector=self.code_switch_detector
        )
        self.tts_synthesizer = tts_synthesizer or QwenOmniTTSSynthesizer()
        self.streamer = streamer or LowLatencyAudioStreamer(
            synthesizer=self.tts_synthesizer,
            chunk_duration_ms=120,
            first_chunk_duration_ms=40,
            inter_segment_pause_ms=0,
        )

    def run_file(
        self,
        input_audio_path: str | Path,
        *,
        task_instruction: str = "Generate a natural conversational response.",
        language_hint: Optional[str] = None,
    ) -> STSRunResult:
        waveform, sample_rate = self._load_audio(input_audio_path)
        return self.run_waveform(
            waveform=waveform,
            sample_rate=sample_rate,
            task_instruction=task_instruction,
            language_hint=language_hint,
        )

    def run_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        task_instruction: str = "Generate a natural conversational response.",
        language_hint: Optional[str] = None,
        use_vad_in_asr: bool = True,
    ) -> STSRunResult:
        stt_waveform = self._prepare_for_stt(waveform, sample_rate)
        transcript_text = self._transcribe_speech_chunks(
            waveform=stt_waveform,
            sample_rate=TARGET_STT_SAMPLE_RATE,
            language_hint=language_hint,
            use_vad=use_vad_in_asr,
        )
        if not transcript_text:
            raise RuntimeError("No speech transcript produced from the provided audio.")

        text_stage = self.text_pipeline.generate_response(
            transcript_text=transcript_text,
            task_instruction=task_instruction,
        )
        response_text = text_stage.llm_response.response_text.strip()
        if not response_text:
            raise RuntimeError("LLM response is empty; cannot continue to TTS.")

        routing_result = self.tts_router.route_response(
            text=response_text,
            dialect_signal=text_stage.dialect_signal,
        )

        if not routing_result.segments:
            raise RuntimeError("TTS routing produced no segments to synthesize.")

        speech_result = self.tts_synthesizer.synthesize_routed_response(routing_result)
        streamed_chunks = list(self.streamer.stream_routed_response(routing_result))

        self.logger.info(
            "STS complete transcript_chars=%s response_chars=%s segments=%s chunks=%s",
            len(transcript_text),
            len(response_text),
            len(routing_result.segments),
            len(streamed_chunks),
        )

        return STSRunResult(
            transcript_text=transcript_text,
            response_text=response_text,
            routing_result=routing_result,
            speech_result=speech_result,
            streamed_chunks=streamed_chunks,
        )

    def save_waveform(
        self, waveform: torch.Tensor, sample_rate: int, output_path: str | Path
    ) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tensor = (
            torch.as_tensor(waveform, dtype=torch.float32).detach().cpu().reshape(1, -1)
        )
        torchaudio.save(str(output), tensor, sample_rate)

    def _transcribe_speech_chunks(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        language_hint: Optional[str],
        use_vad: bool,
    ) -> str:
        chunk_samples = max(1, int(sample_rate * self.chunk_duration_ms / 1000.0))
        pieces: list[str] = []

        chunk_index = 0
        for start in range(0, waveform.numel(), chunk_samples):
            end = min(start + chunk_samples, waveform.numel())
            chunk_waveform = waveform[start:end]
            chunk_id = f"chunk-{chunk_index}"

            if use_vad:
                vad_result = self.vad.process_chunk(
                    AudioChunk(
                        samples=chunk_waveform,
                        sample_rate=sample_rate,
                        chunk_id=chunk_id,
                    )
                )
                if not vad_result.is_speech:
                    chunk_index += 1
                    continue

            chunk_start_sec = start / float(sample_rate)
            asr_result = self.asr.transcribe_chunk(
                ASRAudioChunk(
                    samples=chunk_waveform,
                    sample_rate=sample_rate,
                    chunk_id=chunk_id,
                    chunk_start_sec=chunk_start_sec,
                ),
                language_hint=language_hint,
                return_timestamps=True,
            )

            if asr_result.text.strip():
                pieces.append(asr_result.text.strip())

            chunk_index += 1

        transcript = " ".join(pieces).strip()
        self.logger.info(
            "Transcription complete chunks=%s transcript_chars=%s",
            chunk_index,
            len(transcript),
        )
        return transcript

    def _prepare_for_stt(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        audio = torch.as_tensor(waveform, dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        elif audio.ndim != 1:
            raise ValueError("Expected mono or multi-channel waveform tensor")

        audio = audio.contiguous()
        if sample_rate == TARGET_STT_SAMPLE_RATE:
            return audio

        resampled = torchaudio.functional.resample(
            audio,
            orig_freq=sample_rate,
            new_freq=TARGET_STT_SAMPLE_RATE,
        )
        return resampled.contiguous()

    def _load_audio(self, input_audio_path: str | Path) -> tuple[torch.Tensor, int]:
        path = Path(input_audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, sample_rate = torchaudio.load(str(path))
        return waveform, int(sample_rate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full speech-to-speech pipeline"
    )
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument(
        "--output",
        default="outputs/tts_response.wav",
        help="Output path for synthesized full-waveform audio",
    )
    parser.add_argument(
        "--task",
        default="Generate a natural conversational response.",
        help="Task instruction for Stage 7 response generation",
    )
    parser.add_argument(
        "--language-hint",
        default=None,
        help="Optional hint passed to ASR (example: ar-en)",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=1200,
        help="VAD+ASR chunk size in milliseconds",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = get_logger("run_pipeline")

    pipeline = EndToEndSTSPipeline(chunk_duration_ms=args.chunk_ms)
    result = pipeline.run_file(
        input_audio_path=args.input,
        task_instruction=args.task,
        language_hint=args.language_hint,
    )

    pipeline.save_waveform(
        waveform=result.speech_result.waveform,
        sample_rate=result.speech_result.sample_rate,
        output_path=args.output,
    )

    logger.info("Transcript: %s", result.transcript_text)
    logger.info("Response: %s", result.response_text)
    logger.info(
        "Routing segments=%s stream_chunks=%s output=%s",
        len(result.routing_result.segments),
        len(result.streamed_chunks),
        args.output,
    )


if __name__ == "__main__":
    main()
