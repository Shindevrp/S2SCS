import io
from contextlib import asynccontextmanager
import wave

import httpx
import pytest
import torch

from app.api.main import create_app
from app.config import load_app_config
from app.cs_detection.cs_features import TokenPrediction, analyze_code_switch
from app.dialect.camel_dialect import DialectSignal
from app.llm.prompt_builder import ResponsePromptInput
from app.monitoring.metrics import MetricsRegistry
from app.normalization.arabic_normalizer import NormalizationResult
from app.pipeline.e2e_pipeline import (
    DetectedSpeechTurn,
    TextResponseArtifacts,
    TranscriptionArtifacts,
    VoiceActivityTrace,
)
from app.pipeline.main_pipeline import Stage3ConditionedTextResult
from app.stt.asr_model import TranscriptionSegment
from app.streaming.streamer import StreamingAudioChunk
from app.tts.tts_model import BilingualSpeechResult, SynthesizedSegment, TTSVoiceRegistry
from app.tts.tts_router import ResponseRoutingResult, RoutedTTSSegment
from app.cs_detection.xlmr_model import CodeSwitchResult


class FakePipelineProvider:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.is_initialized = True

    def get_pipeline(self):
        return self.pipeline


class FakePipeline:
    def __init__(self) -> None:
        self.registry = TTSVoiceRegistry()

    def transcribe_audio(self, samples, sample_rate, **kwargs) -> TranscriptionArtifacts:
        return TranscriptionArtifacts(
            transcript_text="مرحبا hello",
            sample_rate=16000,
            duration_sec=1.0,
            segments=[
                TranscriptionSegment(start_sec=0.0, end_sec=1.0, text="مرحبا hello")
            ],
            vad_trace=VoiceActivityTrace(
                sample_rate=16000,
                frame_ms=32,
                total_frames=10,
                speech_frames=6,
                turns=[
                    DetectedSpeechTurn(
                        turn_index=1,
                        start_sec=0.0,
                        end_sec=1.0,
                        sample_rate=16000,
                        samples=torch.ones(16000),
                        speech_frames=6,
                        total_frames=10,
                    )
                ],
            ),
        )

    def respond_to_text(self, text: str, **kwargs) -> TextResponseArtifacts:
        signal = DialectSignal(
            conditioning_label="Gulf",
            confidence=0.91,
            raw_label="RIY",
            raw_city="Riyadh",
            raw_country="Saudi Arabia",
            raw_region="Gulf",
            normalized_text="مرحبا",
            bucket_scores={"MSA": 0.05, "Gulf": 0.9, "Hejazi": 0.05},
            is_fallback=False,
        )
        normalization = NormalizationResult(
            original_text=text,
            normalized_text="مرحبا hello",
            dialect_label="Gulf",
            applied_rules=["normalize_alef"],
        )
        predictions = [
            TokenPrediction("مرحبا", "AR", 0.97, 0, 5, "Gulf"),
            TokenPrediction("hello", "EN", 0.98, 6, 11, "Gulf"),
        ]
        metrics = analyze_code_switch(predictions, dialect_label="Gulf")
        prompt_input = ResponsePromptInput(
            normalized_text=normalization.normalized_text,
            dialect_signal=signal,
            code_switch_metrics=metrics,
        )
        llm_response = type(
            "FakeResponse",
            (),
            {
                "response_text": "ياهلا hello",
                "model_name_or_path": "models/Qwen/Qwen2.5-7B-Instruct",
            },
        )()
        stage_result = Stage3ConditionedTextResult(
            original_text=text,
            dialect_signal=signal,
            normalization_result=normalization,
            code_switch_result=CodeSwitchResult(
                text=normalization.normalized_text,
                dialect_label="Gulf",
                predictions=predictions,
            ),
            code_switch_metrics=metrics,
            prompt_input=prompt_input,
            llm_response=llm_response,
        )
        routing = ResponseRoutingResult(
            text="ياهلا hello",
            dialect_label="Gulf",
            matrix_language="AR",
            segments=[
                RoutedTTSSegment(
                    text="ياهلا",
                    language="AR",
                    voice=self.registry.resolve("AR", "Gulf"),
                    start_char=0,
                    end_char=5,
                    token_start_index=0,
                    token_end_index=0,
                ),
                RoutedTTSSegment(
                    text="hello",
                    language="EN",
                    voice=self.registry.resolve("EN"),
                    start_char=6,
                    end_char=11,
                    token_start_index=1,
                    token_end_index=1,
                ),
            ],
            token_predictions=predictions,
        )
        return TextResponseArtifacts(
            input_text=text,
            stage_result=stage_result,
            response_text="ياهلا hello",
            routing_result=routing,
        )

    def synthesize_response(self, response: TextResponseArtifacts) -> BilingualSpeechResult:
        waveform = torch.tensor([0.0, 0.25, -0.25, 0.1], dtype=torch.float32)
        segment = SynthesizedSegment(
            text=response.response_text,
            voice=self.registry.resolve("EN"),
            waveform=waveform,
            sample_rate=24000,
            segment_index=0,
            start_char=0,
            end_char=len(response.response_text),
        )
        return BilingualSpeechResult(
            waveform=waveform,
            sample_rate=24000,
            segments=[segment],
            model_name_or_path="models/Qwen/Qwen2.5-Omni-3B",
        )

    def iter_audio_chunks(self, response: TextResponseArtifacts):
        yield StreamingAudioChunk(
            waveform=torch.tensor([0.1, -0.1], dtype=torch.float32),
            sample_rate=24000,
            chunk_index=0,
            segment_index=0,
            stream_start_sample=0,
            stream_end_sample=2,
            segment_start_sample=0,
            segment_end_sample=2,
            text="ياهلا",
            voice_id="ar_gulf_default",
            is_first_chunk=True,
            is_last_chunk=True,
            is_stream_end=True,
        )


@asynccontextmanager
async def build_test_client():
    app = create_app(
        config=load_app_config(),
        pipeline_provider=FakePipelineProvider(),
        metrics_registry=MetricsRegistry(window_size=20),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client


def build_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((b"\x00\x00" * 160))
    return buffer.getvalue()


@pytest.mark.anyio
async def test_transcribe_endpoint_returns_transcript_and_vad_summary() -> None:
    async with build_test_client() as client:
        response = await client.post(
            "/v1/transcribe",
            files={"audio": ("sample.wav", build_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transcript_text"] == "مرحبا hello"
    assert payload["vad"]["turn_count"] == 1


@pytest.mark.anyio
async def test_respond_endpoint_can_return_audio_payload() -> None:
    async with build_test_client() as client:
        response = await client.post(
            "/v1/respond",
            json={"text": "مرحبا hello", "synthesize_audio": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_text"] == "ياهلا hello"
    assert payload["audio"]["sample_rate"] == 24000
    assert payload["routing"]["segments"][0]["voice_id"] == "ar_gulf_default"


@pytest.mark.anyio
async def test_stream_endpoint_emits_ndjson_events() -> None:
    async with build_test_client() as client:
        response = await client.post("/v1/stream", data={"text": "مرحبا hello"})

    assert response.status_code == 200
    events = [line for line in response.text.strip().splitlines() if line]
    assert any('"event": "analysis"' in line for line in events)
    assert any('"event": "audio_chunk"' in line for line in events)
    assert any('"event": "complete"' in line for line in events)


@pytest.mark.anyio
async def test_health_and_metrics_endpoints_are_available() -> None:
    async with build_test_client() as client:
        ready = await client.get("/health/ready")
        metrics = await client.get("/metrics")

    assert ready.status_code == 200
    assert ready.json()["status"] in {"ready", "degraded"}
    assert metrics.status_code == 200
    assert "uptime_sec" in metrics.json()
