from __future__ import annotations

import asyncio
import base64
import json
from time import perf_counter
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.audio.capture import decode_audio_bytes
from app.audio.playback import waveform_to_pcm16_bytes
from app.pipeline.e2e_pipeline import (
    TextResponseArtifacts,
    TranscriptionArtifacts,
)


router = APIRouter()


class RespondRequest(BaseModel):
    text: str = Field(min_length=1)
    task_instruction: str | None = None
    synthesize_audio: bool = False


@router.get("/health/live")
async def live_health(request: Request) -> dict[str, object]:
    return _services(request).health.live()


@router.get("/health/ready")
async def ready_health(request: Request) -> dict[str, object]:
    return _services(request).health.ready()


@router.get("/health/warmup")
async def warmup_status(request: Request) -> dict[str, object]:
    services = _services(request)
    return {
        "warmed_up": services.cache_manager.is_warmed_up(),
        "cache_stats": services.cache_manager.cache_stats(),
    }


@router.post("/health/warmup")
async def trigger_warmup(request: Request) -> dict[str, object]:
    """Manually trigger model warmup."""
    services = _services(request)
    if services.cache_manager.is_warmed_up():
        return {
            "status": "already_warmed_up",
            "warmed_up": True,
        }

    def _warmup():
        services.pipeline_provider.get_pipeline()

    import threading
    thread = threading.Thread(target=_warmup, daemon=True)
    thread.start()

    return {
        "status": "warmup_started",
        "warmed_up": False,
    }


@router.delete("/cache")
async def invalidate_cache(request: Request) -> dict[str, object]:
    """Invalidate model cache (for hot-reload)."""
    services = _services(request)
    services.cache_manager.invalidate_pipeline()
    return {"status": "cache_invalidated"}


@router.get("/cache/stats")
async def cache_stats(request: Request) -> dict[str, object]:
    """Get cache statistics."""
    services = _services(request)
    return services.cache_manager.cache_stats()


@router.get("/metrics")
async def metrics_snapshot(request: Request) -> dict[str, object]:
    return _services(request).metrics.snapshot()


@router.post("/v1/transcribe")
async def transcribe_audio_endpoint(
    request: Request,
    audio: UploadFile = File(...),
    language_hint: str | None = Form(None),
    apply_vad: bool = Form(True),
) -> dict[str, object]:
    services = _services(request)
    started = perf_counter()
    success = False
    status_code = 200

    try:
        payload = await audio.read()
        samples, sample_rate = decode_audio_bytes(payload)
        transcription = services.pipeline_provider.get_pipeline().transcribe_audio(
            samples,
            sample_rate,
            language_hint=language_hint,
            apply_vad=apply_vad,
        )
        success = True
        return _serialize_transcription(transcription)
    except ValueError as exc:
        status_code = 400
        services.metrics.record_error("/v1/transcribe", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        status_code = 500
        services.metrics.record_error("/v1/transcribe", str(exc))
        raise HTTPException(status_code=500, detail="transcription failed") from exc
    finally:
        services.metrics.record_endpoint(
            "/v1/transcribe",
            duration_ms=(perf_counter() - started) * 1000.0,
            success=success,
            status_code=status_code,
        )


@router.post("/v1/respond")
async def respond_endpoint(request: Request, payload: RespondRequest) -> dict[str, object]:
    services = _services(request)
    started = perf_counter()
    success = False
    status_code = 200

    try:
        pipeline = services.pipeline_provider.get_pipeline()
        response = pipeline.respond_to_text(
            payload.text,
            task_instruction=payload.task_instruction,
        )
        body = _serialize_text_response(response)

        if payload.synthesize_audio:
            speech_result = pipeline.synthesize_response(response)
            audio_bytes = waveform_to_pcm16_bytes(speech_result.waveform)
            body["audio"] = {
                "audio_format": "pcm_s16le",
                "sample_rate": speech_result.sample_rate,
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "duration_sec": round(
                    speech_result.waveform.numel() / float(speech_result.sample_rate),
                    3,
                ),
                "segment_count": len(speech_result.segments),
            }

        success = True
        return body
    except ValueError as exc:
        status_code = 400
        services.metrics.record_error("/v1/respond", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        status_code = 500
        services.metrics.record_error("/v1/respond", str(exc))
        raise HTTPException(status_code=500, detail="response generation failed") from exc
    finally:
        services.metrics.record_endpoint(
            "/v1/respond",
            duration_ms=(perf_counter() - started) * 1000.0,
            success=success,
            status_code=status_code,
        )


@router.post("/v1/stream")
async def stream_endpoint(
    request: Request,
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    task_instruction: str | None = Form(None),
    language_hint: str | None = Form(None),
    apply_vad: bool = Form(True),
) -> StreamingResponse:
    services = _services(request)
    pipeline = services.pipeline_provider.get_pipeline()

    if audio is None and (text is None or not text.strip()):
        raise HTTPException(status_code=400, detail="provide either text or audio")

    audio_samples = None
    audio_sample_rate = None
    if audio is not None:
        payload = await audio.read()
        try:
            audio_samples, audio_sample_rate = decode_audio_bytes(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _event_stream():
        started = perf_counter()
        success = False

        try:
            response: TextResponseArtifacts
            if audio_samples is not None and audio_sample_rate is not None:
                transcription = pipeline.transcribe_audio(
                    audio_samples,
                    audio_sample_rate,
                    language_hint=language_hint,
                    apply_vad=apply_vad,
                )
                yield _json_line(
                    {
                        "event": "transcript",
                        "data": _serialize_transcription(transcription),
                    }
                )
                if not transcription.transcript_text:
                    yield _json_line(
                        {
                            "event": "complete",
                            "data": {"response_text": "", "audio_chunk_count": 0},
                        }
                    )
                    success = True
                    return

                response = pipeline.respond_to_text(
                    transcription.transcript_text,
                    task_instruction=task_instruction,
                )
            else:
                response = pipeline.respond_to_text(
                    text or "",
                    task_instruction=task_instruction,
                )

            yield _json_line(
                {"event": "analysis", "data": _serialize_text_response(response)}
            )
            yield _json_line(
                {
                    "event": "response",
                    "data": {"text": response.response_text},
                }
            )

            audio_chunk_count = 0
            for chunk in pipeline.iter_audio_chunks(response):
                audio_chunk_count += 1
                pcm_bytes = waveform_to_pcm16_bytes(chunk.waveform)
                services.metrics.record_stream_chunk(sample_count=chunk.waveform.numel())
                yield _json_line(
                    {
                        "event": "audio_chunk",
                        "data": {
                            "chunk_index": chunk.chunk_index,
                            "segment_index": chunk.segment_index,
                            "sample_rate": chunk.sample_rate,
                            "voice_id": chunk.voice_id,
                            "text": chunk.text,
                            "audio_format": "pcm_s16le",
                            "audio_base64": base64.b64encode(pcm_bytes).decode("ascii"),
                            "is_first_chunk": chunk.is_first_chunk,
                            "is_last_chunk": chunk.is_last_chunk,
                            "is_stream_end": chunk.is_stream_end,
                        },
                    }
                )

            yield _json_line(
                {
                    "event": "complete",
                    "data": {
                        "response_text": response.response_text,
                        "audio_chunk_count": audio_chunk_count,
                    },
                }
            )
            success = True
        except Exception as exc:
            services.metrics.record_error("/v1/stream", str(exc))
            yield _json_line(
                {"event": "error", "data": {"message": str(exc)}}
            )
        finally:
            services.metrics.record_endpoint(
                "/v1/stream",
                duration_ms=(perf_counter() - started) * 1000.0,
                success=success,
                status_code=200 if success else 500,
            )

    return StreamingResponse(_event_stream(), media_type="application/x-ndjson")


def _services(request: Request) -> Any:
    return request.app.state.services


def _serialize_transcription(transcription: TranscriptionArtifacts) -> dict[str, object]:
    return {
        "transcript_text": transcription.transcript_text,
        "sample_rate": transcription.sample_rate,
        "duration_sec": round(transcription.duration_sec, 3),
        "segments": [
            {
                "start_sec": round(segment.start_sec, 3),
                "end_sec": round(segment.end_sec, 3),
                "text": segment.text,
            }
            for segment in transcription.segments
        ],
        "vad": _serialize_vad_trace(transcription.vad_trace),
    }


def _serialize_text_response(response: TextResponseArtifacts) -> dict[str, object]:
    stage_result = response.stage_result
    code_switch = stage_result.code_switch_metrics
    dialect = stage_result.dialect_signal
    return {
        "input_text": response.input_text,
        "normalized_text": stage_result.normalization_result.normalized_text,
        "dialect": {
            "label": dialect.conditioning_label,
            "confidence": round(dialect.confidence, 3),
            "raw_label": dialect.raw_label,
            "is_fallback": dialect.is_fallback,
            "fallback_reason": dialect.fallback_reason,
        },
        "code_switch": {
            "cs_index": round(code_switch.cs_index, 3),
            "switch_count": code_switch.switch_count,
            "valid_transition_count": code_switch.valid_transition_count,
            "matrix_language": code_switch.matrix_language,
            "secondary_language": code_switch.secondary_language,
            "language_token_count": code_switch.language_token_count,
            "embedded_language_islands": [
                {
                    "language": island.language,
                    "text": island.text,
                    "start_token_index": island.start_token_index,
                    "end_token_index": island.end_token_index,
                }
                for island in code_switch.embedded_language_islands
            ],
        },
        "response_text": response.response_text,
        "routing": {
            "matrix_language": response.routing_result.matrix_language,
            "dialect_label": response.routing_result.dialect_label,
            "segments": [
                {
                    "text": segment.text,
                    "language": segment.language,
                    "voice_id": segment.voice.voice_id,
                }
                for segment in response.routing_result.segments
            ],
        },
    }


def _serialize_vad_trace(vad_trace) -> dict[str, object] | None:
    if vad_trace is None:
        return None

    return {
        "frame_ms": vad_trace.frame_ms,
        "total_frames": vad_trace.total_frames,
        "speech_frames": vad_trace.speech_frames,
        "turn_count": len(vad_trace.turns),
        "turns": [
            {
                "turn_index": turn.turn_index,
                "start_sec": round(turn.start_sec, 3),
                "end_sec": round(turn.end_sec, 3),
                "speech_frames": turn.speech_frames,
                "total_frames": turn.total_frames,
            }
            for turn in vad_trace.turns
        ],
    }


def _json_line(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@router.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """WebSocket endpoint for low-latency real-time conversation."""
    from app.api.websocket import AsyncConversationHandler
    from app.pipeline.e2e_pipeline import LazyPipelineProvider

    services = websocket.app.state.services
    provider = services.pipeline_provider

    if isinstance(provider, LazyPipelineProvider):
        pipeline = provider.get_pipeline()
    else:
        pipeline = provider

    handler = AsyncConversationHandler(pipeline, services.metrics)
    await handler.handle_client(websocket)
