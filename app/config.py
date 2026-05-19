from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _infer_device() -> str:
    """Infer the best available device: cuda, mps, or cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            sm = torch.cuda.get_device_capability(0)
            if sm[0] >= 12:
                return "cuda"
            if sm[0] >= 8:
                return "cuda"
    except Exception:
        pass
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEFAULT_DEVICE = _infer_device()


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_allow_origins: list[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    warmup_models_on_startup: bool = False


@dataclass(frozen=True)
class VADConfig:
    threshold: float = 0.5
    device: str = DEFAULT_DEVICE


@dataclass(frozen=True)
class ASRConfig:
    model_name_or_path: str = "models/Nabeh_QwBaseerSTT"
    device: str = DEFAULT_DEVICE
    max_new_tokens: int = 256
    local_files_only: bool = True


@dataclass(frozen=True)
class DialectConfig:
    confidence_threshold: float = 0.40
    minimum_arabic_chars: int = 6


@dataclass(frozen=True)
class CodeSwitchConfig:
    model_name_or_path: str = "models/1716Shinde/xlmr-cs-finetuned"
    device: str = DEFAULT_DEVICE
    max_length: int = 256


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "jais"
    model_name_or_path: str = "models/mlconvexai/jais-13b-chat_bitsandbytes_4bit"
    device: str = DEFAULT_DEVICE
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    local_files_only: bool = True


@dataclass(frozen=True)
class TTSConfig:
    model_name_or_path: str = "models/Qwen/Qwen2.5-Omni-3B"
    device: str = DEFAULT_DEVICE
    local_files_only: bool = True
    pause_duration_ms: int = 120


@dataclass(frozen=True)
class ModelConfig:
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    dialect: DialectConfig = field(default_factory=DialectConfig)
    code_switch: CodeSwitchConfig = field(default_factory=CodeSwitchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)


@dataclass(frozen=True)
class AudioConfig:
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    frame_ms: int = 32
    end_silence_ms: int = 700
    min_speech_ms: int = 350
    max_utterance_s: float = 12.0
    asr_chunk_ms: int = 1200


@dataclass(frozen=True)
class PipelineConfig:
    task_instruction: str = "Generate a natural conversational response."
    language_hint: str | None = None
    sliding_window_tokens: int = 48
    max_context_updates: int = 8
    stream_chunk_duration_ms: int = 120
    first_chunk_duration_ms: int = 40
    inter_segment_pause_ms: int = 0


@dataclass(frozen=True)
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_window_size: int = 200
    health_model_path_checks: bool = True
    model_cache_ttl_seconds: int = 3600
    model_warmup_on_startup: bool = False
    lazy_load_models: bool = True


@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    config_path: Path = DEFAULT_CONFIG_PATH
    project_root: Path = PROJECT_ROOT

    def resolve_reference(self, reference: str | Path, *, local_only: bool = False) -> str:
        reference_str = str(reference)
        reference_path = Path(reference_str)
        if reference_path.is_absolute():
            return str(reference_path)

        candidate = (self.project_root / reference_path).resolve()
        if local_only or candidate.exists():
            return str(candidate)

        return reference_str


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    resolved_path = Path(
        config_path or os.getenv("S2SCS_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    ).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / resolved_path).resolve()

    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    config = AppConfig(
        server=_load_server_config(payload.get("server", {})),
        models=_load_model_config(payload.get("models", {})),
        audio=_load_audio_config(payload.get("audio", {})),
        pipeline=_load_pipeline_config(payload.get("pipeline", {})),
        monitoring=_load_monitoring_config(payload.get("monitoring", {})),
        config_path=resolved_path,
        project_root=PROJECT_ROOT,
    )
    _validate_config(config)
    return config


def _load_server_config(payload: dict[str, Any]) -> ServerConfig:
    return ServerConfig(
        host=str(payload.get("host", "0.0.0.0")),
        port=int(payload.get("port", 8000)),
        reload=bool(payload.get("reload", False)),
        cors_allow_origins=_string_list(payload.get("cors_allow_origins"), default=["*"]),
        rate_limit_per_minute=int(payload.get("rate_limit_per_minute", 60)),
        warmup_models_on_startup=bool(payload.get("warmup_models_on_startup", False)),
    )


def _load_model_config(payload: dict[str, Any]) -> ModelConfig:
    vad_payload = payload.get("vad", {})
    asr_payload = payload.get("asr", {})
    dialect_payload = payload.get("dialect", {})
    code_switch_payload = payload.get("code_switch", {})
    llm_payload = payload.get("llm", {})
    tts_payload = payload.get("tts", {})

    return ModelConfig(
        vad=VADConfig(
            threshold=float(vad_payload.get("threshold", 0.5)),
            device=str(vad_payload.get("device", "cpu")),
        ),
        asr=ASRConfig(
            model_name_or_path=str(
                asr_payload.get("model_name_or_path", "models/Nabeh_QwBaseerSTT")
            ),
            device=str(asr_payload.get("device", "cpu")),
            max_new_tokens=int(asr_payload.get("max_new_tokens", 256)),
            local_files_only=bool(asr_payload.get("local_files_only", True)),
        ),
        dialect=DialectConfig(
            confidence_threshold=float(
                dialect_payload.get("confidence_threshold", 0.40)
            ),
            minimum_arabic_chars=int(
                dialect_payload.get("minimum_arabic_chars", 6)
            ),
        ),
        code_switch=CodeSwitchConfig(
            model_name_or_path=str(
                code_switch_payload.get("model_name_or_path", "models/1716Shinde/xlmr-cs-finetuned")
            ),
            device=str(code_switch_payload.get("device", "cpu")),
            max_length=int(code_switch_payload.get("max_length", 256)),
        ),
        llm=LLMConfig(
            provider=str(llm_payload.get("provider", "jais")),
            model_name_or_path=str(
                llm_payload.get("model_name_or_path", "models/mlconvexai/jais-13b-chat_bitsandbytes_4bit")
            ),
            device=str(llm_payload.get("device", "cpu")),
            max_new_tokens=int(llm_payload.get("max_new_tokens", 128)),
            temperature=float(llm_payload.get("temperature", 0.7)),
            top_p=float(llm_payload.get("top_p", 0.9)),
            local_files_only=bool(llm_payload.get("local_files_only", True)),
        ),
        tts=TTSConfig(
            model_name_or_path=str(
                tts_payload.get("model_name_or_path", "models/Qwen/Qwen2.5-Omni-3B")
            ),
            device=str(tts_payload.get("device", "cpu")),
            local_files_only=bool(tts_payload.get("local_files_only", True)),
            pause_duration_ms=int(tts_payload.get("pause_duration_ms", 120)),
        ),
    )


def _load_audio_config(payload: dict[str, Any]) -> AudioConfig:
    return AudioConfig(
        input_sample_rate=int(payload.get("input_sample_rate", 16000)),
        output_sample_rate=int(payload.get("output_sample_rate", 24000)),
        frame_ms=int(payload.get("frame_ms", 32)),
        end_silence_ms=int(payload.get("end_silence_ms", 700)),
        min_speech_ms=int(payload.get("min_speech_ms", 350)),
        max_utterance_s=float(payload.get("max_utterance_s", 12.0)),
        asr_chunk_ms=int(payload.get("asr_chunk_ms", 1200)),
    )


def _load_pipeline_config(payload: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        task_instruction=str(
            payload.get(
                "task_instruction",
                "Generate a natural conversational response.",
            )
        ),
        language_hint=payload.get("language_hint"),
        sliding_window_tokens=int(payload.get("sliding_window_tokens", 48)),
        max_context_updates=int(payload.get("max_context_updates", 8)),
        stream_chunk_duration_ms=int(payload.get("stream_chunk_duration_ms", 120)),
        first_chunk_duration_ms=int(payload.get("first_chunk_duration_ms", 40)),
        inter_segment_pause_ms=int(payload.get("inter_segment_pause_ms", 0)),
    )


def _load_monitoring_config(payload: dict[str, Any]) -> MonitoringConfig:
    return MonitoringConfig(
        enable_metrics=bool(payload.get("enable_metrics", True)),
        metrics_window_size=int(payload.get("metrics_window_size", 200)),
        health_model_path_checks=bool(payload.get("health_model_path_checks", True)),
        model_cache_ttl_seconds=int(payload.get("model_cache_ttl_seconds", 3600)),
        model_warmup_on_startup=bool(payload.get("model_warmup_on_startup", False)),
        lazy_load_models=bool(payload.get("lazy_load_models", True)),
    )


def _string_list(value: Any, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _validate_config(config: AppConfig) -> None:
    if config.server.port < 1:
        raise ValueError("server.port must be positive")
    if config.server.rate_limit_per_minute < 0:
        raise ValueError("server.rate_limit_per_minute must be non-negative")
    if config.audio.input_sample_rate not in {8000, 16000}:
        raise ValueError("audio.input_sample_rate must be 8000 or 16000 for Silero VAD")
    if config.audio.frame_ms <= 0:
        raise ValueError("audio.frame_ms must be positive")
    if config.audio.end_silence_ms <= 0:
        raise ValueError("audio.end_silence_ms must be positive")
    if config.audio.min_speech_ms <= 0:
        raise ValueError("audio.min_speech_ms must be positive")
    if config.pipeline.stream_chunk_duration_ms <= 0:
        raise ValueError("pipeline.stream_chunk_duration_ms must be positive")
    if config.pipeline.first_chunk_duration_ms <= 0:
        raise ValueError("pipeline.first_chunk_duration_ms must be positive")
