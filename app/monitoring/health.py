from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import AppConfig


class HealthChecker:
    """Readiness probes focused on config validity and local asset availability."""

    def __init__(self, config: AppConfig, pipeline_provider: Any) -> None:
        self.config = config
        self.pipeline_provider = pipeline_provider

    def live(self) -> dict[str, object]:
        return {
            "status": "live",
            "config_path": str(self.config.config_path),
        }

    def ready(self) -> dict[str, object]:
        checks = [
            self._model_reference_status(
                "asr",
                self.config.models.asr.model_name_or_path,
                local_only=self.config.models.asr.local_files_only,
            ),
            self._model_reference_status(
                "llm",
                self.config.models.llm.model_name_or_path,
                local_only=self.config.models.llm.local_files_only,
            ),
            self._model_reference_status(
                "tts",
                self.config.models.tts.model_name_or_path,
                local_only=self.config.models.tts.local_files_only,
            ),
            self._model_reference_status(
                "code_switch",
                self.config.models.code_switch.model_name_or_path,
                local_only=False,
            ),
        ]

        if self.config.monitoring.health_model_path_checks:
            is_ready = all(check["ready"] for check in checks)
        else:
            is_ready = True

        return {
            "status": "ready" if is_ready else "degraded",
            "pipeline_initialized": bool(
                getattr(self.pipeline_provider, "is_initialized", False)
            ),
            "checks": checks,
        }

    def _model_reference_status(
        self,
        name: str,
        reference: str,
        *,
        local_only: bool,
    ) -> dict[str, object]:
        resolved_reference = self.config.resolve_reference(reference, local_only=local_only)
        resolved_path = Path(resolved_reference)

        if resolved_path.exists():
            return {
                "name": name,
                "ready": True,
                "reference": str(reference),
                "detail": "local_path_present",
                "resolved_path": str(resolved_path),
            }

        if local_only:
            return {
                "name": name,
                "ready": False,
                "reference": str(reference),
                "detail": "required_local_path_missing",
                "resolved_path": str(resolved_path),
            }

        return {
            "name": name,
            "ready": True,
            "reference": str(reference),
            "detail": "external_reference_allowed",
            "resolved_path": str(resolved_path),
        }
