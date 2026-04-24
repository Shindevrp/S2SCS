from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock


@dataclass
class TimingStats:
    count: int = 0
    error_count: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0

    def record(self, duration_ms: float, *, success: bool) -> None:
        self.count += 1
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        if not success:
            self.error_count += 1

    def to_dict(self) -> dict[str, float | int]:
        average_ms = self.total_ms / self.count if self.count else 0.0
        return {
            "count": self.count,
            "error_count": self.error_count,
            "average_ms": round(average_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }


class MetricsRegistry:
    """Small in-memory metrics store for HTTP and pipeline-stage telemetry."""

    def __init__(self, *, window_size: int = 200) -> None:
        self.started_at = time.time()
        self._window_size = max(1, int(window_size))
        self._lock = Lock()
        self._endpoint_metrics: dict[str, TimingStats] = defaultdict(TimingStats)
        self._stage_metrics: dict[str, TimingStats] = defaultdict(TimingStats)
        self._stream_chunk_count = 0
        self._stream_sample_count = 0
        self._recent_errors: deque[dict[str, str | float]] = deque(
            maxlen=self._window_size
        )

    def record_endpoint(
        self,
        endpoint: str,
        *,
        duration_ms: float,
        success: bool,
        status_code: int,
    ) -> None:
        with self._lock:
            metric = self._endpoint_metrics[f"{endpoint}:{status_code}"]
            metric.record(duration_ms, success=success)

    def record_stage(self, stage: str, *, duration_ms: float, success: bool) -> None:
        with self._lock:
            self._stage_metrics[stage].record(duration_ms, success=success)

    def record_stream_chunk(self, *, sample_count: int) -> None:
        with self._lock:
            self._stream_chunk_count += 1
            self._stream_sample_count += int(sample_count)

    def record_error(self, source: str, message: str) -> None:
        with self._lock:
            self._recent_errors.append(
                {
                    "source": source,
                    "message": message,
                    "timestamp": round(time.time(), 3),
                }
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "uptime_sec": round(time.time() - self.started_at, 3),
                "endpoints": {
                    name: metric.to_dict()
                    for name, metric in sorted(self._endpoint_metrics.items())
                },
                "pipeline_stages": {
                    name: metric.to_dict()
                    for name, metric in sorted(self._stage_metrics.items())
                },
                "streaming": {
                    "audio_chunk_count": self._stream_chunk_count,
                    "audio_sample_count": self._stream_sample_count,
                },
                "recent_errors": list(self._recent_errors),
            }
