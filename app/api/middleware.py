from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.config import AppConfig


class InMemoryRateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-client sliding-window rate limiter."""

    def __init__(
        self,
        app,
        *,
        requests_per_minute: int,
        exempt_paths: set[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = max(0, int(requests_per_minute))
        self.exempt_paths = exempt_paths or set()
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):
        if (
            self.requests_per_minute == 0
            or request.method == "OPTIONS"
            or request.url.path in self.exempt_paths
        ):
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()

        with self._lock:
            timestamps = self._timestamps[client_host]
            while timestamps and now - timestamps[0] > 60.0:
                timestamps.popleft()

            if len(timestamps) >= self.requests_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                )

            timestamps.append(now)

        return await call_next(request)


def configure_middleware(app: FastAPI, config: AppConfig) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        InMemoryRateLimitMiddleware,
        requests_per_minute=config.server.rate_limit_per_minute,
        exempt_paths={"/health/live", "/health/ready", "/metrics"},
    )
