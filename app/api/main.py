from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI

from app.api.middleware import configure_middleware
from app.api.routes import router
from app.cache.cache_manager import ModelCacheManager, get_cache_manager
from app.config import AppConfig, load_app_config
from app.monitoring.health import HealthChecker
from app.monitoring.metrics import MetricsRegistry
from app.pipeline.e2e_pipeline import EndToEndSpeechPipeline, LazyPipelineProvider


@dataclass
class AppServices:
    config: AppConfig
    pipeline_provider: Any
    metrics: MetricsRegistry
    health: HealthChecker
    cache_manager: ModelCacheManager


def create_app(
    *,
    config: AppConfig | None = None,
    pipeline_provider: Any | None = None,
    metrics_registry: MetricsRegistry | None = None,
    cache_manager: ModelCacheManager | None = None,
) -> FastAPI:
    resolved_config = config or load_app_config()
    metrics = metrics_registry or MetricsRegistry(
        window_size=resolved_config.monitoring.metrics_window_size
    )
    provider = pipeline_provider or LazyPipelineProvider(
        lambda: EndToEndSpeechPipeline.from_config(resolved_config, metrics=metrics)
    )
    resolved_cache_manager = cache_manager or get_cache_manager(resolved_config)
    health = HealthChecker(resolved_config, provider)
    services = AppServices(
        config=resolved_config,
        pipeline_provider=provider,
        metrics=metrics,
        health=health,
        cache_manager=resolved_cache_manager,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services
        if resolved_config.monitoring.model_warmup_on_startup:
            def _warmup():
                return provider.get_pipeline()

            thread = threading.Thread(target=_warmup, daemon=True)
            thread.start()
        yield

    app = FastAPI(
        title="S2SCS Speech API",
        version="0.1.0",
        lifespan=lifespan,
    )
    configure_middleware(app, resolved_config)
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required to run the API server. Install it with `uv pip install uvicorn`."
        ) from exc

    config = load_app_config()
    uvicorn.run(
        "app.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )


if __name__ == "__main__":
    main()
