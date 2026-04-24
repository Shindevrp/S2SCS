from __future__ import annotations

import threading
from typing import Any, Callable

from app.cache.model_cache import ModelCache, get_model_cache
from app.config import AppConfig
from app.utils.logger import get_logger


class ModelCacheManager:
    """Manages model caching and warmup lifecycle."""

    def __init__(self, config: AppConfig, cache: ModelCache[Any] | None = None) -> None:
        self.config = config
        self.cache = cache or get_model_cache(config.monitoring.model_cache_ttl_seconds)
        self.logger = get_logger(self.__class__.__name__)
        self._warmup_done = False
        self._warmup_lock = threading.Lock()

    def warmup_models(
        self,
        pipeline_factory: Callable[[], Any],
    ) -> None:
        """Preload all models to avoid cold start delays."""
        if self._warmup_done:
            self.logger.debug("Model warmup already completed")
            return

        with self._warmup_lock:
            if self._warmup_done:
                return

            self.logger.info("Starting model warmup...")

            try:
                pipeline = pipeline_factory()
                self._warmup_done = True
                self.logger.info("Model warmup completed successfully")
            except Exception as exc:
                self.logger.error("Model warmup failed: %s", exc)
                raise

    def get_cached_pipeline(
        self,
        pipeline_factory: Callable[[], Any],
    ) -> Any:
        """Get pipeline with caching."""
        return self.cache.get_or_load(
            "e2e_pipeline",
            pipeline_factory,
        )

    def invalidate_pipeline(self) -> None:
        """Clear pipeline cache (for hot-reload scenarios)."""
        self.cache.invalidate("e2e_pipeline")
        self._warmup_done = False

    def is_warmed_up(self) -> bool:
        """Check if warmup has completed."""
        return self._warmup_done

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


_default_manager: ModelCacheManager | None = None


def get_cache_manager(config: AppConfig) -> ModelCacheManager:
    """Get the global cache manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelCacheManager(config)
    return _default_manager