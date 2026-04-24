from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CachedModel(Generic[T]):
    value: T
    loaded_at: float = field(default_factory=time.time)


class ModelCache:
    """Thread-safe model cache with TTL support."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, CachedModel[Any]] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = ttl_seconds

    def get_or_load(
        self,
        key: str,
        factory: Callable[[], T],
    ) -> T:
        """Get from cache or load via factory."""
        with self._lock:
            cached = self._cache.get(key)
            now = time.time()

            if cached is not None:
                age = now - cached.loaded_at
                if age < self._ttl_seconds:
                    return cached.value

                del self._cache[key]

            model = factory()
            self._cache[key] = CachedModel(value=model)
            return model

    def invalidate(self, key: str) -> None:
        """Remove a specific model from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired = [
            key
            for key, cached in self._cache.items()
            if (now - cached.loaded_at) >= self._ttl_seconds
        ]
        for key in expired:
            del self._cache[key]

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            self._cleanup_expired()
            return {
                "cached_models": len(self._cache),
                "ttl_seconds": self._ttl_seconds,
            }


_global_model_cache: ModelCache[Any] | None = None
_cache_lock = threading.Lock()


def get_model_cache(ttl_seconds: int = 3600) -> ModelCache[Any]:
    """Get or create the global model cache singleton."""
    global _global_model_cache
    if _global_model_cache is None:
        with _cache_lock:
            if _global_model_cache is None:
                _global_model_cache = ModelCache(ttl_seconds=ttl_seconds)
    return _global_model_cache


def cached_model(key: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Decorator to cache model loaders."""

    def decorator(factory: Callable[[], T]) -> Callable[[], T]:
        def wrapper() -> T:
            cache = get_model_cache()
            return cache.get_or_load(key, factory)

        return wrapper

    return decorator