import time
from typing import Any, Optional, Dict

class SimpleMemoryCache:
    """
    In-memory caching layer implementing Cache-Aside pattern.
    Provides fast, non-blocking cache lookups for price history, quotes, and predictions.
    Tracks hits, misses, and active entries for observability.
    """
    def __init__(self, default_ttl: int = 300):
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Optional[Any]:
        item = self._cache.get(key)
        if not item:
            self.misses += 1
            return None
        if time.time() > item["expires_at"]:
            del self._cache[key]
            self.misses += 1
            return None
        self.hits += 1
        return item["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        duration = ttl if ttl is not None else self.default_ttl
        self._cache[key] = {
            "value": value,
            "expires_at": time.time() + duration,
            "created_at": time.time()
        }

    def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v["expires_at"]]
        for k in expired_keys:
            del self._cache[k]
        total_requests = self.hits + self.misses
        return {
            "entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hits / total_requests, 3) if total_requests > 0 else 0.0
        }

cache = SimpleMemoryCache(default_ttl=300)
