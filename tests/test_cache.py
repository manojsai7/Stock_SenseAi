import time
from backend.core.cache import SimpleMemoryCache

def test_cache_set_and_get():
    c = SimpleMemoryCache(default_ttl=10)
    c.set("stock:TCS.NS", {"price": 3500.0})
    val = c.get("stock:TCS.NS")
    assert val is not None
    assert val["price"] == 3500.0

def test_cache_expiration():
    c = SimpleMemoryCache(default_ttl=1)  # 1 second TTL
    c.set("temp_key", "hello", ttl=1)
    assert c.get("temp_key") == "hello"
    time.sleep(1.1)
    assert c.get("temp_key") is None

def test_cache_stats_and_hit_ratio():
    c = SimpleMemoryCache()
    c.set("key1", "val1")
    # 1 hit
    _ = c.get("key1")
    # 1 miss
    _ = c.get("key2")

    stats = c.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_ratio"] == 0.5
    assert stats["entries"] == 1

def test_cache_delete():
    c = SimpleMemoryCache()
    c.set("k", "v")
    assert c.delete("k") is True
    assert c.get("k") is None
    assert c.delete("k") is False
