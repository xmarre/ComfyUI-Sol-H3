import threading
import time

from sol_h3 import compiler_attribution


def test_cache_distinguishes_cold_and_hit_without_cuda_sync():
    cache = compiler_attribution._AttributedCache()
    cold = {}
    with compiler_attribution.attribution_scope(cold):
        assert cache.get(("a",)) is None
    assert cold["compile_cache_initial_hit"] is False

    cache[("a",)] = lambda: None
    hit = {}
    with compiler_attribution.attribution_scope(hit):
        value = cache.get(("a",))
        value()
    assert hit["compile_hit"] is True
    assert hit["dispatch_host_enqueue_s"] >= 0.0


def test_timed_lock_records_host_wait_only():
    lock = compiler_attribution._TimedLock(threading.Lock())
    telemetry = {}
    with compiler_attribution.attribution_scope(telemetry):
        with lock:
            time.sleep(0)
    assert telemetry["compile_lock_wait_s"] >= 0.0
