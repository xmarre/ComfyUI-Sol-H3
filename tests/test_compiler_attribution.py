from types import SimpleNamespace
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



def test_cache_generation_changes_only_for_destructive_mutation():
    cache = compiler_attribution._AttributedCache()
    assert cache.destructive_generation == 0

    def first():
        return None

    cache[("a",)] = first
    cache[("b",)] = lambda: None
    assert cache.destructive_generation == 0

    cache[("a",)] = lambda: None
    assert cache.destructive_generation == 1

    cache.pop(("b",))
    assert cache.destructive_generation == 2

    cache[("c",)] = lambda: None
    assert cache.destructive_generation == 2
    cache.clear()
    assert cache.destructive_generation == 3



def test_refresh_runtime_objects_rewraps_replaced_cache_and_lock():
    interface = SimpleNamespace(
        _compiled={("a",): lambda: None},
        _compile_lock=threading.Lock(),
    )
    compiler_attribution._refresh_runtime_objects(interface)
    assert isinstance(interface._compiled, compiler_attribution._AttributedCache)
    assert isinstance(interface._compile_lock, compiler_attribution._TimedLock)
    first_cache = interface._compiled

    interface._compiled = {("b",): lambda: None}
    interface._compile_lock = threading.Lock()
    compiler_attribution._refresh_runtime_objects(interface)
    assert isinstance(interface._compiled, compiler_attribution._AttributedCache)
    assert isinstance(interface._compile_lock, compiler_attribution._TimedLock)
    assert interface._compiled is not first_cache
    assert ("b",) in interface._compiled
