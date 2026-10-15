"""Opt-in process-wide CUDA allocation attribution for real sampler requests."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import os
import time

_PROFILE = ContextVar("sol_h3_memory_profile", default=None)
log = logging.getLogger("comfy.sol_h3")


def enabled():
    return _PROFILE.get() is not None


def storage_bytes(*tensors):
    """Count backing allocations once, including padding and views' dead slabs."""
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes() for t in tensors}
    return sum(storages.values())


def mark(stage, **fields):
    profile = _PROFILE.get()
    if profile is None:
        return
    import torch
    device = profile["device_index"]
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)
    entry = profile["stages"].setdefault(stage, {"count": 0, "max_live_allocated": 0, "max_reserved": 0})
    entry["count"] += 1
    entry["max_live_allocated"] = max(entry["max_live_allocated"], allocated)
    entry["max_reserved"] = max(entry["max_reserved"], reserved)
    if peak > profile["observed_peak_allocated"]:
        profile["observed_peak_allocated"] = peak
        profile["peak_first_observed_at"] = {"stage": stage, **fields}
    # Bounded per-evaluation records distinguish first use from warm calls.
    if stage in {"evaluation_enter", "evaluation_exit"} and len(profile["evaluations"]) < 256:
        profile["evaluations"].append({"stage": stage, "allocated": allocated, "reserved": reserved,
                                       "peak_allocated_since_request_start": peak, **fields})


@contextmanager
def sampling(config):
    if os.environ.get("SOL_H3_MEMORY_PROFILE") != "1" or _PROFILE.get() is not None:
        yield
        return
    import torch
    if not torch.cuda.is_available():
        yield
        return
    device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    profile = {
        "contract": "sol-h3-sampler-memory-v1", "device_index": device,
        "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "config": config.metadata(), "process_wide_allocator": True,
        "baseline_allocated": baseline, "baseline_reserved": torch.cuda.memory_reserved(device),
        "observed_peak_allocated": baseline, "peak_first_observed_at": None,
        "stages": {}, "evaluations": [],
    }
    token = _PROFILE.set(profile)
    started = time.perf_counter()
    success = False
    try:
        yield
        success = True
    finally:
        try:
            torch.cuda.synchronize(device)
            mark("sampler_exit")
            profile.update(
                success=success, sampler_wall_s=time.perf_counter() - started,
                peak_allocated=torch.cuda.max_memory_allocated(device),
                peak_reserved=torch.cuda.max_memory_reserved(device),
                sampler_exit_allocated=torch.cuda.memory_allocated(device),
                sampler_exit_reserved=torch.cuda.memory_reserved(device),
            )
            profile["incremental_peak_allocated"] = profile["peak_allocated"] - baseline
            log.info("Sol-H3 memory %s", json.dumps(profile, sort_keys=True))
        finally:
            _PROFILE.reset(token)
