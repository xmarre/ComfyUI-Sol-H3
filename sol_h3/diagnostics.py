"""Opt-in bounded CUDA-event diagnostics for Sol-H3 validation/performance attribution.

Production execution does not synchronize for these measurements unless
SOL_H3_CUDA_DIAGNOSTICS is explicitly enabled. The state owns CUDA Event objects
only; it never retains Q/K/V, dense references, model weights, or outputs.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import os
import threading
import time


DEFAULT_MAX_SAMPLES = 128
_ENV = "SOL_H3_CUDA_DIAGNOSTICS"


def _enabled_from_env() -> bool:
    value = os.environ.get(_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class CudaDiagnosticState:
    """Request-owned bounded CUDA-event recorder.

    Gate misses are always preferred. Production calls are sampled densely at
    first and then at powers of two so later evaluations remain represented.
    Event resolution occurs once at request teardown and is reported separately.
    """

    def __init__(
        self,
        *,
        enabled=False,
        max_samples=DEFAULT_MAX_SAMPLES,
        event_factory=None,
        synchronize=None,
    ):
        self.enabled = bool(enabled)
        self.max_samples = int(max_samples)
        if self.max_samples <= 0:
            raise ValueError("CUDA diagnostic max_samples must be positive")
        self._event_factory = event_factory
        self._synchronize = synchronize
        self._lock = threading.Lock()
        self._samples = []
        self._kind_counts = Counter()
        self._overflow = Counter()
        self._resolved = []
        self._resolved_count = 0
        self._drain_count = 0
        self._resolve_sync_wall_s = 0.0
        self._resolution_error = None

    @classmethod
    def from_env(cls):
        return cls(enabled=_enabled_from_env())

    def _factory(self):
        if self._event_factory is not None:
            return self._event_factory
        import torch

        return lambda: torch.cuda.Event(enable_timing=True)

    def _sync(self, device):
        if self._synchronize is not None:
            return self._synchronize(device)
        import torch

        return torch.cuda.synchronize(device)

    @staticmethod
    def _representative_ordinal(ordinal):
        return ordinal <= 8 or (ordinal & (ordinal - 1)) == 0

    def begin_sample(self, kind, device, *, context=None, important=False):
        if not self.enabled:
            return None
        kind = str(kind)
        with self._lock:
            self._kind_counts[kind] += 1
            ordinal = int(self._kind_counts[kind])
            selected = bool(important or self._representative_ordinal(ordinal))
            if not selected:
                return None
            if len(self._samples) >= self.max_samples:
                self._overflow[kind] += 1
                return None
            sample = {
                "kind": kind,
                "ordinal": ordinal,
                "device": device,
                "device_name": str(device),
                "context": dict(context or {}),
                "initial_stream_drain_host_wall_s": None,
                "spans": [],
            }
            self._samples.append(sample)
            return sample

    def initial_stream_drain(self, sample):
        """Diagnostic-only clean-start synchronization, excluded from gate wall."""
        if sample is None:
            return
        started = time.perf_counter()
        self._sync(sample["device"])
        sample["initial_stream_drain_host_wall_s"] = time.perf_counter() - started

    @contextmanager
    def span(self, sample, name):
        if sample is None:
            yield
            return
        factory = self._factory()
        start = factory()
        end = factory()
        start.record()
        try:
            yield
        finally:
            end.record()
            sample["spans"].append((str(name), start, end))

    def drain_pending(self):
        """Resolve pending events at an existing model-evaluation/stage boundary.

        A diagnostic failure is retained in telemetry instead of changing model
        output or masking the sampling exception.
        """
        if not self.enabled:
            return
        with self._lock:
            start_index = self._resolved_count
            pending = list(self._samples[start_index:])
            self._resolved_count = len(self._samples)
        if not pending:
            return
        self._drain_count += 1
        try:
            devices = []
            seen = set()
            for sample in pending:
                key = str(sample["device"])
                if key not in seen:
                    seen.add(key)
                    devices.append(sample["device"])
            started = time.perf_counter()
            for device in devices:
                self._sync(device)
            self._resolve_sync_wall_s += time.perf_counter() - started

            for sample in pending:
                spans = {}
                for name, start, end in sample["spans"]:
                    elapsed = float(start.elapsed_time(end))
                    spans[name] = float(spans.get(name, 0.0)) + elapsed
                self._resolved.append({
                    "kind": sample["kind"],
                    "ordinal": sample["ordinal"],
                    "device": sample["device_name"],
                    "context": dict(sample["context"]),
                    "initial_stream_drain_host_wall_s": sample[
                        "initial_stream_drain_host_wall_s"
                    ],
                    "cuda_event_ms": spans,
                })
        except BaseException as exc:
            self._resolution_error = f"{type(exc).__name__}: {exc}"

    def resolve(self):
        """Compatibility alias used by the outer request teardown."""
        self.drain_pending()

    def summary(self):
        details = self._resolved if self._resolved else []
        return {
            "enabled": self.enabled,
            "max_samples": self.max_samples,
            "selected_samples": len(self._samples),
            "resolved_samples": len(details),
            "observed_calls": dict(self._kind_counts),
            "overflow": dict(self._overflow),
            "drain_count": self._drain_count,
            "resolve_sync_wall_s": self._resolve_sync_wall_s,
            "resolution_error": self._resolution_error,
            "details": list(details),
        }


__all__ = ["CudaDiagnosticState", "DEFAULT_MAX_SAMPLES"]
