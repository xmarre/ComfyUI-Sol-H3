"""Request-owned arithmetic validation and runtime-lifetime contracts.

This module deliberately keeps successful numerical proof request-local.  The
process-global CuTe executable cache remains a separate compiler concern.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import secrets
import struct
import sys
import threading
import time


MAX_SUCCESS_ENTRIES = 256
MAX_SUCCESS_BYTES = 256 * 1024
MAX_EXAMPLES = 32
EVICTED_DIGEST_HISTORY = 512
KEY_ABI = "sol_h3_arithmetic_key_v1"

_REQUEST_SERIAL = itertools.count(1)
_PROCESS_GENERATION = secrets.token_hex(16)


def _request_id():
    return f"sol-h3-{os.getpid()}-{next(_REQUEST_SERIAL)}"


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_driver_version():
    try:
        from cuda.bindings import driver

        result = driver.cuDriverGetVersion()
        if isinstance(result, tuple) and len(result) >= 2:
            return int(result[1])
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return None


def _device_identity(device):
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return {
            "type": device.type,
            "index": device.index,
            "process": os.getpid(),
            "process_generation": _PROCESS_GENERATION,
        }
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    properties = torch.cuda.get_device_properties(index)
    return {
        "type": "cuda",
        "index": index,
        "current_device": int(torch.cuda.current_device()),
        "name": str(properties.name),
        "total_memory": int(properties.total_memory),
        "multi_processor_count": int(properties.multi_processor_count),
        "sm": list(torch.cuda.get_device_capability(index)),
        "cuda_driver_version": _cuda_driver_version(),
        "process": os.getpid(),
        "process_generation": _PROCESS_GENERATION,
        # PyTorch owns the primary CUDA context in the supported runtime.  The
        # process/device pair plus the per-process nonce is therefore the stable
        # context generation until process restart; no private driver handle is retained.
        "context_scope": "pytorch-primary-process-device",
    }


@dataclass
class RuntimeLease:
    """One OUTER_SAMPLE lifetime for verified source and loaded runtime identity."""

    request_id: str = field(default_factory=_request_id)
    source_verified: bool = False
    source_verify_count: int = 0
    source_verify_s: float = 0.0
    manifest_digest: str | None = None
    source_generation: str | None = None
    implementation_generation: str | None = None
    device_identity: dict | None = None
    ordinary_runtime_identity: tuple | None = None
    partitioned_runtime_identity: tuple | None = None
    ordinary_compiler_namespace: tuple | None = None
    partitioned_compiler_namespace: tuple | None = None
    compiler_environment: dict = field(default_factory=dict)
    _lock: object = field(default_factory=threading.Lock, repr=False)

    def ensure_source_verified(self, device):
        """Verify packaged bytes once, then bind the request to one device context."""
        import torch

        identity = _device_identity(device)
        with self._lock:
            if self.device_identity is None:
                self.device_identity = identity
            elif self.device_identity != identity:
                raise RuntimeError("Sol-H3 runtime lease changed CUDA/device context within one request")
            if self.source_verified:
                return
            from .provenance import CONTRACT, REVISION, SOURCE, verify_source

            started = time.perf_counter()
            manifest = verify_source()
            elapsed = time.perf_counter() - started
            manifest_digest = _digest(manifest)
            source_generation = _digest({
                "source": SOURCE,
                "revision": REVISION,
                "contract": CONTRACT,
                "manifest": manifest_digest,
            })
            self.manifest_digest = manifest_digest
            self.source_generation = source_generation
            self.implementation_generation = _digest({
                "source_generation": source_generation,
                "validation_abi": KEY_ABI,
                "lifetime_abi": "request_runtime_lease_v1",
            })
            self.compiler_environment = {
                "python": platform.python_version(),
                "platform": platform.system(),
                "torch": str(torch.__version__),
                "torch_cuda": str(torch.version.cuda),
                "triton": _distribution_version("triton"),
                "nvidia_cutlass_dsl": _distribution_version("nvidia-cutlass-dsl"),
                "cuda_python": _distribution_version("cuda-python"),
                "apache_tvm_ffi": _distribution_version("apache-tvm-ffi"),
            }
            self.source_verify_s += elapsed
            self.source_verify_count += 1
            self.source_verified = True

    def bind_kernel(self, kernel, device):
        self.ensure_source_verified(device)
        module_name = getattr(kernel, "__module__", type(kernel).__module__)
        module = sys.modules.get(module_name)
        identity = (
            module_name,
            getattr(module, "__file__", None),
            getattr(kernel, "__qualname__", type(kernel).__qualname__),
            id(kernel),
            getattr(kernel, "backend_name", None),
            getattr(kernel, "block_size", None),
        )
        namespace_provider = getattr(kernel, "compiler_namespace_provider", None)
        namespace = tuple(
            namespace_provider()
            if callable(namespace_provider)
            else getattr(kernel, "compiler_namespace", ("test_substitute", id(kernel)))
        )
        with self._lock:
            changed = self.ordinary_runtime_identity is not None and (
                self.ordinary_runtime_identity != identity
                or self.ordinary_compiler_namespace != namespace
            )
            self.ordinary_runtime_identity = identity
            self.ordinary_compiler_namespace = namespace
            return changed

    def bind_partitioned(self, interface, device):
        self.ensure_source_verified(device)
        from .compiler_attribution import install_hooks

        namespace = tuple(install_hooks(interface))
        original_compile = getattr(
            interface, "_sol_h3_original_compile_sm120", interface._compile_sm120
        )
        identity = (
            "partitioned",
            getattr(interface, "__file__", None),
            id(interface),
            id(original_compile),
            getattr(interface, "MAPPED_NEIGHBOR_CONTRACT", None),
        )
        with self._lock:
            changed = self.partitioned_runtime_identity is not None and (
                self.partitioned_runtime_identity != identity
                or self.partitioned_compiler_namespace != namespace
            )
            self.partitioned_runtime_identity = identity
            self.partitioned_compiler_namespace = namespace
            return changed

    def runtime_identity_for(self, mode):
        value = (
            self.partitioned_runtime_identity
            if mode.startswith("partitioned_")
            else self.ordinary_runtime_identity
        )
        if value is None:
            raise RuntimeError("Sol-H3 arithmetic key requested before runtime identity binding")
        return value

    def compiler_namespace_for(self, mode):
        value = (
            self.partitioned_compiler_namespace
            if mode.startswith("partitioned_")
            else self.ordinary_compiler_namespace
        )
        if value is None:
            raise RuntimeError("Sol-H3 arithmetic key requested before compiler namespace binding")
        return value

    def summary(self):
        return {
            "request_id": self.request_id,
            "source_verified": self.source_verified,
            "source_verify_count": self.source_verify_count,
            "source_verify_s": self.source_verify_s,
            "manifest_digest": self.manifest_digest,
            "source_generation": self.source_generation,
            "implementation_generation": self.implementation_generation,
            "device_identity": self.device_identity,
            "ordinary_runtime_identity": repr(self.ordinary_runtime_identity),
            "partitioned_runtime_identity": repr(self.partitioned_runtime_identity),
            "ordinary_compiler_namespace": repr(self.ordinary_compiler_namespace),
            "partitioned_compiler_namespace": repr(self.partitioned_compiler_namespace),
            "compiler_environment": dict(self.compiler_environment),
        }


@dataclass(frozen=True)
class ValidationTicket:
    digest: str
    key: dict
    payload_size: int
    generation: int
    validate: bool
    publish: bool
    event: object | None = None


class ArithmeticValidationState:
    """Bounded request-local successful-proof service with per-key in-flight ownership."""

    def __init__(self, *, max_entries=MAX_SUCCESS_ENTRIES, max_bytes=MAX_SUCCESS_BYTES):
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.generation = 0
        self._entries = OrderedDict()
        self._bytes = 0
        self._inflight = {}
        self._lock = threading.Lock()
        self._evicted_queue = deque()
        self._evicted = set()
        self._pending_miss_reason = None
        self.stats = Counter()
        self.miss_reasons = Counter()
        self.examples = []
        self.gate_total_s = 0.0
        self.untimed_gate_count = 0
        self.production_host_wall_s = 0.0

    @staticmethod
    def _geometry(key):
        names = (
            "layout", "dtype", "batch", "heads", "head_dim", "q_rows", "kv_rows",
            "q_stride", "k_stride", "v_stride", "alignment", "scale",
        )
        return tuple((name, repr(key.get(name))) for name in names)

    def _classify_miss(self, digest, key):
        if self._pending_miss_reason is not None:
            reason = self._pending_miss_reason
            self._pending_miss_reason = None
            return reason
        if digest in self._evicted:
            return "evicted"
        if not self._entries:
            return "new_request" if self.stats["attempts"] == 0 else "new_geometry"
        values = [entry[2] for entry in self._entries.values()]
        same_mode = [old for old in values if old.get("mode") == key.get("mode")]
        if not same_mode:
            return "new_geometry"
        for old in same_mode:
            if (
                old.get("source_generation") != key.get("source_generation")
                or old.get("runtime_identity") != key.get("runtime_identity")
                or old.get("compiler_namespace") != key.get("compiler_namespace")
            ):
                return "new_runtime"
        for old in same_mode:
            if old.get("validation_generation") != key.get("validation_generation"):
                return "numerical_transition"
        for old in same_mode:
            if old.get("bias") != key.get("bias"):
                return "new_bias"
        if all(self._geometry(old) != self._geometry(key) for old in same_mode):
            return "new_geometry"
        return "new_geometry"

    def begin(self, key):
        try:
            payload = _canonical_bytes(key)
        except (TypeError, ValueError):
            with self._lock:
                self.stats["misses"] += 1
                self.stats["attempts"] += 1
                self.miss_reasons["unrepresentable_identity"] += 1
            return ValidationTicket("", key, 0, self.generation, True, False)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) > self.max_bytes:
            with self._lock:
                self.stats["misses"] += 1
                self.stats["attempts"] += 1
                self.miss_reasons["unrepresentable_identity"] += 1
            return ValidationTicket(digest, key, len(payload), self.generation, True, False)

        owner = threading.get_ident()
        while True:
            with self._lock:
                hit = self._entries.get(digest)
                if hit is not None and hit[1] == payload:
                    self._entries.move_to_end(digest)
                    self.stats["hits"] += 1
                    return ValidationTicket(digest, key, len(payload), self.generation, False, False)
                inflight = self._inflight.get(digest)
                if inflight is None:
                    event = threading.Event()
                    self._inflight[digest] = (owner, event, self.generation)
                    self.stats["misses"] += 1
                    self.stats["attempts"] += 1
                    self.miss_reasons[self._classify_miss(digest, key)] += 1
                    return ValidationTicket(digest, key, len(payload), self.generation, True, True, event)
                inflight_owner, event, _generation = inflight
                if inflight_owner == owner:
                    self.stats["misses"] += 1
                    self.stats["attempts"] += 1
                    self.miss_reasons["unrepresentable_identity"] += 1
                    return ValidationTicket(digest, key, len(payload), self.generation, True, False)
            event.wait()

    def _remember_eviction(self, digest):
        if digest in self._evicted:
            return
        self._evicted.add(digest)
        self._evicted_queue.append(digest)
        while len(self._evicted_queue) > EVICTED_DIGEST_HISTORY:
            old = self._evicted_queue.popleft()
            self._evicted.discard(old)

    def publish_success(self, ticket, metrics):
        if not ticket.publish:
            return
        scalar_metrics = {
            name: value for name, value in metrics.items()
            if value is None or type(value) in {bool, int, float, str}
        }
        record_size = ticket.payload_size + len(_canonical_bytes(scalar_metrics))
        with self._lock:
            inflight = self._inflight.pop(ticket.digest, None)
            try:
                if ticket.generation != self.generation:
                    self.stats["generation_publish_rejected"] += 1
                    return
                if record_size > self.max_bytes:
                    self.stats["oversized_success_bypass"] += 1
                    return
                old = self._entries.pop(ticket.digest, None)
                if old is not None:
                    self._bytes -= old[0]
                self._entries[ticket.digest] = (record_size, _canonical_bytes(ticket.key), ticket.key, scalar_metrics)
                self._bytes += record_size
                while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                    old_digest, old_entry = self._entries.popitem(last=False)
                    self._bytes -= old_entry[0]
                    self.stats["evictions"] += 1
                    self._remember_eviction(old_digest)
                self.stats["successes"] += 1
            finally:
                if inflight is not None:
                    inflight[1].set()

    def publish_failure(self, ticket):
        self.stats["failures"] += 1
        if not ticket.publish:
            return
        with self._lock:
            inflight = self._inflight.pop(ticket.digest, None)
            if inflight is not None:
                inflight[1].set()

    def invalidate(self, reason="numerical_transition"):
        with self._lock:
            self.generation += 1
            self._entries.clear()
            self._bytes = 0
            self.stats["invalidations"] += 1
            self._pending_miss_reason = reason

    def record_gate(self, ticket, *, mode, gate_wall_s, metrics, telemetry, context=None):
        if gate_wall_s is None:
            self.untimed_gate_count += 1
        else:
            self.gate_total_s += float(gate_wall_s)
        if telemetry.get("compile_hit"):
            self.stats["compile_hits"] += 1
        if telemetry.get("compile_miss"):
            self.stats["compile_misses"] += 1
        if telemetry.get("compile_race_hit"):
            self.stats["compile_race_hits"] += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append({
                "mode": mode,
                "context": dict(context or {}),
                "arithmetic_key_digest": ticket.digest or None,
                "compiler_key_digest": hashlib.sha256(
                    str(telemetry.get("compiler_key", "")).encode("utf-8")
                ).hexdigest() if telemetry.get("compiler_key") else None,
                "gate_wall_s": gate_wall_s,
                "compile_hit": bool(telemetry.get("compile_hit")),
                "compile_miss": bool(telemetry.get("compile_miss")),
                "compile_lock_wait_s": telemetry.get("compile_lock_wait_s"),
                "compile_body_s": telemetry.get("compile_body_s"),
                "prepare_jit_host_wall_s": telemetry.get("prepare_jit_host_wall_s"),
                "dispatch_host_enqueue_s": telemetry.get("dispatch_host_enqueue_s"),
                "all_selected_host_wall_s": telemetry.get("all_selected_host_wall_s"),
                "reference_host_wall_s": telemetry.get("reference_host_wall_s"),
                "reduction_host_wall_s": telemetry.get("reduction_host_wall_s"),
                **{
                    name: metrics.get(name)
                    for name in (
                        "finite", "max_abs", "mean_abs", "rel_l2",
                        "reference_peak_abs", "catastrophic_max_abs_limit",
                    )
                },
            })

    def record_production(self, host_wall_s, telemetry=None):
        telemetry = telemetry or {}
        self.stats["production_calls"] += 1
        self.production_host_wall_s += float(host_wall_s)
        if telemetry.get("compile_hit"):
            self.stats["compile_hits"] += 1
        if telemetry.get("compile_miss"):
            self.stats["compile_misses"] += 1
        if telemetry.get("compile_race_hit"):
            self.stats["compile_race_hits"] += 1

    def summary(self):
        return {
            "validation_generation": self.generation,
            "cache_entries": len(self._entries),
            "cache_bytes": self._bytes,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "attempts": self.stats["attempts"],
            "successes": self.stats["successes"],
            "failures": self.stats["failures"],
            "evictions": self.stats["evictions"],
            "invalidations": self.stats["invalidations"],
            "compile_hits": self.stats["compile_hits"],
            "compile_misses": self.stats["compile_misses"],
            "compile_race_hits": self.stats["compile_race_hits"],
            "miss_reasons": dict(self.miss_reasons),
            "gate_total_s": self.gate_total_s,
            "untimed_gate_count": self.untimed_gate_count,
            "production_calls": self.stats["production_calls"],
            "production_host_wall_s": self.production_host_wall_s,
            "examples": list(self.examples),
        }


def build_arithmetic_key(
    *,
    state,
    q,
    k,
    v,
    mode,
    layout,
    scale,
    mapped_abi=None,
    bias_range=None,
    bias_log_measure=None,
    bias_identity=None,
    physical_identity=None,
):
    """Build the conservative v1 proof key without hashing Q/K/V contents."""
    lease = state.runtime_lease
    if not lease.source_verified:
        raise RuntimeError("Sol-H3 arithmetic key requires a verified runtime lease")

    if layout == "bhtd":
        batch, heads, q_rows, head_dim = map(int, q.shape)
        kv_rows = int(k.shape[2])
    elif layout == "thd":
        q_rows, heads, head_dim = map(int, q.shape)
        kv_rows = int(k.shape[0])
        batch = 1
    else:
        raise ValueError("unsupported arithmetic-key layout")

    bias = {
        "enabled": bias_range is not None or bias_identity is not None,
        "range": bias_range,
        "log_measure_fp32": (
            None
            if bias_log_measure is None
            else struct.unpack(">f", struct.pack(">f", float(bias_log_measure)))[0].hex()
        ),
        "identity": bias_identity,
    }
    return {
        "abi": KEY_ABI,
        "mode": mode,
        "source_generation": lease.source_generation,
        "implementation_generation": lease.implementation_generation,
        "runtime_identity": repr(lease.runtime_identity_for(mode)),
        "compiler_namespace": repr(lease.compiler_namespace_for(mode)),
        "device": lease.device_identity,
        "validation_generation": state.validation_state.generation,
        "config_fingerprint": state.config.metadata()["fingerprint"],
        "layout": layout,
        "dtype": str(q.dtype),
        "batch": batch,
        "heads": heads,
        "head_dim": head_dim,
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "q_stride": list(map(int, q.stride())),
        "k_stride": list(map(int, k.stride())),
        "v_stride": list(map(int, v.stride())),
        "alignment": [
            int(q.data_ptr()) & 15,
            int(k.data_ptr()) & 15,
            int(v.data_ptr()) & 15,
            int(q.storage_offset()),
            int(k.storage_offset()),
            int(v.storage_offset()),
        ],
        "scale": float(scale).hex(),
        "mapped_abi": mapped_abi,
        "bias": bias,
        # Physical identity remains conservative for partitioned v1.  It may be
        # quotiented only after CUDA evidence proves all-selected independence.
        "physical_identity": physical_identity,
    }


__all__ = [
    "ArithmeticValidationState",
    "KEY_ABI",
    "RuntimeLease",
    "ValidationTicket",
    "build_arithmetic_key",
]
