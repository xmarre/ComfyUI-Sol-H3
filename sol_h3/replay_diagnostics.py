"""Opt-in same-input replay for Sol-H3 arithmetic/performance attribution.

Replay is diagnostic-only. It operates on detached Q/K/V snapshots taken after
preprocessing/gathering, calls only the Sol arithmetic operator and dense
reference, discards every replay output, and never re-enters model/provider/VDN
or Spectrum code. Successful arithmetic proofs remain Request-local; replay uses
private validation-state instances so it cannot accept a production call.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import random
import time

import torch

from .validation import ArithmeticValidationState


_ENV = "SOL_H3_REPLAY_DIAGNOSTICS"
TARGETS = (
    "ordinary_low",
    "ordinary_continuation_high",
    "partitioned_suffix",
)
ARMS = (
    "first_executable_fresh_validation",
    "primed_executable_fresh_validation",
    "primed_executable_retained_validation",
)


def _enabled_from_env() -> bool:
    value = os.environ.get(_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _storage_extent(tensor: torch.Tensor) -> int:
    if any(int(stride) < 0 for stride in tensor.stride()):
        raise RuntimeError("Sol-H3 replay does not support negative-stride tensors")
    if tensor.numel() == 0:
        return 0
    return 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )


def clone_tensors_preserve_layout(tensors):
    """Clone tensors while preserving exact shape/stride compiler geometry.

    Tensors sharing one storage are cloned into one replacement storage with the
    same relative storage offsets. This avoids tripling the interleaved H3 Q/K/V
    padding while retaining the stride tuple used by the CuTe specialization key.
    """
    tensors = tuple(tensors)
    if not tensors:
        return ()
    if any(not torch.is_tensor(tensor) for tensor in tensors):
        raise TypeError("Sol-H3 replay snapshots require tensors")

    first = tensors[0]
    same_storage = all(
        tensor.dtype == first.dtype
        and tensor.device == first.device
        and tensor.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
        for tensor in tensors
    )
    if same_storage:
        minimum = min(int(tensor.storage_offset()) for tensor in tensors)
        maximum = max(
            int(tensor.storage_offset()) + _storage_extent(tensor)
            for tensor in tensors
        )
        storage = torch.empty(maximum - minimum, dtype=first.dtype, device=first.device)
        copies = []
        for tensor in tensors:
            view = storage.as_strided(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                int(tensor.storage_offset()) - minimum,
            )
            view.copy_(tensor)
            copies.append(view)
        return tuple(copies)

    copies = []
    for tensor in tensors:
        storage = torch.empty(_storage_extent(tensor), dtype=tensor.dtype, device=tensor.device)
        view = storage.as_strided(tuple(tensor.shape), tuple(tensor.stride()), 0)
        view.copy_(tensor)
        copies.append(view)
    return tuple(copies)


@contextmanager
def _preserve_rng(device):
    python_state = random.getstate()
    cpu_state = torch.get_rng_state().clone()
    cuda_device = None
    cuda_state = None
    if device.type == "cuda":
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_device).clone()
    status = {"restored": False, "error": None}
    try:
        yield status
    finally:
        errors = []
        try:
            random.setstate(python_state)
        except (TypeError, ValueError) as exc:
            errors.append(f"python_rng:{exc}")
        try:
            torch.set_rng_state(cpu_state)
        except RuntimeError as exc:
            errors.append(f"cpu_rng:{exc}")
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state(cuda_state, cuda_device)
            except RuntimeError as exc:
                errors.append(f"cuda_rng:{exc}")
        try:
            python_ok = random.getstate() == python_state
            cpu_ok = bool(torch.equal(torch.get_rng_state(), cpu_state))
            cuda_ok = True
            if cuda_state is not None:
                cuda_ok = bool(torch.equal(torch.cuda.get_rng_state(cuda_device), cuda_state))
            status["restored"] = bool(python_ok and cpu_ok and cuda_ok)
            if not status["restored"]:
                errors.append("rng_state_mismatch")
        except RuntimeError as exc:
            errors.append(f"rng_verify:{exc}")
        if errors:
            status["error"] = "; ".join(errors)


class ReplayDiagnosticState:
    """Request-owned, bounded replay controller.

    At most one snapshot of each target class is admitted. Snapshot tensors are
    owned by the caller only for the duration of execute(); this object retains
    scalar receipts and context metadata only.
    """

    def __init__(self, *, enabled=False):
        self.enabled = bool(enabled)
        self._claimed = set()
        self._reports = []
        self.total_host_wall_s = 0.0
        self.errors = 0
        self.configuration_error = None

    @classmethod
    def from_env(cls):
        return cls(enabled=_enabled_from_env())

    def claim_ordinary(self, context):
        if not self.enabled:
            return None
        stage = (context or {}).get("flow_stage")
        if stage == "low":
            target = "ordinary_low"
        elif stage == "high":
            target = "ordinary_continuation_high"
        else:
            return None
        return self._claim(target)

    def claim_partitioned_suffix(self, *, kind, mapped, force_dense):
        if not self.enabled or kind != "local" or not mapped or force_dense:
            return None
        return self._claim("partitioned_suffix")

    def _claim(self, target):
        if target not in TARGETS or target in self._claimed:
            return None
        self._claimed.add(target)
        return target

    @staticmethod
    def _arm_report(name, ticket, gate_telemetry, production_telemetry, gate_wall_s, production_wall_s):
        gate_telemetry = gate_telemetry or {}
        production_telemetry = production_telemetry or {}
        return {
            "arm": name,
            "validation_generation": int(ticket.generation),
            "proof_hit": not bool(ticket.validate),
            "gate_performed": bool(ticket.validate),
            "gate_host_wall_s": gate_wall_s,
            "production_host_wall_s": production_wall_s,
            "compile_hits": int(bool(gate_telemetry.get("compile_hit")))
            + int(bool(production_telemetry.get("compile_hit"))),
            "compile_misses": int(bool(gate_telemetry.get("compile_miss")))
            + int(bool(production_telemetry.get("compile_miss"))),
            "compile_race_hits": int(bool(gate_telemetry.get("compile_race_hit")))
            + int(bool(production_telemetry.get("compile_race_hit"))),
            "gate_telemetry": {
                key: value
                for key, value in gate_telemetry.items()
                if key != "_compile_cache_lookups"
            },
            "production_telemetry": {
                key: value
                for key, value in production_telemetry.items()
                if key != "_compile_cache_lookups"
            },
        }

    def execute(
        self,
        *,
        target,
        arithmetic_key,
        device,
        context,
        cuda_diagnostics,
        gate,
        production,
    ):
        """Run executable/proof-lifetime replay without changing live proof state.

        gate receives (sample, arm) and returns (metrics, telemetry).
        production receives (sample, arm) and returns telemetry. Callers own and
        release all replay tensors around this method.
        """
        if not self.enabled:
            return
        if not cuda_diagnostics.enabled:
            self.configuration_error = (
                "SOL_H3_REPLAY_DIAGNOSTICS requires SOL_H3_CUDA_DIAGNOSTICS"
            )
            self.errors += 1
            return

        started = time.perf_counter()
        report = {
            "target": target,
            "context": dict(context or {}),
            "arithmetic_key_digest": None,
            "arms": [],
            "rng_restored": False,
            "rng_restore_error": None,
            "provider_history_reentered": False,
            "vdn_runtime_reentered": False,
            "bsa_pool_reentered": False,
            "stream_policy": "current_stream_only",
            "output_policy": "discarded",
            "error": None,
        }
        with _preserve_rng(device) as rng_status:
            try:
                first = ArithmeticValidationState()
                primed = ArithmeticValidationState()
                services = (first, primed, primed)
                for arm, service in zip(ARMS, services, strict=True):
                    ticket = service.begin(arithmetic_key)
                    report["arithmetic_key_digest"] = ticket.digest or None
                    gate_telemetry = {}
                    gate_wall_s = None
                    if ticket.validate:
                        sample = cuda_diagnostics.begin_sample(
                            "replay_arithmetic_gate",
                            device,
                            context={
                                **dict(context or {}),
                                "replay_target": target,
                                "replay_arm": arm,
                            },
                            important=True,
                        )
                        cuda_diagnostics.initial_stream_drain(sample)
                        gate_started = time.perf_counter()
                        try:
                            metrics, gate_telemetry = gate(sample, arm)
                        except BaseException:
                            service.publish_failure(ticket)
                            raise
                        gate_wall_s = time.perf_counter() - gate_started
                        service.publish_success(ticket, metrics)
                    else:
                        metrics = None

                    production_sample = cuda_diagnostics.begin_sample(
                        "replay_production_sparse",
                        device,
                        context={
                            **dict(context or {}),
                            "replay_target": target,
                            "replay_arm": arm,
                        },
                        important=True,
                    )
                    production_started = time.perf_counter()
                    production_telemetry = production(production_sample, arm)
                    production_wall_s = time.perf_counter() - production_started
                    arm_report = self._arm_report(
                        arm,
                        ticket,
                        gate_telemetry,
                        production_telemetry,
                        gate_wall_s,
                        production_wall_s,
                    )
                    if metrics is not None:
                        arm_report["gate_metrics"] = {
                            key: value
                            for key, value in metrics.items()
                            if value is None or type(value) in {bool, int, float, str}
                        }
                    report["arms"].append(arm_report)

                if not report["arms"][0]["gate_performed"]:
                    raise RuntimeError("first fresh replay validation unexpectedly hit")
                if not report["arms"][1]["gate_performed"]:
                    raise RuntimeError("second fresh replay validation unexpectedly hit")
                if not report["arms"][2]["proof_hit"] or report["arms"][2]["gate_performed"]:
                    raise RuntimeError("retained replay validation did not reuse its successful proof")
            except BaseException as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                self.errors += 1

        report["rng_restored"] = bool(rng_status["restored"])
        report["rng_restore_error"] = rng_status["error"]
        if not report["rng_restored"]:
            self.errors += 1
            if report["error"] is None:
                report["error"] = "RNG state did not restore exactly"
        report["host_wall_s"] = time.perf_counter() - started
        self.total_host_wall_s += report["host_wall_s"]
        self._reports.append(report)

    def summary(self):
        return {
            "enabled": self.enabled,
            "configuration_error": self.configuration_error,
            "claimed_targets": sorted(self._claimed),
            "completed_targets": [
                report["target"] for report in self._reports if report.get("error") is None
            ],
            "errors": self.errors,
            "total_host_wall_s": self.total_host_wall_s,
            "reports": list(self._reports),
        }


__all__ = [
    "ARMS",
    "ReplayDiagnosticState",
    "TARGETS",
    "clone_tensors_preserve_layout",
]
