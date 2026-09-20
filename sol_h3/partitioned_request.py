"""Request-owned single-union Sol route for exact-prefix progressive attention.

The first partition prototype evaluated physical K/V domains independently and
merged their LSEs. That is algebraically exact for dense attention, but it changes
Sol's sparse routing because each partition receives an independently computed
routing threshold. The production-shaped route in this module instead keeps one
physical K/V union, applies the target-prefix key measure as an additive natural-
log bias, and combines that with VDN-owned mapped-neighbor metadata in one SM120
kernel invocation.

This module is opt-in. Released Sol-H3 routing is unchanged unless the Flow/VDN
partitioned development path imports and calls ``partitioned_request_attention``.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F

from .mapped_neighbors import (
    POLICY as MAPPED_POLICY,
    MappingUnavailable,
    compile_descriptor,
    device_descriptor,
    validate_wire_map,
)
from .sparse import arithmetic_gate_passes, error_metrics

PARTITIONED_REQUEST_ABI = "sol-h3-partitioned-single-union-v1"
PARTITIONED_RECEIPT_TAG = "sol_h3_partitioned_exact_prefix_v1"
PARTITIONED_DENSE_ROUTE = "partitioned_dense"
PARTITIONED_SOL_ROUTE = "partitioned_sol"
PARTITIONED_MAPPED_ROUTE = "partitioned_sol_mapped"
MAX_BIAS_CACHE_ENTRIES = 64
MAX_BIAS_CACHE_BYTES = 4 * 1024 * 1024
BLOCK_SIZE = 64
_FORCE_DENSE_SUFFIX_DIAGNOSTIC_ENV = "SOL_H3_FORCE_DENSE_PARTITIONED_SUFFIX_DIAGNOSTIC"


def _force_dense_partitioned_suffix_diagnostic_enabled() -> bool:
    value = os.environ.get(_FORCE_DENSE_SUFFIX_DIAGNOSTIC_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_force_dense_suffix(*, kind: str, force_dense: bool, warmup: bool) -> bool:
    return bool(
        not force_dense
        and not warmup
        and kind == "local"
        and _force_dense_partitioned_suffix_diagnostic_enabled()
    )


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _request_state():
    from .runtime import _REQUEST

    state = _REQUEST.get()
    if state is None:
        raise RuntimeError("partitioned Sol must execute inside the native Sol OUTER_SAMPLE lifecycle")
    return state


def _active_forward():
    from .runtime import _FORWARD

    active = _FORWARD.get()
    if active is None or len(active) != 5:
        raise RuntimeError("partitioned Sol must execute inside the native Sol diffusion lifecycle")
    return active


def _forward_evaluation() -> int:
    return int(_active_forward()[2])


def _validate_thd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if (
        any(t.ndim != 3 for t in (q, k, v))
        or k.shape != v.shape
        or q.shape[1:] != k.shape[1:]
        or q.shape[0] <= 0
        or k.shape[0] <= 0
        or q.shape[1] <= 0
        or q.shape[-1] != 128
    ):
        raise RuntimeError("partitioned Sol requires Q [Tq,H,128] and KV [Tkv,H,128]")
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        raise RuntimeError("partitioned Sol requires BF16 QKV on one device")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("partitioned Sol currently requires SM120")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise RuntimeError("partitioned Sol requires a contiguous head dimension")
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("partitioned Sol is unavailable under autograd, torch.compile, or CUDA graph capture")


def _bias_cache(state) -> OrderedDict:
    cache = getattr(state, "partitioned_measure_bias_cache", None)
    if cache is None:
        cache = OrderedDict()
        state.partitioned_measure_bias_cache = cache
        state.partitioned_measure_bias_bytes = 0
    if not isinstance(cache, OrderedDict):
        raise RuntimeError("partitioned Sol bias-cache ownership was replaced")
    return cache


def _key_bias(
    state,
    *,
    kv_rows: int,
    prefix_range: tuple[int, int] | None,
    log_measure: float,
    device: torch.device,
    semantic_digest: str,
) -> torch.Tensor | None:
    if prefix_range is None:
        if log_measure != 0.0:
            raise RuntimeError("partitioned Sol received a measure without a target-prefix K/V range")
        return None
    if (
        not isinstance(prefix_range, tuple)
        or len(prefix_range) != 2
        or any(type(value) is not int for value in prefix_range)
    ):
        raise RuntimeError("partitioned Sol prefix K/V range must be an integer pair")
    start, end = prefix_range
    if not 0 <= start < end <= kv_rows:
        raise RuntimeError("partitioned Sol prefix K/V range is outside the current domain")
    log_measure = float(log_measure)
    if not math.isfinite(log_measure) or log_measure > 0.0:
        raise RuntimeError("partitioned Sol target-prefix log measure must be finite and non-positive")
    if not _digest(semantic_digest):
        raise RuntimeError("partitioned Sol semantic digest is invalid")

    cache = _bias_cache(state)
    key = (
        PARTITIONED_REQUEST_ABI,
        semantic_digest,
        kv_rows,
        start,
        end,
        log_measure,
        str(device),
    )
    with torch.cuda.device(device):
        current = torch.cuda.current_stream(device)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            bias, event, _size = hit
            if event is None:
                raise RuntimeError("partitioned Sol CUDA bias cache entry is missing its stream event")
            current.wait_event(event)
            bias.record_stream(current)
            return bias
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("partitioned Sol cannot allocate key-measure metadata during CUDA graph capture")
        bias = torch.zeros(kv_rows, dtype=torch.float32, device=device)
        bias[start:end] = log_measure
        event = torch.cuda.Event()
        event.record(current)
        bias.record_stream(current)

    size = bias.numel() * bias.element_size()
    cache[key] = (bias, event, size)
    cache.move_to_end(key)
    state.partitioned_measure_bias_bytes += size
    while len(cache) > MAX_BIAS_CACHE_ENTRIES or state.partitioned_measure_bias_bytes > MAX_BIAS_CACHE_BYTES:
        _, (_evicted, _event, evicted_size) = cache.popitem(last=False)
        state.partitioned_measure_bias_bytes -= evicted_size
    return bias


def _weighted_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_bias: torch.Tensor | None,
    *,
    scale: float,
) -> torch.Tensor:
    qh = q.transpose(0, 1).unsqueeze(0)
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    mask = None if key_bias is None else key_bias.to(q.dtype).view(1, 1, 1, -1)
    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=float(scale))
    return out.squeeze(0).transpose(0, 1).contiguous()


def _sm120_union(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tau: float,
    scale: float,
    sink_rows: int,
    key_bias: torch.Tensor | None,
    mapped_neighbor_intervals: torch.Tensor | None,
    telemetry: dict | None = None,
    diagnostic_state=None,
    diagnostic_sample=None,
) -> torch.Tensor:
    """Execute one SM120 sparse attention union with optional bias + mapped metadata."""
    from ._vendor.sol_attn import interface
    from ._vendor.sol_attn.preprocess import prepare

    qb = q.unsqueeze(0)
    kb = k.unsqueeze(0)
    vb = v.unsqueeze(0)
    arch = tuple(torch.cuda.get_device_capability(q.device))
    if interface._backend_for_arch(arch) != "cute_sm120":
        raise RuntimeError("partitioned Sol requires the packaged cute_sm120 backend")
    interface._validate_cute(arch, kb.shape[1], 1)
    if type(sink_rows) is not int or not 0 <= sink_rows <= kb.shape[1]:
        raise RuntimeError("partitioned Sol exact K prefix is outside the K/V domain")

    if key_bias is not None and (
        key_bias.ndim != 1
        or key_bias.shape[0] != kb.shape[1]
        or key_bias.dtype != torch.float32
        or key_bias.device != q.device
        or not key_bias.is_contiguous()
    ):
        raise RuntimeError("partitioned Sol key bias must be contiguous FP32 [Tkv]")
    if mapped_neighbor_intervals is not None:
        expected = ((q.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE, 2)
        if (
            mapped_neighbor_intervals.ndim != 2
            or tuple(mapped_neighbor_intervals.shape) != expected
            or mapped_neighbor_intervals.dtype != torch.int32
            or mapped_neighbor_intervals.device != q.device
            or not mapped_neighbor_intervals.is_contiguous()
        ):
            raise RuntimeError("partitioned Sol mapped-neighbor metadata is invalid")

    batch, q_rows, heads, _ = qb.shape
    kv_rows = kb.shape[1]
    with torch.cuda.device(q.device):
        prepare_started = time.perf_counter()
        manager = (
            diagnostic_state.span(diagnostic_sample, "prepare")
            if diagnostic_state is not None
            else __import__("contextlib").nullcontext()
        )
        with manager:
            kc, vc, threshold = prepare(
                qb,
                kb,
                vb,
                scale=float(scale),
                tau=float(tau),
                thresh_type="diag",
                valid_tokens=q_rows,
                valid_kv_tokens=kv_rows,
            )
        if telemetry is not None:
            telemetry["prepare_jit_host_wall_s"] = telemetry.get("prepare_jit_host_wall_s", 0.0) + (time.perf_counter() - prepare_started)
        output = torch.empty_like(qb)
        lse = torch.empty((batch, q_rows, heads), device=q.device, dtype=torch.float32)
        stream = interface._stream(q.device)
        sink_start_block, sink_end_block = interface._sink_block_range(kv_rows, 0, sink_rows)
        key_bias_arg = key_bias if key_bias is not None else threshold
        mapped_arg = mapped_neighbor_intervals if mapped_neighbor_intervals is not None else threshold
        tensors = [qb, kb, vb, output, kc, vc, threshold, key_bias_arg, mapped_arg, lse]
        layout_key = tuple(tuple(int(value) for value in tensor.stride()) for tensor in (qb, kb, vb))
        key = (
            PARTITIONED_REQUEST_ABI,
            q.device.index,
            arch,
            batch,
            q_rows,
            kv_rows,
            heads,
            layout_key,
            key_bias is not None,
            mapped_neighbor_intervals is not None,
        )
        if telemetry is not None:
            telemetry["compiler_key"] = repr(key)
        compiled = interface._compiled.get(key)
        if telemetry is not None:
            telemetry["compile_cache_initial_hit"] = compiled is not None
        if compiled is None:
            lock_started = time.perf_counter()
            with interface._compile_lock:
                if telemetry is not None:
                    telemetry["compile_lock_wait_s"] = telemetry.get("compile_lock_wait_s", 0.0) + (time.perf_counter() - lock_started)
                compiled = interface._compiled.get(key)
                if compiled is None:
                    compile_started = time.perf_counter()
                    compiled, args = interface._compile_sm120(
                        key,
                        tensors,
                        float(scale),
                        sink_start_block,
                        sink_end_block,
                        stream,
                        key_bias is not None,
                        mapped_neighbor_intervals is not None,
                    )
                    if telemetry is not None:
                        telemetry["compile_body_s"] = telemetry.get("compile_body_s", 0.0) + (time.perf_counter() - compile_started)
                        telemetry["compile_miss"] = True
                else:
                    args = interface._to_cute_tensors(tensors)
                    if telemetry is not None:
                        telemetry["compile_race_hit"] = True
        else:
            args = interface._to_cute_tensors(tensors)
            if telemetry is not None:
                telemetry["compile_hit"] = True
        dispatch_started = time.perf_counter()
        manager = (
            diagnostic_state.span(diagnostic_sample, "compiled_dispatch")
            if diagnostic_state is not None
            else __import__("contextlib").nullcontext()
        )
        with manager:
            compiled(*args, float(scale), sink_start_block, sink_end_block, stream=stream)
        if telemetry is not None:
            telemetry["dispatch_host_enqueue_s"] = telemetry.get("dispatch_host_enqueue_s", 0.0) + (time.perf_counter() - dispatch_started)
    return output[0]


def _replay_partitioned_suffix(
    q,
    k,
    v,
    *,
    key_bias,
    mapped_neighbor_intervals,
    exact_end,
    config,
    state,
    arithmetic_key,
    validation_context,
    kind,
):
    target = state.replay_diagnostics.claim_partitioned_suffix(
        kind=kind,
        mapped=mapped_neighbor_intervals is not None,
        force_dense=False,
    )
    if target is None:
        return
    diagnostics = state.cuda_diagnostics
    if not diagnostics.enabled:
        if state.replay_diagnostics.configuration_error is None:
            state.replay_diagnostics.configuration_error = (
                "SOL_H3_REPLAY_DIAGNOSTICS requires SOL_H3_CUDA_DIAGNOSTICS"
            )
            state.replay_diagnostics.errors += 1
        return

    from .replay_diagnostics import clone_tensors_preserve_layout

    context = {
        "mode": arithmetic_key.get("mode"),
        **dict(validation_context or {}),
        "snapshot_logical_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in (q, k, v)
        ),
    }
    try:
        replay_q, replay_k, replay_v = clone_tensors_preserve_layout((q, k, v))
        replay_bias = None if key_bias is None else key_bias.clone()
        replay_mapped = (
            None
            if mapped_neighbor_intervals is None
            else mapped_neighbor_intervals.clone()
        )

        def gate(sample, _arm):
            telemetry = {}
            all_selected_started = time.perf_counter()
            with diagnostics.span(sample, "all_selected_call"):
                got = _sm120_union(
                    replay_q,
                    replay_k,
                    replay_v,
                    tau=config.tau,
                    scale=replay_q.shape[-1] ** -0.5,
                    sink_rows=int(replay_k.shape[0]),
                    key_bias=replay_bias,
                    mapped_neighbor_intervals=replay_mapped,
                    telemetry=telemetry,
                    diagnostic_state=diagnostics,
                    diagnostic_sample=sample,
                )
            telemetry["all_selected_host_wall_s"] = (
                time.perf_counter() - all_selected_started
            )
            reference_started = time.perf_counter()
            with diagnostics.span(sample, "dense_reference"):
                want = _weighted_dense(
                    replay_q,
                    replay_k,
                    replay_v,
                    replay_bias,
                    scale=replay_q.shape[-1] ** -0.5,
                )
            telemetry["reference_host_wall_s"] = (
                time.perf_counter() - reference_started
            )
            reduction_started = time.perf_counter()
            with diagnostics.span(sample, "error_reduction"):
                metrics = error_metrics(
                    got.transpose(0, 1).unsqueeze(0),
                    want.transpose(0, 1).unsqueeze(0),
                )
            telemetry["reduction_host_wall_s"] = (
                time.perf_counter() - reduction_started
            )
            del got, want
            if not arithmetic_gate_passes(metrics):
                raise RuntimeError(
                    f"partitioned Sol replay arithmetic gate failed: {metrics}"
                )
            return metrics, telemetry

        def production(sample, _arm):
            telemetry = {}
            with diagnostics.span(sample, "production_call"):
                output = _sm120_union(
                    replay_q,
                    replay_k,
                    replay_v,
                    tau=config.tau,
                    scale=replay_q.shape[-1] ** -0.5,
                    sink_rows=exact_end,
                    key_bias=replay_bias,
                    mapped_neighbor_intervals=replay_mapped,
                    telemetry=telemetry,
                    diagnostic_state=diagnostics,
                    diagnostic_sample=sample,
                )
            # The existing arithmetic gate proves only the all-selected kernel
            # against weighted dense attention. That does not test the sparse
            # selection actually used in production. Replay is already an
            # opt-in diagnostic path, so compare that exact sparse output with
            # the same weighted dense reference before discarding both tensors.
            with diagnostics.span(sample, "production_dense_reference"):
                dense = _weighted_dense(
                    replay_q,
                    replay_k,
                    replay_v,
                    replay_bias,
                    scale=replay_q.shape[-1] ** -0.5,
                )
            with diagnostics.span(sample, "production_error_reduction"):
                production_error = error_metrics(
                    output.transpose(0, 1).unsqueeze(0),
                    dense.transpose(0, 1).unsqueeze(0),
                )
            for name, value in production_error.items():
                if value is None or type(value) in {bool, int, float, str}:
                    telemetry[f"production_vs_dense_{name}"] = value
            del output, dense
            return telemetry

        state.replay_diagnostics.execute(
            target=target,
            arithmetic_key=arithmetic_key,
            device=q.device,
            context=context,
            cuda_diagnostics=diagnostics,
            gate=gate,
            production=production,
        )
    except Exception as exc:
        state.replay_diagnostics.record_capture_error(target, context, exc)


def _descriptor_for_wire(
    state,
    wire,
    *,
    q_rows: int,
    kv_rows: int,
    sink_rows: int,
    device,
    materialize: bool,
):
    if wire is None:
        return None, None, None
    try:
        validated = validate_wire_map(wire, q_rows=q_rows, kv_rows=kv_rows, sink_rows=sink_rows)
        descriptor = compile_descriptor(validated)
        mapped = None
        if descriptor is not None and materialize:
            mapped = device_descriptor(state, descriptor, device)
        return validated, descriptor, mapped
    except MappingUnavailable as exc:
        raise RuntimeError(f"partitioned Sol mapped-neighbor contract is unavailable: {exc.reason}") from exc


def _completion_fields(
    *,
    evaluation: int,
    semantic_digest: str,
    kind: str,
    execution_mode: str,
    q_rows: int,
    kv_rows: int,
    sink_rows: int,
    prefix_k_range,
    prefix_log_key_measure: float,
    validated,
    descriptor,
    kernel_contract: str | None,
):
    return (
        PARTITIONED_RECEIPT_TAG,
        ("sol_h3_evaluation", int(evaluation)),
        PARTITIONED_REQUEST_ABI,
        semantic_digest,
        kind,
        execution_mode,
        int(q_rows),
        int(kv_rows),
        int(sink_rows),
        prefix_k_range,
        float(prefix_log_key_measure),
        None if validated is None else validated.map_digest,
        None if descriptor is None else descriptor.descriptor_digest,
        None if validated is None else MAPPED_POLICY,
        kernel_contract,
        True,
    )


def _record_completion(
    state,
    transformer_options,
    *,
    block_index: int,
    route: str,
    fields: tuple,
) -> None:
    active = _active_forward()
    if active[1] is not state:
        raise RuntimeError("partitioned Sol request ownership changed during an H3 block")
    routes = active[4]
    routes.append((block_index, route))
    owned = getattr(state, "partitioned_validated_receipts", None)
    if owned is None:
        owned = set()
        state.partitioned_validated_receipts = owned
    owned.add((block_index, fields))
    from .interop import receipt

    receipt(transformer_options, block_index, route, fields=fields)
    state.partitioned_receipts = getattr(state, "partitioned_receipts", 0) + 1


def partitioned_request_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    transformer_options: dict,
    block_index: int,
    kind: str,
    scale: float,
    sink_rows: int,
    prefix_k_range: tuple[int, int] | None,
    prefix_log_key_measure: float,
    semantic_digest: str,
    query_position_map=None,
    force_dense: bool = False,
) -> torch.Tensor:
    """Run one request-owned partitioned attention subcall over a single K/V union."""
    _validate_thd(q, k, v)
    if not isinstance(transformer_options, dict):
        raise RuntimeError("partitioned Sol requires transformer options")
    if kind not in {"global", "local", "anchor", "full"}:
        raise RuntimeError("partitioned Sol attention kind is unsupported")
    if type(block_index) is not int or block_index < 0:
        raise RuntimeError("partitioned Sol block index is invalid")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0 or scale != q.shape[-1] ** -0.5:
        raise RuntimeError("partitioned Sol requires the native H3 attention scale")
    if type(sink_rows) is not int or not 0 <= sink_rows <= k.shape[0]:
        raise RuntimeError("partitioned Sol sink-row count is invalid")
    if not _digest(semantic_digest):
        raise RuntimeError("partitioned Sol semantic digest is invalid")

    state = _request_state()
    config = state.config
    evaluation = _forward_evaluation()
    stage_runtime = transformer_options.get("h3_flow_partitioned_stage_v1")
    validation_context = {
        "flow_request_id": transformer_options.get("h3_flow_request_id_v1"),
        "flow_stage": transformer_options.get("h3_flow_stage"),
        "flow_stage_id": transformer_options.get("h3_flow_stage_id_v1"),
        "flow_evaluation_id": transformer_options.get("h3_flow_evaluation_id_v1"),
        "sol_evaluation": int(evaluation),
        "block_index": int(block_index),
        "owner_generation": getattr(stage_runtime, "owner_generation", None),
        "route": f"partitioned_{kind}",
    }
    from .interop import dense_evaluation_warmup

    warmup = dense_evaluation_warmup(config, evaluation, transformer_options)
    diagnostic_force_dense_suffix = _diagnostic_force_dense_suffix(
        kind=kind,
        force_dense=bool(force_dense),
        warmup=bool(warmup),
    )
    dense_execution = bool(force_dense or warmup or diagnostic_force_dense_suffix)
    key_bias = _key_bias(
        state,
        kv_rows=int(k.shape[0]),
        prefix_range=prefix_k_range,
        log_measure=float(prefix_log_key_measure),
        device=q.device,
        semantic_digest=semantic_digest,
    )
    exact_end = sink_rows
    if prefix_k_range is not None:
        if prefix_k_range[0] < sink_rows:
            raise RuntimeError("partitioned Sol target-prefix K/V range overlaps the global sink")
        if not dense_execution and prefix_k_range[0] != sink_rows:
            raise RuntimeError(
                "partitioned Sol sparse key measure must begin exactly at the global sink boundary"
            )
        exact_end = max(exact_end, prefix_k_range[1])

    validated, descriptor, mapped = _descriptor_for_wire(
        state,
        query_position_map,
        q_rows=int(q.shape[0]),
        kv_rows=int(k.shape[0]),
        sink_rows=sink_rows,
        device=q.device,
        materialize=not dense_execution,
    )
    state.eligible_calls += 1

    if dense_execution:
        result = _weighted_dense(q, k, v, key_bias, scale=scale)
        state.partitioned_dense_calls = getattr(state, "partitioned_dense_calls", 0) + 1
        if diagnostic_force_dense_suffix:
            state.partitioned_diagnostic_dense_suffix_calls += 1
            mode = "dense_diagnostic_suffix"
        elif force_dense:
            mode = "dense_forced"
        else:
            state.dense_calls += 1
            mode = "dense_warmup"
        fields = _completion_fields(
            evaluation=evaluation,
            semantic_digest=semantic_digest,
            kind=kind,
            execution_mode=mode,
            q_rows=int(q.shape[0]),
            kv_rows=int(k.shape[0]),
            sink_rows=sink_rows,
            prefix_k_range=prefix_k_range,
            prefix_log_key_measure=float(prefix_log_key_measure),
            validated=validated,
            descriptor=descriptor,
            kernel_contract=None,
        )
        _record_completion(
            state,
            transformer_options,
            block_index=block_index,
            route=PARTITIONED_DENSE_ROUTE,
            fields=fields,
        )
        return result

    calibration_identity = _sha256_json(
        {
            "abi": PARTITIONED_REQUEST_ABI,
            "semantic_digest": semantic_digest,
            "kind": kind,
            "q_rows": int(q.shape[0]),
            "kv_rows": int(k.shape[0]),
            "sink_rows": sink_rows,
            "exact_end": exact_end,
            "mapped": mapped is not None,
            "map_digest": None if validated is None else validated.map_digest,
            "descriptor_digest": None if descriptor is None else descriptor.descriptor_digest,
        }
    )
    from ._vendor.sol_attn import interface as sol_interface
    runtime_changed = state.runtime_lease.bind_partitioned(sol_interface, q.device)
    if runtime_changed:
        state.validation_state.invalidate("new_runtime")
        if hasattr(state, "partitioned_sparse_verified"):
            state.partitioned_sparse_verified.clear()
    from .validation import build_arithmetic_key
    mode = (
        "partitioned_mapped_weighted_v1" if mapped is not None and key_bias is not None
        else "partitioned_mapped_v1" if mapped is not None
        else "partitioned_weighted_v1" if key_bias is not None
        else "partitioned_unweighted_v1"
    )
    arithmetic_key = build_arithmetic_key(
        state=state,
        q=q,
        k=k,
        v=v,
        mode=mode,
        layout="thd",
        scale=scale,
        mapped_abi=MAPPED_POLICY if mapped is not None else None,
        bias_range=None if prefix_k_range is None else list(prefix_k_range),
        bias_log_measure=None if prefix_k_range is None else float(prefix_log_key_measure),
        bias_identity=semantic_digest if key_bias is not None else None,
        # Conservative by design: current partitioned proof identity still
        # retains map/group ownership until CUDA replay proves quotient safety.
        physical_identity=calibration_identity,
    )
    _replay_partitioned_suffix(
        q,
        k,
        v,
        key_bias=key_bias,
        mapped_neighbor_intervals=mapped,
        exact_end=exact_end,
        config=config,
        state=state,
        arithmetic_key=arithmetic_key,
        validation_context=validation_context,
        kind=kind,
    )
    ticket = state.validation_state.begin(arithmetic_key)
    diagnostics = state.cuda_diagnostics
    diagnostic_context = {
        "mode": mode,
        "arithmetic_key_digest": ticket.digest or None,
        **validation_context,
    }
    if ticket.validate:
        from .sparse import _accumulate_attribution
        gate_sample = diagnostics.begin_sample(
            "partitioned_arithmetic_gate",
            q.device,
            context=diagnostic_context,
            important=True,
        )
        diagnostics.initial_stream_drain(gate_sample)
        gate_started = time.perf_counter()
        gate_telemetry = {}
        try:
            all_selected_started = time.perf_counter()
            with diagnostics.span(gate_sample, "all_selected_call"):
                got = _sm120_union(
                    q,
                    k,
                    v,
                    tau=config.tau,
                    scale=scale,
                    sink_rows=int(k.shape[0]),
                    key_bias=key_bias,
                    mapped_neighbor_intervals=mapped,
                    telemetry=gate_telemetry,
                    diagnostic_state=diagnostics,
                    diagnostic_sample=gate_sample,
                )
            gate_telemetry["all_selected_host_wall_s"] = (
                time.perf_counter() - all_selected_started
            )
            reference_started = time.perf_counter()
            with diagnostics.span(gate_sample, "dense_reference"):
                want = _weighted_dense(q, k, v, key_bias, scale=scale)
            gate_telemetry["reference_host_wall_s"] = (
                time.perf_counter() - reference_started
            )
            reduction_started = time.perf_counter()
            with diagnostics.span(gate_sample, "error_reduction"):
                gate = error_metrics(
                    got.transpose(0, 1).unsqueeze(0),
                    want.transpose(0, 1).unsqueeze(0),
                )
            gate_telemetry["reduction_host_wall_s"] = (
                time.perf_counter() - reduction_started
            )
            gate_wall_s = time.perf_counter() - gate_started
            _accumulate_attribution(
                state, "partitioned_arithmetic_gate", gate_telemetry, gate_wall_s
            )
            if not arithmetic_gate_passes(gate):
                raise RuntimeError(
                    f"partitioned Sol all-selected arithmetic gate failed: {gate}"
                )
        except BaseException:
            state.validation_state.publish_failure(ticket)
            raise
        state.validation_state.record_gate(
            ticket, mode=mode, gate_wall_s=gate_wall_s,
            metrics=gate, telemetry=gate_telemetry,
            context=validation_context,
        )
        state.validation_state.publish_success(ticket, gate)
        if len(state.gates) < 32:
            state.gates.append(
                {
                    "route": PARTITIONED_REQUEST_ABI,
                    "kind": kind,
                    "shape": [1, q.shape[1], q.shape[0], q.shape[2]],
                    "kv_shape": [1, k.shape[1], k.shape[0], k.shape[2]],
                    "mapped_neighbor_abi": mapped is not None,
                    "key_measure_bias": key_bias is not None,
                    "gate_wall_s": gate_wall_s,
                    "attribution": dict(gate_telemetry),
                    "arithmetic_key_digest": ticket.digest or None,
                    **gate,
                }
            )

    from .sparse import _accumulate_attribution
    production_telemetry = {}
    production_sample = diagnostics.begin_sample(
        "partitioned_production_sparse", q.device, context=diagnostic_context
    )
    production_started = time.perf_counter()
    with diagnostics.span(production_sample, "production_call"):
        result = _sm120_union(
            q,
            k,
            v,
            tau=config.tau,
            scale=scale,
            sink_rows=exact_end,
            key_bias=key_bias,
            mapped_neighbor_intervals=mapped,
            telemetry=production_telemetry,
            diagnostic_state=diagnostics,
            diagnostic_sample=production_sample,
        )
    production_host_wall_s = time.perf_counter() - production_started
    _accumulate_attribution(
        state, "partitioned_production_sparse", production_telemetry,
        production_host_wall_s,
    )
    state.validation_state.record_production(
        production_host_wall_s, production_telemetry
    )
    state.partitioned_sparse_calls = getattr(state, "partitioned_sparse_calls", 0) + 1
    state.partitioned_requested_q_rows = getattr(state, "partitioned_requested_q_rows", 0) + int(q.shape[0])
    state.partitioned_kernel_q_rows = getattr(state, "partitioned_kernel_q_rows", 0) + int(q.shape[0])
    state.sparse_calls += 1
    if kind == "local":
        state.vdn_local_sol_calls += 1
        state.vdn_requested_q_rows += int(q.shape[0])
        state.vdn_kernel_q_rows += int(q.shape[0])
        if q.shape[0] != k.shape[0]:
            state.vdn_rectangular_sol_calls += 1
        if mapped is not None:
            state.vdn_mapped_sol_calls += 1

    from .provenance import CONTRACT

    route = PARTITIONED_MAPPED_ROUTE if mapped is not None else PARTITIONED_SOL_ROUTE
    mode = "sm120_mapped" if mapped is not None else "sm120_union"
    fields = _completion_fields(
        evaluation=evaluation,
        semantic_digest=semantic_digest,
        kind=kind,
        execution_mode=mode,
        q_rows=int(q.shape[0]),
        kv_rows=int(k.shape[0]),
        sink_rows=sink_rows,
        prefix_k_range=prefix_k_range,
        prefix_log_key_measure=float(prefix_log_key_measure),
        validated=validated,
        descriptor=descriptor,
        kernel_contract=CONTRACT,
    )
    _record_completion(
        state,
        transformer_options,
        block_index=block_index,
        route=route,
        fields=fields,
    )
    return result


__all__ = [
    "PARTITIONED_DENSE_ROUTE",
    "PARTITIONED_MAPPED_ROUTE",
    "PARTITIONED_RECEIPT_TAG",
    "PARTITIONED_REQUEST_ABI",
    "PARTITIONED_SOL_ROUTE",
    "partitioned_request_attention",
]
