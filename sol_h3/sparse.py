"""Native attention bridge to the packaged Sana Sol-H3 implementation."""
import math
import sys
import time

import torch
import torch.nn.functional as F


BLOCK_SIZE = 64
MAPPED_ABI_VERSION = "sm120-mapped-neighbor-runtime-v4"
ARITH_MEAN_ABS_LIMIT = 0.002
ARITH_REL_L2_LIMIT = 0.005
ARITH_CATASTROPHIC_MAX_FLOOR = 0.5
ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER = 4.0


class KernelUnavailable(RuntimeError):
    """The native provider can execute safely without this optional sparse kernel."""


def _sink_blocks(start, tokens, rows):
    if type(start) is not int or type(tokens) is not int or type(rows) is not int:
        raise RuntimeError("SOL sink geometry must use integer row counts")
    if start < 0 or tokens < 0 or rows < 0 or start > rows or start + tokens > rows:
        raise RuntimeError("SOL sink row range is outside the current sequence")
    if tokens == 0:
        block = start // BLOCK_SIZE
        return [block, block]
    return [start // BLOCK_SIZE, (start + tokens + BLOCK_SIZE - 1) // BLOCK_SIZE]


def load_kernel(device):
    """Load the verified node-local public API; SM120 requires its CuTe backend."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120 only")
    try:
        from ._vendor.sol_attn import get_sol_attn_backend, interface, sol_attn
        from .compiler_attribution import attribution_scope, install_hooks
        backend = get_sol_attn_backend(device)
        if backend != "cute_sm120":
            if sys.platform == "win32":
                raise RuntimeError(
                    "native Windows cannot execute the required NVIDIA CUTLASS CuTe DSL backend; "
                    "SM120 SOL attention currently requires Linux/WSL2"
                )
            raise RuntimeError(
                f"Sana selected {backend}; SM120 requires CuTe (cutlass.cute and cuda.bindings.driver)"
            )
        from ._vendor.sol_attn import preprocess  # noqa: F401
        from ._vendor.sol_attn.sm120 import make_kernel  # noqa: F401
        compiler_namespace = install_hooks(interface)
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Sana Sol-Attn initialization failed: {exc}") from exc

    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0, key_bias=None,
               mapped_neighbor_intervals=None, _telemetry=None):
        _sink_blocks(sink_start, sink_tokens, k.shape[1])
        with attribution_scope(_telemetry):
            return sol_attn(
                q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
                thresh_type=thresh_type, kv_splits=kv_splits,
                sink_start=sink_start, sink_tokens=sink_tokens,
                key_bias=key_bias,
                mapped_neighbor_intervals=mapped_neighbor_intervals,
            )

    kernel.backend_name = backend
    kernel.source_tree_verified = True
    kernel.block_size = BLOCK_SIZE
    kernel.supports_attribution = True
    kernel.compiler_namespace = compiler_namespace
    return kernel


def error_metrics(got, want):
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    abs_delta = delta.abs()
    finite = bool(torch.isfinite(got_f).all().item() and
                  torch.isfinite(want_f).all().item() and
                  torch.isfinite(delta).all().item())
    if not finite:
        return {
            "finite": False,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "rel_l2": math.inf,
            "reference_peak_abs": math.nan,
            "catastrophic_max_abs_limit": math.nan,
        }

    reference_peak_abs = float(want_f.abs().max().item())
    max_abs = float(abs_delta.max().item())
    mean_abs = float(abs_delta.mean().item())
    rel_l2 = float((torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(want_f).clamp_min(1e-12)).item())
    catastrophic_limit = max(
        ARITH_CATASTROPHIC_MAX_FLOOR,
        ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER * reference_peak_abs,
    )
    return {
        "finite": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "reference_peak_abs": reference_peak_abs,
        "catastrophic_max_abs_limit": catastrophic_limit,
    }


def arithmetic_gate_passes(metrics):
    return bool(
        metrics.get("finite")
        and metrics["mean_abs"] <= ARITH_MEAN_ABS_LIMIT
        and metrics["rel_l2"] <= ARITH_REL_L2_LIMIT
        and metrics["max_abs"] <= metrics["catastrophic_max_abs_limit"]
    )


def _dense_reference(q, k, v, dense_attention, key_bias=None):
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    bias = None if key_bias is None else key_bias.to(dtype=q.dtype).view(1, 1, 1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2)


def _bthd_layout_key(*tensors):
    return tuple((tuple(x.shape), tuple(x.stride())) for x in tensors)


def _validate_mapped_descriptor(mapped, q, k):
    if mapped is None:
        return False
    q_tiles = (q.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE
    if (
        not torch.is_tensor(mapped)
        or mapped.ndim != 2
        or tuple(mapped.shape) != (q_tiles, 2)
        or mapped.dtype != torch.int32
        or mapped.device != q.device
        or not mapped.is_contiguous()
    ):
        raise RuntimeError(
            "mapped SOL requires contiguous int32 [ceil(Tq/64), 2] metadata on the QKV device"
        )
    return True



def _accumulate_attribution(state, phase, telemetry, host_wall_s=None):
    """Aggregate host-side attribution without synchronizing the CUDA device."""
    if telemetry is None:
        telemetry = {}
    target = state.runtime_attribution
    target[phase + "_calls"] = int(target.get(phase + "_calls", 0)) + 1
    if host_wall_s is not None:
        target[phase + "_host_wall_s"] = float(target.get(phase + "_host_wall_s", 0.0)) + float(host_wall_s)
    for name in (
        "prepare_jit_host_wall_s",
        "compile_lock_wait_s",
        "compile_body_s",
        "dispatch_host_enqueue_s",
        "source_verify_s",
    ):
        value = telemetry.get(name)
        if value is not None:
            target[phase + "_" + name] = float(target.get(phase + "_" + name, 0.0)) + float(value)
    for name in ("compile_hit", "compile_miss", "compile_race_hit"):
        if telemetry.get(name):
            target[phase + "_" + name] = int(target.get(phase + "_" + name, 0)) + 1


def _kernel_call(kernel, q, k, v, *, telemetry, **kwargs):
    if getattr(kernel, "supports_attribution", False):
        kwargs["_telemetry"] = telemetry
    return kernel(q, k, v, **kwargs)


def attention(q, k, v, prefix, config, state, dense_attention=None,
              recompute_prefix_queries=True, key_bias=None, exact_k_blocks=None,
              calibration_identity=None, mapped_neighbor_intervals=None,
              mapped_calibration_identity=None, validation_bias_identity=None):
    if (any(x.ndim != 4 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
            or q.shape[0] != 1 or q.shape[-1] != 128
            or q.shape[2] == 0 or k.shape[2] == 0):
        raise RuntimeError("SOL requires Q [1, heads, Tq, 128], KV [1, heads, Tkv, 128]")
    _sink_blocks(0, prefix, k.shape[2])
    if recompute_prefix_queries and prefix > q.shape[2]:
        raise RuntimeError("Prefix query recomputation exceeds the available Q rows")
    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):
        raise RuntimeError("SOL requires BF16 QKV on the same device")

    mapped_enabled = _validate_mapped_descriptor(mapped_neighbor_intervals, q, k)
    if mapped_enabled:
        if key_bias is not None or exact_k_blocks is not None or calibration_identity is not None:
            raise RuntimeError("weighted/exact-range SOL cannot be combined with mapped-neighbor metadata")
        if mapped_calibration_identity is not None and not isinstance(mapped_calibration_identity, str):
            raise RuntimeError("mapped SOL calibration identity must be a string when supplied")
    elif mapped_calibration_identity is not None:
        raise RuntimeError("mapped SOL calibration identity was supplied without metadata")

    if key_bias is not None:
        if (not torch.is_tensor(key_bias) or key_bias.ndim != 1
                or key_bias.shape[0] != k.shape[2] or key_bias.device != q.device
                or key_bias.dtype != torch.float32 or not key_bias.is_contiguous()):
            raise RuntimeError("weighted SOL requires contiguous FP32 key_bias with one value per K row")
        if not isinstance(calibration_identity, str) or not calibration_identity:
            raise RuntimeError("weighted SOL requires a semantic calibration identity")
        if (not isinstance(exact_k_blocks, (tuple, list)) or len(exact_k_blocks) != 2
                or any(type(x) is not int for x in exact_k_blocks)):
            raise RuntimeError("weighted SOL requires one exact K block interval")
        max_blocks = (k.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE
        if exact_k_blocks[0] < 0 or exact_k_blocks[1] < exact_k_blocks[0] or exact_k_blocks[1] > max_blocks:
            raise RuntimeError("weighted SOL exact K block interval is out of range")
    elif not mapped_enabled and (
        exact_k_blocks is not None
        or calibration_identity is not None
        or validation_bias_identity is not None
    ):
        raise RuntimeError("weighted SOL routing metadata was supplied without key_bias")

    state.runtime_lease.ensure_source_verified(q.device)
    kernel_loader_s = 0.0
    if state.kernel is None:
        failure = getattr(state, "kernel_failure", None)
        if failure:
            raise KernelUnavailable(failure)
        started = time.perf_counter()
        try:
            state.kernel = load_kernel(q.device)
        except RuntimeError as exc:
            state.kernel_failure = str(exc)
            raise KernelUnavailable(str(exc)) from exc
        kernel_loader_s = time.perf_counter() - started
        state.kernel_device = q.device
    elif q.device != state.kernel_device:
        raise RuntimeError("SOL compute device changed within a sampling request")
    state.runtime_lease.bind_kernel(state.kernel, q.device)

    qb, kb, vb = (x.transpose(1, 2) for x in (q, k, v))
    if any(x.stride(-1) != 1 for x in (qb, kb, vb)):
        raise RuntimeError("SOL BTHD bridge requires a contiguous head dimension")

    # Descriptor values are intentionally absent: every map with this layout reuses
    # one compiled mapped ABI. The runtime tensor remains a kernel argument.
    from .validation import build_arithmetic_key
    mode = (
        "ordinary_weighted_v1" if key_bias is not None
        else "ordinary_mapped_v1" if mapped_enabled
        else "ordinary_unweighted_v1"
    )
    bias_range = None
    bias_log_measure = None
    if key_bias is not None and exact_k_blocks is not None:
        bias_range = [int(exact_k_blocks[0]) * BLOCK_SIZE, int(exact_k_blocks[1]) * BLOCK_SIZE]
        # The semantic identity remains authoritative; reading device bias values
        # merely to build a key would introduce a synchronization.
    arithmetic_key = build_arithmetic_key(
        state=state,
        q=q,
        k=k,
        v=v,
        mode=mode,
        layout="bhtd",
        scale=q.shape[-1] ** -0.5,
        mapped_abi=MAPPED_ABI_VERSION if mapped_enabled else None,
        bias_range=bias_range,
        bias_log_measure=bias_log_measure,
        bias_identity=(
            validation_bias_identity
            if validation_bias_identity is not None
            else calibration_identity
        ),
        physical_identity=None,
    )
    ticket = state.validation_state.begin(arithmetic_key)
    calibrating = ticket.validate
    gate_started = time.perf_counter() if calibrating else None
    if calibrating:
        gate_telemetry = {}
        try:
            got = _kernel_call(
                state.kernel, qb, kb, vb, telemetry=gate_telemetry,
                tau=config.tau, thresh_type="diag", kv_splits=1,
                sink_start=0, sink_tokens=kb.shape[1], key_bias=key_bias,
                mapped_neighbor_intervals=mapped_neighbor_intervals,
            )
            want = _dense_reference(q, k, v, None, key_bias=key_bias)
            metrics = error_metrics(got, want)
            gate_wall_s = time.perf_counter() - gate_started
            _accumulate_attribution(state, "arithmetic_gate", gate_telemetry, gate_wall_s)
            if not arithmetic_gate_passes(metrics):
                raise RuntimeError(f"SOL all-selected arithmetic gate failed: {metrics}")
        except BaseException:
            state.validation_state.publish_failure(ticket)
            raise
        state.validation_state.record_gate(
            ticket, mode=mode, gate_wall_s=gate_wall_s,
            metrics=metrics, telemetry=gate_telemetry,
        )
        state.validation_state.publish_success(ticket, metrics)
        # Compatibility telemetry only. Routing decisions use validation_state.
        state.sparse_verified.add(ticket.digest)
        if len(state.gates) < 32:
            state.gates.append({
                "shape": list(q.shape),
                "kv_shape": list(k.shape),
                "backend": getattr(state.kernel, "backend_name", "test_substitute"),
                "mapped_neighbor_abi": mapped_enabled,
                "mapped_abi_version": MAPPED_ABI_VERSION,
                "kernel_loader_s": kernel_loader_s,
                "gate_wall_s": gate_wall_s,
                "materialized_qkv_bytes": 0,
                "bthd_qkv_bytes": sum(x.numel() * x.element_size() for x in (qb, kb, vb)),
                "bthd_strides": [list(x.stride()) for x in (qb, kb, vb)],
                "attribution": dict(gate_telemetry),
                "arithmetic_key_digest": ticket.digest or None,
                **metrics,
            })
        del got, want

    if exact_k_blocks is None:
        sink_start, sink_tokens = 0, prefix
    else:
        sink_start = min(k.shape[2], exact_k_blocks[0] * BLOCK_SIZE)
        sink_end = min(k.shape[2], exact_k_blocks[1] * BLOCK_SIZE)
        sink_tokens = max(0, sink_end - sink_start)
    production_telemetry = {}
    production_started = time.perf_counter()
    out = _kernel_call(
        state.kernel, qb, kb, vb, telemetry=production_telemetry,
        tau=config.tau, thresh_type="diag", kv_splits=1,
        sink_start=sink_start, sink_tokens=sink_tokens, key_bias=key_bias,
        mapped_neighbor_intervals=mapped_neighbor_intervals,
    )
    production_host_wall_s = time.perf_counter() - production_started
    _accumulate_attribution(
        state, "production_sparse", production_telemetry,
        production_host_wall_s,
    )
    state.validation_state.record_production(production_host_wall_s)
    if prefix and recompute_prefix_queries:
        out[:, :prefix] = _dense_reference(
            q[:, :, :prefix], k, v, dense_attention, key_bias=key_bias
        )
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
