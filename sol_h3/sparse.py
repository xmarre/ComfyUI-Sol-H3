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
    from .provenance import verify_source
    cap = torch.cuda.get_device_capability(device) if device.type == "cuda" else None
    if cap not in ((12, 0), (12, 1)):
        raise RuntimeError("This experimental SOL integration currently targets single-GPU SM120/SM121 only")
    try:
        verify_source()
        from ._vendor.sol_attn import get_sol_attn_backend, sol_attn
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
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Sana Sol-Attn initialization failed: {exc}") from exc

    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0, key_bias=None,
               mapped_neighbor_intervals=None, query_start=0, prepared_kv=None):
        _sink_blocks(sink_start, sink_tokens, k.shape[1])
        return sol_attn(
            q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
            thresh_type=thresh_type, kv_splits=kv_splits,
            sink_start=sink_start, sink_tokens=sink_tokens,
            key_bias=key_bias,
            mapped_neighbor_intervals=mapped_neighbor_intervals,
            query_start=query_start, prepared_kv=prepared_kv,
        )

    from ._vendor.sol_attn.interface import PreparedKV
    kernel.prepare_kv = PreparedKV
    kernel.backend_name = backend
    kernel.source_tree_verified = True
    kernel.block_size = BLOCK_SIZE
    return kernel


METRIC_CHUNK_ELEMENTS = 262144
GATE_QUERY_CHUNK_SIZE = 4096


def _metric_tiles(tensor):
    """Slice before flattening so a strided output never creates a full copy."""
    if tensor.numel() <= METRIC_CHUNK_ELEMENTS:
        yield tensor
        return
    row_elements = math.prod(tensor.shape[1:])
    if row_elements > METRIC_CHUNK_ELEMENTS:
        for row in tensor.unbind(0):
            yield from _metric_tiles(row)
    else:
        rows = max(1, METRIC_CHUNK_ELEMENTS // row_elements)
        for start in range(0, tensor.shape[0], rows):
            yield tensor[start:start + rows]


class ErrorAccumulator:
    """Whole-output metrics with bounded temporary storage and one host read."""

    def __init__(self, device, *, enable_fused=True):
        self.stats = torch.zeros(6, dtype=torch.float64, device=device)
        self.stats[0] = 1
        self.elements = 0
        self._enable_fused = bool(enable_fused)
        self._fused_partials = []

    def add(self, got, want):
        if got.shape != want.shape or got.device != want.device or not got.numel():
            raise ValueError("Arithmetic metrics require equal nonempty shapes on the same device")
        stats = self.stats
        self.elements += got.numel()
        if self._enable_fused and got.device.type == "cuda":
            from .gate_metrics_triton import gate_partials
            partial = gate_partials(got, want)
            if partial is not None:
                # The entire chunk remains validated, but its statistics can
                # be aggregated with other chunks in one final device read.
                self._fused_partials.append(partial)
                return
        for g, w in zip(_metric_tiles(got), _metric_tiles(want)):
            gf, wf = g.float(), w.float()
            delta = gf - wf
            absolute = delta.abs()
            stats[0].mul_(torch.isfinite(gf).all() & torch.isfinite(wf).all() & torch.isfinite(delta).all())
            stats[1] = torch.maximum(stats[1], wf.abs().max())
            stats[2] = torch.maximum(stats[2], absolute.max())
            stats[3].add_(absolute.sum(dtype=torch.float64))
            stats[4].add_(delta.double().square().sum())
            stats[5].add_(wf.double().square().sum())
            del gf, wf, delta, absolute

    def statistics(self):
        stats = self.stats
        if not self.elements:
            raise ValueError("No arithmetic samples were accumulated")
        if self._fused_partials:
            partial = torch.cat(self._fused_partials, dim=1)
            stats[0].mul_(partial[0].amin())
            stats[1] = torch.maximum(stats[1], partial[1].amax())
            stats[2] = torch.maximum(stats[2], partial[2].amax())
            stats[3].add_(partial[3].sum(dtype=torch.float64))
            stats[4].add_(partial[4].sum(dtype=torch.float64))
            stats[5].add_(partial[5].sum(dtype=torch.float64))
            self._fused_partials.clear()
        return torch.stack((
            stats[0], stats[1], stats[2], stats[3] / self.elements,
            stats[4].sqrt() / stats[5].sqrt().clamp_min(1e-12),
        )).tolist()


def error_metrics(got, want):
    if got.shape != want.shape or got.device != want.device or not got.numel():
        raise ValueError("Arithmetic metrics require equal nonempty shapes on the same device")
    if got.numel() > METRIC_CHUNK_ELEMENTS:
        # Preserve exhaustive all-Q/all-K arithmetic qualification. On SM120,
        # thousands of small PyTorch reductions used to dominate gate overhead;
        # the optional fused implementation computes the SAME full-domain
        # statistics with bounded partial storage and one host read. CPU and
        # unsupported layouts retain the reference reduction unchanged.
        if got.device.type == "cuda":
            from .gate_metrics_triton import gate_statistics
            statistics = gate_statistics(got, want)
            if statistics is not None:
                return _metrics_from_statistics(statistics)
        accumulator = ErrorAccumulator(got.device)
        accumulator.add(got, want)
        return _metrics_from_statistics(accumulator.statistics())
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    abs_delta = delta.abs()
    # Queue the same reductions before one small host transfer. Reading each
    # scalar separately drains the CUDA stream repeatedly during qualification.
    statistics = torch.stack((
        torch.isfinite(got_f).all() & torch.isfinite(want_f).all() & torch.isfinite(delta).all(),
        want_f.abs().max(),
        abs_delta.max(),
        abs_delta.mean(),
        torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(want_f).clamp_min(1e-12),
    )).tolist()
    return _metrics_from_statistics(statistics)


def _metrics_from_statistics(statistics):
    finite = bool(statistics[0])
    if not finite:
        return {
            "finite": False,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "rel_l2": math.inf,
            "reference_peak_abs": math.nan,
            "catastrophic_max_abs_limit": math.nan,
        }

    _, reference_peak_abs, max_abs, mean_abs, rel_l2 = statistics
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
    try:
        from comfy.ops import scaled_dot_product_attention as dense_sdpa
        dispatcher = "core"
    except ImportError:
        dense_sdpa = F.scaled_dot_product_attention
        dispatcher = "torch"
    # Keep an independent all-selected reference while using native VDN's
    # backend policy. Execution failures must propagate without a second try.
    result = dense_sdpa(q, k, v, attn_mask=bias).transpose(1, 2)
    from .runtime import _REQUEST
    state = _REQUEST.get()
    if state is not None:
        counter = f"native_{dispatcher}_reference_calls"
        setattr(state, counter, getattr(state, counter, 0) + 1)
    return result


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


def attention(q, k, v, prefix, config, state, dense_attention=None,
              recompute_prefix_queries=True, key_bias=None, exact_k_blocks=None,
              calibration_identity=None, mapped_neighbor_intervals=None,
              mapped_calibration_identity=None, query_start=0, kv_workspace=None):
    if (any(x.ndim != 4 for x in (q, k, v)) or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]
            or q.shape[0] != 1 or q.shape[-1] != 128
            or q.shape[2] == 0 or k.shape[2] == 0):
        raise RuntimeError("SOL requires Q [1, heads, Tq, 128], KV [1, heads, Tkv, 128]")
    _sink_blocks(0, prefix, k.shape[2])
    if type(query_start) is not int or query_start < 0 or query_start % BLOCK_SIZE:
        raise RuntimeError("SOL query_start must be a nonnegative multiple of 64")
    if recompute_prefix_queries and not query_start and kv_workspace is None and prefix > q.shape[2]:
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
    elif not mapped_enabled and (exact_k_blocks is not None or calibration_identity is not None):
        raise RuntimeError("weighted SOL routing metadata was supplied without key_bias")

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

    qb, kb, vb = (x.transpose(1, 2) for x in (q, k, v))
    if any(x.stride(-1) != 1 for x in (qb, kb, vb)):
        raise RuntimeError("SOL BTHD bridge requires a contiguous head dimension")
    streamed_kwargs = {}
    from .grouped_kv import current as current_grouped_kv
    grouped_prepared = current_grouped_kv(kb, vb)
    if grouped_prepared is not None:
        streamed_kwargs["prepared_kv"] = grouped_prepared
    if kv_workspace is not None:
        kv_workspace.check(q, k, v)
        if kv_workspace.prepared_kv is None:
            kv_workspace.prepared_kv = state.kernel.prepare_kv(kb, vb)
        streamed_kwargs["prepared_kv"] = kv_workspace.prepared_kv
    if query_start:
        streamed_kwargs["query_start"] = query_start

    layout_key = _bthd_layout_key(qb, kb, vb)
    # Descriptor values are intentionally absent: every map with this layout reuses
    # one compiled mapped ABI. The runtime tensor remains a kernel argument.
    mapped_identity = (MAPPED_ABI_VERSION, True) if mapped_enabled else (MAPPED_ABI_VERSION, False)
    # Qualify every query origin once per request, even when all streamed
    # chunks share one layout. Reusing a compile shape must not sample only
    # the first chunk of the original full query sequence.
    key = (q.device, q.dtype, layout_key, calibration_identity, mapped_identity, query_start)
    calibrating = key not in state.sparse_verified
    gate_started = time.perf_counter() if calibrating else None
    if calibrating:
        from .memory_profile import mark, storage_bytes
        mark("arithmetic_gate_enter", q_rows=q.shape[2], kv_rows=k.shape[2])
        # Qualification still covers EVERY Q row against EVERY original K/V.
        # Bound both candidate/reference outputs; retain no all-selected full
        # output beside the inputs. A weighted SDPA fallback can allocate scores,
        # so restrict its query side to one M64 tile independently of producer size.
        gate_rows = GATE_QUERY_CHUNK_SIZE if key_bias is None else BLOCK_SIZE
        if (q.shape[2] > gate_rows and "prepared_kv" not in streamed_kwargs
                and callable(getattr(state.kernel, "prepare_kv", None))):
            streamed_kwargs["prepared_kv"] = state.kernel.prepare_kv(kb, vb)
        accumulator = ErrorAccumulator(q.device) if q.shape[2] > gate_rows else None
        for start in range(0, q.shape[2], gate_rows):
            stop = min(start + gate_rows, q.shape[2])
            gate_kw = dict(streamed_kwargs)
            if query_start + start:
                gate_kw["query_start"] = query_start + start
            mapping = None if mapped_neighbor_intervals is None else mapped_neighbor_intervals[start // 64:(stop + 63) // 64]
            got = state.kernel(
                qb[:, start:stop], kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                sink_start=0, sink_tokens=kb.shape[1], key_bias=key_bias,
                mapped_neighbor_intervals=mapping, **gate_kw,
            )
            want = _dense_reference(q[:, :, start:stop], k, v, None, key_bias=key_bias)
            if accumulator is None:
                metrics = error_metrics(got, want)
            else:
                accumulator.add(got, want)
            del got, want
        if accumulator is not None:
            metrics = _metrics_from_statistics(accumulator.statistics())
        mark("arithmetic_gate_statistics_done", q_rows=q.shape[2], kv_rows=k.shape[2])
        gate_wall_s = time.perf_counter() - gate_started
        if not arithmetic_gate_passes(metrics):
            raise RuntimeError(f"SOL all-selected arithmetic gate failed: {metrics}")
        state.sparse_verified.add(key)
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
            "qkv_storage_bytes": storage_bytes(qb, kb, vb),
            "bthd_strides": [list(x.stride()) for x in (qb, kb, vb)],
            "gate_query_chunk_size": gate_rows,
            "query_start": query_start,
            **metrics,
        })

    if exact_k_blocks is None:
        sink_start, sink_tokens = 0, prefix
    else:
        sink_start = min(k.shape[2], exact_k_blocks[0] * BLOCK_SIZE)
        sink_end = min(k.shape[2], exact_k_blocks[1] * BLOCK_SIZE)
        sink_tokens = max(0, sink_end - sink_start)
    out = state.kernel(
        qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
        sink_start=sink_start, sink_tokens=sink_tokens, key_bias=key_bias,
        mapped_neighbor_intervals=mapped_neighbor_intervals,
        **streamed_kwargs,
    )
    prefix_queries = min(q.shape[2], max(0, prefix - query_start))
    if prefix_queries and recompute_prefix_queries:
        out[:, :prefix_queries] = _dense_reference(
            q[:, :, :prefix_queries], k, v, dense_attention, key_bias=key_bias
        )
    state.sparse_calls += 1
    return out.reshape(1, q.shape[2], -1)
