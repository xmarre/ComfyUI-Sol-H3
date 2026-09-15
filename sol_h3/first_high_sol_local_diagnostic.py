"""Bounded first-high Sol-local diagnostic E.

This module is inert unless Flow publishes ``h3_first_high_sol_local_diagnostic_v1``.
E keeps VDN's restricted local support and complement unchanged, but makes the
returned Sol local call all-selected by changing only the final kernel sink range.
Same-input operator witnesses run outside the H3 model call and never emit backend
receipts or retain production counters.
"""
from __future__ import annotations

import contextvars
import hashlib
import importlib.metadata
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REQUEST_KEY = "h3_first_high_sol_local_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_sol_local_receipts_v1"
EVIDENCE_KEY = "h3_first_high_sol_local_evidence_v1"
MODE = "all_selected_e"
WITNESS_GROUPS = frozenset({0, 2, 10})
_REQUIRED_FIELDS = (
    "api",
    "capture_id",
    "mode",
    "stage",
    "logical_call_limit",
    "sigma",
    "target_shapes_digest",
    "source_contract_digest",
)
_HEX = frozenset("0123456789abcdef")
_INSTALLED = False
_ORIGINAL_SPARSE_ATTENTION = None
_ORIGINAL_RUNTIME_RECEIPT = None
_ORIGINAL_BLOCK_CALL = None
_DEBUG_COMPILED: dict[tuple[Any, ...], Any] = {}
_LSE_COMPILED: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True, slots=True)
class LocalGroup:
    block_index: int
    group_index: int
    q_rows: int
    kv_rows: int
    original_sink_rows: int
    scale: float
    witness_record: dict[str, Any] | None = None


_LOCAL_GROUP: contextvars.ContextVar[LocalGroup | None] = contextvars.ContextVar(
    "h3_first_high_sol_local_group", default=None
)


def parse_request(options: dict[str, Any] | None) -> dict[str, Any] | None:
    value = (options or {}).get(REQUEST_KEY)
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != len(_REQUIRED_FIELDS):
        raise RuntimeError("first-high Sol-local diagnostic request must be an immutable exact-field tuple")
    result: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str):
            raise RuntimeError("first-high Sol-local diagnostic entries must be (name, value) tuples")
        key, field_value = item
        if key in result:
            raise RuntimeError(f"duplicate first-high Sol-local diagnostic field: {key}")
        result[key] = field_value
    if tuple(result) != _REQUIRED_FIELDS or result["api"] != 1:
        raise RuntimeError("first-high Sol-local diagnostic request does not match API 1")
    if result["mode"] != MODE:
        raise RuntimeError(f"unsupported first-high Sol-local diagnostic mode: {result['mode']!r}")
    if result["stage"] != "high" or result["logical_call_limit"] != 1:
        raise RuntimeError("first-high Sol-local diagnostic is restricted to one high-stage logical call")
    if not isinstance(result["capture_id"], str) or not result["capture_id"]:
        raise RuntimeError("first-high Sol-local diagnostic capture_id is invalid")
    sigma = result["sigma"]
    if type(sigma) is not float or not math.isfinite(sigma) or not 0.0 < sigma <= 1.0:
        raise RuntimeError("first-high Sol-local diagnostic sigma must be a finite float in (0, 1]")
    for name in ("target_shapes_digest", "source_contract_digest"):
        digest = result[name]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in _HEX for ch in digest):
            raise RuntimeError(f"first-high Sol-local diagnostic {name} must be lowercase SHA-256")
    options = options or {}
    if options.get("h3_first_high_operator_diagnostic_v1") is not None:
        raise RuntimeError("first-high Sol-local diagnostic cannot coexist with W")
    if options.get("vdn_h3_external_sequence_v1") is not None:
        raise RuntimeError("first-high Sol-local diagnostic forbids external/reduced VDN sequence")
    if options.get("attention_measure_v1") is not None or options.get("h3_flow_mixed_grid_attention_measure_v1") is not None:
        raise RuntimeError("first-high Sol-local diagnostic forbids weighted/Mixed-Grid attention")
    if options.get("minimax_h3_untwist_rope") is not None:
        raise RuntimeError("first-high Sol-local diagnostic requires the no-Untwist R control")
    stage = options.get("h3_flow_stage")
    if stage not in {None, "high"}:
        raise RuntimeError("first-high Sol-local diagnostic reached Sol outside the high stage")
    return result


def enter_local_group(options: dict[str, Any], metadata: dict[str, Any]):
    request = parse_request(options)
    if request is None:
        raise RuntimeError("first-high Sol-local group context requires an active E request")
    required = ("block_index", "group_index", "q_rows", "kv_rows", "original_sink_rows", "scale")
    if any(name not in metadata for name in required):
        raise RuntimeError("first-high Sol-local group metadata is incomplete")
    group = LocalGroup(
        block_index=int(metadata["block_index"]),
        group_index=int(metadata["group_index"]),
        q_rows=int(metadata["q_rows"]),
        kv_rows=int(metadata["kv_rows"]),
        original_sink_rows=int(metadata["original_sink_rows"]),
        scale=float(metadata["scale"]),
        witness_record=metadata.get("witness_record"),
    )
    if not 0 <= group.block_index < 50 or not 0 <= group.group_index < 11:
        raise RuntimeError("first-high Sol-local group identity is out of range")
    if group.q_rows <= 0 or group.kv_rows <= 0 or not 0 <= group.original_sink_rows <= group.kv_rows:
        raise RuntimeError("first-high Sol-local group geometry is invalid")
    if not math.isfinite(group.scale) or group.scale <= 0.0:
        raise RuntimeError("first-high Sol-local group scale is invalid")
    if _LOCAL_GROUP.get() is not None:
        raise RuntimeError("nested first-high Sol-local group context is unsupported")
    return _LOCAL_GROUP.set(group)


def exit_local_group(token) -> None:
    _LOCAL_GROUP.reset(token)


def history_identity(options: dict[str, Any] | None):
    request = parse_request(options)
    if request is None:
        return None
    return (
        "h3_first_high_sol_local_diagnostic_v1",
        request["capture_id"],
        request["mode"],
        request["logical_call_limit"],
        request["sigma"],
        request["target_shapes_digest"],
        request["source_contract_digest"],
    )


def _tensor_sha256_cpu(value: torch.Tensor) -> str:
    work = value.detach().to(device="cpu").contiguous()
    raw = work.view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    from .sparse import error_metrics

    return error_metrics(got, want)


def _metric_accumulator(device: torch.device):
    return {
        "count": 0,
        "abs_sum": torch.zeros((), device=device, dtype=torch.float64),
        "abs_max": torch.zeros((), device=device, dtype=torch.float32),
        "delta_sq": torch.zeros((), device=device, dtype=torch.float64),
        "ref_sq": torch.zeros((), device=device, dtype=torch.float64),
        "finite": torch.ones((), device=device, dtype=torch.bool),
    }


def _accumulate_metric(acc, got: torch.Tensor, want: torch.Tensor) -> None:
    got_f = got.float()
    want_f = want.float()
    delta = got_f - want_f
    abs_delta = delta.abs()
    acc["count"] += int(delta.numel())
    acc["abs_sum"] = acc["abs_sum"] + abs_delta.double().sum()
    acc["abs_max"] = torch.maximum(acc["abs_max"], abs_delta.max())
    acc["delta_sq"] = acc["delta_sq"] + delta.double().square().sum()
    acc["ref_sq"] = acc["ref_sq"] + want_f.double().square().sum()
    acc["finite"] = acc["finite"] & torch.isfinite(got_f).all() & torch.isfinite(want_f).all()


def _finish_metric(acc) -> dict[str, Any]:
    count = max(int(acc["count"]), 1)
    ref_sq = float(acc["ref_sq"].item())
    return {
        "finite": bool(acc["finite"].item()),
        "max_abs": float(acc["abs_max"].item()),
        "mean_abs": float(acc["abs_sum"].item()) / count,
        "rel_l2": math.sqrt(float(acc["delta_sq"].item()) / max(ref_sq, 1.0e-24)),
        "count": int(acc["count"]),
    }


def _module_file(module_name: str) -> Path:
    suffix = f".{module_name}"
    candidates: dict[str, Path] = {}
    for loaded_name, module in tuple(sys.modules.items()):
        if module is None or (loaded_name != module_name and not loaded_name.endswith(suffix)):
            continue
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            continue
        try:
            path = Path(raw).resolve(strict=True)
        except OSError:
            continue
        candidates[str(path)] = path
    if len(candidates) != 1:
        raise RuntimeError(
            f"first-high Sol-local diagnostic requires one loaded source identity for {module_name}; "
            f"got {sorted(candidates)}"
        )
    return next(iter(candidates.values()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def loaded_sparse_provenance() -> dict[str, Any]:
    """Inventory the actual lazy-loaded packaged kernel/compiler bytes."""
    from . import provenance

    manifest = provenance.verify_source()
    critical = (
        "sol_h3._vendor.sol_attn.interface",
        "sol_h3._vendor.sol_attn.preprocess",
        "sol_h3._vendor.sol_attn.common.selector",
        "sol_h3._vendor.sol_attn.common.layout_utils",
        "sol_h3._vendor.sol_attn.sm120.kernel",
        "sol_h3._vendor.sol_attn.sm120.mainloop",
        "sol_h3._vendor.sol_attn._vendor.flash_attn.cute.utils",
    )
    files = []
    source_root = (Path(__file__).resolve().parent / "_vendor" / "sol_attn").resolve(strict=True)
    manifest_files = manifest.get("files") or {}
    for module_name in critical:
        path = _module_file(module_name)
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"loaded Sol-Attn module escaped packaged source root: {path}") from exc
        expected = (manifest_files.get(relative) or {}).get("packaged_sha256")
        actual = _sha256_file(path)
        if expected is None:
            raise RuntimeError(f"loaded Sol-Attn module is absent from packaged manifest: {relative}")
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        canonical_sha = hashlib.sha256(canonical).hexdigest()
        if actual != expected and canonical_sha != expected:
            raise RuntimeError(f"loaded Sol-Attn module bytes differ from packaged manifest: {relative}")
        files.append(
            {
                "module": module_name,
                "path": str(path),
                "relative_path": relative,
                "sha256": actual,
                "manifest_sha256": expected,
            }
        )

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "source": manifest.get("source"),
        "revision": manifest.get("revision"),
        "contract": provenance.CONTRACT,
        "files": files,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "cutlass_dsl": version("nvidia-cutlass-dsl") or version("cutlass"),
        "cuda_python": version("cuda-python"),
        "triton": version("triton"),
    }


def _prepare_independent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, tau: float, scale: float):
    """Independent PyTorch summaries/diag threshold from original BF16 BTHD."""
    block = 64
    _, tq, heads, dim = q.shape
    tkv = int(k.shape[1])
    k_blocks = (tkv + block - 1) // block
    q_blocks = (tq + block - 1) // block
    kc_parts = []
    vc_parts = []
    for index in range(k_blocks):
        start = index * block
        stop = min(tkv, start + block)
        ks = k[:, start:stop].float()
        vs = v[:, start:stop].float()
        kc_parts.append(ks.mean(dim=1))
        vc_parts.append(vs.sum(dim=1))
    kc_fp32 = torch.stack(kc_parts, dim=1)
    vc_fp32 = torch.stack(vc_parts, dim=1)
    kc_bf16 = kc_fp32.to(torch.bfloat16)
    vc_bf16 = vc_fp32.to(torch.bfloat16)
    qbar = []
    for index in range(q_blocks):
        start = index * block
        stop = min(tq, start + block)
        qbar.append(q[:, start:stop].float().mean(dim=1))
    qbar = torch.stack(qbar, dim=1)
    kc_stats = kc_bf16.float()
    kc_mean = kc_stats.mean(dim=1)
    kc_var = (kc_stats.square().mean(dim=1) - kc_mean.square()).clamp_min(0.0)
    log2_scale = float(scale) * 1.4426950408889634
    mean = (qbar * kc_mean[:, None]).sum(dim=-1) * log2_scale
    variance = (qbar.square() * kc_var[:, None]).sum(dim=-1) * (log2_scale * log2_scale)
    threshold = mean + float(tau) * torch.sqrt(variance + 1.0e-6)
    return kc_bf16, vc_bf16, threshold, qbar


def _decode_route_trace(trace: torch.Tensor, k_blocks: int) -> torch.Tensor:
    if trace.dtype != torch.int32 or trace.ndim != 5 or trace.shape[-1] != 2:
        raise RuntimeError("first-high Sol-local debug route trace has unexpected layout")
    _, q_blocks, heads, groups, _ = trace.shape
    result = torch.zeros((q_blocks, heads, k_blocks), dtype=torch.bool, device=trace.device)
    for group in range(groups):
        for word in range(2):
            raw = trace[0, :, :, group, word].to(torch.int64) & 0xFFFFFFFF
            for bit in range(32):
                block = group * 64 + word * 32 + bit
                if block >= k_blocks:
                    break
                result[:, :, block] = ((raw >> bit) & 1).bool()
    return result


def _independent_routes(
    qbar: torch.Tensor,
    kc: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale: float,
    sink_rows: int,
    k_tokens: int,
):
    q_blocks = int(qbar.shape[1])
    k_blocks = int(kc.shape[1])
    log2_scale = float(scale) * 1.4426950408889634
    means = torch.empty((q_blocks, qbar.shape[2], k_blocks), device=qbar.device, dtype=torch.float32)
    routes = torch.empty_like(means, dtype=torch.bool)
    forced = torch.empty_like(routes, dtype=torch.uint8)
    sink_end = (int(sink_rows) + 63) // 64
    for q_block in range(q_blocks):
        row = torch.einsum("bhd,bkhd->bhk", qbar[:, q_block], kc.float())[0] * log2_scale
        means[q_block] = row
        threshold_route = row > threshold[0, q_block, :, None]
        ordinal = torch.arange(k_blocks, device=row.device)
        neighbor = (ordinal - q_block).abs() <= 1
        sink = ordinal < sink_end
        routes[q_block] = threshold_route | neighbor[None, :] | sink[None, :]
        forced[q_block] = (
            threshold_route.to(torch.uint8)
            | (neighbor[None, :].to(torch.uint8) << 1)
            | (sink[None, :].to(torch.uint8) << 2)
        )
    margins = means - threshold[0, :, :, None]
    return routes, means, margins, forced


def _diagnostic_sm120_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    threshold: torch.Tensor,
    *,
    scale: float,
    sink_rows: int,
    trace: bool,
):
    """Compile a diagnostic-only SM120 specialization in an isolated cache."""
    import cutlass.cute as cute

    from ._vendor.sol_attn.interface import _sink_block_range, _stream, _to_cute_tensors
    from ._vendor.sol_attn.sm120 import make_kernel

    if tuple(torch.cuda.get_device_capability(q.device)) != (12, 0):
        raise RuntimeError("first-high Sol-local diagnostic route trace requires SM120")
    output = torch.empty_like(q)
    q_blocks = (int(q.shape[1]) + 63) // 64
    k_blocks = (int(k.shape[1]) + 63) // 64
    route_groups = (k_blocks + 63) // 64
    if trace:
        auxiliary = torch.zeros(
            (int(q.shape[0]), q_blocks, int(q.shape[2]), route_groups, 2),
            device=q.device,
            dtype=torch.int32,
        )
        cache = _DEBUG_COMPILED
    else:
        auxiliary = torch.empty(
            (int(q.shape[0]), int(q.shape[1]), int(q.shape[2])),
            device=q.device,
            dtype=torch.float32,
        )
        cache = _LSE_COMPILED
    sink_start_block, sink_end_block = _sink_block_range(int(k.shape[1]), 0, int(sink_rows))
    operator = make_kernel(debug_route_trace=trace, key_bias_enabled=False)
    key_bias_arg = threshold
    tensors = [q, k, v, output, kc, vc, threshold, key_bias_arg, auxiliary]
    args = _to_cute_tensors(tensors)
    stream = _stream(q.device)
    layout_key = tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype)) for x in tensors)
    key = (str(q.device), layout_key, float(scale), sink_start_block, sink_end_block)
    compiled = cache.get(key)
    if compiled is None:
        compiled = cute.compile(
            operator,
            *args,
            float(scale),
            sink_start_block,
            sink_end_block,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache[key] = compiled
    compiled(
        *args,
        float(scale),
        sink_start_block,
        sink_end_block,
        stream=stream,
    )
    return output, auxiliary


def _frozen_route_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    routes: torch.Tensor,
    sparse_output: torch.Tensor,
    kernel_lse: torch.Tensor,
    *,
    scale: float,
) -> dict[str, Any]:
    """Streaming FP32 mixed exact/approx reference with kernel routes frozen."""
    device = q.device
    q_blocks, heads, k_blocks = routes.shape
    tkv = int(k.shape[1])
    output_acc = _metric_accumulator(device)
    numerator_acc = _metric_accumulator(device)
    denominator_acc = _metric_accumulator(device)
    lse_acc = _metric_accumulator(device)
    finite = torch.ones((), device=device, dtype=torch.bool)
    for q_block in range(q_blocks):
        q_start = q_block * 64
        q_stop = min(int(q.shape[1]), q_start + 64)
        for head in range(heads):
            qh = q[0, q_start:q_stop, head].float()
            exact_blocks = routes[q_block, head]
            row_mask = exact_blocks.repeat_interleave(64)[:tkv]
            approx_blocks = ~exact_blocks
            row_max = torch.full((qh.shape[0],), -torch.inf, device=device, dtype=torch.float32)
            exact_scores = None
            approx_scores = None
            if bool(row_mask.any().item()):
                exact_scores = qh @ k[0, row_mask, head].float().T
                exact_scores.mul_(float(scale))
                row_max = torch.maximum(row_max, exact_scores.max(dim=1).values)
            if bool(approx_blocks.any().item()):
                approx_scores = qh @ kc[0, approx_blocks, head].float().T
                approx_scores.mul_(float(scale))
                row_max = torch.maximum(row_max, approx_scores.max(dim=1).values)
            if not bool(torch.isfinite(row_max).all().item()):
                raise RuntimeError("first-high Sol-local frozen-route reference produced no finite attention mass")
            denominator = torch.zeros_like(row_max)
            numerator = torch.zeros((qh.shape[0], qh.shape[1]), device=device, dtype=torch.float32)
            if exact_scores is not None:
                p = torch.exp(exact_scores - row_max[:, None])
                denominator.add_(p.sum(dim=1))
                numerator.add_(p @ v[0, row_mask, head].float())
            if approx_scores is not None:
                p = torch.exp(approx_scores - row_max[:, None])
                block_ids = torch.arange(k_blocks, device=device)[approx_blocks]
                lengths = (tkv - block_ids * 64).clamp(min=0, max=64).to(torch.float32)
                denominator.add_((p * lengths[None, :]).sum(dim=1))
                numerator.add_(p @ vc[0, approx_blocks, head].float())
            reference = numerator / denominator[:, None]
            reference_lse = row_max + torch.log(denominator)
            got = sparse_output[0, q_start:q_stop, head].float()
            got_lse = kernel_lse[0, q_start:q_stop, head].float()
            kernel_denom_in_ref_scale = torch.exp(got_lse - row_max)
            kernel_num_in_ref_scale = got * denominator[:, None]
            _accumulate_metric(output_acc, got, reference)
            _accumulate_metric(numerator_acc, kernel_num_in_ref_scale, numerator)
            _accumulate_metric(denominator_acc, kernel_denom_in_ref_scale, denominator)
            _accumulate_metric(lse_acc, got_lse, reference_lse)
            finite = finite & torch.isfinite(reference).all() & torch.isfinite(reference_lse).all()
    return {
        "finite": bool(finite.item()),
        "output": _finish_metric(output_acc),
        "numerator_scaled_to_reference_rowmax": _finish_metric(numerator_acc),
        "denominator_scaled_to_reference_rowmax": _finish_metric(denominator_acc),
        "lse": _finish_metric(lse_acc),
        "reference": "FP32 streaming; exact routed rows plus zeroth-order KC/VC approximation",
    }


def _append_provenance_once(options: dict[str, Any]) -> None:
    sink = options.get(EVIDENCE_KEY)
    if sink is None:
        raise RuntimeError("first-high Sol-local diagnostic evidence sink is missing")
    items = getattr(sink, "items", None)
    if not isinstance(items, list):
        raise RuntimeError("first-high Sol-local diagnostic evidence sink is invalid")
    if any(isinstance(item, dict) and item.get("kind") == "sol_sparse_provenance" for item in items):
        return
    sink.append({"kind": "sol_sparse_provenance", "provenance": loaded_sparse_provenance()})


def _run_witness(
    options: dict[str, Any],
    group: LocalGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Any,
    config: Any,
    returned: torch.Tensor,
) -> None:
    record = group.witness_record
    if record is None:
        raise RuntimeError("first-high Sol-local witness group has no preserved VDN record")
    if record.get("completed"):
        raise RuntimeError("first-high Sol-local witness was completed more than once")
    if int(q.shape[2]) != group.q_rows or int(k.shape[2]) != group.kv_rows:
        raise RuntimeError("first-high Sol-local witness Q/KV geometry changed across the provider boundary")
    if float(group.scale) != float(q.shape[-1] ** -0.5):
        raise RuntimeError("first-high Sol-local witness attention scale changed")
    if state.kernel is None:
        raise RuntimeError("first-high Sol-local witness requires the returned E call to load the real kernel first")
    if getattr(state.kernel, "backend_name", None) != "cute_sm120" or getattr(state.kernel, "source_tree_verified", False) is not True:
        raise RuntimeError("first-high Sol-local witness did not execute the packaged verified SM120 backend")

    qb, kb, vb = (value.transpose(1, 2) for value in (q, k, v))
    native = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    sparse_out = state.kernel(
        qb,
        kb,
        vb,
        tau=config.tau,
        thresh_type="diag",
        kv_splits=1,
        sink_start=0,
        sink_tokens=group.original_sink_rows,
    )

    from ._vendor.sol_attn.preprocess import prepare

    kc, vc, threshold, _qbar_packaged = prepare(
        qb,
        kb,
        vb,
        tau=float(config.tau),
        scale=float(group.scale),
        thresh_type="diag",
        valid_tokens=int(qb.shape[1]),
        valid_kv_tokens=int(kb.shape[1]),
        return_q_bar=True,
    )
    debug_out, trace = _diagnostic_sm120_launch(
        qb,
        kb,
        vb,
        kc,
        vc,
        threshold,
        scale=group.scale,
        sink_rows=group.original_sink_rows,
        trace=True,
    )
    lse_out, lse = _diagnostic_sm120_launch(
        qb,
        kb,
        vb,
        kc,
        vc,
        threshold,
        scale=group.scale,
        sink_rows=group.original_sink_rows,
        trace=False,
    )
    if not torch.equal(debug_out, lse_out) or not torch.equal(debug_out, sparse_out):
        raise RuntimeError("first-high Sol-local debug/LSE specialization changed ordinary sparse output")

    independent_kc, independent_vc, independent_threshold, qbar = _prepare_independent(
        qb, kb, vb, tau=float(config.tau), scale=float(group.scale)
    )
    summary_metrics = {
        "kc": _metrics(kc, independent_kc),
        "vc": _metrics(vc, independent_vc),
        "threshold": _metrics(threshold, independent_threshold),
    }
    k_blocks = int(kc.shape[1])
    traced_routes = _decode_route_trace(trace, k_blocks)
    independent_routes, column_means, margins, forced = _independent_routes(
        qbar,
        kc,
        threshold,
        scale=group.scale,
        sink_rows=group.original_sink_rows,
        k_tokens=int(kb.shape[1]),
    )
    mismatch = traced_routes ^ independent_routes
    mismatch_count = int(mismatch.sum().item())
    mismatch_examples = []
    if mismatch_count:
        indices = mismatch.nonzero(as_tuple=False)[:128].detach().cpu().tolist()
        for q_block, head, k_block in indices:
            flags = int(forced[q_block, head, k_block].item())
            mismatch_examples.append(
                {
                    "q_block": int(q_block),
                    "head": int(head),
                    "kv_block": int(k_block),
                    "trace_exact": bool(traced_routes[q_block, head, k_block].item()),
                    "independent_exact": bool(independent_routes[q_block, head, k_block].item()),
                    "column_mean_log2": float(column_means[q_block, head, k_block].item()),
                    "threshold_log2": float(threshold[0, q_block, head].item()),
                    "margin_log2": float(margins[q_block, head, k_block].item()),
                    "threshold_selected": bool(flags & 1),
                    "ordinal_neighbor_forced": bool(flags & 2),
                    "sink_forced": bool(flags & 4),
                }
            )

    frozen = _frozen_route_reference(
        qb,
        kb,
        vb,
        kc,
        vc,
        traced_routes,
        sparse_out,
        lse,
        scale=group.scale,
    )
    returned_bthd = returned.reshape(1, int(q.shape[2]), int(q.shape[1]), int(q.shape[3]))
    all_selected_vs_native = _metrics(returned_bthd, native)
    from .sparse import arithmetic_gate_passes

    record.update(
        {
            "all_selected_output": returned_bthd.detach().to(device="cpu", copy=True),
            "native_sdpa_output": native.detach().to(device="cpu", copy=True),
            "production_sparse_output": sparse_out.detach().to(device="cpu", copy=True),
            "kc": kc.detach().to(device="cpu", copy=True),
            "vc": vc.detach().to(device="cpu", copy=True),
            "threshold": threshold.detach().to(device="cpu", copy=True),
            "route_trace": trace.detach().to(device="cpu", copy=True),
            "route_column_means": column_means.detach().to(device="cpu", copy=True),
            "route_margins": margins.detach().to(device="cpu", copy=True),
            "kernel_lse": lse.detach().to(device="cpu", copy=True),
            "all_selected_vs_native": all_selected_vs_native,
            "all_selected_arithmetic_gate_pass": bool(arithmetic_gate_passes(all_selected_vs_native)),
            "summary_metrics": summary_metrics,
            "route_trace_matches_independent": mismatch_count == 0,
            "route_mismatch_count": mismatch_count,
            "route_mismatch_examples": mismatch_examples,
            "frozen_route_reference": frozen,
            "debug_output_sha256": _tensor_sha256_cpu(debug_out.detach().to(device="cpu", copy=True)),
            "debug_matches_ordinary_sparse": True,
            "packaged_backend": getattr(state.kernel, "backend_name", None),
            "packaged_source_tree_verified": getattr(state.kernel, "source_tree_verified", False),
            "completed": True,
        }
    )
    _append_provenance_once(options)


def _install_sparse_patch() -> None:
    global _ORIGINAL_SPARSE_ATTENTION
    from . import sparse

    if getattr(sparse.attention, "_first_high_sol_local_e_v1", False):
        return
    _ORIGINAL_SPARSE_ATTENTION = sparse.attention

    def attention(q, k, v, prefix, config, state, *args, **kwargs):
        group = _LOCAL_GROUP.get()
        if group is None:
            return _ORIGINAL_SPARSE_ATTENTION(q, k, v, prefix, config, state, *args, **kwargs)
        if int(prefix) != group.original_sink_rows or int(k.shape[2]) != group.kv_rows:
            raise RuntimeError("first-high Sol-local E final sink boundary does not match VDN-owned geometry")
        before_sparse = int(state.sparse_calls)
        try:
            result = _ORIGINAL_SPARSE_ATTENTION(
                q,
                k,
                v,
                int(k.shape[2]),
                config,
                state,
                *args,
                **kwargs,
            )
        except sparse.KernelUnavailable as exc:
            raise RuntimeError(f"first-high Sol-local E packaged SM120 kernel unavailable: {exc}") from exc
        if int(state.sparse_calls) != before_sparse + 1:
            raise RuntimeError("first-high Sol-local E returned call changed sparse counter unexpectedly")
        state.sparse_calls = before_sparse
        if group.block_index == 2 and group.group_index in WITNESS_GROUPS:
            options = kwargs.get("diagnostic_transformer_options")
            if not isinstance(options, dict):
                raise RuntimeError("first-high Sol-local E witness lost transformer-option ownership")
            _run_witness(options, group, q, k, v, state, config, result)
        return result

    attention._first_high_sol_local_e_v1 = True
    attention._first_high_sol_local_original = _ORIGINAL_SPARSE_ATTENTION
    sparse.attention = attention


def _install_receipt_patch() -> None:
    global _ORIGINAL_RUNTIME_RECEIPT
    from . import runtime

    if getattr(runtime.receipt, "_first_high_sol_local_e_v1", False):
        return
    _ORIGINAL_RUNTIME_RECEIPT = runtime.receipt

    def receipt(options, block_index, route, *args, **kwargs):
        group = _LOCAL_GROUP.get()
        if group is not None and route == "vdn_local_sol":
            if int(block_index) != group.block_index:
                raise RuntimeError("first-high Sol-local E receipt block identity diverged")
            route = "vdn_local_sol_all_selected_e"
        return _ORIGINAL_RUNTIME_RECEIPT(options, block_index, route, *args, **kwargs)

    receipt._first_high_sol_local_e_v1 = True
    receipt._first_high_sol_local_original = _ORIGINAL_RUNTIME_RECEIPT
    runtime.receipt = receipt


def _install_block_counter_patch() -> None:
    global _ORIGINAL_BLOCK_CALL
    from . import runtime

    cls = runtime.BlockPatch
    if getattr(cls.__call__, "_first_high_sol_local_e_v1", False):
        return
    _ORIGINAL_BLOCK_CALL = cls.__call__
    fields = (
        "eligible_calls",
        "vdn_local_sol_calls",
        "vdn_requested_q_rows",
        "vdn_kernel_q_rows",
        "vdn_rectangular_sol_calls",
    )

    def call(self, args, extra):
        options = args.get("transformer_options") if isinstance(args, dict) else None
        request = parse_request(options) if isinstance(options, dict) else None
        active = runtime._FORWARD.get()
        state = active[1] if active is not None else None
        restore = request is not None and self.index >= 2 and state is not None
        before = {name: int(getattr(state, name)) for name in fields} if restore else None
        result = _ORIGINAL_BLOCK_CALL(self, args, extra)
        if restore:
            after = {name: int(getattr(state, name)) for name in fields}
            local_calls = after["vdn_local_sol_calls"] - before["vdn_local_sol_calls"]
            if local_calls != 11:
                raise RuntimeError(
                    f"first-high Sol-local E expected 11 returned local calls in block {self.index}, got {local_calls}"
                )
            if after["vdn_rectangular_sol_calls"] - before["vdn_rectangular_sol_calls"] != 11:
                raise RuntimeError("first-high Sol-local E local calls stopped using direct rectangular VDN API v3")
            for name, value in before.items():
                setattr(state, name, value)
        return result

    call._first_high_sol_local_e_v1 = True
    call._first_high_sol_local_original = _ORIGINAL_BLOCK_CALL
    cls.__call__ = call


def _install_history_patches() -> None:
    from . import interop

    current_vdn = interop._vdn_history_identity
    if not getattr(current_vdn, "_first_high_sol_local_e_v1", False):
        def vdn_history_identity(forward, options, layout):
            if getattr(forward, "_h3_first_high_sol_local_diagnostic_v1", False) is True:
                original = getattr(forward, "_h3_first_high_sol_local_original_forward", None)
                if not callable(original) or original is forward:
                    return None
                forward = original
            return current_vdn(forward, options, layout)

        vdn_history_identity._first_high_sol_local_e_v1 = True
        vdn_history_identity._first_high_sol_local_original = current_vdn
        interop._vdn_history_identity = vdn_history_identity

    cls = interop.HistoryPolicy
    if getattr(cls, "_first_high_sol_local_e_v1", False):
        return
    original_call = cls.__call__
    original_accept = cls.accept_receipts

    def call(self, *, layout, options, model):
        identity = original_call(self, layout=layout, options=options, model=model)
        diagnostic = history_identity(options)
        if diagnostic is None or identity is None:
            return identity
        return (*identity, diagnostic)

    def accept_receipts(self, receipts):
        if not receipts:
            return False
        normalized = []
        for item in receipts:
            if isinstance(item, tuple) and len(item) >= 3 and item[0] == "sol_h3" and item[2] == "vdn_local_sol_all_selected_e":
                normalized.append((item[0], item[1], "vdn_local_sol"))
            else:
                normalized.append(item)
        return original_accept(self, normalized)

    cls.__call__ = call
    cls.accept_receipts = accept_receipts
    cls._first_high_sol_local_e_v1 = True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_sparse_patch()
    _install_receipt_patch()
    _install_block_counter_patch()
    _install_history_patches()
    _INSTALLED = True


install()

__all__ = [
    "EVIDENCE_KEY",
    "MODE",
    "RECEIPTS_KEY",
    "REQUEST_KEY",
    "WITNESS_GROUPS",
    "enter_local_group",
    "exit_local_group",
    "history_identity",
    "loaded_sparse_provenance",
    "parse_request",
]
