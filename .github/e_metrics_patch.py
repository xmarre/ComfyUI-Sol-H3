from pathlib import Path

path = Path("sol_h3/first_high_sol_local_diagnostic.py")
text = path.read_text(encoding="utf-8")

anchor = '''def _metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    from .sparse import error_metrics

    return error_metrics(got, want)


'''
insert = r'''def _metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    from .sparse import error_metrics

    return error_metrics(got, want)


def _sampled_p99_abs(got: torch.Tensor, want: torch.Tensor, *, max_values: int = 262144) -> dict[str, Any]:
    """Bounded deterministic p99 estimate without materializing full FP32 error."""
    if got.shape != want.shape:
        raise RuntimeError("first-high Sol-local detailed metrics require equal tensor shapes")
    flat_got = got.reshape(-1)
    flat_want = want.reshape(-1)
    total = int(flat_got.numel())
    count = min(total, int(max_values))
    if count <= 0:
        return {"p99_abs": 0.0, "sample_count": 0, "population_count": total, "method": "empty"}
    if count == total:
        delta = (flat_got.float() - flat_want.float()).abs()
        method = "exact_all_values"
    else:
        ordinal = torch.arange(count, device=flat_got.device, dtype=torch.int64)
        index = torch.div(ordinal * (total - 1), count - 1, rounding_mode="floor")
        delta = (flat_got.index_select(0, index).float() - flat_want.index_select(0, index).float()).abs()
        method = "deterministic_even_sample"
    return {
        "p99_abs": float(torch.quantile(delta, 0.99).item()),
        "sample_count": count,
        "population_count": total,
        "method": method,
    }


def _detailed_bthd_metrics(got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    """Bounded exact tile/head aggregates plus p99 and worst coordinates for BTHD."""
    if got.shape != want.shape or got.ndim != 4 or int(got.shape[0]) != 1:
        raise RuntimeError("first-high Sol-local detailed metrics require equal BTHD tensors with B=1")
    tokens, heads, dim = (int(got.shape[1]), int(got.shape[2]), int(got.shape[3]))
    q_blocks = (tokens + 63) // 64
    device = got.device
    head_abs_sum = torch.zeros(heads, device=device, dtype=torch.float64)
    head_delta_sq = torch.zeros(heads, device=device, dtype=torch.float64)
    head_ref_sq = torch.zeros(heads, device=device, dtype=torch.float64)
    head_abs_max = torch.zeros(heads, device=device, dtype=torch.float32)
    per_q64 = []
    per_head_q64_mean = torch.empty((q_blocks, heads), device=device, dtype=torch.float32)
    per_head_q64_max = torch.empty_like(per_head_q64_mean)
    per_head_q64_rel_l2 = torch.empty_like(per_head_q64_mean)
    global_abs_sum = torch.zeros((), device=device, dtype=torch.float64)
    global_delta_sq = torch.zeros((), device=device, dtype=torch.float64)
    global_ref_sq = torch.zeros((), device=device, dtype=torch.float64)
    global_max = torch.tensor(-1.0, device=device, dtype=torch.float32)
    global_max_row = torch.zeros((), device=device, dtype=torch.int64)
    global_max_head = torch.zeros((), device=device, dtype=torch.int64)
    global_max_dim = torch.zeros((), device=device, dtype=torch.int64)
    finite = torch.ones((), device=device, dtype=torch.bool)

    for q_block in range(q_blocks):
        start = q_block * 64
        stop = min(tokens, start + 64)
        got_f = got[0, start:stop].float()
        want_f = want[0, start:stop].float()
        delta = got_f - want_f
        abs_delta = delta.abs()
        delta_sq = delta.double().square()
        ref_sq = want_f.double().square()
        finite = finite & torch.isfinite(got_f).all() & torch.isfinite(want_f).all()
        global_abs_sum += abs_delta.double().sum()
        global_delta_sq += delta_sq.sum()
        global_ref_sq += ref_sq.sum()

        head_abs_sum += abs_delta.double().sum(dim=(0, 2))
        head_delta_sq += delta_sq.sum(dim=(0, 2))
        head_ref_sq += ref_sq.sum(dim=(0, 2))
        tile_head_max = abs_delta.amax(dim=(0, 2))
        head_abs_max = torch.maximum(head_abs_max, tile_head_max)
        tile_head_count = max((stop - start) * dim, 1)
        per_head_q64_mean[q_block] = abs_delta.mean(dim=(0, 2))
        per_head_q64_max[q_block] = tile_head_max
        per_head_q64_rel_l2[q_block] = torch.sqrt(
            delta_sq.sum(dim=(0, 2)) / ref_sq.sum(dim=(0, 2)).clamp_min(1.0e-24)
        ).float()

        local_max, local_index = abs_delta.reshape(-1).max(dim=0)
        if bool((local_max > global_max).item()):
            flat = int(local_index.item())
            local_row = flat // (heads * dim)
            remainder = flat % (heads * dim)
            global_max = local_max
            global_max_row = torch.tensor(start + local_row, device=device, dtype=torch.int64)
            global_max_head = torch.tensor(remainder // dim, device=device, dtype=torch.int64)
            global_max_dim = torch.tensor(remainder % dim, device=device, dtype=torch.int64)
        per_q64.append(
            {
                "q_block": q_block,
                "row_start": start,
                "row_stop": stop,
                "max_abs": float(abs_delta.max().item()),
                "mean_abs": float(abs_delta.mean().item()),
                "rel_l2": float(
                    torch.sqrt(delta_sq.sum() / ref_sq.sum().clamp_min(1.0e-24)).item()
                ),
            }
        )

    head_count = max(tokens * dim, 1)
    per_head = []
    for head in range(heads):
        per_head.append(
            {
                "head": head,
                "max_abs": float(head_abs_max[head].item()),
                "mean_abs": float((head_abs_sum[head] / head_count).item()),
                "rel_l2": float(
                    torch.sqrt(head_delta_sq[head] / head_ref_sq[head].clamp_min(1.0e-24)).item()
                ),
            }
        )
    p99 = _sampled_p99_abs(got, want)
    total = max(tokens * heads * dim, 1)
    result = {
        "finite": bool(finite.item()),
        "max_abs": float(global_max.item()),
        "mean_abs": float((global_abs_sum / total).item()),
        "rel_l2": float(torch.sqrt(global_delta_sq / global_ref_sq.clamp_min(1.0e-24)).item()),
        **p99,
        "worst_coordinate": {
            "row": int(global_max_row.item()),
            "q_block": int(global_max_row.item()) // 64,
            "head": int(global_max_head.item()),
            "dim": int(global_max_dim.item()),
        },
        "per_head": per_head,
        "per_q64": per_q64,
        "per_head_q64": {
            "mean_abs": per_head_q64_mean.detach().cpu().tolist(),
            "max_abs": per_head_q64_max.detach().cpu().tolist(),
            "rel_l2": per_head_q64_rel_l2.detach().cpu().tolist(),
        },
    }
    return result


def _relative_error_summary(got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    got_f = got.float()
    want_f = want.float()
    rel = (got_f - want_f).abs() / want_f.abs().clamp_min(1.0e-12)
    flat = rel.reshape(-1)
    maximum, index = flat.max(dim=0)
    return {
        "finite": bool(torch.isfinite(rel).all().item()),
        "max": float(maximum.item()),
        "mean": float(rel.mean().item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "worst_flat_index": int(index.item()),
        "count": int(flat.numel()),
    }


def _route_margin_summary(margins: torch.Tensor, forced: torch.Tensor, mismatch: torch.Tensor) -> dict[str, Any]:
    abs_margin = margins.abs()
    geometry_forced = (forced & 0b110) != 0
    threshold_only = ~geometry_forced
    selected = margins > 0
    threshold_values = abs_margin[threshold_only]
    if threshold_values.numel():
        min_abs = float(threshold_values.min().item())
        p01_abs = float(torch.quantile(threshold_values, 0.01).item())
    else:
        min_abs = math.nan
        p01_abs = math.nan
    near_1e4 = abs_margin <= 1.0e-4
    near_1e3 = abs_margin <= 1.0e-3
    near_1e2 = abs_margin <= 1.0e-2
    return {
        "min_abs_threshold_margin_log2": min_abs,
        "p01_abs_threshold_margin_log2": p01_abs,
        "threshold_selected_pairs": int((selected & threshold_only).sum().item()),
        "geometry_forced_pairs": int(geometry_forced.sum().item()),
        "pairs_within_1e-4_log2": int((near_1e4 & threshold_only).sum().item()),
        "pairs_within_1e-3_log2": int((near_1e3 & threshold_only).sum().item()),
        "pairs_within_1e-2_log2": int((near_1e2 & threshold_only).sum().item()),
        "mismatch_within_1e-3_log2": int((mismatch & near_1e3).sum().item()),
        "mismatch_farther_than_1e-3_log2": int((mismatch & ~near_1e3).sum().item()),
    }


'''
if anchor not in text:
    raise SystemExit("metrics anchor not found")
text = text.replace(anchor, insert, 1)

text = text.replace(
    '''            if not bool(torch.isfinite(row_max).all().item()):
                raise RuntimeError("first-high Sol-local frozen-route reference produced no finite attention mass")

            denominator = torch.zeros_like(row_max)
''',
    '''            finite = finite & torch.isfinite(row_max).all()

            denominator = torch.zeros_like(row_max)
''',
    1,
)

text = text.replace(
    '''    lse_acc = _metric_accumulator(device)
    finite = torch.ones((), device=device, dtype=torch.bool)
    max_live_score_elements = 0
''',
    '''    lse_acc = _metric_accumulator(device)
    finite = torch.ones((), device=device, dtype=torch.bool)
    denominator_relative = torch.empty(
        (int(q.shape[1]), int(q.shape[2])), device=device, dtype=torch.float32
    )
    max_live_score_elements = 0
''',
    1,
)

text = text.replace(
    '''            kernel_denom_in_ref_scale = torch.exp(got_lse - row_max)
            kernel_num_in_ref_scale = got * kernel_denom_in_ref_scale[:, None]
            _accumulate_metric(output_acc, got, reference)
''',
    '''            kernel_denom_in_ref_scale = torch.exp(got_lse - row_max)
            kernel_num_in_ref_scale = got * kernel_denom_in_ref_scale[:, None]
            denominator_relative[q_start:q_stop, head] = (
                (kernel_denom_in_ref_scale - denominator).abs()
                / denominator.abs().clamp_min(1.0e-12)
            )
            _accumulate_metric(output_acc, got, reference)
''',
    1,
)

text = text.replace(
    '''        "lse": _finish_metric(lse_acc),
        "reference": "FP32 two-pass streaming; exact routed rows plus zeroth-order KC/VC approximation",
''',
    '''        "lse": _finish_metric(lse_acc),
        "denominator_relative_error": {
            "finite": bool(torch.isfinite(denominator_relative).all().item()),
            "max": float(denominator_relative.max().item()),
            "mean": float(denominator_relative.mean().item()),
            "p99": float(torch.quantile(denominator_relative.reshape(-1), 0.99).item()),
            "count": int(denominator_relative.numel()),
        },
        "reference": "FP32 two-pass streaming; exact routed rows plus zeroth-order KC/VC approximation",
''',
    1,
)

text = text.replace(
    '''    all_selected_vs_native = _metrics(returned_bthd, native)
    exit_hashes = {
''',
    '''    all_selected_vs_native = _detailed_bthd_metrics(returned_bthd, native)
    margin_summary = _route_margin_summary(margins, forced, mismatch)
    exit_hashes = {
''',
    1,
)

text = text.replace(
    '''            "route_mismatch_examples": mismatch_examples,
            "sparse_selected_block_pairs": sparse_selected_pairs,
''',
    '''            "route_mismatch_examples": mismatch_examples,
            "route_margin_summary": margin_summary,
            "sparse_selected_block_pairs": sparse_selected_pairs,
''',
    1,
)

path.write_text(text, encoding="utf-8")
