from pathlib import Path

path = Path("sol_h3/first_high_sol_local_diagnostic.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "_LSE_COMPILED: dict[tuple[Any, ...], Any] = {}\n",
    "_LSE_COMPILED: dict[tuple[Any, ...], Any] = {}\n_REFERENCE_KEY_CHUNK = 1024\n",
    1,
)

start = text.index("def _frozen_route_reference(\n")
end = text.index("\ndef _append_provenance_once(", start)
replacement = r'''def _frozen_route_reference(
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
    """Streaming FP32 mixed exact/approx reference with kernel routes frozen.

    Score storage is capped to one Q64 tile by at most ``_REFERENCE_KEY_CHUNK``
    exact rows or approximate blocks.  The first pass establishes one common
    stable row maximum; the second accumulates numerator and denominator in that
    shared scale.  No full QxKVxH score tensor is materialized.
    """
    device = q.device
    q_blocks, heads, k_blocks = routes.shape
    tkv = int(k.shape[1])
    output_acc = _metric_accumulator(device)
    numerator_acc = _metric_accumulator(device)
    denominator_acc = _metric_accumulator(device)
    lse_acc = _metric_accumulator(device)
    finite = torch.ones((), device=device, dtype=torch.bool)
    max_live_score_elements = 0

    for q_block in range(q_blocks):
        q_start = q_block * 64
        q_stop = min(int(q.shape[1]), q_start + 64)
        for head in range(heads):
            qh = q[0, q_start:q_stop, head].float()
            exact_blocks = routes[q_block, head]
            exact_rows_mask = exact_blocks.repeat_interleave(64)[:tkv]
            exact_rows = exact_rows_mask.nonzero(as_tuple=False).flatten()
            approx_blocks = (~exact_blocks).nonzero(as_tuple=False).flatten()
            row_max = torch.full((qh.shape[0],), -torch.inf, device=device, dtype=torch.float32)

            for offset in range(0, int(exact_rows.numel()), _REFERENCE_KEY_CHUNK):
                row_ids = exact_rows[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = qh @ k[0, row_ids, head].float().T
                scores.mul_(float(scale))
                max_live_score_elements = max(max_live_score_elements, int(scores.numel()))
                row_max = torch.maximum(row_max, scores.max(dim=1).values)
                del scores
            for offset in range(0, int(approx_blocks.numel()), _REFERENCE_KEY_CHUNK):
                block_ids = approx_blocks[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = qh @ kc[0, block_ids, head].float().T
                scores.mul_(float(scale))
                max_live_score_elements = max(max_live_score_elements, int(scores.numel()))
                row_max = torch.maximum(row_max, scores.max(dim=1).values)
                del scores

            if not bool(torch.isfinite(row_max).all().item()):
                raise RuntimeError("first-high Sol-local frozen-route reference produced no finite attention mass")

            denominator = torch.zeros_like(row_max)
            numerator = torch.zeros((qh.shape[0], qh.shape[1]), device=device, dtype=torch.float32)
            for offset in range(0, int(exact_rows.numel()), _REFERENCE_KEY_CHUNK):
                row_ids = exact_rows[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = qh @ k[0, row_ids, head].float().T
                scores.mul_(float(scale))
                probabilities = torch.exp(scores - row_max[:, None])
                denominator.add_(probabilities.sum(dim=1))
                numerator.add_(probabilities @ v[0, row_ids, head].float())
                del scores, probabilities
            for offset in range(0, int(approx_blocks.numel()), _REFERENCE_KEY_CHUNK):
                block_ids = approx_blocks[offset : offset + _REFERENCE_KEY_CHUNK]
                scores = qh @ kc[0, block_ids, head].float().T
                scores.mul_(float(scale))
                probabilities = torch.exp(scores - row_max[:, None])
                lengths = (tkv - block_ids * 64).clamp(min=0, max=64).to(torch.float32)
                denominator.add_((probabilities * lengths[None, :]).sum(dim=1))
                numerator.add_(probabilities @ vc[0, block_ids, head].float())
                del scores, probabilities

            reference = numerator / denominator[:, None]
            reference_lse = row_max + torch.log(denominator)
            got = sparse_output[0, q_start:q_stop, head].float()
            got_lse = kernel_lse[0, q_start:q_stop, head].float()
            kernel_denom_in_ref_scale = torch.exp(got_lse - row_max)
            kernel_num_in_ref_scale = got * kernel_denom_in_ref_scale[:, None]
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
        "reference": "FP32 two-pass streaming; exact routed rows plus zeroth-order KC/VC approximation",
        "score_chunk_keys": _REFERENCE_KEY_CHUNK,
        "max_live_score_elements": max_live_score_elements,
        "max_live_score_bytes_fp32": max_live_score_elements * 4,
    }
'''
text = text[:start] + replacement + text[end:]

start = text.index("def _run_witness(\n")
end = text.index("\ndef _install_sparse_patch(", start)
replacement = r'''def _run_witness(
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
    if (
        getattr(state.kernel, "backend_name", None) != "cute_sm120"
        or getattr(state.kernel, "source_tree_verified", False) is not True
    ):
        raise RuntimeError("first-high Sol-local witness did not execute the packaged verified SM120 backend")

    qb, kb, vb = (value.transpose(1, 2) for value in (q, k, v))
    saved_q = record.get("q")
    saved_k = record.get("k")
    saved_v = record.get("v")
    if not all(torch.is_tensor(value) and value.device.type == "cpu" for value in (saved_q, saved_k, saved_v)):
        raise RuntimeError("first-high Sol-local witness lost its pre-scratch CPU Q/K/V preservation")
    preserved_hashes = {
        "q": _tensor_sha256_cpu(saved_q),
        "k": _tensor_sha256_cpu(saved_k),
        "v": _tensor_sha256_cpu(saved_v),
    }
    entry_hashes = {
        "q": _tensor_sha256_cpu(qb[0]),
        "k": _tensor_sha256_cpu(kb[0]),
        "v": _tensor_sha256_cpu(vb[0]),
    }
    input_exact_on_entry = entry_hashes == preserved_hashes

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
    from .sparse import arithmetic_gate_passes

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
    debug_out, sparse_trace = _diagnostic_sm120_launch(
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
    all_selected_debug_out, all_selected_trace = _diagnostic_sm120_launch(
        qb,
        kb,
        vb,
        kc,
        vc,
        threshold,
        scale=group.scale,
        sink_rows=group.kv_rows,
        trace=True,
    )

    returned_bthd = returned.reshape(1, int(q.shape[2]), int(q.shape[1]), int(q.shape[3]))
    debug_sparse_metrics = _metrics(debug_out, sparse_out)
    lse_sparse_metrics = _metrics(lse_out, sparse_out)
    debug_all_selected_metrics = _metrics(all_selected_debug_out, returned_bthd)
    debug_specializations_conform = bool(
        arithmetic_gate_passes(debug_sparse_metrics)
        and arithmetic_gate_passes(lse_sparse_metrics)
        and arithmetic_gate_passes(debug_all_selected_metrics)
    )

    independent_kc, independent_vc, independent_threshold, qbar = _prepare_independent(
        qb, kb, vb, tau=float(config.tau), scale=float(group.scale)
    )
    summary_metrics = {
        "kc": _metrics(kc, independent_kc),
        "vc": _metrics(vc, independent_vc),
        "threshold": _metrics(threshold, independent_threshold),
    }
    k_blocks = int(kc.shape[1])
    traced_routes = _decode_route_trace(sparse_trace, k_blocks)
    all_selected_routes = _decode_route_trace(all_selected_trace, k_blocks)
    expected_all_selected_pairs = int(traced_routes.shape[0]) * int(traced_routes.shape[1]) * k_blocks
    all_selected_pairs = int(all_selected_routes.sum().item())
    all_selected_trace_complete = bool(all_selected_routes.all().item())
    sparse_selected_pairs = int(traced_routes.sum().item())

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
    all_selected_vs_native = _metrics(returned_bthd, native)
    exit_hashes = {
        "q": _tensor_sha256_cpu(qb[0]),
        "k": _tensor_sha256_cpu(kb[0]),
        "v": _tensor_sha256_cpu(vb[0]),
    }
    input_exact_after_sidecars = exit_hashes == preserved_hashes

    record.update(
        {
            "all_selected_output": returned_bthd.detach().to(device="cpu", copy=True),
            "native_sdpa_output": native.detach().to(device="cpu", copy=True),
            "production_sparse_output": sparse_out.detach().to(device="cpu", copy=True),
            "kc": kc.detach().to(device="cpu", copy=True),
            "vc": vc.detach().to(device="cpu", copy=True),
            "threshold": threshold.detach().to(device="cpu", copy=True),
            "route_trace": sparse_trace.detach().to(device="cpu", copy=True),
            "all_selected_route_trace": all_selected_trace.detach().to(device="cpu", copy=True),
            "route_column_means": column_means.detach().to(device="cpu", copy=True),
            "route_margins": margins.detach().to(device="cpu", copy=True),
            "kernel_lse": lse.detach().to(device="cpu", copy=True),
            "preserved_qkv_sha256": preserved_hashes,
            "entry_qkv_sha256": entry_hashes,
            "exit_qkv_sha256": exit_hashes,
            "input_exact_on_entry": input_exact_on_entry,
            "input_exact_after_sidecars": input_exact_after_sidecars,
            "all_selected_vs_native": all_selected_vs_native,
            "all_selected_arithmetic_gate_pass": bool(arithmetic_gate_passes(all_selected_vs_native)),
            "summary_metrics": summary_metrics,
            "route_trace_matches_independent": mismatch_count == 0,
            "route_mismatch_count": mismatch_count,
            "route_mismatch_examples": mismatch_examples,
            "sparse_selected_block_pairs": sparse_selected_pairs,
            "all_selected_selected_block_pairs": all_selected_pairs,
            "all_selected_expected_block_pairs": expected_all_selected_pairs,
            "all_selected_trace_complete": all_selected_trace_complete,
            "frozen_route_reference": frozen,
            "debug_sparse_vs_ordinary": debug_sparse_metrics,
            "lse_specialization_vs_ordinary": lse_sparse_metrics,
            "debug_all_selected_vs_returned": debug_all_selected_metrics,
            "debug_specializations_conform": debug_specializations_conform,
            "packaged_backend": getattr(state.kernel, "backend_name", None),
            "packaged_source_tree_verified": getattr(state.kernel, "source_tree_verified", False),
            "completed": True,
        }
    )
    _append_provenance_once(options)
'''
text = text[:start] + replacement + text[end:]

old = '''        vdn_history_identity._first_high_sol_local_e_v1 = True\n        vdn_history_identity._first_high_sol_local_original = current_vdn\n        interop._vdn_history_identity = vdn_history_identity\n'''
new = '''        vdn_history_identity.__dict__.update(getattr(current_vdn, "__dict__", {}))\n        vdn_history_identity._first_high_sol_local_e_v1 = True\n        vdn_history_identity._first_high_sol_local_original = current_vdn\n        interop._vdn_history_identity = vdn_history_identity\n'''
if old not in text:
    raise SystemExit("history marker anchor not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
