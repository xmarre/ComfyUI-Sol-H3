from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))


# Runtime: add the generic route alongside, not in place of, the existing
# legacy representative-KV comparator.
p = "sol_h3/runtime.py"
replace_once(
    p,
    "from .mixed_measure import FLOW_MIXED_MEASURE_KEY, reduce_kv, validate_measure_contract\n",
    "from .mixed_measure import FLOW_MIXED_MEASURE_KEY, reduce_kv, validate_measure_contract\n"
    "from . import weighted_measure\n",
)
replace_once(
    p,
    "    external_mixed_measure_removed_rows: int = 0\n"
    "    vdn_local_sol_calls: int = 0\n",
    "    external_mixed_measure_removed_rows: int = 0\n"
    "    external_mixed_weighted_measure_calls: int = 0\n"
    "    external_mixed_weighted_measure_q_rows: int = 0\n"
    "    external_mixed_weighted_measure_kv_rows: int = 0\n"
    "    measure_plans: dict = field(default_factory=dict)\n"
    "    measure_biases: dict = field(default_factory=dict)\n"
    "    vdn_local_sol_calls: int = 0\n",
)
replace_once(
    p,
    '                        "external_mixed_measure_removed_rows": state.external_mixed_measure_removed_rows,\n'
    '                        "compatibility_fallbacks": dict(state.fallbacks),\n',
    '                        "external_mixed_measure_removed_rows": state.external_mixed_measure_removed_rows,\n'
    '                        "external_mixed_weighted_measure_calls": state.external_mixed_weighted_measure_calls,\n'
    '                        "external_mixed_weighted_measure_q_rows": state.external_mixed_weighted_measure_q_rows,\n'
    '                        "external_mixed_weighted_measure_kv_rows": state.external_mixed_weighted_measure_kv_rows,\n'
    '                        "compatibility_fallbacks": dict(state.fallbacks),\n',
)
replace_once(
    p,
    "        def record(route, fallback=False):\n"
    "            routes.append((self.index, route))\n"
    "            receipt(options, self.index, route)\n"
    "            if fallback:\n",
    "        def record(route, fallback=False, measure_plan=None):\n"
    "            routes.append((self.index, route))\n"
    "            receipt(options, self.index, route, measure_plan=measure_plan, call_token=evaluation)\n"
    "            if fallback:\n",
)
replace_once(
    p,
    "                measure_contract = current_options.get(FLOW_MIXED_MEASURE_KEY)\n"
    "                external_mixed = False\n",
    "                legacy_measure_contract = current_options.get(FLOW_MIXED_MEASURE_KEY)\n"
    "                generic_measure_contract = current_options.get(weighted_measure.ATTENTION_MEASURE_KEY)\n"
    "                if legacy_measure_contract is not None and generic_measure_contract is not None:\n"
    "                    raise RuntimeError(\"legacy and generic Mixed-Grid attention-measure contracts cannot coexist\")\n"
    "                external_mixed = False\n",
)
replace_once(
    p,
    "                if measure_contract is not None and (reason is not None or not external_mixed):\n"
    "                    raise RuntimeError(\n"
    '                        "Flow mixed-grid attention-measure contract requires a valid external mixed sequence"\n'
    "                    )\n",
    "                if (legacy_measure_contract is not None or generic_measure_contract is not None) and (\n"
    "                    reason is not None or not external_mixed\n"
    "                ):\n"
    "                    raise RuntimeError(\n"
    '                        "Flow mixed-grid attention-measure contract requires a valid external mixed sequence"\n'
    "                    )\n",
)
marker = '''                measure_validated = None
                measure_stats = None
                if measure_contract is not None:
                    measure_validated = validate_measure_contract(
                        measure_contract,
                        external_contract,
                        q_rows=int(q.shape[2]),
                        kv_rows=int(k.shape[2]),
                    )

                state.eligible_calls += 1
'''
generic = '''                if generic_measure_contract is not None:
                    state.eligible_calls += 1
                    preprocess_identity = weighted_measure.preprocess_digest(dense_provider)
                    existing_sink = (0, (int(prefix) + 63) // 64)
                    measure_plan = weighted_measure.prepare(
                        state,
                        current_options,
                        generic_measure_contract,
                        block_index=self.index,
                        layout=layout,
                        q_rows=int(q.shape[2]),
                        kv_rows=int(k.shape[2]),
                        dtype=q.dtype,
                        device=q.device,
                        head_dim=int(q.shape[-1]),
                        existing_sink=existing_sink,
                        external_sequence=external_contract,
                        preprocess_identity=preprocess_identity,
                    )
                    # Bind against the original mixed coordinates, then run the
                    # full-domain preprocessing chain exactly once.
                    q, k, v, dense_provider = _preprocess_chain(dense_provider, q, k, v, heads, kw)

                    def weighted_dense_result(*, output_heads=False, qd=q, kd=k, vd=v):
                        return weighted_measure.dense(
                            qd,
                            kd,
                            vd,
                            heads,
                            measure_plan,
                            scale=kw.get("scale"),
                            output_heads=output_heads,
                        )

                    if warmup:
                        result = weighted_dense_result()
                        state.dense_calls += 1
                        state.external_mixed_weighted_measure_calls += 1
                        state.external_mixed_weighted_measure_q_rows += q.shape[2]
                        state.external_mixed_weighted_measure_kv_rows += k.shape[2]
                        record("dense_warmup", measure_plan=measure_plan)
                        return result

                    def weighted_prefix_dense(qd, kd, vd):
                        out = weighted_dense_result(output_heads=True, qd=qd, kd=kd, vd=vd)
                        if out.shape != qd.shape:
                            raise RuntimeError("weighted dense prefix returned an invalid output shape")
                        return out.transpose(1, 2)

                    from .sparse import attention, KernelUnavailable

                    try:
                        result = attention(
                            q,
                            k,
                            v,
                            prefix,
                            config,
                            state,
                            dense_attention=weighted_prefix_dense,
                            key_bias=measure_plan.key_log_measure,
                            exact_k_blocks=measure_plan.exact_k_block_range,
                            calibration_identity=measure_plan.semantic_digest,
                        )
                    except KernelUnavailable as exc:
                        result = weighted_dense_result()
                        record("kernel_unavailable:" + str(exc), True, measure_plan=measure_plan)
                        return result
                    state.external_mixed_sol_calls += 1
                    state.external_mixed_q_rows += q.shape[2]
                    state.external_mixed_kernel_q_rows += q.shape[2]
                    state.external_mixed_weighted_measure_calls += 1
                    state.external_mixed_weighted_measure_q_rows += q.shape[2]
                    state.external_mixed_weighted_measure_kv_rows += k.shape[2]
                    record("sol_external_mixed_weighted_measure", measure_plan=measure_plan)
                    return result

                measure_validated = None
                measure_stats = None
                if legacy_measure_contract is not None:
                    measure_validated = validate_measure_contract(
                        legacy_measure_contract,
                        external_contract,
                        q_rows=int(q.shape[2]),
                        kv_rows=int(k.shape[2]),
                    )

                state.eligible_calls += 1
'''
replace_once(p, marker, generic)
replace_once(
    p,
    '    if config.backend == "sol":\n'
    '        to[HISTORY_KEY] = {**to.get(HISTORY_KEY, {}), "sol_h3": HistoryPolicy(config)}\n',
    '    if config.backend == "sol":\n'
    '        to[HISTORY_KEY] = {**to.get(HISTORY_KEY, {}), "sol_h3": HistoryPolicy(config)}\n'
    "        weighted_measure.register(to)\n",
)

# Receipt/history fail closed for the new route until the Spectrum companion
# explicitly learns the richer completed receipt schema.
p = "sol_h3/interop.py"
replace_once(
    p,
    'FLOW_REFINEMENT_KEY = "h3_refinement"\n\n\n'
    "def provider_name(provider):",
    'FLOW_REFINEMENT_KEY = "h3_refinement"\n'
    'ATTENTION_MEASURE_KEY = "attention_measure_v1"\n\n\n'
    "def provider_name(provider):",
)
replace_once(
    p,
    "def receipt(options, block, route):\n"
    "    sink = options.get(RECEIPTS_KEY)\n"
    "    if sink is not None:\n"
    '        sink.append(("sol_h3", block, route))\n',
    "def receipt(options, block, route, *, measure_plan=None, call_token=None):\n"
    "    sink = options.get(RECEIPTS_KEY)\n"
    "    if sink is None:\n"
    "        return\n"
    "    if measure_plan is None:\n"
    '        sink.append(("sol_h3", block, route))\n'
    "        return\n"
    "    from .weighted_measure import receipt_fields\n"
    '    sink.append(("sol_h3", block, route, receipt_fields(measure_plan, call_token=call_token)))\n',
)
replace_once(
    p,
    '        from .runtime import _REQUEST\n'
    "        state = _REQUEST.get()\n"
    "        if state is None:\n"
    "            return None\n"
    '        replacements = options.get("patches_replace", {}).get("dit", {})\n',
    '        from .runtime import _REQUEST\n'
    "        state = _REQUEST.get()\n"
    "        if state is None:\n"
    "            return None\n"
    "        # Generic weighted measure becomes forecastable only after Spectrum\n"
    "        # binds its completed-receipt identity companion.\n"
    "        if options.get(ATTENTION_MEASURE_KEY) is not None:\n"
    "            return None\n"
    '        replacements = options.get("patches_replace", {}).get("dit", {})\n',
)

# Sparse bridge: separate dense-query prefix ownership from exact-KV coverage,
# and make weighted arithmetic calibration measure-specific.
p = "sol_h3/sparse.py"
replace_once(
    p,
    '''def _sink_blocks(start, tokens, rows):
    """Validate an exact prefix interval and describe its overlapping 64-row blocks.

    Sol-H3 only requests prefix sinks beginning at row zero. The final partial
    block is intentionally rounded outward: this makes a few extra keys exact,
    which is semantics-preserving and only slightly more expensive.
    """
    if type(start) is not int or type(tokens) is not int or type(rows) is not int:
        raise RuntimeError("SOL sink geometry must use integer row counts")
    if start != 0:
        raise RuntimeError("Sol-H3 currently supports only a prefix sink beginning at row zero")
    if tokens < 0 or tokens > rows:
        raise RuntimeError("SOL sink row count is outside the current sequence")
    if tokens == 0:
        return [0, 0]
    return [0, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE]
''',
    '''def _sink_blocks(start, tokens, rows):
    """Validate one exact K interval and return its overlapping 64-row blocks."""
    if type(start) is not int or type(tokens) is not int or type(rows) is not int:
        raise RuntimeError("SOL sink geometry must use integer row counts")
    if start < 0 or tokens < 0 or rows < 0 or start > rows or start + tokens > rows:
        raise RuntimeError("SOL sink row range is outside the current sequence")
    if tokens == 0:
        block = start // BLOCK_SIZE
        return [block, block]
    return [start // BLOCK_SIZE, (start + tokens + BLOCK_SIZE - 1) // BLOCK_SIZE]
''',
)
replace_once(
    p,
    '''    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0):
        _sink_blocks(sink_start, sink_tokens, k.shape[1])  # validate prefix geometry
        return sol_attn(q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
                        thresh_type=thresh_type, kv_splits=kv_splits,
                        sink_start=sink_start, sink_tokens=sink_tokens)
''',
    '''    def kernel(q, k, v, *, tau, thresh_type="diag", kv_splits=1,
               sink_start=0, sink_tokens=0, key_bias=None):
        _sink_blocks(sink_start, sink_tokens, k.shape[1])
        return sol_attn(q, k, v, scale=q.shape[-1] ** -0.5, tau=float(tau),
                        thresh_type=thresh_type, kv_splits=kv_splits,
                        sink_start=sink_start, sink_tokens=sink_tokens,
                        key_bias=key_bias)
''',
)
replace_once(
    p,
    '''def _dense_reference(q, k, v, dense_attention):
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
''',
    '''def _dense_reference(q, k, v, dense_attention, key_bias=None):
    if dense_attention is not None:
        out = dense_attention(q, k, v)
        expected = (q.shape[0], q.shape[2], q.shape[1], q.shape[3])
        if out.shape != expected:
            raise RuntimeError(f"Dense SOL reference returned {tuple(out.shape)}, expected {expected}")
        return out
    bias = None if key_bias is None else key_bias.view(1, 1, 1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2)
''',
)
replace_once(
    p,
    "def attention(q, k, v, prefix, config, state, dense_attention=None,\n"
    "              recompute_prefix_queries=True):\n",
    "def attention(q, k, v, prefix, config, state, dense_attention=None,\n"
    "              recompute_prefix_queries=True, key_bias=None, exact_k_blocks=None,\n"
    "              calibration_identity=None):\n",
)
replace_once(
    p,
    '    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):\n'
    '        raise RuntimeError("SOL requires BF16 QKV on the same device")\n'
    "    kernel_loader_s = 0.0\n",
    '    if any(x.dtype != torch.bfloat16 or x.device != q.device for x in (q, k, v)):\n'
    '        raise RuntimeError("SOL requires BF16 QKV on the same device")\n'
    "    if key_bias is not None:\n"
    "        if (not torch.is_tensor(key_bias) or key_bias.ndim != 1\n"
    "                or key_bias.shape[0] != k.shape[2] or key_bias.device != q.device\n"
    "                or key_bias.dtype != torch.float32 or not key_bias.is_contiguous()):\n"
    '            raise RuntimeError("weighted SOL requires contiguous FP32 key_bias with one value per K row")\n'
    "        if not isinstance(calibration_identity, str) or not calibration_identity:\n"
    '            raise RuntimeError("weighted SOL requires a semantic calibration identity")\n'
    "        if (not isinstance(exact_k_blocks, (tuple, list)) or len(exact_k_blocks) != 2\n"
    "                or any(type(x) is not int for x in exact_k_blocks)):\n"
    '            raise RuntimeError("weighted SOL requires one exact K block interval")\n'
    "        max_blocks = (k.shape[2] + BLOCK_SIZE - 1) // BLOCK_SIZE\n"
    "        if exact_k_blocks[0] < 0 or exact_k_blocks[1] < exact_k_blocks[0] or exact_k_blocks[1] > max_blocks:\n"
    '            raise RuntimeError("weighted SOL exact K block interval is out of range")\n'
    "    elif exact_k_blocks is not None or calibration_identity is not None:\n"
    '        raise RuntimeError("weighted SOL routing metadata was supplied without key_bias")\n'
    "    kernel_loader_s = 0.0\n",
)
replace_once(
    p,
    "    key = (q.device, q.dtype, layout_key)\n"
    "    calibrating = key not in state.sparse_verified\n",
    "    key = (q.device, q.dtype, layout_key, calibration_identity)\n"
    "    calibrating = key not in state.sparse_verified\n",
)
replace_once(
    p,
    '''        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=kb.shape[1])
        want = _dense_reference(q, k, v, None)
''',
    '''        got = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                           sink_start=0, sink_tokens=kb.shape[1], key_bias=key_bias)
        want = _dense_reference(q, k, v, None, key_bias=key_bias)
''',
)
replace_once(
    p,
    '''    out = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                       sink_start=0, sink_tokens=prefix)
''',
    '''    if exact_k_blocks is None:
        sink_start, sink_tokens = 0, prefix
    else:
        sink_start = exact_k_blocks[0] * BLOCK_SIZE
        sink_end = min(k.shape[2], exact_k_blocks[1] * BLOCK_SIZE)
        sink_tokens = max(0, sink_end - sink_start)
    out = state.kernel(qb, kb, vb, tau=config.tau, thresh_type="diag", kv_splits=1,
                       sink_start=sink_start, sink_tokens=sink_tokens, key_bias=key_bias)
''',
)
replace_once(
    p,
    "        out[:, :prefix] = _dense_reference(q[:, :, :prefix], k, v, dense_attention)\n",
    "        out[:, :prefix] = _dense_reference(\n"
    "            q[:, :, :prefix], k, v, dense_attention, key_bias=key_bias\n"
    "        )\n",
)

# Vendored public interface: weighted bias is SM120-only and appended to the API
# so existing positional callers retain their contract.
p = "sol_h3/_vendor/sol_attn/interface.py"
replace_once(p, "import functools\n\nimport torch\n", "import functools\nimport math\n\nimport torch\n")
replace_once(
    p,
    '''def _compile_sm120(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
''',
    '''def _compile_sm120(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
    key_bias_enabled,
):
''',
)
replace_once(p, "    operator = make_kernel()\n", "    operator = make_kernel(key_bias_enabled=key_bias_enabled)\n")
replace_once(
    p,
    "    sink_start,\n"
    "    valid_tokens=None,\n"
    "):\n",
    "    sink_start,\n"
    "    valid_tokens=None,\n"
    "    key_bias=None,\n"
    "):\n",
)
replace_once(
    p,
    "        key = (q.device.index, arch, batch, capacity_tokens, k.shape[1], heads, kv_splits, layout_key)\n",
    "        key = (\n"
    "            q.device.index, arch, batch, capacity_tokens, k.shape[1], heads, kv_splits,\n"
    "            layout_key, key_bias is not None,\n"
    "        )\n",
)
replace_once(
    p,
    '''            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm120(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
''',
    '''            key_bias_arg = key_bias if key_bias is not None else threshold
            tensors = [q, k, v, output, kc, vc, threshold, key_bias_arg, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm120(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                    key_bias is not None,
                )
''',
)
replace_once(
    p,
    "    sink_start: int | None = None,\n"
    "    compile_bucket_size: int | None = None,\n"
    ") -> torch.Tensor:\n",
    "    sink_start: int | None = None,\n"
    "    compile_bucket_size: int | None = None,\n"
    "    key_bias: torch.Tensor | None = None,\n"
    ") -> torch.Tensor:\n",
)
replace_once(
    p,
    '    backend = _backend_for_arch(arch)\n'
    '    if q.shape[1] != k.shape[1] and backend != "cute_sm120":\n',
    '    backend = _backend_for_arch(arch)\n'
    "    if key_bias is not None:\n"
    '        if backend != "cute_sm120":\n'
    '            raise ValueError("key_bias is currently supported by cute_sm120 only")\n'
    "        if (not torch.is_tensor(key_bias) or key_bias.ndim != 1\n"
    "                or key_bias.shape[0] != k.shape[1] or key_bias.device != q.device\n"
    "                or key_bias.dtype != torch.float32 or not key_bias.is_contiguous()):\n"
    '            raise ValueError("key_bias must be contiguous float32 with one natural-log value per K/V row")\n'
    "        weighted_scale = q.shape[-1] ** -0.5 if scale is None else float(scale)\n"
    "        if not math.isfinite(weighted_scale) or weighted_scale <= 0.0:\n"
    '            raise ValueError("weighted Sol-Attn requires a finite positive attention scale")\n'
    '    if q.shape[1] != k.shape[1] and backend != "cute_sm120":\n',
)
replace_once(
    p,
    "        valid_tokens=valid_tokens,\n"
    "    )\n",
    "        valid_tokens=valid_tokens,\n"
    "        key_bias=key_bias,\n"
    "    )\n",
)

# SM120 recipe and exact-score bias injection. Natural-log key bias is added as
# b/scale before the existing helper applies scale*log2(e), so it contributes
# b*log2(e) exactly once to the exponent.
p = "sol_h3/_vendor/sol_attn/sm120/kernel.py"
replace_once(
    p,
    "    prefetch_next_route_k: bool = True,\n"
    "):\n",
    "    prefetch_next_route_k: bool = True,\n"
    "    key_bias_enabled: bool = False,\n"
    "):\n",
)
replace_once(
    p,
    "        prefetch_next_route_k=prefetch_next_route_k,\n"
    "    )\n",
    "        prefetch_next_route_k=prefetch_next_route_k,\n"
    "        key_bias_enabled=key_bias_enabled,\n"
    "    )\n",
)

p = "sol_h3/_vendor/sol_attn/sm120/mainloop.py"
replace_once(
    p,
    "        prefetch_next_route_k: bool = True,\n"
    "    ):\n",
    "        prefetch_next_route_k: bool = True,\n"
    "        key_bias_enabled: bool = False,\n"
    "    ):\n",
)
replace_once(
    p,
    "        self.prefetch_next_route_k = prefetch_next_route_k\n",
    "        self.prefetch_next_route_k = prefetch_next_route_k\n"
    "        self.key_bias_enabled = key_bias_enabled\n",
)
replace_once(
    p,
    "        mThreshold: cute.Tensor,\n"
    "        mLSE: cute.Tensor,\n",
    "        mThreshold: cute.Tensor,\n"
    "        mKeyBias: cute.Tensor,\n"
    "        mLSE: cute.Tensor,\n",
)
replace_once(
    p,
    "                mask_exact_scores(tSrS, tScS, block_len, q_len)\n"
    "                row_scale = online_softmax(\n",
    "                mask_exact_scores(tSrS, tScS, block_len, q_len)\n"
    "                if cutlass.const_expr(self.key_bias_enabled):\n"
    "                    add_exact_key_bias(\n"
    "                        tSrS, tScS, mKeyBias, exact_block, block_len, q_len,\n"
    "                        cutlass.Float32(1.4426950408889634) / scale_softmax_log2e,\n"
    "                    )\n"
    "                row_scale = online_softmax(\n",
)
replace_once(
    p,
    "        threshold: cute.Tensor,\n"
    "        lse: cute.Tensor,\n",
    "        threshold: cute.Tensor,\n"
    "        key_bias: cute.Tensor,\n"
    "        lse: cute.Tensor,\n",
)
replace_once(
    p,
    "            threshold,\n"
    "            lse_target,\n"
    "            tma_atom_Q,\n",
    "            threshold,\n"
    "            key_bias,\n"
    "            lse_target,\n"
    "            tma_atom_Q,\n",
)
insertion = '''

@cute.jit
def add_exact_key_bias(
    scores: cute.Tensor,
    coords: cute.Tensor,
    key_bias: cute.Tensor,
    exact_block: cutlass.Int32,
    block_len: cutlass.Int32,
    q_len: cutlass.Int32,
    inv_softmax_scale: cutlass.Float32,
):
    """Add natural-log key measure as b/scale to valid exact raw QK scores."""
    scores_mn = layout_utils.reshape_acc_to_mn(scores)
    coords_mn = layout_utils.reshape_acc_to_mn(coords)
    for m in cutlass.range_constexpr(cute.size(scores_mn, mode=[0])):
        valid_row = coords_mn[m, 0][0] < q_len
        for n in cutlass.range_constexpr(cute.size(scores_mn, mode=[1])):
            local_col = coords_mn[m, n][1]
            if valid_row and local_col < block_len:
                absolute_col = exact_block * cutlass.Int32(N) + local_col
                scores_mn[m, n] = (
                    cutlass.Float32(scores_mn[m, n])
                    + cutlass.Float32(key_bias[absolute_col]) * inv_softmax_scale
                )
'''
anchor = "\n\n@cute.jit\ndef online_softmax(\n"
text = Path(p).read_text()
if text.count(anchor) != 1:
    raise SystemExit(f"{p}: online_softmax insertion anchor mismatch")
Path(p).write_text(text.replace(anchor, insertion + anchor, 1))

# Provenance profile moves from rectangular-only v2 to a reproducible
# rectangular + weighted-SM120 v3 patch.
p = "tools/vendor_sol_attn.py"
replace_once(p, "['rectangular-sm120-v2']", "['rectangular-sm120-v3']")
replace_once(
    p,
    "'relative-imports-v1; rectangular-sm120-v2 (tools/rectangular_sm120.patch)'",
    "'relative-imports-v1; rectangular-sm120-v3 (tools/rectangular_sm120.patch)'",
)
p = "sol_h3/provenance.py"
replace_once(
    p,
    "CONTRACT = 'sana-sol-engine-sol-attn-64-rect-sm120-v2'",
    "CONTRACT = 'sana-sol-engine-sol-attn-64-rect-sm120-v3'",
)

# Tests: preserve old behavior and add weighted calibration/sink semantics plus
# public-interface fail-closed checks.
p = "tests/test_sparse.py"
replace_once(
    p,
    '    with pytest.raises(RuntimeError, match="beginning at row zero"):\n'
    "        sparse._sink_blocks(64, 64, 130)\n",
    "    assert sparse._sink_blocks(64, 64, 130) == [1, 2]\n",
)
Path(p).write_text(
    Path(p).read_text()
    + r'''


def test_weighted_bridge_uses_measure_specific_gate_and_exact_k_range(monkeypatch):
    calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        bias = kw.get("key_bias")
        mask = None if bias is None else bias.view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q, k, v = (torch.randn(1, 2, 260, 128).to(torch.bfloat16) for _ in range(3))
    bias = torch.zeros(260, dtype=torch.float32)
    bias[64:192] = torch.log(torch.tensor(0.5))
    state = Request(Config(exact=False, backend="sol"))
    out = sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=bias,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-a",
    )
    assert out.shape == (1, 260, 256)
    assert [c["sink_tokens"] for c in calls] == [260, 192]
    assert all(c["sink_start"] == 0 for c in calls)
    assert all(c["key_bias"] is bias for c in calls)
    assert len(state.gates) == 1

    sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=bias,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-a",
    )
    assert len(calls) == 3
    assert len(state.gates) == 1

    other = bias.clone()
    other[64:192] = torch.log(torch.tensor(0.25))
    sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=other,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-b",
    )
    assert len(calls) == 5
    assert len(state.gates) == 2


def test_weighted_bridge_rejects_bias_without_bound_route_metadata(monkeypatch):
    monkeypatch.setattr(sparse, "load_kernel", lambda device: None)
    q = torch.ones(1, 1, 65, 128, dtype=torch.bfloat16)
    state = Request(Config(exact=False, backend="sol"))
    with pytest.raises(RuntimeError, match="calibration identity"):
        sparse.attention(q, q, q, 1, state.config, state, key_bias=torch.zeros(65))
'''
)

p = "tests/test_sana_contract.py"
replace_once(
    p,
    "        'sink_tokens', 'sink_start', 'compile_bucket_size')\n",
    "        'sink_tokens', 'sink_start', 'compile_bucket_size', 'key_bias')\n",
)
Path(p).write_text(
    Path(p).read_text()
    + r'''


def test_weighted_public_api_is_sm120_only_and_validates_bias(monkeypatch):
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    monkeypatch.setattr(interface, '_validate_inputs', lambda *a, **k: (10, 0))
    monkeypatch.setattr(interface, '_cute_runtime_available', lambda: True)
    with pytest.raises(ValueError, match='cute_sm120 only'):
        interface.sol_attn(q, q, q, key_bias=torch.zeros(65))

    monkeypatch.setattr(interface, '_validate_inputs', lambda *a, **k: (12, 0))
    with pytest.raises(ValueError, match='one natural-log value'):
        interface.sol_attn(q, q, q, key_bias=torch.zeros(64))
    with pytest.raises(ValueError, match='finite positive'):
        interface.sol_attn(q, q, q, scale=0.0, key_bias=torch.zeros(65))


def test_sm120_source_injects_bias_only_between_exact_mask_and_softmax():
    source = (Path(interface.__file__).parent / 'sm120' / 'mainloop.py').read_text()
    exact = source.index('mask_exact_scores(tSrS, tScS, block_len, q_len)')
    inject = source.index('add_exact_key_bias(', exact)
    softmax = source.index('row_scale = online_softmax(', inject)
    assert exact < inject < softmax
    assert 'key_bias[absolute_col]' in source
'''
)
