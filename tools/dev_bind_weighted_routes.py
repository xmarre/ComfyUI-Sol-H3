from pathlib import Path


path = Path("sol_h3/runtime.py")
text = path.read_text()
old = '''                if generic_measure_contract is not None:
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
'''
new = '''                if generic_measure_contract is not None:
                    state.eligible_calls += 1
                    preprocess_identity = weighted_measure.preprocess_digest(dense_provider)
                    existing_sink = (0, (int(prefix) + 63) // 64)
                    bind_kwargs = {
                        "block_index": self.index,
                        "layout": layout,
                        "q_rows": int(q.shape[2]),
                        "kv_rows": int(k.shape[2]),
                        "dtype": q.dtype,
                        "device": q.device,
                        "head_dim": int(q.shape[-1]),
                        "existing_sink": existing_sink,
                        "external_sequence": external_contract,
                        "preprocess_identity": preprocess_identity,
                    }
                    if warmup:
                        measure_plan = weighted_measure.prepare(
                            state,
                            current_options,
                            generic_measure_contract,
                            **bind_kwargs,
                            implementation_profile=weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
                            numerical_route=weighted_measure.DENSE_NUMERICAL_ROUTE,
                        )
                    else:
                        measure_plan = weighted_measure.prepare(
                            state,
                            current_options,
                            generic_measure_contract,
                            **bind_kwargs,
                            implementation_profile=weighted_measure.SPARSE_IMPLEMENTATION_PROFILE,
                            numerical_route=weighted_measure.SPARSE_NUMERICAL_ROUTE,
                        )
                    # Bind against the original mixed coordinates, then run the
                    # full-domain preprocessing chain exactly once.
                    q, k, v, dense_provider = _preprocess_chain(dense_provider, q, k, v, heads, kw)

                    def weighted_dense_result(plan, *, output_heads=False, qd=q, kd=k, vd=v):
                        return weighted_measure.dense(
                            qd,
                            kd,
                            vd,
                            heads,
                            plan,
                            scale=kw.get("scale"),
                            output_heads=output_heads,
                        )

                    if warmup:
                        result = weighted_dense_result(measure_plan)
                        state.dense_calls += 1
                        state.external_mixed_weighted_measure_calls += 1
                        state.external_mixed_weighted_measure_q_rows += q.shape[2]
                        state.external_mixed_weighted_measure_kv_rows += k.shape[2]
                        record("dense_warmup", measure_plan=measure_plan)
                        return result

                    def weighted_prefix_dense(qd, kd, vd):
                        out = weighted_dense_result(measure_plan, output_heads=True, qd=qd, kd=kd, vd=vd)
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
                        dense_plan = weighted_measure.prepare(
                            state,
                            current_options,
                            generic_measure_contract,
                            **bind_kwargs,
                            implementation_profile=weighted_measure.DENSE_IMPLEMENTATION_PROFILE,
                            numerical_route=weighted_measure.DENSE_NUMERICAL_ROUTE,
                        )
                        result = weighted_dense_result(dense_plan)
                        state.external_mixed_weighted_measure_calls += 1
                        state.external_mixed_weighted_measure_q_rows += q.shape[2]
                        state.external_mixed_weighted_measure_kv_rows += k.shape[2]
                        record("kernel_unavailable:" + str(exc), True, measure_plan=dense_plan)
                        return result
                    state.external_mixed_sol_calls += 1
                    state.external_mixed_q_rows += q.shape[2]
                    state.external_mixed_kernel_q_rows += q.shape[2]
                    state.external_mixed_weighted_measure_calls += 1
                    state.external_mixed_weighted_measure_q_rows += q.shape[2]
                    state.external_mixed_weighted_measure_kv_rows += k.shape[2]
                    record("sol_external_mixed_weighted_measure", measure_plan=measure_plan)
                    return result
'''
count = text.count(old)
if count != 1:
    raise SystemExit(f"expected one weighted runtime block, found {count}")
path.write_text(text.replace(old, new, 1))
