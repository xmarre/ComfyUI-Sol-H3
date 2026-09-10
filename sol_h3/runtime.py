from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import json
import logging

from .attention_measure import ATTENTION_MEASURE_KEY
from .contracts import KEY, Config, adaln_status, prefix_length
from .mixed_measure import FLOW_MIXED_MEASURE_KEY, reduce_kv, validate_measure_contract
from .interop import (
    HISTORY_KEY,
    VDN_KEY,
    VDN_KEY_V2,
    VDN_KEY_V3,
    VDN_PREPROCESS_KEY,
    HistoryPolicy,
    dense_evaluation_warmup,
    provider_identity,
    provider_name,
    receipt,
)

log = logging.getLogger("comfy.sol_h3")
_REQUEST = ContextVar("sol_h3_request", default=None)
_FORWARD = ContextVar("sol_h3_forward", default=None)


@dataclass
class Request:
    config: Config
    evaluations: int = 0
    eligible_calls: int = 0
    sparse_calls: int = 0
    dense_calls: int = 0
    external_mixed_sol_calls: int = 0
    external_mixed_q_rows: int = 0
    external_mixed_kernel_q_rows: int = 0
    external_mixed_measure_calls: int = 0
    external_mixed_measure_q_rows: int = 0
    external_mixed_measure_kv_rows_before: int = 0
    external_mixed_measure_kv_rows_after: int = 0
    external_mixed_measure_removed_rows: int = 0
    external_mixed_weighted_calls: int = 0
    external_mixed_weighted_q_rows: int = 0
    external_mixed_weighted_kv_rows: int = 0
    external_mixed_weighted_exact_k_rows: int = 0
    vdn_local_sol_calls: int = 0
    vdn_rectangular_sol_calls: int = 0
    vdn_requested_q_rows: int = 0
    vdn_kernel_q_rows: int = 0
    vdn_square_expanded_calls: int = 0
    vdn_square_requested_rows: int = 0
    vdn_square_kernel_rows: int = 0
    exact_blocks: int = 0
    fallbacks: Counter = field(default_factory=Counter)
    dense_provider_failures: Counter = field(default_factory=Counter)
    disabled_dense_providers: set = field(default_factory=set)
    exact_verified: set = field(default_factory=set)
    sparse_verified: set = field(default_factory=set)
    dense_attention_backends: set = field(default_factory=set)
    gates: list = field(default_factory=list)
    kernel: object = None
    kernel_device: object = None
    native_verified: bool = False
    native_reason: str | None = None
    last_routes: tuple | None = None
    backend_transitions: int = 0


@dataclass(frozen=True)
class SamplingWrapper:
    config: Config

    def __call__(self, executor, *args, **kwargs):
        state = Request(self.config)
        token = _REQUEST.set(state)
        success = False
        try:
            result = executor(*args, **kwargs)
            success = True
            return result
        finally:
            _REQUEST.reset(token)
            log.info(
                "Sol-H3 %s",
                json.dumps(
                    {
                        "success": success,
                        **self.config.metadata(),
                        "actual_evaluations": state.evaluations,
                        "sol_eligible_calls": state.eligible_calls,
                        "sol_backend": getattr(state.kernel, "backend_name", None),
                        "sol_source_tree_verified": getattr(state.kernel, "source_tree_verified", False),
                        "sparse_calls": state.sparse_calls,
                        "dense_warmup": state.dense_calls,
                        "external_mixed_sol_calls": state.external_mixed_sol_calls,
                        "external_mixed_q_rows": state.external_mixed_q_rows,
                        "external_mixed_kernel_q_rows": state.external_mixed_kernel_q_rows,
                        "external_mixed_measure_calls": state.external_mixed_measure_calls,
                        "external_mixed_measure_q_rows": state.external_mixed_measure_q_rows,
                        "external_mixed_measure_kv_rows_before": state.external_mixed_measure_kv_rows_before,
                        "external_mixed_measure_kv_rows_after": state.external_mixed_measure_kv_rows_after,
                        "external_mixed_measure_removed_rows": state.external_mixed_measure_removed_rows,
                        "external_mixed_weighted_calls": state.external_mixed_weighted_calls,
                        "external_mixed_weighted_q_rows": state.external_mixed_weighted_q_rows,
                        "external_mixed_weighted_kv_rows": state.external_mixed_weighted_kv_rows,
                        "external_mixed_weighted_exact_k_rows": state.external_mixed_weighted_exact_k_rows,
                        "compatibility_fallbacks": dict(state.fallbacks),
                        "dense_provider_failures": dict(state.dense_provider_failures),
                        "vdn_local_sol_calls": state.vdn_local_sol_calls,
                        "vdn_rectangular_sol_calls": state.vdn_rectangular_sol_calls,
                        "vdn_requested_q_rows": state.vdn_requested_q_rows,
                        "vdn_kernel_q_rows": state.vdn_kernel_q_rows,
                        "vdn_square_expanded_calls": state.vdn_square_expanded_calls,
                        "vdn_square_requested_rows": state.vdn_square_requested_rows,
                        "vdn_square_kernel_rows": state.vdn_square_kernel_rows,
                        "numerical_backend_transitions": state.backend_transitions,
                        "exact_blocks": state.exact_blocks,
                        "inherited_dense_backends": sorted(state.dense_attention_backends),
                        "arithmetic_gates": state.gates,
                    }
                ),
            )


@dataclass(frozen=True)
class DiffusionWrapper:
    config: Config

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        if options.get(KEY) != self.config.metadata():
            raise RuntimeError("Sol-H3 runtime metadata changed after installation")
        state = _REQUEST.get()
        if state is None or state.config != self.config:
            raise RuntimeError("Sol-H3 must execute inside its native OUTER_SAMPLE lifecycle")
        model = executor.class_obj
        seen = set()
        routes = []
        token = _FORWARD.set((model, state, state.evaluations, seen, routes))
        try:
            result = executor(x, timestep, context, options, **kwargs)
            if seen:
                identity = tuple(routes)
                if state.last_routes is not None and identity != state.last_routes:
                    state.backend_transitions += 1
                state.last_routes = identity
            return result
        finally:
            _FORWARD.reset(token)


def _shape_reason(q, k, v, heads, mask, kw, *, rectangular=False):
    import torch

    if mask is not None:
        return "attention_mask"
    if not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
        return "attention_layout_flags"
    allowed = {"skip_reshape", "skip_output_reshape", "transformer_options", "_inside_attn_wrapper", "attn_precision"}
    if set(kw) - allowed or kw.get("attn_precision") is not None:
        return "attention_arguments"
    if (
        any(t.ndim != 4 for t in (q, k, v))
        or k.shape != v.shape
        or q.shape[:2] != k.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or q.shape[0] != 1
        or min(q.shape[2], k.shape[2]) == 0
    ):
        return "qkv_geometry"
    if not rectangular and q.shape != k.shape:
        return "non_square_qkv"
    if q.shape[1] != heads or q.shape[-1] != 128:
        return "head_geometry"
    if any(t.dtype != torch.bfloat16 or t.device != q.device for t in (q, k, v)):
        return "qkv_dtype_device"
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) != (12, 0):
        return "device_not_sm120"
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return "autograd_or_graph_capture"
    return None


def _preprocess_chain(provider, q, k, v, heads, kw):
    """Apply only explicit QKV transforms and return the remaining dense leaf."""
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            raise RuntimeError("Cyclic attention preprocessing contract")
        seen.add(id(provider))
        transform, provider = provider.attention_preprocess_v1
        shapes = (q.shape, k.shape, v.shape)
        dtypes = (q.dtype, k.dtype, v.dtype)
        devices = (q.device, k.device, v.device)
        q, k, v = transform(q, k, v, heads, **kw)
        if (q.shape, k.shape, v.shape) != shapes:
            raise RuntimeError("Attention preprocessing changed its promised QKV topology")
        if (q.dtype, k.dtype, v.dtype) != dtypes or (q.device, k.device, v.device) != devices:
            raise RuntimeError("Attention preprocessing changed its promised QKV dtype/device")
    return q, k, v, provider


def _preprocess_identity(provider):
    """Describe the exact full-domain QKV preprocessing owner chain without running it."""
    seen = set()
    identity = []
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            raise RuntimeError("Cyclic attention preprocessing contract")
        seen.add(id(provider))
        transform, provider = provider.attention_preprocess_v1
        identity.append((provider_name(transform), id(transform)))
    return tuple(identity)


def _provider_leaf(provider):
    """Return the dense leaf without executing preprocessing transforms."""
    seen = set()
    while hasattr(provider, "attention_preprocess_v1"):
        if id(provider) in seen:
            raise RuntimeError("Cyclic attention preprocessing contract")
        seen.add(id(provider))
        _, provider = provider.attention_preprocess_v1
    return provider


def _provider_unavailable_reason(exc):
    """Classify optional-provider loader failures; never absorb compute errors."""
    if not isinstance(exc, (ImportError, OSError)):
        return None
    message = str(exc)
    if "GLIBCXX_" in message:
        return "binary_abi"
    if "undefined symbol" in message:
        return "binary_symbol"
    if "cannot open shared object file" in message or "No such file or directory" in message:
        return "binary_dependency"
    return "import_error" if isinstance(exc, ImportError) else "loader_error"


def _external_sequence_prefix(contract, layout, rows):
    """Validate Flow mixed-grid API 2 and return its exact global-prefix sink rows."""
    if not isinstance(contract, dict) or contract.get("api") != 2:
        return None, "external_sequence_native"
    if contract.get("mode") != "dense_gate_no_linear" or contract.get("topology") != "mixed_grid_low_suffix":
        return None, "external_sequence_native"
    names = (
        "native_sequence_rows",
        "sequence_rows",
        "video_start",
        "temporal",
        "prefix_t",
        "source_rows_per_frame",
        "prefix_rows_per_frame",
    )
    if any(type(contract.get(name)) is not int for name in names):
        return None, "external_sequence_contract"
    native, actual, start, temporal, prefix_t, source_rows, prefix_rows = (contract[name] for name in names)
    if (
        actual != rows
        or not 0 < start < actual
        or not 0 < prefix_t < temporal
        or not 0 < source_rows < prefix_rows
        or native != start + temporal * source_rows
        or actual != start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows
    ):
        return None, "external_sequence_contract"

    # Flow's mixed-grid wrapper deliberately keeps the native low-grid carrier
    # layout in the model payload while each wrapped transformer block executes
    # a larger mixed hidden stream. Depending on wrapper ordering, Sol-H3 may
    # therefore see either the mixed block layout (`actual`) or the native
    # carrier layout (`native`). Both are valid evidence as long as the packed
    # prefix agrees with the explicit API-2 contract.
    layout_rows = getattr(layout, "seq_len", None)
    if layout_rows not in {native, actual}:
        return None, "external_sequence_layout"
    try:
        current_prefix = prefix_length(layout, layout_rows)
    except RuntimeError:
        return None, "external_sequence_layout"
    if current_prefix != start:
        return None, "external_sequence_layout"
    return start, None


def _vdn_provider_api():
    """Return the installed VDN softmax-provider capability without owning hybrid.py."""
    try:
        from vdn_h3.softmax_provider import PROVIDER_API_VERSION
    except (ImportError, AttributeError):
        return None
    try:
        return int(PROVIDER_API_VERSION)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class BlockPatch:
    index: int
    config: Config
    previous: object = None

    def __call__(self, args, extra):
        active = _FORWARD.get()
        if active is None:
            raise RuntimeError("Sol-H3 block called outside its diffusion scope")
        model, state, evaluation, seen, routes = active
        config = state.config
        if not seen:
            state.evaluations += 1
        seen.add(self.index)
        options = dict(args["transformer_options"])
        forwarded = {**args, "transformer_options": options}
        route_start = len(routes)

        def record(route, fallback=False, evidence=None):
            routes.append((self.index, route))
            receipt(options, self.index, route, evidence=evidence)
            if fallback:
                state.fallbacks[route] += 1

        if config.backend == "sol":
            previous = options.get("optimized_attention_override")
            warmup = dense_evaluation_warmup(config, evaluation, options) or self.index < config.dense_layers

            def override(original, q, k, v, heads, mask=None, **kw):
                dense_provider = previous

                def dense(qd=q, kd=k, vd=v, *, output_heads=None):
                    dense_kw = {**kw, "_inside_attn_wrapper": True}
                    if output_heads is not None:
                        dense_kw["skip_reshape"] = True
                        dense_kw["skip_output_reshape"] = output_heads

                    provider = dense_provider
                    q_call, k_call, v_call = qd, kd, vd
                    leaf = _provider_leaf(provider) if provider is not None else None
                    if provider is not None and hasattr(provider, "attention_preprocess_v1"):
                        q_call, k_call, v_call, leaf = _preprocess_chain(
                            provider, q_call, k_call, v_call, heads, dense_kw
                        )
                        provider = leaf
                    if leaf is not None and id(leaf) in state.disabled_dense_providers:
                        provider = None

                    if provider is not None:
                        try:
                            out = provider(original, q_call, k_call, v_call, heads, mask=mask, **dense_kw)
                        except (ImportError, OSError) as exc:
                            reason = _provider_unavailable_reason(exc)
                            if reason is None:
                                raise
                            name = provider_name(leaf)
                            first_failure = id(leaf) not in state.disabled_dense_providers
                            state.disabled_dense_providers.add(id(leaf))
                            state.dense_provider_failures[f"{name}:{reason}"] += 1
                            state.fallbacks[f"dense_provider_unavailable:{reason}"] += 1
                            if first_failure:
                                state.backend_transitions += 1
                                log.warning(
                                    "Sol-H3 inherited dense provider %s unavailable (%s); "
                                    "demoting it to the original Comfy attention for this request: %s",
                                    name,
                                    reason,
                                    exc,
                                )
                        else:
                            state.dense_attention_backends.add(provider_name(leaf))
                            return out

                    if original is None:
                        raise RuntimeError(
                            "Inherited dense provider is unavailable and no original attention fallback was supplied"
                        )
                    out = original(q_call, k_call, v_call, heads, mask=mask, **dense_kw)
                    state.dense_attention_backends.add(provider_name(original))
                    return out

                current_options = kw.get("transformer_options") or options
                reason = _shape_reason(q, k, v, heads, mask, kw)
                if current_options.get("minimax_h3_untwist_rope", {}).get("enabled") and not hasattr(
                    previous, "attention_preprocess_v1"
                ):
                    reason = "inherited_key_transform"
                consumer = current_options.get("spectrum_h3_runtime")
                if consumer is not None and not callable(getattr(consumer, "prepare_backend_history", None)):
                    reason = "forecast_consumer_no_history_contract"
                layout = current_options.get("minimax_h3_layout", args.get("layout"))
                external_contract = current_options.get("vdn_h3_external_sequence_v1")
                legacy_measure_contract = current_options.get(FLOW_MIXED_MEASURE_KEY)
                generic_measure_request = current_options.get(ATTENTION_MEASURE_KEY)
                external_mixed = False
                prefix = 0
                if reason is None and external_contract is not None:
                    prefix, reason = _external_sequence_prefix(external_contract, layout, q.shape[2])
                    external_mixed = reason is None
                elif reason is None:
                    try:
                        prefix = prefix_length(layout, q.shape[2])
                    except RuntimeError:
                        reason = "packed_layout_not_representable"

                if legacy_measure_contract is not None and generic_measure_request is not None:
                    raise RuntimeError("Flow cannot publish legacy and generic Mixed-Grid measure contracts together")
                # Both measure contracts are meaningful only on a fully validated
                # external mixed stream. Never silently drop or reinterpret either
                # request as the old square path.
                if (
                    (legacy_measure_contract is not None or generic_measure_request is not None)
                    and (reason is not None or not external_mixed)
                ):
                    raise RuntimeError(
                        "Flow mixed-grid attention measure requires a valid external mixed sequence"
                    )
                if reason is not None:
                    record(reason, True)
                    return dense()

                legacy_measure = None
                measure_stats = None
                generic_measure = None
                preprocess_id = _preprocess_identity(dense_provider)
                if generic_measure_request is not None:
                    from . import attention_measure as sol_measure

                    # Generic measure keeps every Q/K/V row. Full-domain key
                    # transforms execute once before provider binding/routing.
                    q, k, v, dense_provider = _preprocess_chain(
                        dense_provider, q, k, v, heads, kw
                    )
                    generic_measure = sol_measure.prepare(
                        current_options,
                        generic_measure_request,
                        owner=self,
                        block_index=self.index,
                        layout=layout,
                        q_rows=int(q.shape[2]),
                        kv_rows=int(k.shape[2]),
                        dtype=q.dtype,
                        device=q.device,
                        head_dim=int(q.shape[-1]),
                        existing_sink=(0, (int(prefix) + 63) // 64),
                        external_sequence=external_contract,
                        preprocess_identity=preprocess_id,
                        dense=warmup,
                    )
                    if warmup:
                        from comfy import attention_measure as core_measure

                        state.dense_calls += 1
                        result = core_measure.weighted_dense(
                            q,
                            k,
                            v,
                            heads,
                            key_bias=generic_measure.key_log_measure,
                            mask=None,
                            skip_reshape=True,
                            skip_output_reshape=False,
                        )
                        record(
                            "dense_warmup_weighted",
                            evidence=generic_measure.receipt_fields(),
                        )
                        return result
                elif legacy_measure_contract is not None:
                    legacy_measure = validate_measure_contract(
                        legacy_measure_contract,
                        external_contract,
                        q_rows=int(q.shape[2]),
                        kv_rows=int(k.shape[2]),
                    )

                state.eligible_calls += 1
                if legacy_measure is not None:
                    # Legacy representative mode retains its existing behavior:
                    # preprocess once on the full mixed coordinates, then gather
                    # representative K/V rows. New generic weighted mode never
                    # enters this branch.
                    q, k, v, dense_provider = _preprocess_chain(dense_provider, q, k, v, heads, kw)
                    k, v, measure_stats = reduce_kv(k, v, legacy_measure)
                    if warmup:
                        state.dense_calls += 1
                        record("dense_warmup")
                        return dense(q, k, v)
                elif generic_measure is None:
                    if warmup:
                        state.dense_calls += 1
                        record("dense_warmup")
                        return dense()
                    q, k, v, dense_provider = _preprocess_chain(dense_provider, q, k, v, heads, kw)

                def dense_attention(qd, kd, vd):
                    out = dense(qd, kd, vd, output_heads=True)
                    if out.shape != qd.shape:
                        raise RuntimeError("Inherited attention returned an invalid prefix output shape")
                    return out.transpose(1, 2)

                from .sparse import attention, KernelUnavailable

                try:
                    result = attention(
                        q,
                        k,
                        v,
                        generic_measure.exact_rows if generic_measure is not None else prefix,
                        config,
                        state,
                        dense_attention=dense_attention,
                        dense_query_prefix=prefix if generic_measure is not None else None,
                        key_bias=None if generic_measure is None else generic_measure.key_log_measure,
                        key_bias_range=(
                            None
                            if generic_measure is None
                            else (generic_measure.bias_start, generic_measure.bias_stop)
                        ),
                        measure_identity=(
                            None if generic_measure is None else generic_measure.semantic_digest
                        ),
                    )
                except KernelUnavailable as exc:
                    if generic_measure is not None:
                        raise RuntimeError(
                            "attention_measure_v1 selected Sol-H3 but the weighted SM120 kernel is unavailable"
                        ) from exc
                    record("kernel_unavailable:" + str(exc), True)
                    dense_provider = previous
                    return dense()
                if external_mixed:
                    state.external_mixed_sol_calls += 1
                    state.external_mixed_q_rows += q.shape[2]
                    state.external_mixed_kernel_q_rows += q.shape[2]
                    if generic_measure is not None:
                        state.external_mixed_weighted_calls += 1
                        state.external_mixed_weighted_q_rows += q.shape[2]
                        state.external_mixed_weighted_kv_rows += k.shape[2]
                        state.external_mixed_weighted_exact_k_rows += generic_measure.exact_rows
                        record(
                            "sol_external_mixed_weighted_measure",
                            evidence=generic_measure.receipt_fields(),
                        )
                    elif legacy_measure is not None:
                        if measure_stats is None:
                            raise RuntimeError("Flow mixed-grid attention-measure telemetry was not produced")
                        state.external_mixed_measure_calls += 1
                        state.external_mixed_measure_q_rows += q.shape[2]
                        state.external_mixed_measure_kv_rows_before += measure_stats["kv_rows_before"]
                        state.external_mixed_measure_kv_rows_after += measure_stats["kv_rows_after"]
                        state.external_mixed_measure_removed_rows += measure_stats["kv_rows_removed"]
                        record("sol_external_mixed_measure")
                    else:
                        record("sol_external_mixed")
                else:
                    record("sol")
                return result

            def vdn_provider_v1(native, q, k, v, *, kind, scale, square_aligned=False):
                if not square_aligned or q.shape != k.shape or q.shape != v.shape:
                    record("vdn_" + kind + "_native", True)
                    return native()
                return vdn_provider_v2(native, q, k, v, kind=kind, scale=scale, square_aligned=square_aligned)

            def vdn_provider_v2(
                native, q, k, v, *, kind, scale, square_aligned=False, square_q=None, query_positions=None, sink_rows=0
            ):
                if kind != "local":
                    record("vdn_" + kind + "_native", True)
                    return native()

                # API v2 owns gathering and supplies the exact restricted KV domain.
                # square_q/query_positions remain accepted for existing VDN #11 callers;
                # neither is needed to evaluate the requested Q rows directly.
                if any(t.ndim != 3 for t in (q, k, v)) or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
                    record("vdn_local_domain", True)
                    return native()
                qc, kc, vc = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                reason = _shape_reason(qc, kc, vc, q.shape[1], None, {"skip_reshape": True}, rectangular=True)
                if scale != q.shape[-1] ** -0.5:
                    reason = "vdn_scale"
                if not isinstance(sink_rows, int) or not 0 <= sink_rows <= k.shape[0]:
                    reason = "vdn_sink_rows"
                if reason:
                    record(reason, True)
                    return native()
                state.eligible_calls += 1
                if warmup:
                    state.dense_calls += 1
                    record("vdn_dense_warmup")
                    return native()
                from .sparse import attention, KernelUnavailable

                try:
                    result = attention(qc, kc, vc, sink_rows, config, state, recompute_prefix_queries=False)
                except KernelUnavailable as exc:
                    record("kernel_unavailable:" + str(exc), True)
                    return native()
                result = result.reshape(q.shape[0], q.shape[1], q.shape[2])
                state.vdn_requested_q_rows += q.shape[0]
                state.vdn_kernel_q_rows += qc.shape[2]
                if q.shape[0] != k.shape[0]:
                    state.vdn_rectangular_sol_calls += 1
                state.vdn_local_sol_calls += 1
                record("vdn_local_sol")
                return result

            def vdn_provider_v3(native, q, k, v, *, kind, scale, square_aligned=False, sink_rows=0):
                return vdn_provider_v2(
                    native,
                    q,
                    k,
                    v,
                    kind=kind,
                    scale=scale,
                    square_aligned=square_aligned,
                    sink_rows=sink_rows,
                )

            def vdn_preprocess(q, k, v, *, heads, transformer_options):
                # VDN exposes post-RoPE [T,H,D]. Comfy attention preprocessors use
                # [B,H,T,D], so adapt only at this explicit full-domain boundary.
                qc, kc, vc = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                kw = {"transformer_options": transformer_options, "skip_reshape": True}
                qc, kc, vc, _ = _preprocess_chain(previous, qc, kc, vc, heads, kw)
                return tuple(t.squeeze(0).transpose(0, 1) for t in (qc, kc, vc))

            options["optimized_attention_override"] = override
            options[VDN_KEY] = vdn_provider_v1
            options[VDN_KEY_V2] = vdn_provider_v2
            options[VDN_KEY_V3] = vdn_provider_v3
            if hasattr(previous, "attention_preprocess_v1"):
                options[VDN_PREPROCESS_KEY] = vdn_preprocess

        def original_block(call_args):
            block = model.blocks[self.index]
            if config.exact:
                from .exact import execute_block, ineligible_reason

                reason = ineligible_reason(block, call_args)
                if reason is None and not state.native_verified and state.native_reason is None:
                    from .native_contract import verify_native

                    try:
                        verify_native()
                        state.native_verified = True
                    except RuntimeError as exc:
                        state.native_reason = str(exc)
                reason = reason or state.native_reason
                if reason is None:
                    result = execute_block(block, call_args, state.exact_verified)
                    state.exact_blocks += 1
                    return result
                state.fallbacks["exact:" + reason] += 1
            return extra["original_block"](call_args)

        result = (
            original_block(forwarded)
            if self.previous is None
            else self.previous(forwarded, {**extra, "original_block": original_block})
        )
        if config.backend == "sol" and len(routes) == route_start:
            attn_forward = getattr(getattr(model.blocks[self.index], "attn", None), "forward", None)
            if getattr(attn_forward, "_vdn_forward", False):
                api = _vdn_provider_api()
                if api is None:
                    record("vdn_provider_contract_missing", True)
                elif api < 2:
                    record("vdn_provider_v1_not_consumed", True)
                else:
                    record("vdn_provider_not_consumed", True)
            else:
                record("inherited_block_or_attention_owner", True)
        return result


def merge_config(old, new):
    if old.backend == new.backend == "sol":
        if replace(old, exact=new.exact) != new:
            raise ValueError("Conflicting SOL policies on one MODEL; branch before SOL to use different policies")
    policy = old if old.backend == "sol" else new
    return replace(policy, exact=old.exact or new.exact)


def _find_block_patch(value, index):
    """Find this install's Sol BlockPatch through known previous links only."""
    seen = set()
    current = value
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, BlockPatch):
            return current if current.index == index else None
        if hasattr(current, "_h3_flow_previous"):
            current = current._h3_flow_previous
        elif hasattr(current, "existing_patch"):
            current = current.existing_patch
        else:
            return None
    return None


def install(model, config):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrappersMP

    inner = model.get_model_object("diffusion_model")
    if not isinstance(inner, MiniMaxH3Model) or not len(inner.blocks):
        raise RuntimeError("Sol-H3 requires a native ComfyUI MiniMaxH3 MODEL")
    options = model.model_options.get("transformer_options", {})
    reapplied = KEY in options
    if reapplied:
        old = Config(**{name: options[KEY][name] for name in Config.__dataclass_fields__})
        config = merge_config(old, config)
    cloned = model.clone()
    to = cloned.model_options["transformer_options"]
    to[KEY] = config.metadata()
    if config.backend == "sol":
        to[HISTORY_KEY] = {**to.get(HISTORY_KEY, {}), "sol_h3": HistoryPolicy(config)}
    for kind, wrapper in (
        (WrappersMP.OUTER_SAMPLE, SamplingWrapper(config)),
        (WrappersMP.DIFFUSION_MODEL, DiffusionWrapper(config)),
    ):
        if reapplied:
            cloned.remove_wrappers_with_key(kind, KEY)
        cloned.add_wrapper_with_key(kind, KEY, wrapper)

    measure_owners = {}
    if not reapplied:
        replacements = options.get("patches_replace", {}).get("dit", {})
        for i in range(len(inner.blocks)):
            previous = replacements.get(("double_block", i))
            patch = BlockPatch(i, config, previous)
            measure_owners[i] = patch
            cloned.set_model_patch_replace(patch, "dit", "double_block", i)
    else:
        replacements = to.get("patches_replace", {}).get("dit", {})
        for i in range(len(inner.blocks)):
            patch = _find_block_patch(replacements.get(("double_block", i)), i)
            if patch is None:
                raise RuntimeError("Sol-H3 reapply cannot recover its block owner for attention measure binding")
            measure_owners[i] = patch

    if config.backend == "sol":
        from .attention_measure import register as register_attention_measure

        register_attention_measure(to, measure_owners)
        vdn_patches = sum(
            bool(getattr(cloned.object_patches.get(f"diffusion_model.blocks.{i}.attn.forward"), "_vdn_forward", False))
            for i in range(len(inner.blocks))
        )
        if vdn_patches:
            api = _vdn_provider_api()
            log.info(
                "Sol-H3 detected VDN object patches=%d; softmax-provider module API=%s",
                vdn_patches,
                api if api is not None else "missing",
            )
    log.info("Sol-H3 active: %s; AdaLN precompute=%s", config.metadata(), adaln_status(inner))
    return cloned
