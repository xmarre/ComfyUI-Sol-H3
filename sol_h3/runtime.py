from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import json
import logging

from .contracts import KEY, Config, adaln_status, prefix_length
from .interop import HISTORY_KEY, VDN_KEY, HistoryPolicy, provider_name, receipt

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
    vdn_local_sol_calls: int = 0
    exact_blocks: int = 0
    fallbacks: Counter = field(default_factory=Counter)
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
            log.info("Sol-H3 %s", json.dumps({"success": success, **self.config.metadata(),
                     "actual_evaluations": state.evaluations, "sol_eligible_calls": state.eligible_calls,
                     "sparse_calls": state.sparse_calls, "dense_warmup": state.dense_calls,
                     "compatibility_fallbacks": dict(state.fallbacks),
                     "vdn_local_sol_calls": state.vdn_local_sol_calls,
                     "numerical_backend_transitions": state.backend_transitions,
                     "exact_blocks": state.exact_blocks,
                     "inherited_dense_backends": sorted(state.dense_attention_backends),
                     "arithmetic_gates": state.gates}))


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
        # The first executed block, not the diffusion wrapper, consumes warmup.
        # This works with forecasting wrappers on either side of this wrapper.
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


def _shape_reason(q, k, v, heads, mask, kw):
    import torch
    if mask is not None:
        return "attention_mask"
    if not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
        return "attention_layout_flags"
    allowed = {"skip_reshape", "skip_output_reshape", "transformer_options",
               "_inside_attn_wrapper", "attn_precision"}
    if set(kw) - allowed or kw.get("attn_precision") is not None:
        return "attention_arguments"
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1:
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
        config = state.config  # merged configuration; clones may share old immutable callbacks
        if not seen:
            state.evaluations += 1
        seen.add(self.index)
        options = dict(args["transformer_options"])
        forwarded = {**args, "transformer_options": options}
        route_start = len(routes)

        def record(route, fallback=False):
            routes.append((self.index, route))
            receipt(options, self.index, route)
            if fallback:
                state.fallbacks[route] += 1

        if config.backend == "sol":
            previous = options.get("optimized_attention_override")
            warmup = evaluation < config.dense_evaluations or self.index < config.dense_layers

            def override(original, q, k, v, heads, mask=None, **kw):
                dense_provider = previous
                state.dense_attention_backends.add(provider_name(dense_provider or original))

                def dense(qd=q, kd=k, vd=v, *, output_heads=None):
                    dense_kw = {**kw, "_inside_attn_wrapper": True}
                    if output_heads is not None:
                        dense_kw["skip_reshape"] = True
                        dense_kw["skip_output_reshape"] = output_heads
                    if dense_provider is not None:
                        return dense_provider(original, qd, kd, vd, heads, mask=mask, **dense_kw)
                    return original(qd, kd, vd, heads, mask=mask, **dense_kw)

                current_options = kw.get("transformer_options") or options
                reason = _shape_reason(q, k, v, heads, mask, kw)
                # This override transforms K rather than merely evaluating dense
                # attention. Until it exposes a transform contract, preserve it whole.
                if (current_options.get("minimax_h3_untwist_rope", {}).get("enabled")
                        and not hasattr(previous, "attention_preprocess_v1")):
                    reason = "inherited_key_transform"
                consumer = current_options.get("spectrum_h3_runtime")
                if consumer is not None and not callable(getattr(consumer, "prepare_backend_history", None)):
                    reason = "forecast_consumer_no_history_contract"
                if current_options.get("vdn_h3_external_sequence_v1"):
                    reason = "external_sequence_native"
                layout = current_options.get("minimax_h3_layout", args.get("layout"))
                prefix = 0
                if reason is None:
                    try:
                        prefix = prefix_length(layout, q.shape[2])
                    except RuntimeError:
                        reason = "packed_layout_not_representable"
                if reason is not None:
                    record(reason, True)
                    return dense()
                state.eligible_calls += 1
                if warmup:
                    state.dense_calls += 1
                    record("dense_warmup")
                    return dense()

                # Pure QKV transforms explicitly expose their inherited provider.
                # Apply them once to sparse QKV; prefix queries use only the dense
                # leaf afterwards, so neither RoPE nor reference K scaling repeats.
                chain_seen = set()
                while hasattr(dense_provider, "attention_preprocess_v1"):
                    if id(dense_provider) in chain_seen:
                        raise RuntimeError("Cyclic attention preprocessing contract")
                    chain_seen.add(id(dense_provider))
                    transform, dense_provider = dense_provider.attention_preprocess_v1
                    shapes = (q.shape, k.shape, v.shape)
                    q, k, v = transform(q, k, v, heads, **kw)
                    if (q.shape, k.shape, v.shape) != shapes:
                        raise RuntimeError("Attention preprocessing changed its promised QKV topology")

                def dense_attention(qd, kd, vd):
                    out = dense(qd, kd, vd, output_heads=True)
                    if out.shape != qd.shape:
                        raise RuntimeError("Inherited attention returned an invalid prefix output shape")
                    return out.transpose(1, 2)

                from .sparse import attention, KernelUnavailable
                try:
                    result = attention(q, k, v, prefix, config, state, dense_attention=dense_attention)
                except KernelUnavailable as exc:
                    record("kernel_unavailable:" + str(exc), True)
                    dense_provider = previous
                    return dense()
                record("sol")
                return result

            def vdn_provider(native, q, k, v, *, kind, scale, square_aligned=False):
                # VDN supplies already restricted Q/K/V; never gather unrestricted
                # sequence rows here. Global/anchor rows remain native and exact.
                if kind != "local" or not square_aligned:
                    record("vdn_" + kind + "_native", True)
                    return native()
                qc, kc, vc = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                reason = _shape_reason(qc, kc, vc, q.shape[1], None, {"skip_reshape": True})
                if scale != q.shape[-1] ** -0.5:
                    reason = "vdn_scale"
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
                    result = attention(qc, kc, vc, 0, config, state)
                except KernelUnavailable as exc:
                    record("kernel_unavailable:" + str(exc), True)
                    return native()
                state.vdn_local_sol_calls += 1
                record("vdn_local_sol")
                return result.reshape_as(q)

            options["optimized_attention_override"] = override
            options[VDN_KEY] = vdn_provider

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

        result = (original_block(forwarded) if self.previous is None else
                  self.previous(forwarded, {**extra, "original_block": original_block}))
        if config.backend == "sol" and len(routes) == route_start:
            record("inherited_block_or_attention_owner", True)
        return result


def merge_config(old, new):
    if old.backend == new.backend == "sol":
        if replace(old, exact=new.exact) != new:
            raise ValueError("Conflicting SOL policies on one MODEL; branch before SOL to use different policies")
    policy = old if old.backend == "sol" else new
    return replace(policy, exact=old.exact or new.exact)


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
    for kind, wrapper in ((WrappersMP.OUTER_SAMPLE, SamplingWrapper(config)),
                          (WrappersMP.DIFFUSION_MODEL, DiffusionWrapper(config))):
        if reapplied:
            cloned.remove_wrappers_with_key(kind, KEY)
        cloned.add_wrapper_with_key(kind, KEY, wrapper)
    if not reapplied:
        replacements = options.get("patches_replace", {}).get("dit", {})
        for i in range(len(inner.blocks)):
            previous = replacements.get(("double_block", i))
            cloned.set_model_patch_replace(BlockPatch(i, config, previous), "dit", "double_block", i)
    log.info("Sol-H3 active: %s; AdaLN precompute=%s", config.metadata(), adaln_status(inner))
    return cloned
