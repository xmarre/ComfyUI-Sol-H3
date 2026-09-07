from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging

from .contracts import (
    KEY,
    Config,
    adaln_status,
    prefix_length,
    reject_forecasting_conflicts,
    reject_sparse_conflicts,
)

log = logging.getLogger("comfy.sol_h3")
_REQUEST = ContextVar("sol_h3_request", default=None)
_FORWARD = ContextVar("sol_h3_forward", default=None)


@dataclass
class Request:
    config: Config
    evaluations: int = 0
    sparse_calls: int = 0
    dense_calls: int = 0
    exact_blocks: int = 0
    exact_verified: set = field(default_factory=set)
    sparse_verified: set = field(default_factory=set)
    dense_attention_backends: set = field(default_factory=set)
    gates: list = field(default_factory=list)
    kernel: object = None
    kernel_device: object = None
    native_verified: bool = False


@dataclass(frozen=True)
class SamplingWrapper:
    config: Config

    def __call__(self, executor, *args, **kwargs):
        state = Request(self.config)
        token = _REQUEST.set(state)
        success = False
        try:
            result = executor(*args, **kwargs)
            if self.config.backend == "sol" and state.sparse_calls == 0:
                raise RuntimeError("SOL requested but zero sparse calls executed; reduce warmup or remove bypassing patches")
            if self.config.exact and state.exact_blocks == 0:
                raise RuntimeError("Exact Sol-H3 requested but no fused blocks executed")
            success = True
            return result
        finally:
            _REQUEST.reset(token)
            log.info("Sol-H3 %s", json.dumps({"success": success, **self.config.metadata(),
                     "actual_evaluations": state.evaluations, "sparse_calls": state.sparse_calls,
                     "dense_calls": state.dense_calls, "exact_blocks": state.exact_blocks,
                     "dense_attention_backends": sorted(state.dense_attention_backends),
                     "arithmetic_gates": state.gates}))


@dataclass(frozen=True)
class DiffusionWrapper:
    config: Config

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        if options.get(KEY) != self.config.metadata():
            raise RuntimeError("Sol-H3 runtime metadata changed after installation")
        if self.config.backend == "sol":
            reject_sparse_conflicts(options)
        else:
            reject_forecasting_conflicts(options)
        state = _REQUEST.get()
        if state is None or state.config != self.config:
            raise RuntimeError("Sol-H3 must execute inside its native OUTER_SAMPLE lifecycle")
        model = executor.class_obj
        import torch
        if torch.is_grad_enabled():
            raise RuntimeError("Sol-H3 supports inference only (no_grad)")
        if torch.compiler.is_compiling() or (x[0].device.type == "cuda" and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("Sol-H3 validation/lifecycle is not compatible with whole-model graph capture")
        if self.config.exact and not state.native_verified:
            from .native_contract import verify_native
            verify_native()
            state.native_verified = True
        evaluation = state.evaluations
        state.evaluations += 1
        # Explicit scope handles arbitrary sigmas, restarts and nested samplers;
        # no timestep-direction heuristic or persistent model activation cache.
        seen = set()
        token = _FORWARD.set((model, state, evaluation, seen))
        try:
            result = executor(x, timestep, context, options, **kwargs)
            if seen != set(range(len(model.blocks))):
                raise RuntimeError(
                    "Another provider bypassed Sol-H3 block patches inside this diffusion call; "
                    "apply Sol-H3 after wrappers that may short-circuit transformer execution "
                    "(for example Spectrum), and after compatible block-replacement patches"
                )
            return result
        finally:
            _FORWARD.reset(token)


@dataclass(frozen=True)
class BlockPatch:
    index: int
    config: Config
    previous: object = None

    def __call__(self, args, extra):
        active = _FORWARD.get()
        if active is None:
            raise RuntimeError("Sol-H3 block called outside its diffusion scope")
        model, state, evaluation, seen = active
        if state.config != self.config:
            raise RuntimeError("Sol-H3 block configuration differs from the active sampling scope")
        if self.index in seen:
            raise RuntimeError("A provider called the same Sol-H3 block twice in one actual evaluation")
        seen.add(self.index)
        options = dict(args["transformer_options"])
        forwarded = {**args, "transformer_options": options}
        sparse_before = state.sparse_calls
        sparse_expected = False
        if self.config.backend == "sol":
            reject_sparse_conflicts(options)
            if getattr(model.blocks[self.index].attn.forward, "_vdn_forward", False):
                raise RuntimeError("SOL cannot own attention alongside VDN hybrid/window attention; use exact mode")
            if args.get("attention") is not None:
                raise RuntimeError("SOL and another attention provider own the same operation")
            prefix = prefix_length(args["layout"], len(args["img"]))
            if evaluation >= self.config.dense_evaluations and self.index >= self.config.dense_layers:
                sparse_expected = True

                def override(original, q, k, v, heads, mask=None, **kw):
                    if mask is not None or not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
                        raise RuntimeError("SOL received unsupported attention mask/layout flags")
                    allowed = {"skip_reshape", "skip_output_reshape", "transformer_options",
                               "_inside_attn_wrapper", "attn_precision"}
                    if set(kw) - allowed or kw.get("attn_precision") is not None:
                        raise RuntimeError("SOL received unsupported attention precision/arguments")
                    if q.ndim != 4 or q.shape[1] != heads or q.shape[2] != len(args["img"]):
                        raise RuntimeError("SOL QKV do not match the current packed block")
                    if not callable(original):
                        raise RuntimeError("SOL could not access the active dense attention provider")
                    state.dense_attention_backends.add(
                        f"{getattr(original, '__module__', '<unknown>')}.{getattr(original, '__name__', type(original).__name__)}"
                    )

                    def dense_attention(qd, kd, vd):
                        # Preserve the attention wrapper's recursion guard. Some raw
                        # providers (for example Sage on an unsupported shape) call a
                        # wrapped dense fallback internally; dropping this marker would
                        # re-enter this SOL override recursively through transformer_options.
                        dense_kw = dict(kw)
                        dense_kw["_inside_attn_wrapper"] = True
                        dense_kw["skip_reshape"] = True
                        dense_kw["skip_output_reshape"] = True
                        out = original(qd, kd, vd, heads, mask=None, **dense_kw)
                        if out.ndim != 4 or out.shape[:3] != (qd.shape[0], heads, qd.shape[2]):
                            raise RuntimeError("Active dense attention provider returned an unexpected SOL reference shape")
                        return out.transpose(1, 2)

                    from .sparse import attention
                    return attention(q, k, v, prefix, self.config, state, dense_attention=dense_attention)

                options["optimized_attention_override"] = override
            else:
                state.dense_calls += 1

        exact_before = state.exact_blocks if self.config.exact else None

        def original_block(call_args):
            if self.config.exact:
                from .exact import execute_block
                result = execute_block(model.blocks[self.index], call_args, state.exact_verified)
                state.exact_blocks += 1
                return result
            return extra["original_block"](call_args)

        if self.previous is None:
            result = original_block(forwarded)
        else:
            previous_extra = {**extra, "original_block": original_block}
            result = self.previous(forwarded, previous_extra)

        if self.config.exact and state.exact_blocks != exact_before + 1:
            raise RuntimeError(
                "An existing H3 block replacement did not delegate exactly once to the Sol-H3 fused native block"
            )
        if sparse_expected and state.sparse_calls != sparse_before + 1:
            raise RuntimeError("An eligible SOL block did not execute exactly one sparse attention call")
        return result


def install(model, config):
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrappersMP
    inner = model.get_model_object("diffusion_model")
    if not isinstance(inner, MiniMaxH3Model) or not len(inner.blocks):
        raise RuntimeError("Sol-H3 requires a native ComfyUI MiniMaxH3 MODEL")
    options = model.model_options.get("transformer_options", {})
    if KEY in options:
        raise RuntimeError("Sol-H3 is already applied; branch from the original MODEL to change configuration")
    replacements = options.get("patches_replace", {}).get("dit", {})
    existing = {i: replacements.get(("double_block", i)) for i in range(len(inner.blocks))}
    if config.backend == "sol" and any(patch is not None for patch in existing.values()):
        raise RuntimeError(
            "An existing patch owns H3 blocks; external SOL must be the sole sparse/block owner. "
            "Use exact mode to wrap compatible block replacements, or remove the competing sparse provider."
        )
    if config.backend == "sol":
        reject_sparse_conflicts(options)
        if config.dense_layers >= len(inner.blocks):
            raise RuntimeError("dense_layers would disable every sparse layer")
    else:
        reject_forecasting_conflicts(options)
    cloned = model.clone()
    # Config and callbacks are immutable. All counters, gates and model references
    # live in per-invocation ContextVars, never in a shared clone attachment.
    cloned.model_options["transformer_options"][KEY] = config.metadata()
    cloned.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, SamplingWrapper(config))
    cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, DiffusionWrapper(config))
    for i in range(len(inner.blocks)):
        # Exact mode intentionally wraps a pre-existing block replacement. This lets
        # compatibility-preserving providers transform args/output while Sol-H3 remains
        # the final native block implementation. The runtime verifies delegation exactly once.
        cloned.set_model_patch_replace(BlockPatch(i, config, existing[i]), "dit", "double_block", i)
    log.info("Sol-H3 active: %s; AdaLN precompute=%s", config.metadata(), adaln_status(inner))
    return cloned
