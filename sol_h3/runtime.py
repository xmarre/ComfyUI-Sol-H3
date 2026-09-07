from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging

from .contracts import KEY, Config, adaln_status, prefix_length, reject_sparse_conflicts

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
    gates: list = field(default_factory=list)
    kernel: object = None


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
            log.info("Sol-H3 %s", json.dumps({"success": success, **self.config.metadata(),
                     "actual_evaluations": state.evaluations, "sparse_calls": state.sparse_calls,
                     "dense_calls": state.dense_calls, "exact_blocks": state.exact_blocks,
                     "arithmetic_gates": state.gates}))
            _REQUEST.reset(token)


@dataclass(frozen=True)
class DiffusionWrapper:
    config: Config

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        if options.get(KEY) != self.config.metadata():
            raise RuntimeError("Sol-H3 runtime metadata changed after installation")
        if self.config.backend == "sol":
            reject_sparse_conflicts(options)
        state = _REQUEST.get()
        if state is None or state.config != self.config:
            raise RuntimeError("Sol-H3 must execute inside its native OUTER_SAMPLE lifecycle")
        model = executor.class_obj
        import torch
        if torch.is_grad_enabled() or model.training:
            raise RuntimeError("Sol-H3 supports inference only (eval and no_grad)")
        evaluation = state.evaluations
        state.evaluations += 1
        # Explicit scope handles arbitrary sigmas, restarts and nested samplers;
        # no timestep-direction heuristic or persistent model activation cache.
        token = _FORWARD.set((model, state, evaluation))
        try:
            return executor(x, timestep, context, options, **kwargs)
        finally:
            _FORWARD.reset(token)


@dataclass(frozen=True)
class BlockPatch:
    index: int
    config: Config

    def __call__(self, args, extra):
        active = _FORWARD.get()
        if active is None:
            raise RuntimeError("Sol-H3 block called outside its diffusion scope")
        model, state, evaluation = active
        options = dict(args["transformer_options"])
        forwarded = {**args, "transformer_options": options}
        if self.config.backend == "sol":
            reject_sparse_conflicts(options)
            if options.get("optimized_attention_override") is not None or args.get("attention") is not None:
                raise RuntimeError("SOL and another attention provider own the same operation")
            prefix = prefix_length(args["layout"], len(args["img"]))
            if evaluation >= self.config.dense_evaluations and self.index >= self.config.dense_layers:
                def override(original, q, k, v, heads, mask=None, **kw):
                    if mask is not None or not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
                        raise RuntimeError("SOL received unsupported attention mask/layout flags")
                    from .sparse import attention
                    return attention(q, k, v, prefix, self.config, state)
                options["optimized_attention_override"] = override
            else:
                state.dense_calls += 1
        if self.config.exact:
            from .exact import execute_block
            result = execute_block(model.blocks[self.index], forwarded, state.exact_verified)
            state.exact_blocks += 1
            return result
        return extra["original_block"](forwarded)


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
    if any(("double_block", i) in replacements for i in range(len(inner.blocks))):
        raise RuntimeError("An existing patch owns H3 blocks; apply Sol-H3 before compatible wrapping patches")
    if config.backend == "sol":
        reject_sparse_conflicts(options)
        if options.get("optimized_attention_override") is not None:
            raise RuntimeError("SOL conflicts with an existing optimized attention override")
        if config.dense_layers >= len(inner.blocks):
            raise RuntimeError("dense_layers would disable every sparse layer")
    cloned = model.clone()
    # Config and callbacks are immutable. All counters, gates and model references
    # live in per-invocation ContextVars, never in a shared clone attachment.
    cloned.model_options["transformer_options"][KEY] = config.metadata()
    cloned.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, KEY, SamplingWrapper(config))
    cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, KEY, DiffusionWrapper(config))
    for i in range(len(inner.blocks)):
        cloned.set_model_patch_replace(BlockPatch(i, config), "dit", "double_block", i)
    log.info("Sol-H3 active: %s; AdaLN precompute=%s", config.metadata(), adaln_status(inner))
    return cloned
