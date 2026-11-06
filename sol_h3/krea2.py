"""Opt-in native Sol-Attn routing for ComfyUI's Krea 2 DiT.

This adapter owns only a per-MODEL attention override. MiniMax-H3's block
replacement, VDN, history and exact-fusion contracts are not involved.
"""

from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
import math

import torch

from .sparse import KernelUnavailable, attention as sol_attention

_LOG = logging.getLogger("comfy.sol_h3.krea2")
_KEY = "sol_krea2_sparse_attention_v1"
_ACTIVE = ContextVar("sol_krea2_active", default=None)
_EXECUTION = ContextVar("sol_krea2_model_execution", default=None)


@dataclass(frozen=True)
class Settings:
    tau: float = 1.0
    min_tokens: int = 4096
    dense_evaluations: int = 1
    dense_layers: int = 2
    extra_tokens: int = 256

    def __post_init__(self):
        if not math.isfinite(self.tau) or not 0 <= self.tau <= 3:
            raise ValueError("Krea 2 Sol-Attn tau must be finite and between 0 and 3")
        for name in ("min_tokens", "dense_evaluations", "dense_layers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Krea 2 Sol-Attn {name} must be a nonnegative integer")
        if type(self.extra_tokens) is not int or self.extra_tokens not in (0, 64, 128, 192, 256):
            raise ValueError("Krea 2 FP16 Sol extra_tokens must be 0, 64, 128, 192, or 256")


@dataclass
class _Run:
    owner: object
    settings: Settings
    evaluations: int = 0
    sparse_calls: int = 0
    fallbacks: Counter = field(default_factory=Counter)
    kernel: object = None
    kernel_failure: str | None = None
    kernel_device: object = None
    sparse_verified: set = field(default_factory=set)
    gates: list = field(default_factory=list)


class _SamplingScope:
    def __init__(self, owner, settings, verbose):
        self.owner = owner
        self.settings = settings
        self.verbose = verbose

    def __call__(self, executor, *args, **kwargs):
        state = _Run(self.owner, self.settings)
        token = _ACTIVE.set(state)
        try:
            return executor(*args, **kwargs)
        finally:
            _ACTIVE.reset(token)
            if self.verbose:
                _LOG.info(
                    "Sol-Krea2: model_evaluations=%d sparse_attention_calls=%d "
                    "arithmetic_gates=%d dense_fallbacks=%s",
                    state.evaluations, state.sparse_calls, len(state.gates),
                    dict(state.fallbacks),
                )


class _DiffusionScope:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, executor, *args, **kwargs):
        state = _ACTIVE.get()
        if state is None or state.owner is not self.owner:
            return executor(*args, **kwargs)
        # An outer forecasting wrapper may return a prediction without ever
        # calling the DiT. Count only a witnessed native main-block execution.
        scope = {"owner": self.owner, "counted": False}
        token = _EXECUTION.set(scope)
        try:
            return executor(*args, **kwargs)
        finally:
            _EXECUTION.reset(token)


def _can_sparse(q, k, v, heads, mask, precision, skip_reshape, skip_output_reshape, kwargs,
                settings, state, blocks):
    if state is None:
        return "outside_sampling_scope", None
    options = kwargs.get("transformer_options")
    if not isinstance(options, dict) or set(kwargs) - {"transformer_options", "_inside_attn_wrapper"}:
        return "attention_arguments", None
    if mask is not None or precision is not None or not skip_reshape or skip_output_reshape:
        return "attention_format", None
    if not all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in (q, k, v)):
        return "attention_tensors", None
    if (q.shape != k.shape or q.shape != v.shape or q.shape[0] != 1
            or q.shape[1] != heads or q.shape[-1] != 128 or q.shape[2] == 0):
        return "qkv_geometry", None
    if (q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype or v.dtype != q.dtype
            or k.device != q.device or v.device != q.device or any(t.stride(-1) != 1 for t in (q, k, v))):
        return "qkv_dtype_device_stride", None
    block = options.get("block_index")
    image = options.get("img_slice")
    if (options.get("block_type") != "single" or options.get("total_blocks") != blocks
            or type(block) is not int or not 0 <= block < blocks
            or not isinstance(image, (tuple, list)) or len(image) != 2
            or type(image[0]) is not int or type(image[1]) is not int
            or not 0 < image[0] < image[1] == q.shape[2]):
        return "not_krea2_main_attention", None
    # Krea reference-image K/V is a noncontiguous tail; a prefix-only sink
    # cannot preserve it, so reference-guided calls remain fully dense.
    if options.get("reference_image_num_tokens"):
        return "reference_images_need_exact_kv", None
    if state.evaluations <= settings.dense_evaluations:
        return "dense_evaluation_warmup", None
    if block < settings.dense_layers:
        return "dense_leading_layers", None
    if q.shape[2] < settings.min_tokens:
        return "sequence_below_threshold", None
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) not in ((12, 0), (12, 1)):
        return "device_not_sm120_sm121", None
    if (torch.is_grad_enabled() or torch.compiler.is_compiling()
            or torch.cuda.is_current_stream_capturing()):
        return "autograd_compile_or_capture", None
    return None, image[0]


def _override(previous, owner, settings, blocks):
    def patched(func, q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kwargs):
        state = _ACTIVE.get()
        if state is not None and state.owner is not owner:
            state = None

        options = kwargs.get("transformer_options")
        scope = _EXECUTION.get()
        if (state is not None and isinstance(scope, dict) and scope["owner"] is owner
                and not scope["counted"] and isinstance(options, dict)
                and options.get("block_type") == "single"
                and options.get("total_blocks") == blocks
                and options.get("block_index") == 0
                and isinstance(options.get("img_slice"), (tuple, list))
                and len(options["img_slice"]) == 2):
            scope["counted"] = True
            state.evaluations += 1

        def dense(reason):
            if state is not None:
                state.fallbacks[reason] += 1
            delegate = previous if previous is not None else lambda f, *a, **kw: f(*a, **kw)
            return delegate(
                func, q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs,
            )

        reason, prefix = _can_sparse(
            q, k, v, heads, mask, attn_precision, skip_reshape, skip_output_reshape,
            kwargs, settings, state, blocks,
        )
        if reason is not None:
            return dense(reason)
        if previous is not None and hasattr(previous, "attention_preprocess_v1"):
            return dense("inherited_attention_preprocessing")
        try:
            # All text K/V blocks are exact for image queries, while text
            # query rows are recomputed with dense SDPA by the Sol bridge.
            if q.dtype == torch.float16:
                from .krea2_fp16 import attention as fp16_attention
                return fp16_attention(q, k, v, prefix, settings, state)
            return sol_attention(q, k, v, prefix, settings, state)
        except KernelUnavailable:
            return dense("native_sol_unavailable")

    patched._sol_krea2_owner = owner
    patched._sol_krea2_preprocess = bool(previous is not None and hasattr(previous, "attention_preprocess_v1"))
    return patched



def spectrum_runtime_status(override):
    """A read-only snapshot for qualified forecast adapters.

    The callable's owner marker must match the current OUTER_SAMPLE scope.
    No state is fabricated for a foreign or inactive MODEL branch.
    """
    owner = getattr(override, "_sol_krea2_owner", None)
    state = _ACTIVE.get()
    if owner is None or state is None or state.owner is not owner:
        return None
    return {"api": 1, "settings": state.settings,
            "actual_evaluations": state.evaluations,
            "sparse_calls": state.sparse_calls,
            "kernel_unavailable": state.kernel_failure is not None}

def fp16_backend_available(device):
    """Predict the FP16 kernel route for Spectrum's feature-history partition."""
    from .krea2_fp16 import available
    return available(device)


def install(model, settings, *, verbose=False):
    """Attach Sol to one native Krea 2 model, preserving its prior dense owner."""
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.patcher_extension import WrappersMP

    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, SingleStreamDiT) or not diffusion_model.blocks:
        raise TypeError("Sol-Krea2 requires the native ComfyUI Krea 2 SingleStreamDiT MODEL")
    model_options = model.model_options.get("transformer_options", {})
    if _KEY in model_options:
        raise ValueError("Sol-Krea2 is already installed on this MODEL branch")

    previous = model_options.get("optimized_attention_override")
    if previous is None and any(
        getattr(getattr(getattr(block, "attn", None), "comfy_attention", None), "function", None) is not None
        for block in diffusion_model.blocks
    ):
        raise RuntimeError("Sol-Krea2 cannot preserve a configured per-block attention backend without an inherited override")

    cloned = model.clone()
    options = cloned.model_options.setdefault("transformer_options", {})
    owner = object()
    previous = options.get("optimized_attention_override")
    options["optimized_attention_override"] = _override(previous, owner, settings, len(diffusion_model.blocks))
    options[_KEY] = {"api": 1, "tau": settings.tau,
                     "min_tokens": settings.min_tokens,
                     "dense_evaluations": settings.dense_evaluations,
                     "dense_layers": settings.dense_layers,
                     "extra_tokens": settings.extra_tokens}
    cloned.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, _KEY, _SamplingScope(owner, settings, verbose))
    cloned.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, _KEY, _DiffusionScope(owner))
    return cloned
