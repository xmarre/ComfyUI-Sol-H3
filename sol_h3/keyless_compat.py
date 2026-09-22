"""Materialized-route compatibility for MiniMax-H3-Keyless.

This module deliberately does not add a key projection or a ``qkv_proj`` alias.
Canonical Keyless attention exposes raw projected V plus a routing specification.
The Sol bridge owns a Keyless provider for the ordinary route, materializes the
logical route from that specification, scores against it, and retrieves raw V.
This keeps the provider boundary identical to the future fused Q+V kernel while
retaining the current proven materialized arithmetic as the reference path.

Only the ordinary materialized Sol route is history-qualified here. Keyless+VDN,
weighted/external continuation routes, masks/measures, foreign Keyless providers,
kernel-unavailable fallbacks and other compositions remain actual-only until their
own Keyless contracts are proven.
"""
from __future__ import annotations

import math
from typing import Any

import torch

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_RUNTIME_IDENTITY_KEY = "sol_h3_keyless_semantic_identity_v1"
KEYLESS_RECEIPT_TAG = "sol_h3_keyless_materialized_route_v1"
KEYLESS_PROVIDER_KEY = "minimax_h3_keyless_provider_v1"
KEYLESS_PROVIDER_IDENTITY = "sol_h3_keyless_provider_v1"

_KEYLESS_PROJECTION_ROWS = 14_336
_KEYLESS_HIDDEN_SIZE = 5_376
_KEYLESS_HEADS = 56
_KEYLESS_HEAD_DIM = 128
_KEYLESS_INNER_DIM = 7_168
_KEYLESS_CORE_BLOCKS = 50
_KEYLESS_REFINER_BLOCKS = 2
_KEYLESS_ROPE_POLICY = "h3_split_half_96_v1"
_KEYLESS_QV_ORDER = "q_effective;v"
_KEYLESS_MATERIALIZED_ROUTES = frozenset({"sol", "dense_warmup"})
_UNKNOWN_PROVIDER = object()

_INSTALLED = False
_ORIGINAL_HISTORY_CALL = None
_ORIGINAL_HISTORY_ACCEPT = None
_ORIGINAL_RUNTIME_INSTALL = None
_ORIGINAL_RUNTIME_RECEIPT = None
_ORIGINAL_DIFFUSION_CALL = None


class KeylessCompatibilityError(TypeError):
    """An advertised Keyless model does not satisfy the reviewed Sol bridge."""


def _field(contract: Any, name: str) -> Any:
    if not hasattr(contract, name):
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY} is missing required field {name!r}"
        )
    return getattr(contract, name)


def validate_keyless_contract(model: Any) -> Any | None:
    """Validate the public Keyless v1 boundary without importing its package."""
    if model is None or not hasattr(model, KEYLESS_CONTRACT_KEY):
        return None
    contract = getattr(model, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": _KEYLESS_CORE_BLOCKS,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": _KEYLESS_REFINER_BLOCKS,
        "heads": _KEYLESS_HEADS,
        "head_dim": _KEYLESS_HEAD_DIM,
        "inner_dim": _KEYLESS_INNER_DIM,
        "hidden_size": _KEYLESS_HIDDEN_SIZE,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": 1e-5,
        "rope_policy": _KEYLESS_ROPE_POLICY,
        "qv_order": _KEYLESS_QV_ORDER,
        "projection_attr": "qv_proj",
        "checkpoint_format_version": 1,
    }
    mismatches = []
    for name, wanted in expected.items():
        actual = _field(contract, name)
        if actual != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise KeylessCompatibilityError(
            f"unsupported {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches)
        )

    blocks = getattr(model, "blocks", None)
    refiner = getattr(getattr(model, "token_refiner", None), "blocks", None)
    try:
        block_count = len(blocks)
        refiner_count = len(refiner)
    except TypeError as exc:
        raise KeylessCompatibilityError("Keyless H3 block topology is not sized") from exc
    if block_count != _KEYLESS_CORE_BLOCKS or refiner_count != _KEYLESS_REFINER_BLOCKS:
        raise KeylessCompatibilityError(
            "Keyless H3 must expose 50 core blocks and two native-QKV token-refiner blocks"
        )

    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise KeylessCompatibilityError(f"Keyless block {index} has no attention module")
        if hasattr(attention, "qkv_proj"):
            raise KeylessCompatibilityError(
                f"Keyless block {index} exposes qkv_proj; Sol will not accept a fake/dead K path"
            )
        projection = getattr(attention, "qv_proj", None)
        weight = getattr(projection, "weight", None)
        shape = tuple(int(value) for value in getattr(weight, "shape", ()))
        if shape != (_KEYLESS_PROJECTION_ROWS, _KEYLESS_HIDDEN_SIZE):
            raise KeylessCompatibilityError(
                f"Keyless block {index} does not expose canonical qv_proj.weight geometry"
            )
        if not hasattr(attention, "q_norm") or not hasattr(attention, "route_norm"):
            raise KeylessCompatibilityError(
                f"Keyless block {index} is missing q_norm/route_norm semantics"
            )

    identity_fn = getattr(contract, "identity", None)
    if not callable(identity_fn):
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY} must expose identity()")
    try:
        identity = tuple(identity_fn())
        hash(identity)
    except (TypeError, ValueError) as exc:
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() must return a hashable tuple-like value"
        ) from exc
    if not identity:
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY}.identity() may not be empty")
    return contract


def keyless_semantic_identity(model: Any) -> tuple[Any, ...] | None:
    contract = validate_keyless_contract(model)
    if contract is None:
        return None
    return (KEYLESS_CONTRACT_KEY, *tuple(contract.identity()))


def _provider_owned(provider: Any, semantic: tuple[Any, ...]) -> bool:
    return bool(
        isinstance(provider, _KeylessSolProviderV1)
        and provider.semantic == semantic
        and provider.identity == (KEYLESS_PROVIDER_IDENTITY, 1, semantic)
    )


def _history_call(self, *, layout, options, model):
    assert _ORIGINAL_HISTORY_CALL is not None
    identity = _ORIGINAL_HISTORY_CALL(self, layout=layout, options=options, model=model)
    semantic = keyless_semantic_identity(model)
    if semantic is None:
        return identity
    if identity is None:
        return None
    if options.get(KEYLESS_RUNTIME_IDENTITY_KEY) != semantic:
        return None
    provider = options.get(KEYLESS_PROVIDER_KEY)
    if provider is not None and not _provider_owned(provider, semantic):
        # A foreign Keyless provider owns the numerical attention. Sol replacement
        # metadata alone cannot prove what that provider will execute next.
        return None
    return (*identity, ("keyless_materialized_route", semantic, KEYLESS_RECEIPT_TAG))


def _valid_keyless_receipts(receipts) -> bool | None:
    """Return True/False for Keyless-tagged sets, None for ordinary native receipts."""
    if not receipts:
        return None
    tagged = []
    for item in receipts:
        if not isinstance(item, (tuple, list)) or len(item) != 4:
            tagged.append(False)
            continue
        fields = item[3]
        tagged.append(
            isinstance(fields, tuple)
            and len(fields) == 2
            and fields[0] == KEYLESS_RECEIPT_TAG
        )
    if not any(tagged):
        return None
    if not all(tagged):
        return False

    semantic = None
    for item in receipts:
        provider, block, route, fields = item
        if provider != "sol_h3" or type(block) is not int or block < 0:
            return False
        if route not in _KEYLESS_MATERIALIZED_ROUTES:
            return False
        current = fields[1]
        try:
            current = tuple(current)
            hash(current)
        except (TypeError, ValueError):
            return False
        if not current or current[0] != KEYLESS_CONTRACT_KEY:
            return False
        if semantic is None:
            semantic = current
        elif current != semantic:
            return False
    return True


def _history_accept(self, receipts):
    verdict = _valid_keyless_receipts(receipts)
    if verdict is not None:
        return verdict
    assert _ORIGINAL_HISTORY_ACCEPT is not None
    return _ORIGINAL_HISTORY_ACCEPT(self, receipts)


def _runtime_receipt(
    options,
    block,
    route,
    *,
    measure_plan=None,
    call_token=None,
    fields=None,
):
    """Tag only the proven materialized Keyless routes; force all others actual-only."""
    assert _ORIGINAL_RUNTIME_RECEIPT is not None
    semantic = options.get(KEYLESS_RUNTIME_IDENTITY_KEY)
    if semantic is not None:
        if fields is None and measure_plan is None and route in _KEYLESS_MATERIALIZED_ROUTES:
            fields = (KEYLESS_RECEIPT_TAG, semantic)
        else:
            # Existing native-QKV parsers must not accidentally accept an unreviewed
            # Keyless VDN/Flow/weighted/fallback composition.
            route = "keyless_unvalidated:" + str(route)
    return _ORIGINAL_RUNTIME_RECEIPT(
        options,
        block,
        route,
        measure_plan=measure_plan,
        call_token=call_token,
        fields=fields,
    )


def _runtime_install(model, config):
    """Validate Keyless before installation and mark only the returned clone."""
    assert _ORIGINAL_RUNTIME_INSTALL is not None
    inner = model.get_model_object("diffusion_model")
    semantic = keyless_semantic_identity(inner)
    installed = _ORIGINAL_RUNTIME_INSTALL(model, config)
    if semantic is None:
        return installed

    to = installed.model_options["transformer_options"]
    to[KEYLESS_RUNTIME_IDENTITY_KEY] = semantic

    if getattr(config, "backend", None) == "sol":
        vdn_patches = [
            index
            for index in range(len(inner.blocks))
            if getattr(
                installed.object_patches.get(
                    f"diffusion_model.blocks.{index}.attn.forward"
                ),
                "_vdn_forward",
                False,
            )
        ]
        if vdn_patches:
            raise RuntimeError(
                "Sol-H3 Keyless materialized-route bridge does not yet authorize VDN-owned "
                "attention forwards. Keyless VDN requires its own value-space provider/checkpoint "
                "contract; refusing to reinterpret the released QKV VDN path."
            )
    return installed


def _sol_override_previous(options: dict[str, Any]) -> Any:
    """Prove the active override is Sol's current block-local owner and return its parent."""
    current = options.get("optimized_attention_override")
    if current is None:
        return _UNKNOWN_PROVIDER
    module = str(getattr(current, "__module__", ""))
    qualname = str(getattr(current, "__qualname__", ""))
    if not (
        (module == "sol_h3.runtime" or module.endswith(".sol_h3.runtime"))
        and qualname.endswith("BlockPatch.__call__.<locals>.override")
    ):
        return _UNKNOWN_PROVIDER
    from . import interop

    values = interop._closure_values(current)
    if not isinstance(values, dict) or "previous" not in values:
        return _UNKNOWN_PROVIDER
    return values["previous"]


def _dense_without_sol_override(
    q: torch.Tensor,
    route: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    scale: float,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    """Execute the inherited Comfy dense backend on materialized Keyless tensors.

    Inputs use Sol's [B,H,T,D] convention. The result is [B,T,H,D], matching
    ``sparse._dense_reference`` and the dense-prefix callback contract. Sol's own
    optimized-attention override and this Keyless provider are removed only from
    the local call options, preventing recursion while preserving the selected
    global dense backend.
    """
    from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

    clean = dict(transformer_options)
    clean.pop("optimized_attention_override", None)
    clean.pop(KEYLESS_PROVIDER_KEY, None)
    qc = AttentionTensorContainer(q.transpose(1, 2))
    rc = AttentionTensorContainer(route.transpose(1, 2))
    vc = AttentionTensorContainer(v.transpose(1, 2))
    out = optimized_attention(
        qc,
        rc,
        vc,
        heads,
        mask=None,
        skip_reshape=True,
        skip_output_reshape=True,
        scale=scale,
        transformer_options=clean,
    )
    if out.shape != q.shape:
        raise RuntimeError(
            f"Keyless dense prefix returned {tuple(out.shape)}, expected {tuple(q.shape)}"
        )
    return out.transpose(1, 2)


class _KeylessSolProviderV1:
    """Owned API-1 provider for the materialized Keyless Sol reference route.

    This provider intentionally materializes ``routing(V)``. It establishes the
    Q+V/routing-spec boundary that the future fused SM120 executor will consume;
    it is not itself the no-route-tensor production kernel.
    """

    api = 1

    def __init__(self, semantic: tuple[Any, ...]) -> None:
        self.semantic = semantic
        self.identity = (KEYLESS_PROVIDER_IDENTITY, 1, semantic)

    def __call__(
        self,
        *,
        q,
        v,
        heads,
        scale,
        routing,
        mask,
        log_measure,
        exact_blocks,
        query_domain,
        value_domain,
        dense_fallback,
        transformer_options,
    ):
        from . import runtime, sparse
        from .contracts import prefix_length
        from .interop import dense_evaluation_warmup

        options = transformer_options
        if options.get(KEYLESS_RUNTIME_IDENTITY_KEY) != self.semantic:
            raise RuntimeError("Keyless Sol provider semantic identity changed inside the model call")
        active = runtime._FORWARD.get()
        if active is None:
            raise RuntimeError("Keyless Sol provider called outside the Sol diffusion scope")
        _model, state, evaluation, seen, routes = active
        if state.config.backend != "sol":
            return dense_fallback()

        block_index = getattr(routing, "block_index", None)
        if type(block_index) is not int or block_index < 0 or block_index >= _KEYLESS_CORE_BLOCKS:
            raise RuntimeError("Keyless routing spec has an invalid core block index")
        if block_index not in seen:
            raise RuntimeError("Keyless Sol provider executed before its owning block patch")
        if (
            not torch.is_tensor(q)
            or not torch.is_tensor(v)
            or q.ndim != 3
            or q.shape != v.shape
            or q.shape[1] != heads
            or q.shape[-1] != _KEYLESS_HEAD_DIM
        ):
            raise RuntimeError("Keyless Sol provider requires Q/V [T, heads, 128] with identical geometry")
        expected_scale = q.shape[-1] ** -0.5
        if not math.isclose(float(scale), expected_scale, rel_tol=0.0, abs_tol=1e-12):
            return dense_fallback()

        # Mask/log-measure and selected/rectangular domains remain on the canonical
        # Keyless dense fallback until their sparse semantics are separately audited.
        # exact_blocks has no generic dense meaning, so refusing it is safer than
        # silently dropping a requested exact-row policy.
        if exact_blocks is not None:
            raise RuntimeError(
                "Keyless Sol materialized provider does not yet implement exact_blocks semantics"
            )
        if (
            mask is not None
            or log_measure is not None
            or query_domain is not None
            or value_domain is not None
            or getattr(routing, "value_domain", None) is not None
            or getattr(routing, "routing_position_domain", None) is not None
        ):
            return dense_fallback()

        previous = _sol_override_previous(options)
        if previous is _UNKNOWN_PROVIDER or previous is not None:
            # Preserve reviewed/foreign inherited preprocessors and dense owners via
            # the existing materialized optimized-attention bridge. Do not bypass a
            # layer merely because this provider can compute an ordinary Sol route.
            return dense_fallback()

        materialize = getattr(routing, "materialize", None)
        if not callable(materialize):
            raise RuntimeError("Keyless routing spec does not expose materialize(V)")
        route = materialize(v)
        if (
            not torch.is_tensor(route)
            or route.shape != v.shape
            or route.dtype != v.dtype
            or route.device != v.device
        ):
            raise RuntimeError("Keyless routing materialization changed V geometry/dtype/device")

        layout = options.get("minimax_h3_layout")
        try:
            prefix = prefix_length(layout, int(q.shape[0]))
        except RuntimeError:
            return dense_fallback()

        qc = q.transpose(0, 1).unsqueeze(0)
        rc = route.transpose(0, 1).unsqueeze(0)
        vc = v.transpose(0, 1).unsqueeze(0)

        def dense_attention(qd, rd, vd):
            return _dense_without_sol_override(
                qd,
                rd,
                vd,
                heads=heads,
                scale=float(scale),
                transformer_options=options,
            )

        def record(route_name: str, *, fallback: bool = False) -> None:
            routes.append((block_index, route_name))
            runtime.receipt(options, block_index, route_name)
            if fallback:
                state.fallbacks[route_name] += 1

        state.eligible_calls += 1
        warmup = dense_evaluation_warmup(state.config, evaluation, options) or block_index < state.config.dense_layers
        if warmup:
            result = dense_attention(qc, rc, vc)
            state.dense_calls += 1
            record("dense_warmup")
            return result.squeeze(0)

        try:
            result = sparse.attention(
                qc,
                rc,
                vc,
                prefix,
                state.config,
                state,
                dense_attention=dense_attention,
            )
        except sparse.KernelUnavailable as exc:
            result = dense_attention(qc, rc, vc)
            record("kernel_unavailable:" + str(exc), fallback=True)
            return result.squeeze(0)

        record("sol")
        if result.shape != (1, q.shape[0], heads * q.shape[-1]):
            raise RuntimeError(
                "Keyless Sol sparse provider returned an invalid flattened output shape"
            )
        return result.reshape(q.shape[0], heads, q.shape[-1])


def _diffusion_call(self, executor, x, timestep, context, transformer_options=None, **kwargs):
    """Install one request-stable owned provider without replacing foreign providers."""
    assert _ORIGINAL_DIFFUSION_CALL is not None
    options = dict(transformer_options or {})
    semantic = options.get(KEYLESS_RUNTIME_IDENTITY_KEY)
    if semantic is None or getattr(self.config, "backend", None) != "sol":
        return _ORIGINAL_DIFFUSION_CALL(
            self, executor, x, timestep, context, options, **kwargs
        )

    existing = options.get(KEYLESS_PROVIDER_KEY)
    if existing is None:
        from . import runtime

        state = runtime._REQUEST.get()
        if state is None:
            raise RuntimeError("Keyless Sol provider requires an active Sol sampling request")
        provider = getattr(state, "_keyless_provider_v1", None)
        if provider is None:
            provider = _KeylessSolProviderV1(tuple(semantic))
            state._keyless_provider_v1 = provider
        elif not _provider_owned(provider, tuple(semantic)):
            raise RuntimeError("Keyless Sol request provider identity changed inside the sampling request")
        options[KEYLESS_PROVIDER_KEY] = provider
    # A foreign provider remains authoritative. _history_call returns None for it,
    # so Spectrum cannot forecast under Sol's materialized-route identity.
    return _ORIGINAL_DIFFUSION_CALL(
        self, executor, x, timestep, context, options, **kwargs
    )


def install() -> None:
    """Compose after history diagnostics so native-QKV behavior remains its inner path."""
    global _INSTALLED
    global _ORIGINAL_HISTORY_CALL, _ORIGINAL_HISTORY_ACCEPT
    global _ORIGINAL_RUNTIME_INSTALL, _ORIGINAL_RUNTIME_RECEIPT, _ORIGINAL_DIFFUSION_CALL
    if _INSTALLED:
        return

    from . import interop, runtime

    _ORIGINAL_HISTORY_CALL = interop.HistoryPolicy.__call__
    _ORIGINAL_HISTORY_ACCEPT = interop.HistoryPolicy.accept_receipts
    _ORIGINAL_RUNTIME_INSTALL = runtime.install
    _ORIGINAL_RUNTIME_RECEIPT = runtime.receipt
    _ORIGINAL_DIFFUSION_CALL = runtime.DiffusionWrapper.__call__

    interop.HistoryPolicy.__call__ = _history_call
    interop.HistoryPolicy.accept_receipts = _history_accept
    runtime.install = _runtime_install
    runtime.receipt = _runtime_receipt
    runtime.DiffusionWrapper.__call__ = _diffusion_call
    _INSTALLED = True


__all__ = [
    "KEYLESS_CONTRACT_KEY",
    "KEYLESS_PROVIDER_KEY",
    "KEYLESS_RECEIPT_TAG",
    "KEYLESS_RUNTIME_IDENTITY_KEY",
    "KeylessCompatibilityError",
    "install",
    "keyless_semantic_identity",
    "validate_keyless_contract",
]
