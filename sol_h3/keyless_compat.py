"""Materialized-route compatibility for MiniMax-H3-Keyless.

This module deliberately does not add a key projection or a ``qkv_proj`` alias.
Canonical Keyless attention materializes its routing view from projected V and then
calls ComfyUI's optimized-attention interface with ``(Q, route(V), raw V)``.  Sol's
existing attention override can consume exactly that tensor contract, so the first
production bridge is an identity/history layer around the existing route rather than
a second attention implementation.

Only the ordinary materialized Sol route is history-qualified here. Keyless+VDN,
weighted/external continuation routes, kernel-unavailable fallbacks and other
compositions remain actual-only until their own Keyless contracts are proven.
"""
from __future__ import annotations

from typing import Any

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_RUNTIME_IDENTITY_KEY = "sol_h3_keyless_semantic_identity_v1"
KEYLESS_RECEIPT_TAG = "sol_h3_keyless_materialized_route_v1"

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

_INSTALLED = False
_ORIGINAL_HISTORY_CALL = None
_ORIGINAL_HISTORY_ACCEPT = None
_ORIGINAL_RUNTIME_INSTALL = None
_ORIGINAL_RUNTIME_RECEIPT = None


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


def install() -> None:
    """Compose after history diagnostics so native-QKV behavior remains its inner path."""
    global _INSTALLED
    global _ORIGINAL_HISTORY_CALL, _ORIGINAL_HISTORY_ACCEPT
    global _ORIGINAL_RUNTIME_INSTALL, _ORIGINAL_RUNTIME_RECEIPT
    if _INSTALLED:
        return

    from . import interop, runtime

    _ORIGINAL_HISTORY_CALL = interop.HistoryPolicy.__call__
    _ORIGINAL_HISTORY_ACCEPT = interop.HistoryPolicy.accept_receipts
    _ORIGINAL_RUNTIME_INSTALL = runtime.install
    _ORIGINAL_RUNTIME_RECEIPT = runtime.receipt

    interop.HistoryPolicy.__call__ = _history_call
    interop.HistoryPolicy.accept_receipts = _history_accept
    runtime.install = _runtime_install
    runtime.receipt = _runtime_receipt
    _INSTALLED = True


__all__ = [
    "KEYLESS_CONTRACT_KEY",
    "KEYLESS_RECEIPT_TAG",
    "KEYLESS_RUNTIME_IDENTITY_KEY",
    "KeylessCompatibilityError",
    "install",
    "keyless_semantic_identity",
    "validate_keyless_contract",
]
