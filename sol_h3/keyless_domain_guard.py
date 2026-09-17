"""Fail closed on Keyless row domains that the materialized Sol bridge cannot own.

The public Keyless provider contract can carry explicit query/value selections and
routing-owned value/position domains.  The current Sol reference provider does not
implement those physical-row semantics.  Delegating such a call through its generic
``dense_fallback`` is not safe: that fallback can re-enter the inherited Comfy
optimized-attention stack, including Sol's QKV-oriented block override, without
proving that the Keyless row-domain mapping was preserved.

Keep this guard additive until row-domain execution has its own reviewed provider
contract.  Masks and log-measure remain on the existing dense fallback because that
path already forces the direct materialized Keyless SDPA route; ``exact_blocks`` is
rejected by the provider itself for the same fail-closed reason.
"""
from __future__ import annotations

from typing import Any

_INSTALLED = False
_ORIGINAL_PROVIDER_CALL = None


def _guarded_provider_call(
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
    domains = []
    if query_domain is not None:
        domains.append("query_domain")
    if value_domain is not None:
        domains.append("value_domain")
    if getattr(routing, "value_domain", None) is not None:
        domains.append("routing.value_domain")
    if getattr(routing, "routing_position_domain", None) is not None:
        domains.append("routing.routing_position_domain")
    if domains:
        raise RuntimeError(
            "Keyless Sol materialized provider does not yet implement explicit row-domain "
            "semantics (" + ", ".join(domains) + "); refusing to delegate them through "
            "the inherited QKV optimized-attention fallback"
        )

    assert _ORIGINAL_PROVIDER_CALL is not None
    return _ORIGINAL_PROVIDER_CALL(
        self,
        q=q,
        v=v,
        heads=heads,
        scale=scale,
        routing=routing,
        mask=mask,
        log_measure=log_measure,
        exact_blocks=exact_blocks,
        query_domain=query_domain,
        value_domain=value_domain,
        dense_fallback=dense_fallback,
        transformer_options=transformer_options,
    )


def install() -> None:
    """Wrap the Sol-owned Keyless provider once, after keyless_compat is installed."""
    global _INSTALLED, _ORIGINAL_PROVIDER_CALL
    if _INSTALLED:
        return

    from . import keyless_compat

    provider_type: Any = keyless_compat._KeylessSolProviderV1
    current = provider_type.__call__
    if getattr(current, "_keyless_domain_guard_v1", False):
        _INSTALLED = True
        return
    _ORIGINAL_PROVIDER_CALL = current
    _guarded_provider_call._keyless_domain_guard_v1 = True
    provider_type.__call__ = _guarded_provider_call
    _INSTALLED = True


__all__ = ["install"]
