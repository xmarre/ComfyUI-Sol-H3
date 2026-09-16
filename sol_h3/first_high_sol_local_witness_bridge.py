"""Internal witness-owner adapter for the bounded first-high Sol-local diagnostic.

The production ``sparse.attention`` signature deliberately has no transformer
options argument.  E keeps that API unchanged: this diagnostic-only adapter
passes the clone-stable Flow evidence owner to the already-installed E wrapper
only for the three preserved witness calls, and strips the private keyword before
delegating to the untouched production function.
"""
from __future__ import annotations

from . import first_high_sol_local_diagnostic as diagnostic
from . import sparse

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    current = sparse.attention
    if not getattr(current, "_first_high_sol_local_e_v1", False):
        raise RuntimeError("Sol-local witness bridge requires the E sparse wrapper to be installed first")
    production = diagnostic._ORIGINAL_SPARSE_ATTENTION
    if not callable(production):
        raise RuntimeError("Sol-local witness bridge cannot resolve the production sparse attention owner")

    def production_adapter(q, k, v, prefix, config, state, *args, diagnostic_transformer_options=None, **kwargs):
        _ = diagnostic_transformer_options
        return production(q, k, v, prefix, config, state, *args, **kwargs)

    diagnostic._ORIGINAL_SPARSE_ATTENTION = production_adapter

    def witness_owner_adapter(q, k, v, prefix, config, state, *args, **kwargs):
        group = diagnostic._LOCAL_GROUP.get()
        if group is not None and group.witness_record is not None:
            options = group.witness_record.get("_transformer_options")
            if not isinstance(options, dict):
                raise RuntimeError("first-high Sol-local witness lost its transformer-option owner")
            kwargs["diagnostic_transformer_options"] = options
        return current(q, k, v, prefix, config, state, *args, **kwargs)

    witness_owner_adapter._first_high_sol_local_witness_bridge_v1 = True
    witness_owner_adapter._first_high_sol_local_inner = current
    sparse.attention = witness_owner_adapter
    _INSTALLED = True


install()

__all__ = ["install"]
