"""Internal witness-owner adapter for the bounded first-high Sol-local diagnostic.

The production ``sparse.attention`` signature deliberately has no transformer
options argument. E keeps that API unchanged: this diagnostic-only adapter
passes the clone-stable Flow evidence owner to the already-installed E wrapper
only for the three preserved witness calls, strips the private keyword before
delegating to the untouched production function, and aligns the bounded rich
all-selected-vs-native metric payload with the production arithmetic-gate schema.
"""
from __future__ import annotations

import math

from . import first_high_sol_local_diagnostic as diagnostic
from . import sparse

_INSTALLED = False


def _install_metric_gate_contract() -> None:
    current = diagnostic._detailed_bthd_metrics
    if getattr(current, "_first_high_sol_local_gate_schema_v1", False):
        return

    def detailed_bthd_metrics_with_gate_schema(got, want):
        metrics = current(got, want)
        if not bool(metrics.get("finite")):
            reference_peak_abs = math.nan
            catastrophic_limit = math.nan
        else:
            reference_peak_abs = 0.0
            tokens = int(want.shape[1])
            for start in range(0, tokens, 64):
                stop = min(tokens, start + 64)
                tile_peak = float(want[0, start:stop].float().abs().max().item())
                reference_peak_abs = max(reference_peak_abs, tile_peak)
            catastrophic_limit = max(
                sparse.ARITH_CATASTROPHIC_MAX_FLOOR,
                sparse.ARITH_CATASTROPHIC_REFERENCE_PEAK_MULTIPLIER * reference_peak_abs,
            )
        metrics["reference_peak_abs"] = reference_peak_abs
        metrics["catastrophic_max_abs_limit"] = catastrophic_limit
        return metrics

    detailed_bthd_metrics_with_gate_schema._first_high_sol_local_gate_schema_v1 = True
    detailed_bthd_metrics_with_gate_schema._first_high_sol_local_inner = current
    diagnostic._detailed_bthd_metrics = detailed_bthd_metrics_with_gate_schema


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

    _install_metric_gate_contract()

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
