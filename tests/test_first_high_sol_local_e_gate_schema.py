from __future__ import annotations

import math

import torch

from sol_h3 import first_high_sol_local_diagnostic as e
from sol_h3 import sparse


def test_detailed_metric_payload_satisfies_production_arithmetic_gate_schema():
    want = torch.zeros((1, 4, 2, 3), dtype=torch.float32)
    want[0, 1, 0, 2] = 18.125
    got = want.clone()
    got[0, 2, 1, 1] = 0.01

    metrics = e._detailed_bthd_metrics(got, want)

    assert metrics["finite"] is True
    assert metrics["reference_peak_abs"] == 18.125
    assert metrics["catastrophic_max_abs_limit"] == 72.5
    assert sparse.arithmetic_gate_passes(metrics) is True


def test_detailed_metric_payload_remains_fail_closed_for_nonfinite_reference():
    want = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    want[0, 0, 0, 0] = math.nan
    got = torch.zeros_like(want)

    metrics = e._detailed_bthd_metrics(got, want)

    assert metrics["finite"] is False
    assert math.isnan(metrics["reference_peak_abs"])
    assert math.isnan(metrics["catastrophic_max_abs_limit"])
    assert sparse.arithmetic_gate_passes(metrics) is False


def test_production_arithmetic_gate_still_enforces_catastrophic_maximum():
    metrics = {
        "finite": True,
        "mean_abs": 0.0,
        "rel_l2": 0.0,
        "max_abs": 72.5001,
        "reference_peak_abs": 18.125,
        "catastrophic_max_abs_limit": 72.5,
    }

    assert sparse.arithmetic_gate_passes(metrics) is False
