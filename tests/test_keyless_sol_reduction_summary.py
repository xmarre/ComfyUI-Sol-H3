from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sol_h3.keyless_route_summary import (
    NORM_EPS,
    SOL_REDUCTION_CONTRACT,
    route_summary_sol_reduction,
)


def _source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "sol_h3"
        / "keyless_route_summary.py"
    ).read_text(encoding="utf-8")


def test_sol_reduction_k1_contract_is_versioned():
    assert (
        SOL_REDUCTION_CONTRACT
        == "sol-h3-keyless-route-summary-sol-reduction-v4"
    )


def test_sol_reduction_k1_reuses_exact_released_kv_reduction_for_vc():
    source = _source()
    assert "from ._vendor.sol_attn.preprocess import _reduce_kv_kernel" in source
    assert "_reduce_kv_kernel[grid](" in source
    assert "scratch_kc4 = torch.empty_like(rc4)" in source
    assert "TensorDescriptor.from_tensor(" in source


def test_sol_reduction_k1_rc_keeps_route_row_local_before_reduction():
    source = _source()
    rc_kernel = source[
        source.index("def _reduce_route_rc_sol_kernel"):
        source.index("def _reduce_route_vc_kernel")
    ]
    assert "mean_square = tl.sum(raw * raw, axis=1) / D" not in rc_kernel
    assert "route = tl.where(" in rc_kernel
    assert "summary = tl.sum(route.to(tl.float32), axis=0) / block_len" in rc_kernel
    assert "lane_offsets = tl.arange(0, 32)" in rc_kernel
    assert "lane_sum = tl.fma(lane3, lane3, lane_sum)" in rc_kernel
    assert "mean_square = tl.sum(lane_sum, axis=1) / D" in rc_kernel
    assert "torch.empty_like(v)" not in source[
        source.index("def route_summary_sol_reduction"):
        source.index("def route_summary(")
    ]


def test_sol_reduction_k1_fails_closed_on_cpu():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.bfloat16)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        route_summary_sol_reduction(v, weight, NORM_EPS, rope)


def test_sol_reduction_k1_rejects_wrong_dtype_before_gpu_execution():
    rows = 65
    v = torch.zeros((rows, 56, 128), dtype=torch.float32)
    weight = torch.ones((128,), dtype=torch.bfloat16)
    rope = torch.zeros((1, rows, 1, 48, 2, 2), dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="requires BF16"):
        route_summary_sol_reduction(v, weight, NORM_EPS, rope)
