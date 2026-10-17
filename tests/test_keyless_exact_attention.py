from __future__ import annotations

import pytest
import torch

from sol_h3 import keyless_exact_attention as exact
from sol_h3 import keyless_route_summary as summary


def _rope(rows: int, dtype=torch.float32):
    torch.manual_seed(41)
    angles = torch.randn(rows, summary.ROPE_HALF_DIM, dtype=torch.float32) * 0.2
    rope = torch.zeros(
        1, rows, 1, summary.ROPE_HALF_DIM, 2, 2, dtype=dtype
    )
    rope[0, :, 0, :, 0, 0] = angles.cos().to(dtype)
    rope[0, :, 0, :, 1, 0] = angles.sin().to(dtype)
    return rope


def _manual_attention(q, v, weight, rope):
    route = summary.materialized_route_reference(
        v, weight, summary.NORM_EPS, rope
    )
    logits = torch.einsum(
        "qhd,khd->hqk", q.float(), route.float()
    ) * exact.CANONICAL_SCALE
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum(
        "hqk,khd->qhd", probs, v.float()
    ).to(v.dtype)


def test_materialized_exact_reference_scores_route_and_retrieves_raw_v():
    torch.manual_seed(42)
    q = torch.randn(7, 3, summary.HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(9, 3, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.linspace(
        0.4, 1.6, summary.HEAD_DIM, dtype=torch.bfloat16
    )
    rope = _rope(v.shape[0])

    got = exact.materialized_exact_reference(
        q, v, weight, summary.NORM_EPS, rope
    )
    expected = _manual_attention(q, v, weight, rope)

    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)


def test_materialized_exact_reference_is_not_route_as_retrieval_value():
    torch.manual_seed(43)
    q = torch.randn(5, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(8, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.linspace(
        0.25, 1.75, summary.HEAD_DIM, dtype=torch.bfloat16
    )
    rope = _rope(v.shape[0])
    route = summary.materialized_route_reference(
        v, weight, summary.NORM_EPS, rope
    )

    correct = exact.materialized_exact_reference(
        q, v, weight, summary.NORM_EPS, rope
    )
    q4 = q.transpose(0, 1).unsqueeze(0)
    r4 = route.transpose(0, 1).unsqueeze(0)
    wrong = torch.nn.functional.scaled_dot_product_attention(
        q4,
        r4,
        r4,
        dropout_p=0.0,
        scale=exact.CANONICAL_SCALE,
    ).squeeze(0).transpose(0, 1)

    assert float((correct.float() - wrong.float()).abs().max()) > 1e-3


def test_materialized_exact_reference_supports_rectangular_q_v_rows():
    torch.manual_seed(44)
    q = torch.randn(3, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(11, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    out = exact.materialized_exact_reference(
        q, v, weight, summary.NORM_EPS, rope
    )
    assert out.shape == q.shape


def test_exact_attention_fails_closed_without_cuda():
    q = torch.ones(3, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    v = torch.ones(4, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        exact.exact_attention(q, v, weight, summary.NORM_EPS, rope)


def test_exact_attention_rejects_incompatible_head_geometry():
    q = torch.ones(3, 2, 64, dtype=torch.bfloat16)
    v = torch.ones(4, 2, 64, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    with pytest.raises(ValueError, match=r"Q \[Tq,H,128\]"):
        exact.exact_attention(q, v, weight, summary.NORM_EPS, rope)


def test_k2_contract_is_explicit_and_separate_from_k1():
    assert exact.CONTRACT == "sol-h3-keyless-exact-allselected-v3-k1-route-identity"
    assert summary.CONTRACT == "sol-h3-keyless-route-summary-v2"
    assert exact.CONTRACT != summary.CONTRACT
    assert exact.BLOCK_M == 64
    assert exact.BLOCK_N == 64


def test_k2_source_uses_fp32_rope_compute_after_bf16_norm():
    from pathlib import Path

    source = Path(exact.__file__).read_text(encoding="utf-8")
    assert ").to(tl.bfloat16)" in source
    assert "norm_math = norm.to(tl.float32)" in source
    assert "partner_math = partner_norm.to(tl.float32)" in source
    assert "cos_values = tl.load(" in source
    assert ").to(tl.float32)" in source


def test_k2_v3_uses_k1_comfy_lane_fma_rms_ordering():
    from pathlib import Path

    source = Path(exact.__file__).read_text(encoding="utf-8")
    kernel = source[
        source.index("def _keyless_exact_allselected_kernel"):
        source.index("else:\n    _keyless_exact_allselected_kernel = None")
    ]
    assert "lane_offsets = tl.arange(0, 32)" in kernel
    assert "lane_sum = tl.fma(lane0, lane0, lane_sum)" in kernel
    assert "lane_sum = tl.fma(lane1, lane1, lane_sum)" in kernel
    assert "lane_sum = tl.fma(lane2, lane2, lane_sum)" in kernel
    assert "lane_sum = tl.fma(lane3, lane3, lane_sum)" in kernel
    assert "mean_square = tl.sum(lane_sum, axis=1) / head_dim" in kernel
    assert "mean_square = tl.sum(raw * raw, axis=1) / head_dim" not in kernel
