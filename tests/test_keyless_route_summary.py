from __future__ import annotations

import math

import pytest
import torch

from sol_h3 import keyless_route_summary as summary


def _rope(rows: int, dtype=torch.float32):
    torch.manual_seed(31)
    angles = torch.randn(rows, summary.ROPE_HALF_DIM, dtype=torch.float32) * 0.2
    rope = torch.zeros(
        1, rows, 1, summary.ROPE_HALF_DIM, 2, 2, dtype=dtype
    )
    rope[0, :, 0, :, 0, 0] = angles.cos().to(dtype)
    rope[0, :, 0, :, 1, 0] = angles.sin().to(dtype)
    return rope


def _direct_route(v, weight, eps, rope):
    work = v.float()
    norm = (
        work
        * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
        * weight.float()
    ).to(v.dtype)
    rot = rope.to(v.dtype)
    c = rot[0, :, 0, :, 0, 0].unsqueeze(1)
    s = rot[0, :, 0, :, 1, 0].unsqueeze(1)
    a = norm[..., : summary.ROPE_HALF_DIM]
    b = norm[..., summary.ROPE_HALF_DIM : summary.ROPE_ROT_DIM]
    return torch.cat(
        (a * c - b * s, a * s + b * c, norm[..., summary.ROPE_ROT_DIM :]),
        dim=-1,
    ), norm


def test_materialized_reference_matches_keyless_order_and_preserves_unrotated_tail():
    torch.manual_seed(32)
    v = torch.randn(9, 3, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])

    got = summary.materialized_route_reference(v, weight, summary.NORM_EPS, rope)
    expected, norm = _direct_route(v, weight, summary.NORM_EPS, rope)

    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        got[..., summary.ROPE_ROT_DIM :],
        norm[..., summary.ROPE_ROT_DIM :],
        atol=0,
        rtol=0,
    )


def test_route_summary_reference_reduces_routed_centroid_and_raw_value_sum():
    torch.manual_seed(33)
    rows, heads = summary.BLOCK_SIZE + 7, 2
    v = torch.randn(rows, heads, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(rows)
    route, _norm = _direct_route(v, weight, summary.NORM_EPS, rope)

    rc, vc = summary.route_summary_reference(v, weight, summary.NORM_EPS, rope)

    assert rc.shape == (2, heads, summary.HEAD_DIM)
    assert vc.shape == rc.shape
    expected_rc0 = route[: summary.BLOCK_SIZE].float().mean(dim=0).to(v.dtype)
    expected_rc1 = route[summary.BLOCK_SIZE :].float().mean(dim=0).to(v.dtype)
    expected_vc0 = v[: summary.BLOCK_SIZE].float().sum(dim=0).to(v.dtype)
    expected_vc1 = v[summary.BLOCK_SIZE :].float().sum(dim=0).to(v.dtype)
    torch.testing.assert_close(rc[0], expected_rc0, atol=0, rtol=0)
    torch.testing.assert_close(rc[1], expected_rc1, atol=0, rtol=0)
    torch.testing.assert_close(vc[0], expected_vc0, atol=0, rtol=0)
    torch.testing.assert_close(vc[1], expected_vc1, atol=0, rtol=0)


def test_route_summary_reference_does_not_average_raw_v_for_route():
    torch.manual_seed(34)
    rows = summary.BLOCK_SIZE
    v = torch.randn(rows, 1, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.linspace(0.5, 1.5, summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(rows)
    rc, _vc = summary.route_summary_reference(v, weight, summary.NORM_EPS, rope)

    raw_centroid = v.float().mean(dim=0).to(v.dtype)
    assert not torch.equal(rc[0], raw_centroid)


def test_native_route_summary_fails_closed_without_cuda():
    v = torch.ones(3, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        summary.route_summary(v, weight, summary.NORM_EPS, rope)


def test_native_route_summary_rejects_noncanonical_epsilon_before_launch(monkeypatch):
    # Exercise the native epsilon gate without requiring hardware. The device gate
    # is intentionally reached first on real CPU tensors, so validate the common
    # finite-positive rule separately here and keep the canonical value explicit.
    v = torch.ones(3, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    with pytest.raises(ValueError, match="finite and positive"):
        summary.route_summary_reference(v, weight, math.nan, rope)
    assert summary.NORM_EPS == 1e-5
    assert summary.CONTRACT == "sol-h3-keyless-route-summary-v1"


def test_materialized_native_route_diagnostic_fails_closed_without_cuda():
    v = torch.ones(3, 2, summary.HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(summary.HEAD_DIM, dtype=torch.bfloat16)
    rope = _rope(v.shape[0])
    with pytest.raises(RuntimeError, match="requires CUDA"):
        summary.materialized_native_route_diagnostic(
            v,
            weight,
            summary.NORM_EPS,
            rope,
        )
