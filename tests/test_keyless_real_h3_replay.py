from __future__ import annotations

import pytest
import torch

from sol_h3.keyless_real_h3_replay import (
    ENVELOPE,
    apply_split_half_rope_fp32_from_normalized,
    apply_split_half_rope_from_normalized,
    block_summary_oracle,
    checkpoint_tensor_names,
    identity_split_half_rope_like,
    metric_within_limit,
    public_rms_norm,
    replay_requires_process_failure,
    split_projection,
    tensor_metrics,
    tensor_scale_diagnostics,
    value_sum_within_limit,
)


def test_frozen_replay_envelope_values():
    assert ENVELOPE.contract == "sol-h3-keyless-real-h3-replay-envelope-v1"
    assert ENVELOPE.k1_route_centroid.rel_l2 == 0.004
    assert ENVELOPE.k1_route_centroid.mean_abs == 0.0005
    assert ENVELOPE.k1_route_centroid.max_abs == 0.04
    assert ENVELOPE.k2_output.rel_l2 == 0.0045
    assert ENVELOPE.k2_output.mean_abs == 0.00065
    assert ENVELOPE.k2_output.max_abs == 0.02
    assert ENVELOPE.k1_value_max_abs == 0.0


def test_metric_gate_is_inclusive_and_fails_nonfinite():
    limit = ENVELOPE.k2_output
    passing = {
        "finite": True,
        "rel_l2": limit.rel_l2,
        "mean_abs": limit.mean_abs,
        "max_abs": limit.max_abs,
    }
    assert metric_within_limit(passing, limit)
    assert not metric_within_limit({**passing, "finite": False}, limit)
    assert not metric_within_limit(
        {**passing, "max_abs": limit.max_abs + 1e-6},
        limit,
    )


def test_value_sum_gate_keeps_exact_current_contract():
    exact = {"finite": True, "max_abs": 0.0}
    changed = {"finite": True, "max_abs": 1e-7}
    assert value_sum_within_limit(exact, ENVELOPE.k1_value_max_abs)
    assert not value_sum_within_limit(changed, ENVELOPE.k1_value_max_abs)


def test_tensor_metrics_zero_delta():
    x = torch.randn(5, 3, 7)
    metrics = tensor_metrics(x, x.clone())
    assert metrics == {
        "finite": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rel_l2": 0.0,
    }


def test_block_summary_oracle_handles_partial_last_block():
    route = torch.arange(5 * 2 * 3, dtype=torch.float32).reshape(5, 2, 3)
    values = route + 1
    rc, vc = block_summary_oracle(route, values, block_size=4)
    assert rc.shape == (2, 2, 3)
    assert vc.shape == (2, 2, 3)
    torch.testing.assert_close(rc[0], route[:4].mean(dim=0))
    torch.testing.assert_close(rc[1], route[4])
    torch.testing.assert_close(vc[0], values[:4].sum(dim=0))
    torch.testing.assert_close(vc[1], values[4])


def test_checkpoint_tensor_names_distinguish_teacher_and_keyless():
    teacher = checkpoint_tensor_names("teacher", 25)
    keyless = checkpoint_tensor_names("keyless", 25)
    assert teacher["projection"] == "blocks.25.attn.qkv_proj.weight"
    assert teacher["route_norm"] == "blocks.25.attn.k_norm.weight"
    assert keyless["projection"] == "blocks.25.attn.qv_proj.weight"
    assert keyless["route_norm"] == "blocks.25.attn.route_norm.weight"
    with pytest.raises(ValueError, match="unsupported checkpoint kind"):
        checkpoint_tensor_names("fake", 0)


def test_split_projection_drops_teacher_k_and_preserves_keyless_qv():
    heads, dim, hidden = 2, 3, 5
    inner = heads * dim
    teacher = torch.arange(3 * inner * hidden).reshape(3 * inner, hidden)
    q, v = split_projection(
        teacher,
        kind="teacher",
        heads=heads,
        head_dim=dim,
        hidden_size=hidden,
    )
    assert torch.equal(q, teacher[:inner])
    assert torch.equal(v, teacher[2 * inner:])

    keyless = torch.arange(2 * inner * hidden).reshape(2 * inner, hidden)
    q2, v2 = split_projection(
        keyless,
        kind="keyless",
        heads=heads,
        head_dim=dim,
        hidden_size=hidden,
    )
    assert torch.equal(q2, keyless[:inner])
    assert torch.equal(v2, keyless[inner:])


def test_split_projection_rejects_wrong_geometry():
    with pytest.raises(ValueError, match="teacher qkv projection expected"):
        split_projection(torch.zeros(1, 1), kind="teacher")
    with pytest.raises(ValueError, match="Keyless qv projection expected"):
        split_projection(torch.zeros(1, 1), kind="keyless")


def test_tensor_scale_diagnostics_reports_worst_element():
    want = torch.tensor([[1.0, 16.0], [2.0, -4.0]], dtype=torch.bfloat16)
    got = want.clone()
    got[0, 1] = torch.tensor(16.25, dtype=torch.bfloat16)
    diagnostics = tensor_scale_diagnostics(got, want)
    assert diagnostics["finite"] is True
    assert diagnostics["want_abs_max"] == 16.0
    assert diagnostics["got_abs_max"] == 16.25
    assert diagnostics["worst"]["index"] == [0, 1]
    assert diagnostics["worst"]["want"] == 16.0
    assert diagnostics["worst"]["got"] == 16.25
    assert diagnostics["worst"]["delta"] == 0.25
    assert diagnostics["worst"]["want_bf16_ulp_estimate"] == 0.125
    assert diagnostics["worst"]["abs_error_in_want_bf16_ulps"] == 2.0


def test_identity_split_half_rope_like_is_exact_identity():
    rope = torch.randn(1, 3, 1, 2, 2, 2, dtype=torch.bfloat16)
    identity = identity_split_half_rope_like(rope)
    assert identity.shape == rope.shape
    assert identity.dtype == rope.dtype
    assert torch.count_nonzero(identity[..., 0, 0] - 1) == 0
    assert torch.count_nonzero(identity[..., 1, 1] - 1) == 0
    assert torch.count_nonzero(identity[..., 0, 1]) == 0
    assert torch.count_nonzero(identity[..., 1, 0]) == 0


def test_public_rms_norm_matches_explicit_formula():
    raw = torch.tensor(
        [[[1.0, -2.0, 3.0, -4.0], [0.5, 1.5, -2.5, 3.5]]],
        dtype=torch.bfloat16,
    )
    weight = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.bfloat16)
    got = public_rms_norm(raw, weight, 1e-5)
    work = raw.float()
    inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + 1e-5)
    want = (work * inv * weight.float()).to(torch.bfloat16)
    assert torch.equal(got, want)


def test_apply_split_half_rope_identity_preserves_normalized_tensor():
    normalized = torch.randn(3, 2, 6, dtype=torch.bfloat16)
    rope = torch.zeros(1, 3, 1, 2, 2, 2, dtype=torch.bfloat16)
    rope[..., 0, 0] = 1
    rope[..., 1, 1] = 1
    got = apply_split_half_rope_from_normalized(normalized, rope, rot_dim=4)
    assert torch.equal(got, normalized)


def test_apply_split_half_rope_uses_full_rotation_matrix_and_preserves_tail():
    normalized = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 9.0, 10.0]]],
        dtype=torch.bfloat16,
    )
    rope = torch.zeros(1, 1, 1, 2, 2, 2, dtype=torch.bfloat16)
    rope[..., 0, 0] = 2
    rope[..., 0, 1] = 3
    rope[..., 1, 0] = 5
    rope[..., 1, 1] = 7
    got = apply_split_half_rope_from_normalized(normalized, rope, rot_dim=4)
    want = torch.tensor(
        [[[11.0, 16.0, 26.0, 38.0, 9.0, 10.0]]],
        dtype=torch.bfloat16,
    )
    assert torch.equal(got, want)


def test_diagnostic_only_preserves_gate_but_suppresses_process_failure():
    assert replay_requires_process_failure(
        all_blocks_pass=False,
        diagnostic_only=False,
    )
    assert not replay_requires_process_failure(
        all_blocks_pass=False,
        diagnostic_only=True,
    )
    assert not replay_requires_process_failure(
        all_blocks_pass=True,
        diagnostic_only=False,
    )


def test_apply_split_half_rope_fp32_promotes_rotation_math():
    normalized = torch.tensor(
        [[[1.234375, -0.765625, 0.6171875, -1.1171875, 9.0, 10.0]]],
        dtype=torch.bfloat16,
    )
    rope = torch.zeros(1, 1, 1, 2, 2, 2, dtype=torch.bfloat16)
    rope[..., 0, 0] = torch.tensor(0.69921875, dtype=torch.bfloat16)
    rope[..., 0, 1] = torch.tensor(-0.71484375, dtype=torch.bfloat16)
    rope[..., 1, 0] = torch.tensor(0.71484375, dtype=torch.bfloat16)
    rope[..., 1, 1] = torch.tensor(0.69921875, dtype=torch.bfloat16)

    got = apply_split_half_rope_fp32_from_normalized(normalized, rope, rot_dim=4)
    freqs = rope.float()[0, 0, 0]
    a = normalized[..., :2].float()
    b = normalized[..., 2:4].float()
    first = freqs[:, 0, 0].view(1, 1, 2) * a + freqs[:, 0, 1].view(1, 1, 2) * b
    second = freqs[:, 1, 0].view(1, 1, 2) * a + freqs[:, 1, 1].view(1, 1, 2) * b
    want = torch.cat(
        ((torch.cat((first, second), dim=-1)).to(torch.bfloat16), normalized[..., 4:]),
        dim=-1,
    )
    assert torch.equal(got, want)
