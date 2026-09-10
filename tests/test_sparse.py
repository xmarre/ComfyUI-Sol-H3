import pytest
import torch
import torch.nn.functional as F

from sol_h3 import sparse
from sol_h3.contracts import Config
from sol_h3.runtime import Request


def test_real_loader_rejects_cpu():
    with pytest.raises(RuntimeError, match="SM120"):
        sparse.load_kernel(torch.device("cpu"))


def test_sana_loader_requires_cute_on_sm120(monkeypatch):
    from sol_h3._vendor.sol_attn import interface
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(interface, "_cute_runtime_available", lambda: False)
    with pytest.raises(RuntimeError, match="Sana selected triton; SM120 requires CuTe"):
        sparse.load_kernel(torch.device("cuda"))


def test_sink_geometry_rejects_nonprefix_and_out_of_range():
    assert sparse._sink_blocks(0, 0, 130) == [0, 0]
    assert sparse._sink_blocks(0, 64, 130) == [0, 1]
    assert sparse._sink_blocks(0, 65, 130) == [0, 2]
    assert sparse._sink_blocks(64, 64, 130) == [1, 2]
    with pytest.raises(RuntimeError, match="outside"):
        sparse._sink_blocks(0, 131, 130)


def test_production_calibration_metrics_allow_isolated_sm120_peak():
    # Real RTX PRO 6000 full-stack calibration reached this aggregate error but
    # the old max-only <=0.08 rule rejected a 0.125 peak. Aggregate error is far
    # inside the BF16 SDPA calibration budget, so that peak must not invalidate
    # an otherwise coherent approximate provider.
    metrics = {
        "finite": True,
        "max_abs": 0.125,
        "mean_abs": 0.0002412556204944849,
        "rel_l2": 0.0010161730460822582,
        "reference_peak_abs": 0.25,
        "catastrophic_max_abs_limit": 1.0,
    }
    assert sparse.arithmetic_gate_passes(metrics)


def test_arithmetic_gate_rejects_broad_mean_error():
    metrics = {
        "finite": True,
        "max_abs": 0.10,
        "mean_abs": sparse.ARITH_MEAN_ABS_LIMIT + 1e-4,
        "rel_l2": 0.002,
        "reference_peak_abs": 0.25,
        "catastrophic_max_abs_limit": 1.0,
    }
    assert not sparse.arithmetic_gate_passes(metrics)


def test_arithmetic_gate_rejects_bad_relative_error():
    metrics = {
        "finite": True,
        "max_abs": 0.10,
        "mean_abs": 0.001,
        "rel_l2": sparse.ARITH_REL_L2_LIMIT + 1e-4,
        "reference_peak_abs": 0.25,
        "catastrophic_max_abs_limit": 1.0,
    }
    assert not sparse.arithmetic_gate_passes(metrics)


def test_arithmetic_gate_rejects_catastrophic_peak_and_nonfinite():
    base = {
        "finite": True,
        "max_abs": 0.10,
        "mean_abs": 0.001,
        "rel_l2": 0.002,
        "reference_peak_abs": 0.01,
        "catastrophic_max_abs_limit": sparse.ARITH_CATASTROPHIC_MAX_FLOOR,
    }
    assert not sparse.arithmetic_gate_passes({
        **base,
        "max_abs": sparse.ARITH_CATASTROPHIC_MAX_FLOOR + 0.1,
    })
    assert not sparse.arithmetic_gate_passes({**base, "finite": False})


def test_error_metrics_report_scale_aware_peak_cap():
    want = torch.ones(1, 1, 1000, 1)
    got = want.clone()
    got.reshape(-1)[0] += 0.125
    metrics = sparse.error_metrics(got, want)
    assert metrics["finite"] is True
    assert metrics["max_abs"] == pytest.approx(0.125)
    assert metrics["reference_peak_abs"] == pytest.approx(1.0)
    assert metrics["catastrophic_max_abs_limit"] == pytest.approx(4.0)
    assert sparse.arithmetic_gate_passes(metrics)


def test_bridge_protects_prefix_and_uses_full_sink_gate(monkeypatch):
    # CPU bridge oracle only: deliberately fake the sparse output after the
    # all-selected call to verify that ALL prefix queries are overwritten.
    calls = []
    dense_calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        dense = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return dense if kw["sink_tokens"] == q.shape[1] else torch.full_like(dense, 9)

    def dense_attention(q, k, v):
        dense_calls.append(q.shape[2])
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q, k, v = (torch.randn(1, 2, 11, 128).to(torch.bfloat16) for _ in range(3))
    config = Config(exact=False, backend="sol")
    state = Request(config)
    out = sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention).reshape(1, 11, 2, 128)
    want = F.scaled_dot_product_attention(q[:, :, :5], k, v).transpose(1, 2)
    assert torch.equal(out[:, :5], want)
    assert (out[:, 5:] == 9).all()
    assert [c["sink_tokens"] for c in calls] == [11, 5]
    assert all(c["sink_start"] == 0 for c in calls)
    assert dense_calls == [5]
    assert state.sparse_calls == 1
    assert state.gates[0]["kernel_loader_s"] >= 0.0
    assert state.gates[0]["gate_wall_s"] >= 0.0
    assert state.gates[0]["materialized_qkv_bytes"] == 0
    assert state.gates[0]["bthd_qkv_bytes"] == 3 * q.numel() * q.element_size()
    sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention)
    assert len(calls) == 3  # same-request shape+stride gate reused
    assert dense_calls == [5, 5]


def test_failed_arithmetic_never_counts_sparse(monkeypatch):
    monkeypatch.setattr(
        sparse,
        "load_kernel",
        lambda device: lambda q, k, v, **kw: torch.full_like(v, float("nan")),
    )
    state = Request(Config(exact=False, backend="sol"))
    q = torch.ones(1, 1, 3, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="arithmetic gate failed"):
        sparse.attention(q, q, q, 1, state.config, state)
    assert state.sparse_calls == 0



def test_weighted_bridge_uses_measure_specific_gate_and_exact_k_range(monkeypatch):
    calls = []

    def kernel(q, k, v, **kw):
        calls.append(kw)
        bias = kw.get("key_bias")
        mask = None if bias is None else bias.view(1, 1, 1, -1)
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    q, k, v = (torch.randn(1, 2, 260, 128).to(torch.bfloat16) for _ in range(3))
    bias = torch.zeros(260, dtype=torch.float32)
    bias[64:192] = torch.log(torch.tensor(0.5))
    state = Request(Config(exact=False, backend="sol"))
    out = sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=bias,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-a",
    )
    assert out.shape == (1, 260, 256)
    assert [c["sink_tokens"] for c in calls] == [260, 192]
    assert all(c["sink_start"] == 0 for c in calls)
    assert all(c["key_bias"] is bias for c in calls)
    assert len(state.gates) == 1

    sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=bias,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-a",
    )
    assert len(calls) == 3
    assert len(state.gates) == 1

    other = bias.clone()
    other[64:192] = torch.log(torch.tensor(0.25))
    sparse.attention(
        q, k, v, 5, state.config, state,
        key_bias=other,
        exact_k_blocks=(0, 3),
        calibration_identity="measure-b",
    )
    assert len(calls) == 5
    assert len(state.gates) == 2


def test_weighted_bridge_rejects_bias_without_bound_route_metadata(monkeypatch):
    monkeypatch.setattr(sparse, "load_kernel", lambda device: None)
    q = torch.ones(1, 1, 65, 128, dtype=torch.bfloat16)
    state = Request(Config(exact=False, backend="sol"))
    with pytest.raises(RuntimeError, match="calibration identity"):
        sparse.attention(q, q, q, 1, state.config, state, key_bias=torch.zeros(65))
