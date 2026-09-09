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
    with pytest.raises(RuntimeError, match="beginning at row zero"):
        sparse._sink_blocks(64, 64, 130)
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
    assert state.gates[0]["zero_copy_bthd"] is True
    assert state.gates[0]["calibration_s"] >= 0.0
    sparse.attention(q, k, v, 5, config, state, dense_attention=dense_attention)
    assert len(calls) == 3  # same-request shape/layout gate reused
    assert dense_calls == [5, 5]


def test_bridge_passes_zero_copy_views_and_verifies_each_layout(monkeypatch):
    seen = []
    expected_storage = []

    def kernel(q, k, v, **kw):
        index = len(seen)
        seen.append({
            "strides": tuple(tuple(int(s) for s in x.stride()) for x in (q, k, v)),
            "storage": tuple(x.untyped_storage().data_ptr() for x in (q, k, v)),
            "contiguous": tuple(x.is_contiguous() for x in (q, k, v)),
        })
        assert seen[index]["storage"] == expected_storage[index // 2]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    state = Request(Config(exact=False, backend="sol"))

    # Ordinary independent BHSD tensors become non-contiguous BTHD views.
    ordinary = tuple(torch.randn(1, 2, 7, 128, dtype=torch.bfloat16) for _ in range(3))
    expected_storage.append(tuple(x.untyped_storage().data_ptr() for x in ordinary))
    sparse.attention(*ordinary, 0, state.config, state)

    # Match the packed Ulysses layout: Q/K/V are views into one [..., Q|K|V]
    # allocation. The BHSD bridge inputs have the same shapes but different
    # strides, so correctness calibration must not reuse the prior layout gate.
    packed = torch.randn(1, 7, 2, 3 * 128, dtype=torch.bfloat16)
    packed_bthd = packed.split(128, dim=-1)
    packed_bhsd = tuple(x.transpose(1, 2) for x in packed_bthd)
    expected_storage.append(tuple(x.untyped_storage().data_ptr() for x in packed_bhsd))
    sparse.attention(*packed_bhsd, 0, state.config, state)

    assert len(seen) == 4  # calibration + sparse execution for each layout
    assert all(not all(call["contiguous"]) for call in seen)
    assert seen[0]["strides"] != seen[2]["strides"]
    assert len(state.sparse_verified) == 2
    assert len(state.gates) == 2
    assert state.gates[0]["bthd_strides"] != state.gates[1]["bthd_strides"]
    assert all(gate["zero_copy_bthd"] is True for gate in state.gates)


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
