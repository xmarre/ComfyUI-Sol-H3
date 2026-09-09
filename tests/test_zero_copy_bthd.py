"""Zero-copy BTHD bridge contracts for Sana SM120 SOL attention."""

import pytest
import torch

from sol_h3 import sparse
from sol_h3.contracts import Config
from sol_h3.runtime import Request


HEAD_DIM = 128


def _packed_component(rows, heads, component, *, device="cpu"):
    packed = torch.randn(
        1,
        rows,
        heads,
        3 * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    return packed, packed.split(HEAD_DIM, dim=-1)[component]


def _dense_bthd(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)


def test_sparse_bridge_preserves_strided_views_and_calibrates_per_layout(monkeypatch):
    """No-copy bridge identity follows the exact BTHD shape+stride contract."""
    _, q_bthd = _packed_component(5, 2, 0)
    kv_packed = torch.randn(1, 9, 2, 3 * HEAD_DIM, dtype=torch.bfloat16)
    _, k_bthd, v_bthd = kv_packed.split(HEAD_DIM, dim=-1)
    expected_ptrs = tuple(x.data_ptr() for x in (q_bthd, k_bthd, v_bthd))
    expected_strides = tuple(tuple(x.stride()) for x in (q_bthd, k_bthd, v_bthd))
    assert all(x.stride(-1) == 1 and not x.is_contiguous()
               for x in (q_bthd, k_bthd, v_bthd))

    calls = []

    def kernel(q, k, v, **_):
        calls.append((
            tuple(x.data_ptr() for x in (q, k, v)),
            tuple(tuple(x.stride()) for x in (q, k, v)),
        ))
        return torch.zeros_like(q)

    monkeypatch.setattr(sparse, "load_kernel", lambda device: kernel)
    monkeypatch.setattr(
        sparse,
        "_dense_reference",
        lambda q, k, v, dense_attention: torch.zeros(
            (q.shape[0], q.shape[2], q.shape[1], q.shape[3]),
            dtype=q.dtype,
            device=q.device,
        ),
    )

    cfg = Config(exact=False, backend="sol")
    state = Request(cfg)
    q, k, v = (x.transpose(1, 2) for x in (q_bthd, k_bthd, v_bthd))

    # First layout calibrates once; the second identical call must reuse it.
    sparse.attention(q, k, v, 0, cfg, state, recompute_prefix_queries=False)
    sparse.attention(q, k, v, 0, cfg, state, recompute_prefix_queries=False)
    assert calls[0] == (expected_ptrs, expected_strides)
    assert calls[1] == (expected_ptrs, expected_strides)
    assert len(state.gates) == 1

    # Same shapes but contiguous BTHD storage are a different CuTe layout and
    # therefore require an independent arithmetic gate.
    q2_bthd, k2_bthd, v2_bthd = (x.contiguous() for x in (q_bthd, k_bthd, v_bthd))
    q2, k2, v2 = (x.transpose(1, 2) for x in (q2_bthd, k2_bthd, v2_bthd))
    sparse.attention(q2, k2, v2, 0, cfg, state, recompute_prefix_queries=False)

    assert len(state.gates) == 2
    assert len(state.sparse_verified) == 2
    assert state.gates[0]["bthd_strides"] == [list(s) for s in expected_strides]
    assert state.gates[0]["bthd_strides"] != state.gates[1]["bthd_strides"]
    assert all(gate["materialized_qkv_bytes"] == 0 for gate in state.gates)
    assert all(gate["bthd_qkv_bytes"] > 0 for gate in state.gates)


@pytest.mark.gpu
@pytest.mark.parametrize("tq,tk", [(193, 193), (65, 449)])
def test_real_sm120_sparse_bridge_accepts_exact_strided_bthd_views(tq, tk):
    """Exercise the production transpose-view bridge on real SM120 CuTe."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires real SM120")

    torch.manual_seed(91 + tq + tk)
    q_packed = torch.randn(
        1, tq, 2, 3 * HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    if tq == tk:
        q_bthd, k_bthd, v_bthd = q_packed.split(HEAD_DIM, dim=-1)
    else:
        q_bthd = q_packed.split(HEAD_DIM, dim=-1)[0]
        kv_packed = torch.randn(
            1, tk, 2, 3 * HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        _, k_bthd, v_bthd = kv_packed.split(HEAD_DIM, dim=-1)

    assert all(x.stride(-1) == 1 and not x.is_contiguous()
               for x in (q_bthd, k_bthd, v_bthd))
    expected_strides = [list(x.stride()) for x in (q_bthd, k_bthd, v_bthd)]
    q, k, v = (x.transpose(1, 2) for x in (q_bthd, k_bthd, v_bthd))

    cfg = Config(exact=False, backend="sol")
    state = Request(cfg)
    with torch.inference_mode():
        # Making every KV row a sink turns both the calibration and production
        # call into an all-selected reference for this layout.
        got = sparse.attention(
            q,
            k,
            v,
            tk,
            cfg,
            state,
            recompute_prefix_queries=False,
        ).reshape(1, tq, 2, HEAD_DIM)
        want = _dense_bthd(q_bthd, k_bthd, v_bthd)
        metrics = sparse.error_metrics(got, want)

    assert sparse.arithmetic_gate_passes(metrics), metrics
    assert len(state.gates) == 1
    assert state.gates[0]["backend"] == "cute_sm120"
    assert state.gates[0]["bthd_strides"] == expected_strides
    assert state.gates[0]["materialized_qkv_bytes"] == 0
