import random

import torch

from sol_h3.diagnostics import CudaDiagnosticState
from sol_h3.replay_diagnostics import (
    ARMS,
    ReplayDiagnosticState,
    clone_tensors_preserve_layout,
)


def _key():
    return {
        "api": 1,
        "mode": "ordinary_unweighted_v1",
        "layout": "bhtd",
        "dtype": "torch.bfloat16",
        "batch": 1,
        "heads": 2,
        "head_dim": 128,
        "q_rows": 8,
        "kv_rows": 8,
        "q_stride": [2048, 128, 256, 1],
        "k_stride": [2048, 128, 256, 1],
        "v_stride": [2048, 128, 256, 1],
        "alignment": 16,
        "scale": float(128**-0.5).hex(),
        "source_generation": "source",
        "implementation_generation": "impl",
        "runtime_identity": "runtime",
        "compiler_namespace": "compiler",
        "validation_generation": 0,
        "bias": None,
        "physical_identity": None,
    }


def test_replay_uses_fresh_then_fresh_then_retained_validation():
    state = ReplayDiagnosticState(enabled=True)
    diagnostics = CudaDiagnosticState(enabled=True, event_factory=None)
    diagnostics.enabled = True

    class NoCudaDiagnostics:
        enabled = True

        @staticmethod
        def begin_sample(*args, **kwargs):
            return None

        @staticmethod
        def initial_stream_drain(sample):
            return None

    compiled = False
    gates = []
    productions = []

    def gate(_sample, arm):
        nonlocal compiled
        telemetry = {"compiler_key": "key"}
        if compiled:
            telemetry["compile_hit"] = True
        else:
            telemetry["compile_miss"] = True
            compiled = True
        gates.append(arm)
        return {
            "finite": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rel_l2": 0.0,
            "reference_peak_abs": 1.0,
            "catastrophic_max_abs_limit": 4.0,
        }, telemetry

    def production(_sample, arm):
        productions.append(arm)
        return {"compiler_key": "key", "compile_hit": True}

    random.seed(73)
    before = random.getstate()
    state.execute(
        target="ordinary_low",
        arithmetic_key=_key(),
        device=torch.device("cpu"),
        context={"flow_stage": "low"},
        cuda_diagnostics=NoCudaDiagnostics(),
        gate=gate,
        production=production,
    )

    summary = state.summary()
    assert summary["errors"] == 0
    assert summary["completed_targets"] == ["ordinary_low"]
    report = summary["reports"][0]
    assert report["rng_restored"] is True
    assert random.getstate() == before
    assert gates == list(ARMS[:2])
    assert productions == list(ARMS)
    assert report["arms"][0]["gate_performed"] is True
    assert report["arms"][0]["compile_misses"] == 1
    assert report["arms"][1]["gate_performed"] is True
    assert report["arms"][1]["compile_misses"] == 0
    assert report["arms"][2]["gate_performed"] is False
    assert report["arms"][2]["proof_hit"] is True
    assert report["arms"][2]["compile_misses"] == 0


def test_replay_target_selection_is_bounded_and_stage_specific():
    state = ReplayDiagnosticState(enabled=True)
    assert state.claim_ordinary({"flow_stage": "single"}) is None
    assert state.claim_ordinary({"flow_stage": "low"}) == "ordinary_low"
    assert state.claim_ordinary({"flow_stage": "low"}) is None
    assert (
        state.claim_ordinary({"flow_stage": "high"})
        == "ordinary_continuation_high"
    )
    assert (
        state.claim_partitioned_suffix(kind="local", mapped=True, force_dense=False)
        == "partitioned_suffix"
    )
    assert (
        state.claim_partitioned_suffix(kind="local", mapped=True, force_dense=False)
        is None
    )


def test_layout_clone_preserves_interleaved_qkv_strides_and_values():
    rows = 7
    heads = 2
    dim = 4
    inner = heads * dim
    packed = torch.arange(rows * 3 * inner, dtype=torch.float32).reshape(rows, 3 * inner)
    q, k, v = (
        part.view(rows, heads, dim).transpose(0, 1).unsqueeze(0)
        for part in packed.split(inner, dim=-1)
    )
    copies = clone_tensors_preserve_layout((q, k, v))

    assert [tuple(tensor.shape) for tensor in copies] == [
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
    ]
    assert [tuple(tensor.stride()) for tensor in copies] == [
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
    ]
    for original, copied in zip((q, k, v), copies, strict=True):
        assert torch.equal(original, copied)
    assert copies[0].untyped_storage().data_ptr() == copies[1].untyped_storage().data_ptr()
    assert copies[1].untyped_storage().data_ptr() == copies[2].untyped_storage().data_ptr()
    assert copies[0].untyped_storage().data_ptr() != q.untyped_storage().data_ptr()
