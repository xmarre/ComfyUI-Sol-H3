from types import SimpleNamespace

import torch

from sol_h3 import mapped_neighbors, runtime, sparse
from sol_h3.contracts import Config
from sol_h3.interop import RECEIPTS_KEY, VDN_KEY_V4, HistoryPolicy
from sol_h3.runtime import BlockPatch, Request, _FORWARD, _REQUEST


OWNER = "vdn-runtime-owner"
PLAN = "a" * 64


def _wire(*, owner=OWNER, kv_begin=65):
    return (
        "vdn_query_positions",
        1,
        owner,
        PLAN,
        0,
        64,
        256,
        1,
        ((0, 64, kv_begin),),
    )


def _summary(wire):
    return SimpleNamespace(
        tag="vdn_query_position_plan_v1",
        schema=1,
        mode="grouped",
        owner_generation=OWNER,
        plan_digest=PLAN,
        seq_len=256,
        video_start=1,
        video_end=256,
        num_frames=1,
        tokens_per_frame=64,
        anchor_frames="none",
        groups=(wire,),
    )


def test_v4_success_records_only_completed_request_owned_mapped_receipt(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    wire = _wire()
    summary = _summary(wire)

    def forward(*args, **kwargs):
        raise AssertionError("test forward is only a provider-v4 ownership hook")

    forward._vdn_forward = True
    forward.vdn_query_position_plan_v1 = lambda options, layout: summary
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=forward))])

    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        mapped_neighbors,
        "device_descriptor",
        lambda state, descriptor, device: torch.tensor(
            descriptor.intervals, dtype=torch.int32, device=device
        ),
    )
    calls = []

    def attention(q, k, v, prefix, config, got_state, **kwargs):
        calls.append((q.shape, k.shape, prefix, kwargs))
        assert got_state is state
        mapped = kwargs["mapped_neighbor_intervals"]
        assert mapped.dtype == torch.int32
        assert tuple(mapped.shape) == (1, 2)
        assert kwargs["recompute_prefix_queries"] is False
        return torch.zeros((1, q.shape[2], q.shape[1] * q.shape[3]), dtype=q.dtype)

    monkeypatch.setattr(sparse, "attention", attention)
    q = torch.zeros((64, 1, 128), dtype=torch.bfloat16)
    k = torch.zeros((256, 1, 128), dtype=q.dtype)
    v = torch.zeros_like(k)
    receipts = []
    options = {RECEIPTS_KEY: receipts}
    layout = SimpleNamespace()

    request_token = _REQUEST.set(state)
    forward_token = _FORWARD.set((model, state, 0, set(), []))
    try:
        def block(args):
            provider = args["transformer_options"][VDN_KEY_V4]
            got = provider(
                lambda: (_ for _ in ()).throw(AssertionError("mapped success used native")),
                q,
                k,
                v,
                kind="local",
                scale=128 ** -0.5,
                sink_rows=1,
                query_position_map=wire,
            )
            assert got.shape == q.shape
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {
                "img": torch.zeros((1, 128)),
                "layout": layout,
                "transformer_options": options,
            },
            {"original_block": block},
        )

        assert len(calls) == 1
        assert state.vdn_mapped_sol_calls == 1
        assert state.vdn_local_sol_calls == 1
        assert state.vdn_rectangular_sol_calls == 1
        assert len(receipts) == 1
        item = receipts[0]
        assert item[:3] == ("sol_h3", 0, "vdn_local_sol_mapped_v1")
        assert (0, item[3]) in state.mapped_validated_receipts
        assert HistoryPolicy(cfg).accept_receipts(receipts) is True

        forged = list(item[3])
        forged[7] = "b" * 64
        assert HistoryPolicy(cfg).accept_receipts(
            [(item[0], item[1], item[2], tuple(forged))]
        ) is False
    finally:
        _FORWARD.reset(forward_token)
        _REQUEST.reset(request_token)


def test_v4_stale_owner_falls_native_and_is_actual_only(monkeypatch):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    expected_wire = _wire()
    stale_wire = _wire(owner="other-owner")
    summary = _summary(expected_wire)

    def forward(*args, **kwargs):
        raise AssertionError("test forward is only a provider-v4 ownership hook")

    forward._vdn_forward = True
    forward.vdn_query_position_plan_v1 = lambda options, layout: summary
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(forward=forward))])
    monkeypatch.setattr(runtime, "_shape_reason", lambda *a, **k: None)
    monkeypatch.setattr(
        sparse,
        "attention",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stale map reached sparse attention")),
    )

    q = torch.zeros((64, 1, 128), dtype=torch.bfloat16)
    k = torch.zeros((256, 1, 128), dtype=q.dtype)
    receipts = []
    native_calls = []
    request_token = _REQUEST.set(state)
    forward_token = _FORWARD.set((model, state, 0, set(), []))
    try:
        def block(args):
            provider = args["transformer_options"][VDN_KEY_V4]
            got = provider(
                lambda: native_calls.append(True) or q,
                q,
                k,
                k,
                kind="local",
                scale=128 ** -0.5,
                sink_rows=1,
                query_position_map=stale_wire,
            )
            assert got is q
            return {"img": args["img"]}

        BlockPatch(0, cfg)(
            {
                "img": torch.zeros((1, 128)),
                "layout": SimpleNamespace(),
                "transformer_options": {RECEIPTS_KEY: receipts},
            },
            {"original_block": block},
        )
        assert native_calls == [True]
        assert state.vdn_mapped_sol_calls == 0
        assert receipts == [("sol_h3", 0, "vdn_local_native_mapping:owner")]
        assert HistoryPolicy(cfg).accept_receipts(receipts) is False
    finally:
        _FORWARD.reset(forward_token)
        _REQUEST.reset(request_token)


def test_mapped_calibration_is_structural_not_descriptor_value_keyed():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    state.kernel_device = torch.device("cpu")
    launches = []

    def kernel(q, k, v, **kwargs):
        launches.append(kwargs.get("mapped_neighbor_intervals").clone())
        return torch.zeros_like(q)

    kernel.backend_name = "test_sm120"
    state.kernel = kernel
    q = torch.zeros((1, 1, 65, 128), dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 257, 128), dtype=q.dtype)
    mapped_a = torch.tensor(((0, 2), (1, 3)), dtype=torch.int32)
    mapped_b = torch.tensor(((1, 3), (2, 4)), dtype=torch.int32)

    for mapped in (mapped_a, mapped_b):
        out = sparse.attention(
            q,
            k,
            k,
            0,
            cfg,
            state,
            recompute_prefix_queries=False,
            mapped_neighbor_intervals=mapped,
        )
        assert out.shape == (1, 65, 128)

    # First layout use performs mapped all-selected calibration plus the actual
    # call. A second descriptor with the same structural layout reuses that gate.
    assert len(launches) == 3
    assert torch.equal(launches[0], mapped_a)
    assert torch.equal(launches[1], mapped_a)
    assert torch.equal(launches[2], mapped_b)
    assert len(state.gates) == 1
    assert state.gates[0]["mapped_neighbor_abi"] is True
    assert state.sparse_calls == 2
