import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sol_h3 import interop
from sol_h3.partitioned_history import (
    PARTITIONED_FLOW_IDENTITY,
    VDN_EXTERNAL_SEQUENCE_KEY,
    _accept_partitioned_receipt,
    _partitioned_flow_replacement_identity,
    _partitioned_history_layout_valid,
)
from sol_h3.partitioned_request import (
    PARTITIONED_DENSE_ROUTE,
    PARTITIONED_MAPPED_ROUTE,
    PARTITIONED_RECEIPT_TAG,
    PARTITIONED_REQUEST_ABI,
    PARTITIONED_SOL_ROUTE,
)
from sol_h3.runtime import _FORWARD, _REQUEST


DIGEST = "a" * 64
MAP_DIGEST = "b" * 64
DESCRIPTOR_DIGEST = "c" * 64
MAPPED_POLICY = "k64-union-radius1-additive-interval4-v1"


def _fields(
    *,
    route=PARTITIONED_DENSE_ROUTE,
    execution_mode="dense_forced",
    prefix_range=(7, 23),
    prefix_measure=-0.5,
    map_digest=None,
    descriptor_digest=None,
    mapped_policy=None,
    kernel_contract=None,
    completed=True,
):
    return (
        PARTITIONED_RECEIPT_TAG,
        PARTITIONED_REQUEST_ABI,
        DIGEST,
        "local",
        execution_mode,
        8,
        31,
        7,
        prefix_range,
        float(prefix_measure),
        map_digest,
        descriptor_digest,
        mapped_policy,
        kernel_contract,
        completed,
    )


def _owned(item):
    state = SimpleNamespace(
        partitioned_validated_receipts={(item[1], item[3])}, partitioned_receipt_evaluation=2,
    )
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 2, None, []))
    try:
        return _accept_partitioned_receipt(item)
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)


def _partitioned_history_fixture():
    contract = {
        "sequence_rows": 49,
        "video_start": 7,
        "temporal": 5,
        "prefix_t": 2,
        "source_rows_per_frame": 6,
        "target_rows_per_frame": 12,
        "semantic_digest": DIGEST,
    }
    external = {
        "api": 4,
        "mode": "partitioned_attention_variable_grid_linear",
        "topology": "target_prefix_source_suffix",
        "sequence_rows": 49,
        "video_start": 7,
        "temporal": 5,
        "prefix_t": 2,
        "source_rows_per_frame": 6,
        "target_rows_per_frame": 12,
        "flow_semantic_digest": DIGEST,
    }
    options = {
        PARTITIONED_FLOW_IDENTITY: contract,
        VDN_EXTERNAL_SEQUENCE_KEY: external,
    }
    layout = SimpleNamespace(
        seq_len=49,
        segments=[(0, 7, "nonvideo"), (7, 49, "video")],
        signature=(PARTITIONED_FLOW_IDENTITY, "test"),
    )
    return options, layout


def _flow_replacement_fixture():
    layer = 0
    previous = object()
    plan = SimpleNamespace(
        prefix_t=2,
        temporal=5,
        source_rows=4,
        target_rows=16,
        prefix_rows=32,
        partitioned_rows=44,
        target_hw=(8, 8),
    )
    video_start = 7
    video_end = 27
    carrier_prefix_rows = 8
    layout = SimpleNamespace(
        seq_len=27,
        segments=[(0, 7, "nonvideo"), (7, 27, "video")],
        signature=("native", 5, 4, 4),
    )
    partitioned_layout = SimpleNamespace(
        seq_len=51,
        segments=[(0, 7, "nonvideo"), (7, 51, "video")],
        signature=(PARTITIONED_FLOW_IDENTITY, "partitioned"),
    )
    inner = SimpleNamespace(blocks=[object(), object()])
    partition_contract = {"semantic_digest": DIGEST}

    def call():
        return (
            layer,
            previous,
            plan,
            layout,
            partitioned_layout,
            video_start,
            video_end,
            carrier_prefix_rows,
            inner,
            partition_contract,
        )

    call.__module__ = "h3_flow_regenerate.partitioned_transformer"
    call.__qualname__ = "partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call"
    return call, previous


def test_owned_partitioned_dense_receipt_is_accepted():
    fields = _fields()
    item = ("sol_h3", 3, PARTITIONED_DENSE_ROUTE, fields)
    assert _owned(item)


def test_partitioned_receipt_requires_request_owned_completion():
    fields = _fields()
    item = ("sol_h3", 3, PARTITIONED_DENSE_ROUTE, fields)
    token = _REQUEST.set(SimpleNamespace(partitioned_validated_receipts=set()))
    try:
        assert not _accept_partitioned_receipt(item)
    finally:
        _REQUEST.reset(token)


def test_partitioned_receipt_requires_completion_in_the_current_forward():
    from sol_h3.partitioned_request import _record_completion

    fields = _fields()
    item = ("sol_h3", 3, PARTITIONED_DENSE_ROUTE, fields)
    state = SimpleNamespace()
    options = {"attention_backend_receipts_v1": []}
    token = _REQUEST.set(state)
    forward = _FORWARD.set((None, state, 2, None, []))
    try:
        assert not _accept_partitioned_receipt(item)
        _record_completion(state, options, block_index=3, route=PARTITIONED_DENSE_ROUTE, fields=fields)
        assert _accept_partitioned_receipt(item)
        next_forward = _FORWARD.set((None, state, 3, None, []))
        try:
            assert not _accept_partitioned_receipt(item)
            _record_completion(state, options, block_index=4, route=PARTITIONED_DENSE_ROUTE, fields=fields)
            assert not _accept_partitioned_receipt(item)
            assert state.partitioned_validated_receipts == {(4, fields)}
            _record_completion(state, options, block_index=3, route=PARTITIONED_DENSE_ROUTE, fields=fields)
            assert _accept_partitioned_receipt(item)
            other_request = _REQUEST.set(SimpleNamespace())
            try:
                assert not _accept_partitioned_receipt(item)
            finally:
                _REQUEST.reset(other_request)
        finally:
            _FORWARD.reset(next_forward)
    finally:
        _FORWARD.reset(forward)
        _REQUEST.reset(token)
    assert not _accept_partitioned_receipt(item)


def test_partitioned_sparse_receipt_rejects_measure_gap_after_sink():
    fields = _fields(
        route=PARTITIONED_SOL_ROUTE,
        execution_mode="sm120_union",
        prefix_range=(8, 23),
        kernel_contract="test-sm120-contract",
    )
    item = ("sol_h3", 3, PARTITIONED_SOL_ROUTE, fields)
    assert not _owned(item)


def test_owned_partitioned_mapped_receipt_requires_exact_map_proof():
    fields = _fields(
        route=PARTITIONED_MAPPED_ROUTE,
        execution_mode="sm120_mapped",
        map_digest=MAP_DIGEST,
        descriptor_digest=DESCRIPTOR_DIGEST,
        mapped_policy=MAPPED_POLICY,
        kernel_contract="test-sm120-contract",
    )
    item = ("sol_h3", 3, PARTITIONED_MAPPED_ROUTE, fields)
    assert _owned(item)

    tampered = list(fields)
    tampered[11] = None
    bad = ("sol_h3", 3, PARTITIONED_MAPPED_ROUTE, tuple(tampered))
    assert not _owned(bad)


def test_partitioned_history_requires_current_partitioned_layout_and_vdn_binding():
    options, layout = _partitioned_history_fixture()
    assert _partitioned_history_layout_valid(options, layout)

    stale_layout = SimpleNamespace(
        seq_len=48,
        segments=layout.segments,
        signature=layout.signature,
    )
    assert not _partitioned_history_layout_valid(options, stale_layout)

    bad_external = {
        **options,
        VDN_EXTERNAL_SEQUENCE_KEY: {
            **options[VDN_EXTERNAL_SEQUENCE_KEY],
            "flow_semantic_digest": "d" * 64,
        },
    }
    assert not _partitioned_history_layout_valid(bad_external, layout)


def test_partitioned_history_recognizes_dedicated_flow_transformer_closure():
    patch, previous = _flow_replacement_fixture()
    resolved = _partitioned_flow_replacement_identity(interop, patch, 0)
    assert resolved is not None
    identity, inherited = resolved
    assert identity[0] == PARTITIONED_FLOW_IDENTITY
    assert identity[1] == DIGEST
    assert identity[2:10] == (27, 51, 7, 5, 2, 4, 16, (8, 8))
    assert inherited is previous

    patch.__module__ = "h3_flow_regenerate.partitioned_mixed"
    assert _partitioned_flow_replacement_identity(interop, patch, 0) is None


def _same_grid_history_fixture():
    # Canonical Flow v1 control: patch grid 4x6, latent grid 8x12.
    contract = {
        "api": 1,
        "topology": "target_prefix_source_suffix",
        "sequence_rows": 127,
        "video_start": 7,
        "temporal": 5,
        "prefix_t": 2,
        "source_grid_h": 4,
        "source_grid_w": 6,
        "target_grid_h": 4,
        "target_grid_w": 6,
        "source_rows_per_frame": 24,
        "target_rows_per_frame": 24,
        "prefix_range": [7, 55],
        "suffix_range": [55, 127],
        "prefix_log_key_measure": 0.0,
        "nonvideo_log_key_measure": 0.0,
        "suffix_log_key_measure": 0.0,
        "exact_prefix_queries_preserved": True,
        "generated_suffix_queries_preserved": True,
        "heterogeneous_spatial_domains": False,
    }
    contract["semantic_digest"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    options, layout = _partitioned_history_fixture()
    options[PARTITIONED_FLOW_IDENTITY] = contract
    external = options[VDN_EXTERNAL_SEQUENCE_KEY]
    for name in ("sequence_rows", "source_rows_per_frame", "target_rows_per_frame"):
        external[name] = contract[name]
    external["flow_semantic_digest"] = contract["semantic_digest"]
    layout.seq_len = 127
    layout.segments[-1] = (7, 127, "video")
    patch, previous = _flow_replacement_fixture()
    values = interop._closure_values(patch)
    plan = values["plan"]
    plan.source_h, plan.source_w = 8, 12
    plan.source_rows = plan.target_rows = 24
    plan.prefix_rows, plan.partitioned_rows = 48, 120
    plan.target_hw = (8, 12)
    values["layout"].seq_len = 127
    values["layout"].segments[-1] = (7, 127, "video")
    values["partitioned_layout"].seq_len = 127
    values["partitioned_layout"].segments[-1] = (7, 127, "video")
    values["partition_contract"].clear()
    values["partition_contract"].update(contract)
    # The fixture closure has integer cells; the real Flow closure computes these.
    values.update(video_end=127, carrier_prefix_rows=48)
    return options, layout, patch, previous, values


def test_same_grid_control_has_a_current_layout_history_identity():
    options, layout, _patch, _previous, _values = _same_grid_history_fixture()
    before = copy.deepcopy(options)
    assert _partitioned_history_layout_valid(options, layout)
    assert options == before


def test_same_grid_control_replacement_identity_retains_geometry_and_inherited_owner():
    options, _layout, patch, previous, values = _same_grid_history_fixture()
    classifier = SimpleNamespace(_closure_values=lambda _patch: values)
    resolved = _partitioned_flow_replacement_identity(classifier, patch, 0)
    assert resolved is not None
    identity, inherited = resolved
    assert identity[1] == options[PARTITIONED_FLOW_IDENTITY]["semantic_digest"]
    assert identity[2:10] == (127, 127, 7, 5, 2, 24, 24, (8, 12))
    assert inherited is previous


@pytest.mark.parametrize("name,value", [
    ("api", 2),
    ("topology", "foreign"),
    ("source_grid_h", 3),
    ("source_grid_w", 8),
    ("target_grid_h", 6),
    ("target_grid_w", 4),
    ("source_grid_h", True),
    ("source_grid_h", 4.0),
    ("prefix_range", [7, 54]),
    ("suffix_range", [54, 127]),
    ("prefix_log_key_measure", -0.5),
    ("nonvideo_log_key_measure", 1.0),
    ("suffix_log_key_measure", 1.0),
    ("exact_prefix_queries_preserved", False),
    ("generated_suffix_queries_preserved", False),
    ("heterogeneous_spatial_domains", True),
    ("heterogeneous_spatial_domains", 0),
    ("semantic_digest", "d" * 64),
])
def test_same_grid_control_rejects_noncanonical_geometry_measure_or_digest(name, value):
    options, layout, patch, _previous, values = _same_grid_history_fixture()
    options[PARTITIONED_FLOW_IDENTITY][name] = value
    values["partition_contract"][name] = value
    classifier = SimpleNamespace(_closure_values=lambda _patch: values)
    assert not _partitioned_history_layout_valid(options, layout)
    assert _partitioned_flow_replacement_identity(classifier, patch, 0) is None


def test_same_grid_control_rejects_stale_layout_external_binding_and_closure_geometry():
    options, layout, patch, _previous, values = _same_grid_history_fixture()
    layout.seq_len -= 1
    assert not _partitioned_history_layout_valid(options, layout)
    layout.seq_len += 1
    options[VDN_EXTERNAL_SEQUENCE_KEY]["flow_semantic_digest"] = "d" * 64
    assert not _partitioned_history_layout_valid(options, layout)
    values["plan"].source_h, values["plan"].source_w = 12, 8
    classifier = SimpleNamespace(_closure_values=lambda _patch: values)
    assert _partitioned_flow_replacement_identity(classifier, patch, 0) is None


def test_equal_row_products_on_different_grids_remain_opaque_with_a_recomputed_digest():
    options, layout, patch, _previous, values = _same_grid_history_fixture()
    contract = options[PARTITIONED_FLOW_IDENTITY]
    contract["target_grid_h"], contract["target_grid_w"] = 6, 4
    payload = {name: value for name, value in contract.items() if name != "semantic_digest"}
    contract["semantic_digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    options[VDN_EXTERNAL_SEQUENCE_KEY]["flow_semantic_digest"] = contract["semantic_digest"]
    values["partition_contract"].update(contract)
    classifier = SimpleNamespace(_closure_values=lambda _patch: values)
    assert not _partitioned_history_layout_valid(options, layout)
    assert _partitioned_flow_replacement_identity(classifier, patch, 0) is None
