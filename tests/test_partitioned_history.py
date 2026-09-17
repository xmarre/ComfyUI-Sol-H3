from types import SimpleNamespace

from sol_h3.partitioned_history import (
    PARTITIONED_FLOW_IDENTITY,
    VDN_EXTERNAL_SEQUENCE_KEY,
    _accept_partitioned_receipt,
    _partitioned_history_layout_valid,
)
from sol_h3.partitioned_request import (
    PARTITIONED_DENSE_ROUTE,
    PARTITIONED_MAPPED_ROUTE,
    PARTITIONED_RECEIPT_TAG,
    PARTITIONED_REQUEST_ABI,
    PARTITIONED_SOL_ROUTE,
)
from sol_h3.runtime import _REQUEST


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
        ("sol_h3_evaluation", 2),
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
    state = SimpleNamespace(partitioned_validated_receipts={(item[1], item[3])})
    token = _REQUEST.set(state)
    try:
        return _accept_partitioned_receipt(item)
    finally:
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
        "api": 3,
        "mode": "partitioned_attention_no_linear",
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


def test_partitioned_sparse_receipt_rejects_measure_gap_after_sink():
    fields = _fields(
        route=PARTITIONED_SOL_ROUTE,
        execution_mode="sm120_union",
        prefix_range=(8, 23),
        kernel_contract="test-sm120-contract",
    )
    item = ("sol_h3", 3, PARTITIONED_SOL_ROUTE, fields)
    state = SimpleNamespace(partitioned_validated_receipts={(3, fields)})
    token = _REQUEST.set(state)
    try:
        assert not _accept_partitioned_receipt(item)
    finally:
        _REQUEST.reset(token)


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
    tampered[12] = None
    bad = ("sol_h3", 3, PARTITIONED_MAPPED_ROUTE, tuple(tampered))
    state = SimpleNamespace(partitioned_validated_receipts={(3, tuple(tampered))})
    token = _REQUEST.set(state)
    try:
        assert not _accept_partitioned_receipt(bad)
    finally:
        _REQUEST.reset(token)


def test_partitioned_history_requires_current_mixed_layout_and_vdn_binding():
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
