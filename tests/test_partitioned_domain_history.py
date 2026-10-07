"""Sol history identity for Flow's two-stream domain-uniform block replacement."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from sol_h3 import interop
from sol_h3.partitioned_history import (
    PARTITIONED_DOMAIN_STREAM_API,
    PARTITIONED_DOMAIN_STREAM_KEY,
    PARTITIONED_DOMAIN_UNIFORM_IDENTITY,
    PARTITIONED_FLOW_IDENTITY,
    _partitioned_flow_domain_replacement_identity,
    _partitioned_flow_replacement_identity,
    install_partitioned_history_bridge,
)

POLICY = "domain_uniform_v1"
NATIVE_ROWS, NATIVE_VIDEO_START = 7 + 6 * 48, 7


def _contract(video_start, temporal, prefix_t, grid_h, grid_w):
    rows = grid_h * grid_w
    prefix_end = video_start + prefix_t * rows
    sequence_rows = video_start + temporal * rows
    contract = {
        "api": 1,
        "topology": "target_prefix_source_suffix",
        "sequence_rows": sequence_rows,
        "video_start": video_start,
        "temporal": temporal,
        "prefix_t": prefix_t,
        "source_grid_h": grid_h,
        "source_grid_w": grid_w,
        "target_grid_h": grid_h,
        "target_grid_w": grid_w,
        "source_rows_per_frame": rows,
        "target_rows_per_frame": rows,
        "prefix_range": [video_start, prefix_end],
        "suffix_range": [prefix_end, sequence_rows],
        "prefix_log_key_measure": 0.0,
        "nonvideo_log_key_measure": 0.0,
        "suffix_log_key_measure": 0.0,
        "exact_prefix_queries_preserved": True,
        "generated_suffix_queries_preserved": True,
        "heterogeneous_spatial_domains": False,
    }
    raw = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    contract["semantic_digest"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return contract


def _stream(name, contract, head):
    rows = contract["sequence_rows"]
    return SimpleNamespace(
        name=name,
        contract=contract,
        layout=SimpleNamespace(
            seq_len=rows,
            segments=[(0, contract["video_start"], "nonvideo"), (contract["video_start"], rows, "video")],
            signature=(PARTITIONED_FLOW_IDENTITY, name, (PARTITIONED_DOMAIN_STREAM_KEY, POLICY, name)),
        ),
        leaf={
            "api": PARTITIONED_DOMAIN_STREAM_API,
            "policy": POLICY,
            "stream": name,
            "flow_semantic_digest": contract["semantic_digest"],
            "native_sequence_rows": NATIVE_ROWS,
            "native_video_start": NATIVE_VIDEO_START,
        },
        attention_head_t=head,
    )


def _fixture():
    layer = 0
    previous = object()
    streams = (
        _stream("target", _contract(5, 4, 2, 6, 8), 3),
        _stream("source", _contract(NATIVE_VIDEO_START, 6, 2, 3, 4), 4),
    )
    layout = SimpleNamespace(
        seq_len=NATIVE_ROWS,
        segments=[(0, NATIVE_VIDEO_START, "nonvideo"), (NATIVE_VIDEO_START, NATIVE_ROWS, "video")],
        signature=("native", 6, 12, 16),
    )
    video_start = NATIVE_VIDEO_START
    inner = SimpleNamespace(blocks=[object(), object()])
    domain_policy = POLICY

    def call():
        return layer, previous, streams, layout, video_start, inner, domain_policy

    call.__module__ = "h3_flow_regenerate.partitioned_transformer"
    call.__qualname__ = "_domain_uniform_forward.<locals>.wrap.<locals>.call"
    return call, previous


def test_domain_replacement_identity_is_recognized_and_retains_the_inherited_owner():
    patch, previous = _fixture()
    resolved = _partitioned_flow_domain_replacement_identity(interop, patch, 0)
    assert resolved is not None
    identity, inherited = resolved
    assert inherited is previous
    assert identity[:4] == (PARTITIONED_DOMAIN_UNIFORM_IDENTITY, POLICY, NATIVE_ROWS, NATIVE_VIDEO_START)
    assert [stream[0] for stream in identity[5]] == ["target", "source"]
    assert _partitioned_flow_replacement_identity(interop, patch, 0) is None
    install_partitioned_history_bridge()
    assert interop._flow_mixed_grid_replacement_identity(patch, 0) == resolved
    assert _partitioned_flow_domain_replacement_identity(interop, patch, 1) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "policy",
        "order",
        "stream_count",
        "heterogeneous",
        "digest",
        "leaf",
        "signature",
        "head",
        "native_rows",
        "qualname",
    ],
)
def test_domain_replacement_identity_fails_closed(mutation):
    patch, _previous = _fixture()
    values = interop._closure_values(patch)
    streams = values["streams"]
    if mutation == "policy":
        patch.__closure__[patch.__code__.co_freevars.index("domain_policy")].cell_contents = "mixed_grid"
    elif mutation == "order":
        patch.__closure__[patch.__code__.co_freevars.index("streams")].cell_contents = streams[::-1]
    elif mutation == "stream_count":
        patch.__closure__[patch.__code__.co_freevars.index("streams")].cell_contents = streams[:1]
    elif mutation == "heterogeneous":
        streams[1].contract["target_grid_h"] = 6
    elif mutation == "digest":
        streams[0].contract["semantic_digest"] = "0" * 64
    elif mutation == "leaf":
        streams[1].leaf["native_video_start"] += 1
    elif mutation == "signature":
        streams[0].layout.signature = (PARTITIONED_FLOW_IDENTITY, "target")
    elif mutation == "head":
        streams[0].attention_head_t = streams[0].contract["temporal"]
    elif mutation == "native_rows":
        values["layout"].seq_len += 1
    elif mutation == "qualname":
        patch.__qualname__ = "partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call"
    assert _partitioned_flow_domain_replacement_identity(interop, patch, 0) is None
