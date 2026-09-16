from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sol_h3 import mapped_neighbors
from sol_h3._vendor.sol_attn import interface
from sol_h3.contracts import Config
from sol_h3.mapped_neighbors import (
    MappingUnavailable,
    POLICY,
    compile_descriptor,
    validate_preflight_summary,
    validate_wire_map,
)
from sol_h3.runtime import Request, _REQUEST


DIGEST = "a" * 64
OWNER = "vdn-owner-test"


def wire(*, q_rows, kv_rows, runs, sink_rows=0, group_index=0,
         owner=OWNER, plan_digest=DIGEST):
    return (
        "vdn_query_positions",
        1,
        owner,
        plan_digest,
        group_index,
        q_rows,
        kv_rows,
        sink_rows,
        tuple(runs),
    )


def plan(value):
    validated = validate_wire_map(
        value, q_rows=value[5], kv_rows=value[6], sink_rows=value[7]
    )
    return compile_descriptor(validated)


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, ((0, 2),)),
        (1, ((0, 3),)),
        (63, ((0, 3),)),
        (64, ((0, 3),)),
        (65, ((0, 4),)),
    ],
)
def test_exact_physical_k64_neighbor_union_for_offsets(offset, expected):
    descriptor = plan(wire(
        q_rows=64,
        kv_rows=256,
        runs=((0, 64, offset),),
    ))
    assert descriptor is not None
    assert descriptor.intervals == expected


def test_partial_q64_tail_uses_only_represented_physical_positions():
    descriptor = plan(wire(
        q_rows=65,
        kv_rows=320,
        runs=((0, 65, 190),),
    ))
    assert descriptor is not None
    assert descriptor.intervals == ((1, 5), (2, 5))


def test_non_contiguous_exact_union_fails_instead_of_filling_hull():
    value = wire(
        q_rows=64,
        kv_rows=512,
        runs=((0, 32, 0), (32, 64, 320)),
    )
    validated = validate_wire_map(value, q_rows=64, kv_rows=512, sink_rows=0)
    with pytest.raises(MappingUnavailable, match="fragmented"):
        compile_descriptor(validated)


def test_contiguous_union_wider_than_four_blocks_fails_closed():
    value = wire(
        q_rows=64,
        kv_rows=512,
        runs=((0, 32, 64), (32, 64, 192)),
    )
    validated = validate_wire_map(value, q_rows=64, kv_rows=512, sink_rows=0)
    with pytest.raises(MappingUnavailable, match="interval_width"):
        compile_descriptor(validated)


def test_identity_aligned_square_map_keeps_existing_ordinal_route():
    value = wire(q_rows=128, kv_rows=128, runs=((0, 128, 0),))
    validated = validate_wire_map(value, q_rows=128, kv_rows=128, sink_rows=0)
    assert validated.identity_aligned is True
    assert compile_descriptor(validated) is None


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda value: value[:-1], "schema"),
        (lambda value: ("wrong", *value[1:]), "schema"),
        (lambda value: (value[0], True, *value[2:]), "schema"),
        (lambda value: (*value[:2], "", *value[3:]), "owner"),
        (lambda value: (*value[:3], "not-a-digest", *value[4:]), "owner"),
        (lambda value: (*value[:4], True, *value[5:]), "schema"),
        (lambda value: (*value[:5], 65, *value[6:]), "domain"),
        (lambda value: (*value[:6], 257, *value[7:]), "domain"),
        (lambda value: (*value[:7], 2, value[8]), "domain"),
        (lambda value: (*value[:8], ((0, 31, 1), (32, 64, 40))), "schema"),
        (lambda value: (*value[:8], ((0, 32, 40), (32, 64, 20))), "domain"),
        (lambda value: (*value[:8], ((0, 64, 200),)), "domain"),
    ],
)
def test_malformed_or_stale_wire_map_fails_closed(mutate, reason):
    value = wire(q_rows=64, kv_rows=256, sink_rows=1, runs=((0, 64, 1),))
    bad = mutate(value)
    with pytest.raises(MappingUnavailable, match=reason):
        validate_wire_map(bad, q_rows=64, kv_rows=256, sink_rows=1)


def test_runtime_counts_are_strict_ints_not_bool_aliases():
    value = wire(q_rows=64, kv_rows=256, sink_rows=1, runs=((0, 64, 1),))
    for counts in (
        dict(q_rows=True, kv_rows=256, sink_rows=1),
        dict(q_rows=64, kv_rows=True, sink_rows=1),
        dict(q_rows=64, kv_rows=256, sink_rows=True),
    ):
        with pytest.raises(MappingUnavailable, match="domain"):
            validate_wire_map(value, **counts)


def test_request_local_cpu_plan_cache_reuses_validated_map_and_descriptor():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    value = wire(q_rows=64, kv_rows=256, sink_rows=1, runs=((0, 64, 65),))
    first_state = Request(cfg)
    first_token = _REQUEST.set(first_state)
    try:
        validated_a = validate_wire_map(value, q_rows=64, kv_rows=256, sink_rows=1)
        descriptor_a = compile_descriptor(validated_a)
        validated_b = validate_wire_map(value, q_rows=64, kv_rows=256, sink_rows=1)
        descriptor_b = compile_descriptor(validated_b)
        assert validated_b is validated_a
        assert descriptor_b is descriptor_a
        assert len(first_state._mapped_cpu_plan_cache) == 1
    finally:
        _REQUEST.reset(first_token)

    second_state = Request(cfg)
    second_token = _REQUEST.set(second_state)
    try:
        validated_c = validate_wire_map(value, q_rows=64, kv_rows=256, sink_rows=1)
        descriptor_c = compile_descriptor(validated_c)
        assert validated_c is not validated_a
        assert descriptor_c is not descriptor_a
        assert len(second_state._mapped_cpu_plan_cache) == 1
    finally:
        _REQUEST.reset(second_token)


def test_request_local_cpu_plan_cache_is_bounded_lru():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    token = _REQUEST.set(state)
    try:
        first = None
        for index in range(mapped_neighbors.MAX_CPU_PLAN_CACHE_ENTRIES + 1):
            value = wire(
                q_rows=64,
                kv_rows=256,
                sink_rows=1,
                runs=((0, 64, 65),),
                owner=f"owner-{index}",
            )
            validated = validate_wire_map(value, q_rows=64, kv_rows=256, sink_rows=1)
            compile_descriptor(validated)
            if index == 0:
                first = validated
        assert len(state._mapped_cpu_plan_cache) == mapped_neighbors.MAX_CPU_PLAN_CACHE_ENTRIES
        first_key = mapped_neighbors._validated_cache_key(first)
        assert first_key not in state._mapped_cpu_plan_cache
    finally:
        _REQUEST.reset(token)


def test_unhashable_malformed_wire_bypasses_cache_and_keeps_schema_rejection():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    token = _REQUEST.set(state)
    try:
        value = (
            "vdn_query_positions",
            1,
            OWNER,
            DIGEST,
            0,
            64,
            256,
            1,
            [(0, 64, 65)],
        )
        with pytest.raises(MappingUnavailable, match="schema"):
            validate_wire_map(value, q_rows=64, kv_rows=256, sink_rows=1)
        cache = getattr(state, "_mapped_cpu_plan_cache", None)
        assert cache is not None
        assert not cache
    finally:
        _REQUEST.reset(token)


def test_preflight_requires_exact_owner_plan_group_and_wire_identity():
    value = wire(q_rows=64, kv_rows=256, runs=((0, 64, 65),), group_index=1)
    descriptor = plan(value)
    assert descriptor is not None
    summary = SimpleNamespace(
        tag="vdn_query_position_plan_v1",
        schema=1,
        mode="grouped",
        owner_generation=OWNER,
        plan_digest=DIGEST,
        groups=(wire(q_rows=1, kv_rows=1, runs=((0, 1, 0),), group_index=0), value),
    )
    assert validate_preflight_summary(summary, value, descriptor)

    for changed in (
        SimpleNamespace(**{**summary.__dict__, "schema": True}),
        SimpleNamespace(**{**summary.__dict__, "mode": "native"}),
        SimpleNamespace(**{**summary.__dict__, "owner_generation": "other"}),
        SimpleNamespace(**{**summary.__dict__, "plan_digest": "b" * 64}),
        SimpleNamespace(**{**summary.__dict__, "groups": (summary.groups[0], wire(
            q_rows=64, kv_rows=256, runs=((0, 64, 66),), group_index=1))}),
    ):
        assert not validate_preflight_summary(changed, value, descriptor)


def test_descriptor_identity_includes_policy_and_map_not_runtime_tensor_values():
    value = wire(q_rows=64, kv_rows=256, runs=((0, 64, 65),))
    descriptor = plan(value)
    assert descriptor is not None
    assert POLICY == "k64-union-radius1-additive-interval4-v1"
    changed = plan(wire(q_rows=64, kv_rows=256, runs=((0, 64, 66),)))
    assert changed is not None
    assert changed.validated.map_digest != descriptor.validated.map_digest
    assert changed.descriptor_digest != descriptor.descriptor_digest


def test_sm120_mapped_compile_key_ignores_descriptor_values(monkeypatch):
    from sol_h3._vendor.sol_attn import preprocess

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(interface, "_stream", lambda _: None)
    monkeypatch.setattr(interface, "_to_cute_tensors", lambda tensors: tensors)
    monkeypatch.setattr(interface, "_compiled", {})

    thresholds = []
    def prepare(q, k, v, **kwargs):
        blocks = (k.shape[1] + 63) // 64
        kc = torch.empty(1, blocks, q.shape[2], q.shape[3])
        threshold = torch.empty(1, (q.shape[1] + 63) // 64, q.shape[2])
        thresholds.append(threshold)
        return kc, torch.empty_like(kc), threshold

    compiles = []
    launches = []
    def compile_(key, tensors, scale, start, end, stream,
                 key_bias_enabled, mapped_neighbors_enabled):
        compiles.append((key, key_bias_enabled, mapped_neighbors_enabled))
        def compiled(*args, **kwargs):
            launches.append(args)
            args[3].copy_(args[0])
        interface._compiled[key] = compiled
        return compiled, tensors

    monkeypatch.setattr(preprocess, "prepare", prepare)
    monkeypatch.setattr(interface, "_compile_sm120", compile_)

    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    k = torch.zeros(1, 257, 2, 128, dtype=q.dtype)
    mapped_a = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    mapped_b = torch.tensor([[1, 3], [2, 4]], dtype=torch.int32)
    common = dict(
        arch=(12, 0), scale=128 ** -0.5, tau=1.0,
        thresh_type="diag", kv_splits=1, sink_tokens=0,
        sink_start=0, valid_tokens=65,
    )

    interface._sol_attn_cute(q, k, k, mapped_neighbor_intervals=mapped_a, **common)
    interface._sol_attn_cute(q, k, k, mapped_neighbor_intervals=mapped_b, **common)
    interface._sol_attn_cute(q, k, k, mapped_neighbor_intervals=None, **common)

    assert [(bias, mapped) for _, bias, mapped in compiles] == [
        (False, True),
        (False, False),
    ]
    assert launches[0][8] is mapped_a
    assert launches[1][8] is mapped_b
    assert launches[2][8] is thresholds[2]
