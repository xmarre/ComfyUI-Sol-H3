from types import SimpleNamespace

import torch

from sol_h3.runtime_diagnostics import (
    deferred_tensor_delta_receipt,
    deferred_tensor_receipt,
    finalize,
    tensor_metadata_receipt,
    tensor_receipt,
)


def test_tensor_receipt_is_stable_and_layout_bound():
    x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    first = tensor_receipt(x)
    second = tensor_receipt(x.clone())
    assert first == second

    transposed = tensor_receipt(x.t())
    assert transposed["sample_sha256"] != first["sample_sha256"]
    assert transposed["stride"] != first["stride"]


def test_tensor_receipt_detects_sampled_value_change():
    x = torch.arange(64, dtype=torch.float32)
    before = tensor_receipt(x)
    x[0] += 1
    after = tensor_receipt(x)
    assert before["sample_sha256"] != after["sample_sha256"]


def test_tensor_receipt_handles_empty_tensor():
    got = tensor_receipt(torch.empty(0, 3))
    assert got["numel"] == 0
    assert got["sample_count"] == 0
    assert got["sample_finite"] is True


def test_deferred_receipt_matches_immediate_receipt_after_finalize():
    x = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    expected = tensor_receipt(x)
    state = SimpleNamespace(
        runtime_diagnostic={},
        runtime_diagnostic_pending={
            "sample": {
                "schema": "test",
                "tensor": deferred_tensor_receipt(x),
            }
        },
    )

    finalize(state)

    assert state.runtime_diagnostic_pending == {}
    assert state.runtime_diagnostic["sample"]["tensor"] == expected


def test_finalize_materializes_native_attention_receipt():
    x = torch.arange(64, dtype=torch.float32)
    state = SimpleNamespace(
        runtime_diagnostic={},
        runtime_diagnostic_pending={
            "eval0_block0_vdn_native_global": {
                "schema": "test",
                "kind": "global",
                "output": deferred_tensor_receipt(x),
                "extra_attention_shadow_calls": 0,
            }
        },
    )

    finalize(state)

    got = state.runtime_diagnostic["eval0_block0_vdn_native_global"]
    assert got["kind"] == "global"
    assert got["output"]["sample_sha256"]
    assert got["extra_attention_shadow_calls"] == 0


def test_tensor_metadata_receipt_is_value_free_and_layout_bound():
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    got = tensor_metadata_receipt(x)
    assert got["shape"] == [3, 8]
    assert got["stride"] == [8, 1]
    assert got["dtype"] == "torch.float32"
    assert got["device"] == "cpu"
    assert got["numel"] == 24
    assert got["storage_offset"] == 0
    assert got["data_ptr"] == x.data_ptr()
    assert got["data_ptr_mod_256"] == x.data_ptr() % 256
    assert got["data_ptr_mod_4096"] == x.data_ptr() % 4096
    assert "sample_sha256" not in got


def test_finalize_keeps_route_specific_native_keys():
    x = torch.arange(32, dtype=torch.float32)
    state = SimpleNamespace(
        runtime_diagnostic={},
        runtime_diagnostic_pending={
            "eval0_block0_vdn_native_anchor": {
                "schema": "test",
                "kind": "anchor",
                "output": deferred_tensor_receipt(x),
            }
        },
    )
    finalize(state)
    got = state.runtime_diagnostic["eval0_block0_vdn_native_anchor"]
    assert got["kind"] == "anchor"


def test_deferred_pair_delta_receipt_is_bounded_and_detects_change():
    left = torch.arange(64, dtype=torch.float32)
    right = left.clone()
    state = SimpleNamespace(
        runtime_diagnostic={},
        runtime_diagnostic_pending={
            "same": {
                "schema": "test",
                "delta": deferred_tensor_delta_receipt(left, right),
            },
        },
    )
    finalize(state)
    same = state.runtime_diagnostic["same"]["delta"]
    assert same["receipt_kind"] == "bounded_pair_delta"
    assert same["sample_abs_max"] == 0.0

    changed = right.clone()
    changed[0] += 1.0
    state = SimpleNamespace(
        runtime_diagnostic={},
        runtime_diagnostic_pending={
            "changed": {
                "schema": "test",
                "delta": deferred_tensor_delta_receipt(changed, left),
            },
        },
    )
    finalize(state)
    got = state.runtime_diagnostic["changed"]["delta"]
    assert got["sample_abs_max"] == 1.0