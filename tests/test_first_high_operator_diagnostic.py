from __future__ import annotations

import copy

import pytest

from sol_h3 import first_high_operator_diagnostic as w


def _request(mode="native_window"):
    return (
        ("api", 1),
        ("capture_id", "capture-w"),
        ("mode", mode),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", 0.8780487775802612),
        ("target_shapes_digest", "a" * 64),
        ("source_contract_digest", "b" * 64),
    )


def _options(mode="native_window"):
    return {w.REQUEST_KEY: _request(mode), "h3_flow_stage": "high"}


def test_request_parser_is_strict_immutable_and_fail_closed():
    parsed = w.parse_request(_options())
    assert parsed is not None
    assert parsed["mode"] == "native_window"
    assert parsed["logical_call_limit"] == 1

    mutable = {w.REQUEST_KEY: dict(_request())}
    with pytest.raises(RuntimeError, match="immutable"):
        w.parse_request(mutable)

    wrong_order = list(_request())
    wrong_order[1], wrong_order[2] = wrong_order[2], wrong_order[1]
    with pytest.raises(RuntimeError, match="fields/order"):
        w.parse_request({w.REQUEST_KEY: tuple(wrong_order)})

    bad_mode = list(_request())
    bad_mode[2] = ("mode", "dense")
    with pytest.raises(RuntimeError, match="unsupported"):
        w.parse_request({w.REQUEST_KEY: tuple(bad_mode)})

    bad_sigma = list(_request())
    bad_sigma[5] = ("sigma", float("nan"))
    with pytest.raises(RuntimeError, match="sigma"):
        w.parse_request({w.REQUEST_KEY: tuple(bad_sigma)})

    with pytest.raises(RuntimeError, match="external/reduced"):
        w.parse_request({**_options(), "vdn_h3_external_sequence_v1": object()})
    with pytest.raises(RuntimeError, match="weighted/Mixed-Grid"):
        w.parse_request({**_options(), "attention_measure_v1": object()})
    with pytest.raises(RuntimeError, match="outside the high stage"):
        w.parse_request({**_options(), "h3_flow_stage": "low"})


def test_history_identity_distinguishes_w_arms_and_binds_capture_source_contract():
    window = w.history_identity(_options("native_window"))
    full = w.history_identity(_options("native_full_support"))
    assert window != full
    assert window[0] == "h3_first_high_operator_diagnostic_v1"
    assert window[1] == "native_window"
    assert window[2] == "capture-w"
    assert window[-1] == "b" * 64


def _live_provider(block_index, sink):
    class Owner:
        def __init__(self, index):
            self.index = index

    owner = Owner(block_index)

    def make_record(self):
        def record(route):
            sink.append(("sol_h3", self.index, route))

        return record

    record = make_record(owner)

    def provider(*_args, **_kwargs):
        _ = record
        return None

    return provider


def test_native_local_receipt_uses_live_blockpatch_owner_after_success():
    sink = []
    request = dict(_request("native_window"))
    options = _options("native_window")
    options[w.VDN_PROVIDER_V3_KEY] = _live_provider(7, sink)

    w.record_native_local(options, 7, "vdn_local_native_window_w", request)
    assert sink == [("sol_h3", 7, "vdn_local_native_window_w")]

    with pytest.raises(RuntimeError, match="block ownership"):
        w.record_native_local(options, 8, "vdn_local_native_window_w", request)
    with pytest.raises(RuntimeError, match="selected mode"):
        w.record_native_local(options, 7, "vdn_local_native_full_w", request)


def test_native_local_receipt_rejects_request_or_provider_divergence():
    request = dict(_request())
    options = _options()
    options[w.VDN_PROVIDER_V3_KEY] = _live_provider(0, [])

    changed = copy.deepcopy(request)
    changed["capture_id"] = "other"
    with pytest.raises(RuntimeError, match="request identity diverged"):
        w.record_native_local(options, 0, "vdn_local_native_window_w", changed)

    no_provider = _options()
    with pytest.raises(RuntimeError, match="live Sol VDN v3 provider"):
        w.record_native_local(no_provider, 0, "vdn_local_native_window_w", request)


def test_receipt_normalization_accepts_only_explicit_w_native_routes():
    assert w._diagnostic_receipt(("sol_h3", 3, "vdn_local_native_window_w")) == (
        "sol_h3",
        3,
        "vdn_local_native",
    )
    assert w._diagnostic_receipt(("sol_h3", 3, "vdn_local_native_full_w")) == (
        "sol_h3",
        3,
        "vdn_local_native",
    )
    assert w._diagnostic_receipt(("sol_h3", 3, "vdn_local_sol")) is None
    assert w._diagnostic_receipt(("other", 3, "vdn_local_native_window_w")) is None
    assert w._diagnostic_receipt(("sol_h3", 3, "vdn_local_native_window_w", {})) is None
