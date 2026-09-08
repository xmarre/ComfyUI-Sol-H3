import logging
from types import SimpleNamespace

import torch

from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy
from sol_h3.runtime import BlockPatch, Request, _REQUEST


def _model():
    return SimpleNamespace(
        blocks=[SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None))],
        dtype=torch.bfloat16,
    )


def _layout():
    return SimpleNamespace(
        seq_len=7,
        segments=[(0, 3, "text"), (3, 7, "video")],
        signature=(3, 7),
    )


def _options(cfg):
    return {
        "patches_replace": {
            "dit": {("double_block", 0): BlockPatch(0, cfg)}
        }
    }


def test_history_diagnostic_reports_semantic_identity_component_changes(caplog):
    cfg = Config(exact=False, backend="sol", dense_evaluations=1, dense_layers=0)
    request = Request(cfg)
    token = _REQUEST.set(request)
    caplog.set_level(logging.INFO, logger="comfy.sol_h3")
    try:
        policy = HistoryPolicy(cfg)
        options = _options(cfg)
        first = policy(layout=_layout(), options=options, model=_model())
        assert first is not None and first[1] == "dense"

        request.evaluations = 1
        second = policy(layout=_layout(), options=options, model=_model())
        assert second is not None and second[1] == "sol"
    finally:
        _REQUEST.reset(token)

    messages = [record.getMessage() for record in caplog.records]
    assert any("backend-history diagnostic established" in message for message in messages)
    assert any(
        "backend-history diagnostic transition" in message and "components=phase" in message
        for message in messages
    )


def test_history_diagnostic_reports_opaque_replacement_without_changing_fail_closed_result(caplog):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    request = Request(cfg)
    token = _REQUEST.set(request)
    caplog.set_level(logging.INFO, logger="comfy.sol_h3")
    try:
        policy = HistoryPolicy(cfg)
        options = _options(cfg)
        options["patches_replace"]["dit"][("double_block", 0)] = SimpleNamespace(
            existing_patch=BlockPatch(0, cfg)
        )
        assert policy(layout=_layout(), options=options, model=_model()) is None
    finally:
        _REQUEST.reset(token)

    assert any(
        "backend-history diagnostic opaque" in record.getMessage()
        and "replacement_chain:block=0" in record.getMessage()
        for record in caplog.records
    )


def test_history_diagnostic_reports_dense_provider_identity_change(caplog):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    request = Request(cfg)
    token = _REQUEST.set(request)
    caplog.set_level(logging.INFO, logger="comfy.sol_h3")
    try:
        policy = HistoryPolicy(cfg)
        options = _options(cfg)

        def first_provider(*args, **kwargs):
            return None

        def second_provider(*args, **kwargs):
            return None

        options["optimized_attention_override"] = first_provider
        assert policy(layout=_layout(), options=options, model=_model()) is not None
        options["optimized_attention_override"] = second_provider
        assert policy(layout=_layout(), options=options, model=_model()) is not None
    finally:
        _REQUEST.reset(token)

    assert any(
        "backend-history diagnostic transition" in record.getMessage()
        and "dense_provider" in record.getMessage()
        for record in caplog.records
    )


def test_history_diagnostic_reports_receipt_acceptance_and_rejection_without_changing_result(caplog):
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    request = Request(cfg)
    token = _REQUEST.set(request)
    caplog.set_level(logging.INFO, logger="comfy.sol_h3")
    try:
        policy = HistoryPolicy(cfg)
        assert policy.accept_receipts([("sol_h3", 0, "sol")]) is True
        assert policy.accept_receipts([("sol_h3", 0, "unexpected_route")]) is False
    finally:
        _REQUEST.reset(token)

    messages = [record.getMessage() for record in caplog.records]
    assert any("receipts accepted routes=sol" in message for message in messages)
    assert any("receipts rejected reason=unsupported_route:'unexpected_route'" in message for message in messages)
