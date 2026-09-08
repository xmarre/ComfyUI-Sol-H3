from types import SimpleNamespace

import torch

from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy, SPECTRUM_EXTERNAL_RUNTIME_KEY
from sol_h3.runtime import BlockPatch, Request, _REQUEST


class MiniMaxH3BlockReplacePatch:
    __module__ = "comfyui_diffaid_patches.nodes"

    def __init__(self, existing_patch=None):
        self.config = SimpleNamespace(
            strength=0.5,
            sigma_start=0.0,
            sigma_end=0.95,
            sigma_ramp=0.0,
            token_weight_mode="none",
            token_tail=0.35,
            cond_only=True,
        )
        self.existing_patch = existing_patch


def _runtime_marker(sigma):
    return ({
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "instance_id": "diffaid-h3-1",
        "normalized_sigma": sigma,
    },)


def _fixture_options(cfg, *, marker=True):
    # Exercise both supported node orders: Diff-Aid outside Sol on block 0 and
    # Sol outside Diff-Aid on block 1. One Diff-Aid instance legitimately owns
    # several configured blocks and therefore publishes one runtime instance id.
    replacements = {
        ("double_block", 0): MiniMaxH3BlockReplacePatch(BlockPatch(0, cfg)),
        ("double_block", 1): BlockPatch(1, cfg, MiniMaxH3BlockReplacePatch()),
    }
    options = {"patches_replace": {"dit": replacements}}
    if marker:
        options[SPECTRUM_EXTERNAL_RUNTIME_KEY] = _runtime_marker(0.5)
    return options


def _policy_identity(cfg, options):
    model = SimpleNamespace(
        blocks=[
            SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None)),
            SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None)),
        ],
        dtype=torch.bfloat16,
    )
    layout = SimpleNamespace(
        seq_len=7,
        segments=[(0, 3, "text"), (3, 7, "video")],
        signature=(3, 7),
    )
    request = Request(cfg)
    token = _REQUEST.set(request)
    try:
        return HistoryPolicy(cfg)(layout=layout, options=options, model=model)
    finally:
        _REQUEST.reset(token)


def test_spectrum_declared_diffaid_replacements_are_attention_history_transparent():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    options = _fixture_options(cfg)
    first = _policy_identity(cfg, options)
    assert first is not None

    # Diff-Aid's normalized sigma changes every call. Spectrum's external-patch
    # compatibility layer owns those regime transitions; Sol's attention history
    # identity must remain stable within a regime rather than invalidating every step.
    options[SPECTRUM_EXTERNAL_RUNTIME_KEY] = _runtime_marker(0.25)
    assert _policy_identity(cfg, options) == first


def test_diffaid_wrapper_without_spectrum_runtime_declaration_remains_opaque():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    assert _policy_identity(cfg, _fixture_options(cfg, marker=False)) is None


def test_unknown_replacement_wrapper_remains_opaque():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    options = _fixture_options(cfg)
    options["patches_replace"]["dit"][("double_block", 0)] = SimpleNamespace(
        existing_patch=BlockPatch(0, cfg)
    )
    assert _policy_identity(cfg, options) is None


def test_receipt_coverage_guard_rejects_missing_transformer_blocks():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    policy = HistoryPolicy(cfg, block_count=2)
    assert not policy.accept_receipts((("sol_h3", 0, "sol"),))
    assert policy.accept_receipts((
        ("sol_h3", 0, "sol"),
        ("sol_h3", 1, "vdn_local_sol"),
        ("sol_h3", 1, "vdn_global_native"),
    ))
