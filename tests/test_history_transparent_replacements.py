from types import SimpleNamespace

import torch

from sol_h3.contracts import Config
from sol_h3.interop import HistoryPolicy, SPECTRUM_EXTERNAL_RUNTIME_KEY
from sol_h3.runtime import BlockPatch, Request, _REQUEST


class MiniMaxH3BlockReplacePatch:
    # The loader-specific module name is intentionally unimportant. The real
    # producer identity comes from Diff-Aid's Spectrum runtime declaration.
    __module__ = "nodes"

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


def _policy_identity(cfg, options, *, blocks=2, layout=None):
    model = SimpleNamespace(
        blocks=[
            SimpleNamespace(attn=SimpleNamespace(forward=lambda *a, **k: None))
            for _ in range(blocks)
        ],
        dtype=torch.bfloat16,
    )
    if layout is None:
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


def _flow_mixed_wrapper(previous, layer, *, source_rows=4, target_rows=16, attention_measure=False):
    prefix_t = 2
    temporal = 4
    va = 3
    old_prefix = prefix_t * source_rows
    prefix_rows = prefix_t * target_rows
    mixed_rows = prefix_rows + (temporal - prefix_t) * source_rows
    vb = va + temporal * source_rows
    plan = SimpleNamespace(
        prefix_t=prefix_t,
        temporal=temporal,
        source_rows=source_rows,
        target_rows=target_rows,
        prefix_rows=prefix_rows,
        mixed_rows=mixed_rows,
        target_hw=(8, 8),
        attention_measure=attention_measure,
    )
    layout = SimpleNamespace(
        seq_len=vb,
        segments=[(0, va, "text"), (va, vb, "video")],
        signature=(3, temporal, 4, 4, 0),
    )
    mixed_layout = SimpleNamespace(
        seq_len=va + mixed_rows,
        segments=[(0, va, "text"), (va, va + mixed_rows, "video")],
        signature=("h3_flow_mixed_grid_v1", *layout.signature, prefix_t, 8, 8),
    )
    inner = SimpleNamespace(blocks=[object(), object()])

    # Reference every audited free variable so the closure shape mirrors Flow's
    # actual mixed_diffusion_wrapper.<locals>.wrap.<locals>.call function.
    def call(args, extra):
        return (
            layer,
            previous,
            plan,
            layout,
            mixed_layout,
            va,
            vb,
            old_prefix,
            inner,
            args,
            extra,
        )

    call.__module__ = "h3_flow_regenerate.mixed_grid"
    call.__qualname__ = "mixed_diffusion_wrapper.<locals>.wrap.<locals>.call"
    return call, layout


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


def test_audited_flow_mixed_grid_closures_keep_stable_history_across_rebuilds():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    replacements = {}
    carrier = None
    for layer in range(2):
        wrapped, carrier = _flow_mixed_wrapper(BlockPatch(layer, cfg), layer)
        replacements[("double_block", layer)] = wrapped
    options = {"patches_replace": {"dit": replacements}}
    first = _policy_identity(cfg, options, blocks=2, layout=carrier)
    assert first is not None

    # Flow reconstructs the wrapper closures for every mixed-grid transformer
    # invocation. History identity must describe their semantics, not function ids.
    rebuilt = {}
    for layer in range(2):
        wrapped, rebuilt_carrier = _flow_mixed_wrapper(BlockPatch(layer, cfg), layer)
        rebuilt[("double_block", layer)] = wrapped
    second = _policy_identity(
        cfg,
        {"patches_replace": {"dit": rebuilt}},
        blocks=2,
        layout=rebuilt_carrier,
    )
    assert second == first


def test_flow_mixed_grid_geometry_change_changes_history_identity():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    base = {}
    changed = {}
    base_layout = changed_layout = None
    for layer in range(2):
        base_wrap, base_layout = _flow_mixed_wrapper(BlockPatch(layer, cfg), layer)
        changed_wrap, changed_layout = _flow_mixed_wrapper(
            BlockPatch(layer, cfg), layer, source_rows=8, target_rows=16
        )
        base[("double_block", layer)] = base_wrap
        changed[("double_block", layer)] = changed_wrap
    base_id = _policy_identity(
        cfg, {"patches_replace": {"dit": base}}, blocks=2, layout=base_layout
    )
    changed_id = _policy_identity(
        cfg, {"patches_replace": {"dit": changed}}, blocks=2, layout=changed_layout
    )
    assert base_id is not None and changed_id is not None and changed_id != base_id


def test_flow_mixed_grid_attention_measure_change_changes_history_identity():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    plain = {}
    measured = {}
    plain_layout = measured_layout = None
    for layer in range(2):
        plain_wrap, plain_layout = _flow_mixed_wrapper(BlockPatch(layer, cfg), layer)
        measured_wrap, measured_layout = _flow_mixed_wrapper(
            BlockPatch(layer, cfg), layer, attention_measure=True
        )
        plain[("double_block", layer)] = plain_wrap
        measured[("double_block", layer)] = measured_wrap
    plain_id = _policy_identity(
        cfg, {"patches_replace": {"dit": plain}}, blocks=2, layout=plain_layout
    )
    measured_id = _policy_identity(
        cfg, {"patches_replace": {"dit": measured}}, blocks=2, layout=measured_layout
    )
    assert plain_id is not None and measured_id is not None and measured_id != plain_id


def test_malformed_flow_mixed_grid_closure_remains_opaque():
    cfg = Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0)
    wrapped, carrier = _flow_mixed_wrapper(BlockPatch(0, cfg), 1)
    options = {
        "patches_replace": {
            "dit": {
                ("double_block", 0): wrapped,
                ("double_block", 1): BlockPatch(1, cfg),
            }
        }
    }
    assert _policy_identity(cfg, options, blocks=2, layout=carrier) is None
