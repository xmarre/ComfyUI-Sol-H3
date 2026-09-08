from types import SimpleNamespace

from sol_h3.interop import (
    SPECTRUM_EXTERNAL_RUNTIME_KEY,
    SPECTRUM_RUNTIME_KEY,
    _diffaid_replacement_identity,
    _float32_scalar,
)


class MiniMaxH3BlockReplacePatch:
    def __init__(self, config, existing_patch):
        self.config = config
        self.existing_patch = existing_patch


def _config(*, strength=0.5, sigma_end=0.95):
    return SimpleNamespace(
        strength=strength,
        sigma_start=0.0,
        sigma_end=sigma_end,
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=0.35,
        cond_only=True,
    )


def _descriptor(*, instance="diffaid-h3-1", blocks=(0, 2), strength=0.5, sigma_end=0.95):
    return SimpleNamespace(
        schema_version=1,
        provider="comfyui-diffaid-patches",
        instance_id=instance,
        block_indices_0based=tuple(blocks),
        strength=strength,
        sigma_start=_float32_scalar(0.0),
        sigma_end=_float32_scalar(sigma_end),
        sigma_ramp=0.0,
        token_weight_mode="none",
        token_tail=0.35,
        cond_only=True,
    )


def _options(descriptor, *, runtime_entries=None):
    parsed = SimpleNamespace(descriptors=(descriptor,))
    compat = SimpleNamespace(parsed=parsed)
    runtime = SimpleNamespace(_spectrum_h3_external_patch_compat=compat)
    options = {SPECTRUM_RUNTIME_KEY: runtime}
    if runtime_entries is not None:
        options[SPECTRUM_EXTERNAL_RUNTIME_KEY] = tuple(runtime_entries)
    return options


def test_diffaid_static_spectrum_contract_proves_preflight_before_runtime_injection():
    previous = object()
    patch = MiniMaxH3BlockReplacePatch(_config(), previous)
    descriptor = _descriptor()

    before = _diffaid_replacement_identity(patch, _options(descriptor), 0)
    assert before is not None
    assert before[1] is previous

    runtime_entry = {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "instance_id": "diffaid-h3-1",
        "normalized_sigma": 0.5,
    }
    after = _diffaid_replacement_identity(
        patch,
        _options(descriptor, runtime_entries=(runtime_entry,)),
        0,
    )
    assert after == before


def test_diffaid_static_contract_fails_closed_on_runtime_instance_disagreement():
    patch = MiniMaxH3BlockReplacePatch(_config(), object())
    descriptor = _descriptor()
    conflicting = {
        "schema_version": 1,
        "provider": "comfyui-diffaid-patches",
        "instance_id": "diffaid-h3-other",
        "normalized_sigma": 0.5,
    }
    assert _diffaid_replacement_identity(
        patch,
        _options(descriptor, runtime_entries=(conflicting,)),
        0,
    ) is None


def test_diffaid_static_contract_requires_matching_block_and_config():
    descriptor = _descriptor(blocks=(0, 2))
    assert _diffaid_replacement_identity(
        MiniMaxH3BlockReplacePatch(_config(), object()),
        _options(descriptor),
        1,
    ) is None
    assert _diffaid_replacement_identity(
        MiniMaxH3BlockReplacePatch(_config(strength=0.25), object()),
        _options(descriptor),
        0,
    ) is None
