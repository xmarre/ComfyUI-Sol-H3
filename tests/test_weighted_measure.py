from types import SimpleNamespace

import pytest
import torch

from sol_h3 import weighted_measure


def test_register_refreshes_only_its_own_model_local_capability(monkeypatch):
    class Core:
        @staticmethod
        def register_capability(options, provider_identity, capability):
            registry = dict(options.get(weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY, {}))
            existing = registry.get(provider_identity)
            if existing is not None and existing is not capability:
                raise RuntimeError("duplicate provider")
            registry[provider_identity] = capability
            options[weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY] = registry

    monkeypatch.setattr(weighted_measure, "_core", lambda required=True: Core)
    source = {}
    first = weighted_measure.register(source)
    source_registry = source[weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY]

    # Reproduce MODEL.clone() carrying the source model's nested registry by
    # reference. Refreshing the clone must not mutate or reuse the source owner.
    clone = {weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY: source_registry}
    second = weighted_measure.register(clone, refresh_owned=True)

    assert second is not first
    assert second.owner is not first.owner
    assert second.owner.generation != first.owner.generation
    assert source_registry[weighted_measure.PROVIDER_IDENTITY] is first
    assert clone[weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY][weighted_measure.PROVIDER_IDENTITY] is second


def test_register_rejects_foreign_provider_identity_owner(monkeypatch):
    class Core:
        register_capability = staticmethod(lambda *args, **kwargs: None)

    monkeypatch.setattr(weighted_measure, "_core", lambda required=True: Core)
    options = {
        weighted_measure.ATTENTION_MEASURE_CAPABILITIES_KEY: {
            weighted_measure.PROVIDER_IDENTITY: object(),
        }
    }
    with pytest.raises(RuntimeError, match="another capability"):
        weighted_measure.register(options, refresh_owned=True)


def test_preprocess_digest_tracks_numerical_chain_and_concrete_stateful_owners():
    class Leaf:
        pass

    def transform_a(q, k, v, heads, **kw):
        return q, k, v

    def transform_b(q, k, v, heads, **kw):
        return q, k, v

    leaf_a = Leaf()
    leaf_b = Leaf()
    provider_a = SimpleNamespace(attention_preprocess_v1=(transform_a, leaf_a))
    provider_b = SimpleNamespace(attention_preprocess_v1=(transform_a, leaf_b))
    provider_c = SimpleNamespace(attention_preprocess_v1=(transform_b, leaf_a))

    digest = weighted_measure.preprocess_digest(provider_a)
    assert digest == weighted_measure.preprocess_digest(provider_a)
    assert digest != weighted_measure.preprocess_digest(provider_b)
    assert digest != weighted_measure.preprocess_digest(provider_c)


def test_preprocess_digest_stabilizes_factory_recreated_stateless_functions():
    class Leaf:
        pass

    leaf = Leaf()

    def factory():
        # Mirrors the reviewed Untwist H3 preprocessing shape: the wrapper
        # function is recreated per model invocation, but the preprocessor itself
        # carries no closure/default state and reads execution state from kwargs.
        def preprocess(q, k, v, heads, **kwargs):
            options = kwargs.get("transformer_options")
            return q, k, v if options is not None else v

        return SimpleNamespace(attention_preprocess_v1=(preprocess, leaf))

    first = factory()
    second = factory()
    assert first.attention_preprocess_v1[0] is not second.attention_preprocess_v1[0]
    assert first.attention_preprocess_v1[0].__closure__ is None
    assert weighted_measure.preprocess_digest(first) == weighted_measure.preprocess_digest(second)


def test_preprocess_digest_keeps_stateful_recreated_functions_owner_bound():
    class Leaf:
        pass

    leaf = Leaf()

    def factory(scale):
        def preprocess(q, k, v, heads, **kwargs):
            return q, k * scale, v

        return SimpleNamespace(attention_preprocess_v1=(preprocess, leaf))

    first = factory(0.5)
    second = factory(0.5)
    assert first.attention_preprocess_v1[0].__closure__ is not None
    assert weighted_measure.preprocess_digest(first) != weighted_measure.preprocess_digest(second)


def test_preprocess_digest_rejects_unsafe_nonweakrefable_stateful_owner_and_cycles():
    def transform(q, k, v, heads, **kw):
        return q, k, v

    with pytest.raises(RuntimeError, match="weak references"):
        weighted_measure.preprocess_digest(SimpleNamespace(attention_preprocess_v1=(transform, object())))

    cycle = SimpleNamespace()
    cycle.attention_preprocess_v1 = (transform, cycle)
    with pytest.raises(RuntimeError, match="cyclic"):
        weighted_measure.preprocess_digest(cycle)


def test_provider_binds_real_core_measure_contract_when_available():
    core = pytest.importorskip("comfy.attention_measure")

    options = {}
    capability = weighted_measure.register(options, refresh_owned=True)
    assert capability is not None

    rows = 202
    layout = SimpleNamespace(seq_len=rows, segments=((0, 10, "text"), (10, rows, "video")))
    external = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 138,
        "sequence_rows": rows,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 64,
        "prefix_rows_per_frame": 128,
    }
    request = {
        "api": 1,
        "operator": "key_log_measure",
        "normalization": "h3_native_source_carrier_v1",
        "topology": "mixed_grid_low_suffix",
        "coordinate_policy": "minimax_h3_native_frame_grid_v1",
        "q_rows": rows,
        "kv_rows": rows,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_grid": [8, 8],
        "prefix_grid": [8, 16],
        "segments": [
            {"start": 0, "stop": 10, "mass_num": 1, "mass_den": 1},
            {"start": 10, "stop": 138, "mass_num": 1, "mass_den": 2},
            {"start": 138, "stop": rows, "mass_num": 1, "mass_den": 1},
        ],
    }
    state = SimpleNamespace(measure_plans={}, measure_biases={})
    plan = weighted_measure.prepare(
        state,
        options,
        request,
        block_index=0,
        layout=layout,
        q_rows=rows,
        kv_rows=rows,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        existing_sink=(0, 1),
        external_sequence=external,
        preprocess_identity="test-preprocess",
    )

    assert plan.provider_identity == weighted_measure.PROVIDER_IDENTITY
    assert plan.owner_generation == capability.owner.generation
    assert plan.implementation_profile == weighted_measure.IMPLEMENTATION_PROFILE
    assert plan.exact_k_block_range == (0, 3)
    assert plan.key_log_measure.dtype == torch.float32
    assert plan.key_log_measure.is_contiguous()
    assert torch.count_nonzero(plan.key_log_measure[:10]) == 0
    assert torch.allclose(plan.key_log_measure[10:138], torch.full((128,), torch.log(torch.tensor(0.5))))
    assert torch.count_nonzero(plan.key_log_measure[138:]) == 0

    cached = weighted_measure.prepare(
        state,
        options,
        request,
        block_index=0,
        layout=layout,
        q_rows=rows,
        kv_rows=rows,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        existing_sink=(0, 1),
        external_sequence=external,
        preprocess_identity="test-preprocess",
    )
    assert cached is plan
    assert state.measure_biases[(core.semantic_digest(request), "cpu")] is plan.key_log_measure