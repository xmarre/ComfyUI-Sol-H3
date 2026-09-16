from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from sol_h3 import weighted_measure


pytestmark = pytest.mark.skipif(
    not os.environ.get("COMFYUI_PATH") or not os.environ.get("FLOW_PATH"),
    reason="set COMFYUI_PATH and FLOW_PATH for real weighted Flow/Core contracts",
)


def _activate_real_sources():
    for path in (os.environ["COMFYUI_PATH"], os.environ["FLOW_PATH"]):
        if path not in sys.path:
            sys.path.insert(0, path)

    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def test_real_flow_weighted_request_binds_current_core_and_sol_provider():
    _activate_real_sources()

    from comfy import attention_measure as core
    from h3_flow_regenerate.attention_measure import attention_measure_semantic_digest
    from h3_flow_regenerate.mixed_grid import (
        MIXED_GRID_MEASURE_PROFILE_WEIGHTED,
        MixedGridPlan,
        mixed_attention_measure_contract,
    )

    flow_plan = MixedGridPlan(
        prefix=torch.empty(1, 24, 1, 16, 32),
        temporal=2,
        source_h=16,
        source_w=16,
        measure_profile=MIXED_GRID_MEASURE_PROFILE_WEIGHTED,
    )
    request = mixed_attention_measure_contract(
        flow_plan,
        video_start=10,
        sequence_rows=202,
    )
    assert request is not None
    assert request["operator"] == "key_log_measure"
    assert request["q_rows"] == request["kv_rows"] == 202
    assert request["source_grid"] == [8, 8]
    assert request["prefix_grid"] == [8, 16]
    assert core.normalize(request) == request

    layout = SimpleNamespace(
        seq_len=202,
        segments=((0, 10, "text"), (10, 202, "video")),
    )
    external = {
        "api": 2,
        "mode": "dense_gate_no_linear",
        "topology": "mixed_grid_low_suffix",
        "native_sequence_rows": 138,
        "sequence_rows": 202,
        "video_start": 10,
        "temporal": 2,
        "prefix_t": 1,
        "source_rows_per_frame": 64,
        "prefix_rows_per_frame": 128,
    }
    options = {}
    capability = weighted_measure.register(options, refresh_owned=True)
    assert capability is not None
    state = SimpleNamespace(measure_plans={}, measure_biases={})

    bound = weighted_measure.prepare(
        state,
        options,
        request,
        block_index=3,
        layout=layout,
        q_rows=202,
        kv_rows=202,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        head_dim=128,
        existing_sink=(0, 1),
        external_sequence=external,
        preprocess_identity="real-flow-core-sol-v1",
    )

    digest = attention_measure_semantic_digest(request)
    assert digest == core.semantic_digest(request) == bound.semantic_digest
    assert bound.provider_identity == weighted_measure.PROVIDER_IDENTITY
    assert bound.owner_generation == capability.owner.generation
    assert bound.implementation_profile == weighted_measure.SPARSE_IMPLEMENTATION_PROFILE
    assert bound.numerical_route == weighted_measure.SPARSE_NUMERICAL_ROUTE
    assert bound.exact_k_block_range == (0, 3)
    assert bound.key_log_measure.shape == (202,)
    assert bound.key_log_measure.dtype == torch.float32
    assert bound.key_log_measure.is_contiguous()

    expected = torch.zeros(202, dtype=torch.float32)
    expected[10:138] = math.log(0.5)
    assert torch.equal(bound.key_log_measure, expected)
    assert torch.exp(bound.key_log_measure[10:138]).sum().item() == 64.0
    assert torch.exp(bound.key_log_measure[138:202]).sum().item() == 64.0
