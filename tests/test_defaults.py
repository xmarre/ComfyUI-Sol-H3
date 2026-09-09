from inspect import signature

from sol_h3.contracts import Config
from sol_h3.interop import dense_evaluation_warmup
from sol_h3.nodes import SolH3Experimental


def test_sol_default_starts_on_first_denoiser_evaluation():
    config = Config(backend="sol")
    assert config.dense_evaluations == 0
    assert config.dense_layers == 2
    assert dense_evaluation_warmup(config, 0, {}) is False

    inputs = SolH3Experimental.INPUT_TYPES()["required"]
    assert inputs["dense_evaluations"][1]["default"] == 0
    assert inputs["dense_layers"][1]["default"] == 2
    assert signature(SolH3Experimental.apply).parameters["dense_evaluations"].default == 0


def test_dense_evaluation_warmup_remains_an_explicit_opt_in():
    config = Config(backend="sol", dense_evaluations=1)
    assert dense_evaluation_warmup(config, 0, {}) is True
    assert dense_evaluation_warmup(config, 1, {}) is False


def test_progressive_high_continuation_does_not_restart_single_eval_opt_in():
    config = Config(backend="sol", dense_evaluations=1)
    options = {
        "h3_flow_stage": "high",
        "h3_refinement": {
            "api": 1,
            "active": True,
            "source": "h3_flow_progressive_handoff",
            "min_actual_prefix_steps": 1,
            "sigma_reference": 1.0,
        },
    }
    assert dense_evaluation_warmup(config, 0, options) is False
