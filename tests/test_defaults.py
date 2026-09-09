from inspect import signature

from sol_h3.contracts import Config
from sol_h3.interop import dense_evaluation_warmup
from sol_h3.nodes import SolH3Experimental


def test_sol_default_uses_one_dense_trajectory_evaluation():
    config = Config(backend="sol")
    assert config.dense_evaluations == 1
    assert config.dense_layers == 2
    assert dense_evaluation_warmup(config, 0, {}) is True
    assert dense_evaluation_warmup(config, 1, {}) is False

    inputs = SolH3Experimental.INPUT_TYPES()["required"]
    assert inputs["dense_evaluations"][1]["default"] == 1
    assert inputs["dense_layers"][1]["default"] == 2
    assert signature(SolH3Experimental.apply).parameters["dense_evaluations"].default == 1


def test_sol_first_remains_an_explicit_speed_opt_in():
    config = Config(backend="sol", dense_evaluations=0)
    assert dense_evaluation_warmup(config, 0, {}) is False


def test_progressive_high_continuation_does_not_restart_single_eval_default():
    config = Config(backend="sol")
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
