from types import SimpleNamespace

import pytest

from sol_h3.contracts import Config, adaln_status, prefix_length, reject_sparse_conflicts
from sol_h3.nodes import NODE_CLASS_MAPPINGS


def test_identity_changes_with_every_numerical_option():
    base = Config().metadata()
    for config in (Config(exact=False), Config(backend="sol"), Config(tau=0.5),
                   Config(dense_evaluations=2), Config(dense_layers=3)):
        assert config.metadata()["fingerprint"] != base["fingerprint"]
    assert Config().metadata() == base
    base["tau"] = 99
    assert Config().metadata()["tau"] == 1


@pytest.mark.parametrize("kwargs", [{"backend": "sol_bsa"}, {"backend": "auto"},
    {"tau": float("nan")}, {"tau": float("inf")}, {"dense_layers": -1}, {"dense_evaluations": 1.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("prefix", [[(0, 5, "text"), (5, 9, "audio")],
    [(0, 5, "text"), (5, 12, "ref_img"), (12, 18, "ref_audio"), (18, 22, "audio")],
    [(0, 5, "text"), (5, 12, "cond"), (12, 18, "cond_audio"), (18, 22, "audio")]])
def test_complete_prefix(prefix):
    start = prefix[-1][1]
    layout = SimpleNamespace(seq_len=start + 11, segments=prefix + [(start, start + 11, "video")])
    assert prefix_length(layout, start + 11) == start


@pytest.mark.parametrize("segments", [[(0, 5, "text"), (6, 10, "video")],
    [(0, 5, "video"), (5, 10, "audio")], [(0, 10, "video")],
    [(0, 5, "text"), (5, 8, "video"), (8, 10, "video")]])
def test_invalid_packing(segments):
    with pytest.raises(RuntimeError):
        prefix_length(SimpleNamespace(seq_len=10, segments=segments), 10)


def test_adaln_format():
    assert adaln_status(SimpleNamespace(use_adaln_curves=True, adaln_t_table=object())) == "not_applicable_compact_curve"
    assert adaln_status(SimpleNamespace(use_adaln_curves=False)) == "native_full_width_no_schedule_precompute"
    with pytest.raises(RuntimeError):
        adaln_status(SimpleNamespace(use_adaln_curves=True))


def test_late_sparse_conflicts():
    for options in ({"spectrum_h3_runtime": object()},
                    {"wrappers": {"diffusion_model": {"spectrum_h3": []}}},
                    {"vdn_h3_external_sequence_v1": {}}):
        with pytest.raises(RuntimeError):
            reject_sparse_conflicts(options)


def test_node_schema():
    assert set(NODE_CLASS_MAPPINGS) == {"SolH3Exact", "SolH3Experimental"}
    for cls in NODE_CLASS_MAPPINGS.values():
        assert cls.RETURN_TYPES == ("MODEL",)
        assert cls.INPUT_TYPES()["required"]["model"] == ("MODEL",)
    assert NODE_CLASS_MAPPINGS["SolH3Experimental"].INPUT_TYPES()["required"]["tau"][1]["default"] == 1.0
