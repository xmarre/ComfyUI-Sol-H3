from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    p.write_text(text.replace(old, new, 1))


p = "sol_h3/runtime.py"
replace_once(
    p,
    '''                # The independent Flow measure contract is meaningful only on a
                # fully validated external mixed stream. Never silently drop or
                # reinterpret a malformed measure request as the old square path.
                if (legacy_measure_contract is not None or generic_measure_contract is not None) and (
                    reason is not None or not external_mixed
                ):
                    raise RuntimeError(
                        "Flow mixed-grid attention-measure contract requires a valid external mixed sequence"
                    )
''',
    '''                # The legacy representative-KV contract is coupled to VDN API 2
                # because that contract describes its gather domain. The generic key-measure
                # contract is not: it binds to the actual all-row H3 layout and only cross-checks
                # VDN API 2 when that optional contract is present.
                if legacy_measure_contract is not None and (reason is not None or not external_mixed):
                    raise RuntimeError(
                        "Legacy Flow mixed-grid attention-measure contract requires a valid external mixed sequence"
                    )
                if generic_measure_contract is not None and reason is not None:
                    raise RuntimeError(
                        "Generic mixed-grid attention measure requires the actual representable H3 mixed layout"
                    )
''',
)

p = "tests/test_weighted_runtime.py"
text = Path(p).read_text()
old = "def _run_block(monkeypatch, cfg, *, sparse_impl, dense_impl):"
new = "def _run_block(monkeypatch, cfg, *, sparse_impl, dense_impl, include_external=True):"
if text.count(old) != 1:
    raise SystemExit(f"{p}: helper signature replacement mismatch")
text = text.replace(old, new, 1)
old = '''    options = {
        "minimax_h3_layout": layout,
        "vdn_h3_external_sequence_v1": external,
        weighted_measure.ATTENTION_MEASURE_KEY: measure,
    }
'''
new = '''    options = {
        "minimax_h3_layout": layout,
        weighted_measure.ATTENTION_MEASURE_KEY: measure,
    }
    if include_external:
        options["vdn_h3_external_sequence_v1"] = external
'''
if text.count(old) != 1:
    raise SystemExit(f"{p}: options replacement mismatch")
text = text.replace(old, new, 1)
text += '''\n\ndef test_generic_weighted_measure_does_not_require_vdn_external_sequence(monkeypatch):
    calls = []

    def sparse_impl(q, k, v, prefix, config, state, **kwargs):
        calls.append((prefix, k.shape[2], kwargs["exact_k_blocks"]))
        state.sparse_calls += 1
        return torch.zeros(1, ROWS, HEADS * DIM, dtype=q.dtype)

    def dense_impl(*args, **kwargs):
        raise AssertionError("generic all-row weighted route unexpectedly fell back to dense")

    result, state, _ = _run_block(
        monkeypatch,
        Config(exact=False, backend="sol", dense_evaluations=0, dense_layers=0),
        sparse_impl=sparse_impl,
        dense_impl=dense_impl,
        include_external=False,
    )

    assert result.shape == (1, ROWS, HEADS * DIM)
    assert calls == [(10, ROWS, (0, 3))]
    assert state.external_mixed_weighted_measure_calls == 1
'''
Path(p).write_text(text)
