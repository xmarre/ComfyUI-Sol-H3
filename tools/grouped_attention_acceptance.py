"""SM120 acceptance for fused grouped production, full attention and allocation reuse.

This validates the newly introduced producer, not QKV projection. It does not load
H3 or certify generation quality. Output is saved before a failed acceptance exits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sol_h3 import grouped_kv  # noqa: E402
from sol_h3._vendor.sol_attn.preprocess import _reduce_kv, _compute_diag_threshold  # noqa: E402
from sol_h3.contracts import Config  # noqa: E402
from sol_h3.partitioned_request import _sm120_union  # noqa: E402
from sol_h3.runtime import Request, _FORWARD, _REQUEST  # noqa: E402
from sol_h3.sparse import arithmetic_gate_passes, error_metrics  # noqa: E402

MIB = 1024**2


@contextmanager
def scope():
    state = Request(Config(backend="sol", exact=False, dense_evaluations=0, dense_layers=0))
    rt = _REQUEST.set(state)
    ft = _FORWARD.set((None, state, 0, set(), []))
    try:
        yield state
    finally:
        _FORWARD.reset(ft)
        _REQUEST.reset(rt)


def _allocate(source_rows, q_rows, kv_rows, sinks, heads, *, interleaved=False):
    raw = torch.randn(source_rows, heads, 3 if interleaved else 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = raw[:, :, 0], raw[:, :, -1]
    if not interleaved:
        k, v = k.contiguous(), v.contiguous()
        del raw
    q = torch.randn(q_rows, heads, 128, device="cuda", dtype=k.dtype)
    # Noncontiguous, deliberately reversed/randomized physical row order.
    index = torch.randperm(source_rows, device=k.device)[: kv_rows - sinks].contiguous()
    ko = torch.empty(kv_rows, heads, 128, device=k.device, dtype=k.dtype)
    vo = torch.empty_like(ko)
    ko[:sinks].copy_(k[:sinks])
    vo[:sinks].copy_(v[:sinks])
    return q, k, v, ko, vo, index


def _gather(k, v, ko, vo, index, sinks):
    torch.index_select(k, 0, index, out=ko[sinks:])
    torch.index_select(v, 0, index, out=vo[sinks:])


def _union(q, k, v, sinks, bias=None, mapped=None, preparation=None):
    return _sm120_union(
        q,
        k,
        v,
        tau=1.0,
        scale=128**-0.5,
        sink_rows=sinks,
        key_bias=bias,
        mapped_neighbor_intervals=mapped,
        preparation=preparation,
    )


def _exact_match(got, expected, *, stage, field):
    """Preserve the bitwise gate, but identify the failing group and tensor."""
    try:
        torch.testing.assert_close(got, expected, atol=0, rtol=0)
    except AssertionError as exc:
        raise AssertionError(f"{stage} {field}: {exc}") from exc


def parity_cases():
    reports = []
    torch.manual_seed(6181)
    for sinks, windows, q_rows, mapped, weighted, interleaved in (
        (0, 1, 1, False, False, False),
        (1, 63, 17, True, False, False),
        (63, 66, 65, True, False, True),
        (64, 128, 33, False, True, False),
        (65, 256, 129, True, True, True),
        (129, 0, 65, False, False, False),
    ):
        rows = sinks + windows
        case = f"parity sinks={sinks} windows={windows} q={q_rows} mapped={mapped} weighted={weighted} interleaved={interleaved}"
        q, k, v, ko, vo, index = _allocate(max(519, rows), q_rows, rows, sinks, 2, interleaved=interleaved)
        _gather(k, v, ko, vo, index, sinks)
        original_k, original_v = ko.clone(), vo.clone()
        kc, vc = _reduce_kv(ko.unsqueeze(0), vo.unsqueeze(0))
        descriptor = None
        if mapped:
            descriptor = torch.tensor(
                [[0, min(2, (rows + 63) // 64)]] * ((q_rows + 63) // 64), device=q.device, dtype=torch.int32
            )
        bias = None
        if weighted:
            bias = torch.zeros(rows, device=q.device, dtype=torch.float32)
            bias[: max(sinks, 1)] = -0.69
        baseline = _union(q, ko, vo, sinks, bias, descriptor)
        with scope() as state:

            def consume(kg, vg):
                prepared = grouped_kv.current(kg.unsqueeze(0), vg.unsqueeze(0))
                assert prepared is not None
                # Exhaustive bitwise comparison: summary changes can change sparse routing.
                _exact_match(kg, original_k, stage=case, field="gathered K")
                _exact_match(vg, original_v, stage=case, field="gathered V")
                _exact_match(prepared.kc, kc, stage=case, field="BF16 K centroid")
                _exact_match(prepared.vc, vc, stage=case, field="BF16 V sum")
                return _union(q, kg, vg, sinks, bias, descriptor)

            candidate = grouped_kv.execute(k, v, ko, vo, index, sinks, consume)
            assert state.grouped_kv_fused_calls == 1
        _exact_match(candidate, baseline, stage=case, field="sparse attention output")
        # All-selected comparison also covers the independent dense operator.
        all_selected = _union(q, ko, vo, rows, bias, descriptor)
        mask = None if bias is None else bias.to(q.dtype).view(1, 1, 1, -1)
        reference = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                ko.transpose(0, 1).unsqueeze(0),
                vo.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        metrics = error_metrics(all_selected, reference)
        assert arithmetic_gate_passes(metrics), metrics
        reports.append(
            dict(
                sinks=sinks,
                window_rows=windows,
                q_rows=q_rows,
                mapped=mapped,
                weighted=weighted,
                interleaved=interleaved,
                entire_qkv_summaries_and_sparse_output_bitwise_equal=True,
                dense_reference=metrics,
            )
        )
    return reports


def measure(fn, rounds):
    for _ in range(2):
        result = fn()
        del result
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    walls, milliseconds = [], []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - wall) * 1000)
        milliseconds.append(start.elapsed_time(end))
        del result
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - baseline
    return dict(
        cuda_ms=statistics.median(milliseconds),
        wall_ms=statistics.median(walls),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / MIB,
        incremental_peak_allocated_mib=(torch.cuda.max_memory_allocated() - baseline) / MIB,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / MIB,
        retained_after_release_mib=retained / MIB,
        free_before_mib=free / MIB,
        total_mib=total / MIB,
    )


def production_case(source_rows, q_rows, kv_rows, sinks, rounds):
    q, k, v, ko, vo, index = _allocate(source_rows, q_rows, kv_rows, sinks, 56)
    _gather(k, v, ko, vo, index, sinks)
    from sol_h3.grouped_kv_triton import gather_summaries

    def baseline_prepare():
        _gather(k, v, ko, vo, index, sinks)
        return _reduce_kv(ko.unsqueeze(0), vo.unsqueeze(0))

    def fused_prepare():
        return gather_summaries(k, v, ko, vo, index, sinks)

    kc, vc = baseline_prepare()
    fc, fv = fused_prepare()
    _exact_match(fc, kc, stage=f"production q={q_rows} kv={kv_rows}", field="BF16 K centroid")
    _exact_match(fv, vc, stage=f"production q={q_rows} kv={kv_rows}", field="BF16 V sum")
    del kc, vc, fc, fv
    descriptor = torch.tensor(
        [[1, min(4, (kv_rows + 63) // 64)]] * ((q_rows + 63) // 64), dtype=torch.int32, device=q.device
    )

    def baseline():
        _gather(k, v, ko, vo, index, sinks)
        return _union(q, ko, vo, sinks, mapped=descriptor)

    def candidate():
        return grouped_kv.execute(
            k, v, ko, vo, index, sinks, lambda kg, vg: _union(q, kg, vg, sinks, mapped=descriptor)
        )

    with scope() as state:
        expected = baseline()
        actual = candidate()
        _exact_match(actual, expected, stage=f"production q={q_rows} kv={kv_rows}", field="full sparse attention output")
        del expected, actual
        # Alternate arm order to reduce the effect of warm pool/clock drift.
        first = {"legacy": measure(baseline, rounds), "fused": measure(candidate, rounds)}
        second = {"fused": measure(candidate, rounds), "legacy": measure(baseline, rounds)}
        preparation = {"legacy": measure(baseline_prepare, rounds), "fused": measure(fused_prepare, rounds)}
        # Kernel/threshold work with already-produced K/V and summaries.
        kc, vc = baseline_prepare()
        threshold = _compute_diag_threshold(q.unsqueeze(0), kc, tau=1.0, scale=128**-0.5, valid_kv_tokens=kv_rows)
        kernel = measure(lambda: _union(q, ko, vo, sinks, mapped=descriptor, preparation=[(kc, vc, threshold)]), rounds)
        assert state.grouped_kv_fused_calls > 0
    for result in (*first.values(), *second.values(), *preparation.values(), kernel):
        assert result["retained_after_release_mib"] <= 1.0, result
    return dict(
        source_rows=source_rows,
        q_rows=q_rows,
        kv_rows=kv_rows,
        sinks=sinks,
        first=first,
        reversed_order=second,
        gather_and_summary=preparation,
        prepared_attention=kernel,
        fused_calls=state.grouped_kv_fused_calls,
        full_domain_bitwise_equal=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("grouped_attention_acceptance.json"))
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("--rounds must be >= 3")
    report = dict(
        contract="sol-vdn-grouped-producer-acceptance-v1",
        end_to_end=False,
        passed=False,
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
            raise RuntimeError("This acceptance requires an SM120/SM121 GPU")
        report["device"] = torch.cuda.get_device_name()
        with torch.inference_mode():
            report["phase"] = "bf16_parity"
            report["parity"] = parity_cases()
            report["phase"] = "production_shapes"
            # Synthetic restricted domains sized for recorded long H3 source
            # sequences. These are not captures of a particular H3 workflow.
            report["production_shapes"] = [
                production_case(54744, 3168, 18901, 3101, args.rounds),
                production_case(107630, 6580, 35901, 3101, args.rounds),
            ]
        report["numerical_and_memory_passed"] = True
        report["phase"] = "performance_gate"
        report["component_performance_passed"] = all(
            case[order]["fused"]["cuda_ms"] <= case[order]["legacy"]["cuda_ms"] * 1.05
            for case in report["production_shapes"] for order in ("first", "reversed_order")
        )
        if not report["component_performance_passed"]:
            raise RuntimeError("Fused grouped attention exceeds the 5% component regression budget")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output.resolve()), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
