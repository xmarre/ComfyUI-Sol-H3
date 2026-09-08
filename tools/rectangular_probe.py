"""Real SM120 arithmetic, grouped VDN routing, and matched warmed kernel timing.

The synthetic grouped probe runs the installed VDN dispatcher. --runtime-log also
checks actual workflow telemetry. Timings exclude full-model execution and VDN's
upstream square_q construction (still present in VDN #11).
"""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sol_h3 import sparse  # noqa: E402
from sol_h3.contracts import Config  # noqa: E402
from sol_h3.interop import VDN_KEY_V2  # noqa: E402
from sol_h3.provenance import CONTRACT, REVISION  # noqa: E402
from sol_h3.runtime import BlockPatch, Request, _FORWARD  # noqa: E402

BASELINE = 'c543f4f017c0ddb276ff28148a7e9be291057b75'


def baseline_kernel(path):
    path = path.resolve()
    head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if head != BASELINE:
        raise RuntimeError(f'Baseline checkout must be {BASELINE}; got {head}')
    if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Baseline checkout must be clean')
    spec = importlib.util.spec_from_file_location('_square_baseline', path / 'sol_h3/__init__.py',
                                                 submodule_search_locations=[str(path / 'sol_h3')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return importlib.import_module('_square_baseline.sparse').load_kernel(torch.device('cuda'))


def reference(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2)).transpose(1, 2)


def matched_timing(functions, warmup, repeats):
    # Compile, autotune and warm both paths before any recorded sample.
    for fn in functions.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples = {name: {'cuda_ms': [], 'wall_ms': []} for name in functions}
    names = list(functions)
    for i in range(repeats):
        for name in names[::1 if i % 2 == 0 else -1]:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name]['wall_ms'].append((time.perf_counter() - t0) * 1000)
            samples[name]['cuda_ms'].append(start.elapsed_time(end))
    return {name: {metric: {'median': statistics.median(values), 'samples': values}
                   for metric, values in metrics.items()} for name, metrics in samples.items()}


def grouped_probe(vdn_path, rows, heads):
    sys.path.insert(0, str(vdn_path.resolve()))
    from vdn_h3 import retained, window
    from vdn_h3.softmax_provider import PROVIDER_API_VERSION
    if PROVIDER_API_VERSION != 2:
        raise RuntimeError('Install the VDN #8 -> #11 stack first')
    cfg = Config(exact=False, backend='sol', dense_evaluations=0, dense_layers=0)
    state = Request(cfg)
    captured = []
    # Native grouped dispatcher with four frames and non-video prefix/suffix.
    frame_rows = rows
    total = 65 + 4 * frame_rows + 3
    q, k, v = (torch.randn(total, heads, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    bounds = window.window_bounds(4, 0, 1)
    def block(args):
        options = args['transformer_options']
        provider = options[VDN_KEY_V2]
        def capture(native, q, k, v, **contract):
            if contract['kind'] == 'local' and not captured:
                captured.append((q, k, v, contract))
            return provider(native, q, k, v, **contract)
        options[VDN_KEY_V2] = capture
        return retained.window_softmax_grouped_runtime(
            q, k, v, 65, 65 + 4 * frame_rows, 4, frame_rows, bounds, 128**-.5,
            anchor_frames='both', transformer_options=options)
    token = _FORWARD.set((SimpleNamespace(blocks=[object()]), state, 0, set(), []))
    try:
        out = BlockPatch(0, cfg)({'transformer_options': {}}, {'original_block': block})
    finally:
        _FORWARD.reset(token)
    assert torch.isfinite(out).all() and captured
    assert state.vdn_rectangular_sol_calls > 0 and state.vdn_square_expanded_calls == 0
    assert state.vdn_requested_q_rows == state.vdn_kernel_q_rows > 0
    assert state.kernel.backend_name == 'cute_sm120'
    return captured[0], {
        'scope': 'installed VDN grouped dispatcher with synthetic tensors',
        'vdn_rectangular_sol_calls': state.vdn_rectangular_sol_calls,
        'vdn_requested_q_rows': state.vdn_requested_q_rows,
        'vdn_kernel_q_rows': state.vdn_kernel_q_rows,
        'vdn_square_expanded_calls': state.vdn_square_expanded_calls,
        'fallbacks': dict(state.fallbacks), 'arithmetic_gates': state.gates,
    }


def check_runtime_log(path):
    records = []
    for line in path.read_text(errors='replace').splitlines():
        if 'Sol-H3 {' in line:
            records.append(json.loads(line.split('Sol-H3 ', 1)[1]))
    native = [r for r in records if r.get('vdn_rectangular_sol_calls', 0) > 0]
    if not native:
        raise RuntimeError('No rectangular VDN-local workflow execution in this log')
    for r in native:
        assert r['success'] and r['sol_backend'] == 'cute_sm120' and r['sol_source_tree_verified']
        assert r['vdn_requested_q_rows'] == r['vdn_kernel_q_rows'] > 0
        assert r['vdn_square_expanded_calls'] == 0
    return {'native_stages': len(native), 'requested_rows': sum(r['vdn_requested_q_rows'] for r in native),
            'kernel_rows': sum(r['vdn_kernel_q_rows'] for r in native)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--vdn-path', type=Path, required=True)
    parser.add_argument('--comfy-root', type=Path)
    parser.add_argument('--frame-rows', type=int, default=521)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--runtime-log', type=Path)
    args = parser.parse_args()
    if min(args.frame_rows, args.heads, args.warmup, args.repeats) < 1:
        parser.error('row/head/warmup/repeat counts must be positive')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError('Requires real SM120 CUDA execution')
    if args.comfy_root:
        sys.path.insert(0, str(args.comfy_root.resolve()))
    torch.manual_seed(53)
    with torch.inference_mode():
        kernel = sparse.load_kernel(torch.device('cuda'))
        old = baseline_kernel(args.baseline)
        (q, k, v, contract), routing = grouped_probe(args.vdn_path, args.frame_rows, args.heads)
        square = contract['square_q'].unsqueeze(0).contiguous()
        positions = contract['query_positions']
        qb, kb, vb = (x.unsqueeze(0).contiguous() for x in (q, k, v))
        assert torch.equal(square.index_select(1, positions), qb)
        want = reference(qb, kb, vb)
        arithmetic = {}
        for name, fn in {
            'rectangular': lambda: kernel(qb, kb, vb, tau=1., sink_tokens=kb.shape[1]),
            'square_baseline': lambda: old(square, kb, vb, tau=1., sink_tokens=kb.shape[1]).index_select(1, positions),
        }.items():
            arithmetic[name] = sparse.error_metrics(fn(), want)
            if not sparse.arithmetic_gate_passes(arithmetic[name]):
                raise RuntimeError(f'{name} all-selected gate failed: {arithmetic[name]}')
        functions = {
            'rectangular': lambda: kernel(qb, kb, vb, tau=1., sink_tokens=contract['sink_rows']),
            'square_baseline': lambda: old(square, kb, vb, tau=1., sink_tokens=contract['sink_rows']).index_select(1, positions),
        }
        sparse_difference = sparse.error_metrics(functions['rectangular'](), functions['square_baseline']())
        timing = matched_timing(functions, args.warmup, args.repeats)
        report = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                  'sol_backend': kernel.backend_name, 'sana_revision': REVISION,
                  'kernel_contract': CONTRACT, 'baseline_revision': BASELINE,
                  'q_shape': list(qb.shape), 'kv_shape': list(kb.shape),
                  'requested_q_rows': qb.shape[1], 'kernel_q_rows': qb.shape[1],
                  'baseline_kernel_q_rows': square.shape[1], 'sink_rows': contract['sink_rows'],
                  'correctness': arithmetic, 'vdn_routing': routing,
                  'sparse_output_difference_not_a_parity_gate': sparse_difference,
                  'warmed_timing': timing,
                  'timing_scope': 'packaged API preprocessing + attention + baseline output gather; preallocated inputs; no model or VDN input-gather timing'}
        if args.runtime_log:
            report['production_telemetry'] = check_runtime_log(args.runtime_log)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
