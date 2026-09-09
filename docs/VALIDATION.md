# Validation status and RTX PRO 6000 evidence

This document records the **v0.1.0** validation state for Exact Runtime, rectangular Sana/CuTe SM120 attention, VDN provider API v3, Flow API-2 mixed-grid SOL routing, Spectrum backend-history coordination, Untwist preprocessing, Diff-Aid replacement composition and the zero-copy BTHD bridge.

## Reviewed stack

```text
Sol-H3 v0.1.0
Spectrum #104  9c682c07f4c5ea9de601cda234755a1561b59f59
Untwist #9     cf428e204f42354ce9a9582dd956906f75a52974
VDN #8         b6f0755c4172ec5c17386c56998f454e78b2a2d4
VDN #11        5b63dc670229d419a6350b64f7ceda609dbc8194
Flow v0.3.2    fe0ef8752b92081b5a85bc9b39ad8e2a7037d591
Diff-Aid       ba9d9efbcf7e64c755e068cb76547d8cc85481eb
Sana           2936c47637380842aaa4a4488fac5006cc542b70
Comfy native   efa6c8f804bff78b46a0fd458ebd2e47bba07a30
KJ native      c9869eade9920a1b949de07c4a197156006bcceb
```

Flow-Aligned Regenerate already publishes the explicit API-2 `mixed_grid_low_suffix` contract. Sol-H3 consumes that contract and does not infer arbitrary mixed layouts.

VDN #11 currently applies after the still-unreleased VDN #8 audio-fidelity overlay, so that pair remains a pinned companion stack rather than a mainline VDN release.

## Production environment

Real GPU execution was performed on:

```text
GPU            NVIDIA RTX PRO 6000 Blackwell Workstation Edition
compute        capability (12, 0)
PyTorch        2.10.0+cu130
CUDA runtime   13.0
sol_source     sana-sol-engine
Sana revision  2936c47637380842aaa4a4488fac5006cc542b70
SOL backend    cute_sm120
source verify  true
```

CPU/native CI validates contracts and composition where a real GPU kernel is unavailable; it is not used as GPU-performance evidence.

## Packaged Sana/CuTe SM120

The packaged source compiled and executed the real `cute_sm120` kernel. The original direct 4096-row probe reported:

```text
max_abs  = 0.0009765625
mean_abs = 4.484307282837108e-05
rel_l2   = 0.003004377940669656
```

The rectangular GPU suite subsequently passed the SM120 arithmetic/sparse-sink coverage. Production arithmetic gates remained finite around `~9.5e-4 .. 1.1e-3` relative L2 for the final stack.

## VDN provider API v3

Production confirms the VDN module is actually using provider API v3:

```text
VDN object patches=50
softmax-provider module API=3
```

Representative native stages:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square-expanded calls |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The preferred path therefore executes only the requested local Q rows against VDN's unchanged restricted K/V domain. It does not construct the old v2 square-Q/query-position compatibility payload.

Historical v2 compatibility runs expanded Q work by roughly `4.4x–5.4x` in the affected native stages. v3 removes that kernel-row expansion without changing VDN's learned gate, linear complement, output projection, local-window membership, global ownership or anchor ownership.

## Flow mixed-grid SOL

Final representative mixed execution:

```text
external_mixed_sol_calls           144
external_mixed_q_rows         6,270,480
external_mixed_kernel_q_rows  6,270,480
compatibility_fallbacks              {}
```

The route is `sol_external_mixed`. VDN retains the learned `dense_gate_no_linear` external-mode gate; the geometry-dependent linear complement stays disabled as required by the Flow contract. Later native target-grid stages resume ordinary rectangular SOL.

An earlier production run also established 192 external SOL calls / 8,360,640 requested and kernel Q rows with no compatibility fallback. Both sets of evidence establish routing and arithmetic behavior; neither is presented as a standalone performance A/B.

## Spectrum backend history

A prior production stack executed all `18/18` logical calls as actual transformer NFEs because Sol-H3 could not prove the real replacement topology to Spectrum. The SOL-bypassed control executed `13 actual + 5 forecast`, making the wall-time comparison NFE-confounded.

The final provider-history implementation was then validated against the real wrapper topology:

- audited Diff-Aid activation-only wrapper chains are recognized only with Diff-Aid's runtime declaration;
- Flow's generic marked layout wrapper is transparent only when its marker attributes, closure values, block index, scope and previous-link identity agree;
- Flow's mixed-grid wrapper is recognized only when its captured geometry/contract agrees;
- unknown/malformed wrappers remain opaque and force an actual call;
- Untwist preprocessing remains exactly once;
- receipt/provider changes still invalidate incompatible history.

Progressive high stages are separate sampling lifetimes. Sol-H3 consumes Flow's explicit continuation contract so the default single dense warmup is not restarted spuriously. `dense_evaluations > 1` intentionally remains request-local because the amount already consumed cannot be proven generically.

### Final production schedule

The corrected production stack consistently reaches:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

The SOL-bypassed control is `13 actual + 5 forecast`. The one additional SOL actual is the legitimate initial low-stage `dense -> sol` numerical-backend transition and remains intentionally retained.

## Zero-copy BTHD bridge

The final bridge accepts suitable innermost-contiguous strided BTHD views directly. Arithmetic calibration identity includes Q/K/V shapes **and strides**, so a differently-strided layout cannot reuse a previous gate.

Production confirms:

```text
materialized_qkv_bytes = 0
SOL backend             = cute_sm120
requested/kernel Q rows = 1:1
square expansion         = 0
```

The exact mixed production layout is:

```text
shape      [1, 43545, 56, 128]
Q/V stride [7168, 21504, 128, 1]
K stride   [7168,  7168, 128, 1]
```

### Isolated A/B benchmark

The benchmark feeds identical Q/K/V values to three paths:

```text
strided              current zero-copy path
contiguous_kernel    kernel-only on pre-materialized contiguous BTHD
copy_plus_kernel     historical per-call .contiguous() bridge + kernel
```

Seven-run medians on the production RTX PRO 6000:

| Path | CUDA median | Host-wall median |
|---|---:|---:|
| strided zero-copy | **46.768 ms** | **42.338 ms** |
| contiguous kernel | 46.941 ms | 42.376 ms |
| old copy + kernel | 47.727 ms | 43.136 ms |

The timing clocks are interpreted internally, not against each other. Both agree on the decision:

- strided vs pre-contiguous kernel: effective parity (`-0.37%` CUDA / `-0.09%` wall);
- old materialization overhead: about `0.786 ms` CUDA / `0.761 ms` wall per representative mixed call;
- strided zero-copy vs old copy+kernel: about `2.01%` faster CUDA / `1.85%` faster wall for the isolated call.

The old path materialized `1,248,522,240` bytes per mixed call. Across 144 representative calls, zero-copy avoids about **167.44 GiB** of redundant Q/V materialization and approximately **0.11 s** of direct copy overhead.

The native rectangular control is already contiguous, making `.contiguous()` a no-op; its small spread is treated as timing noise rather than a native-path speed claim.

Decision: **retain zero-copy**. It is a real micro-optimization with no measured strided-kernel penalty, not a claim of a large whole-workflow speedup.

## Whole-workflow timing evidence

### Exact Runtime

Historical matched A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | **247.30 s** | **298.69 s** | 17.18 GB |

This is Exact-only evidence.

### SOL hot run

A pre-zero-copy same-process hot run with the corrected 14/4 Spectrum schedule measured:

```text
end-to-end prompt       274.87 s
H3ContinuumSamplerV34   229.13 s
```

The SOL-bypassed control measured:

```text
end-to-end prompt       287.51 s
H3ContinuumSamplerV34   240.55 s
schedule                 13 actual / 5 forecast
```

The SOL run was faster despite one additional actual NFE, but the release does **not** convert that into a percentage SOL speed claim because the runs are not a controlled routing/content/cache A/B.

Later zero-copy workflow runs measured `287.29/239.10 s` and `298.59/247.63 s` end-to-end/sampler while retaining 14/4. Arithmetic-gate/caching time itself varied materially between those runs. The isolated BTHD benchmark is therefore the authoritative zero-copy measurement.

## Current CI gate

The release candidate remains one implementation commit over neutral Sol `main` while development occurs on mirrors/checkpoints.

Final pre-release validation includes:

- `pip check`;
- Ruff;
- full Sol-H3 CPU suite;
- pinned Sana source reproduction;
- real Comfy ModelPatcher integration;
- KJ wrapper behavior;
- Spectrum history behavior;
- sequential VDN #8 -> #11 application;
- VDN provider API v3 and lazy v2 fallback;
- both audited Diff-Aid/Sol wrapper orders;
- real Flow layout-wrapper history identity;
- Flow mixed-grid SOL routing and mixed -> native resumption;
- direct BTHD benchmark invocation regression;
- zero-copy stride/materialization contracts.

The final release-polish mirror and the final PR head must both pass the CPU-contract and pinned native-interop lanes before merge.

## Validation commands

From an installed checkout:

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/comfyui-sol-h3

python -m pip install -r requirements.txt
python -m pip check
python -m ruff check .
python -m pytest -q
```

Source/kernel sanity:

```bash
python - <<'PY'
import torch
from sol_h3.provenance import verify_source
from sol_h3.sparse import load_kernel

print(verify_source()['revision'])
kernel = load_kernel(torch.device('cuda:0'))
print(kernel.backend_name)
assert kernel.backend_name == 'cute_sm120'
PY
```

Direct packaged-kernel probe:

```bash
python tools/attention_probe.py \
  --backend pytorch \
  --tokens 4096 \
  --prefix 512 \
  --heads 8
```

Zero-copy isolated benchmark:

```bash
python tools/bthd_layout_benchmark.py --warmup 2 --repeats 7
```

The benchmark script bootstraps the repository root when directly executed; it does not require an editable install solely to import `sol_h3`.

## Production success counters

Native grouped VDN:

```text
sol_backend == cute_sm120
sol_source_tree_verified == true
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls == 0
```

Flow mixed-grid:

```text
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
route includes sol_external_mixed
```

Spectrum schedule:

```text
sampler_logical_calls
transformer_actual_nfe
spectrum_forecast_calls
```

Zero-copy:

```text
materialized_qkv_bytes == 0
bthd_strides = exact executed Q/K/V layouts
```

`vdn_global_native` and `vdn_anchor_native` are expected where VDN owns those operations. Unsupported external layouts may report explicit `external_sequence_*` delegation without failing the whole generation.

## Optional SageAttention

Sage is an inherited dense provider, not the SOL kernel. On SM120 use KJNodes `auto`, not `sageattn_qk_int8_pv_fp16_triton`.

Loader/ABI failure can be request-locally classified and demoted to original Comfy dense attention while SOL remains available. Arbitrary provider compute errors remain fatal. See [SAGEATTENTION](SAGEATTENTION.md) for installation/repair guidance.

## Media-quality boundary

Kernel/routing validation does not replace decoded-media inspection. For quality A/Bs preserve model/adapters, quantization, seed, sampler/sigma schedule, prompt/dialogue, reference order, dimensions/duration, VDN settings, Spectrum schedule and all other patches.

Inspect video identity/reference fidelity, motion, grass/detail artifacts, temporal stability and progressive boundaries, plus the full audio track for missing/wrong words, unsolicited speech, whisper/loudness fidelity, synchronization and discontinuities.

Successful sparse execution proves routing; it does not by itself prove decoded-media superiority.