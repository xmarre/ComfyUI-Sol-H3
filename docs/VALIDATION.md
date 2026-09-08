# Validation status and RTX PRO 6000 commands

The current revision combines Exact Runtime, rectangular Sana/CuTe SM120 attention, VDN provider API v3, explicit Flow API-2 mixed-grid SOL routing, and Spectrum backend-history coordination through the audited MiniMax-H3 Diff-Aid replacement chain.

## Current reviewed companion set

```text
Sol-H3         feature/native-sol-h3
Spectrum #104  9c682c07f4c5ea9de601cda234755a1561b59f59
VDN #8         b6f0755c4172ec5c17386c56998f454e78b2a2d4
VDN #11        5b63dc670229d419a6350b64f7ceda609dbc8194
Sana           2936c47637380842aaa4a4488fac5006cc542b70
```

Flow-Aligned Regenerate's API-2 `mixed_grid_low_suffix` producer is already merged/released in v0.3.0. Sol-H3 consumes that explicit contract; it does not infer arbitrary mixed layouts.

## What is already proven

### Packaged Sana/CuTe SM120

Real RTX PRO 6000 Blackwell execution established:

```text
PyTorch 2.10.0+cu130
CUDA runtime 13.0
GPU NVIDIA RTX PRO 6000 Blackwell Workstation Edition
compute capability (12, 0)
sol_source = sana-sol-engine
sana_revision = 2936c47637380842aaa4a4488fac5006cc542b70
sol_backend = cute_sm120
sol_source_tree_verified = true
```

The original 4096-row direct probe compiled/executed the packaged real CuTe kernel twice and reported exact dense-prefix parity. Its all-selected arithmetic gate reported:

```text
max_abs = 0.0009765625
mean_abs = 4.484307282837108e-05
rel_l2 = 0.003004377940669656
```

The rectangular SM120 GPU suite later passed **9 tests**, covering rectangular all-selected and sparse-sink behavior.

### Production native-grid rectangular VDN v3

Full SOL-enabled production runs with VDN + DiffAid + Untwist + SOL + Spectrum + progressive/Continuum now confirm that VDN provider API v3 is active:

```text
VDN object patches=50
softmax-provider module API=3
```

Native stages reported:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square-expanded calls |
|---|---:|---:|---:|---:|
| Native low grid | 2,112 | 4,992,000 | 4,992,000 | 0 |
| First high grid | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later high grid | 1,248 | 5,967,360 | 5,967,360 | 0 |

Thus the preferred v3 path executes direct rectangular Q/KV domains and does not construct the old v2-only square-Q compatibility payload for Sol-H3. This proves route activation and removes the historical square-Q kernel expansion. The available production timings are not a clean v3-only A/B because compile/cache state and Spectrum NFE schedules differ between runs.

### Production Flow mixed-grid SOL

The explicit API-2 mixed stage is also now exercised on the production GPU stack:

```text
sol_eligible_calls               250
external_mixed_sol_calls         192
dense_warmup                      58
external_mixed_q_rows       8,360,640
external_mixed_kernel_q_rows 8,360,640
compatibility_fallbacks           {}
rel_l2                    0.0009817115
```

The 58 dense calls are expected from `dense_evaluations=1` plus two configured dense layers. The remaining eligible mixed calls execute packaged `cute_sm120`. Later target-grid stages resume ordinary rectangular native SOL.

This proves coherent production routing and arithmetic behavior for the mixed path. It does not establish an isolated performance gain.

### Exact Runtime

Matched historical production A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | 247.30 s | 298.69 s | 17.18 GB |

This is Exact-only evidence, not a SOL speed claim.

## Spectrum / Diff-Aid history compatibility

Spectrum #104 is the generic backend-history consumer. It preflights provider identities and observes actual receipts before retaining H3 anchors. Sol-H3 must therefore prove its own replacement/routing topology; Spectrum does not declare Sol, Diff-Aid or Untwist transparent by name.

The final reviewed Spectrum consumer fails closed when a third-party backend-history metadata callback raises: that call becomes unprovable/actual-only instead of aborting the sampling run. CUDA out-of-memory remains a real resource failure and propagates. Spectrum also establishes the first provider identity without resetting an empty offline-capture archive; a provider appearing after backend-dependent evidence exists, provider removal, or a genuine identity/receipt transition still performs the full history reset.

Production exposed one important provider-side gap. The prior Sol history policy accepted only a bare Sol `BlockPatch`, while the real workflow composes MiniMax-H3 Diff-Aid around the DIT replacement. Sol therefore reported the route as opaque and Spectrum correctly forced actual transformer execution.

Current Sol-H3 recognizes only the audited Diff-Aid H3 activation-only chain when Diff-Aid publishes its existing Spectrum runtime declaration:

```text
spectrum_h3_external_patch_runtime
provider = comfyui-diffaid-patches
schema = 1
valid instance id
```

Both valid Sol/Diff-Aid wrapper orders are supported. Static Diff-Aid configuration is included in the history identity. Changing normalized sigma is excluded because Spectrum's existing external-patch compatibility layer owns patch-regime transitions. Cycles, duplicate/mismatched Sol patches, unknown wrappers, and missing Diff-Aid declarations remain opaque/actual-only. Untwist is not generically declared history-transparent through this mechanism.

The native interoperability suite covers both valid wrapper orders, stable identity across changing normalized sigma, repeated sampling scopes, missing declarations, and unknown wrappers.

## Current CI evidence

Spectrum #104 is one commit over v0.2.25/current `main`:

```text
Spectrum head:       9c682c07f4c5ea9de601cda234755a1561b59f59
Spectrum parent:     2482f52604da037c29e3f613e10057e8aaa83abb
Spectrum final CI:   #597 / 34278002378 — 9/9 green
CodeRabbit findings: both substantive backend-history findings verified/resolved
```

Sol-H3's final reviewed Spectrum repin was validated on a mirror before reconsolidation:

```text
Sol repin mirror: mirror/spectrum-104-final-repin-20260908
mirror head:      3929ec3cecc4ef12496aa7378022a730067148f5
mirror CI #168:   34278353336 — CPU-contract + native-interop green
```

The PR branch remains constrained to exactly one implementation commit over neutral Sol `main` `5db282ca836416a32cf114346b946fe75136e4f1`. Documentation updates do not change the validated runtime implementation; the final consolidated PR-head CI is the last CPU/native gate before the production GPU rerun.

CPU tests use explicit CPU substitutes where GPU kernels are unavailable. They establish contracts/composition, not GPU performance or decoded-media quality.

## Current checkout: dependency and source checks

Use the existing checkout as-is. No Git branch-changing command is required for validation.

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/comfyui-sol-h3

git status --short --branch
git rev-parse HEAD

python -m pip install -r requirements.txt
python -m pip check
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

Expected minimum evidence:

```text
sol_source = sana-sol-engine
sana_revision = 2936c47637380842aaa4a4488fac5006cc542b70
sol_backend = cute_sm120
sol_source_tree_verified = true
sparse_calls > 0
prefix_parity = true
```

## Optional SageAttention

Sage is an inherited dense provider, not the SOL kernel.

On SM120 use KJNodes `auto`, not `sageattn_qk_int8_pv_fp16_triton`.

If the installed Sage package fails to import because of a binary ABI problem, Sol-H3 must report the provider failure and continue with original Comfy dense attention where possible. It must not fake Sage success or disable SOL sparse execution globally.

A healthy optional Sage probe is:

```bash
python tools/attention_probe.py \
  --backend sage \
  --tokens 4096 \
  --prefix 512 \
  --heads 8
```

If Sage is unavailable at load time, expected telemetry includes a classified `dense_provider_failures` entry plus continuing SOL sparse calls. Arbitrary provider compute errors remain fatal.

For permanent Sage installation/repair, see [SAGEATTENTION](SAGEATTENTION.md). Do not use `LD_LIBRARY_PATH`/`LD_PRELOAD` workarounds.

## Production acceptance matrix

Restart ComfyUI after updating the node/companions and preserve the workflow settings used for A/B comparisons.

| Run | Combination | Required observation |
|---|---|---|
| A | Native dense + SOL | Eligible normal H3 calls can use `cute_sm120`; dense prefix/warmup delegated |
| B | Sage + SOL | Sage retained when usable; loader failure request-locally demoted while SOL continues |
| C | Spectrum + SOL | Qualified actual receipts establish history; route/provider transitions reset history rather than abort |
| D | VDN grouped + SOL | API v3 local rectangular SOL; requested/kernel Q rows 1:1; no square expansion |
| E | VDN flex + SOL | Masked Flex stays native; existing grouped fallback may then use v3 |
| F | VDN full coverage + SOL | Eligible softmax can use SOL; VDN learned gate remains active |
| G | VDN + Spectrum + SOL | Stable grouped/full routing may forecast only after qualified actual history |
| H | Flow API-2 `mixed_grid_low_suffix` | Explicit contract routes coherent whole-sequence external attention through SOL; VDN external gate retained, linear complement disabled |
| I | Mixed-grid stage -> later native grid | `sol_external_mixed` during valid mixed stage, then normal native SOL resumes |
| J | Malformed/unknown external contract | Local `external_sequence_*` fallback, generation continues |
| K | Repeated Continuum chunks | Fresh Sol-H3 request state, no stale VDN/Spectrum/provider state |
| L | Untwist + SOL + VDN | Full-domain preprocess occurs once before VDN gather; coordinates preserved |
| M | Runtime DoRA/LoRA + DiffAid + preview/progressive stack | Projection/hooks preserved; audited Diff-Aid replacement chain keeps qualified Spectrum history |
| N | Diff-Aid outside Sol / Sol outside Diff-Aid | Both audited wrapper orders produce stable backend identities; unknown wrappers remain actual-only |

## Native-grid success counters

For grouped VDN on the current v3 companion:

```text
sol_source == sana-sol-engine
sana_revision == 2936c47637380842aaa4a4488fac5006cc542b70
sol_backend == cute_sm120
sol_source_tree_verified == true
sol_eligible_calls > 0
sparse_calls > 0
vdn_local_sol_calls > 0
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls == 0
```

`vdn_global_native` and `vdn_anchor_native` may appear and are expected.

## Mixed-grid success counters

A workflow must actually enter the explicit Flow API-2 mixed topology. Then require:

```text
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
```

and route receipts containing:

```text
sol_external_mixed
```

The current implementation intentionally does not infer arbitrary external layouts. `external_sequence_native`, `external_sequence_contract`, or `external_sequence_layout` means that particular call delegated to inherited attention.

Production has already established `192` external mixed SOL calls and `8,360,640` requested/kernel Q rows with no compatibility fallbacks in the cited run. Future A/Bs still need matched NFE schedules before timing attribution.

## Performance interpretation: compare NFE schedules first

The most recent control exposed a large Spectrum schedule confound.

| | SOL enabled, prior hot run | SOL bypassed |
|---|---:|---:|
| End-to-end | 366.90 s | 287.51 s |
| Continuum sampler | 320.98 s | 240.55 s |
| Target-grid invocation | 144.44 s | 96.63 s |
| Mixed-grid invocation | 169.37 s | 136.64 s |
| Logical calls | 18 | 18 |
| Actual transformer NFEs | **18** | **13** |
| Spectrum forecasts | **0** | **5** |

The five avoided transformer calls are roughly an 80-90 s amount of work from neighboring actual-call timings, approximately the observed sampler delta. Therefore the raw wall-time difference is not evidence that SOL itself is ~25% slower.

The next SOL-enabled run must inspect these counters before wall-time interpretation:

```text
sampler_logical_calls
transformer_actual_nfe
spectrum_forecast_calls
```

Spectrum forecasts should reappear with the corrected provider history. Do not require exactly 13 actual + 5 forecast: legitimate dense/SOL ownership transitions can require fresh actual anchors. Once schedules are comparable, normalize target-grid/mixed-grid time by actual transformer NFEs and then evaluate SOL cost.

A possible later optimization target is the native SOL adapter's Q/K/V transpose+contiguous materialization versus suitable innermost-contiguous strided packed views accepted by the packaged Sana interface. Do not change that before the corrected schedule-matched production rerun; the previous five-NFE discrepancy is large enough to invalidate performance attribution.

## Performance and media acceptance

Preserve:

- exact model/adapter files and hashes;
- quantization;
- seed;
- sigma schedule/sampler;
- prompt/dialogue;
- reference order;
- dimensions/duration;
- VDN configuration;
- Spectrum actual/forecast schedule;
- DiffAid/Untwist/runtime adapter settings;
- progressive/Continuum settings.

Record:

```text
cold/warm latency
sampler wall time
end-to-end wall time
peak allocated/reserved VRAM
sampler_logical_calls
transformer_actual_nfe
spectrum_forecast_calls
sol_eligible_calls
sparse_calls
dense_warmup
compatibility_fallbacks
dense_provider_failures
external_mixed_sol_calls
external_mixed_q_rows
external_mixed_kernel_q_rows
vdn_rectangular_sol_calls
vdn_requested_q_rows
vdn_kernel_q_rows
vdn_square_expanded_calls
numerical_backend_transitions
```

Inspect decoded video identity, reference fidelity, motion/grass artifacts, temporal stability, progressive/Continuum boundaries, and the entire audio track for missing/wrong words, unsolicited speech, volume/whisper fidelity, synchronization, and discontinuities.

Successful sparse execution proves routing, not quality or speed.
