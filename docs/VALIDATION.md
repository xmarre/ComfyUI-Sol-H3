# Validation status and RTX PRO 6000 commands

The current revision combines Exact Runtime, rectangular Sana/CuTe SM120 attention, VDN provider API v3, and explicit Flow API-2 mixed-grid SOL routing.

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

### Production native-grid rectangular VDN

A live Comfy production run with VDN + DiffAid + Untwist + SOL + Spectrum + progressive/Continuum completed successfully on the rectangular revision. Native stages reported:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square-expanded calls |
|---|---:|---:|---:|---:|
| Native low grid | 2,112 | 4,992,000 | 4,992,000 | 0 |
| First high grid | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later high grid | 1,248 | 5,967,360 | 5,967,360 | 0 |

That run used the prior VDN provider API v2. Sol-H3 already executed only requested Q, but VDN still constructed the unused v2 `square_q` compatibility payload. Current VDN PR #11 API v3 removes that gather/allocation. The v3 integration is structurally/CI validated; its GPU timing improvement has not yet been measured.

### Exact Runtime

Matched historical production A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | 247.30 s | 298.69 s | 17.18 GB |

This is Exact-only evidence, not a SOL speed claim.

## Current CI evidence

The validated mirror code passes:

```text
full Sol-H3 suite: 90 passed, 42 skipped, 2 warnings
native interoperability: 24 passed
Ruff: pass
pip check: pass
```

The native lane checks the sequential VDN overlay order:

```text
VDN PR #8 b6f0755c4172ec5c17386c56998f454e78b2a2d4
then
VDN PR #11 5b63dc670229d419a6350b64f7ceda609dbc8194
```

and covers real Comfy ModelPatcher integration, KJ wrappers, Spectrum wrapper ordering, VDN API v3, lazy v2 compatibility, Flow API-2 mixed-grid SOL routing, malformed-contract fallback, and mixed -> later-native SOL resumption.

CPU tests use explicit CPU substitutes where GPU kernels are unavailable. They do not establish GPU performance or decoded-media quality.

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
| H | Flow API-2 `mixed_grid_low_suffix` | Explicit contract can route whole-sequence external attention through SOL; VDN external gate semantics retained, linear complement disabled |
| I | Mixed-grid stage -> later native grid | `sol_external_mixed` during valid mixed stage, then normal native SOL may resume |
| J | Malformed/unknown external contract | Local `external_sequence_*` fallback, generation continues |
| K | Repeated Continuum chunks | Fresh Sol-H3 request state, no stale VDN/Spectrum/provider state |
| L | Untwist + SOL + VDN | Full-domain preprocess occurs once before VDN gather; coordinates preserved |
| M | Runtime DoRA/LoRA + DiffAid + preview/progressive stack | Projection/hooks preserved; unsupported ownership delegated explicitly |

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

The newly enabled mixed-grid SOL path is not yet production-GPU/media validated. The previous successful progressive run predates this route and correctly kept the mixed stage dense. A new run is needed before claiming mixed-grid quality or performance.

## Spectrum interpretation

Spectrum may forecast only from qualified history matching the current numerical attention owner. Important transitions include:

- dense provider demotion;
- native <-> SOL route changes;
- external mixed <-> normal native-grid transitions;
- opaque Flex/fallback outcomes.

A history reset is not a runtime incompatibility. It is the conservative response to a changed numerical backend.

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
actual/forecast transformer counts
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
