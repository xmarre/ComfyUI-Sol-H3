# Current rectangular revision

Use [RECTANGULAR](RECTANGULAR.md) for current commands and counters. The earlier square-bridge validation below is historical; its GPU measurements do not validate the rectangular revision.

# Validation status and RTX PRO 6000 commands

**Production RTX PRO 6000 evidence exists for Exact Runtime, and direct synthetic GPU evidence now exists for the packaged Sana CuTe SM120 SOL kernel.** Full ComfyUI/VDN/Spectrum sparse execution, warmed performance and audiovisual acceptance remain outstanding.

## Current source-integration validation

- The node packages the actual `xmarre/Sana` `sol-engine` source at revision `2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`.
- All 51 vendored files match the pinned upstream git objects after the documented import-only transformation.
- Wheel inspection confirms all 51 files, manifest and license texts are packaged.
- Declared dependencies install normally; no Sana checkout, `SOL_ROOT`, custom `PYTHONPATH`, linker environment edit or runtime download is required.
- CPU/native integration CI covers current Comfy ModelPatcher, KJ attention-provider chaining, Spectrum and the ordered VDN PR #8 -> PR #11 stack.
- Optional inherited dense-provider **loader/import** failures are request-locally demoted to the original Comfy attention callable. This is intentionally limited to `ImportError`/`OSError` loader failures; CUDA/runtime compute failures still propagate. Explicit preprocessing contracts such as Untwist remain applied exactly once before the fallback dense owner.
- Provider demotion is recorded in `dense_provider_failures`, increments the numerical backend transition, and changes Spectrum's provider-history identity so a forecast cannot cross the demotion as though the numerical backend were unchanged.

CPU structural tests replace unavailable GPU kernels explicitly. They do not establish GPU kernel correctness, performance or media quality.

### Completed RTX PRO 6000 Sana/CuTe SM120 kernel execution

A direct local probe was run in the production WSL/conda environment before starting ComfyUI:

```text
PyTorch: 2.10.0+cu130
CUDA: 13.0
Triton: 3.6.0
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
compute capability: (12, 0)
```

The packaged source verified all 51 files and reported:

```text
sol_source: sana-sol-engine
sana_revision: 2936c47637380842aaa4a4488fac5006cc542b70
kernel_contract: sana-sol-engine-sol-attn-64-v1
sol_backend: cute_sm120
sol_source_tree_verified: true
```

`tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512 --heads 8` then compiled and executed the real packaged Sana SM120 CuTe path. It completed with:

```text
sparse_calls: 2
prefix_parity: true
fallbacks: {}
```

The independent all-selected arithmetic gate against BF16 SDPA reported:

```text
max_abs: 0.0009765625
mean_abs: 4.484307282837108e-05
rel_l2: 0.003004377940669656
```

This is real GPU compilation/execution evidence for the packaged SOL kernel and its prefix bridge. It is **not** yet evidence for production VDN square-expanded execution, Spectrum forecasting compatibility, end-to-end speed or media quality.

A second probe requesting the installed SageAttention 2 Triton provider exposed an independent binary-loader problem in that Sage installation: its `_fused` extension resolved `/lib/x86_64-linux-gnu/libstdc++.so.6` and required `GLIBCXX_3.4.32`, while the active conda environment already contained a newer compatible `libstdc++`. Sol-H3 must not require users to repair that with `LD_LIBRARY_PATH`, preloading or a second C++ runtime. Current runtime/probe policy therefore treats an unavailable optional dense provider as a local provider demotion while keeping the verified Sana SOL path active. A successful demotion probe must still report `sparse_calls > 0`, `sol_backend=cute_sm120` and exact prefix parity, together with the explicit dense-provider failure.

For a permanent SageAttention repair, rebuild the official package against the same ComfyUI environment/toolchain instead of overriding the process linker. See **[SageAttention installation and repair](SAGEATTENTION.md)**. On SM120, use KJNodes `auto`: upstream SageAttention 2.2.0 dispatches `sageattn` to its CUDA FP8/SageAttention2++ path and explicitly marks its Triton path unusable on SM120.

### Completed production-GPU Exact Runtime A/B

A matched hot A/B was run on the RTX PRO 6000 production workflow with the exact/runtime tier only; approximate SOL attention was not enabled. The stack included the production INT8/ConvRot H3 path with VDN, DiffAid, Untwist RoPE, Spectrum, progressive handoff and Continuum.

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | 247.30 s | 298.69 s | 17.18 GB |

Observed improvement: **6.17% lower sampler time** (1.066x) and **4.44% lower end-to-end time** (1.046x), with unchanged measured peak VRAM.

Exact-mode telemetry reported `success=true`, `backend=inherit`, `approximate=false`, `sparse_calls=0`, and the expected fused-block accounting (for example 3 actual H3 evaluations -> 150 exact blocks on the 50-block model). This is real GPU/runtime evidence for Exact Runtime and its integration with that workflow stack. It is not sparse-SOL evidence.

### Production reproduction of the pre-v2 sparse-routing failure

A later run used the intended production ordering:

```text
VDN -> runtime DoRA -> DiffAid -> Untwist -> SOL -> Spectrum -> preview/progressive/Continuum
```

with Sage enabled in the H3 loader. Native-grid Sol-H3 telemetry still reported zero sparse eligibility/execution (`sol_eligible_calls=0`, `sparse_calls=0`, `vdn_local_sol_calls=0`) while Exact blocks executed. Mixed-grid calls reported `external_sequence_native`, which is expected for that topology.

That run established the interoperability defect that provider API v2 addresses: normal VDN local query rows are rectangular against their restricted KV domain because prefix/global KV rows are included, while the SOL kernel requires square Q/K/V.

The v2 bridge supplies an expanded query tensor over exactly the same restricted KV row domain and a mapping back to VDN's originally requested rows. CPU oracle tests verify that selecting those outputs reproduces the original rectangular attention result, and a real ModelPatcher/object-patch integration test verifies that the VDN object patch actually reaches Sol-H3 after `patch_model()`.

### Production reproduction of the obsolete external-loader failure

The first post-v2 production run reached the intended provider path:

```text
sol_eligible_calls: 2750
```

but still reported:

```text
sparse_calls: 0
kernel_unavailable: Install the pinned Sol-Attn backend and expose its parent directory on PYTHONPATH
```

That run is routing evidence only. The external loader is obsolete: current code packages and verifies the actual Sana source locally, and the direct SM120 probe above proves the packaged CuTe kernel can compile and execute on the target GPU.

## Reproduce CPU contracts

From the Sol-H3 checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
KJNODES_PATH=/path/to/ComfyUI-KJNodes \
SPECTRUM_PATH=/path/to/ComfyUI-Spectrum-MiniMax-H3 \
VDN_PATH=/path/to/ComfyUI-VDN-H3-Plus \
python -m pytest -q
```

Use the linked companion PR branches/heads. For VDN, the integration lane intentionally checks out PR #8 first and cherry-picks the one-commit PR #11 overlay afterward; this is the supported sequential stack and is itself a merge-conflict regression check.

## Production machine: kernel checks

In the active ComfyUI environment, without changing Git state or linker environment:

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/comfyui-sol-h3
python -m pip install -r requirements.txt
python - <<'PYCODE'
import torch
from sol_h3.provenance import verify_source
from sol_h3.sparse import load_kernel
print(verify_source()['revision'])
kernel = load_kernel(torch.device('cuda:0'))
print(kernel.backend_name)
assert kernel.backend_name == 'cute_sm120'
PYCODE
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512 --heads 8
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512 --heads 8
```

The PyTorch probe is the minimum kernel acceptance check and must actually execute sparse calls. Sage is an optional inherited dense provider. If its binary/package loader is unavailable, the probe must not fake Sage success: it should report `dense_provider_fallback=true`, the classified `dense_provider_failures`, the effective Comfy/PyTorch dense owner and continuing SOL sparse calls. Arbitrary provider compute failures remain fatal.

SageAttention3 is deliberately not part of the production Python 3.12 acceptance commands: upstream currently documents SageAttention3 with Python >=3.13, PyTorch >=2.8 and CUDA >=12.8. Test it only in a separate environment satisfying those requirements.

Launch ComfyUI normally after these checks. There must be **no** `SOL_ROOT`, Sana checkout, Sol-H3-specific `PYTHONPATH`, `LD_LIBRARY_PATH` workaround or `ctypes` preload requirement.

## Full-stack acceptance matrix

Start with Exact off to isolate sparse interoperability, then enable it and compare again.

| Run | Combination | Required observation |
|---|---|---|
| A | Native selected dense provider | Baseline latency and media |
| B | Sage + SOL | If Sage is usable, inherited Sage is recorded; if its loader is unavailable, explicit one-time demotion to original Comfy dense attention is recorded while SOL continues |
| C | Spectrum + SOL, then Sage + Spectrum + SOL | SOL actual count agrees with actual transformer calls; history resets at backend/provider demotions and transitions |
| D | VDN grouped + SOL, with and without Sage | v2 local square expansion executes SOL on eligible restricted domains; `vdn_square_kernel_rows > vdn_square_requested_rows > 0`; global/anchor remain native |
| E | VDN flex + SOL | Masked Flex remains native; if Flex falls back to grouped, grouped v2 routing is visible. With Spectrum, Flex remains actual-only because the fallback outcome is not preflight-predictable |
| F | VDN full coverage + SOL | Eligible softmax can use SOL; learned gate remains active |
| G | VDN + Spectrum + SOL | Grouped/full VDN publishes a stable audited history identity and can forecast after qualified actuals; opaque/Flex routing stays actual-only without aborting |
| H | Flow API-1 reduced / API-2 mixed, then normal target grid | `external_sequence_native` on unsupported calls; later native-grid v2 SOL eligibility resumes |
| I | Repeated Continuum chunks | Fresh sampling scopes, correct warmup, no stale VDN/SOL/Spectrum state |
| J | Core Block Sparse Attention + Spectrum, with/without Sol-H3 | Current core provider is actual-only under the companion history consumer; explicit ownership recorded |
| K | DiffAid + Untwist + runtime DoRA/LoRA + KJ preview + production stack | Projection/key transforms preserved; VDN preprocessing occurs before local gather; any remaining fallback is explicitly identified |

For the first current-head production run, the decisive native-grid telemetry is:

```text
sol_source == sana-sol-engine
sana_revision == 2936c47637380842aaa4a4488fac5006cc542b70
sol_backend == cute_sm120
sol_source_tree_verified == true
kernel_contract == sana-sol-engine-sol-attn-64-v1
sol_eligible_calls > 0
sparse_calls > 0
vdn_local_sol_calls > 0
vdn_square_expanded_calls > 0
vdn_square_kernel_rows > vdn_square_requested_rows > 0
```

`vdn_global_native`, `vdn_anchor_native` and mixed-grid `external_sequence_native` may still appear and are expected. Their presence is not a failure. A `kernel_unavailable` reason means Sana initialization failed and the run cannot establish sparse-SOL success.

When an optional inherited dense provider cannot load, expect explicit telemetry such as:

```text
dense_provider_failures: {"<provider>:binary_abi": 1}
compatibility_fallbacks: {"dense_provider_unavailable:binary_abi": 1, ...}
inherited_dense_backends: ["<effective original Comfy provider>", ...]
```

The provider is attempted once per sampling request, then bypassed. Preprocessing transforms remain active, the numerical-history identity changes, and Spectrum may not forecast across the transition as if nothing changed.

Spectrum preflight does not require PR #11 to modify `vdn_h3/hybrid.py`. For the audited VDN closure shape, Sol-H3 derives the grouped routing identity from the captured VDN state/config and includes model layout plus inherited provider identity in the numerical-history key. If that closure shape changes, the backend is unknown, or Flex is selected, preflight returns opaque and Spectrum executes actual calls. That is a conservative performance fallback, not a runtime incompatibility.

Also exercise Exact -> SOL, SOL exact=true -> Exact, SOL exact=false -> Exact, repeated identical applications, independently cloned branches, all-warmup requests and all-fallback requests. Legitimate zero-sparse runs must finish, but they are not evidence of SOL acceleration.

## Performance and media acceptance

Preserve exact model/adapter files and hashes, quantization, seed, sigma schedule, sampler, prompt/dialogue, reference order, dimensions, duration, audio conditioning and patch settings. Progressive lower-resolution stages change composition; matching only the seed does not make them composition-matched A/B samples.

For VDN v2 specifically record the ratio of `vdn_square_kernel_rows / vdn_square_requested_rows` together with kernel time. The square bridge intentionally computes additional Q rows to preserve VDN's restricted KV geometry while satisfying the square kernel, so its gather/query overhead can erase the sparse benefit even when execution is correct.

Record cold/warm latency, end-to-end wall time, peak allocated/reserved VRAM, actual/forecast counts, sparse/eligible/warmup counts, fallback reasons, dense-provider failures, VDN-local SOL calls, VDN square row counts and history resets. Use warmed repetitions at identical shapes before making a speed claim. Compare Sol-H3's packaged Sana direct-QKV policy against ComfyUI core Block Sparse Attention's chunked producer as separate integration policies.

Inspect video identity, reference fidelity, motion/grass artifacts, temporal stability and Continuum boundaries. Listen to the entire audio for wrong/missing words, unsolicited speech, whisper/volume fidelity, synchronization and discontinuities. Tensor metrics and unit tests cannot replace this assessment.
