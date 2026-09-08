# Validation status and RTX PRO 6000 commands

**Production RTX PRO 6000 evidence exists for Exact Runtime, including a matched Exact-off/on timing A/B.** Two later sparse-SOL runs were diagnostically useful: the first exposed the pre-v2 VDN routing defect; the latest post-v2 run reached SOL eligibility but executed zero sparse calls only because the old loader unnecessarily required a separate Sana package. The current head removes that external-loader requirement and uses ComfyUI's installed `comfy_kitchen.sol_attn` directly.

## Completed evidence

- Original SOL baseline: 45 passed, 19 skipped before Triton installation.
- Previous broad local suite: 66 passed, 18 GPU skips.
- Previous v2 Sol mirror: **57 passed, 30 expected GPU/optional-integration skips**; `pip check` and Ruff passed.
- Previous native interoperability CI: **12 passed**, exercising current Comfy ModelPatcher/wrappers, current KJ Python wrapper code, Spectrum and the sequential VDN PR #8 -> PR #11 stack with CPU kernel substitutes.
- VDN v2 companion CI: pinned-Comfy + official-oracle suite, legacy migration and current-Comfy smoke all pass.
- Spectrum companion baseline: 911 passed, 13 skipped against the audited Comfy revision.
- Untwist companion baseline: 40 passed.

CPU structural tests replace unavailable GPU kernels explicitly. They do not establish GPU kernel correctness, performance or media quality.

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

The v2 bridge now supplies an expanded query tensor over exactly the same restricted KV row domain and a mapping back to VDN's originally requested rows. CPU oracle tests verify that selecting those outputs reproduces the original rectangular attention result, and a real ModelPatcher/object-patch integration test verifies that the VDN object patch actually reaches Sol-H3 after `patch_model()`.

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

At the same process startup, ComfyUI reported `comfy-kitchen 0.2.33` and CUDA capability `sol_attn` as available. Therefore this was not a GPU/kernel-capability failure: Sol-H3 was simply bypassing the already-installed ComfyUI kernel and insisting on a duplicate external package. That requirement is removed on the current head. This run is routing evidence only and must not be used as a sparse-SOL timing result.

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

No Sol-H3-specific environment variables are required. Verify the kernel already shipped with the ComfyUI environment:

```bash
conda activate comfy312
cd /home/toor/ComfyUI
python - <<'PY'
import torch
import comfy_kitchen as ck

device = torch.device('cuda:0')
print('gpu:', torch.cuda.get_device_name(device))
print('capability:', torch.cuda.get_device_capability(device))
print('comfy_kitchen sol_attn:', ck.sol_attn_is_available(device))
assert torch.cuda.get_device_capability(device) == (12, 0)
assert ck.sol_attn_is_available(device)
PY

cd /home/toor/ComfyUI/custom_nodes/comfyui-sol-h3
python -m pytest -q tests/test_gpu.py
python tools/gpu_probe.py --tokens 4096 --hidden 5376
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage3 --tokens 4096 --prefix 512
```

Sage/Sage3 commands require their installed extensions. Expected: the kernel availability assertion passes, GPU tests have zero skips, affine probe reports parity, and each attention probe reports sparse execution plus a successful independent arithmetic gate. The first shape includes kernel setup/cache cost; record cold and warmed costs separately.

Launch ComfyUI normally:

```bash
cd /home/toor/ComfyUI
python main.py 2>&1 | tee /home/toor/sol_h3_full_stack.log
```

There must be **no** `SOL_ROOT`, Sana checkout, or Sol-H3-specific `PYTHONPATH` requirement. Before the full run, ensure the startup/install log reports VDN provider API 2 and `comfy_kitchen` reports `sol_attn` available.

## Full-stack acceptance matrix

Start with Exact off to isolate sparse interoperability, then enable it and compare again.

| Run | Combination | Required observation |
|---|---|---|
| A | Native selected dense provider | Baseline latency and media |
| B | Sage + SOL | Inherited dense backend recorded; sparse calls after warmup |
| C | Sage3 + SOL | Same routing checks; independent gate passes |
| D | Spectrum + SOL, then Sage + Spectrum + SOL | SOL actual count agrees with actual transformer calls; history resets at backend transitions |
| E | VDN grouped + SOL, with and without Sage | v2 local square expansion executes SOL on eligible restricted domains; `vdn_square_kernel_rows > vdn_square_requested_rows > 0`; global/anchor remain native |
| F | VDN flex + SOL | Masked Flex remains native; if Flex falls back to grouped, grouped v2 routing is visible. With Spectrum, Flex remains actual-only because the fallback outcome is not preflight-predictable |
| G | VDN full coverage + SOL | Eligible softmax can use SOL; learned gate remains active |
| H | VDN + Spectrum + SOL | Grouped/full VDN publishes a stable audited history identity and can forecast after qualified actuals; opaque/Flex routing stays actual-only without aborting |
| I | Flow API-1 reduced / API-2 mixed, then normal target grid | `external_sequence_native` on unsupported calls; later native-grid v2 SOL eligibility resumes |
| J | Repeated Continuum chunks | Fresh sampling scopes, correct warmup, no stale VDN/SOL/Spectrum state |
| K | Core Block Sparse Attention + Spectrum, with/without Sol-H3 | Current core provider is actual-only under the companion history consumer; explicit ownership recorded |
| L | DiffAid + Untwist + runtime DoRA/LoRA + KJ preview + production stack | Projection/key transforms preserved; VDN preprocessing occurs before local gather; any remaining fallback is explicitly identified |

For the first current-head production run, the decisive native-grid telemetry is:

```text
kernel_contract == comfy-kitchen-sol-attn-64-v1
sol_eligible_calls > 0
sparse_calls > 0
vdn_local_sol_calls > 0
vdn_square_expanded_calls > 0
vdn_square_kernel_rows > vdn_square_requested_rows > 0
```

`vdn_global_native`, `vdn_anchor_native` and mixed-grid `external_sequence_native` may still appear and are expected. Their presence is not a failure. `kernel_unavailable:...comfy-kitchen...` is meaningful only if `comfy_kitchen.sol_attn_is_available(cuda:0)` is false; the code no longer searches for an external Sana package.

Spectrum preflight does not require PR #11 to modify `vdn_h3/hybrid.py`. For the audited VDN closure shape, Sol-H3 derives the grouped routing identity from the captured VDN state/config and includes model layout plus inherited provider identity in the numerical-history key. If that closure shape changes, the backend is unknown, or Flex is selected, preflight returns opaque and Spectrum executes actual calls. That is a conservative performance fallback, not a runtime incompatibility.

Also exercise Exact -> SOL, SOL exact=true -> Exact, SOL exact=false -> Exact, repeated identical applications, independently cloned branches, all-warmup requests and all-fallback requests. Legitimate zero-sparse runs must finish, but they are not evidence of SOL acceleration.

## Performance and media acceptance

Preserve exact model/adapter files and hashes, quantization, seed, sigma schedule, sampler, prompt/dialogue, reference order, dimensions, duration, audio conditioning and patch settings. Progressive lower-resolution stages change composition; matching only the seed does not make them composition-matched A/B samples.

For VDN v2 specifically record the ratio of `vdn_square_kernel_rows / vdn_square_requested_rows` together with kernel time. The square bridge intentionally computes additional Q rows to preserve VDN's restricted KV geometry while satisfying the square kernel, so its gather/query overhead can erase the sparse benefit even when execution is correct.

Record cold/warm latency, end-to-end wall time, peak allocated/reserved VRAM, actual/forecast counts, sparse/eligible/warmup counts, fallback reasons, VDN-local SOL calls, VDN square row counts and history resets. Use warmed repetitions at identical shapes before making a speed claim. Compare Sol-H3's direct-QKV `comfy_kitchen.sol_attn` policy against ComfyUI core Block Sparse Attention's chunked producer as separate integration policies.

Inspect video identity, reference fidelity, motion/grass artifacts, temporal stability and Continuum boundaries. Listen to the entire audio for wrong/missing words, unsolicited speech, whisper/volume fidelity, synchronization and discontinuities. Tensor metrics and unit tests cannot replace this assessment.
