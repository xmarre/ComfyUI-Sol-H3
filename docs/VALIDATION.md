# Validation status and RTX PRO 6000 commands

**Production RTX PRO 6000 evidence exists for Exact Runtime, including a matched Exact-off/on timing A/B.** Two later sparse-SOL runs were diagnostically useful: the first exposed the pre-v2 VDN routing defect; the latest post-v2 run reached SOL eligibility but executed zero sparse calls only because the old loader unnecessarily required a separate Sana package. The current redesign packages the actual Sana source and selects its CuTe SM120 backend. No GPU execution of this redesign is yet established.

## Current source-integration validation

- Full local CPU suite with real Comfy/KJ/Spectrum/VDN integrations: **79 passed, 21 CUDA skips**.
- Source contract + native interoperability lane: **22 passed** before the final two negative provenance/dependency tests were added; both are included in the full-suite count.
- All 51 vendored files match the pinned upstream git objects after the documented import-only transformation.
- Wheel inspection confirms all 51 files, manifest and license texts are packaged.
- Declared dependencies installed successfully; `pip check` and Ruff pass.
- Local environment: Python 3.12, PyTorch 2.14.0, Triton 3.8.0, CUTLASS DSL 4.7.1, CUDA Python 13.3.1, TVM FFI 0.1.13.post3. Real SM120 module import/kernel construction passed; no CUDA device is available here.
- CI additionally retains PyTorch 2.10.0 CPU / Triton 3.6.0 baseline coverage.
- Existing Exact GPU evidence below is carried forward from the handoff; Exact implementation was not changed. Sana SM120 GPU compilation, numerical output, speed and audiovisual quality remain unvalidated.

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

The handoff also reports comfy-kitchen availability at startup. That is evidence about a different integration; it cannot establish Sana CuTe capability. Packaging removes the manual external-source requirement. The reported run is routing evidence only.

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

Install declared dependencies in `comfy312`, then check the node-local source:

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
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
python -m pytest -q tests/test_gpu.py
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage3 --tokens 4096 --prefix 512
```

Sage/Sage3 commands require their installed extensions. Expected: the backend assertion passes and GPU tests have zero skips, and each attention probe reports sparse execution plus a successful independent arithmetic gate. The first shape includes kernel setup/cache cost; record cold and warmed costs separately.

Launch ComfyUI normally:

```bash
cd /home/toor/ComfyUI
python main.py 2>&1 | tee /home/toor/sol_h3_full_stack.log
```

There must be **no** `SOL_ROOT`, Sana checkout, or Sol-H3-specific `PYTHONPATH` requirement. Before the full run, ensure the startup/install log reports VDN provider API 2 and the node-local backend check reports `cute_sm120`.

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

`vdn_global_native`, `vdn_anchor_native` and mixed-grid `external_sequence_native` may still appear and are expected. Their presence is not a failure. A `kernel_unavailable` reason means Sana initialization failed and the run cannot establish sparse-SOL success. Read the concrete dependency/provenance error; no external Sana checkout is needed.

Spectrum preflight does not require PR #11 to modify `vdn_h3/hybrid.py`. For the audited VDN closure shape, Sol-H3 derives the grouped routing identity from the captured VDN state/config and includes model layout plus inherited provider identity in the numerical-history key. If that closure shape changes, the backend is unknown, or Flex is selected, preflight returns opaque and Spectrum executes actual calls. That is a conservative performance fallback, not a runtime incompatibility.

Also exercise Exact -> SOL, SOL exact=true -> Exact, SOL exact=false -> Exact, repeated identical applications, independently cloned branches, all-warmup requests and all-fallback requests. Legitimate zero-sparse runs must finish, but they are not evidence of SOL acceleration.

## Performance and media acceptance

Preserve exact model/adapter files and hashes, quantization, seed, sigma schedule, sampler, prompt/dialogue, reference order, dimensions, duration, audio conditioning and patch settings. Progressive lower-resolution stages change composition; matching only the seed does not make them composition-matched A/B samples.

For VDN v2 specifically record the ratio of `vdn_square_kernel_rows / vdn_square_requested_rows` together with kernel time. The square bridge intentionally computes additional Q rows to preserve VDN's restricted KV geometry while satisfying the square kernel, so its gather/query overhead can erase the sparse benefit even when execution is correct.

Record cold/warm latency, end-to-end wall time, peak allocated/reserved VRAM, actual/forecast counts, sparse/eligible/warmup counts, fallback reasons, VDN-local SOL calls, VDN square row counts and history resets. Use warmed repetitions at identical shapes before making a speed claim. Compare Sol-H3's packaged Sana direct-QKV policy against ComfyUI core Block Sparse Attention's chunked producer as separate integration policies.

Inspect video identity, reference fidelity, motion/grass artifacts, temporal stability and Continuum boundaries. Listen to the entire audio for wrong/missing words, unsolicited speech, whisper/volume fidelity, synchronization and discontinuities. Tensor metrics and unit tests cannot replace this assessment.
