# Validation status and RTX PRO 6000 commands

**Production RTX PRO 6000 evidence exists for Exact Runtime, including a matched Exact-off/on timing A/B.** A later correctly ordered sparse-SOL production attempt also provided a useful failure reproduction: the pre-v2 VDN bridge executed zero SOL sparse calls. The v2 interoperability head fixes that structural routing gap in CPU/native integration tests, but still needs a fresh GPU acceptance run.

## Completed evidence

- Original SOL baseline: 45 passed, 19 skipped before Triton installation.
- Previous broad local suite: 66 passed, 18 GPU skips.
- Current Linux CI on the v2 Sol mirror: **55 passed, 30 expected GPU/optional-integration skips**; `pip check` and Ruff pass.
- Current native interoperability CI: **12 passed**, exercising current Comfy ModelPatcher/wrappers, current KJ Python wrapper code, Spectrum and the consolidated VDN v2 companion with CPU kernel substitutes.
- VDN v2 companion CI: pinned-Comfy + official-oracle suite, legacy workflow migration and current-Comfy smoke all pass.
- Spectrum companion baseline: 911 passed, 13 skipped against the audited Comfy revision.
- Untwist companion baseline: 40 passed.

CPU structural tests replace unavailable external Sage/SOL CUDA kernels explicitly. They do not establish GPU kernel correctness, performance or media quality.

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

This run must **not** be used as a sparse-SOL timing result. It established the interoperability defect that provider API v2 addresses: normal VDN local query rows are rectangular against their restricted KV domain because prefix/global KV rows are included, while the pinned NVIDIA SOL kernel requires square Q/K/V.

The v2 bridge now supplies an expanded query tensor over exactly the same restricted KV row domain and a mapping back to VDN's originally requested rows. CPU oracle tests verify that selecting those outputs reproduces the original rectangular attention result, and a real ModelPatcher/object-patch integration test verifies that the VDN object patch actually reaches Sol-H3 after `patch_model()`.

## Reproduce CPU contracts

From the Sol-H3 checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
KJNODES_PATH=/path/to/ComfyUI-KJNodes \
SPECTRUM_PATH=/path/to/ComfyUI-Spectrum-MiniMax-H3 \
VDN_PATH=/path/to/ComfyUI-VDN-H3-Plus \
python -m pytest -q
```

Use the linked companion PR branches/heads. The CI pins exact tested revisions.

## Production machine: kernel checks

In WSL, using the documented pinned Sana checkout:

```bash
conda activate comfy312
COMFYUI_ROOT=/home/toor/ComfyUI
SOL_ROOT=/home/toor/Sana-sol-h3
cd "$COMFYUI_ROOT/custom_nodes/ComfyUI-Sol-H3"
export PYTHONPATH="$SOL_ROOT/models/minimax_h3/Sol-H3/h3_runtime/third_party${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests/test_gpu.py
python tools/gpu_probe.py --tokens 4096 --hidden 5376
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage3 --tokens 4096 --prefix 512
```

Sage/Sage3 commands require their installed extensions. Expected: GPU tests have zero skips; affine probe reports parity; each attention probe reports sparse execution and successful independent arithmetic gates. The first shape includes compile/verification cost; record cold and warmed costs separately.

Launch the production stack with the same environment:

```bash
cd "$COMFYUI_ROOT"
python main.py 2>&1 | tee /home/toor/sol_h3_full_stack.log
```

Before the full run, ensure the startup/install log reports VDN object-patch provider API **2**. If it reports v1/missing, the VDN checkout is stale.

## Full-stack acceptance matrix

Start with Exact off to isolate sparse interoperability, then enable it and compare again.

| Run | Combination | Required observation |
|---|---|---|
| A | Native selected dense provider | Baseline latency and media |
| B | Sage + SOL | Inherited dense backend recorded; sparse calls after warmup |
| C | Sage3 + SOL | Same routing checks; independent gate passes |
| D | Spectrum + SOL, then Sage + Spectrum + SOL | SOL actual count agrees with actual transformer calls; history resets at backend transitions |
| E | VDN grouped + SOL, with and without Sage | v2 local square expansion executes SOL on eligible restricted domains; `vdn_square_kernel_rows > vdn_square_requested_rows > 0`; global/anchor remain native |
| F | VDN flex + SOL | Masked Flex remains native; if Flex falls back to grouped, grouped v2 routing is visible |
| G | VDN full coverage + SOL | Eligible softmax can use SOL; learned gate remains active |
| H | VDN + Spectrum + SOL | Backend receipts cover VDN multi-call routing; no incompatible cross-backend anchors |
| I | Flow API-1 reduced / API-2 mixed, then normal target grid | `external_sequence_native` on unsupported calls; later native-grid v2 SOL eligibility resumes |
| J | Repeated Continuum chunks | Fresh sampling scopes, correct warmup, no stale VDN/SOL/Spectrum state |
| K | Core Block Sparse Attention + Spectrum, with/without SOL | Current core provider is actual-only under the companion history consumer; explicit ownership recorded |
| L | DiffAid + Untwist + runtime DoRA/LoRA + KJ preview + production stack | Projection/key transforms preserved; VDN preprocessing occurs before local gather; any remaining fallback is explicitly identified |

For the first post-v2 production run, the decisive native-grid telemetry is:

```text
sol_eligible_calls > 0
sparse_calls > 0
vdn_local_sol_calls > 0
vdn_square_expanded_calls > 0
vdn_square_kernel_rows > vdn_square_requested_rows > 0
```

`vdn_global_native`, `vdn_anchor_native` and mixed-grid `external_sequence_native` may still appear and are expected. Their presence is not a failure. If the native-grid stage still reports `vdn_provider_contract_missing`, `vdn_provider_v1_not_consumed` or `vdn_provider_not_consumed`, save the complete log; those reasons now identify the exact lifecycle break instead of collapsing it into generic inherited ownership.

Also exercise Exact → SOL, SOL exact=true → Exact, SOL exact=false → Exact, repeated identical applications, independently cloned branches, all-warmup requests and all-fallback requests. Legitimate zero-sparse runs must finish, but they are not evidence of SOL acceleration.

## Performance and media acceptance

Preserve exact model/adapter files and hashes, quantization, seed, sigma schedule, sampler, prompt/dialogue, reference order, dimensions, duration, audio conditioning and patch settings. Progressive lower-resolution stages change composition; matching only the seed does not make them composition-matched A/B samples.

For VDN v2 specifically record the ratio of `vdn_square_kernel_rows / vdn_square_requested_rows` together with kernel time. The square bridge intentionally computes additional Q rows to preserve VDN's restricted KV geometry while satisfying the pinned SOL kernel, so its gather/query overhead can erase the sparse benefit even when execution is correct.

Record cold/warm latency, end-to-end wall time, peak allocated/reserved VRAM, actual/forecast counts, sparse/eligible/warmup counts, fallback reasons, VDN-local SOL calls, VDN square row counts and history resets. Use warmed repetitions at identical shapes before making a speed claim. Compare current core BSA against external CuTe as separate policies.

Inspect video identity, reference fidelity, motion/grass artifacts, temporal stability and Continuum boundaries. Listen to the entire audio for wrong/missing words, unsolicited speech, whisper/volume fidelity, synchronization and discontinuities. Tensor metrics and unit tests cannot replace this assessment.