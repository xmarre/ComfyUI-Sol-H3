# Validation status and RTX PRO 6000 commands

**Production RTX PRO 6000 evidence already exists for Exact Runtime, including a matched Exact-off/on timing A/B.** The current interoperability head still needs fresh GPU acceptance for sparse SOL and the companion-provider combinations, so PR #1 and companions remain draft. CPU structural correctness, exact-runtime GPU evidence, sparse-kernel execution and decoded-media quality are different evidence.

## Completed evidence

- Original SOL baseline: 45 passed, 19 skipped before Triton installation.
- SOL final local suite: 66 passed, 18 GPU skips, including offline SM120 affine compilation, real Comfy/KJ wrappers and integrated stack tests.
- Additional integrated native H3 + Spectrum + SOL + optional VDN tests: 8 passed, covering both wrapper orders, repeated scopes, full coverage, grouped windows and Flex-to-grouped fallback. External CuTe is replaced by CPU SDPA; VDN linear work is explicitly disabled in this test.
- Spectrum: 911 passed, 13 skipped against current Comfy. Native fixture updates accommodate core's optional attention argument and native options augmentation.
- VDN: 133 passed, 12 skipped. Restricted-domain provider tests preserve exact native results; absent official/GPU prerequisites remain skips.
- Untwist: 40 passed.

Environment for the CPU suites: Python 3.12, torch 2.14.0+cpu, Triton 3.6.0, comfy-kitchen 0.2.33. The original CI also covers torch 2.10 CPU.

### Completed production-GPU Exact Runtime A/B

A matched hot A/B was run on the RTX PRO 6000 production workflow with the exact/runtime tier only; approximate SOL attention was not enabled. The stack included the production INT8/ConvRot H3 path with VDN, DiffAid, Untwist RoPE, Spectrum, progressive handoff and Continuum.

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | 247.30 s | 298.69 s | 17.18 GB |

Observed improvement: **6.17% lower sampler time** (1.066x) and **4.44% lower end-to-end time** (1.046x), with unchanged measured peak VRAM.

Exact-mode telemetry from the production runs reported `success=true`, `backend=inherit`, `approximate=false`, `sparse_calls=0`, and the expected fused-block accounting (for example 3 actual H3 evaluations -> 150 exact blocks on the 50-block model). This is real GPU/runtime evidence for Exact Runtime and its integration with that workflow stack. It is **not** GPU validation of external sparse SOL, Sage+SOL, VDN-local SOL, Spectrum backend-transition handling, or the final post-interoperability PR head; those still require the matrix below. No separate decoded-media acceptance result is claimed from this timing pair.

To reproduce CPU contracts from the Sol-H3 checkout:

```bash
COMFYUI_PATH=/path/to/ComfyUI \
KJNODES_PATH=/path/to/ComfyUI-KJNodes \
SPECTRUM_PATH=/path/to/ComfyUI-Spectrum-MiniMax-H3 \
VDN_PATH=/path/to/ComfyUI-VDN-H3-Plus \
python -m pytest -q
```

Use the linked companion branches. CPU tests replace unavailable Sage/SOL kernels explicitly; they do not secretly run CUDA. The real KJ get_sage_func source and Comfy wrap_attn behavior are exercised separately from the substituted kernel.

## Production machine: kernel checks

In your WSL shell, using the documented pinned Sana checkout:

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

Sage/Sage3 commands require their respective installed extensions. Expected: GPU tests have zero skips; affine probe reports bitwise parity; each attention probe reports `sparse_calls=2`, successful independent arithmetic gates and exact prefix parity against the selected provider. The probe intentionally asserts sparse execution because it is a kernel test; normal sampling has no fatal zero-sparse requirement. Do not relax arithmetic tolerances to hide failures.

Repeat attention probes at representative packed row counts and heads after the small probe passes. The probe uses synthetic tensors, no model weights, and does not establish model speedup. The first shape includes verification/compile cost; record cold and warmed costs separately.

Launch the full stack with the same environment:

```bash
cd "$COMFYUI_ROOT"
python main.py 2>&1 | tee /home/toor/sol_h3_full_stack.log
```

Archived production logs already establish Exact Runtime execution. For the remaining sparse/interoperability comparisons, use the original saved workflow and preserve its settings; save the complete current-head log with each result.

## Full-stack matrix

Use the companion PRs linked in README. Start with Exact off to isolate sparse interoperability, then enable it and compare again.

| Run | Combination | Required observation |
|---|---|---|
| A | Native selected dense provider | Baseline latency and media |
| B | Sage + SOL | Dense inherited backend recorded; sparse calls after warmup |
| C | Sage3 + SOL | Same routing checks; independent gate passes |
| D | Spectrum + SOL, then Sage + Spectrum + SOL | SOL actual count agrees with actual transformer calls; history resets at backend transitions |
| E | VDN grouped + SOL, with and without Sage | Gate/linear semantics and native rectangular window fallbacks preserved |
| F | VDN flex + SOL | Native Flex mask preserved; grouped fallback reason if used |
| G | VDN full coverage + SOL | Eligible softmax can use SOL; learned gate remains active |
| H | VDN + Spectrum + SOL | Backend receipts cover multi-call VDN; no cross-backend anchors |
| I | Flow API-1 reduced / API-2 mixed, then normal target grid | Native fallback on unsupported calls; later eligibility can resume SOL |
| J | Repeated Continuum chunks | New sampling scopes, correct warmup, no stale state or boundary errors |
| K | Core Block Sparse Attention + Spectrum, with/without SOL | Current core provider is actual-only under the companion history consumer; explicit ownership recorded |
| L | DiffAid + Untwist + runtime DoRA/LoRA + KJ preview + production stack | Projection/key transforms preserved; any native/actual-only fallback explicitly identified |

Also exercise Exact → SOL, SOL exact=true → Exact, SOL exact=false → Exact, repeated identical applications, independently cloned branches, all-warmup requests, and all-fallback requests. All legitimate zero-sparse runs must finish. They are not evidence of SOL speedup.

Use ordinary LoRA and runtime DoRA/LoRA separately; change strengths between runs and test offload/reload. Test pruned/curve AdaLN and the actual INT8/ConvRot model separately from BF16.

## Performance and media acceptance

Preserve exact model/adapter files and hashes, quantization, seed, full sigma schedule, sampler, prompt/dialogue, reference order, dimensions, duration, audio conditioning and patch settings. Save each workflow and output metadata. Progressive lower-resolution stages change composition; matching only the seed does not make them composition-matched A/B samples.

Record cold/warm latency, end-to-end wall time, peak allocated/reserved VRAM, actual/forecast counts, sparse/eligible/warmup counts, fallback reasons, VDN-local SOL calls and history resets. Run three warmed repetitions at the same shape. Compare current core BSA against external CuTe as separate policies.

Inspect video identity, reference fidelity, motion/grass artifacts, temporal stability and Continuum boundaries. Listen to the entire audio for wrong/missing words, unsolicited speech, whisper/volume fidelity, synchronization and discontinuities. Tensor metrics and unit tests cannot replace this assessment.
