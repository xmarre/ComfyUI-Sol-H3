# Validation on the production SM120 GPU

GPU and audiovisual validation are outstanding. CPU tests cannot establish kernel correctness, performance, model equivalence or quality.

## Patch ordering for matched tests

For **Exact Runtime**, apply every compatible provider that may short-circuit the diffusion model or replace H3 blocks **before** Sol-H3 Exact. The important Spectrum order is:

`MODEL -> Spectrum -> other compatible MODEL patches -> Sol-H3 Exact Runtime -> sampler`

Spectrum forecast calls should then bypass Sol-H3; Spectrum actual calls should enter it. If Sol-H3 logs a bypass on a forecast call, the patch order is wrong.

For **external NVIDIA SOL**, do not stack Spectrum, ComfyUI Block Sparse Attention, VDN hybrid/window attention or another optimized-attention owner. Those combinations are intentionally rejected.

## Exact-kernel gate

Set the checkout root first:

```bash
COMFYUI_ROOT=/path/to/ComfyUI
```

In the Comfy conda environment:

```bash
cd "$COMFYUI_ROOT/custom_nodes/ComfyUI-Sol-H3"
python -m pip install -e '.[test]'
python -m pytest -q tests/test_gpu.py
python tools/gpu_probe.py --tokens 4096 --hidden 5376
```

Expected: all GPU cases pass, **zero skips**. Tests include BF16/FP16/FP32, strided AdaLN chunk views, indexed mask rows and irregular hidden width. They compare exact bitwise outputs; do not loosen the threshold to hide a regression.

The probe allocates synthetic activations directly on the GPU and loads no model weights. It reports bitwise parity, warmed operator CUDA timings and peak allocated/reserved VRAM. Both paths include the same input clone. Its timing does not establish full-model speedup. Repeat with the production row count.

Offline compilation is also covered by `tests/test_compile.py` when Triton 3.6 is installed. It compiles scalar/indexed variants for SM120 without requiring a GPU; passing that test is not execution validation. `N`, hidden-row stride and scalar modulation row are runtime, non-specialized Triton arguments so segment-length/layout changes do not create value-specialized variants.

## External NVIDIA SOL gate

Install the pinned optional kernel and launch Comfy with its `PYTHONPATH` as described in README. The package root is located without importing `sol_attn`; SHA-256 verification of every pinned package file must complete before package code is imported. The first real QKV shape must then pass the all-selected arithmetic gate. The end-of-sampling `Sol-H3` JSON log must show `success=true`, `backend=sol`, `approximate=true`, and `sparse_calls>0`. `dense_calls` counts configured dense warmup/layers, not sparse execution.

Failed/gated requests are not timing samples. Initial source verification, correctness probes and kernel compilation affect cold latency and must be reported separately from measured warmed runs.

## Matched matrix

Preserve model file/hash, quantization, LoRAs/strengths, prompt, dialogue, seed, references/order, sampler, complete sigma tensor, resolution, duration, audio conditioning and all other patches. Save the actual workflow for every run.

| Run | Configuration | Acceptance |
|---|---|---|
| A | Native dense | Baseline |
| B | Sol-H3 Exact | Packed hidden/output latent parity against A, then media |
| C | Native + Spectrum | Separate forecasting baseline; record actual/forecast counts |
| D | Spectrum -> Sol-H3 Exact | Compare against C with identical Spectrum schedule/counts |
| E-core | ComfyUI Block Sparse Attention | Native sparse baseline; record `comfy-kitchen` version and every node setting |
| E | External NVIDIA CuTe SOL, exact fusion off | Reference-policy sparse path; dialogue/video acceptance |
| E2 | External NVIDIA CuTe SOL + exact fusion | Compare with E; isolate affine interaction |
| F | SOL-BSA | Unavailable; do not label any dense run F |
| G | External SOL + Spectrum | Gated pending backend-history integration and sparse-quality acceptance |
| H | Few-step adapter | Separate comparison; record adapter and changed sigmas |

For **E-core**, use the current upstream `Block Sparse Attention` node deliberately rather than trying to force it to masquerade as E. Its implementation differs from NVIDIA's released H3 policy. At minimum record selection mode, tau/keep-percent, start/end percent, dense blocks, `min_tokens`, `extra_tokens`, and `sink_conditioning`. A useful quality-oriented starting point is Sol-Attn with `tau=1.0`, `sink_conditioning=exact_kv_and_rows`, and dense blocks `0,1`, but this still does **not** reproduce NVIDIA's one-dense-evaluation schedule or full-prefix dense-query recomputation. Treat it as a separate backend.

Do **not** combine E-core with Spectrum in the current validation matrix. The audited Spectrum consumer does not track the core sparse schedule's numerical-backend transitions, so such a run would mix incompatible actual-history evidence rather than establish a valid Spectrum comparison.

Run T2VA, I2V/FL2VA, Ref2VA with seven references, explicit dialogue, whispered dialogue, no dialogue, motion/grass, variable masks, Continuum boundaries and progressive handoff. Change one factor per comparison. Test normal BF16 and the production INT8/ConvRot model separately. Test model offload/reload and LoRA strength changes between runs to expose stale ownership.

Record cold and warm transformer CUDA time, end-to-end wall time, peak allocated and reserved VRAM, backend fingerprint, sparse/dense calls and Spectrum actual/forecast counts. Synchronize CUDA before/after timed regions. Reuse the same warmed shape and run three measured repetitions; report medians and range. The package does not add default CUDA synchronizations or full-sequence copies solely to collect timings. Use existing runtime metrics/profiling tools.

For exact mode, compare packed hidden states and output latents before decoding; bitwise affine parity alone does not prove integration parity. In the Spectrum combination, verify that Sol-H3 `actual_evaluations` tracks Spectrum actual transformer calls rather than logical calls.

For sparse paths, compare decoded video temporal stability, identity, reference fidelity and artifacts. Listen to the entire audio for missing/wrong words, unprompted chatter, volume/whispering, sync and boundary discontinuity. LPIPS or tensor error alone is insufficient. Keep failed dialogue runs in the results even if video looks good.

The most informative sparse comparison is **E-core vs E** at the same seed/model/output geometry. If the built-in Comfy Kitchen path is equal or better in both quality and speed, the external CuTe dependency should not be promoted merely because it comes from the newer Sol-H3 runtime.
