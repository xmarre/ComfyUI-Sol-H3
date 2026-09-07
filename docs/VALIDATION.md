# Validation on the production SM120 GPU

GPU and audiovisual validation are outstanding. CPU tests cannot establish
kernel correctness, performance, model equivalence or quality.

## Kernel gate

In the Comfy conda environment:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
python -m pip install -e '.[test]'
python -m pytest -q tests/test_gpu.py
```

Expected: all GPU cases pass, **zero skips**. Tests include BF16/FP16/FP32,
strided AdaLN chunk views, indexed mask rows and irregular hidden width. They
compare exact bitwise outputs; do not loosen the threshold to hide a regression.

For SOL, install the pinned optional kernel and launch Comfy with its PYTHONPATH
as described in README. First real QKV per shape must pass the all-selected
arithmetic gate. The end-of-sampling `Sol-H3` JSON log must show `success=true`,
`backend=sol`, `approximate=true`, and `sparse_calls>0`. `dense_calls` includes
configured warmup/layers, not sparse execution. Failed/gated requests are not
timing samples. Initial correctness probes and kernel compilation affect cold
latency and must be reported separately from measured warmed runs.

## Matched matrix

Preserve model file/hash, quantization, LoRAs/strengths, prompt, dialogue,
seed, references/order, sampler, complete sigma tensor, resolution, duration,
audio conditioning and all other patches. Save the actual workflow for each run.

| Run | Configuration | Acceptance |
|---|---|---|
| A | Native dense | Baseline |
| B | Exact Sol-H3 | Packed hidden/output latent parity against A, then media |
| C | Native + Spectrum | Separate forecasting baseline; actual/forecast counts |
| D | Exact + Spectrum | Compare against C with identical schedule/counts |
| E | SOL, exact fusion off | Sparse alone; dialogue and video acceptance |
| E2 | SOL + exact fusion | Compare with E; isolate interaction |
| F | SOL-BSA | Unavailable; do not label any dense run F |
| G | SOL + Spectrum | Gated pending backend-history integration and E acceptance |
| H | Few-step adapter | Separate comparison; record adapter and changed sigmas |

Run T2VA, I2V/FL2VA, Ref2VA with seven references, explicit dialogue,
whispered dialogue, no dialogue, motion/grass, variable masks, Continuum
boundaries, and progressive handoff. Change one factor per comparison. Test
normal BF16 and the production INT8/ConvRot model separately. Test model
offload/reload and LoRA strength changes between runs to expose stale ownership.

Record cold and warm transformer CUDA time, end-to-end wall time, peak allocated
and reserved VRAM, backend fingerprint, sparse/dense calls, and Spectrum
actual/forecast counts. Synchronize CUDA before/after timed regions. Reuse the
same warmed shape and run three measured repetitions; report medians and range.
The package does not add default CUDA synchronizations or full-sequence copies
solely to collect timings. Use existing runtime metrics/profiling tools.

For exact mode compare packed hidden states and output latents before decoding;
bitwise affine parity alone does not prove integration parity. Compare decoded
video temporal stability, identity, reference fidelity and artifacts. Listen to
the entire audio for missing/wrong words, unprompted chatter, volume/whispering,
sync and boundary discontinuity. LPIPS or tensor error alone is insufficient.
Keep failed dialogue runs in the results even if video looks good.
