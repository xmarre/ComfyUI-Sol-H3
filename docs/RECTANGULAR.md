# Rectangular Sana SM120 attention

The packaged kernel remains Sana `sol-engine` revision
`2936c47637380842aaa4a4488fac5006cc542b70`, with a recorded functional patch.
Q is `[B,Tq,H,128]`; K/V are `[B,Tkv,H,128]`. Rectangular public calls require
`cute_sm120`. Existing square paths on other architectures retain their behavior.

| Geometry | Owner |
|---|---|
| Query launch grid, Q pooling/tail divisor, thresholds, output and LSE | Q tokens / ceil(Tq/64) |
| K centroids, V sums, centroid statistics, route groups and approximate masses | KV tokens / ceil(Tkv/64) |
| Exact sink interval and outward-rounded sink blocks | KV rows |
| CuTe compile descriptors and arithmetic verification cache | Both Q and KV shapes |

The SM120 query-tail length derives from Q. K/V route traversal, last-block
masses and exact-score masks retain KV geometry. Output stores use Q descriptors;
LSE stores mask the actual Q tail. Preprocessing remains Sana's existing Triton
preprocessing feeding the real CuTe attention mainloop. No alternative attention
implementation is substituted.

VDN v2 passes actual requested Q directly. K/V values, order and membership stay
unchanged. `sink_rows` is the exact global-prefix K/V count, including counts larger
than Tq. A partial sink block is rounded outward for exact computation, without
adding any K/V rows. VDN keeps global/anchor/Flex attention, learned gating and
linear complement. Its existing `square_q` and `query_positions` arguments remain
accepted but unused. VDN #11 still constructs them; eliminating that upstream
allocation is outside this Sol-H3-only change.

Untwist runs before gathering over original packed coordinates. The native packed
shape guard, external/mixed fallback and later eligible SOL resumption are retained.
Exact Runtime and inherited Sage behavior are unchanged. The kernel contract identity
is bumped to `sana-sol-engine-sol-attn-64-rect-sm120-v2`; it participates in the
configuration fingerprint used by Spectrum history. Existing receipt ownership is
retained.

## Approximation and correctness boundary

The requested rows are now pooled into their own 64-row Q tiles. Thresholds use
those Q centroids against all restricted K centroids; the existing local-diagonal
exact heuristic uses Q-tile and KV-block ordinals. Repacking changes both query
centroids and their ordinal alignment relative to square expansion. Sparse output
is therefore not promised to equal square-expanded output. This needs decoded-media
validation. It does not change the set of keys available to any local operation.

All-selected arithmetic uses `sink_tokens=Tkv` and independent rectangular BF16 SDPA.
The aggregate gate is unchanged: finite output, mean absolute error <= 0.002,
relative L2 <= 0.005, plus the existing scale-aware catastrophic-peak guard.
Changing only Tkv triggers fresh calibration. Max error remains telemetry.

## Provenance

`sol_manifest.json` retains every original upstream SHA-256. Packaged hashes and
modification records cover three functionally changed files: `interface.py`,
`preprocess.py`, `sm120/mainloop.py`. Their headers identify the changes.
`tools/rectangular_sm120.patch` applies after import adaptation and is hash-recorded.
`python tools/vendor_sol_attn.py /path/to/pinned/Sana` reproduces the modified tree.
CI compares that reproduction against the packaged bytes. The original
`SANA_SOURCE_SNAPSHOT.json` remains an upstream historical record.

## WSL GPU validation

In the ComfyUI environment, with VDN #8 followed by #11 installed:

```bash
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
python -m pytest -q tests/test_rectangular.py -m gpu
git fetch origin c543f4f017c0ddb276ff28148a7e9be291057b75
git worktree add --detach /tmp/sol-h3-square-c543f4f c543f4f017c0ddb276ff28148a7e9be291057b75
python tools/rectangular_probe.py \
  --baseline /tmp/sol-h3-square-c543f4f \
  --comfy-root /home/toor/ComfyUI \
  --vdn-path /home/toor/ComfyUI/custom_nodes/comfyui-vdn-h3-plus \
  --warmup 5 --repeats 20 > /tmp/sol-rectangular-probe.json
```

The probe requires real `cute_sm120`, verifies the old baseline checkout and both
packaged trees, checks all-selected arithmetic on identical Q/K/V, exercises the
installed VDN grouped dispatcher, then alternates warmed A/B execution order.
Both paths include their preprocessing; the baseline includes output gathering.
Input tensors are preallocated. CUDA-event and synchronized wall-time samples are
reported separately. First-use compilation/autotuning is excluded. This measures
the attention API, not model or end-to-end speed. Adjust `--frame-rows` and `--heads`
to measured production geometry; defaults are a small synthetic routing test.

Run the normal full workflow after restarting ComfyUI. Save its log, then add
`--runtime-log /path/to/comfy.log` to the same probe command to check production
stage telemetry. Eligible native stages must report:

```text
sol_backend = cute_sm120
sol_source_tree_verified = true
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls = 0
```

VDN row counters describe successful production sparse subcalls and exclude the
one-time per-geometry arithmetic calibration. Mixed/external stages may have zero
SOL calls; later native stages must resume. Preserve seeds, resolution, actual/
forecast schedule, tau, reference inputs and all other patches for full-model A/B.
Compare decoded audio/video too: sparse query grouping changed.

## Evidence

Focused CPU contracts: 40 passed, 5 GPU cases skipped. No rectangular GPU execution,
decoded-media comparison or warmed performance result is available in this environment.
CI and final aggregate validation will be recorded in the PR description.
