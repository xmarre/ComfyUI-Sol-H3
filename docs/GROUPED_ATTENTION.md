# Grouped Sol/VDN producer

VDN owns the learned linear complement and its raw pre-normalization, pre-RoPE
Q/K/V. Sol replaces only the restricted-domain softmax provider. The bounded
4096-row QKV producer, learned weights, text state, temporal scans, LoRAs, gates,
conditioning queries, anchors, protected-prefix rules and mapped-neighbor selector
are unchanged by the grouped producer.

## Execution

VDN initializes the conditioning prefix in one scratch pair, then calls the
provider's optional `grouped_kv_v1` capability for each local group. The provider
executes the attention consumer synchronously before returning.

On eligible SM120/SM121 BF16 inputs, a single Triton pass reads the window's
original K/V rows, writes their existing restricted-domain scratch, and computes
the same M64 K centroid and V sum as Sol's upstream reducer. It reads the already
initialized prefix without rewriting it. Arbitrary ordered row indices, partial
last tiles and tiles crossing an unaligned prefix/window boundary are supported.
The computation removes two independent gather launches and the subsequent full
K/V summary-read pass. It retains the existing bounded K/V scratch; it does not
materialize all windows together.

Prepared summaries belong to one synchronous group invocation and its CUDA
stream. A context token restores an enclosing invocation on success or failure;
recursive reuse of active scratch is rejected before writing. Request and forward
ownership, views, mutation versions where available, and stream identity are
checked before consuming summaries. No K/V or summary tensor is cached across
blocks, evaluations or requests.

Native and Flow attention reuse those summaries. Flow also reuses the identical
threshold and summaries between its all-selected arithmetic check and production
call, and drops qualification outputs before allocating the production output.
The existing CuTe mainloop, sparse threshold policy, BF16/FP32 precision and
query-origin-v5 launch ABI are unchanged. Forced-dense requests, warmup, older VDN
and other providers preserve their existing routes. Compilation and execution
failures propagate; they do not trigger an unreported dense retry.

The sampling receipt reports `grouped_kv_fused_calls` and
`grouped_kv_fused_rows`. These count actual fused producers, not VDN learned-linear
participation; retain the separate VDN execution counters when auditing a run.

## Validation and deployment

Use Sol PR #41 with VDN PR #42. VDN PR #41 remains a separate preceding overlay
for its scratch-admission policy. Flow PR #107 continues to accept the unchanged
query-origin-v5 ABI. No Core ownership guard is required.

CPU tests execute the Triton gather in the interpreter against the upstream
reducer in FP32, test nested/failure cleanup and compile the BF16 kernel to SM120.
The interpreter does not establish BF16 GPU parity or throughput.

After applying the PR overlays through ComfyUI Patcher, run from the Sol node's
folder in the existing ComfyUI Python environment:

```bash
python tools/grouped_attention_acceptance.py --rounds 5 --output grouped_attention_acceptance.json
```

The script requires SM120/SM121. It saves a JSON report even on failure. It checks
all gathered K/V elements, all BF16 summaries and all sparse output elements for
bitwise equality against separate gather/preparation, including rectangular,
mapped, biased, unaligned and interleaved layouts. It additionally compares
all-selected output with independent SDPA. Long synthetic domains use 56 heads,
128 channels and source lengths 54,744 and 107,630; these are representative shapes,
not captured model activations. Both execution orders compare complete group
CUDA/wall latency, preparation cost, prepared attention cost, peak allocated and
reserved bytes, free device memory and repeated-call allocation cleanup. A >5%
CUDA-time regression in either order fails component acceptance. Reserved memory
may remain in the allocator pool; that is distinct from live retained allocation.
The script does not rerun the already-qualified QKV projection benchmark.

A component pass does not certify the full-generation speed or quality target.
Full acceptance requires matched native and two-chunk Flow runs, all required VDN
learned-linear calls, actual sparse/dense receipts, fixed real H3 evaluations,
per-stage sampler wall time, peak/live allocation across repeated generations and
decoded audio/video continuity. Keep `VDN_H3_PERF_PROFILE=0` for timing. Hold prompt,
references, model, adapters, LoRAs, seed, grid, frames, schedule, Spectrum, tau,
sinks, precision, power limit and residency state constant. A Core sparse timing
is comparable only when its VDN execution counts and semantics also match.

No end-to-end speedup, Core performance parity, BF16 GPU qualification or
production VRAM improvement follows from CPU tests or offline compilation. Keep
these changes draft until the real-GPU gates pass.
