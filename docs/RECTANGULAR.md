# Rectangular Sana SM120 attention

The packaged kernel remains Sana `sol-engine` revision `2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, with the rectangular SM120 functional patch recorded in `tools/rectangular_sm120.patch`.

The public SM120 path now accepts independent query and key/value lengths:

```text
Q   [B, Tq,  H, 128]
K/V [B, Tkv, H, 128]
```

`cute_sm120` remains the execution backend. Sana's existing preprocessing still feeds the CuTe mainloop; no replacement attention implementation is substituted.

## Geometry ownership

| Geometry | Owner |
|---|---|
| Query launch grid, Q pooling/tail divisor, thresholds, output and LSE | Q tokens / ceil(Tq / 64) |
| K centroids, V sums, centroid statistics, route groups and approximate masses | KV tokens / ceil(Tkv / 64) |
| Exact sink interval and outward-rounded sink blocks | KV rows |
| CuTe compile descriptors and arithmetic verification cache | Both Q and KV shapes |

The SM120 query-tail length derives from Q. K/V route traversal and exact-score masking retain K/V geometry. Output/LSE stores use Q descriptors and mask the actual Q tail.

The kernel contract identity is `sana-sol-engine-sol-attn-64-rect-sm120-v2`; this identity is part of the Spectrum numerical-history fingerprint.

## VDN provider API v3

The current companion [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11) publishes provider API v3.

For a grouped local operation, VDN first constructs exactly the same restricted K/V domain it would use for its own grouped SDPA. It then passes only the originally requested Q rows plus that unchanged restricted K/V domain and `sink_rows`:

```text
Q        = requested local query rows
K/V      = VDN's already-restricted global + permitted-window rows
sink_rows = leading global/prefix K/V rows
```

Sol-H3 evaluates those requested rows directly. It does not broaden the K/V set, reorder K/V, or inject rows from outside VDN's selected domain.

VDN still owns:

- local/window membership;
- global and anchor operations;
- Flex behavior;
- the learned softmax gate;
- the learned linear complement;
- output projection and all other VDN model math.

The legacy provider API v2 remains available only for square-QKV-only providers. Its `square_q` and `query_positions` compatibility payload is now constructed lazily. Because current Sol-H3 publishes v3, production Sol-H3 no longer triggers that square-Q gather/allocation.

Expected v3 telemetry is therefore:

```text
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls == 0
```

## Flow mixed-grid external sequence

Flow's API-2 mixed-grid contract is a different path from VDN grouped-local provider dispatch. In the mixed transformer sequence, VDN deliberately switches to `dense_gate_no_linear`: the gate remains active and the geometry-dependent linear complement is disabled.

Sol-H3 now recognizes only the explicit coherent contract:

```text
api = 2
mode = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

and validates:

- the published native carrier row count;
- the actual mixed sequence row count;
- `video_start`;
- temporal length and protected-prefix frame count;
- source rows per frame;
- target-prefix rows per frame;
- the current packed layout prefix.

If these invariants match, the underlying whole-sequence attention call can use rectangular SOL. The exact sink is the validated leading packed non-video/global prefix (`video_start`). Sol-H3 does not pretend that the mixed video rows form a uniform native target grid.

Successful execution is recorded as:

```text
route = sol_external_mixed
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
```

Malformed, stale, unknown, or differently owned external contracts remain inherited/dense for that call. Later normal native-grid calls can still resume SOL.

This mixed-grid route is currently covered by CPU/native integration tests, including mixed -> native resumption. It has not yet received a production GPU/media rerun. The prior live progressive run predates this route and therefore correctly reported the mixed stage as dense.

## Untwist and preprocessing

VDN's explicit full-domain preprocessing hook runs on the complete post-RoPE packed Q/K/V tensors before grouped-local gathering. This preserves original packed-row coordinates for transforms such as Untwist.

For ordinary model-level attention, Sol-H3 also preserves the inherited `attention_preprocess_v1` chain exactly once before SOL/dense ownership is chosen.

Shape, dtype, and device changes by a preprocessing contract remain invalid.

## Approximation boundary

Rectangular repacking changes the 64-row query groups compared with the old square-expanded bridge. That changes query centroids and their ordinal alignment relative to Sana's local exact heuristic. Sparse output is therefore not promised to equal the historical square-expanded sparse output.

The K/V set available to each operation is unchanged. The approximation boundary is in query grouping/routing, not in VDN domain membership.

The newly enabled mixed-grid route has an additional experimental boundary: the mixed video row sequence is explicitly non-uniform. Sol-H3 relies on the published Flow contract rather than inventing a native-grid lattice. Its sparse quality therefore requires decoded-media validation even though the packed attention semantics are coherent.

## Arithmetic gate

The all-selected calibration uses `sink_tokens=Tkv` and an independent rectangular BF16 SDPA reference. The aggregate gate remains:

- finite output;
- mean absolute error <= `0.002`;
- relative L2 <= `0.005`;
- existing scale-aware catastrophic-peak guard.

Changing either Tq or Tkv creates a new calibration geometry. A failed arithmetic gate is fatal and is never silently accepted as an approximate result.

## Provenance

`sol_manifest.json` retains the original SHA-256 for every upstream source file and separate packaged hashes. Functional modifications beyond import adaptation are restricted to:

- `interface.py`;
- `preprocess.py`;
- `sm120/mainloop.py`.

Their headers identify the changes. `tools/rectangular_sm120.patch` reproduces the functional diff after import adaptation. `python tools/vendor_sol_attn.py /path/to/pinned/Sana` reproduces the packaged tree for CI/source-audit purposes; it is not an installation requirement.

The original `SANA_SOURCE_SNAPSHOT.json` remains the historical upstream snapshot and is not rewritten to pretend upstream validated SM120.

## Validation from the current checkout

Do not change Git state for normal validation. In the active ComfyUI environment:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q tests/test_rectangular.py -m gpu
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512 --heads 8
```

The direct attention probe must report the packaged Sana source and `cute_sm120`; GPU tests must execute the real kernel rather than a CPU substitute.

For normal ComfyUI production validation, restart ComfyUI and run the workflow without changing seed/model/reference/order/sampler settings. Inspect the Sol-H3 telemetry.

### Native grouped VDN success

```text
sol_backend = cute_sm120
sol_source_tree_verified = true
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls = 0
```

### Mixed-grid success

For a workflow that actually enters Flow API-2 `mixed_grid_low_suffix`:

```text
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
```

A malformed/unsupported mixed contract may instead report an explicit `external_sequence_*` fallback. That is a local compatibility fallback, not a reason to hard-fail the entire generation.

## Existing GPU evidence

Real RTX PRO 6000 evidence already exists for:

- packaged Sana/CuTe SM120 compile/execution;
- rectangular GPU arithmetic/sparse-sink tests;
- production native-grid VDN rectangular routing with 1:1 requested/kernel Q rows and zero Sol-H3 square expansion;
- Exact Runtime matched production timing.

The prior production rectangular run used the v2 provider contract and therefore still paid VDN's upstream `square_q` gather even though Sol-H3 itself ignored those extra query values. Current API v3 removes that allocation upstream. GPU timing of that v3 allocation removal is still outstanding.

The new external mixed-grid SOL route is structurally/CI validated but not yet production-GPU/media validated.

## Performance claims

Do not infer speedup solely from successful sparse execution.

For a meaningful A/B, preserve model/adapters, quantization, seed, sigma schedule, sampler, prompt/dialogue, references, resolution, duration, VDN settings, Spectrum forecast schedule and all other patches. Warm both paths and record:

- CUDA-event kernel time;
- synchronized wall time;
- end-to-end sampler time;
- peak allocated/reserved VRAM;
- requested vs kernel Q rows;
- sparse/eligible/warmup/fallback counts;
- actual/forecast counts;
- decoded video and audio quality.

The old v2 square bridge expanded query work by roughly 4.4-5.4x in the cited production stages. The rectangular kernel removed that Sol-H3 compute expansion. API v3 additionally removes VDN's now-unused square-Q gather/allocation. The actual end-to-end benefit of that second optimization still needs a matched measurement.
