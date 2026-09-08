# Rectangular Sana SM120 attention

The packaged kernel remains Sana `sol-engine` revision `2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, with the rectangular SM120 functional patch recorded in `tools/rectangular_sm120.patch`.

The public SM120 path accepts independent query and key/value lengths:

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
Q         = requested local query rows
K/V       = VDN's already-restricted global + permitted-window rows
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

The legacy provider API v2 remains available only for square-QKV-only providers. Its `square_q` and `query_positions` compatibility payload is constructed lazily. Because current Sol-H3 publishes v3, production Sol-H3 does not trigger that square-Q gather/allocation.

Expected v3 telemetry is:

```text
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls == 0
```

Production RTX PRO 6000 runs now confirm API v3 is active (`softmax-provider module API=3`) and native grouped stages satisfy those invariants. The direct rectangular route is therefore not only synthetic/CI coverage.

## Flow mixed-grid external sequence

Flow's API-2 mixed-grid contract is a different path from VDN grouped-local provider dispatch. In the mixed transformer sequence, VDN deliberately switches to `dense_gate_no_linear`: the gate remains active and the geometry-dependent linear complement is disabled.

Sol-H3 recognizes only the explicit coherent contract:

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

The production stack has now exercised this route:

```text
sol_eligible_calls               250
external_mixed_sol_calls         192
dense_warmup                      58
external_mixed_q_rows       8,360,640
external_mixed_kernel_q_rows 8,360,640
compatibility_fallbacks           {}
rel_l2                    0.0009817115
```

The 58 dense calls are expected from one configured dense evaluation plus two configured dense layers. The remaining eligible calls used packaged `cute_sm120`, and later native target-grid stages resumed rectangular SOL. This establishes production routing/arithmetic for the mixed sequence, not an isolated speedup.

## Untwist and preprocessing

VDN's explicit full-domain preprocessing hook runs on the complete post-RoPE packed Q/K/V tensors before grouped-local gathering. This preserves original packed-row coordinates for transforms such as Untwist.

For ordinary model-level attention, Sol-H3 also preserves the inherited `attention_preprocess_v1` chain exactly once before SOL/dense ownership is chosen.

Shape, dtype, and device changes by a preprocessing contract remain invalid.

## Spectrum backend-history boundary

Spectrum #104 tracks generic backend policy identity and actual receipts. It does not infer Sol/VDN/Flow ownership itself. Sol-H3 therefore publishes a policy identity that includes the rectangular kernel contract and the route geometry, then emits actual route receipts.

Production also composes MiniMax-H3 Diff-Aid. Current Sol-H3 accepts only its audited activation-only DIT wrapper chain when Diff-Aid's own Spectrum runtime declaration is present. Both valid wrapper orders are supported. Static Diff-Aid configuration enters the policy identity, while changing normalized sigma remains owned by Spectrum's existing external-patch regime tracking. Unknown wrappers, cycles, duplicate/mismatched Sol patches, and missing Diff-Aid declarations remain opaque/actual-only.

This matters for performance interpretation: a previous hot SOL run executed all 18 logical calls as actual transformer NFEs, whereas a SOL-bypassed control executed 13 actual + 5 Spectrum forecasts. The five skipped transformer calls are roughly the same 80-90 s scale as the raw sampler-time delta. Route geometry and NFE schedule must therefore be matched before judging SOL cost.

## Approximation boundary

Rectangular repacking changes the 64-row query groups compared with the old square-expanded bridge. That changes query centroids and their ordinal alignment relative to Sana's local exact heuristic. Sparse output is therefore not promised to equal the historical square-expanded sparse output.

The K/V set available to each operation is unchanged. The approximation boundary is in query grouping/routing, not in VDN domain membership.

The mixed-grid route has an additional experimental boundary: the mixed video row sequence is explicitly non-uniform. Sol-H3 relies on the published Flow contract rather than inventing a native-grid lattice. The route has now executed successfully in production, but decoded-media and matched performance evaluation remain separate from routing correctness.

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

Real RTX PRO 6000 evidence exists for:

- packaged Sana/CuTe SM120 compile/execution;
- rectangular GPU arithmetic/sparse-sink tests;
- production native-grid VDN provider-v3 rectangular routing with 1:1 requested/kernel Q rows and zero square expansion;
- production Flow mixed-grid SOL routing with 192 external calls and exact 1:1 mixed requested/kernel Q rows;
- later mixed -> native rectangular SOL resumption;
- Exact Runtime matched production timing.

The old square bridge expanded Q work by roughly 4.4-5.4x in the cited native VDN stages. Rectangular Sol-H3 removes that kernel-Q expansion. API v3 additionally avoids constructing the obsolete v2 square-Q/query-position compatibility payload. The available end-to-end runs are not a clean measurement of the v3 allocation saving because compile/cache state and Spectrum NFE schedules differ.

## Performance claims

Do not infer speedup solely from successful sparse execution.

The most recent timing comparison is explicitly confounded:

```text
SOL hot:      18 logical, 18 actual, 0 forecasts, sampler 320.98 s
SOL bypassed: 18 logical, 13 actual, 5 forecasts, sampler 240.55 s
```

Before comparing wall time, first record:

```text
sampler_logical_calls
transformer_actual_nfe
spectrum_forecast_calls
```

Do not require exact `13 actual + 5 forecast`; backend transitions can legitimately force extra actual anchors. Once schedules are comparable, normalize target-grid/mixed-grid time by actual transformer NFEs.

For a meaningful A/B, preserve model/adapters, quantization, seed, sigma schedule, sampler, prompt/dialogue, references, resolution, duration, VDN settings, Spectrum forecast schedule and all other patches. Warm both paths and record:

- CUDA-event kernel time;
- synchronized wall time;
- end-to-end sampler time;
- peak allocated/reserved VRAM;
- requested vs kernel Q rows;
- sparse/eligible/warmup/fallback counts;
- actual/forecast counts;
- decoded video and audio quality.

A credible later optimization target is the native SOL adapter's Q/K/V `transpose(...).contiguous()` materialization. The packaged Sana interface can accept suitable innermost-contiguous strided packed views, so avoiding those copies may matter at long sequence lengths. That optimization should follow, not precede, the corrected schedule-matched production rerun.
