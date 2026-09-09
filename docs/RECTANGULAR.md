# Rectangular Sana SM120 attention

v0.1.0 packages Sana `sol-engine` revision `2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, with the rectangular SM120 functional patch recorded in `tools/rectangular_sm120.patch`.

The public SM120 path accepts independent query and key/value lengths:

```text
Q   [B, Tq,  H, 128]
K/V [B, Tkv, H, 128]
```

`cute_sm120` remains the execution backend. Sana preprocessing still feeds the CuTe mainloop; no replacement attention implementation is substituted.

## Geometry ownership

| Geometry | Owner |
|---|---|
| Query launch grid, Q pooling/tail divisor, thresholds, output and LSE | Q tokens / ceil(Tq / 64) |
| K centroids, V sums, centroid statistics, route groups and approximate masses | K/V tokens / ceil(Tkv / 64) |
| Exact sink interval and outward-rounded sink blocks | K/V rows |
| CuTe compile descriptors and arithmetic verification cache | Q/K/V shape and layout |

The SM120 query tail derives from Q. K/V route traversal and exact-score masking retain K/V geometry. Output/LSE stores use Q descriptors and mask the actual Q tail.

Kernel contract identity:

```text
sana-sol-engine-sol-attn-64-rect-sm120-v2
```

This identity participates in Spectrum numerical-history fingerprinting.

## VDN provider API v3

The reviewed companion [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11) publishes provider API v3.

For grouped local attention VDN first constructs the same restricted K/V domain it would use for grouped SDPA, then passes only the originally requested Q rows plus that unchanged restricted domain:

```text
Q         = requested local query rows
K/V       = VDN's already-restricted global + permitted-window rows
sink_rows = leading global/prefix K/V rows
```

Sol-H3 does not broaden, reorder or synthesize K/V membership.

VDN still owns:

- local/window membership;
- global and anchor operations;
- masked Flex behavior;
- learned softmax gate;
- learned linear complement;
- output projection and all other VDN model math.

Provider API v2 remains available for square-only providers. Its `square_q` and `query_positions` payload is constructed lazily only for a v2 consumer. Current Sol-H3 publishes v3, so production does not construct that obsolete square-Q compatibility payload.

Expected telemetry:

```text
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls == 0
```

Representative production stages satisfy those invariants:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The historical square bridge expanded Q work by roughly `4.4x–5.4x` in affected stages. The v3 rectangular path removes that kernel-row expansion.

## Flow mixed-grid external sequence

Flow's API-2 mixed-grid contract is a separate path from VDN grouped-local provider dispatch. In the mixed transformer sequence, VDN intentionally uses `dense_gate_no_linear`: its learned gate remains active while the geometry-dependent linear complement is disabled.

Sol-H3 recognizes only:

```text
api = 2
mode = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

and validates carrier rows, mixed packed rows, `video_start`, temporal length, protected-prefix frames, source rows/frame, target-prefix rows/frame and packed-layout prefix.

If those invariants agree, the whole-sequence attention call can use SOL. The exact sink is the validated leading packed non-video/global prefix. Sol-H3 does not pretend the mixed video suffix forms a uniform native target lattice.

Successful execution:

```text
route = sol_external_mixed
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
```

Representative final production route:

```text
external_mixed_sol_calls           144
external_mixed_q_rows         6,270,480
external_mixed_kernel_q_rows  6,270,480
compatibility_fallbacks              {}
```

An earlier run also established 192 mixed calls / 8,360,640 requested and kernel rows 1:1. Malformed/stale/unknown contracts delegate locally to inherited attention; later native target-grid calls can resume SOL.

### Experimental K/V attention-measure normalization

Flow PR #24 can publish a second, independent contract alongside API 2:

```text
key  = h3_flow_mixed_grid_attention_measure_v1
api  = 1
mode = prefix_kv_stratified_subsample
```

This contract addresses unequal discrete spatial sampling density inside the mixed attention domain; it does not redefine the mixed sequence or VDN ownership. The motivating geometry has `1064` protected-prefix rows/frame and `540` source-suffix rows/frame (`~1.97037x`).

Sol-H3 validates the measure metadata against the current API-2 mixed contract and full Q/K/V row counts. Explicit preprocessing such as Untwist is applied to the original full mixed domain first. Then only K/V are gathered:

- every Q row is preserved;
- every row before target video is preserved;
- every generated source-grid suffix K/V row is preserved exactly and in order;
- each protected prefix frame keeps one K/V representative nearest every source-grid coordinate under MiniMax-H3's native area-normalized `_frame_grid` geometry.

For the matched production geometry:

```text
Q:   56029 -> 56029
K/V: 56029 -> 49741
```

`49741` equals the native low-carrier packed row count. The SM120 call is therefore rectangular in the ordinary kernel sense: query ownership remains the full mixed sequence while the softmax K/V integration measure is normalized to source-grid spatial density.

The representative mapping is regression-tested against pinned native ComfyUI `_frame_grid` coordinates row-for-row. Contract mismatches fail closed rather than silently selecting a different K/V domain. The experiment remains off by default in Flow and requires decoded-media validation before promotion.

Expected telemetry when active:

```text
external_mixed_measure_calls > 0
external_mixed_measure_q_rows > 0
external_mixed_measure_kv_rows_before > external_mixed_measure_kv_rows_after
external_mixed_measure_removed_rows > 0
```

## Untwist preprocessing

VDN's full-domain preprocessing hook runs on complete post-RoPE packed Q/K/V before grouped gathering. This preserves packed-row coordinates for transforms such as Untwist.

For ordinary model-level attention Sol-H3 also consumes the inherited `attention_preprocess_v1` chain exactly once before sparse/dense ownership is chosen. Shape, dtype or device changes from a preprocessing contract remain invalid.

## Spectrum backend-history boundary

Spectrum #104 tracks generic backend policy identity and actual receipts. It does not infer Sol/VDN/Flow ownership itself.

Sol-H3 therefore publishes a policy identity covering the rectangular kernel contract, current route geometry and audited replacement chain. Production history validation now includes:

- Diff-Aid activation-only wrappers only when Diff-Aid publishes its runtime declaration;
- Flow's marked layout wrapper only when marker, closure, block, scope and previous-link invariants match;
- Flow's mixed-grid wrapper only when captured geometry matches;
- unknown/malformed wrappers remain opaque/actual-only;
- progressive high continuation consumes Flow's explicit continuation contract so the default single dense warmup is not repeated.

The corrected production schedule is:

```text
18 logical
14 actual transformer NFEs
4 Spectrum forecasts
```

The bypass control is 13 actual + 5 forecast. The one additional SOL actual is the intentional initial `dense -> sol` backend transition.

## Zero-copy BTHD input layout

Pinned Comfy MiniMax-H3 produces Q/K/V from a packed projection:

```text
[T, 3*H*D]
  -> split whole Q/K/V slabs
  -> view [T,H,D]
  -> transpose(0,1).unsqueeze(0)
```

The historical Sol bridge then transposed BHTD back to BTHD and forced `.contiguous()` on each tensor. v0.1.0 instead preserves suitable BTHD views directly and requires only `stride(-1) == 1`.

The arithmetic-gate identity includes device, dtype and each Q/K/V `(shape, stride)` tuple. A strided layout therefore cannot reuse calibration from a different contiguous layout.

### Production mixed layout

```text
shape      [1, 43545, 56, 128]
Q/V stride [7168, 21504, 128, 1]
K stride   [7168,  7168, 128, 1]
```

Real production execution confirms `materialized_qkv_bytes=0` and finite arithmetic gates through `cute_sm120`.

### Isolated real-SM120 A/B

`tools/bthd_layout_benchmark.py` feeds identical Q/K/V values through:

```text
strided              current zero-copy path
contiguous_kernel    pre-materialized contiguous BTHD kernel-only
copy_plus_kernel     old per-call .contiguous() copies + kernel
```

Seven-run production medians:

| Path | CUDA median | Host-wall median |
|---|---:|---:|
| strided zero-copy | **46.768 ms** | **42.338 ms** |
| contiguous kernel | 46.941 ms | 42.376 ms |
| old copy + kernel | 47.727 ms | 43.136 ms |

Interpret within each timing clock:

- strided kernel vs contiguous kernel: `-0.37%` CUDA / `-0.09%` wall, i.e. effective parity;
- old copies add about `0.786 ms` CUDA / `0.761 ms` wall per representative mixed call;
- zero-copy vs old copy+kernel is about `2.01%` faster CUDA / `1.85%` faster wall for the isolated call.

The old path materialized `1,248,522,240` bytes per mixed call. Across 144 representative calls, zero-copy avoids about **167.44 GiB** of redundant Q/V materialization and roughly **0.11 s** of direct copy overhead.

The native rectangular control is already contiguous, so historical `.contiguous()` calls are no-ops there; its small timing spread is treated as noise rather than a native-path speedup.

This is a micro-optimization result, not a large whole-workflow speed claim.

## Approximation boundary

Rectangular repacking changes 64-row query groups compared with the historical square-expanded bridge. That changes query centroids and their ordinal alignment relative to Sana's local exact heuristic. Sparse output is therefore not promised to equal historical square-expanded sparse output.

The K/V set available to each operation is unchanged. The approximation boundary is query grouping/routing, not VDN domain membership.

The mixed route has an additional experimental boundary because mixed video rows are explicitly non-uniform. Sol-H3 relies on Flow's published contract rather than inventing a native-grid lattice.

## Arithmetic gate

All-selected calibration uses `sink_tokens=Tkv` and an independent rectangular BF16 SDPA reference. Aggregate requirements remain:

- finite output;
- mean absolute error <= `0.002`;
- relative L2 <= `0.005`;
- scale-aware catastrophic-peak guard.

Changing Q/K/V geometry or layout creates a distinct calibration identity. Failed calibration is fatal and is never silently accepted.

## Provenance

`sol_manifest.json` retains original SHA-256 values for upstream source files and separate packaged hashes. Functional modifications beyond import adaptation are restricted to:

- `interface.py`;
- `preprocess.py`;
- `sm120/mainloop.py`.

Their headers identify the changes. `tools/rectangular_sm120.patch` reproduces the functional diff after import adaptation. `python tools/vendor_sol_attn.py /path/to/pinned/Sana` reproduces the packaged tree for source-audit/CI purposes; it is not an installation requirement.

`SANA_SOURCE_SNAPSHOT.json` remains the historical upstream snapshot rather than being rewritten to imply upstream validated these changes.

## Validation

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q tests/test_rectangular.py -m gpu
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512 --heads 8
python tools/bthd_layout_benchmark.py --warmup 2 --repeats 7
```

The direct benchmark bootstraps the repository root itself and can be run from a normal checkout without an editable install solely for imports.

Native grouped VDN success:

```text
sol_backend = cute_sm120
sol_source_tree_verified = true
vdn_rectangular_sol_calls > 0
vdn_requested_q_rows == vdn_kernel_q_rows > 0
vdn_square_expanded_calls = 0
```

Mixed-grid success:

```text
external_mixed_sol_calls > 0
external_mixed_q_rows == external_mixed_kernel_q_rows > 0
```

Malformed/unsupported external layouts may instead report explicit `external_sequence_*` fallback. That is local delegation, not a reason to hard-fail the generation.

## Performance boundary

Successful sparse execution alone is not a speed claim. Whole-workflow A/Bs must preserve model/adapters, quantization, seed, sampler/sigma schedule, prompt/dialogue, reference order, resolution/duration, VDN settings, Spectrum actual/forecast schedule and all other patches.

The final zero-copy decision is based on the isolated identical-input benchmark above. Whole-workflow timings remain useful operational evidence but are not used to inflate that micro-optimization into an unsupported end-to-end percentage.