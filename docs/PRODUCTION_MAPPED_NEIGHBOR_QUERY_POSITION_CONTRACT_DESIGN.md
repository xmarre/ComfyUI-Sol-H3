# Production mapped-neighbor query-position contract

Status: implementation-ready architectural design, **not a production implementation or promotion approval**. Audited 2026-09-16. This specification covers additive physical-neighbor protection for native grouped VDN local calls into packaged SM120 Sol-Attn. The causal result is limited to the captured first-high failure; full-trajectory quality remains a release gate.

## 1. Decision and invariants

Introduce `vdn_softmax_provider_v4`. VDN supplies an immutable, versioned description of each requested query's position in its **already gathered restricted K/V domain**, compressed as affine row runs. Sol validates that description, derives an exact bounded K64-neighbor interval per Q64 tile, and passes a small runtime device tensor to its packaged SM120 kernel. The kernel ORs this interval into the existing route before ballot, exact-index compaction and approximation masking.

The first implementation supports exactly representable single intervals of at most four K64 blocks per Q64 tile. That covers current production grouped-window geometry and diagnostic M. Non-contiguous query maps are accepted only when their exact neighbor union satisfies the same representation. Otherwise use native attention for that local call. Never fill holes with a min/max hull or silently substitute Q ordinals.

Mandatory invariants:

- `new_exact = old_exact OR mapped_neighbor`, over valid restricted K blocks. Retain all threshold, sink and old ordinal-neighbor selections.
- Preserve Q/K/V values, row order, restricted support, scale, diagonal threshold preparation, tau, KC/VC preparation and mixed arithmetic.
- Preserve VDN's softmax gate, output projection, learned linear complement, global/anchor calls and dense warmup policy.
- No square-Q expansion, full-K/V reconstruction, QxK mask, per-map JIT specialization, selector monkeypatch or per-local-call GPU-to-CPU synchronization.
- Missing, invalid, ambiguous or unsupported mapping uses the supplied restricted-domain `native()` callback. Arithmetic failures, CUDA execution errors and OOM are not disguised as capability fallback.
- Preserve v1/v2/v3 signatures and dispatch compatibility. Old consumers do not receive extra keywords. Ordinary aligned/square and external/weighted callers retain their established semantics.

## 2. Verified facts and evidence boundaries

### 2.1 Repositories and effective stack

The main branches and all open PR heads/bases were fetched through GitHub. Compare results show all three main heads are ancestors of the respective M heads: Sol 30 commits ahead/0 behind; VDN 110/0; Flow 84/0. Recursive trees contain no `AGENTS.md`. Changed-file comparisons show no instruction-file deletion between main and those heads.

| Repository | main | M head |
|---|---|---|
| [Sol-H3](https://github.com/xmarre/ComfyUI-Sol-H3) | `f82ff2693be37dbad3438a30eb389d77136c0276` | `b95ad7b3bc7028465547b22fc61300cda53eb110` |
| [VDN-H3-Plus](https://github.com/xmarre/ComfyUI-VDN-H3-Plus) | `76b31323f9e09019b435237dcd8bad1e05476ce1` | `6ca09ec37cd2dcfad02b790573a79b0462a662f3` |
| [Flow-Aligned-Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) | `970396db839ae7ab431b9718859f6d48a2e5019b` | `9b1c2c1bb32e17592a72f5e8dbc8798b9785c1cb` |

The relevant base relationships are explicit, not inferred from PR numbers:

| Repo/PR | Head SHA | Base |
|---|---|---|
| Sol #9 | `fdd52bbd07bd88dfe3d31c823de0d14c27c74aff` | main |
| Sol #11 W | `fff274559fe08f9b8ede6affe05c5487b2f8747e` | #9 |
| Sol #12 E | `6f9e2f5fb625252f03807f9b3dd09be837b8daab` | #11 |
| Sol #13 M | `b95ad7b3bc7028465547b22fc61300cda53eb110` | #12 |
| VDN #14 | `d5f158a1c1750d79b37cb6ef22b0f14e6a8cbad6` | main |
| VDN #8 | `71549c02bc8e73c4c968a43b3955a3998715660d` | #14 |
| VDN #15 W | `7d2d146c658e0ca4a90aa8a90b3ab35d78a2bbed` | #8 |
| VDN #16 E | `10bf368bee38b9f942961e6d8c374b5af6c4bc16` | #15 |
| VDN #17 M | `6ca09ec37cd2dcfad02b790573a79b0462a662f3` | #16 |
| Flow #33 | `9c400d46e990222cf723c426039b4c543192e185` | main |
| Flow #35 | `a99ed7a7ca20ed3af5282c73cad4bb662e45f804` | #33 |
| Flow #36 | `cf260160df234f7a6c61787725b22eddd522c438` | #35 |
| Flow #41 | `4ae2e35f77ed961151ab5695bef9e1dbe277cc54` | #36 |
| Flow #43 W | `b41a20c8b15e17fcb97329e2bbbb38e058bc3506` | #41 |
| Flow #44 E | `b2f9d42cb99a57b19abfa40a5ff1dbe60ed352b8` | #43 |
| Flow #45 M | `9b1c2c1bb32e17592a72f5e8dbc8798b9785c1cb` | #44 |

W/E/M have no submitted reviews or inline threads at the audited heads; returned check runs are successful. Their issue comments report automatic review skipped, not approval. Continuation comments confirm the preserved W-to-E stack order. Ancestor Flow #41 has two resolved review threads, covering replay selector documentation and derivation of provenance equivalence from validated full provenance. Preserve that tamper protection. All open PR review/check lists were inspected. Independent VDN #12 (`094fd0f4e76df6a436de7c8d522b678fcb94008e`) has changes requested and no returned check runs: the review requires APPLY_MODEL compiler-guard scope, not OUTER_SAMPLE. The M runtime already uses APPLY_MODEL. Independent Flow #34 has failed/cancelled test runs. Neither is an implicit dependency to apply.

Other open Flow overlays include #30 Mixed-Grid (`f6a940fdc2fb6d31249a487aff5e4a1297151e9c`), #32 seam diagnostics, #37 forecast isolation, #25 decode and #29 design. Inspect installed files and active wrappers to determine which actually participate. Do not merge every open PR to manufacture an effective tree.

The M preflight manifest records Patcher-produced installed labels, different from PR heads:

| Owner | Installed HEAD | Dirty |
|---|---|---|
| Sol | `c1b7136cfda9f110192bcca97307cb70e25037e9` | false |
| VDN | `10255747eb225cb20b1aff10ec7c2e39cb080002` | true |
| Flow | `c0e324da8be0294c61888f8bf5238a5249a3bc6c` | false |
| Spectrum | `b1cf2870fd843530c7b7ef56150162c9a6125d21` | false |
| ComfyUI | `e3c077bd8a31eb6a9c0efa64a65b341337032dbd` | true |

The full log identifies ComfyUI 0.35.0 on `patcher/stack`. Relevant loaded file hashes match the fetched M-head bytes: Sol contracts/runtime/interop/sparse/provenance/weighted_measure/M module; VDN apply/hybrid/retained/runtime/window/softmax_provider/mixed_measure_epilogue/M modules; Flow attention/mixed_grid/provenance/runtime-observation/M module. This corroborates the effective attention path despite different Git labels; it does not certify every workstation file or its current state after the capture.

Installed Spectrum `runtime.py` matches current fetched main bytes, SHA-256 `0bd77c4649828209439519eb5f6cd7305ee16a59f3d6d5239858ebd31c4ba7bc`. Installed `backend_history.py` matches PR #110 at `78a9a5b36c185a55af59a44c073bc7f8e534bc97`, SHA-256 `45aa4395d1f2bf16a9564016b52b55c144922414c63dd2565944871c4695d2a8`, rather than main. Its special receipt normalization applies only to weighted core-BSA receipts; generic Sol receipts still compare as complete tuples. No Spectrum production edit is required by this design.

### 2.2 Controlled evidence

Primary diagnostic authority: [FIRST_HIGH_SOL_LOCAL_ROOT_CAUSE_AND_FIX_DESIGN.md at 0b8715f](https://github.com/xmarre/ComfyUI-Sol-H3/blob/0b8715faa0a82c730f5aaf0e44b1185e64291e49/docs/FIRST_HIGH_SOL_LOCAL_ROOT_CAUSE_AND_FIX_DESIGN.md). The [M selection checkpoint](https://github.com/xmarre/ComfyUI-Sol-H3/blob/b95ad7b3bc7028465547b22fc61300cda53eb110/docs/FIRST_HIGH_M_SELECTION_EVIDENCE.md) supplies W/E interpretation, the BF16 numerator correction and the rejected threshold experiment. Those earlier tensor calculations were not rerun during planning.

Inspected machine files: `M_report_00001.json`, durable M JSON, `metrics_00483_.json`, `preflight_provenance_manifest_00007.json`, `executioncontractreport_00017.json`, and full log `Pasted text(20260916-022930).txt`. Downloaded M tensor bytes were SHA-256 checked; tensor deserialization was not performed in the planning runtime, which lacks torch.

For preserved R capture `234ed062128e43ed8d5ec63e27517b22`:

| Check | Verified result |
|---|---|
| Execution/arithmetic/entry-state/first-call-only | all true |
| Core cleanup | complete; loaded-model/inner-model/thread-pool ownership absent |
| Calls | 1 logical, 1 actual, 0 forecast; 0 extra learned-upscaler calls |
| Backend receipts | 700: 528 M local, 22 native dense local, 50 native global, 100 native anchor |
| Route records | all 528 valid; additive-only; restricted domain unchanged |
| Original / added / effective exact pairs | 226,499,197 / 2,817,011 / 229,316,208 |
| Exact work | +1.243717875% aggregate; maximum individual call +2.738717142% |
| Model call | 48.136404779 s from metrics; E comparison approximately 90.566 s from prior evidence |
| Raw/pre-guidance video tensor digest | both `4edbdb4c5abc4a5d77dde1abd6ab6dd92db265e7520b07d96bfe485a6871c09d` |
| Durable JSON file SHA-256 | `1914990091be84bf01c382820b1b12e7f73a9886db114ce0f820fced4122b0ba` (independently matched) |
| M tensor file SHA-256 | `8c6ff8b7781557a5da5dd2b3278e1e0b394f32424e51ea30089f1db23a0b0a2e` (independently matched) |

The execution observer has its own capture ID `3a1be03adf5f43b2b72760a01906c5c0`; its nested first-high contract explicitly binds R capture `234ed...` and mode `mapped_neighbor_m`. Do not confuse observer and replay identifiers.

Packaged provenance reports `cute_sm120`, compute capability (12,0), torch 2.10.0+cu130, CUDA 13.0, Cutlass DSL 4.7.1, source contract `sana-sol-engine-sol-attn-64-rect-sm120-v3`, Sana `2936c47637380842aaa4a4488fac5006cc542b70`. Manifest and loaded-source hashes agree in the report.

`media_clean=null` and `production_sparse_conformant_to_independent_witness=null` are preserved as such. The supplied completed investigation records manual clean-media acceptance of `first_high_model_raw_00002(2).mp4`, `first_high_pre_guidance_00002(1).mp4`, and `MiniMax_H3_00002-audio(2).mp4`. Planning does not claim a new media inspection or general artifact cure. M is an approximation with a corrected route, not native-SDPA equality.

### 2.3 Executed source path

At the pinned Sol/VDN heads:

1. Sol `runtime.BlockPatch.__call__` installs v1/v2/v3 closures in copied block options. `warmup` combines trajectory warmup and `index < dense_layers`.
2. VDN `hybrid.make_vdn_forward` performs projection, QK normalization and RoPE, retaining separate raw inputs for its learned branch.
3. `retained.window_softmax_grouped_runtime` preprocesses the full packed QKV once, builds/reuses the plan, gathers Q and `[global_idx; win_idx]` K/V, dispatches, and scatters results. v2 alone lazily gathers square Q; v3 avoids that allocation.
4. `softmax_provider.dispatch` prefers v3, then v2, then v1. Sol v3 delegates to v2, which currently ignores v2 query positions and calls `sparse.attention(..., recompute_prefix_queries=False)`.
5. `sparse.attention` preserves strided BTHD views, checks the verified package, calibrates new layouts using all-selected attention, then runs ordinary sparse attention.
6. Vendor `interface.sol_attn -> _sol_attn_cute -> prepare -> sm120.make_kernel -> SolAttnForwardSm120` builds KC/VC/thresholds and executes the mixed mainloop.
7. `common.selector.sol_attn_route_is_exact` uses `(column_mean > threshold OR abs(q_block-kv_block)<=1) AND valid`. SM120 adds sink forcing before ballots, exact index compaction and masking selected centroids out of the approximate contribution.
8. VDN applies its trained gate/projection and adds its separately projected learned complement. That complement is not a term in Sol's softmax denominator.

M intercepts retained grouped attention and temporarily replaces a Python selector binding while compiling a specialization keyed by descriptor values. Its group-10 Q rows map to K positions 9245..10268, whereas Q block ordinals are 0..15. This is the coordinate mismatch. Production must change transport and the real ABI, not install those diagnostic wrappers.

## 3. VDN contract and mapping ownership

### 3.1 Negotiation

Add `KEY_V4 = "vdn_softmax_provider_v4"`, `PROVIDER_API_VERSION = 4`, and `has_v4`. Dispatch priority: callable v4, v3, v2, v1, native. A present malformed v4 entry fails to native for that call; do not quietly downgrade a rejected v4 map to v3 sparse execution. Older absent keys continue the existing preference order. Preserve existing validation of returned shape/dtype/device.

New signature:

```python
provider(native, q, k, v, *, kind, scale,
         square_aligned=False, sink_rows=0, query_position_map=None)
```

Q is `[Tq,H,D]`; K/V are `[Tkv,H,D]` in VDN gather order. `sink_rows` describes K/V only. `query_position_map` is required for non-aligned local mapped execution. Global/anchor calls carry `None` and remain native. v4 must suppress the v2 square gather just as v3 does. A mapping rejection cannot rerun preprocessing or change the callback's captured Q/K/V.

Do not reinterpret v3, and do not derive a v4 capability from the mere presence of v2 `query_positions`. New Sol retains callable v1/v2/v3 adapters: aligned local calls retain their old sparse path; non-aligned local calls without the new capability explicitly use native. This deliberately replaces the unsafe old rectangular Sol route in the upgraded provider while keeping old APIs callable. Updating only VDN cannot correct an old Sol binary; the paired release requires both upgrades.

### 3.2 Exact immutable wire schema

Use a tagged tuple, not a process-global registry or a class imported across custom-node namespaces:

```python
("vdn_query_positions", 1,
 owner_generation, plan_digest, group_index,
 q_rows, kv_rows, sink_rows,
 query_position_runs)

# query_position_runs: tuple of (q_begin, q_end, kv_begin)
# For q_begin <= i < q_end: position[i] = kv_begin + i - q_begin.
```

All row counts/indices are Python integers (`bool` is invalid); counts are positive, sink is in `[0,kv_rows]`, and kernel-addressable counts must be below 2**31. Runs are nonempty, ordered, disjoint and cover `[0,q_rows)` exactly. Mapped K positions must be in range and strictly increasing across runs, with no repeated positions. Merge adjacent runs with the same affine offset into one canonical run. `owner_generation` is an opaque nonempty string allocated once per VDNState/Apply-VDN ownership generation, never once per call. `plan_digest` is lowercase SHA-256 of canonical JSON geometry described below, not a digest of tensor addresses or activations.

This transports the exact position function without allocating a CUDA `Tq` vector or `square_q`. It is a narrow geometry contract; it does not carry attention weights, arbitrary masks, thresholds or external-sequence semantics.

The producer binds the tuple to the current immutable window plan. The consumer cannot prove row identity by inspecting floating Q/K values: VDN must guarantee the identity structurally from the shared gather description. Tests independently assert `domain_idx[position] == q_idx` for every row. A tensor-built future plan without an authoritative CPU identity description cannot advertise this version until it can establish the same guarantee.

### 3.3 One CPU geometry description shared with gathering

Extract the CPU frame-list construction from `retained._build_window_plan` into a narrow VDN helper (`vdn_h3/query_positions.py`). Retain the existing group tuple shape and `square_aligned` entries for old users; add parallel immutable map metadata. Both the device indices and mapping runs must be built from this one CPU description.

For each group, the helper owns:

- actual ordered `frames` for Q;
- actual sorted/deduplicated `key_frames` for the window and anchor columns;
- ordered global ranges `[0,video_start)` then `[video_end,seq)`;
- group order, anchor-row slices and alignment flag.

Let `S=tokens_per_frame`, `G=video_start+(seq-video_end)`, and `rank(f)` be frame f's rank in `key_frames`. For query frame f at offset j in the group's Q frame list, create run `(j*S,(j+1)*S,G+rank(f)*S)`, then merge adjoining affine runs. Require every query frame to appear exactly once in the local key-frame list. This is algebraically the existing `searchsorted(win_idx,q_idx)+G` without GPU search/scalar reads. It remains correct when packed global rows contain a suffix: the whole gathered domain need not be globally sorted; only `win_idx` is sorted in the old search.

The canonical plan digest includes schema/layout algorithm versions, packed sequence length, video interval, frame count, S, actual clamped bounds, anchor mode, ordered globals, every group's Q/key frame lists, alignment flags and anchor slices. Device, tensor addresses, source Git labels and call ordinal are excluded. Owner identity is a separate field. Validate `video_end-video_start == num_frames*S`, nonnegative ranges, sorted unique frame lists, positive S and unambiguous group coverage before publishing a capability. A malformed map with an otherwise valid native gather falls back; an intrinsically invalid native plan is not repaired by fallback.

### 3.4 Which geometry is guaranteed

For current `window.window_bounds`, frame mode has monotone clamped `(lo,hi)` and chunk mode has monotone chunk-aligned `(lo,hi)`. Equal pairs occur over contiguous frame runs. Excluding only endpoint anchor rows cannot create an interior gap. Every group's query frames are consecutive within its sorted key-frame domain; added anchor columns only shift their rank. Thus their mapped positions form one contiguous run, regardless of prefix alignment, positive tokens-per-frame or frame/chunk tails.

A source-derived CPU enumeration checked 25,600 combinations: frames 1..100, radius 0..7, chunks `{0,1,2,3,5,7,16,100}`, all four anchor modes. Frame-rank contiguity held. This supports the proof but is not a CUDA or gather execution test. The generic builder accepts arbitrary bounds: `[(0,2),(1,1),(0,2)]` groups frames `[0,2]` together, giving a real non-contiguous counterexample. Future builders must therefore validate, not inherit the production-bounds assumption.

Anchor columns belong to the restricted K domain and count in its coordinates. Global/anchor query calls remain separate native calls. A partial Q64 tile uses only real Q rows; partial K64 blocks are clipped to `ceil(Tkv/64)`, with existing token masks unchanged. `square_aligned` means identical ordered domains, not equal lengths.

## 4. Sol descriptor and packaged kernel

### 4.1 Exact descriptor derivation

Add `sol_h3/mapped_neighbors.py`, a CPU-only validator/compiler for the wire tuple. For Q tile t, enumerate the distinct K64 block numbers represented by positions for real Q rows `[64t,min(Tq,64(t+1)))`; runs permit interval arithmetic without expanding every row. Form:

```text
B_t = { floor(position[i]/64) : i in real Q tile t }
N_t = union { b-1, b, b+1 : b in B_t } intersect [0, ceil(Tkv/64))
```

Merge the exact integer block sets into disjoint intervals. Accept only when `N_t` is one nonempty interval `[lo,hi)` with `hi-lo <= 4`. Do this for every tile; if any fails, fallback for the entire local call. This accepts some non-contiguous maps exactly, but never over-forces through a hole. No truncation or dropped represented block is allowed. Validate before copying/launching.

For a contiguous tile the calculation reduces to `[max(0,first//64-1), min(Kblocks,last//64+2))`, at most four blocks. Group 10 must yield first `[143,147)` and last `[158,162)`. For identity-aligned Q/K, `N_t` is already included in ordinal protection; skip descriptor transport and keep the ordinary path.

Return a canonical CPU tuple of intervals and descriptor digest. Device representation is contiguous int32 `[ceil(Tq/64),2]`, stride `(2,1)`, 8 bytes per Q tile, shared across heads and the current B=1 batch. Positions are head-independent. Do not expand to per-head metadata. Future per-batch/per-head different maps require a new contract rather than broadcasting incorrectly.

### 4.2 ABI and CTA integration

Thread optional `mapped_neighbor_intervals` through `sparse.attention`, `load_kernel`'s closure, vendor `interface.sol_attn`, `_sol_attn_cute`, `_compile_sm120`, `sm120.make_kernel`, `SolAttnForwardSm120.__call__` and `.kernel`.

Add structural `mapped_neighbors_enabled: bool` to the kernel recipe. Descriptor **contents** are runtime values, never `Constexpr`, Python closure constants or cache-key values. Pass a tensor ABI argument; disabled specialization may use an existing dummy tensor that is never read under a compile-time-false branch, so ordinary calls allocate no metadata. Validate the mapped tensor's int32 dtype, device, shape, contiguity and profile before launch. The trusted wrapper owns values derived from validated CPU data; no device readback is required. The low-level vendor tensor argument is internal/trusted metadata, not a replacement for the public VDN validation boundary.

In the SM120 mainloop, routing warp 0 loads this Q tile's two endpoints once before the route-group loop and retains them in registers. Identical per-lane reads of the two addresses are acceptable and avoid introducing new synchronization primitives. For each valid K block:

```text
old_exact = existing selector result OR existing sink predicate
mapped = lo <= kv_block < hi
exact = old_exact OR mapped
```

Keep the `valid` guard around both. Apply this before the existing ballot, `column_masks`, rank and `route_indices` writes. Then exact compaction and approximate-centroid removal see the same effective route, so added blocks are not counted twice. Preserve ascending block order, existing route-group capacity and online-softmax/PV arithmetic. No changes to `common/selector.py` are needed; other backends keep their selector unchanged.

### 4.3 Calibration, caches and execution errors

Extend the compiled key with the structural mapped-profile flag and ABI/source contract version while preserving existing device/arch/batch/Q-length/K-length/head/split/stride identity and weighted flag. Descriptor layout is fixed by Q shape. Changing offset, owner, sink or interval contents at fixed tensor layout must not compile again. Within the supported unweighted path this adds at most one mapped specialization per existing layout, not one per window/layer/plan.

The all-selected calibration must exercise the mapped ABI variant on first use of that layout; sink-all still forces every valid block, so descriptor content cannot affect its arithmetic. Add mapped enablement/source version to `sparse_verified` identity, not every descriptor digest. Keep the existing shape/stride/device/weighted calibration checks. Do not calibrate all maps or perform an extra H3 evaluation.

Capability/representation/backend rejection occurs before launch and produces native fallback with a bounded reason code. Keep `KernelUnavailable` for loader/backend unavailability. Do not catch arbitrary `RuntimeError`, failed arithmetic gates, asynchronous CUDA errors or OOM and retry attention; those failures must propagate. Do not cache a corrupt descriptor as valid or poison every map because one geometry is unsupported.

Sol's public integration still requires Linux/WSL SM120 CuTe for sparse execution. The new vendor argument must reject SM90/SM100/SM103/Triton paths explicitly; it cannot be silently ignored. Windows retains its existing native fallback. Combined mapped+key-bias requests are rejected/fallback through the owner-specific native path in this first release; weighted/external normal dispatch never supplies mapped metadata.

## 5. Lifetime, caching and compilation boundaries

VDN's immutable CPU geometry/maps belong to the retained plan, keyed by the current plan tuple plus mapping schema version. Existing key fields are video interval, frame count, S, bounds, anchors, sequence length and explicit device. The pure CPU helper may use a separate eight-entry `functools.lru_cache`, keyed by the complete immutable geometry values and schema; it stores no model, owner, tensors or mutable cfg. Bind owner generation outside that reusable summary. This lets preflight and all fifty block calls reuse frame lists/digests without acquiring an execution pool or serializing a plan every call. Preserve `_MAX_WINDOW_PLANS=8` and the `RuntimeBufferOwner` lease: nested/concurrent execution without the primary lease uses transient buffers. No module-global device tensors or mapping state on shared attention functions.

Sol owns an LRU of validated descriptors in its existing `runtime.Request` dataclass, cleared with `SamplingWrapper` teardown. Key: wire schema, owner generation, plan/group identity, canonical runs/counts, policy version and explicit CUDA device. Store CPU intervals/digest plus the device tensor and its creation stream/event. Bound it by both 64 entries and 4 MiB of descriptor tensors; this comfortably covers the captured 11 groups and prevents unbounded geometry churn. Oversized individual metadata falls back with `metadata_budget`; eviction drops cache ownership only, retaining the in-flight call's reference. No Q/K/V tensors are cached here. Keep the small immutable validation identities needed for receipts through completion of the current model call even if their device descriptors are evicted; retire them with that request/call scope rather than accepting stale cache entries.

Cold path: build CPU geometry once, validate/derive once per unique map, and copy a small contiguous tensor to the execution device. A synchronous host copy is acceptable on this cold path; no `.cpu()`, `.item()`, `torch.equal` readback or mapping hash of CUDA bytes may occur in a warmed local call. Current captured local queries total 51,200 rows per block (52 frames minus two anchors, 1,024 rows each), so all eleven descriptors together occupy 6,400 bytes before optional deduplication. Do not reserve a QxK route mask.

Use the calling CUDA stream. For reuse on another stream, wait on the creation event and record consumption on that stream before eviction/free. CPU immutability and a Python lease alone do not prove GPU completion. Do not mutate descriptor storage in place to serve another map. Across devices, separate keys and allocations; retain current rejection of a device change within a Sol sampling request. On unload/reload/new Apply-VDN, do not reuse old owner-bound maps. Preserve VDN's cancellation clear path and normal retained-buffer policy; this change adds no independent persistent device cache.

The vendor `_compiled` cache currently lives at module scope. Keep compiled executables reusable across requests, but store no descriptor tensor/value/owner in compiled entries. Source contract changes invalidate specializations; process restart is required when deploying changed loaded modules. Compile on the existing current-device/stream path, publish only a successfully compiled entry, and serialize same-key first compilation if concurrent calls can otherwise race. A lock may protect cache creation; it must not mutate selector globals or hold a global execution lock around attention.

No new claim of torch.compile/CUDA-graph compatibility is made. Keep Python negotiation/CPU planning outside captured graphs. Eager execution is the supported path. A capture must be prewarmed with all immutable descriptors and compiled variants, retain their storage for graph lifetime and prohibit metadata misses; otherwise decline mapped execution before entering capture using the existing native/owner policy. Do not allocate/copy/compile silently inside capture. VDN's APPLY_MODEL allocator-compiler guard remains unchanged.

## 6. Numerical history and receipts

### 6.1 Preflight must know the plan without executing H3

A production routing correction changes numerical policy. Include mapping policy presence/version and deterministic geometry in history. Do not normalize the production mapped route to `vdn_local_sol` as M did.

Add a narrow VDN-owned callable attribute on each VDN forward: `vdn_query_position_plan_v1(options, layout)`. Attach it in `make_vdn_forward` without replacing its closure or changing audio logic. It returns an immutable CPU summary containing owner generation, grouped/native mode, plan digest and ordered wire maps. It uses the same CPU geometry helper as the actual gather. It must derive current geometry from the supplied native PackedLayout (validated segments, seq_len and five-field padded native signature) and captured VDN cfg/branch state, **not** `state.layout` from a previous actual call. Native video dimensions/rows must cross-check. Unsupported signatures, opaque wrappers, external/reduced layouts or Flex routing return `None` for the mapped preflight.

Extend Sol `_vdn_history_identity` to consume and validate this explicit hook before its audited legacy closure fallback. Keep existing replacement-chain, DiffAid, Flow, dense-provider and warmup identities. For upgraded grouped native calls add `(mapped_policy_version, owner_generation, plan_digest, ordered(group_index,q_rows,kv_rows,sink_rows,map_digest,descriptor_digest,route_class))`. Derive CPU descriptor identity without uploading or invoking kernels. Unknown/invalid maps are actual-only initially; deterministic valid mapped geometry can forecast after an accepted actual. Older VDN closure recognition remains valid for aligned/native routes but must not predict mapped sparse execution without the hook.

Avoid storing a single mutable 'last plan' on `HistoryPolicy` or a shared model. Request-local validation caches may hold immutable summaries; options/ContextVar scopes and owner generation must prevent cross-request/nested-model reuse. Actual map validation independently binds q/k lengths, sink, group and owner. A preflight expectation, if cached, must be indexed by complete policy/layout identity and cannot be overwritten by an unrelated nested call.

Concretely, the Sol block's v4 closure binds the current block attention forward and its advertised VDN plan hook. Resolve the CPU summary for that block's current native layout once, compare the wire map to the summary's exact group entry, then check actual Q/K lengths and sink before using a cached descriptor. This binding is required even when Spectrum is disabled. Missing hook, mismatched owner/group/plan or an opaque forward wrapper uses native fallback. A caller-provided digest alone is not ownership proof. Pure summary/descriptor caches amortize this check; do not read CUDA values. The preflight receives the same native PackedLayout from Spectrum `_resolve_layout` before `begin_model_call`, as verified in `comfyui_spectrum_h3/minimax_h3.py`.

### 6.2 Stable completed receipts, separate telemetry

Use route `vdn_local_sol_mapped_v1` after successful completion. Keep the existing outer receipt shape `("sol_h3",block,route,fields)`; mapped fields are a new tagged immutable tuple:

```python
("vdn_mapped_neighbor_v1", owner_generation, plan_digest, group_index,
 q_rows, kv_rows, sink_rows, map_digest, descriptor_digest,
 "k64-union-radius1-additive-interval4-v1", kernel_contract, True)
```

The final boolean means completed. No tensors, device pointers, call token, evaluation number, wall time or activation-dependent selected counts enter this tuple. The numerical receipt must be identical on consecutive actual evaluations with the same routing policy/geometry, even when selected threshold blocks differ. Record it once, after the provider has a valid result; no successful receipt on fallback or exception. Existing global/anchor/dense receipts remain unchanged.

Extend `HistoryPolicy.accept_receipts` to recognize this tag, validate types/profile/completion and identities against request-owned validated maps, reject mixed malformed entries, and keep existing weighted handling unchanged. The Spectrum owner creates a fresh receipt list for each actual call. Add exact block/group/order/count checks in integration tests; do not rely only on the presence of one mapped label. Raw per-call diagnostic events, call IDs and selected counts use a separate optional telemetry sink, not the Spectrum-compared receipt list.

Mapping failures record `vdn_local_native_mapping:<bounded_reason>` with reasons such as `missing`, `schema`, `owner`, `domain`, `fragmented`, `interval_width`, `metadata_budget`, `backend`. These are explicitly rejected for forecast acceptance in v1; the next actual may re-establish mapped safety. Do not label these as a successfully mapped Sol call or use the old accepted `vdn_local_native` label to hide a routing transition. Stable unsupported configurations can be optimized for forecastable native fallback later with separate proof.

The source contract/fingerprint change and geometry/fallback transitions can legitimately invalidate Spectrum history and force actual calls. Stable mapped execution must not generate an artificial reset every evaluation. Do not hardcode NFE/forecast ratios or promise the full-trajectory schedule cannot change: adaptive forecast error can change because the numerical output changes. The controlled first-high replay must still be exactly 1/1/0.

## 7. Compatibility and fallback matrix

| Caller/path | Required behavior |
|---|---|
| New VDN + new Sol, valid unweighted grouped map | v4, additive mapped sparse SM120 after existing warmup |
| New VDN + old v1/v2/v3 consumer | original signature/priority behavior; lazy v2 square payload only when actually needed; no new kwargs |
| Old VDN + new Sol, non-aligned local | native restricted callback; explicit missing-capability receipt; no ordinal inference |
| v2 map supplied without v4 | preserve API acceptance; non-aligned native; no sync/implicit v4 promotion |
| True square-aligned local | existing sparse route; identity map adds nothing; unchanged numerical selections |
| Equal shapes but different/unproven row identity | never infer square alignment; require map or native |
| Ordinary non-VDN square Sol | unchanged route and dense-prefix-query handling; zero descriptor allocation |
| Ordinary external rectangular / Mixed-Grid / weighted Sol | existing separate owner contracts unchanged; no VDN-map injection; existing weighted calibration/epilogue tests remain gates |
| VDN external/reduced sequence | existing dense-gated/external owner route; complement behavior unchanged; no grouped map synthesized |
| Global and anchor queries | native as before, with full K/V where VDN already specifies it |
| Dense layer / trajectory warmup | native as before; no descriptor upload or mapped calibration needed |
| Flex and Flex-to-grouped | native masked behavior unchanged; actual grouped fallback may use v4, but uncertain Flex history remains actual-only |
| Training/reference grouped path | unchanged; do not inject inference-only retained metadata into autograd |
| Missing/invalid/unrepresentable map | native same restricted Q/K/V; complement still executes once |
| Non-SM120 / unavailable CuTe / Windows | existing supported native fallback; mapped argument must not reach an unsupported sparse backend |
| Unknown low-level mapped tensor or mapped+weighted request | reject at wrapper/dispatch boundary; never ignore metadata |

No requirement to repair unrelated external routes is inferred from the first-high result. Conversely, compatibility tests must ensure the new provider and kernel flag do not change those routes accidentally.

## 8. Reproducible packaged source

Current canonical process: `tools/vendor_sol_attn.py` reads the pinned Sana subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, relocates imports, applies `tools/rectangular_sm120.patch`, writes vendor files and `sol_manifest.json`. The rectangular v3 patch already includes weighted-kernel support from the production dependency stack. `SANA_SOURCE_SNAPSHOT.json` is a different snapshot record; do not substitute its revision for the actual Sol package revision.

Add canonical `tools/mapped_neighbor_sm120.patch`, applied **after** the unchanged rectangular v3 patch. Limit it to vendor `interface.py`, `sm120/kernel.py`, `sm120/mainloop.py`; annotate local modifications. Do not change shared selector/preparation or other architecture kernels. Extend vendor tooling to apply/check both patches in order and regenerate all packaged hashes from pinned raw sources. Keep upstream hashes unchanged. Retain the old rectangular patch hash field for audit compatibility and add an ordered patch list with individual SHA-256 values and modification labels. Add manifest contract `sana-sol-engine-sol-attn-64-rect-sm120-mapped-neighbor-v4`; use that same constant in provenance/config/calibration metadata.

`verify_source` must check expected source/revision/contract and the exact file set and packaged hashes, retaining only its existing CRLF-to-LF transport allowance. Regeneration CI verifies canonical patch hashes and byte-for-byte output; runtime package verification must not require a live upstream checkout/network. Update the source-contract tests and documentation together. Native affine manifest/contract is unchanged. No handwritten vendor-only fix is acceptable.

Use the real pinned upstream tree to regenerate. A reverse-applied copy of the current package is not independent provenance. Record upstream-vs-local patch identity, actual loaded module origins and compiler versions during SM120 validation. Old diagnostic source-delta manifests remain frozen evidence: the candidate gets a separate explicit provenance delta, not a widened historical allowlist.

## 9. Exact implementation plan and branch boundaries

Work on separate implementation mirrors. Preserve every W/E/M PR and its base relationship. Production patches should target audited non-diagnostic dependencies: Sol #9 and VDN #8/#14 effective production composition, reconciling newer main as necessary. Do not base a release on importing all M diagnostic installers. Maintain a separate integration checkout/branch for testing the candidate with preserved replay evidence. Do not replace Patcher-managed workstation repositories manually.

| Repo/files | Required changes |
|---|---|
| VDN `vdn_h3/query_positions.py` (new) | immutable CPU frame/range plan, exact compressed mapping runs, canonical plan digest, native-layout summary helper |
| VDN `retained.py` | share geometry description with existing index construction; parallel map entries; v4 avoids v2 allocation; pass correct group map; preserve gathers/scatters/preprocess |
| VDN `softmax_provider.py` | v4 constants/negotiation/signature; preserve old dispatch; explicit map-capability fallback |
| VDN `hybrid.py` | owner generation and attach pure `vdn_query_position_plan_v1` metadata hook; minimal changes around existing state/forward construction; preserve audio, external epilogue and compiler guard |
| VDN `runtime.py` | only schema-aware plan cache/lifecycle changes needed; preserve leases and eight-plan bound |
| VDN docs/tests | `SOFTMAX_PROVIDER.md`, new mapping/v4 tests; extend retained/window/provider-stack tests and real hybrid tests |
| Sol `mapped_neighbors.py` (new) | strict wire validation, exact set-union-to-interval conversion, digest, bounded request-owned device cache helpers |
| Sol `runtime.py` | v4 closure/install, safe legacy adapters, request cache/cleanup/counters, completed mapped/fallback receipts |
| Sol `interop.py` | v4 key; explicit VDN plan preflight; mapped identity/receipt validation; preserve existing replacement/weighted policies |
| Sol `sparse.py` | pass mapped tensor; capability validation; mapped calibration class; lazy-loading/error boundaries |
| Sol vendor/tools/manifests/provenance | ABI/kernel OR change through canonical second patch and regeneration; contract bump; no changes to affine `kernels.py` |
| Sol tests/docs | mapping/provider/history tests, same-input SM120 route tests, cache/provenance tests, paired-upgrade/fallback explanation |
| Flow validation only | new candidate replay adapter/report/source-delta file and tests on a separate validation mirror; use saved bundle and existing strict loader/cleanup machinery; do not modify #43/#44/#45 or M mode |
| Spectrum | no production change; run integration tests against installed-equivalent main + #110 and current live stack |

The Flow candidate adapter must invoke ordinary upgraded VDN retained dispatch and ordinary packaged Sol ABI. It may observe and bound execution, but cannot install W/E/M operator substitutions, compile-time selector replacement or route-label normalization. Give it a distinct candidate mode/schema and exact allowlisted source delta derived from the actual production diff. Preserve validated capture provenance and non-operator state checks. It must not silently relax source, entry-state, topology or cleanup checks just to run changed code.

Implement in reviewable stages: CPU contract/tests; v4 dispatch/native fallback; reproducible ABI/kernel patch; request cache/history integration; CPU/source-contract validation; same-input real SM120 operator gates; controlled candidate replay; full trajectory/performance; paired promotion. Checkpoint pushed GitHub commits before risky changes, hostile review, long tests, and whenever reconstruction would be expensive.

## 10. Validation matrix and acceptance criteria

All criteria below are implementation gates, not tests claimed to have passed for unimplemented code.

| Gate | Cases | Acceptance |
|---|---|---|
| Exact VDN map | prefix/suffix globals; every anchor mode; frame/chunk windows; one frame; non-multiple-of-64 S/Q/K; edge groups | every mapped row matches the independently built gathered domain; unchanged Q/K/V indices and order |
| Descriptor math | contiguous offset 0/1/63/64/65; boundary crossing; one-row and 63/64/65-row tails; group-10 witness; non-contiguous maps with contiguous and fragmented unions | exact equality to set-union oracle; no hull approximation; group-10 endpoints match; invalid/width>4 maps fallback |
| Invalid ownership/schema | duplicate/omitted/reordered Q coverage, duplicate K positions, out-of-range/overflow/bool fields, stale owner/group/sink/counts, mutated plan | rejected before sparse launch; same native callback executes once if its domain remains valid |
| Additivity | random/frozen old threshold+ordinal+sink routes, partial K masks | `old & ~new == 0`; `new == old | neighbor`; no extra blocks outside neighbor set; invalid padded blocks absent |
| Compatibility | old v1/v2/v3 consumers; new Sol with old VDN; true square vs equal shape only; ordinary/weighted/external/Flex/training | expected signatures/routes/counters; no accidental v2 Q gather with v4; no extra preprocessing/gate/complement |
| Metadata lifecycle | repeated groups/layers/evaluations; retain on/off; cancellation/error; nested/concurrent requests; clone/unload/reload; device/stream switch; LRU eviction | no stale tensor/owner, no per-call warmed upload, bounded memory, correct stream lifetime; no descriptor values in JIT key |
| Spectrum | both wrapper orders; known Flow/DiffAid wrappers; stable geometry; changed map at same Q/K shape; warmup transition; fallback/recovery; conditioning subcalls; offline replay | stable numerical receipts and forecasts; real policy changes reset before reuse; fallback actual-only; no call-token/count-driven resets |
| Vendor source | pinned-source regeneration; patch order/hash; exact manifest; LF/CRLF; tamper/file-set negatives; Windows import | byte-identical generated vendor; stale/unsupported capability rejected; ordinary source tests pass |
| JIT structure | same shapes/strides, many descriptor values; mapped flag on/off; weighted ordinary path; different device | no additional compilation for value changes; at most one extra mapped variant per supported ordinary layout; no global selector mutation |
| Real SM120 same-input | preserved E group-10 Q/K/V and other captured layouts where available; strided views; Q/K tails | candidate executed route bit-identical to `old OR exact_map` and diagnostic M oracle; matching output within accepted BF16 arithmetic gate; all-selected variant conforms |
| Controlled first-high | saved R bundle, candidate production path, original adapters/schedule/gates/precision | exact entry hashes, 1 logical/1 actual/0 forecast/0 upscaler; 700 receipts with 528 mapped/22 dense/50 global/100 anchor; cleanup complete; no hidden native local fallback; manually clean complete raw/pre-guidance/final diagnostic clips |
| Full trajectory | original target-input workflow; low/probe/high/later chunks; representative frame/window/resolution/prefix changes; Spectrum on/off; separate weighted compatibility case | reviewed media/audio without regression; report actual/forecast topology and changes with causes; no unsupported generalization from first-high |

Use saved witness Q/K/V for same-input route/arithmetic testing; no H3 call is needed to recompute offline route math. Route equality is stronger than count equality. Debug route ballots may use the existing `debug_route_trace` machinery in a bounded standalone operator test. Production telemetry may optionally accumulate original/added/effective counts in a separate debug buffer using ballots/popcount, drained once after the bounded model call. Disable it for steady-state timing. No full-area persistent mask and no `.item()` per attention call.

The numeric operator gate retains current limits: finite output, mean absolute error <=0.002, relative L2 <=0.005 against the appropriate same-route BF16 reference, and the existing scale-aware catastrophic peak guard. Do not apply output-space absolute tolerance to differently scaled reconstructed numerators. Sparse M vs dense/native approximation error is evidence/telemetry, not required to be <=0.005. Prefer bitwise candidate-vs-diagnostic output where execution order matches; any difference requires explanation under the same route and arithmetic model.

Historical aggregate counts are a regression target only for the same captured activations/execution. With exact same-input route records, require exact original/added/effective counts. If a full model-call rerun differs because upstream numerical inputs changed, it is not a matched reproduction; investigate provenance before loosening the acceptance criterion.

### 10.1 Performance gates

These are proposed engineering acceptance budgets, not measured candidate results. Compare on the same installed runtime/device with paired warmed runs and separately report cold preparation/JIT/calibration. Do not compare a cold candidate to a warm baseline or use wall time alone to infer sparse work.

- Preserve exact M additions for the saved witness. On the captured controlled run target the recorded +1.2437% aggregate work (maximum +2.7387% per call); this percentage is not a universal cap for other geometries. A larger count at identical inputs is a semantic failure, not an allowed performance tradeoff.
- Warm metadata generation/lookup/upload overhead: no transfers after cache warmup; aggregate host overhead <=1% of the H3 model-call time, measured with telemetry disabled. Report cold CPU plan/descriptor time and bytes separately; a >1% model-time preparation overhead requires redesign or evidence-backed exception.
- Warm mapped kernel time: candidate median no more than 5% above diagnostic M for matching same-input operator cases, beyond established measurement noise. Report variance and per-shape outliers; >10% on a common layout requires investigation before promotion.
- Warm full model-call time: <=5% above matched diagnostic M and <=10% above ordinary sparse baseline, absent demonstrated noise/other execution differences. A drift toward all-selected E latency fails the architecture goal. Historical 48.136 s is context, not an exact wall-time requirement.
- Cold compilation: only the bounded structural mapped variant; no compilation as descriptor values/groups change. Record cold total/JIT time separately. More than 25% additional cold compile time relative to the same ordinary shape set requires explanation before promotion; never accept specialization count proportional to distinct maps.
- Memory: descriptors exactly `8*ceil(Tq/64)` bytes per unique cached map plus small CPU metadata/events; request cache <=4 MiB. No Q/K/V copies or full-attention-area allocations introduced. Peak model memory change beyond 1% requires attribution; JIT allocator peaks reported separately.
- Full-trajectory total time and NFE counts are separate gates: an adaptive schedule change must not conceal metadata/kernel regression. Compare actual attention/model-call timings as well as end-to-end sampling.

Start with three warmed operator repetitions and expand only if variability prevents deciding these bounds. The controlled first-high media call is one bounded production-candidate validation; repeat model calls only to answer a concrete performance or reproducibility uncertainty, not to rerun completed R/W/E investigations.

## 11. Rejected and excluded alternatives

| Alternative | Status and evidence | Scope |
|---|---|---|
| Full K/V or removal of VDN complement required | falsified as necessary by clean W native-window with restricted support/complement preserved | captured first-high artifact |
| Permanent all-selected local Sol | valid reference but inferior production candidate: clean E, approximately 90.6 s vs M 48.1 s | controlled call; not a universal performance ratio |
| Generic SM120 arithmetic/stride rewrite | unsupported by all-selected same-input conformance and corrected BF16 mixed witness | tested operator/layout evidence |
| KC/VC/diagonal preparation redesign | excluded; independent E preparation reproduction passed | tested witnesses |
| Full-covariance threshold T | empirically inferior: worse group-10 error with materially changed route density | that witness; not globally disproven |
| Remove old ordinal neighbors | untested independent optimization; excluded to retain M's additive intervention | first production fix |
| Diagnostic selector monkeypatch and per-map constexpr kernel | superseded architecturally: global binding mutation, compilation serialization and value-dependent specialization growth | production lifecycle/cache design |
| Infer map from Q length, offset guess or equal shapes | invalid contract: gathered positions are owned by VDN, prefix/anchors shift coordinates | non-aligned domains |
| One min/max hull for arbitrary maps | mathematically wrong when exact neighbor union has holes | generic/future geometry |
| General CSR/bitmask forced-route ABI now | unnecessary for current proven geometry; broader kernel/memory surface; remains possible future work | layouts that currently fall back |
| Copy M receipt normalization | incorrect production identity; hides a real numerical capability transition | Spectrum integration |

## 12. Remaining runtime questions and assumptions to re-check

Resolved by source: API version choice, mapping owner/domain, current contiguous-group property, bounded descriptor representation/fallback, additive kernel insertion point, source regeneration path, cache ownership and history fields. These are not deferred architectural choices.

Genuinely unresolved until implementation/runtime:

1. CuTe 4.7.1 lowering of the extra int32 tensor and register loads, compiled ABI/runtime-value behavior, resource pressure and performance. Resolve with real SM120 compile, route ballots and paired timing, not CPU tests.
2. Candidate output/media and full-trajectory behavior. M validates the narrow intervention; it does not validate an unbuilt ABI/dispatch/history implementation.
3. Current workstation state at deployment. Supplied manifests establish the successful run, not live remote access. Recollect exact loaded-source/module/compiler provenance before CUDA gates.
4. Captured same-input route reproduction beyond the available reports. The full E witness and R bundle remain required for operator/replay validation; preserve them. Their loss would prevent the specified causal regression gates, not justify guessing inputs.

Implementation-time rechecks:

| Assumption in audited tree | Required verification before editing/promotion |
|---|---|
| v3 is latest provider; hooks absent | inspect live `softmax_provider.py`, installed module origin and `make_vdn_forward`; reconcile any newer API without overwriting it |
| grouped frame construction remains sorted/native | compare live helper/gather order/bounds/anchor semantics; independent identity tests before advertising v4 |
| Spectrum receives current native PackedLayout before actual | exercise both wrapper orders; ensure pure preflight derives identical plan without stale `state.layout`; unknown layout returns opaque |
| native fallback retains local domain | trace callback through current retained path and preprocessing; verify no generic override widens support |
| no hidden operator overlay | record active callable chain and imports; ordinary candidate gate must contain no W/E/M operator substitution |
| source contract still pinned v3 | compare live rectangular patch, manifests, loaded package and upstream revision; preserve later compatible changes and document deviations |
| existing exact gate/failure semantics | ensure mapped flag is represented in calibration identity and errors remain fail-closed |
| stable history interface | re-read installed Spectrum `backend_history` and `observe_backend_history`; no generic Sol normalization currently exists |
| plan cache/leases and compiler guard unchanged | inspect unload/cancel/clone/stream lifecycle and APPLY_MODEL placement; do not import independent #12 |
| diagnostic topology unchanged | re-fetch all W/E/M bases/heads/reviews/CI and verify no historical branch was repurposed |

Justified deviations require a recorded source/runtime finding, replacement decision and updated tests/design. A performance shortfall does not authorize removal of ordinal neighbors, threshold changes or relaxed map validation.

## 13. Planning verification performed

- Fetched live main/open PR state, ancestor comparisons, reviews/review comments/check runs for all three repositories; fetched W/E/M review threads and resolved ancestor Flow #41 threads.
- Read authoritative diagnostic design and M selection evidence; traced provider/gather/kernel/manifest/history/cache paths listed above. Verified 46 downloaded source snapshots against Git blob IDs; matched relevant installed loaded-file SHA-256 values separately.
- Read five machine JSON files plus the successful-run full log; independently summed 528 route records and verified durable M JSON and tensor file SHA-256. Verified observer-to-R binding and installed Spectrum #110 file identity.
- Executed the 25,600-case source-derived CPU geometry check and recorded the arbitrary-bounds counterexample. This checks frame-index reasoning, not tensor gathering or CUDA behavior.
- No production code edits, repository test-suite execution, GPU execution or new media gate occurred during planning. Existing successful CI checks are historical source evidence only. Papers were not needed to resolve these source-level design choices; no paper claim overrides the audited implementation.

## 14. Relevant artifacts outside git

Durable evidence remains external because tensors/media are not source. Preserve the original files; use hashes to identify them. Local planning copies are reproducible from their recorded origins and may be discarded after this design is committed.

| Exact path/location | Contents and significance | Required disposition |
|---|---|---|
| `/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json` and same basename `.pt` | original R entry state/provenance for matched candidate replay | retain and inspect; do not regenerate R |
| `/home/toor/ComfyUI/output/h3_first_high_sol_local_e/234ed062128e43ed8d5ec63e27517b22-all-selected-e.pt` | E same-input witnesses, SHA-256 `e25ff53bb3e6f7d6c1780ba9a1bb44288be1b3e095e87166a77b152971e8a6d5` | retain/use for offline operator validation |
| `/home/toor/ComfyUI/output/h3_first_high_sol_local_e/block2_group10_evidence42.pt` | bounded witness; transported shard SHA-256 `7abacbb03407a7486329ce05bcb4ed2a2778f9eb090c6f08ec890a658bf3f9ec` | inspect/use; also located in supplied file collection, not materialized during planning |
| `/home/toor/ComfyUI/output/h3_first_high_mapped_neighbor_m/234ed062128e43ed8d5ec63e27517b22-mapped-neighbor-m.json` and same basename `.pt` | successful M baseline, hashes in section 2 | retain/use for comparison |
| Supplied complete clips `first_high_model_raw_00002(2).mp4`, `first_high_pre_guidance_00002(1).mp4`, `MiniMax_H3_00002-audio(2).mp4` | manually accepted media reference; exact workstation directories not established | retain; resolve exact files before media comparison, never substitute similarly named incomplete clips |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/M_report_00001.json` | inspected M report | reproduce from supplied file collection or retain |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/234ed062128e43ed8d5ec63e27517b22-mapped-neighbor-m.json` and same basename `.pt` | hash-verified M local copies | reproducible; may discard local copies |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/preflight_provenance_manifest_00007.json` | installed source/compiler/wrapper evidence | retain or reproduce for provenance recheck |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/executioncontractreport_00017.json` | nested R binding and execution/state observations | retain or reproduce |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/metrics_00483_.json` | model-call/topology timing | retain or reproduce |
| `/workspace/scratch/89c3ec918dea/evidence/ComfyUI-Sol-H3/Pasted text(20260916-022930).txt` | successful M full log | retain or reproduce |
| `/workspace/scratch/89c3ec918dea/sources/`, `/workspace/scratch/89c3ec918dea/planning-source-index.json`, `/workspace/scratch/89c3ec918dea/planning-github-audit.json` | read-only fetched source/API snapshots; durable findings are in this document | may discard; re-fetch pinned sources/current GitHub state |

No profiler output, compiled binary, experimental source patch, new route dump or generated benchmark metrics were produced. The only new deliverable is this design; the planning branch contains documentation only.
