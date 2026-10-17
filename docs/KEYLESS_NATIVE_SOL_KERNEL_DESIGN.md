# Native Keyless SM120 Sol kernel design

## Status

This document defines the implementation boundary for the native Sol-H3 backend for MiniMax-H3-Keyless architecture `h3_keyless_core50_v1`.

It is intentionally stricter than the existing materialized compatibility bridge. The current Keyless Sol provider is a correctness/reference path: it receives `(Q, raw V, routing spec)`, materializes the complete logical routing tensor `R = route(V)`, then calls the existing Q/K/V Sol backend as `(Q, R, V)`. That path is useful for arithmetic and ecosystem validation, but it does **not** deliver the production memory objective because the full route tensor exists globally.

The native path specified here must consume Q + raw V + route metadata and derive routing tiles inside the GPU execution path. It must never recreate persistent teacher K weights or advertise a globally materialized route as native Keyless execution.

The implementation is based on released Sol-H3 `main` `0208ddbaaa8a94d70cbf29003680a24d6404fc66`. The independent exact-prefix development PR is not an implementation base for this work.

## 1. Fixed Keyless semantics

For every converted core block:

- Q is already projected, query-normalized and H3-positioned when the provider is called;
- retrieval V is the **raw projected value** tensor;
- routing is derived from raw V as:

  ```text
  R = preprocess(rope(rmsnorm(V, route_norm.weight, eps=1e-5)))
  ```

- H3 RoPE is split-half partial RoPE over 96 of 128 head channels:
  - first half: channels `[0:48]`;
  - second half: channels `[48:96]`;
  - channels `[96:128]` pass through unchanged;
- attention scores use `Q @ R^T`;
- output accumulation uses raw V, never R;
- scale is `128**-0.5` for the canonical H3 path.

The native executor therefore has two logical views of one projected V tensor. They are not interchangeable. `V != R` whenever route normalization, RoPE or routing preprocessing is nontrivial.

## 2. Current SM120 QKV implementation and the incompatibility boundary

The released SM120 Sol path is structurally QKV-specific even though the attention kernel itself accepts arbitrary Q/K/V tensors.

`sol_h3/_vendor/sol_attn/preprocess.py::prepare` currently:

1. receives complete global K and V;
2. computes K block centroids `KC` and V block sums `VC`;
3. computes the Sol route threshold from Q and `KC`.

`sol_h3/_vendor/sol_attn/sm120/mainloop.py::SolAttnForwardSm120` currently receives:

```text
mQ, mK, mV, mO, mKC, mVC, threshold, ...
```

and constructs independent TMA/pipeline paths for K/KC and V/VC. Route-group selection scores Q against `KC`. Selected exact blocks then load K for QK MMA and V for PV accumulation.

Consequences:

- passing raw V as K is mathematically wrong because it omits route RMSNorm/RoPE/preprocessing;
- producing a full `R` before `sol_attn(...)` is the existing materialized reference bridge, not a fused implementation;
- only changing the exact-block K load is insufficient because the **coarse selector** also depends on routed K centroids;
- route centroids cannot in general be obtained by transforming an average V centroid: RMSNorm is nonlinear and RoPE varies by row position.

Both coarse routing and selected exact-score execution must therefore derive route features from physical V rows.

## 3. Native provider ABI

The public Keyless provider API remains version 1. The Sol-owned provider receives:

```text
q                 [Tq, H, 128]
v                 [Tv, H, 128] raw projected V
routing.norm_weight
routing.norm_epsilon
routing.rope_freqs
routing.rope_policy
routing.preprocessors
routing.value_domain
routing.routing_position_domain
query_domain
value_domain
mask
log_measure
exact_blocks
```

The native kernel implementation is an internal Sol ABI and gets its own identity. Do not overload VDN provider API numbers or the public Keyless provider version.

Initial native eligibility is deliberately narrow:

- CUDA SM120;
- BF16 Q and V on one device;
- batch-equivalent H3 provider geometry, head count 56, head dimension 128;
- canonical scale `128**-0.5`;
- `routing.rope_policy == "h3_split_half_96_v1"`;
- norm epsilon exactly `1e-5`;
- explicit RoPE rows exactly aligned to V;
- no mask;
- no log measure;
- no explicit query/value/routing-position subdomain;
- no `exact_blocks` request;
- no generic callable routing preprocessor;
- no inherited optimized-attention owner below Sol.

Anything outside this first native contract uses the existing materialized reference route or another already-proven owner. Unsupported metadata is never ignored.

## 4. Coarse route preprocessing without a global route tensor

The QKV preprocessing stage must be split into a Keyless-specific path.

### 4.1 Required outputs

For each V block of up to 64 physical rows, compute:

```text
RC[b] = mean(route(V[row]))      # routed centroid used by Sol selection
VC[b] = sum(V[row])              # raw retrieval sum used by approximate PV
```

`RC` replaces current `KC`. `VC` retains its existing raw-V meaning.

The transform must occur per V row **before** the reduction:

```text
row = V[row].float()
inv = rsqrt(mean(row^2) + eps)
norm = (row * inv) * route_norm_weight
route[0:48]   = norm[0:48]  * cos - norm[48:96] * sin
route[48:96]  = norm[0:48]  * sin + norm[48:96] * cos
route[96:128] = norm[96:128]
```

The first implementation may use a dedicated Triton reduction kernel that reads raw V and writes only bounded block summaries `RC`/`VC`. It must not write `[T,H,128]` route output.

The route reduction should accumulate normalization/RoPE arithmetic in FP32 where practical, then store `RC` in the dtype required by the existing selector MMA. Any BF16 rounding point must be matched by the dense materialized oracle used for the arithmetic gate.

### 4.2 Thresholds

Existing diagonal/exact threshold code can be reused only after its input contract is renamed/reinterpreted from `KC` to routed centroid `RC` and tested against materialized `R` centroids.

Do not compare raw-V statistics against Q and call the result a Keyless route threshold.

## 5. Selected exact blocks: derive route tiles from raw V

For each selected V64 block the mainloop needs simultaneously:

- routed view for Q×R MMA;
- raw V view for softmax-probability×V MMA.

### 5.1 Correct baseline

The first correct no-global-route implementation may read the selected raw V block into two bounded shared-memory staging views if required by the current QK/PV MMA layouts, then transform only the QK staging view in place.

This can temporarily perform two global reads of the same selected V tile if the existing K-major and V-major TMA layouts make one-load conversion unsafe. That is still materially different from global route materialization because route storage is bounded to CTA/shared-memory tiles. It is an optimization target, not permission to restore a full route tensor.

### 5.2 Production optimization

After arithmetic correctness is proven, reduce duplicate raw-V traffic where practical:

1. load raw V once into a canonical shared tile;
2. preserve or expose the raw layout needed by PV;
3. derive the routed QK shared layout from that tile;
4. synchronize before QK/PV consumers read their views.

The transform requires a per-row 128-channel RMS reduction followed by the 48+48 split-half rotation. Synchronization and shared-memory lifetime must be audited against the existing single-stage K/V pipelines and Q-SMEM route scratch reuse. No transform may race the PV consumer or the epilogue reuse of Q shared memory.

## 6. RoPE and route metadata transport

The provider already receives Comfy H3 `rope_freqs` aligned to V. Native launch metadata must preserve its exact row mapping.

The native kernel may consume either:

- the existing `[1,T,1,48,2,2]` tensor/strides directly; or
- validated cos/sin views derived without changing values.

The implementation must not rebuild positions from a guessed frame/grid layout when exact rope rows are already available.

For selected domains added later, the provider must use the Keyless `RoutingSpecV1.select_value_rows(...)` semantics so V rows, rope rows and row measures are gathered once from the same physical selection. Independently selecting a pseudo-K domain is forbidden.

## 7. Routing preprocessors / Untwist

Generic Python routing preprocessors cannot be executed inside the CuTe kernel.

Initial fused eligibility therefore requires `routing.preprocessors == ()` and no inherited attention preprocessing owner. Such calls continue to use the current materialized bridge.

A later native Untwist contract may transport only bounded declarative data already proven by the reviewed implementation:

- selected reference row ranges;
- the 128-channel route scaling vector/schedule identity;
- source/runtime identity preventing double application.

That transform applies to routed values only. Retrieval V remains raw.

Unknown callable preprocessing never becomes fused merely because it preserves shape.

## 8. Sparse policy features added after the plain route

The following features are staged after plain full-domain arithmetic is proven:

1. exact/sink physical V-row block ranges;
2. mapped-neighbor intervals owned by VDN query-position provider v4;
3. key/log measure on the same selected value/routing rows;
4. rectangular query domains;
5. explicit selected value domains;
6. reviewed native Untwist route scaling.

Each addition requires its own CPU contract plus a real SM120 arithmetic/route receipt before backend history may forecast it.

The existing VDN mapped-neighbor descriptor remains an orthogonal query-position contract. Keyless does not change its API number or reinterpret its physical coordinates.

## 9. Calibration and history identity

The native Keyless path needs an independent arithmetic gate. A QKV Sol gate result is not transferable.

For each previously unseen structural Keyless execution identity:

1. run an all-selected fused Keyless call;
2. compute the dense materialized Keyless oracle from the **same Q and raw V** using `routing.materialize(V)`;
3. compare with the existing finite/mean-absolute/relative-L2/catastrophic-max framework only after confirming those thresholds are meaningful for this path;
4. retain the structural identity only after the comparison passes.

At minimum the calibration/history identity must bind:

- Keyless architecture and checkpoint/provenance semantic identity;
- block index;
- head geometry/dtype/device ABI;
- native Keyless kernel contract/version;
- route norm epsilon and immutable norm-weight owner;
- RoPE policy and physical route-layout identity;
- routing preprocessor identity (none for phase 1);
- query/value/routing-position domain identity;
- exact/sink/mapped policy when enabled;
- measure policy when enabled.

First use of a newly introduced fused route is actual-only. Spectrum may forecast only after a provider preflight identity and completed actual receipt agree.

Use a distinct receipt such as `sol_keyless_fused_v1`; never reuse `sol` materialized-route receipts for a different numerical implementation.

## 10. Source provenance

The vendored Sol source is guarded by `sol_h3/provenance.py` and `sol_manifest.json`.

Any edit under `sol_h3/_vendor/sol_attn` therefore requires:

- an explicit patch/provenance record for the Keyless changes;
- updated packaged SHA-256 values;
- a new kernel contract string identifying the Keyless-capable source tree;
- provenance tests proving modified/unexpected files fail closed;
- no mutation of the old released contract identity to mean new arithmetic.

The ordinary QKV route must remain available and byte/behavior compatible outside Keyless execution.

## 11. Implementation sequence

### Phase K0 — completed structural prerequisite

- Sol-owned public Keyless provider boundary;
- request-stable ownership;
- materialized route reference execution;
- raw-V retrieval tests;
- foreign-provider history fail-closed behavior;
- CPU/native-interop CI.

### Phase K1 — route-summary primitive

- add Keyless-only raw-V -> `(RC, VC)` block-summary primitive;
- compare RC/VC with `materialize_route(V)` + existing reduction on CUDA;
- no full route allocation in implementation or test path except the oracle;
- record temporary bytes and kernel timing.

### Phase K2 — all-selected exact tile path

- add Keyless SM120 exact Q×route(V), probability×raw-V tile execution;
- force all blocks selected;
- compare against dense materialized oracle;
- establish BF16 rounding/calibration behavior before sparse selection is enabled.

### Phase K3 — Sol selector

- feed RC into existing threshold/routing policy;
- preserve sink/exact behavior over physical V blocks;
- compare selected-block descriptors and outputs against a materialized-route Sol reference using identical policy inputs.

### Phase K4 — provider promotion

- let `_KeylessSolProviderV1` select the fused route only when every K1-K3 eligibility condition is proven;
- otherwise retain materialized fallback;
- emit distinct fused receipts and history identity;
- verify repeated calls, backend transitions, cancellation and request teardown.

### Phase K5 — additive ecosystem contracts

- mapped-neighbor;
- measure/exact ranges;
- rectangular/selected domains;
- reviewed native Untwist;
- Flow/VDN/Spectrum composition.

## 12. Required tests

CPU/structural:

- provider never aliases raw V as route;
- native eligibility rejects masks/measures/domains/preprocessors it cannot implement;
- route metadata binds exact block/provenance identity;
- foreign provider and unknown preprocessing remain actual-only;
- ordinary QKV Sol behavior is unchanged;
- provenance manifest rejects source drift;
- no persistent K or synthetic `qkv_proj` is introduced.

CUDA arithmetic:

- route-summary primitive vs materialized route reduction;
- all-selected fused output vs dense materialized oracle;
- partial-RoPE 96/128 channel boundary, including unrotated tail;
- multiple sequence lengths including non-multiples of 64;
- first/last partial V block;
- every representative early/mid/late block route-norm weight;
- cold/compiled repeated-call behavior;
- cancellation/re-entry does not retain stale descriptors.

Production SM120:

- matched materialized-vs-fused same-input arithmetic;
- decoded fixed-seed media before any speed claim;
- route tensor allocation absent from the fused path;
- measured peak VRAM and wall time;
- Spectrum NFE/history behavior with fused receipt transitions;
- Untwist/Flow/VDN combinations only after their specific native contracts land.

## 13. Explicitly rejected shortcuts

The following are implementation bugs, not acceptable temporary optimizations:

- `K = V`;
- storing a full route tensor and labeling the path fused/native;
- retaining teacher K projection weights for normal inference;
- transforming a V block centroid instead of transforming each V row before reduction;
- applying route normalization/RoPE to retrieval V in place;
- silently dropping routing preprocessors, masks, measures or row-domain metadata;
- inheriting QKV arithmetic-gate success for Keyless;
- reusing a materialized-route receipt for the fused numerical route;
- changing vendored kernel source without a new audited provenance contract.
