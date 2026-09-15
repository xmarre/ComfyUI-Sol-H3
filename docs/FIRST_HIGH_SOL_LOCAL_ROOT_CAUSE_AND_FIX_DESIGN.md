# First-high Sol-local root-cause isolation and gated fix design

Status: implementation-ready **diagnostic design**, not a production-fix authorization. Scope: the captured MiniMax-H3 first-high artifact. R and W are complete; neither is to be repeated for confidence. The precise defective arithmetic/policy is not yet causally identified. The next step is one all-selected local-Sol first-high call, with bounded same-input operator witnesses, followed only by the evidence-selected intervention below.

## 1. Evidence and authority

R capture `234ed062128e43ed8d5ec63e27517b22` reproduces the broken first-high output without retained low/probe/reload lifecycle state. Valid W native-window and native-full-support outputs are both clean. Native-window retains restricted K/V support and the existing VDN complement; therefore neither full support nor removing that complement is necessary to remove this captured artifact. The removed Sol-local path is the remaining causal boundary. This does not establish that every later-trajectory artifact has the same cause.

The preceding Flow design remains evidence: repository `xmarre/MiniMax-H3-Flow-Aligned-Regenerate`, commit `41b54405867a31e1a9e3261d4d79469a640682c4`:

| File | Git blob | SHA-256 of fetched bytes |
|---|---|---|
| `docs/FIRST_HIGH_EXPLICIT_CONTRACT_SOLUTION_DESIGN.md` | `18491dfa6088785b8e8a8132fe27d548c54faeca` | `a437594b49ce6c4ae5aad3a7d4519af5bab4b35ac802e9f573f2af64dc156b7b` |
| `docs/FIRST_HIGH_EXPLICIT_CONTRACT_SOURCE_AUDIT.json` | `4731c8369a71d949985c088261b598175b4d300d` | `3f2cda0a6e2d18c8d69d81ef227e150d6d3a4be9cfb337373021d92b1b9b28b9` |

This document selects the post-W investigation. Preserve the older architecture at `3b86aa212743e972cdd922bd9f235c0824346569`; do not reactivate its completed lifecycle experiment.

Both downloaded W reports were inspected alongside their execution reports, provenance manifests and associated full logs. Both have `w_invariants_valid=true`, matching capture/input hashes, 1 logical / 1 actual / 0 forecasts, zero upscaler calls, and exactly 700 detailed receipts: 550 local, 50 global, 100 anchor. Global/anchor providers remain native VDN. All ordinary/local/external Sol execution counters are zero. Source-delta, export and cleanup gates pass. Raw and pre-guidance MP4 bytes are identical within each arm and differ across arms. Sampled decoded frames corroborate the supplied clean-media assessment; no new full-trajectory result is claimed.

| Boundary | Required SHA-256 |
|---|---|
| Sampler-entry video | `79dca62849b2a159061a1d6204828af7a4c5995c41beec256c1719a5932a41ee` |
| Sampler-entry audio | `f5fd588101bf2ab68eb5aeeb49bfaf8e58f4c71aa99864de89cb1ddef8c859f3` |
| First H3 video input | `b49f17a317530795be43bc775486777b07d5b5dbb28996819033cdede64195c0` |
| First H3 audio input | `f54b22ff25a675c47a4c32aba54b5642033b048441ad6eec9bf5ef935cd29002` |
| Broken R raw/pre-guidance video | `73587c900e2ac2bfefa4d9163874c105655e97b1cb0493cd1451d1fc04e6dd6b` |
| W-window raw/pre-guidance video | `e2f7f0ce8ab138fa432f2982591d2acfe2d36942d0260a841d4a0c7857050134` |
| W-full raw/pre-guidance video | `630328849015f574bc913c3190225d93c3de024275d8e392e8bd562d2c4db483` |
| Block-0 sampled pre-attention QKV | `38cc6dc9bec83456a41e8ea3183b242442e8c9a8fe368185131e6f5666672936` |
| Gate configuration fingerprint | `55afe755b441274502061cbd29de1febe9e469d6047fb391a40fc0cad1b9f082` |
| Complete per-block adapter fingerprint digest | `409e9cd7cd1657493d6cd5d7fadf88e79aab726b50b508f8ab727bf562b72d61` |

## 2. Live repository state and installed provenance

State refreshed on 2026-09-15. Re-fetch before implementation; these are audited baselines, not permission to overwrite newer work.

| Repository / PR | Head | Base / relationship |
|---|---|---|
| Sol main | `f82ff2693be37dbad3438a30eb389d77136c0276` | released tree |
| Sol #9 | `fdd52bbd07bd88dfe3d31c823de0d14c27c74aff` | main; one commit |
| Sol #11 | `fff274559fe08f9b8ede6affe05c5487b2f8747e` | #9; one diagnostic commit |
| VDN main | `76b31323f9e09019b435237dcd8bad1e05476ce1` | released tree |
| VDN #14 | `d5f158a1c1750d79b37cb6ef22b0f14e6a8cbad6` | main; one commit |
| VDN #8 | `71549c02bc8e73c4c968a43b3955a3998715660d` | #14; reconciled, 91 commits |
| VDN #15 | `7d2d146c658e0ca4a90aa8a90b3ab35d78a2bbed` | #8; two diagnostic commits |
| Flow main | `970396db839ae7ab431b9718859f6d48a2e5019b` | released tree |
| Flow #33 | `9c400d46e990222cf723c426039b4c543192e185` | main; independent contract |
| Flow #35 | `a99ed7a7ca20ed3af5282c73cad4bb662e45f804` | #33 |
| Flow #36 | `cf260160df234f7a6c61787725b22eddd522c438` | #35 |
| Flow #41 | `4ae2e35f77ed961151ab5695bef9e1dbe277cc54` | #36 |
| Flow #43 | `b41a20c8b15e17fcb97329e2bbbb38e058bc3506` | #41; 15 diagnostic commits |

Flow #37 remains open diagnostic evidence at `174e0c2e884f268387897e06d4f11c0adebafacd`. #39 (`205b5b1380593de0c9501d24f9b406a9f2f42cad`) and #40 (`ff3fc7481f54ef26dde96a1f4b534ba088984028`) remain closed/unmerged. Sol #1 is merged: head `ab4833c67fca9ed437304347dba09f1fd58c2eba`, parent `5db282ca836416a32cf114346b946fe75136e4f1`, one implementation commit.

Exact diagnostic-head CI is green: Sol runs `34996134985` / `34996130917`; VDN `35007776174` / `35007492312`; Flow `35009988149` / `35009982017` / `35009705604`. #11/#15/#43 have no submitted reviews or inline review threads; their bot comments say automatic review was skipped. Sol #1 has six resolved threads; its final large review was skipped by the reviewer file limit. Historical CI and review establish neither current CUDA arithmetic nor media quality.

The W logs identify ComfyUI 0.35.0, real workstation SM120, torch 2.10.0+cu130, CUDA 13.0. The manifests record TF32 matmul disabled. Installed Git labels differ from PR heads (Sol `28e9be9...`, VDN `ae3b17b...`, VDN dirty); **do not deploy by those labels**. Exact fetched bytes match both W manifests for:

| File | Installed/fetched SHA-256 |
|---|---|
| `sol_h3/runtime.py` | `5fa136dd67b8b25df5451b3359971c5a128202d11cd12031839f29cebe75e1f7` |
| `sol_h3/interop.py` | `b9a10efceea9873ddb5a52d840e17508a070db857fff27662fa1567658973b76` |
| `sol_h3/first_high_operator_diagnostic.py` | `cff9da51615248cfed6d88c49dc0305af7f567207b6293233c57128c66879f67` |
| `vdn_h3/hybrid.py` | `eb290b17cfd566d9466030067db4acc7dd9f57248db739d1e70b5aab3b404a5c` |
| `vdn_h3/retained.py` | `39fe2a6aba50ba11a8d49c2a129f20aadf0a0f31a6c54ac47e1cf3815aaa55a0` |

`vdn_h3/retained.py`, `window.py`, `softmax_provider.py`, and `first_high_operator_diagnostic.py` were also byte-matched, not inferred from Git status. Core source gates in W verify model sampling, latent formats, model patcher and k-diffusion sampling separately. The prior source audit's earlier missing-Core caveat is superseded for these W runs.

**Remaining provenance prerequisite:** W intentionally never loads Sol sparse execution: its logs report `sol_backend=null`, `sol_source_tree_verified=false`, no arithmetic gates. Thus W proves the bridge/source delta but cannot prove the lazy-loaded packaged kernel's executed bytes. The original real-SM120 path is established by supplied R history; the next arm must additionally hash the actual loaded sparse module, manifest, preprocessing, selector, mainloop and transitive helpers, verify the packaged manifest, record module origins and compiler versions, and reject stale/multiple module identities. This is a gate to collect automatically before the new call, not a reason to rerun R/W or postpone diagnostic implementation. No current remote workstation access is assumed.

## 3. Exact production call graph and labels

Trace at the pinned heads, using functions rather than stale line numbers:

1. Sol `runtime.BlockPatch.__call__` preserves `Config.exact` affine fusion and installs VDN providers v1/v2/v3. `warmup = dense_evaluation_warmup(...) or block_index < dense_layers`. Flow's explicit high-continuation contract consumes only the default trajectory-start dense evaluation; `dense_layers=2` still preserves blocks 0 and 1 as native.
2. VDN `hybrid.make_vdn_forward().vdn_forward`: QKV projection/adapters; save raw pre-norm/pre-RoPE video inputs for the learned linear branch; Q/K normalization and RoPE; call retained grouped attention. Preserve learned weights, gates, projection, audio controls and branch state.
3. `retained.window_softmax_grouped_runtime`: apply any declared preprocessing once to full packed QKV, then `_build_window_plan`. Global and anchor query calls use full K/V and native VDN. Each local call gathers only its requested Q and builds K/V as `[global_idx; win_idx]`. `sink_rows=global_count`. Retained scratch is reused after each subcall. V3 avoids v2's square-Q allocation; legacy square payload is not consumed by Sol.
4. `softmax_provider.dispatch` selects v3; Sol `vdn_provider_v3 -> vdn_provider_v2` validates shapes/scale/sink, preserves warmup-native calls, transposes `[T,H,D] -> [1,H,T,D]`, and calls `sparse.attention(..., recompute_prefix_queries=False)`. No full-domain reconstruction or measure argument enters this route.
5. `sparse.attention`: validates BF16/B1/D128/device; uses zero-copy BTHD transpose views; loads the node-local package with `load_kernel/verify_source`. SM120 must select `cute_sm120`, not Triton fallback. New shape/stride calibration compares an all-local-keys-sink kernel call with SDPA; the actual sparse call then uses only the original prefix sink. Calibration caches are shape/stride based, not proof for every activation.
6. Vendor `interface.sol_attn -> _sol_attn_cute -> preprocess.prepare`: independent Q and K/V lengths; KC/VC summaries; Q thresholds; Q-sized output/LSE; stride-sensitive compiled cache; SM120 `make_kernel -> SolAttnForwardSm120`.
7. Mainloop scans pooled K blocks, computes token-to-centroid scores, reduces columns over valid Q rows, routes/compacts selected indices, masks selected/invalid columns out of approximation, then combines approximate and selected exact contributions in one online-softmax state. Returns Q-sized output.
8. Sol reshapes to `[Tq,H,D]`; VDN scatters into the original query slots, applies learned softmax gate and `out_proj`, then adds its separately projected linear branch on video rows. Sol LSE is not passed to VDN. The complement is not merged into Sol's softmax denominator.

`exact=true` / `exact_kernel=rounded-affine-v1` mean the separate AdaLN affine optimization in `exact.py` / `kernels.py`, preserving native intermediate rounding and disabling FP fusion. They are not a correction applied to attention. `approximate=true` means Sol owns the sparse backend. `threshold=diag` is a diagonal-covariance estimator, not a geometric diagonal mask. Public `thresh_type='exact'` computes full second moments for the threshold; it does **not** disable sparsification. `tau=1` is the paper's standardized cutoff coefficient beta, not the per-query threshold itself.

Malformed provider domains and unavailable kernels can fall back to native in ordinary production; arithmetic-gate failures propagate. Diagnostic validity must reject **any unplanned fallback**, even if its output is clean. Keep ordinary fallback behavior unchanged when diagnostics are absent.

## 4. Packaged source, history, and mathematical contract

Primary sources: uploaded **Sol-Attn 2607.24027v1**, sections 3.1/3.2, equations 4–11, Algorithm 1, Appendix B; **Sol Video Inference Engine 2606.23743v2**, sparse-attention integration and human-validation discussion only. The latter supports deployment-specific quality validation, not a new state-transport or full-stack investigation.

The actual package is `xmarre/Sana@2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn`, transformed by relative imports and `tools/rectangular_sm120.patch`. Do not confuse this with the separate delivery snapshot's Sana revision.

| Component | Verified packaged SHA-256 |
|---|---|
| `interface.py` | `cd278f6efbba842bec5684997925d7bd10746dd1f2da2758e18ac571e0666166` |
| `preprocess.py` | `97c9f5f3a9c491442b3298819153f090221305398379b55825db54e5b3aa28ec` |
| `sm120/mainloop.py` | `3568c456083fc1ebf468648628e8026507da6efcd3f256ca8cc5eccea965b2a6` |
| `sm120/kernel.py` | `03e7c3da49e2af2e75366765784fab809529ce461993068da96aedcfdefab9df` |
| `common/selector.py` | `b75eab435ec873af383a61ca2f32eb3eb75fd3fec94c30302fbfe8149f9bb0a1` |

Fetched exact upstream mainloop and preprocessing hashes match manifest upstream digests `8b6b87be7fe21cb0ebfbab34cd0ae819faa96b829c57c7b27c1c14c29e771570` and `d38c924280e00c34072652cfedf6fd65c85db6a4fe93e7858dd919e28d89d958`. These prove audited repository provenance, not the missing lazy-loaded workstation inventory.

PR #1's rectangular patch separates K/V summary count/statistics/sinks from Q thresholds/output/LSE, derives Q tail from Q rather than K, keys compilation by both lengths, and rejects unaudited rectangular backends. V3 eliminates square-Q expansion; the later zero-copy change preserves strides and keys calibration accordingly. The current v3 patch adds optional weighted exact-key bias under #9; unweighted VDN local calls pass no bias, so that specialization is inactive. Current mainloop path history contains the original integration `447b81e3c7d79b2ba1417f776e3fd5ca3d137cf4` and #9. The shared selector and rejected-block mass formula were not rectangularized by #1 because lengths alone did not alter them.

PR #1 includes a real-kernel all-selected test and `forced_route_reference` / `test_real_sm120_sink_and_approximate_kv_tail_mass`, including Q=193/KV=449 tails and sinks. The latter compares a tau=infinity forced ordinal route with a compressed-softmax reference. This is meaningful coverage of tail mass; it explicitly preserves ordinal-neighbor policy, does not test physical self-key correspondence, and does not exercise long finite-tau production routing. Historical ~1e-3 aggregate arithmetic gates cannot eliminate activation-dependent approximation error or localized corruption.

### Local-domain equations

For one local Q group, partition **only its supplied K/V** into 64-row blocks j with valid length n_j. Let c_j=mean(K_j), u_j=sum(V_j), a_rj=(q_r dot c_j)/sqrt(128). For Q block i, proxy p_ij=mean_r(a_rj). Packaged summaries are stored BF16 after accumulation; thresholds are FP32.

The diagonal estimator uses mean and diagonal covariance of local KC blocks: t_i = mean(p_i) + tau * sqrt(sum_d(qbar_id^2 * var(KC_d)) * scale^2 + epsilon), in log2 units internally. K-block statistics are uniformly block-weighted, including a partial tail centroid. This follows the block-proxy policy; it is distinct from per-key softmax multiplicity. Full-covariance `exact` is a conditional routing comparator, not a dense oracle.

Selected set S_i is the union of threshold-selected blocks, forced sink blocks, and the packaged `abs(i-j)<=1` rule. For each valid query row:

```
D_r = sum_{j in S_i} sum_{k in j} exp(q_r k / sqrt(128))
    + sum_{j not in S_i} n_j * exp(a_rj)
N_r = sum_{j in S_i} sum_{k in j} exp(q_r k / sqrt(128)) * v_k
    + sum_{j not in S_i} exp(a_rj) * u_j
O_r = N_r / D_r
```

VC is a **sum**, so the approximate numerator must not receive another factor n_j. The denominator must receive n_j, not Tq, packed-domain length, 64 for every tail, or route-chunk size. Selected blocks have zero approximate contribution. Exact and approximate updates rescale the same accumulated numerator and denominator when the running maximum changes. BF16 probability conversion/PV and approximate exponent/reciprocal instructions introduce numerical error even when every block is selected.

SM120 uses physical Q/K tiles 64x64 and routing chunks of 64 **centroids** (up to 4096 keys). K tail masking, n_j and KC/VC reduction are local-KV-owned. Q tail reduction/stores are Q-owned. Tensor descriptors mask loads outside actual tensor shapes; no explicit compile-bucket padding is allowed on SM120. Slice backing capacity must never be mistaken for logical rows. Check descriptor/tail behavior on real CUDA rather than assuming zero-fill from labels.

Changing routing chunk C with fixed 64-row grouping, thresholds and selected set should change only reduction roundoff/performance. Changing the underlying K block partition or Q pooling partition changes approximation/routing and is not a mathematical invariant. Arbitrary gathered-domain reordering is therefore not an innocuous experiment. Non-video globals are exact sinks; outward rounding may promote adjacent video keys, but never adds rows. Local gathered video/anchor boundaries may lie inside tiles and produce heterogeneous rejected blocks; the zeroth-order error depends on within-block score spread and value correlation (paper Appendix B), not just proxy magnitude.

### Nested-domain assessment and surviving square assumption

No inspected local-Sol calculation imports full packed-domain mass. KC/VC, threshold statistics, selected traversal, n_j, exact tails and sink bounds are all based on supplied local K/V. VDN's learned outside-window branch is separately projected and added; there is no shared normalization or obvious duplicate complement term. This rules out the specific source hypothesis of using packed T or Q length as local rejected mass in this path; CUDA descriptor errors and approximation interacting with the trained complement remain unresolved.

There is a concrete inherited policy mismatch: `common.selector.sol_attn_route_is_exact` forces `abs(q_block-kv_block)<=1`, while V3 discards square-Q/query-position information. A local Q block's ordinal is not its position in `[globals; gathered window]`. With 3101 globals and 1024 rows/frame, even the first queried video row is offset in K/V; later groups and anchor gaps add offsets. The first 49 K blocks are sinks, so some wrongly forced neighbors are already sinks while true self/near keys are not guaranteed exact. This is a **verified coordinate-policy mismatch**, not yet a demonstrated cause of the visible artifact. Reconstructing the saved 52-frame geometry gives local-K positions 4125–8220 for group 0 Q, 8221–13340 for group 1, 9245–14364 for groups 2–9, and 9245–10268 for group 10. Across all 800 Q64 tiles per layer, the old ordinal-neighbor-plus-sink rule does not guarantee all corresponding self-key blocks; threshold selection may still select them. These counts describe geometry, not measured sparse ballots. Do not remove the neighbor rule or impose a new geometric interpretation in production without the conditional mapped-route experiment.

## 5. Minimal causal experiment E

Use the existing R-bundle/one-first-high-call machinery, preserving normal Core cleanup and callback-x0 externalization. Introduce a separately versioned diagnostic request/report (suggested `h3_first_high_sol_local_diagnostic_v1`), never rename a Sol arm `native_window` or claim `w_invariants_valid`. Existing W behavior and reports remain unchanged.

**E: all-selected Sol on production-eligible local calls.** The sole returned-operator intervention is promotion of every valid local K block to exact for calls that production would route to Sol. Keep the actual K/V tensor, order, strides, original sink metadata, Q groups, tau=1, diagonal preprocessing, scale, gates, adapters, complement, global/anchor routes and affine fusion unchanged. Pass a diagnostic sink override `sink_start=0, sink_tokens=Tkv` only at the final kernel invocation. Keep `recompute_prefix_queries=False`.

Proof: sink range becomes [0,ceil(Tkv/64)); every valid ballot bit is selected, selected columns are excluded from approximation, `exact_count==valid_blocks`, `has_approx=false` for every route group, and every valid key contributes once through the exact branch. No threshold choice is needed. Do not use tau=0, tau=infinity, `Config.exact`, or threshold='exact' as substitutes.

The replay preserves Flow's high-continuation marker. Expected returned attention receipts are **700 total: 528 `vdn_local_sol_all_selected_e`, 22 native dense-layer local, 50 native global, 100 native anchor**. Blocks 0/1 remain native; blocks 2–49 have 11 Sol calls each. Expect zero ordinary/external/weighted Sol calls, zero square expansion, and no additional dense warmup. The new diagnostic route must not fabricate production counters; distinguish returned calls, calibration calls and sidecar kernel calls. Each arm is 1L/1A/0F and zero upscaler calls; additional attention calculations below add **zero H3 evaluations**.

Per-block geometry from the saved window receipts:

| Group | Q rows | K/V rows | K blocks | Tail keys |
|---|---:|---:|---:|---:|
| 0 | 4096 | 14365 | 225 | 29 |
| 1 | 5120 | 19485 | 305 | 29 |
| 2–8 | 5120 | 20509 | 321 | 29 |
| 9 | 5120 | 16413 | 257 | 29 |
| 10 | 1024 | 11293 | 177 | 29 |

All have B1/H56/D128, sink rows 3101 (49 outward-rounded blocks, 35 additional video rows in the boundary block). Packed rows 56349; video span [3101,56349); global Q=3101, anchor Q=1024 each. Preserve the complete suffix `[0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]`; stop after the first callback, not by changing sigmas.

### Same-input witnesses inside E

At block **2**, groups **0, 2, 10**, before scratch reuse, compare E's all-selected output with native SDPA and the **unchanged** production Sol operator on the same tensors. Block 2 is the first eligible sparse layer; preceding layers stay native and no earlier local intervention has changed these inputs. These are non-returned sidecars, not additional model calls or another W run. Save the full Q/K/V witness once, row-index maps, shape/stride/storage-offset metadata, original sink, KC/VC/thresholds, and outputs to CPU before reuse. Report SHA-256 for full logical tensor bytes and geometry. Capture all heads; do not replace the kernel invocation with sliced Q because that changes Q block ordinals and pooling.

Run the existing `debug_route_trace=True` specialization only on these witnesses, using a distinct compiled-cache namespace and trace buffer. It replaces LSE storage with ballots; it is a diagnostic specialization, not the ordinary production binary. Require its output to match the ordinary invocation before trusting its ballots. Ordinary production/all-selected calls remain the authoritative returned outputs. Expected sidecars: three native calls and three ordinary sparse calls; three sparse-route traces and three all-selected traces. None produces an additional backend-history completion receipt. Trace E has every valid bit set and no invalid bit; sparse trace counts are measured, never invented. Exact all-selected selected-block-pair count is H*ceil(Tq/64)*ceil(Tkv/64) per call.

Then, using saved witnesses **without H3**, evaluate a streaming FP32 reference for the mixed operator using the recorded production route and the actual BF16 summaries. Compare numerator/denominator/LSE as well as output. Separately compute independent summaries and diagonal thresholds, including FP32 accumulation followed by the same BF16 storage. Compare route column means, thresholds, margins and forced-bit reasons. This divides Python preparation, selector/reduction, and fused mixed accumulation without another media arm. Never compute each block's normalized attention and average outputs: combine unnormalized contributions with a common stable maximum. If a frozen-route mixed mismatch survives independent preparation, instrument only the first mismatching head/Q64 tile on the saved witness: record row maxima, lane-partial denominator and unnormalized output accumulator immediately before/after the approximate update and each selected exact update, with route-group/block IDs. Compare against the corresponding stable FP32 recurrence after the same lane reduction. This distinguishes rejected-branch mass/PV error, selected-branch arithmetic, and inter-branch rescaling without another H3 call. Keep producer/consumer and prefetch order intact; do not skip a pipeline stage to make an exact-only ablation. Bound snapshots to one head/tile and stop at the first divergence; verify instrumented versus ordinary final output before interpreting intermediate state.

Memory: baseline retained buffers stay unchanged. Maximum group QKV logical bytes are `2*56*128*(5120+2*20509)` = about 0.62 GiB; a Q-sized BF16 output is 70 MiB. Three saved CPU QKV witnesses total about 1.37 GiB. Transfer/release sequentially; do not retain three CUDA clones. Streaming reference chunks must cap live score storage (e.g. 64 Q rows, 1024 keys, one head: 256 KiB FP32 scores), never materialize full QxKVxH scores. Record actual peak allocation; permit a configurable diagnostic budget with default 2 GiB additional CUDA and 2 GiB CPU witness storage, failing before allocation if insufficient. Compilation and sidecar time are excluded from performance conclusions. The full R bundle's existing memory is separate.

### Validity and exactness boundaries

All entry hashes and fingerprints in section 1 must match; verify every saved per-block adapter entry, not just aggregate presence. Require exact logical Q/K/V equality across sidecars before and after execution, exact index-map/scale/sink equality, finite output and normalization, one use of every local output, unchanged native/global/anchor paths, and normal cleanup. The existing block-0 QKV digest samples 128 values and is not a full-tensor identity proof; add full witness digests.

Do not require later-block activations or gate **values** to equal W: returned attention changes propagate normally. Gate weights/configuration, adapter state and operations remain fixed. Do not require audio output to equal R/W when changed video attention can influence it later in the same transformer; preserve audio inputs/carry/ownership and measure audio output/regression. Do not relabel intended downstream changes as validity failures.

Reject unexpected warmup, backend fallback, missing/duplicate receipts, wrong source bytes, wrong compiler/kernel backend, changed checkpoint/conditioning, nonfinite tensors, mutated inputs, external/weighted/Untwist activity, extra H3/upscaler calls, export mismatch, incomplete cleanup, or trace/output disagreement. An arithmetic mismatch with an otherwise valid real kernel is **diagnostic evidence**, not a reason to silently fall back. Separate execution-valid, arithmetic-conformant, and media-clean booleans. Preserve the production arithmetic gate unchanged; if it aborts E, keep the saved witness and investigate operator-only before spending another H3 call.

Use existing aggregate calibration limits (mean absolute <=0.002 and relative L2 <=0.005) only as baseline alarms. Also report per-head/per-Q64 errors, maximum/p99 absolute error and offending row/block coordinates, denominator relative error and route margins. No aggregate pass alone establishes media correctness; trace mismatches far from numerical thresholds require diagnosis even when aggregate error is small. Near-threshold flips must be identified as such, not hidden by a new tolerance.

## 6. Conditional decision table and fix ownership

Run only the branch selected by E and its witnesses. No combinatorial media matrix.

| Result | Interpretation and next minimum action | Likely owner |
|---|---|---|
| E invalid | No causal interpretation; correct the specific diagnostic contract defect, then rerun only invalid E | diagnostic owner |
| All-selected differs materially from same-input SDPA/reference | Exact arithmetic, descriptors, strides, masking or normalization unresolved. On saved witnesses compare exact same values through contiguous versus original-stride invocation; compare prepare summaries and tails. No new H3 call until localized | packaged SM120 or bridge |
| E broken but all-selected witness checks pass | Witnesses are insufficient or accumulated numerical drift matters. Collect error summaries at the first later divergent layer during one bounded E extension; do not infer routing innocence or restart lifecycle research | unresolved local path |
| E clean; sparse differs from independent mixed reference with identical recorded routes/summaries | Fused mixed arithmetic, compaction, tail masks, rescaling or PV implementation defect. Isolate first mismatching route group on saved witness; do not alter threshold policy | packaged SM120 |
| E clean; KC/VC/threshold differs from independent preparation | Preparation/domain/dtype/stride defect. Identify first wrong summary or threshold and its exact rows | packaged preprocessing or Sol bridge |
| E clean; finite-tau ballots disagree with independent threshold rule | Route reduction/selection error; distinguish numeric-margin flips from wrong indices/means. Compare ordinary/debug output first | selector / SM120 reduction |
| E clean; sparse conforms to its mixed operator | Approximation policy error remains. Quantify rejected-block exact versus approximate N/D and missing physical-neighbor blocks before selecting one route intervention | policy / shared mapping contract |

For conformant sparse arithmetic, prioritize the verified coordinate mismatch **only if witnesses show it actually leaves self/near blocks rejected with material N/D error**. Reconstruct `query_positions = searchsorted(win_idx,q_idx)+global_count` from VDN's own exact plan. For each requested Q tile, map its represented queries to local K blocks; union their corresponding blocks and immediate local-K block neighbors, clamped to valid K blocks. This explicitly restores the existing local ordinal-neighbor intent in K coordinates; do not claim it is a new physical-distance metric across gathered gaps.

**M (conditional media arm):** keep original threshold, sink and ordinary selected set; add only those missing mapped-neighbor blocks to the exact set. Do not remove previously selected blocks. This single additive intervention preserves approximation everywhere else and avoids a sparsity-increase confound. Use real SM120 with a reviewed bounded per-Q-tile forced-block descriptor, unchanged Q/K/V, 528 Sol local +22 native local+150 nonlocal receipts, 1L/1A/0F, zero upscaler calls. Record added blocks, route reasons, exact-work increase, and resulting media. Reject if descriptor changes existing threshold decisions or expands K/V. Metadata is bounded by O(number of Q blocks) intervals for contiguous groups; general maps require a validated bounded representation, not a dense QxKV mask.

If M is clean with matched invariants and retained acceleration, the supported fix is a narrow VDN-to-Sol query-position contract and SM120 forced-route mapping. Version provider capability rather than reinterpreting v3 or reintroducing square Q. Preserve ordinary square callers and old consumers; if required mapping is unavailable, report that narrow capability failure with explicit native fallback for that call. Do not silently approximate a missing map. Keep the first proven fix additive; removing accidental old neighbors is a separate performance change requiring evidence.

If mapped neighbors are already selected or carry negligible error, **do not run M**. If independent full-covariance thresholds select the responsible omitted blocks and same-input reference errors improve materially, run **T** instead: change only `thresh_type='diag'` to `'exact'` through the diagnostic kernel call; keep tau/sinks/domains/complement/dense layers fixed. Same counts and one H3 call; real production kernel with different preprocessing. Full covariance uses additional O(H*D^2) storage, not QxKV storage. A clean T supports threshold-estimator policy ownership only after measured cost and finite-tau route parity; it does not prove a general diagonal estimator bug.

If neither mapping nor estimator explains error, identify the dominant rejected blocks via exact-versus-approximate N/D on witnesses. A single **C** arm may promote only an independently specified offending class (e.g. partial tail or a measured heterogeneous block class) while leaving all other routing intact. Specify that class and evidence in a checkpointed amendment **before** CUDA. Merely disabling correction would also change normalization and discard support; it cannot identify a correction implementation bug by itself and is not an authorized default arm. A second-order/heterogeneity-aware approximation is research until its math, bounded cost and causal benefit are established; do not silently authorize it here.

A source-local arithmetic fix first passes saved-witness validation, then one one-call media confirmation at the original production tau/sink policy. A clean all-selected E alone never authorizes permanently dense local attention. If sparse arithmetic is conformant but no bounded policy intervention resolves quality, report the remaining approximation-quality limitation and keep promotion blocked rather than claiming an implementation bug.

## 7. Concrete implementation plan and isolation

1. Work on separate implementation mirrors from the refreshed effective stacks. Keep #43/#15/#11 as immutable diagnostic evidence during development; no replacement PRs and no consolidation into production by implication. This design branch is `mirror/first-high-sol-local-causal-design-20260915` in Sol-H3 and adds documentation only.
2. Add a sibling Flow diagnostic module for the new request/report; reuse reviewed R loading, `_sampler_entry_wrapper`, `_outer_wrapper`, `_FirstCallComplete`, `_ReceiptSink`, export and cleanup behavior from `first_high_operator_comparison.py`. Preserve source-byte gates, exact namespaced loader resolution, clone-stable receipt ownership, full schedule, and wrapper-order normalization. Prefer a small explicit extension seam over copying a divergent sampling implementation; prove W request/output behavior unchanged if any shared helper moves.
3. VDN owns plan/gather/witness metadata. Add a request-scoped sibling dispatch overlay using the existing grouped geometry; preserve normal `make_vdn_forward`, raw complement buffers, preprocessing, native dense layers, global/anchor dispatch and final projection. Do not use `_FullSupportStateProxy` for E. Keep original W parser/routes untouched. Never retain reused `k_scratch/v_scratch` views after return without a completed CPU copy or owned clone.
4. Sol owns eligible/native selection, all-selected override, real-kernel execution identity, and completion receipts. Add a sibling diagnostic module and a narrowly guarded seam in `BlockPatch`/`sparse.attention`; leave production config/affine execution unchanged. Record original and effective sink ranges separately. Add new history identity before Spectrum preflight; never normalize E/M/T routes into native W or forecast-safe production identity.
5. Implement operator-only reference/witness tooling outside production modules. Reuse `forced_route_reference` mathematics, generalized to recorded finite-tau routes and streaming N/D. Keep original/full-domain logical Q indices for tracing; never silently reset tile ordinals when slicing. Preserve FP32 reference computation with TF32 disabled and separate BF16-emulated comparison.
6. Diagnostic trace launch uses existing SM120 debug capability; changes to its launcher must not enter production compiled-cache keys. If later arithmetic instrumentation is necessary, isolate it as a diagnostic specialization and require output equivalence before trusting results. Update manifests/packaging transformation records for any reviewed packaged-source modification; never bypass `verify_source` or reuse an old numerical contract identity for a changed operator.
7. Implement only E and witness/reference tooling initially. Implement M/T/C or a production patch after applying the decision table. Checkpoint evidence interpretation and precise selected intervention before its CUDA execution.

## 8. Validation, compatibility and rollback

Meaningful CPU/reference tests: unequal Q/KV counts; Q/K tails 1/63/64/65 and production K remainder 29; sink 3101 rounding; constant-value preservation; single exact accounting for sinks/selected blocks; selected-all equivalence; partial-block centroid divisor and V sum; finite-tau diagonal/full-covariance statistics; chunk-order/chunk-size invariance with frozen routing; index maps spanning prefix, anchors and gathered gaps. Explicitly test that covariance off-diagonal terms may change thresholds. Do not assert arbitrary block repartition invariance.

Source/host tests: no diagnostic request gives original behavior; E cannot enter full-support mode or change Config; 22 native/528 Sol expected ownership; fail closed on malformed/duplicated request, extra callbacks, loader ambiguity, stale source, clone/reentrancy/cleanup errors and unexpected fallback. Trace sidecars cannot alter counters, history, RNG, QKV, retained buffers or compiled production dispatch. Restore context tokens/options in `finally`; reject overlapping diagnostic owners. Preserve W's report schema and existing tests.

Packaged-kernel/real-SM120 tests: run targeted existing rectangular/all-selected/tail/zero-copy tests plus finite-tau route/reference tests on saved real witnesses. Test noncontiguous backing buffers, more than one 64-centroid route group, 29-key tail, all-selected, mostly-rejected and mixed exact/approximate groups, and forced sink boundary. No GPU result can be replaced by a mocked CPU launch or another GPU architecture. Existing tests were inspected; new CUDA validation was not executed for this design.

Media gate: valid E then only the selected branch; decode raw and pre-guidance with the same VAE and compare original defect regions/time slices with saved R/W. Require user review of complete clips; scalar error, hashes, CI, sampled frames and source correctness are insufficient substitutes. Do not expose diagnostic internal details in ordinary product UI.

After the local cause/fix is proven: matched full-production trajectory with unchanged seed, checkpoint, references, noise, sigma schedule, audio carry, geometry, VDN gates/complement and Spectrum settings. Verify original 18L/14A/4F when that captured production workflow is used, but derive counters from its actual runtime rather than enforcing that ratio on arbitrary workflows. Check first/later chunks, motion, framing, audio continuity, protected prefixes, no extra H3 calls, route ownership, hot timing and peak VRAM. Benchmark the actual targeted hot path once sufficiently warmed; do not infer speed from diagnostic sidecars.

Promotion requires causal media benefit, exact affected operator/contract identification, targeted math/CUDA tests, retained acceleration outside and within the applicable route, and compatibility with untouched stacked work. Keep a clean minimal production commit separate from diagnostic infrastructure. Select its base/companion PR only after ownership is known; do not pull weighted research or W overlays into main just to land a local fix.

Rollback: disable/remove only the new diagnostic request to restore the original path; preserve R/W artifacts and all existing PR heads. A production candidate stays opt-in/unpromoted until gates pass. If promoted, version its backend-history identity and clear only incompatible request/compile state through ordinary ownership boundaries. Roll back the specific candidate, not Flow transport, VDN support/complement or all Sol acceleration. Never rewrite the user's workstation branch topology or delete evidence to obtain a clean provenance report.

## 9. Rejected and unresolved hypotheses

| Hypothesis / approach | Status and scope |
|---|---|
| Retained low/probe/reload lifecycle state is necessary | Falsified by R for this captured first-high output |
| Restricted VDN local support alone causes the artifact | Falsified by valid clean W-window with unchanged support/complement |
| Full K/V support is necessary for cleanliness | Falsified by W-window |
| Broad stochastic/state-transport rework; resurrect #39/#40 | Unsupported after R/W; prohibited as fallback |
| Force full attention or broadly disable Sol | Already diagnostic boundary; not a root-cause fix |
| Add dense first-high warmup / extra H3 calls / change Spectrum | No causal support; prohibited shortcut |
| Weighted Mixed-Grid research | Absent from controlled call; keep out |
| SM120 arithmetic / descriptors / partial-tail handling | Unresolved on captured tensors despite existing tests |
| Rectangular index preparation / ordinal neighbor policy | Coordinate mismatch verified; media causality unresolved |
| Threshold estimator / sparse selection / rejected correction | Unresolved; divided by E and same-input witnesses |
| Complement interaction | No shared-mass leak found; trained complement can still interact with local approximation error |

## 10. Artifacts to preserve and assumptions to re-check

Preserve workstation R files **in place**:

```
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json
/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.pt
```

Their whole-file hashes were not available in this design workspace; do not invent them or confuse tensor hashes with file hashes. The new loader must verify the saved bundle's own integrity and record whole-file hashes without overwriting/regenerating it.

The following exact File Library files were retrieved and inspected. Preserve all; none needs reproduction. These are file-byte SHA-256 values, distinct from latent hashes above:

| Filename | SHA-256 |
|---|---|
| `W_report_00002.json` | `4ef2d1b6ecacb18a3f669246fc5a52cd5f62a0aa22dafab1372cc0ff86fea71c` |
| `executioncontractreport_00014.json` | `958884ed1ee2de832664e55d664561a79f6a47e5745259373761a193377c8c7d` |
| `preflightprovenancemanifest_00026.json` | `8a3e245a2c81a012b72d2128b6ea918694e26a6ae72732bcaaf062668d810acd` |
| `first_high_model_raw_00002.mp4` and `first_high_pre_guidance_00002.mp4` | `3449374d5be5fd69101b38a56322cbe99d1621d0561e2018a88b8c6cb5490d49` |
| `MiniMax_H3_00003-audio(1).mp4` | `6d40aac8dc330efd5ed8695bffd32042bf47879bdcba0dc001fc067e2f835720` |
| `Pasted text(20260915-192858).txt` (window log) | `4eee09c32ec476d42d3e0bb3d6ac809f3f6b54302b8f90e881805aa9c27053a4` |
| `W_report_00003.json` | `de25cc7232cbb2a4c826002e3f82e4dbc02da55fac415a6a075c31028d128a2d` |
| `executioncontractreport_00015.json` | `275a4095eb81b7272842ab1401d484418e092a1b4045f466766124b5e9304478` |
| `preflightprovenancemanifest_00028.json` | `2e0f137b02c77ae0ef99ec33c91e808ca5a9970f1286ce7c10f43a6e6be1a901` |
| `first_high_model_raw_00003.mp4` and `first_high_pre_guidance_00003.mp4` | `43907eda9ac03f842ad2eb97b313c161382f356ba267fda0cc22886324aed504` |
| `MiniMax_H3_00004-audio(2).mp4` | `de20d4a4cb39c9cc2fb913661c44fe566cb30e9c262b8cdfaf8ecadd2be234b7` |
| `Pasted text(20260915-194544).txt` (full-support log) | `570ec7e6c5e35bcbe46025464d5ff18542bac1f61fa0249ab0a345dbad6a2201` |

Local fetched source copies, extracted PDF text and sampled PNGs are reproducible scratch, not authoritative deployment artifacts. No exploratory production code or uncommitted implementation is required by this design. Preserve future E witness tensor bundles, route bitsets, source inventories and media before CUDA process exit; store their hashes and interpretation on the investigation mirror. Do not leave expensive evidence only in scratch.

Before editing or executing: re-fetch main/PRs/reviews/CI; check for applicable repository instructions; re-check effective overlay bytes including lazy kernel modules, actual ComfyUI conversions/cleanup, checkpoint/adapters, dense-layer ownership, high-continuation marker, native SDPA provider, full schedule, and geometry. If source changed, adapt this design explicitly with source evidence rather than copying stale file/line details. A mismatch is a reason to reconcile the precise contract, never a license to relax validity or return to rejected broad fixes.

The implementation can begin with E and operator witnesses now. Production-fix ownership remains gated on their real SM120 CUDA/media results. There is no justified production patch yet.
