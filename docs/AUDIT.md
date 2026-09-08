# Interoperability source audit — 2026-09-08

Existing PR #1 is continued on a mirror branch. This audit supersedes both the initial exclusive-provider design and the later requirement for a separately exposed Sana `sol_attn` checkout.

| Source | Inspected/tested revision |
|---|---|
| ComfyUI | `efa6c8f804bff78b46a0fd458ebd2e47bba07a30` |
| Sana `sol-engine` | `2936c47637380842aaa4a4488fac5006cc542b70`, `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn` |
| KJNodes | `c9869eade9920a1b949de07c4a197156006bcceb` |
| VDN-H3-Plus companion | sequential PR #8 -> PR #11 stack; PR #11 head `99ff31128d66efa508873f5072f8de88c677e55e` |
| Spectrum companion | `31de0cb89b472c965794fdc340de857360a58107` |
| Flow | `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591` |
| Continuum | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| Untwist | PR #9 preprocessing contract |
| DiffAid | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` |

## Production reproductions that changed the design

A correctly ordered production stack — VDN, runtime DoRA, DiffAid, Untwist, SOL, Spectrum, then preview/progressive/Continuum — ran on the RTX PRO 6000 with Sage enabled. Exact fusion executed, but the first interoperability revision still reported `sol_eligible_calls=0`, `sparse_calls=0`, `vdn_local_sol_calls=0`. Mixed-grid calls reported the expected `external_sequence_native` fallback.

That ruled out node ordering, dense warmup and tau. The deeper limitation was VDN provider API v1: normal H3 VDN local queries attend to an already-restricted KV domain that includes prefix/global rows, so the requested query set is usually rectangular against K/V while the SOL kernel requires square Q/K/V.

VDN PR #11 therefore exposes provider API v2. For each local operation VDN still constructs its original restricted KV domain, then supplies an expanded query tensor over that same ordered domain plus a mapping back to the originally requested query rows. SOL may evaluate that square restricted domain and VDN keeps only the original query outputs. The operation does not gain KV rows and therefore does not become unrestricted full-sequence attention. Global/anchor operations remain native, masked Flex remains native, and VDN retains its learned gate and linear complement.

The post-v2 handoff reports `sol_eligible_calls=2750` but zero sparse execution because the source loader required an external package. This establishes routing only. The subsequent comfy-kitchen substitution is replaced here by the packaged Sana implementation. A ComfyUI startup capability report for comfy-kitchen does not prove Sana CuTe availability.

## Source, licensing and dependencies

The entire 51-file Sol-Attn subtree is retained to preserve its public architecture dispatch and internal support imports; no other engine code is imported. `interface.py`, `preprocess.py`, `common/`, SM120 mainloop and the upstream H3 sparse policy were inspected. `SOURCE_SNAPSHOT.json` identifies upstream validation on SM103 and explicitly lists SM120 as unvalidated; a copy is packaged for provenance.

Sana's pinned root README licenses repository code under Apache-2.0. The subtree's third-party notice identifies FlashAttention BSD-3-Clause and NVIDIA Apache-2.0 adaptations. Copyright/SPDX headers and the original third-party notice/BSD license are retained; the Apache-2.0 text is included. Modified files identify the sole change: absolute `sol_attn.*` imports become relative imports. This avoids global module aliases, path mutations and collisions with other nodes, including when ComfyUI imports the node beneath a generated package name. The manifest records original and transformed hashes, verified before GPU runtime import.

Imports require PyTorch, Triton (also used by CuTe preprocessing), CUDA Python, CUTLASS DSL and TVM FFI (`enable_tvm_ffi=True`). The pinned cuDNN frontend's `cutedsl` dependency metadata declares `nvidia-cutlass-dsl[cu13]>=4.5.0`, `cuda-python`, `apache-tvm-ffi>=0.1.11`; these dependencies are declared directly. No `cudnn` import exists in the packaged subtree. Upstream pins Triton 3.6.0; the node declares >=3.6,<4 so installation does not force a newer PyTorch environment to downgrade. CI retains the upstream 3.6.0 baseline; newer versions require GPU validation. No diffusers, transformers, model downloader or full cuDNN frontend dependency is added.

Upstream maps SM120 to `cute_sm120` only when CuTe imports succeed; otherwise it selects Triton. This node explicitly requires `cute_sm120` on its existing SM120 integration path and records initialization failure as native fallback. It does not mislabel Triton execution as CuTe. Other architectures remain available in the packaged public source but retain the existing node-level per-call native fallback.

## Compatibility-gate disposition

| Former rejection or invariant | Classification | Current behavior |
|---|---|---|
| Provider-name bans | Unvalidated/composable ownership | Removed; per-call dispatch/telemetry |
| SOL + Spectrum blanket gate | Numerical-history ownership | Consumer preflight/receipt protocol; incompatible history becomes actual/reset rather than RuntimeError |
| Generic optimized override including Sage/Sage3 | Composable dense provider | Retained as inherited dense provider, recursion marker preserved |
| `_vdn_forward` | Multiple operation ownership | Explicit VDN restricted-subcall v1/v2 hooks; learned gate/linear math retained |
| Rectangular VDN local Q vs KV | Kernel topology mismatch with coherent bridge | VDN v2 expands Q only over the same restricted KV domain, runs square SOL when eligible, then selects original query rows |
| VDN external sequence API 1/2 | Per-call unsupported topology | Inherited gated attention for that call; later normal grid can resume SOL |
| Preexisting block replacement | Composable ownership | Wrap/delegate; attention replacement remains its own operation |
| Repeated block / zero or multiple sparse subcalls | Legitimate provider execution | Counters and routing receipts, no exactly-one enforcement |
| Zero sparse / zero exact post-run checks | Optimization-usage policy | Removed; report counters |
| Dense layers covering all blocks | Valid all-dense configuration | Allowed |
| Duplicate Sol-H3 application | Composable configuration | Compatible exact/SOL requests merge; one keyed lifecycle |
| Different SOL policies on same branch | Ambiguous configuration | Explicit ValueError; use separate branches |
| Current PackedLayout/prefix unavailable or unsupported | Per-call kernel topology | Native fallback; no cached layout inference |
| Mask / attention flags / precision / dtype / device unsupported | Per-call kernel topology | Inherited attention |
| Whole-block forward hooks, compiled/custom block forward | Exact arithmetic ownership | Preserve native block invocation |
| Native source hashes drift | Exact implementation unverified | Native block fallback |
| Packaged Sana source/CuTe unavailable for the current GPU | Optional optimization unavailable | Native fallback with explicit telemetry; no external package search/download |
| Arithmetic gate actually fails | Computed output violates tested invariant | Hard failure; never silently accept failed arithmetic |
| Missing request scope / tampered metadata / cyclic transform | Broken internal lifecycle/contract | Hard failure |
| Malformed affine indices/strides | Unsafe fused indexing | Existing hard invariant retained |

## Numerical and ownership details

Comfy's `wrap_attn` sets `_inside_attn_wrapper` before invoking the override and materializes `AttentionTensorContainer` inputs when the override has no container implementation. SOL accepts that tensor contract and retains the marker on inherited dense calls, including providers that themselves use wrapped fallbacks. It does not special-case Sage names.

Independent BF16 SDPA verifies the all-selected SOL kernel. Dense-prefix execution on ordinary native H3 attention uses the inherited provider and may therefore be approximate Sage. Pure `attention_preprocess_v1` transformations expose their inherited provider: SOL transforms QKV once and uses the dense leaf on transformed prefix queries.

The direct kernel call uses the packaged upstream `sol_attn` API on BF16 BTHD tensors with explicit native scale, direct tau, `thresh_type="diag"` and `kv_splits=1`, matching H3's upstream default policy. Sol-H3 always passes `sink_start=0`: upstream's omitted-start default is a suffix. Upstream `_sink_block_range` rounds the prefix end outward to the next 64-row boundary. Prefix queries remain dense on normal H3; the all-selected gate uses a full-length exact sink.

VDN v2 preprocessing happens on the full post-RoPE VDN domain before local gather, so transforms such as Untwist retain their original packed-row coordinates. For an expanded VDN local SOL call, prefix/global rows remain sink KV. Their expanded query outputs are auxiliary and discarded, so SOL does not perform the ordinary native-H3 dense prefix-query recomputation for those unused rows.

VDN's square expansion has a real performance cost: it computes additional query rows and may perform additional gathers. Telemetry separately records `vdn_square_expanded_calls`, requested query rows and kernel rows so GPU tests can determine whether the sparse work still wins. Correct execution is established structurally; speedup is not.

Spectrum decides whether to forecast before executing H3, but captures anchors inside the final block replacement. Preflight therefore establishes expected policy and actual per-block receipts are consumed before `observe_actual`. Unknown/opaque routing is actual-only. Backend transitions reset stage forecasters/controllers and invalidate incompatible offline archives rather than aborting execution.

Flow can change layout, explicit RoPE rows and modulation rows through block callbacks. Exact uses current delegated arguments. SOL validates call-time row counts; stale or unsupported layouts fall back rather than inventing a lattice. Continuum's separate sampler invocations receive fresh request state.

Projection/norm/MLP hooks remain invoked, preserving runtime adapters and ordinary Comfy weight patches structurally. Whole-block hooks force native Exact execution. Curve AdaLN remains native; no table eviction or stale projection cache is introduced.

## Optimization decisions

| Sol-H3 component | Classification and decision |
|---|---|
| Fused QKV projection | Already native in `Attention.qkv_proj`; retained |
| Q/K RMSNorm + partial split-half RoPE | Already native `ck.rms_rope_split_half_`; retained, including quantized ops |
| SwiGLU + down projection | Already native `linear_input_act`; retained |
| Residual/gate | Native in-place `addcmul_` already fused per segment; retained |
| Scale/shift modulation | Exact candidate: fused native add/multiply/add launches while preserving dtype rounding |
| RMSNorm + modulation | Not fused here: native RMSNorm implementations/hooks and reduction order are retained |
| AdaLN schedule tables | Compact curve format already removes the large full-width table cost; no fixed schedule cache is introduced |
| Layout/copy elimination | Native packed target-video tail is already contiguous; direct SOL uses current contiguous BTHD Q/K/V |
| Compilation/caches | Triton specializes the small affine kernel; no whole-model compile, CUDA graph capture or process-global activation cache |
| Sol-Attn numerical kernel | Uses packaged Sana `sol_attn` / CuTe SM120; approximate, opt-in, complete prefix KV sink |
| VDN v2 square bridge | Preserves VDN restricted KV geometry while adapting rectangular local calls to the square kernel; extra-Q overhead must be measured |
| ComfyUI Block Sparse Attention node | Separate kernel integration and H3 producer/policy (`sol_attn_chunked`, pooled state, Sol/SLA/VSA options); retained as a separate A/B path |
| SOL-BSA + residual | Approximate; not exposed without supported validation |
| FastH3/DMD2/VSA adapter | Learned acceleration, separate concern; not installed or required |
| Ulysses / distributed execution / parallel VAE | Runtime-specific, outside this native MODEL patch; omitted |

## ComfyUI built-in sparse attention

Core BSA uses comfy-kitchen, including its chunked H3 QKV producer. Sol-H3 uses the pinned Sana source at the native attention/provider boundary. These are separate integrations; neither one's availability or benchmark establishes correctness/performance of the other.

## Evidence limits

CPU CI exercises real ModelPatcher object-patch application, current KJ wrapper behavior, native H3, Spectrum capture and the sequential VDN PR #8 -> PR #11 stack with explicit CPU SOL substitutes.

Real RTX PRO 6000 evidence exists for Exact Runtime, for the pre-v2 routing failure, and for post-v2 eligibility reaching 2750 SOL-capable calls. That last run executed zero sparse calls because of the now-removed external loader requirement; it is not a sparse performance result. Fresh GPU execution is still required to confirm packaged Sana CuTe sparse calls, VDN-local execution, speed, Spectrum interaction and decoded-media quality.
