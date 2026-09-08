# Interoperability source audit — 2026-09-08

Existing PR #1 is continued on a mirror branch. This audit supersedes the initial exclusive-provider design and incorporates the production RTX PRO 6000 reproduction logs that were supplied after the first interoperability pass.

| Source | Inspected/tested revision |
|---|---|
| ComfyUI | `efa6c8f804bff78b46a0fd458ebd2e47bba07a30` |
| KJNodes | `c9869eade9920a1b949de07c4a197156006bcceb` |
| VDN-H3-Plus companion | PR #11 v2 tree consolidated as `c05a7a8ddd457e17bc459f6eca97b4683c3803c9` |
| Spectrum companion | `31de0cb89b472c965794fdc340de857360a58107` |
| Flow | `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591` |
| Continuum | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| Untwist | PR #9 preprocessing contract based on the audited implementation |
| DiffAid | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` |
| Pinned Sana kernel | `2936c47637380842aaa4a4488fac5006cc542b70` |

## Production reproduction that changed the design

A correctly ordered production stack — VDN, runtime DoRA, DiffAid, Untwist, SOL, Spectrum, then preview/progressive/Continuum — ran on the RTX PRO 6000 with Sage enabled. Exact fusion executed, but native-grid stages still reported `sol_eligible_calls=0`, `sparse_calls=0`, `vdn_local_sol_calls=0`, with every block recorded as inherited attention ownership. Mixed-grid calls reported the expected `external_sequence_native` fallback.

That ruled out node ordering, dense warmup, tau and the external kernel as the immediate cause: SOL never reached kernel eligibility. The first CPU stack test had also bypassed the real failure mode by assigning `block.attn.forward` directly instead of exercising VDN's actual `ModelPatcher.object_patches` lifecycle.

The deeper limitation was VDN provider API v1. Normal H3 VDN local queries attend to an already-restricted KV domain that includes prefix/global rows, so the requested query set is usually rectangular against K/V. NVIDIA's pinned SOL interface requires square Q/K/V. The v1 contract therefore could not make normal production VDN local calls eligible even when it was consumed correctly.

VDN PR #11 now exposes provider API v2. For each local operation VDN still constructs its original restricted KV domain, then supplies an expanded query tensor over that same ordered domain plus a mapping back to the originally requested query rows. SOL may evaluate that square restricted domain and VDN keeps only the original query outputs. The operation does not gain any KV rows and therefore does not become unrestricted full-sequence attention. Global/anchor operations remain native, masked Flex remains native, and VDN retains its learned gate and linear complement.

The Sol-H3 native interoperability test now installs VDN with its real `apply_vdn()` ModelPatcher object patches, calls `patch_model()`, executes native H3 through actual Comfy wrappers, and verifies that v2 reaches the SOL provider with square-expansion telemetry. This closes the previous false-green test gap.

## Compatibility-gate disposition

| Former rejection or invariant | Classification | New behavior |
|---|---|---|
| `reject_sparse_conflicts` / provider-name bans | Unvalidated/composable ownership | Removed; per-call dispatch/telemetry |
| `reject_forecasting_conflicts` / SOL+Spectrum | Numerical-history ownership | Consumer preflight/receipt protocol; resets before anchor capture |
| Generic optimized override including Sage/Sage3 | Composable dense provider | Retained as dense provider, recursion marker preserved |
| `_vdn_forward` | Multiple operation ownership | Explicit VDN restricted-subcall v1/v2 hooks; learned gate/linear math retained |
| Rectangular VDN local Q vs KV | Kernel topology mismatch with coherent bridge | VDN v2 expands Q only over the same restricted KV domain, runs square SOL when eligible, then selects original query rows |
| VDN external sequence API 1/2 | Per-call unsupported native-grid interpretation | Inherited gated attention for that call; normal grid can resume SOL |
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
| Optional kernel absent/mismatched | Kernel cannot be accepted | Native fallback; verified sources required before kernel import |
| Arithmetic gate actually fails | Computed output violates tested invariant | Hard failure; never silently accept failed arithmetic |
| Missing request scope / tampered metadata / cyclic transform | Broken internal lifecycle/contract | Hard failure |
| Malformed affine indices/strides | Unsafe fused indexing | Existing hard invariant retained |

## Numerical and ownership details

Comfy's `wrap_attn` sets `_inside_attn_wrapper` before invoking the override and materializes `AttentionTensorContainer` inputs when the override has no container implementation. SOL accepts that tensor contract and retains the marker on inherited dense calls, including providers that themselves use wrapped fallbacks. It does not special-case Sage names.

Independent BF16 SDPA verifies the all-selected SOL kernel. Dense-prefix execution on ordinary native H3 attention uses the inherited provider and may therefore be approximate Sage. Pure `attention_preprocess_v1` transformations expose their inherited provider: SOL transforms QKV once and uses the dense leaf on transformed prefix queries.

VDN v2 preprocessing happens on the full post-RoPE VDN domain before local gather, so transforms such as Untwist retain their original packed-row coordinates. For an expanded VDN local SOL call, prefix/global rows remain sink KV. Their expanded query outputs are auxiliary and discarded, so SOL does not perform the ordinary native-H3 dense prefix-query recomputation for those unused rows.

VDN's square expansion has a real performance cost: it computes additional query rows and may perform additional gathers. Telemetry separately records `vdn_square_expanded_calls`, requested query rows and kernel rows so GPU tests can determine whether the sparse work still wins. Correct execution is established structurally; speedup is not.

Spectrum decides whether to forecast before executing H3, but captures anchors inside the final block replacement. Preflight therefore establishes expected policy and actual per-block receipts are consumed before `observe_actual`. Both diffusion-wrapper orders are exercised. Unknown block replacement routing is actual-only. Core BSA currently lacks the predictive contract and is also actual-only under the companion consumer.

Backend transitions reset all stage forecasters/controllers and invalidate old offline archives. Same-step mixed receipts are not retained as one numerical anchor. Backend identity is included in rollback snapshots. A late backend reset also discards pending residual shadow/hold evidence.

Flow can change layout, explicit RoPE rows and modulation rows through block callbacks. Exact uses current delegated arguments. SOL validates call-time row counts; stale or unsupported layouts fall back rather than inventing a lattice. Continuum's separate sampler invocations receive fresh request state. The supplied production log verifies those paths execute together, but sparse SOL still needs a fresh post-v2 GPU run.

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
| Layout/copy elimination | Native packed target-video tail is already contiguous. No Morton reorder. External NVIDIA SOL uses contiguous SM120 buffers until strided execution is validated |
| Compilation/caches | Triton specializes the small affine kernel. No whole-model compile, CUDA graph capture or process-global activation cache |
| External NVIDIA SOL attention | Approximate, opt-in, complete prefix KV sink; ordinary native-H3 prefix queries are dense; SM120 experimental |
| VDN v2 square bridge | Preserves VDN restricted KV geometry while adapting rectangular local calls to the square kernel; extra-Q overhead must be measured |
| ComfyUI Block Sparse Attention | Already upstream in core and backed by `comfy-kitchen`; retained as a separate comparison path rather than reimplemented |
| SOL-BSA + residual | Approximate; not exposed without supported SM120 validation |
| FastH3/DMD2/VSA adapter | Learned acceleration, separate concern; not installed or required |
| Ulysses, INT8 QKV/FP8 output communication | Multi-GPU-only in NVIDIA's Sol-H3 runtime; omitted |
| Parallel VAE | Distributed/runtime-specific, outside native MODEL patch; omitted |

## ComfyUI built-in sparse attention

ComfyUI commit `e308cc73b466584b0c17be695e5de1a17438bb40` added `BlockSparseAttention`, including a MiniMax-H3-specific producer path. The current core path is not a thin call to NVIDIA's H3 runtime: it projects QKV in chunks into `comfy_kitchen.sol_attn_chunked`, keeps its own pooled K/V state and supports Sol-Attn, SLA-style top-k and VSA policies.

The built-in path remains a separate A/B target because its producer, quantization/carriers, schedule defaults, exact-row policy and approximation implementation differ from the pinned NVIDIA BF16 CuTe path here. The external node is justified as a controlled reference-policy comparison, not because ComfyUI lacks sparse attention.

## Evidence limits

Current CPU CI exercises real ModelPatcher object-patch application, current KJ wrapper behavior, native H3, Spectrum capture and VDN v2 dispatch with an explicit CPU SOL oracle. The Sol mirror's latest validated code before documentation changes passed **55 tests with 30 expected GPU/integration skips**, while the native interoperability job passed **12 tests**. The consolidated VDN v2 tree passed its pinned-Comfy/oracle suite, legacy migration and current-Comfy smoke jobs.

Real RTX PRO 6000 evidence exists for Exact Runtime and for the pre-v2 failed sparse-routing reproduction. There is still no successful post-v2 sparse SOL GPU execution, VDN-local sparse performance result, or decoded-media acceptance result. Those are the next acceptance tests; they must not be inferred from CPU CI.