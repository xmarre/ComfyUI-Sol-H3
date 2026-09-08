# Interoperability source audit — 2026-09-08

Existing PR #1 is continued on a mirror branch. This audit supersedes the initial exclusive-provider design. The full user runtime log was not attached in this session: only its reported errors and five papers were available. We reproduced the old rejection logic in source/tests, not the user's complete workflow.

| Source | Inspected base revision |
|---|---|
| ComfyUI | `efa6c8f804bff78b46a0fd458ebd2e47bba07a30` |
| KJNodes | `c9869eade9920a1b949de07c4a197156006bcceb` |
| VDN-H3-Plus | `3516368a09de0c4bc785faff434ea9d5a9479cb0`; refreshed main before companion publication |
| Spectrum | `a360f64fbfa54681ded100a64ded86a5713ddf17` |
| Flow | `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591` |
| Continuum | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| Untwist | `8ab3f38a621076de9f25cf64c60f52a9b5457827` |
| DiffAid | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` |
| Pinned Sana kernel | `2936c47637380842aaa4a4488fac5006cc542b70` |

## Compatibility-gate disposition

| Former rejection or invariant | Classification | New behavior |
|---|---|---|
| `reject_sparse_conflicts` / provider-name bans | Unvalidated/composable ownership | Removed; per-call dispatch/telemetry |
| `reject_forecasting_conflicts` / SOL+Spectrum | Numerical-history ownership | Consumer preflight/receipt protocol; resets before anchor capture |
| Generic optimized override including Sage/Sage3 | Composable dense provider | Retained as dense provider, recursion marker preserved |
| `_vdn_forward` | Multiple operation ownership | Explicit VDN restricted-subcall hook; learned gate/linear math retained |
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

Comfy's `wrap_attn` sets `_inside_attn_wrapper` before invoking the override and materializes AttentionTensorContainer inputs when the override has no container implementation. SOL accepts that tensor contract and retains the marker on inherited dense calls, including providers that themselves use wrapped fallbacks. It does not special-case Sage names.

Independent BF16 SDPA verifies the all-selected SOL kernel. Dense-prefix execution uses the inherited provider and may therefore be approximate Sage. Pure `attention_preprocess_v1` transformations expose their inherited provider: SOL transforms QKV once and uses the dense leaf on transformed prefix queries. Legacy Untwist remains native because treating its K transform as a mere dense leaf would lose that transform on sparse calls.

VDN retains exact window/global/anchor domains and its learned sigmoid softmax gate, output projection and learned linear complement. The new hook receives restricted QKV and a square-aligned flag derived from the row-index plan. Equal tensor sizes alone are insufficient to promise alignment. Normal prefix-bearing windows are rectangular and remain native. Masked Flex remains native; existing per-call grouped fallback is retained. API-2 mixed sequences retain VDN's existing dense gate with geometry-dependent linear processing disabled by VDN itself.

Spectrum decides whether to forecast before executing H3, but captures anchors inside the final block replacement. Preflight therefore establishes expected policy and actual per-block receipts are consumed before `observe_actual`. Both diffusion-wrapper orders are exercised. Unknown block replacement routing is actual-only; this is a deliberate current limitation, not a compatibility RuntimeError. Core BSA currently lacks the predictive contract and is also actual-only under the companion consumer.

Backend transitions reset all stage forecasters/controllers and invalidate old offline archives. Same-step mixed receipts are not retained as one numerical anchor. Backend identity is included in rollback snapshots. A late backend reset also discards pending residual shadow/hold evidence. Tests must keep auditing retry/replay and conditioning subcalls; media behavior remains unverified.

Flow can change layout, explicit RoPE rows and modulation rows through block callbacks. Exact uses current delegated arguments. SOL validates call-time row counts; stale or unsupported layouts fall back rather than inventing a lattice. Continuum's separate sampler invocations receive fresh request state. Repeated scopes are tested, but actual Flow/Continuum workflows are not generated in this environment.

Projection/norm/MLP hooks remain invoked, preserving runtime adapters and ordinary Comfy weight patches structurally. Whole-block hooks (including previews) force native Exact execution. Curve AdaLN remains native; no table eviction or stale projection cache is introduced.

## Optimization decisions

| Sol-H3 component | Classification and decision |
|---|---|
| Fused QKV projection | Already native in `Attention.qkv_proj`; retained |
| Q/K RMSNorm + partial split-half RoPE | Already native `ck.rms_rope_split_half_`; retained, including quantized ops |
| SwiGLU + down projection | Already native `linear_input_act`; retained |
| Residual/gate | Native in-place `addcmul_` already fused per segment; retained |
| Scale/shift modulation | Useful exact candidate: fuse native add/multiply/add launches while preserving each dtype rounding |
| RMSNorm + modulation | Not fused here: native RMSNorm implementations/hooks and reduction order are retained |
| AdaLN schedule tables | Compact curve format already removes the large full-width table cost. Full-width table eviction is not copied because arbitrary masks, independent audio shifts, conditioning timesteps and dynamic adapters invalidate the fixed pipeline assumptions |
| Layout/copy elimination | Native packed target-video tail is already contiguous. No Morton reorder. External NVIDIA SOL uses contiguous SM120 buffers until strided execution is validated |
| Compilation/caches | Triton specializes the small affine kernel. No whole-model compile, CUDA graph capture or process-global activation cache |
| External NVIDIA SOL attention | Approximate, opt-in, complete prefix KV sink and exact dense prefix queries; SM120 experimental |
| ComfyUI Block Sparse Attention | Already upstream in core and backed by `comfy-kitchen`; retained as a separate comparison path rather than reimplemented |
| SOL-BSA + residual | Approximate; not exposed without supported SM120 validation. Discarding omitted blocks would lose upstream centroid/LSE semantics |
| FastH3/DMD2/VSA adapter | Learned acceleration, separate concern; not installed or required |
| Ulysses, INT8 QKV/FP8 output communication | Multi-GPU-only in NVIDIA's Sol-H3 runtime; omitted |
| Parallel VAE | Distributed/runtime-specific, outside native MODEL patch; omitted |

This is a bounded native port. Full-width schedule-aware AdaLN precompute is not implemented: a table keyed solely by sampler sigmas is insufficient for per-token masks, independent audio shifts and conditioning noise values, and cannot silently freeze adapter changes. Existing native weight offloading is retained. Exact fusion reduces launches but has no measured end-to-end gain yet.

## Important upstream discovery: ComfyUI already has Sol-Attn

ComfyUI commit `e308cc73b466584b0c17be695e5de1a17438bb40` (2026-09-05) added `BlockSparseAttention`, including a MiniMax-H3-specific producer path. The current core path is not a thin call to NVIDIA's September 7 H3 runtime. It projects QKV in 4096-row chunks directly into `comfy_kitchen.sol_attn_chunked`, avoids materializing full Q/K/V for that path, keeps per-block pooled K/V statistics across steps, and supports Sol-Attn, SLA-style top-k and VSA. Core currently pins `comfy-kitchen==0.2.33`.

That means the first version of this audit was incomplete when it described native ComfyUI attention only in terms of dense `optimized_attention`: native ComfyUI now has a strong built-in sparse option and it must be part of every comparison.

The built-in path is nevertheless **not equivalent** to the released NVIDIA H3 policy implemented by this repository's experimental node:

- Comfy Kitchen's fused H3 producer uses its own CUDA Sol-Attn implementation with quantized/int8 carriers; NVIDIA's released H3 SOL path passes BF16 QKV to the pinned CuTe kernel.
- Core defaults use tau 1.3, a schedule window, `min_tokens=12288`, optional `extra_tokens=256`, and `exact_kv_and_rows`; NVIDIA's H3 policy uses tau 1.0, one dense transformer evaluation and two dense transformer layers.
- Core `exact_kv_and_rows` keeps the packed prefix as exact KV but makes the target-audio query blocks dense. NVIDIA's released H3 integration recomputes **every prefix query row** densely after the SOL call: text, references/conditioning and generated audio.
- Comfy Kitchen's eager/reference algorithm retains omitted blocks through pooled summary terms; NVIDIA's released CuTe implementation has its own validated arithmetic and routing implementation. Similar algorithmic intent does not make the kernels bitwise or perceptually interchangeable.

Therefore the external NVIDIA node remains justified only as a controlled **reference-policy A/B path**, not because ComfyUI lacks Sol-Attn. If the built-in Comfy path is equal or better in speed/quality on SM120, there is no reason to prefer the external dependency.


## Evidence limits

CPU tests exercise real ModelPatcher methods, current KJ Sage/Sage3 Python wrappers, native H3, Spectrum capture and VDN dispatch. External SOL/Sage kernels are explicitly substituted in CPU integration tests. VDN's linear branch is disabled in the cross-package stack test; its separate math/oracle tests remain necessary. No RTX PRO 6000 execution, full model inference, performance benchmark or decoded media comparison has occurred.

The supplied Sol Engine, Sol-Attn, DMD2, VSA and Spectrum papers remain background provenance from the original implementation. This interoperability pass used current source code for execution contracts and did not re-extract those PDFs. Full runtime-log reproduction is outstanding because that log was missing from the actual attachments.
