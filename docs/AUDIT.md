# Source audit — 2026-09-08

Sources were re-fetched before implementation and again during the handoff review. Target repository was empty; main was initialized with a neutral README. Implementation lives on a feature branch.

| Source | Audited revision |
|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | `41db8f4fa1587d139e412a57b9b69394e3b13f95` |
| [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen) | `0.2.33` / source `21003fa97bf3b180393446d729ae630ceb6c2a52` |
| [Sana sol-engine](https://github.com/xmarre/Sana/tree/sol-engine/models/minimax_h3/Sol-H3) | `2936c47637380842aaa4a4488fac5006cc542b70` |
| [Spectrum H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) | `a360f64fbfa54681ded100a64ded86a5713ddf17` |
| [VDN H3 Plus](https://github.com/xmarre/ComfyUI-VDN-H3-Plus) | `3516368a09de0c4bc785faff434ea9d5a9479cb0` |
| [Flow-Aligned-Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) | `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591` |
| Continuum Plus | `a5b8943844594545301b20d01af5d9e3fa38ae29` |
| RefDelta Solver | `034e4c4c14c56bf76813cee4765e7164b0c7e0db` |
| Untwisting RoPE | `8ab3f38a621076de9f25cf64c60f52a9b5457827` |
| DiffAid Patches | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` |

The original native source-contract audit used ComfyUI `9ac7352f70b2206d4ef7a345b30106d0fa3807d1`. The current head is one commit later and that commit changes only `comfy/model_management.py`; `comfy/ldm/minimax/model.py` is unchanged, so the canonical source hashes for `_mod_row`, `_mod_scale_shift`, `_mod_gate` and `DiTBlock.forward` remain valid.

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

## Current external sparse contract

Audited NVIDIA `h3_runtime/sparse_attention.py`, `bsa_metadata.py`, `sol_residual.py`, the vendored `sol_attn/interface.py`, and the earlier RTX5090 adapter. The current interface explicitly maps `(12, 0)` to `cute_sm120`; the current Sol-H3 engine's world-size restriction is deployment policy, not absence of a single-GPU kernel. Its README still marks reachable SM120 paths as unvalidated. The earlier [H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/) demonstrates single-GPU sparse execution but predates the released H3 prefix-protection contract and does not validate this port.

The bridge retains current `tau=1`, diagonal threshold, one dense evaluation, two dense layers, complete prefix sink and exact prefix-query recomputation. All-selected arithmetic uses a full-sequence exact KV sink, not a guessed extreme tau. Both that arithmetic gate and the prefix-query recomputation now call the **active Comfy dense-attention provider** rather than hard-coding PyTorch SDPA, so the reference follows the user's selected dense backend. The Comfy attention wrapper's `_inside_attn_wrapper` recursion guard is deliberately retained when a raw provider falls back to another wrapped provider; dropping it can recursively re-enter SOL through the same `transformer_options`. Gate tolerances follow the current H3 kernel gate. These tolerances test kernel arithmetic only; they are not a perceptual bound for sparse routing.

The BSA implementation routes 64-token blocks and includes omitted-block centroid residual with LSE/state merging. Its all-selected gate verifies the arithmetic path, not sparsification quality. This package does not implement a weaker top-k/drop-only BSA approximation.

## Exact-patch composition and ordering

A second review found an ordering condition that the first implementation did not model correctly. Spectrum's diffusion wrapper can return a forecast without calling the inner native H3 `_forward` at all. If Sol-H3 is installed **before** Spectrum, Sol-H3's diffusion wrapper enters first, expects all block replacements to execute, and correctly reports a bypass on Spectrum forecast calls. Therefore the correct composition order is:

`model / adapters -> Spectrum or other short-circuit wrapper -> compatible block/attention patches -> Sol-H3 Exact Runtime -> sampler`

With Sol-H3 Exact installed last, Spectrum remains outermost. Forecast calls return before entering Sol-H3. Actual calls delegate inward to Sol-H3, so Sol-H3's evaluation/block counters correspond to actual transformer evaluations rather than logical Spectrum steps.

Exact mode wraps pre-existing `dit/double_block` replacements instead of rejecting them. The existing provider receives the current arguments and a replacement `original_block` callback that executes the fused native block. Sol-H3 verifies that each existing replacement delegates **exactly once**; zero or multiple delegation aborts. This preserves providers such as DiffAid or ComfyUI's H3 sparse producer that intentionally modify block inputs/attention and then call the native block, while rejecting providers that fully replace the arithmetic being fused.

External NVIDIA SOL remains exclusive. It rejects pre-existing block replacements, `optimized_attention_override`, ComfyUI `block_sparse_attention`, Spectrum, VDN external/mixed-grid sequences and VDN hybrid attention. A later provider that overwrites its blocks is caught by per-evaluation block accounting.

The exact affine parity cache is keyed by dtype, width, indexed/scalar mode **and physical activation/modulation strides**. A later provider that presents a different strided layout therefore receives a fresh real-activation parity gate instead of inheriting validation from a numerically similar but physically different layout.

## Compatibility evidence

Spectrum `_execute_actual` copies the current `patches_replace` table, wraps the first/final block for observations and calls the pre-existing replacement from its capture callback. With Spectrum outside Sol-H3 Exact, those capture callbacks delegate to the Sol-H3 block patch, which then executes the fused native block. Spectrum forecast calls never enter Sol-H3. This establishes structural compatibility; GPU parity remains mandatory.

Spectrum sparse compatibility is deliberately different. Its historical forecaster assumes actual features are samples from one numerical backend. The external Sol-H3 sparse policy changes from dense warmup to sparse SOL within the same sampling run. `sol_h3_runtime_v1` publishes a fingerprint, but the audited Spectrum consumer does not read it or reset history at dense/sparse transitions. External SOL + Spectrum therefore remains gated.

The same issue exists for **ComfyUI Block Sparse Attention + Spectrum**: the core sparse node has an active schedule and can produce dense actual evaluations before/after sparse ones. Spectrum has no audited identity hook for those transitions. Sol-H3 Exact therefore rejects a workflow in which it detects both Spectrum and core `block_sparse_attention`. Spectrum + Exact and core Block Sparse Attention + Exact remain individually structurally composable; the three-way combination is not.

VDN `apply.py` uses projection hooks/native weight patches and can replace `attn.forward`. Exact mode invokes the active attention module and preserves those submodule hooks. External SOL rejects `_vdn_forward`: VDN window/grouped dispatch can bypass or multiply optimized-attention calls and cannot share attention ownership.

Flow mixed-grid wrappers forward changed layout, modulation segments, RoPE and attention state through existing block patches. Exact mode consumes the current delegated arguments. External SOL rejects VDN external/reduced/mixed-grid sequence markers.

Continuum owns APPLY_MODEL conditioning/layout preparation and invokes the native sampler; each invocation receives fresh Sol-H3 state. RefDelta owns scheduler/sampler configuration rather than affine arithmetic. Untwist installs an optimized-attention override; Exact mode inherits it, external SOL rejects it. DiffAid transforms block arguments then delegates to the previous block; apply DiffAid before Sol-H3 Exact so the transformed call reaches the fused native block.

ComfyUI `BlockSparseAttention` owns every H3 block replacement and an optimized-attention override. Apply it **before** Sol-H3 Exact when testing that pair: Sol-H3 Exact wraps the core block producer and remains lossless relative to that active sparse backend. Exact metadata now uses `backend=inherit`, with sparse-only fields unset, because the Exact node does not own or certify the inherited attention backend. External SOL + core Block Sparse Attention is rejected. Do not put Spectrum into the BlockSparse + Exact workflow until backend-history coordination is implemented.

These are structural findings. Combined GPU/media behavior remains unvalidated. Same-operation ownership is rejected rather than silently resolved by patch order.

Exact arithmetic replacement additionally checks canonical native function source hashes, allowing unrelated upstream edits while rejecting drift in the block/modulation operations being replaced. CPU source-derived tests exercise those operations and dynamic projection hooks with real PyTorch tensors; they substitute the literal affine reference for the unavailable GPU kernel.

## Supplied papers

The five supplied PDF versions were inspected. Their roles in this implementation are:

- **Sol Video Inference Engine**, 2606.23743v2: instance-specific optimization and human quality validation; headline gains do not transfer across runtime stacks.
- **Sol-Attn**, 2607.24027v1: online dynamic routing and approximation correction; training-free sparse execution remains approximate.
- **DMD2**, 2405.14867v2: distribution matching distillation, critic training and few-step training/inference alignment; background for learned acceleration, not a runtime optimization.
- **VSA**, 2505.13389v5: trainable coarse/fine sparse attention and joint sparse distillation; does not establish lossless inference-only SOL substitution.
- **Spectrum**, 2603.01623v1: forecasting from historical transformer features; history must correspond to the numerical backend currently being evaluated.

The newer [Sol-H3 page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) and [earlier H3 page](https://nvlabs.github.io/Sana/Sol-Engine/H3/) provide benchmark context. The actual native and released runtime code governs implementation decisions.
