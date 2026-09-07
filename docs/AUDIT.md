# Source audit — 2026-09-07

Sources were re-fetched before implementation. Target repository was empty;
main was initialized with a neutral README. Implementation lives on a feature branch.

| Source | Audited revision |
|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | `9ac7352f70b2206d4ef7a345b30106d0fa3807d1` |
| [Sana sol-engine](https://github.com/xmarre/Sana/tree/sol-engine/models/minimax_h3/Sol-H3) | `2936c47637380842aaa4a4488fac5006cc542b70` |
| [Spectrum H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) | `a360f64fbfa54681ded100a64ded86a5713ddf17` |
| [VDN H3 Plus](https://github.com/xmarre/ComfyUI-VDN-H3-Plus) | `3516368a09de0c4bc785faff434ea9d5a9479cb0` |
| [Flow-Aligned-Regenerate](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate) | `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591` |

## Optimization decisions

| Sol-H3 component | Classification and decision |
|---|---|
| Fused QKV projection | Already native in `Attention.qkv_proj`; retained |
| Q/K RMSNorm + partial split-half RoPE | Already native `ck.rms_rope_split_half_`; retained, including quantized ops |
| SwiGLU + down projection | Already native `linear_input_act`; retained |
| Residual/gate | Native in-place `addcmul_` already fused per segment; retained |
| Scale/shift modulation | Useful exact candidate: fuse three elementwise launches, preserve each dtype rounding |
| RMSNorm + modulation | Not fused here: native RMSNorm implementations/hooks and reduction order retained |
| AdaLN schedule tables | Compact curve format already reduces width. Full-width table eviction incompatible with arbitrary dynamic native patch/schedule assumptions; remains native, no precompute claim |
| Layout/copy elimination | Native packed target-video tail already contiguous. No Morton reorder. SOL uses contiguous SM120 buffers until strided execution is validated |
| Compilation/caches | Triton specializes the small affine kernel. No whole-model compile, CUDA graph capture or process-global activation cache |
| SOL attention | Useful approximate, opt-in, exact prefix KV and dense prefix queries; SM120 experimental |
| SOL-BSA + residual | Approximate; not exposed without supported hardware validation. Discarding omitted blocks would lose upstream centroid/LSE semantics |
| FastH3/DMD2/VSA adapter | Learned acceleration, separate concern; not installed or required |
| Ulysses, INT8 QKV/FP8 output communication | Multi-GPU-only here; omitted |
| Parallel VAE | Distributed/runtime-specific, outside native MODEL patch; omitted |

This is a bounded native port. Full-width schedule-aware AdaLN precompute is
not implemented: a table keyed solely by sampler sigmas is insufficient for
per-token masks, independent audio shifts and conditioning noise values, and
cannot silently freeze adapter changes. Existing native weight offloading is
retained. Exact fusion reduces launches but has no measured end-to-end gain yet.

## Current sparse contract

Audited `h3_runtime/sparse_attention.py`, `bsa_metadata.py`, `sol_residual.py`,
the vendored `sol_attn/interface.py`, and the earlier RTX5090 adapter.
The current interface explicitly maps `(12, 0)` to `cute_sm120`; the current
runtime's world-size restriction is a deployment policy, not absence of a
single-GPU kernel. Its README says reachable SM120 paths are unvalidated.
The earlier [H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/)
demonstrates single-GPU sparse execution, but predates the released H3 prefix
protection contract and does not validate this port.

The bridge retains current `tau=1`, diagonal threshold, one dense evaluation,
two dense layers, complete prefix sink and exact prefix query recomputation.
All-selected arithmetic uses a full-sequence exact KV sink, not a guessed
extreme tau. Gate tolerances follow the current H3 kernel gate. These tolerances
test kernel arithmetic only; they are not a perceptual bound for sparse routing.

The BSA implementation routes 64-token blocks and includes omitted-block
centroid residual with LSE/state merging. Its all-selected gate verifies the
arithmetic path, not learned or training-free sparsification quality. This
package does not implement a weaker top-k/drop-only BSA approximation.

## Compatibility evidence

Spectrum `_execute_actual` chains existing first/last block replacements and
observes final packed features. Exact patches can occupy those existing slots.
Its forecasting path is independent of exact affine execution; dense numerical
parity therefore remains mandatory. Sparse histories need explicit coordination;
the current package rejects Spectrum in sparse mode at installation and runtime.

Flow mixed-grid wrappers forward changed `layout`, `mod_segments`, RoPE and
attention state into previous block patches; the exact patch consumes those
current arguments. SOL rejects its VDN external-sequence marker. VDN `apply.py`
uses projection hooks and native weight patches; those modules are still called
by exact fusion, but a VDN replacement of block.forward is incompatible and
raises. No external repository is modified or imported as a hard dependency.

Continuum/RefDelta/Untwist/DiffAid interaction is not established by this audit:
their effects must be checked with active production settings. Same-operation
ownership is rejected rather than silently resolved by patch order.

## Supplied papers

The five supplied PDF versions were extracted locally and inspected. Their roles:

* **Sol Video Inference Engine**, 2606.23743v2: instance-specific optimization
  and human quality validation; headline gains do not transfer across stacks.
* **Sol-Attn**, 2607.24027v1: online dynamic routing and approximation correction;
  training-free sparse execution remains approximate.
* **DMD2**, 2405.14867v2: distribution matching distillation, critic training and
  few-step training/inference alignment; background for learned acceleration,
  not a runtime optimization.
* **VSA**, 2505.13389v5: trainable coarse/fine sparse attention and joint sparse
  distillation; does not establish lossless inference-only SOL substitution.
* **Spectrum**, 2603.01623v1: forecasting from historical transformer features;
  history must correspond to the numerical backend currently being evaluated.

The newer [Sol-H3 page](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/)
and [earlier H3 page](https://nvlabs.github.io/Sana/Sol-Engine/H3/) supply context.
The actual native and released runtime code governs implementation decisions.
