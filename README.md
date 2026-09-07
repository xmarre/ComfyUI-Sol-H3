# ComfyUI-Sol-H3

Native MiniMax-H3 affine fusion and optional experimental NVIDIA Sol-Attn reference-policy integration for ComfyUI.
Development preview: **no SM120 runtime, speedup, or media-quality validation yet**.

Current ComfyUI already ships its own **Block Sparse Attention** node backed by `comfy-kitchen` Sol-Attn. This repository does **not** claim that ComfyUI lacks Sol-Attn. The experimental node here instead targets the newer NVIDIA/Sana H3 policy and released CuTe `cute_sm120` path so the two implementations can be compared directly. The implementations are not arithmetic-equivalent: ComfyUI's H3 path uses `comfy_kitchen.sol_attn_chunked` with chunked QKV/int8 carriers and different routing/sink defaults, while the pinned NVIDIA path used here keeps BF16 QKV, `tau=1.0`, a complete prefix KV sink and dense recomputation of every prefix query row.

## Nodes

Connect the MODEL through every compatible model patch that may short-circuit the diffusion model or replace H3 blocks, then place **Sol-H3 Exact Runtime last** before the sampler. In particular, apply Spectrum, DiffAid, ComfyUI Block Sparse Attention, Untwist/attention patches and similar compatible providers **before** the Exact Runtime node. Exact mode wraps existing H3 block replacements and verifies that each one delegates exactly once to the fused native block. This ordering also lets Spectrum forecast calls bypass Sol-H3 entirely while Spectrum actual calls still execute the exact fusion.

The Exact Runtime node fuses per-segment affine modulation while explicitly preserving native intermediate rounding. Native norms, AdaLN projections, QKV/RoPE, attention, MLP/INT8/ConvRot ops, residual gates, blocks, model offloading, sampling schedules, references and output projection remain in the active native path. The first real affine activation of each dtype/indexing/physical-layout variant gets a bitwise parity probe; a failure stops execution. GPU tests and matched full-model parity are still required before treating this as production-ready.

**Sol-H3 SOL Attention (Experimental)** separately opts into the pinned NVIDIA CuTe sparse path. Use it instead of the Exact Runtime node, with `exact_fusion` enabled to combine both. It requires SM120, BF16 activations and head dimension 128. Defaults follow the released H3 policy: tau 1.0, `diag` threshold, first actual evaluation dense, first two layers dense. Warmup counts actual transformer evaluations, not outer sampler steps; no assumptions are made about sigma direction or number of solver evaluations. Every sampling invocation starts a fresh scope.

The complete prefix before target video is an exact KV sink, including generated audio, references and conditioning. Every prefix query row is recomputed densely after the SOL call. No Morton reordering is applied. The first real QKV shape is checked with an all-selected arithmetic gate; that validates kernel arithmetic only and does not establish perceptual equivalence of sparse routing.

External SOL is intentionally exclusive: it rejects Spectrum, ComfyUI Block Sparse Attention, VDN hybrid/window attention, competing optimized-attention overrides and pre-existing H3 block replacements. Unsupported sparse execution raises; it never silently substitutes dense. A request with zero sparse calls raises even when warmup explains why.

There is no SOL-BSA node: NVIDIA's cuDNN BSA + omitted-block residual path is not validated on SM120 here, and replacing it with a weaker drop-only approximation would not be equivalent.

## Install

Set the two checkout roots for your machine first:

```bash
COMFYUI_ROOT=/path/to/ComfyUI
SOL_ROOT=/path/to/Sana-sol-h3
```

Clone this custom node into the configured ComfyUI checkout:

```bash
cd "$COMFYUI_ROOT/custom_nodes"
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
git switch feature/native-sol-h3
```

Exact fusion uses the CUDA PyTorch and Triton already installed with ComfyUI. No models, Diffusers, schedulers, adapters or sparse kernels are downloaded by the Exact Runtime node.

The native source contract was audited against ComfyUI `41db8f4fa1587d139e412a57b9b69394e3b13f95`. The only commit after the original `9ac7352` audit changed `comfy/model_management.py`; the MiniMax-H3 block/modulation source did not change. Runtime source hashes still reject drift in the exact arithmetic being replaced while allowing unrelated ComfyUI updates.

Experimental NVIDIA SOL additionally uses the released, pinned kernel as an external dependency. This package does not redistribute NVIDIA/Sana kernel sources. Use a separate dependency checkout and retain its notices:

```bash
git clone --branch sol-engine https://github.com/xmarre/Sana.git "$SOL_ROOT"
git -C "$SOL_ROOT" checkout 2936c47637380842aaa4a4488fac5006cc542b70
```

Upstream recommends Python 3.12, CUDA 13.0, PyTorch 2.10 and Triton 3.6; its released integration requires CuTe DSL >=4.5 and cuda-python. Install those in the Comfy environment only after checking compatibility with its existing PyTorch/CUDA stack. Upstream also pins `nvidia-cudnn-frontend[cutedsl]` at `29106622617bfd9031a53099a6fbbc5e74a474e9`; BSA itself is not used by this package.

Expose the pinned `sol_attn` package when launching ComfyUI:

```bash
cd "$COMFYUI_ROOT"
PYTHONPATH="$SOL_ROOT/models/minimax_h3/Sol-H3/h3_runtime/third_party${PYTHONPATH:+:$PYTHONPATH}" python main.py
```

The loader finds the package root without importing it, verifies every pinned package file with SHA-256 before any `sol_attn` package code executes, retains Git blob IDs separately for provenance, and requires actual `cute_sm120` dispatch. Missing CuTe, altered sources, wrong architecture or wrong dtype stops execution. Triton's reference backend is not accepted as a substitute for the requested CuTe kernel. If `sol_attn` was imported before this integrity gate ran, external SOL fails closed and requires a clean ComfyUI restart.

## Composition and limits

For **Exact Runtime**, apply compatible patches first and Sol-H3 Exact last. Existing `dit/double_block` replacements are wrapped rather than overwritten. If an existing replacement fails to delegate exactly once, execution aborts instead of silently bypassing either provider. Projection/norm/MLP hooks and attention overrides remain in the active submodules. A later diffusion wrapper that short-circuits after Sol-H3 was installed is detected when Sol-H3 sees an incomplete block set; reorder the nodes so that wrapper is outside Sol-H3.

Exact + Spectrum is structurally composable only with that ordering: Spectrum first, Sol-H3 Exact last. Spectrum forecast calls never enter Sol-H3; Spectrum actual calls do, so the exact-fusion counters represent actual transformer evaluations rather than logical solver calls. Matched GPU parity is still required.

Exact Runtime is lossless **relative to the active upstream attention/model patch stack**. Its metadata uses `backend=inherit`: Sol-H3 Exact does not own attention and therefore does not certify that the upstream model is dense. Sage, Block Sparse Attention, or another compatible provider may still be active upstream.

**Spectrum + scheduled sparse attention remains rejected unless the consumer tracks backend-history transitions.** The Spectrum revision audited here does not consume a sparse backend identity or reset history when actual H3 evaluations change numerical backend. That applies both to external SOL's dense-warmup-to-sparse transition and to ComfyUI Block Sparse Attention schedules. `Spectrum + Sol-H3 Exact` remains valid because Exact does not change the attention backend; `ComfyUI Block Sparse Attention + Sol-H3 Exact` is also valid when Spectrum is absent.

The stable `transformer_options['sol_h3_runtime_v1']` dictionary includes `api`, backend ownership, approximate policy, kernel identity where applicable and a SHA-256 `fingerprint`, but metadata publication alone is not Spectrum integration.

VDN hybrid attention and mixed/reduced VDN external sequences are rejected in external SOL mode. Exact mode uses the current block's modulation/RoPE inputs and invokes the active attention module normally. No broad production compatibility claim is made for Continuum, RefDelta, progressive handoff, Untwist RoPE, DiffAid, ComfyUI Block Sparse Attention or dynamic adapters before media tests; the structural audit only establishes where ownership is preserved or rejected.

Compact curve AdaLN is detected and reported as not applicable to NVIDIA's full-width schedule-table eviction. Full-width native models keep their projections: this release does not precompute or evict AdaLN weights. Native masked timestep rows, independent audio shifts, conditioning timesteps and dynamic adapter changes make the fixed-pipeline table replacement unsafe to copy unchanged.

FastH3, VDN/Turbo and VSA-trained acceleration are separate model adaptations. Use compatible adapters through normal Comfy loaders with their documented sampling settings. These nodes never select four forwards or change sigmas. NVIDIA's 8xB300 headline results also include learned acceleration and distributed communication/VAE work that is absent here.

## Tests

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m pytest -q
python -m pytest -q tests/test_gpu.py  # CUDA required; CPU skips are not validation
```

CPU CI checks contracts, lifecycle, wrapper/block ownership, chained replacements and node schema. GPU tests compare affine outputs bitwise against the native arithmetic expression. Full-model numerical parity, performance and audiovisual acceptance remain separate gates.

The initial reviewed suite passed 40 CPU/offline-compiler tests with 18 CUDA execution tests skipped. The current revision adds explicit tests for compatible block chaining, late sparse-provider conflicts, global hook rejection, pre-import source verification and non-specialized affine runtime scalars; rely on the current GitHub Actions result rather than the historical count. Triton 3.6 previously compiled BF16/FP16/FP32 scalar and indexed affine variants for SM120 and inspected BF16 PTX retained separate rounded add/multiply/add instructions. That is compilation evidence only.

## Sources and license

GPL-3.0-or-later; see LICENSE and NOTICE. See [the source audit](docs/AUDIT.md) for pinned ComfyUI/Sana revisions, the built-in ComfyUI sparse implementation comparison, and the supplied Sol Engine, Sol-Attn, DMD2, VSA and Spectrum papers. Original integration code is provided here; external kernels retain their own licenses and notices.
