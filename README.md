# ComfyUI-Sol-H3

Native MiniMax-H3 affine fusion and optional experimental Sol-Attn for ComfyUI.
Development preview: **no SM120 runtime, speedup, or media-quality validation yet**.

## Nodes

Connect the MODEL output from your model/LoRA loader to **Sol-H3 Exact Runtime**,
then connect its MODEL output to your existing sampler or compatible downstream
MODEL patch. The node fuses per-segment affine modulation while explicitly
preserving native intermediate rounding. Native norms, QKV/RoPE, attention,
AdaLN projections, MLP/INT8/ConvRot ops, residual gates, blocks, model offloading,
sampling schedules, references and output projection remain in the native path.
The first real affine activation of each dtype/indexing variant gets a bitwise
parity probe; a failure stops execution. GPU tests and matched full-model parity
are still required before treating this as production-ready.

**Sol-H3 SOL Attention (Experimental)** separately opts into approximate
attention. Use it instead of the Exact Runtime node, with `exact_fusion` enabled
to combine both. It requires SM120, BF16 activations and head dimension 128.
Defaults follow the released H3 policy: tau 1.0, `diag` threshold, first actual
evaluation dense, first two layers dense. Warmup counts actual transformer
evaluations, not outer sampler steps; no assumptions about sigma direction or
number of solver evaluations. Every sampling invocation starts a fresh scope.
The complete prefix before target video is an exact KV sink, including generated
audio. Prefix queries are recomputed densely. No Morton reordering is applied.
The all-selected arithmetic gate does not validate approximate routing quality.

There is no SOL-BSA node: its cuDNN block-sparse path is not validated on SM120.
Unsupported sparse execution raises; it never silently substitutes dense.
The log records the backend fingerprint and actual sparse call count. A request
with zero sparse calls raises even when warmup explains why.

## Install

```bash
cd /home/toor/ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
git switch feature/native-sol-h3
```

Exact fusion uses the CUDA PyTorch and Triton already installed with ComfyUI.
No models, Diffusers, schedulers, or adapters are downloaded by these nodes.
The native API audited here is ComfyUI `9ac7352f70b2206d4ef7a345b30106d0fa3807d1`.
See [the source audit](docs/AUDIT.md) before using a different native version.

Experimental SOL additionally uses the released, pinned kernel as an external
dependency. This package does not redistribute NVIDIA/Sana kernel sources.
Use a separate dependency checkout; retain its notices:

```bash
git clone --branch sol-engine https://github.com/xmarre/Sana.git /home/toor/Sana-sol-h3
git -C /home/toor/Sana-sol-h3 checkout 2936c47637380842aaa4a4488fac5006cc542b70
```

Upstream recommends Python 3.12, CUDA 13.0, PyTorch 2.10 and Triton 3.6;
its released integration requires CuTe DSL >=4.5 and cuda-python. Install those
in your Comfy environment only after checking compatibility with its existing
PyTorch/CUDA stack. The upstream requirements also pin
`nvidia-cudnn-frontend[cutedsl]` at
`29106622617bfd9031a53099a6fbbc5e74a474e9`; BSA itself is not used here.
This combination has not been tested on this package's SM120 path.

Expose the pinned package when launching ComfyUI:

```bash
cd /home/toor/ComfyUI
PYTHONPATH="/home/toor/Sana-sol-h3/models/minimax_h3/Sol-H3/h3_runtime/third_party${PYTHONPATH:+:$PYTHONPATH}" python main.py
```

The loader checks every pinned source hash and requires the actual
`cute_sm120` dispatch. Missing CuTe, altered sources, wrong architecture or
wrong dtype stops execution. Triton's reference backend is not accepted as a
substitute for the requested CuTe kernel.

## Composition and limits

Apply Sol-H3 before providers that wrap existing block replacements. Exact mode
retains compatible attention overrides. It rejects an already-owned block slot,
replaced block forward or block-level hooks rather than bypassing their behavior.
Projection/norm/MLP hooks remain in the invoked native submodules. It does not
rewrite weight patches, mutate model forward methods or retain activation
caches on ModelPatcher clones. Sampling state is released on exceptions.

Exact + Spectrum is structurally composable through Spectrum's chained block
observation. Matched native/GPU testing is pending. **Sparse + Spectrum is
currently rejected**: the audited Spectrum does not consume the backend
fingerprint and cannot safely distinguish dense/sparse histories. The stable
`transformer_options['sol_h3_runtime_v1']` dictionary includes `api`, `backend`,
`approximate`, policy, kernel identity and a SHA-256 `fingerprint`. Consumers
must reset history when numerical identity changes and before transitions
between dense warmup and sparse execution. Metadata alone is not integration.

Mixed/reduced VDN external sequences are rejected in SOL mode. Exact mode uses
the current block's modulation/rope inputs. VDN hybrid block ownership may
require disabling exact fusion. No broad compatibility claim is made for
Continuum, RefDelta, Untwist RoPE, DiffAid or dynamic adapters before media tests.
See [audit](docs/AUDIT.md) and [validation](docs/VALIDATION.md).

Compact curve AdaLN is detected and reported as not applicable to schedule
precomputation. Full-width models keep native projections: this release does
not precompute or evict their weights. Native masked timestep rows and dynamic
adapter changes make NVIDIA's fixed pipeline table replacement unsafe to copy.

FastH3 and Turbo/VDN learned acceleration are separate model adaptations.
Use compatible adapters through normal Comfy loaders with their documented
sampling settings. This node never selects four forwards or changes sigmas.
NVIDIA's 8×B300 results include learned acceleration and distributed transport/
VAE work absent here; they are not speedups measured for this package.

## Tests

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m pytest -q
python -m pytest -q tests/test_gpu.py  # CUDA required; CPU skips are not validation
```

CPU CI checks contracts, lifecycle, ownership and node schema. GPU tests compare
affine outputs bitwise against the native arithmetic expression. Full-model
numerical parity, performance and audiovisual acceptance remain separate gates.

## Sources and license

GPL-3.0-or-later; see LICENSE and NOTICE. See [audit](docs/AUDIT.md) for pinned
native/Sana sources and the supplied Sol Engine, Sol-Attn, DMD2, VSA and Spectrum
papers. Original kernel integration code is provided here; external kernels
retain their own licenses and notices.
