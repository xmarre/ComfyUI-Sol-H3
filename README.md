# ComfyUI-Sol-H3

Native MiniMax-H3 affine fusion and experimental NVIDIA BF16 CuTe SOL attention for ComfyUI. **Draft: Exact Runtime has production RTX PRO 6000 evidence; sparse SOL and the new interoperability paths still require current-head GPU/performance/audiovisual validation.**

ComfyUI already provides a separate Block Sparse Attention implementation through `comfy-kitchen`. Its chunked/int8 H3 producer differs from this repository's pinned NVIDIA BF16 kernel and routing policy. Keep both as separate A/B baselines.

## Nodes and composition

Connect your MODEL patches, then **Sol-H3 SOL Attention (Experimental)** before sampling. `exact_fusion=true` also requests the Exact affine optimization. A subsequent **Sol-H3 Exact Runtime** node merges that request; it does not install a second lifecycle. Exact → SOL, SOL → Exact, and identical repeated applications are supported. Two different SOL policies on one branch are ambiguous and require separate MODEL branches.

SOL wraps existing block replacements. Each call either uses eligible SOL attention or delegates to the inherited implementation with a recorded reason. Generic `optimized_attention_override` providers, including KJ Sage/Sage3, supply dense warmup and dense-prefix attention. Comfy's attention recursion guard is preserved. No post-sampling error is raised merely because sparse calls or fused blocks were zero.

The defaults are tau 1.0, diagonal threshold, one dense **actual transformer evaluation**, and two dense layers. Forecast-only Spectrum calls do not consume warmup, in either diffusion-wrapper order. Sampling invocations use fresh request state. Apply SOL after block-replacing nodes to preserve their callbacks; a later node that overwrites a block slot can still bypass that slot's SOL optimization, reported as inherited ownership.

Exact fusion preserves native intermediate affine rounding. It retains norms, AdaLN projections, QKV/RoPE, attention, MLP, gates, projection hooks and output heads. Whole-block hooks/custom forwards, graph capture, unsupported activation formats or unaudited native source use the native block. A real affine parity failure still stops the operation.

## Optional interoperability companions

- [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11): restricted softmax-subcall provider contract, including v2 restricted-domain square expansion for SOL's square-QKV kernel.
- [Spectrum #104](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/104): numerical attention policy/receipt history coordination.
- [Untwisting RoPE #9](https://github.com/xmarre/ComfyUI-Untwisting-RoPE/pull/9): pure QKV preprocessing contract.

These remain draft and require combined runtime validation. They are not automatically installed.

| Combination | Execution policy | Current evidence |
|---|---|---|
| Sage/Sage3 or generic dense override + SOL | SOL on eligible calls; inherited provider on dense-required rows/calls | Real Comfy/KJ Python wrappers, CPU kernel substitutions |
| VDN full coverage + SOL | Softmax component can use SOL; learned gate remains active | Real VDN/H3/Spectrum dispatch with CPU oracle |
| VDN grouped + SOL | v2 expands each already-restricted local KV domain to matching square Q, runs SOL there when eligible, then selects only the original VDN query rows; global/anchor operations remain native | Exact restricted-domain oracle tests plus real ModelPatcher object-patch integration test |
| VDN flex + SOL | Masked Flex remains native; its existing grouped fallback can then use the grouped v2 contract | Mask tests and Flex-to-grouped CPU fallback; no Flex GPU execution |
| VDN reduced/mixed API 1/2 | Inherited gated attention for the current external sequence; later eligible native calls may resume SOL | Component contract tests; full Flow workflow GPU rerun outstanding |
| Spectrum + SOL | Actual backend receipts qualify history; policy/receipt changes reset histories before capture | Both real wrapper orders and repeated CPU sampling scopes |
| Core Block Sparse Attention + SOL | Explicit block attention ownership wins; other calls follow SOL eligibility | Source audit and ownership contracts; GPU stack outstanding |
| Core Block Sparse Attention + Spectrum | Actual-only while core lacks a predictive backend-history contract | Consumer contract test; no forecasting speedup claimed |
| Untwist + SOL | Pure QKV contract preserves K scaling once, including VDN's full post-RoPE domain before local gathers | Transform tests plus real ModelPatcher VDN/SOL preprocessing regression |
| DiffAid / Flow / other block replacements | Delegate current arguments; unsupported layouts use inherited calls | Source/contract evidence; opaque replacement history remains actual-only |
| Runtime adapters, ordinary LoRA, curve AdaLN, KJ preview | Native submodules/hooks remain active; unsupported Exact ownership uses native block | Source and projection-hook tests; full GPU/media stack outstanding |

VDN remains the owner of its trained local/window geometry, anchors, learned softmax gate and linear complement. The v2 bridge does **not** broaden a local VDN operation to unrestricted model attention: it evaluates extra query rows only over the same already-restricted KV domain so the NVIDIA square-QKV kernel can run, then discards those extra query outputs. This adds query/gather overhead, so successful VDN-local SOL execution does not by itself establish a speedup.

Spectrum histories use `attention_backend_history_v1` preflight policies and `attention_backend_receipts_v1` actual receipts. Unpredictable routing executes actual calls rather than using unqualified anchors. Transitions clear stage histories/controllers and incompatible offline archives; offline replay may therefore be unavailable for a changing backend. With older Spectrum lacking the consumer contract, SOL delegates its affected attention calls to the inherited provider.

## SOL kernel contract

Eligible Q/K/V are BF16, matching `[1, heads, rows, 128]` tensors on SM120, with supported unmasked attention flags and a current contiguous packed prefix/video-tail layout. Unsupported calls delegate locally and do not permanently disable later eligible calls.

For ordinary native H3 attention the full prefix is a KV sink and every prefix query is recomputed by the inherited dense backend. VDN v2 differs deliberately: its square-expanded prefix/global query rows are auxiliary outputs that VDN discards, so SOL keeps those rows as sink KV without paying a second dense query recomputation. The all-selected arithmetic gate uses **independent BF16 SDPA**, so approximate Sage arithmetic cannot falsely fail the SOL kernel gate. The gate tests arithmetic, not sparse output quality. Pure QKV preprocessing contracts run before SOL and its reference without repeating the transformation.

Sources are SHA-256 verified before import. Missing or mismatched optional kernel dependencies cause recorded native fallback; their code is not accepted as the requested kernel. A kernel that actually runs and fails its arithmetic gate still raises. SOL-BSA, learned distillation, full-width AdaLN schedule-table eviction and distributed execution are not implemented.

## Installation

Use the existing PR branch:

```bash
cd /home/toor/ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
git switch feature/native-sol-h3
```

Exact fusion uses ComfyUI's CUDA PyTorch/Triton environment. The optional SOL kernel requires a separately installed pinned Sana checkout:

```bash
SOL_ROOT=/home/toor/Sana-sol-h3
git clone --branch sol-engine https://github.com/xmarre/Sana.git "$SOL_ROOT"
git -C "$SOL_ROOT" checkout 2936c47637380842aaa4a4488fac5006cc542b70
```

The audited upstream recommends Python 3.12, CUDA 13.0, PyTorch 2.10, Triton 3.6, CuTe DSL >=4.5 and cuda-python. Check compatibility with your existing environment before changing these dependencies. Kernel source is not redistributed here.

```bash
cd /home/toor/ComfyUI
PYTHONPATH="$SOL_ROOT/models/minimax_h3/Sol-H3/h3_runtime/third_party${PYTHONPATH:+:$PYTHONPATH}" python main.py
```

## Validation and diagnostics

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m pytest -q
```

Set `COMFYUI_PATH`, `KJNODES_PATH`, `SPECTRUM_PATH`, and `VDN_PATH` to their checkouts to enable the real-wrapper/stack tests. GPU skips and CPU oracle substitutions are not GPU validation. See [VALIDATION](docs/VALIDATION.md) for RTX PRO 6000 evidence, commands and the remaining media matrix.

Sampling logs expose actual evaluations, SOL-eligible calls, sparse calls, dense warmup, fallback reasons, VDN-local SOL calls, VDN square-expansion requested/kernel row counts, backend transitions, inherited dense providers and arithmetic gates. Spectrum reports backend-history resets and opaque actual calls separately. A successful run with zero sparse calls is valid telemetry and cannot be used as a SOL performance sample.

See [AUDIT](docs/AUDIT.md) for ownership decisions and remaining limitations. GPL-3.0-or-later; see LICENSE and NOTICE.