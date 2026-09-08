# ComfyUI-Sol-H3

Native MiniMax-H3 affine fusion and experimental Sol-Attn integration for ComfyUI. **Draft: Exact Runtime has production RTX PRO 6000 evidence, and the packaged Sana CuTe SM120 SOL kernel has now compiled and executed successfully on that GPU. Full VDN/Spectrum/progressive-stack performance and audiovisual validation remain outstanding.**

This node packages the real Sol-Attn source from [`xmarre/Sana`, branch `sol-engine`](https://github.com/xmarre/Sana/tree/2936c47637380842aaa4a4488fac5006cc542b70/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn), pinned at `2936c47637380842aaa4a4488fac5006cc542b70`. SM120 uses its **CuTe `cute_sm120` backend**. Normal installation installs the declared dependencies; no manual Sana checkout, `PYTHONPATH`, `SOL_ROOT`, runtime source download or special launch command is required. `comfy_kitchen.sol_attn` is not used by this node. ComfyUI's Block Sparse Attention node is a separate integration.

## Nodes and composition

Connect your MODEL patches, then **Sol-H3 SOL Attention (Experimental)** before sampling. `exact_fusion=true` also requests the Exact affine optimization. A subsequent **Sol-H3 Exact Runtime** node merges that request; it does not install a second lifecycle. Exact → SOL, SOL → Exact, and identical repeated applications are supported. Two different SOL policies on one branch are ambiguous and require separate MODEL branches.

SOL wraps existing block replacements. Each call either uses eligible SOL attention or delegates to the inherited implementation with a recorded reason. Generic `optimized_attention_override` providers, including KJ Sage/Sage3, supply dense warmup and dense-prefix attention when available. If an optional inherited provider cannot load because of an `ImportError`/`OSError` binary or package-loader failure, Sol-H3 demotes that provider for the current sampling request and uses the original Comfy attention callable instead. It does **not** mutate `LD_LIBRARY_PATH`, preload a different `libstdc++`, install another runtime, or swallow arbitrary CUDA/compute errors. Explicit preprocessing contracts such as Untwist are still applied exactly once before the fallback dense owner. Comfy's attention recursion guard is preserved. No post-sampling error is raised merely because sparse calls or fused blocks were zero.

The defaults are tau 1.0, diagonal threshold, one dense **actual transformer evaluation**, and two dense layers. Forecast-only Spectrum calls do not consume warmup, in either diffusion-wrapper order. Sampling invocations use fresh request state. Apply SOL after block-replacing nodes to preserve their callbacks; a later node that overwrites a block slot can still bypass that slot's SOL optimization, reported as inherited ownership.

Exact fusion preserves native intermediate affine rounding. It retains norms, AdaLN projections, QKV/RoPE, attention, MLP, gates, projection hooks and output heads. Whole-block hooks/custom forwards, graph capture, unsupported activation formats or unaudited native source use the native block. A real affine parity failure still stops the operation.

## Optional interoperability companions

- [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11): restricted softmax-subcall provider contract, including requested Q, restricted K/V and exact global-prefix sink rows.
- [Spectrum #104](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/104): numerical attention policy/receipt history coordination.
- [Untwisting RoPE #9](https://github.com/xmarre/ComfyUI-Untwisting-RoPE/pull/9): pure QKV preprocessing contract.

These remain draft and require combined runtime validation. They are not automatically installed.

| Combination | Execution policy | Current evidence |
|---|---|---|
| Native dense + SOL | SOL on eligible calls; original Comfy attention on dense-required rows/calls | **Real RTX PRO 6000 CuTe SM120 compile/execution; `sparse_calls=2`, prefix parity passed** |
| Sage/Sage3 or generic dense override + SOL | SOL on eligible calls; inherited provider when usable; loader/import failure demotes request-locally to original Comfy attention with explicit telemetry | Real Comfy/KJ Python wrappers and CPU fallback contracts; installed Sage2 ABI-failure reproduction classified, GPU fallback rerun pending |
| VDN full coverage + SOL | Softmax component can use SOL; learned gate remains active | Real VDN/H3/Spectrum dispatch with CPU oracle |
| VDN grouped + SOL | v2 passes requested Q directly against its restricted K/V; global/anchor operations remain native | Exact restricted-domain oracle tests plus real ModelPatcher object-patch integration test |
| VDN flex + SOL | Masked Flex remains native; its existing grouped fallback can then use the grouped v2 contract | Mask tests and Flex-to-grouped CPU fallback; no Flex GPU execution |
| VDN reduced/mixed API 1/2 | Inherited gated attention for the current external sequence; later eligible native calls may resume SOL | Component contract tests; full Flow workflow GPU rerun outstanding |
| Spectrum + SOL | Actual backend receipts qualify history; policy/receipt/provider-demotion changes reset history before capture | Real wrapper orders, repeated CPU sampling scopes, provider-demotion identity regression |
| Core Block Sparse Attention + SOL | Explicit block attention ownership wins; other calls follow SOL eligibility | Source audit and ownership contracts; GPU stack outstanding |
| Core Block Sparse Attention + Spectrum | Actual-only while core lacks a predictive backend-history contract | Consumer contract test; no forecasting speedup claimed |
| Untwist + SOL | Pure QKV contract preserves K scaling once, including when the inherited dense leaf is unavailable | Transform tests plus provider-demotion preprocessing regression |
| DiffAid / Flow / other block replacements | Delegate current arguments; unsupported layouts use inherited calls | Source/contract evidence; opaque replacement history remains actual-only |
| Runtime adapters, ordinary LoRA, curve AdaLN, KJ preview | Native submodules/hooks remain active; unsupported Exact ownership uses native block | Source and projection-hook tests; full GPU/media stack outstanding |

VDN owns its local/window geometry, anchors, learned softmax gate and linear complement. The rectangular SM120 bridge evaluates only requested Q rows over the unchanged restricted K/V domain. VDN #11 still constructs its compatibility `square_q` payload; Sol-H3 accepts that contract but does not read or execute those extra queries. Untwist runs over the full original packed sequence before VDN gathering. See [rectangular design and GPU validation](docs/RECTANGULAR.md).

Spectrum histories use `attention_backend_history_v1` preflight policies and `attention_backend_receipts_v1` actual receipts. Unpredictable routing executes actual calls rather than using unqualified anchors. Provider demotion changes the inherited-provider identity and increments the backend transition, so Spectrum cannot forecast across an optional-provider failure as if the numerical backend were unchanged. Transitions clear stage histories/controllers and incompatible offline archives; offline replay may therefore be unavailable for a changing backend. With older Spectrum lacking the consumer contract, SOL delegates its affected attention calls to the inherited provider.

## SOL kernel contract

Native packed Q/K/V are BF16, matching `[1, heads, rows, 128]` tensors on SM120; VDN v2 local calls use Q `[1, heads, Tq, 128]` and K/V `[1, heads, Tkv, 128]`, with supported unmasked attention flags and a current contiguous packed prefix/video-tail layout. Unsupported calls delegate locally and do not permanently disable later eligible calls.

The bridge converts Comfy BHTD tensors to upstream BTHD, preserving BF16 and the native scale `128**-0.5`. It passes `tau` directly with the upstream H3 `diag` threshold policy and `kv_splits=1`. Explicit `sink_start=0` keeps prefix KV exact; upstream rounds partially overlapping 64-row blocks outward. Omitting this argument would incorrectly select a suffix sink. Ordinary H3 prefix queries are recomputed through the inherited dense provider or its explicit original-Comfy fallback. VDN v2 has no auxiliary kernel queries; its `sink_rows` refers only to global-prefix K/V.

An all-selected sink call is checked against independent BF16 SDPA. This gate checks arithmetic, not sparse quality. On the RTX PRO 6000 4096-row synthetic probe, the real packaged `cute_sm120` kernel produced `max_abs=0.0009765625`, `mean_abs=4.484307282837108e-05`, and `rel_l2=0.003004377940669656`, with exact prefix parity and two sparse calls. Untwist preprocessing runs exactly once before SOL and the dense reference/fallback. Missing Sana dependencies, failed source verification or CuTe initialization produce a local native fallback with the reason; SM120 never silently substitutes Triton or comfy-kitchen. Arithmetic-gate failures and arbitrary provider compute errors remain fatal.

SOL-BSA, learned distillation, full-width AdaLN schedule-table eviction and distributed execution are not implemented.

## Installation

Use the existing PR branch:

```bash
cd /home/toor/ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
git switch feature/native-sol-h3
python -m pip install -r requirements.txt
```

Run the dependency command in the same environment as ComfyUI (for example `comfy312`), including when updating an existing checkout. Manager installations use `requirements.txt`. Dependencies are PyTorch, Triton >=3.6,<4 (Linux), NVIDIA CUTLASS DSL with its CUDA 13 extra, CUDA Python and Apache TVM FFI. The cuDNN frontend and full Sana engine are not required. The current production target is Linux/WSL SM120; native Windows dependency/kernel execution remains unvalidated.

### SageAttention on Blackwell

SageAttention is optional and is **not** installed by Sol-H3. On SM120, upstream SageAttention 2.2.0 routes `sageattn`/`auto` to its CUDA FP8 SageAttention2++ path and explicitly states that its Triton path is currently not usable on SM120. If using KJNodes on an RTX 50-series or RTX PRO 6000 Blackwell GPU, select **`auto`**, not `sageattn_qk_int8_pv_fp16_triton`.

If Sage fails with a binary ABI error such as `GLIBCXX_3.4.32 not found`, do not repair it with a global `LD_LIBRARY_PATH`/`LD_PRELOAD` override. Rebuild the official SageAttention package from source against the same Python environment, host compiler and CUDA toolkit that run ComfyUI. The complete pinned install/repair commands and verification probes are in **[SageAttention installation and repair](docs/SAGEATTENTION.md)**.

`sol_h3/sol_manifest.json` records upstream and packaged SHA-256 hashes. Import adaptation is followed by functional rectangular-SM120 changes in `interface.py`, `preprocess.py`, and `sm120/mainloop.py`. Original source hashes remain intact; packaged hashes describe the modified files. `tools/rectangular_sm120.patch` records the reproducible functional diff. Developers can reproduce the snapshot with `python tools/vendor_sol_attn.py /path/to/pinned/Sana`; this is not an installation step.

## Validation and diagnostics

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m pytest -q
```

Set `COMFYUI_PATH`, `KJNODES_PATH`, `SPECTRUM_PATH`, and `VDN_PATH` to their checkouts to enable the real-wrapper/stack tests. GPU skips and CPU oracle substitutions are not GPU validation. See [VALIDATION](docs/VALIDATION.md) for the RTX PRO 6000 evidence, exact commands and remaining media matrix.

Sampling logs expose `sol_source`, `sana_revision`, `sol_backend`, `sol_source_tree_verified`, actual evaluations, SOL-eligible calls, sparse calls, dense warmup, fallback reasons, `dense_provider_failures`, VDN-local SOL calls, `vdn_rectangular_sol_calls`, `vdn_requested_q_rows`, `vdn_kernel_q_rows` (expected 1:1); legacy square-expansion counters remain zero, backend transitions, effective inherited dense providers and arithmetic gates. Spectrum reports backend-history resets and opaque actual calls separately. A successful run with zero sparse calls is valid telemetry and cannot be used as a SOL performance sample.

See [AUDIT](docs/AUDIT.md) for ownership decisions and remaining limitations. GPL-3.0-or-later; see LICENSE and NOTICE.
