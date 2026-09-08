# ComfyUI-Sol-H3

Native MiniMax-H3 affine fusion and experimental Sana Sol-Attn integration for ComfyUI.

This PR packages the actual Sol-Attn implementation from [`xmarre/Sana`, branch `sol-engine`](https://github.com/xmarre/Sana/tree/2936c47637380842aaa4a4488fac5006cc542b70/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn), pinned at revision `2936c47637380842aaa4a4488fac5006cc542b70`. On SM120 it uses Sana's real CuTe `cute_sm120` backend. `comfy_kitchen.sol_attn` is not used by this node.

The current draft has three distinct pieces:

- **Exact Runtime**: exact native H3 affine/runtime optimizations with production RTX PRO 6000 evidence.
- **Rectangular SOL**: Sana/CuTe SM120 attention with independent Q and K/V lengths, allowing VDN's requested local Q rows to run directly against its unchanged restricted K/V domain.
- **Composable interoperability**: inherited dense providers, VDN grouped attention, Spectrum history, Untwist preprocessing, and explicitly validated Flow mixed-grid external sequences can coexist without blanket hard gates.

Unvalidated combinations are treated as experimental telemetry unless the runtime cannot define coherent semantics. Hard failures are reserved for broken contracts, unsafe indexing, failed arithmetic gates, or actual execution errors.

## Nodes and composition

Connect MODEL patches and then apply **Sol-H3 SOL Attention (Experimental)** before sampling. `exact_fusion=true` also requests the Exact optimization. A later **Sol-H3 Exact Runtime** node merges with the existing lifecycle rather than installing a second one. Exact -> SOL, SOL -> Exact, and repeated identical applications are supported. Different SOL policies on the same MODEL branch remain ambiguous and require separate branches.

SOL wraps existing block replacements. Each attention call either executes SOL or delegates to the inherited owner with an explicit route/fallback receipt. A valid run may legitimately contain zero sparse calls.

Generic `optimized_attention_override` providers such as KJ Sage remain the dense owner for warmup and dense-required rows when usable. Loader-level `ImportError`/`OSError` failures are demoted only for the current sampling request to the original Comfy attention callable, with telemetry. Explicit preprocessing contracts are preserved exactly once. Arbitrary CUDA/runtime compute failures are not swallowed.

## Interoperability companions

- [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11): grouped restricted-domain provider API v3, plus lazy v2 square compatibility and full-domain QKV preprocessing.
- [Spectrum #104](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/104): numerical attention history/receipt coordination.
- [Untwisting RoPE #9](https://github.com/xmarre/ComfyUI-Untwisting-RoPE/pull/9): shape-preserving QKV preprocessing contract.

These companions are not automatically installed.

### VDN grouped attention

VDN retains ownership of its trained local-window geometry, global/anchor operations, learned softmax gate, linear complement, and output projection.

Provider API v3 passes:

```text
requested Q rows
unchanged restricted K rows
unchanged restricted V rows
leading global/prefix KV sink count
```

directly to Sol-H3. No extra square query domain is constructed for a v3 provider. VDN's v2 `square_q`/`query_positions` compatibility payload is now lazy and is created only when a v2-only provider is actually installed.

For the Sol-H3 v3 path:

```text
vdn_requested_q_rows == vdn_kernel_q_rows
vdn_square_expanded_calls == 0
```

Global/anchor calls remain VDN-native. Masked Flex remains VDN-native; if its existing fallback switches to grouped attention, that grouped path can use v3.

### Flow mixed-grid external sequences

Flow's explicit API-2 contract:

```text
api = 2
mode = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

is no longer automatically forced to dense attention. Sol-H3 validates the contract against the current mixed packed row count and layout, then allows the underlying whole-sequence VDN softmax call to use SOL. VDN still owns the external-mode gate semantics and keeps the geometry-dependent linear complement disabled.

This support is intentionally narrow: arbitrary reduced/external layouts are not inferred. Unknown, stale, or malformed contracts delegate to the inherited dense implementation. Successful mixed-grid sparse execution is recorded separately as:

```text
external_mixed_sol_calls
external_mixed_q_rows
external_mixed_kernel_q_rows
route = sol_external_mixed
```

CPU/native integration tests cover this route and mixed -> later-native SOL resumption. A production GPU/media rerun of the new mixed-grid sparse path is still outstanding; the previous live mixed-grid run predates this support and correctly remained dense.

### Spectrum

Spectrum consumes preflight policy plus actual backend receipts. Provider demotion or route changes alter the numerical identity and reset incompatible history instead of aborting execution. Opaque/unpredictable routing executes actual transformer calls rather than forecasting from an unqualified anchor.

## SOL kernel contract

The packaged SM120 implementation supports rectangular BTHD attention:

```text
Q: [B, Tq, H, 128]
K: [B, Tkv, H, 128]
V: [B, Tkv, H, 128]
```

Q owns query launch geometry, Q pooling/tails, output and LSE. K/V own KV centroids, route groups, exact sink blocks and approximate masses. CuTe compile descriptors and arithmetic-cache keys include both Q and K/V geometry.

The bridge preserves BF16 and the native H3 scale. `sink_start=0` keeps the leading prefix/global KV exact, with Sana's outward 64-row block rounding. An all-selected call is checked against independent BF16 SDPA. Failed arithmetic calibration is fatal rather than silently accepted.

The kernel contract identity is:

```text
sana-sol-engine-sol-attn-64-rect-sm120-v2
```

and participates in Spectrum's numerical-history fingerprint.

## Evidence

### Direct packaged Sana/CuTe SM120 execution

On the RTX PRO 6000 Blackwell production environment the packaged source verified and the real `cute_sm120` kernel compiled/executed successfully. The original 4096-row probe reported:

```text
sol_source = sana-sol-engine
sana_revision = 2936c47637380842aaa4a4488fac5006cc542b70
sol_backend = cute_sm120
sol_source_tree_verified = true
sparse_calls = 2
prefix_parity = true
max_abs = 0.0009765625
mean_abs = 4.484307282837108e-05
rel_l2 = 0.003004377940669656
```

The rectangular GPU suite subsequently passed **9 tests** on the same SM120 class of GPU, including rectangular all-selected and sparse-sink coverage.

### Production rectangular VDN execution

A prior live Comfy run with VDN + DiffAid + Untwist + SOL + Spectrum + progressive/Continuum established real rectangular native-grid execution:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square-expanded calls |
|---|---:|---:|---:|---:|
| Native low grid | 2,112 | 4,992,000 | 4,992,000 | 0 |
| First high grid | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later high grid | 1,248 | 5,967,360 | 5,967,360 | 0 |

That run used the previous v2 companion contract, while Sol-H3 already ignored the disposable square-Q values. Current VDN API v3 removes the remaining upstream `square_q` gather/allocation itself. The v3 integration is covered by current CI; its allocation/time improvement has not yet been measured on GPU.

The same prior workflow's mixed-grid stage remained dense because external mixed SOL support did not exist yet. Do not treat that historical zero as evidence against the current mixed-grid route.

### Exact Runtime

Historical matched production A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | 247.30 s | 298.69 s | 17.18 GB |

This is Exact-only evidence and does not establish SOL speed.

### Current CI

The current mirror validation covers:

- `pip check`;
- Ruff;
- full Sol-H3 CPU suite;
- pinned Sana source reproduction;
- real Comfy ModelPatcher integration;
- KJ wrapper behavior;
- Spectrum stack behavior;
- sequential VDN #8 -> #11 application;
- VDN provider API v3 and lazy v2 compatibility;
- Flow API-2 mixed-grid SOL routing and mixed -> native resumption.

GPU/media validation remains separate from CPU CI.

## Installation

Run in the same Python environment as ComfyUI:

```bash
cd /home/toor/ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
git switch feature/native-sol-h3
python -m pip install -r requirements.txt
```

Dependencies are PyTorch, Triton >=3.6,<4 on Linux, NVIDIA CUTLASS DSL with the CUDA 13 extra, CUDA Python, and Apache TVM FFI. No Sana checkout, `SOL_ROOT`, special `PYTHONPATH`, runtime source download, or linker override is required.

The current production target is Linux/WSL SM120. Native Windows kernel execution remains unvalidated.

## SageAttention on Blackwell

SageAttention is optional and is not installed by Sol-H3. On SM120 use KJNodes **`auto`**, not `sageattn_qk_int8_pv_fp16_triton`; upstream SageAttention 2 dispatches SM120 away from its unusable Triton path.

If Sage fails with a binary ABI error such as `GLIBCXX_3.4.32 not found`, do not repair it with `LD_LIBRARY_PATH`, `LD_PRELOAD`, or runtime preloading. Rebuild the official package against the same ComfyUI Python/compiler/CUDA toolkit. See [SageAttention installation and repair](docs/SAGEATTENTION.md).

## Validation and diagnostics

From the current checkout, without changing Git state:

```bash
python -m pip install -e '.[test]'
python -m pip check
python -m ruff check .
python -m pytest -q
```

GPU validation and production telemetry are documented in [VALIDATION](docs/VALIDATION.md). Rectangular kernel ownership and remaining approximation boundaries are documented in [RECTANGULAR](docs/RECTANGULAR.md). The source/interoperability audit is in [AUDIT](docs/AUDIT.md).

Useful successful-run counters include:

```text
sol_backend
sol_source_tree_verified
sol_eligible_calls
sparse_calls
external_mixed_sol_calls
external_mixed_q_rows
external_mixed_kernel_q_rows
vdn_local_sol_calls
vdn_rectangular_sol_calls
vdn_requested_q_rows
vdn_kernel_q_rows
vdn_square_expanded_calls
dense_provider_failures
numerical_backend_transitions
compatibility_fallbacks
```

A successful run with zero sparse calls is valid execution telemetry but is not evidence of SOL acceleration.

`sol_h3/sol_manifest.json` records original upstream hashes and packaged hashes. `tools/rectangular_sm120.patch` records the functional rectangular changes after import adaptation. GPL-3.0-or-later; see `LICENSE` and `NOTICE`.
