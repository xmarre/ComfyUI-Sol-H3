# ComfyUI-Sol-H3

Native MiniMax-H3 exact-runtime optimization and composable Sana Sol-Attn integration for ComfyUI.

**v0.1.0** packages the real Sol-Attn implementation from [`xmarre/Sana`, branch `sol-engine`](https://github.com/xmarre/Sana/tree/2936c47637380842aaa4a4488fac5006cc542b70/models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn), pinned at revision `2936c47637380842aaa4a4488fac5006cc542b70`. On SM120 it executes Sana's CuTe `cute_sm120` backend; `comfy_kitchen.sol_attn` is not substituted for it.

The release has three parts:

- **Exact Runtime** — exact native H3 affine/runtime optimizations.
- **Rectangular SOL** — Sana/CuTe SM120 attention with independent query and K/V lengths.
- **Composable interoperability** — inherited dense providers, VDN grouped attention, Spectrum backend history, Untwist preprocessing, Diff-Aid and Flow mixed-grid routing can coexist when their ownership contracts are coherent.

Unvalidated combinations are experimental telemetry rather than blanket errors. Hard failures are reserved for broken contracts, unsafe geometry/indexing, failed arithmetic verification or real execution failures.

## v0.1.0 production status

The full production stack has been exercised on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120)** with PyTorch `2.10.0+cu130`.

The final validated stack preserves the expected Spectrum schedule:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

The SOL-bypassed control executes `13 actual + 5 forecast`; the one additional SOL actual is the deliberate first low-stage `dense -> sol` numerical-backend transition and remains a safety anchor.

The real packaged kernel, VDN API-v3 rectangular route, Flow mixed-grid route, Spectrum receipts/history, Untwist preprocessing and zero-copy BTHD bridge all passed production execution. See [Validation](docs/VALIDATION.md) for the full evidence matrix.

## Nodes and composition

Apply MODEL patches and then apply **Sol-H3 SOL Attention (Experimental)** before sampling. `exact_fusion=true` also requests Exact Runtime. A later **Sol-H3 Exact Runtime** node merges with the same lifecycle rather than installing a second one.

Supported composition includes Exact -> SOL, SOL -> Exact and repeated identical applications. Different SOL policies on the same MODEL branch are ambiguous and require separate branches.

SOL wraps existing block replacements. Every attention call either executes SOL or delegates to the inherited owner with an explicit route/fallback receipt. A valid run may legitimately contain zero sparse calls.

Generic `optimized_attention_override` providers such as KJ Sage remain the dense owner for warmup/dense-required rows when usable. Loader-level `ImportError`/`OSError` failures are request-locally demoted to original Comfy attention with telemetry; arbitrary CUDA/runtime compute failures are not swallowed.

## Interoperability companions

The companion contracts are opt-in and are not automatically installed:

- [VDN-H3-Plus #11](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/11) — grouped restricted-domain provider API v3 and lazy v2 square compatibility. It currently applies after the still-unreleased VDN audio-fidelity overlay #8, so it remains a pinned companion rather than a mainline release.
- [Spectrum #104](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/104) — generic numerical-attention backend history and receipt coordination.
- [Untwisting RoPE #9](https://github.com/xmarre/ComfyUI-Untwisting-RoPE/pull/9) — shape-preserving Q/K/V preprocessing contract.

Reviewed pins used by v0.1.0 validation:

```text
Spectrum #104  9c682c07f4c5ea9de601cda234755a1561b59f59
Untwist #9     cf428e204f42354ce9a9582dd956906f75a52974
VDN #8         b6f0755c4172ec5c17386c56998f454e78b2a2d4
VDN #11        5b63dc670229d419a6350b64f7ceda609dbc8194
Flow v0.3.2    fe0ef8752b92081b5a85bc9b39ad8e2a7037d591
Diff-Aid       ba9d9efbcf7e64c755e068cb76547d8cc85481eb
Sana           2936c47637380842aaa4a4488fac5006cc542b70
```

### VDN grouped attention

VDN retains ownership of its trained local-window geometry, global/anchor operations, learned softmax gate, learned linear complement and output projection.

Provider API v3 passes only:

```text
requested local Q rows
VDN's unchanged restricted K rows
VDN's unchanged restricted V rows
leading global/prefix K/V sink count
```

to Sol-H3. No square query domain is constructed for a v3 provider. VDN's v2-only `square_q` / `query_positions` payload is created lazily only for a v2-only provider.

Production invariants:

```text
vdn_requested_q_rows == vdn_kernel_q_rows
vdn_square_expanded_calls == 0
```

Representative native production stages:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The historical v2 compatibility bridge expanded Q work by roughly **4.4–5.4x** in the affected stages. v3 removes that kernel-row expansion while leaving VDN's K/V domain semantics intact.

Global/anchor calls remain VDN-native. Masked Flex remains VDN-native; its existing grouped fallback can use v3.

### Flow mixed-grid external sequence

Flow's explicit API-2 contract:

```text
api = 2
mode = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

is validated against the current packed geometry rather than being forced dense by default. VDN retains the learned external-mode gate; its geometry-dependent linear complement remains disabled in this mode.

Representative final mixed-stage execution:

```text
external_mixed_sol_calls           144
external_mixed_q_rows         6,270,480
external_mixed_kernel_q_rows  6,270,480
compatibility_fallbacks              {}
```

Unknown/stale/malformed external contracts delegate locally to inherited attention. Later target-grid stages resume native rectangular SOL.

### Spectrum history and receipts

Spectrum preflights numerical backend policy and observes actual route receipts before retaining forecasting history. Sol-H3 provides a stable identity for the actual attention ownership chain; Spectrum remains provider-generic.

Production-only history issues found during validation were fixed narrowly:

- audited Diff-Aid activation wrappers are transparent only when Diff-Aid publishes its runtime declaration;
- Flow's marked layout wrapper and mixed-grid wrapper are recognized only when exact marker/closure/geometry invariants agree;
- progressive high stages consume Flow's explicit continuation contract so the default one-evaluation dense warmup is not spuriously restarted;
- unknown/malformed wrappers remain opaque and force an actual call rather than weakening the gate.

Untwist preprocessing remains exactly once. Receipt/provider transitions still reset incompatible history.

## SOL kernel contract

The packaged SM120 implementation supports rectangular BTHD attention:

```text
Q:   [B, Tq,  H, 128]
K/V: [B, Tkv, H, 128]
```

Q owns query launch geometry, Q pooling/tails, output and LSE. K/V own centroids, route groups, exact sink blocks and approximate masses. CuTe compile descriptors and arithmetic-cache keys include Q/K/V geometry and layout.

The bridge preserves BF16 and native H3 scaling. `sink_start=0` keeps the leading prefix/global K/V exact with Sana's outward 64-row block rounding. An all-selected call is checked against independent BF16 SDPA; failed arithmetic calibration is fatal.

Kernel contract identity:

```text
sana-sol-engine-sol-attn-64-rect-sm120-v2
```

### Zero-copy BTHD bridge

Pinned Comfy MiniMax-H3 produces BTHD views from `[T,3*H*D] -> split -> view[T,H,D] -> transpose(0,1).unsqueeze(0)`. v0.1.0 preserves suitable innermost-contiguous strided views instead of forcing `transpose(...).contiguous()` copies.

The arithmetic gate is stride-sensitive, so different Q/K/V layouts cannot reuse a calibration result accidentally.

The isolated real-SM120 A/B used identical Q/K/V values and the exact mixed production stride:

```text
Q/V stride [7168, 21504, 128, 1]
K stride   [7168,  7168, 128, 1]
shape      [1, 43545, 56, 128]
```

Seven-run medians:

| Path | CUDA median | Host-wall median |
|---|---:|---:|
| strided zero-copy | **46.768 ms** | **42.338 ms** |
| pre-contiguous kernel | 46.941 ms | 42.376 ms |
| old copy + kernel | 47.727 ms | 43.136 ms |

Interpretation:

- strided CuTe execution is effectively parity with pre-contiguous CuTe (`-0.37%` CUDA / `-0.09%` wall);
- the old copies add about `0.786 ms` CUDA / `0.761 ms` host wall per representative mixed call;
- the zero-copy path is about `2.01%` faster than old copy+kernel in CUDA timing and `1.85%` faster in host-wall timing for this isolated call;
- the old bridge materialized `1,248,522,240` bytes per mixed call. Across 144 representative mixed calls, zero-copy avoids about **167.44 GiB** of redundant Q/V materialization and about **0.11 s** of direct copy overhead.

This is intentionally a **micro-optimization** claim. It does not explain multi-second whole-workflow variance.

## Performance evidence

### Exact Runtime

Historical matched production A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | **247.30 s** | **298.69 s** | 17.18 GB |

This is Exact-only evidence, not a SOL speed claim.

### SOL whole-workflow timing

A pre-zero-copy same-process hot SOL run with the final 14/4 Spectrum schedule measured:

```text
end-to-end prompt       274.87 s
H3ContinuumSamplerV34   229.13 s
```

That run was faster than the SOL-bypassed control (`287.51 s` end-to-end / `240.55 s` sampler) despite executing one extra actual transformer NFE, but this is **not** claimed as a clean SOL speed percentage because routing/content/cache state were not a controlled A/B.

Subsequent zero-copy workflow runs varied materially (`287.29/239.10 s` and `298.59/247.63 s` end-to-end/sampler) while preserving the same 14/4 schedule. Arithmetic-gate/caching cost also varied. Therefore the release makes no large whole-workflow zero-copy claim; the isolated kernel/layout A/B above is the authoritative zero-copy result.

## Installation

Set your ComfyUI root once and install in the same Python environment as ComfyUI:

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
cd "$COMFYUI_ROOT/custom_nodes"
git clone https://github.com/xmarre/ComfyUI-Sol-H3.git
cd ComfyUI-Sol-H3
python -m pip install -r requirements.txt
```

For an existing checkout:

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
cd "$COMFYUI_ROOT/custom_nodes/ComfyUI-Sol-H3"
git pull
python -m pip install -r requirements.txt
```

Dependencies are PyTorch, Triton `>=3.6,<4` on Linux, NVIDIA CUTLASS DSL with the CUDA 13 extra, CUDA Python and Apache TVM FFI. No Sana checkout, `SOL_ROOT`, special `PYTHONPATH`, runtime source download or linker override is required.

The validated production target is **Linux/WSL on SM120**. Native Windows kernel execution remains unvalidated.

## SageAttention on Blackwell

SageAttention is optional and is not installed by Sol-H3. On SM120 use KJNodes **`auto`**, not `sageattn_qk_int8_pv_fp16_triton`; upstream SageAttention 2 dispatches SM120 away from the unusable Triton path.

If Sage fails with a binary ABI error such as `GLIBCXX_3.4.32 not found`, rebuild the official package against the same ComfyUI Python/compiler/CUDA toolkit. Do not repair it with `LD_LIBRARY_PATH`, `LD_PRELOAD` or runtime preloading. See [SageAttention installation and repair](docs/SAGEATTENTION.md).

## Validation and diagnostics

From the node checkout:

```bash
python -m pip install -e '.[test]'
python -m pip check
python -m ruff check .
python -m pytest -q
```

GPU validation and production telemetry are documented in [VALIDATION](docs/VALIDATION.md). Rectangular ownership, zero-copy layout behavior and approximation boundaries are documented in [RECTANGULAR](docs/RECTANGULAR.md). Source/interoperability provenance is in [AUDIT](docs/AUDIT.md).

Useful counters include:

```text
sampler_logical_calls
transformer_actual_nfe
spectrum_forecast_calls
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
materialized_qkv_bytes
bthd_strides
dense_provider_failures
numerical_backend_transitions
compatibility_fallbacks
```

A successful run with zero sparse calls is valid execution telemetry but is not evidence of SOL acceleration.

## Release notes

See [CHANGELOG.md](CHANGELOG.md) for the v0.1.0 release summary and validation boundaries.

`sol_h3/sol_manifest.json` records original upstream hashes and packaged hashes. `tools/rectangular_sm120.patch` records the functional rectangular changes after import adaptation.

GPL-3.0-or-later; see `LICENSE` and `NOTICE`.