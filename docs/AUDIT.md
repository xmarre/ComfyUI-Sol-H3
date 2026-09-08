# Interoperability source audit — 2026-09-08

This audit records the ownership decisions behind the current rectangular/mixed-grid revision of the existing Sol-H3 PR #1.

## Audited sources

| Source | Inspected/tested revision |
|---|---|
| ComfyUI | `efa6c8f804bff78b46a0fd458ebd2e47bba07a30` in the pinned native-interoperability lane |
| Sana `sol-engine` | `2936c47637380842aaa4a4488fac5006cc542b70`, subtree `models/minimax_h3/Sol-H3/h3_runtime/third_party/sol_attn` |
| KJNodes | `c9869eade9920a1b949de07c4a197156006bcceb` |
| VDN-H3-Plus companion | PR #8 `b6f0755c4172ec5c17386c56998f454e78b2a2d4` followed by PR #11 `5b63dc670229d419a6350b64f7ceda609dbc8194` |
| Spectrum companion | `31de0cb89b472c965794fdc340de857360a58107` |
| Flow mixed-grid producer | `h3_flow_regenerate.mixed_grid.v1` API-2 contract on current Flow main inspected during this revision |
| Untwist | PR #9 preprocessing contract |

The original PR remains the integration vehicle. Development was performed on a mirror branch and is consolidated back to `feature/native-sol-h3` as one implementation commit.

## Reproductions that changed the design

### 1. Blanket interoperability gates were wrong

Earlier revisions treated Sage/KJ, Spectrum, VDN and some external layouts as mutually exclusive with SOL. That was too restrictive. The current rule is narrower:

- if coherent semantics exist, execute experimentally and record the route;
- if ownership is unknown, delegate that operation rather than aborting the generation;
- hard-fail only broken contracts, unsafe indexing, failed arithmetic verification, or actual execution errors.

### 2. VDN local attention is rectangular

Normal VDN grouped-local queries attend to an already-restricted K/V domain containing global/prefix rows plus permitted video-window rows. Requested video Q rows are therefore usually rectangular against that K/V domain.

The historical v2 bridge adapted this to a square-only kernel by expanding Q over the whole restricted K/V domain and selecting requested rows afterward. Production telemetry showed approximately 4.38-5.41x more kernel Q rows than requested in representative stages.

The packaged Sana SM120 path was then modified to support rectangular Q/KV directly. Production native-grid runs established 1:1 requested/kernel Q rows with zero Sol-H3 square expansion.

### 3. VDN API v3 removes the remaining dead square-Q allocation

Even after Sol-H3 stopped executing expanded Q, VDN provider API v2 still constructed `square_q` and `query_positions` before calling the provider. That allocation/gather was no longer useful to a rectangular provider.

VDN PR #11 now publishes provider API v3:

```text
requested q
restricted k/v
sink_rows
```

with dispatch priority v3 -> v2 -> v1. The v2 square compatibility payload is built lazily only when a v2-only provider is installed. Sol-H3 publishes v3, so the current production path no longer requests or constructs `square_q`.

PR #11 remains stack-compatible with PR #8 because the provider integration is implemented in the retained grouped-attention layer and does not edit `vdn_h3/hybrid.py`.

### 4. Flow API-2 mixed-grid attention has coherent sparse semantics

The prior Sol-H3 revision forced all VDN external/mixed sequences to inherited dense attention. Inspection of Flow's published API-2 mixed-grid contract showed a narrower coherent case:

```text
api = 2
mode = dense_gate_no_linear
topology = mixed_grid_low_suffix
```

Flow publishes the exact native carrier count, actual mixed row count, video start, temporal length, protected-prefix length, source rows/frame, and target-prefix rows/frame. VDN explicitly switches to whole-sequence gated attention and disables the geometry-dependent linear complement for that external sequence.

Sol-H3 now validates those values against the current packed layout and allows the underlying whole-sequence attention operation to use SOL. It does not invent a uniform target-grid lattice. The leading non-video/global prefix remains the exact sink.

Unknown or malformed external contracts still delegate. Later native-grid calls may resume SOL.

This new mixed route is covered by CPU/native integration tests. It has not yet received a production GPU/media rerun; the previous live mixed-grid run predates this support and correctly remained dense.

### 5. Optional Sage failure is a provider availability issue, not a Sol-H3 incompatibility

The production environment exposed an installed SageAttention binary whose `_fused` extension required a newer `GLIBCXX` than the system `libstdc++` it resolved. Sol-H3 must not repair third-party ABI problems with global linker mutations.

Loader-level `ImportError`/`OSError` from an inherited dense provider is therefore classified and request-locally demoted to the original Comfy attention callable. Explicit preprocessing remains active. Arbitrary CUDA/runtime compute failures continue to propagate.

## Source and provenance

The complete 51-file Sol-Attn subtree is packaged so its real public dispatch and support imports remain intact. The node does not use `comfy_kitchen.sol_attn`.

Original upstream SHA-256 values remain recorded separately from packaged hashes. Functional changes beyond import adaptation are restricted to:

- `interface.py`;
- `preprocess.py`;
- `sm120/mainloop.py`.

`tools/rectangular_sm120.patch` records those changes. The import-adapted + patched tree is reproducible from the pinned Sana checkout and compared in CI.

The original upstream snapshot is preserved as historical upstream evidence; this project does not rewrite it to claim upstream SM120 validation.

Dependencies required by the packaged path are declared directly: PyTorch, Triton, CUDA Python, NVIDIA CUTLASS DSL with CUDA 13 support, and Apache TVM FFI. No runtime Sana checkout or `PYTHONPATH` injection is required.

## Ownership table

| Situation | Classification | Current behavior |
|---|---|---|
| Generic inherited Sage/dense override | Composable dense owner | Used for warmup/dense-required work; SOL owns eligible sparse calls |
| Optional dense provider loader/import failure | Provider availability | Request-local demotion to original Comfy attention with telemetry |
| Dense provider compute/CUDA failure | Real execution failure | Propagates |
| SOL + Spectrum | Numerical history ownership | Receipt/preflight coordination; route changes reset history instead of aborting |
| VDN grouped local q vs restricted k/v | Coherent rectangular domain | v3 direct rectangular SOL when eligible |
| VDN v2-only provider | Legacy compatibility | Lazy square-Q payload only when actually needed |
| VDN global/anchor | VDN-owned operation | Native VDN attention |
| VDN Flex masked attention | VDN-owned masked operation | Native Flex; existing grouped fallback can then use v3 |
| Flow API-2 `mixed_grid_low_suffix` + `dense_gate_no_linear` | Explicit coherent external topology | Contract-validated whole-sequence SOL, experimental telemetry |
| Unknown/malformed external sequence | Undefined/unsupported topology | Local inherited fallback |
| Later native grid after external stage | New eligible operation | SOL may resume |
| Preexisting block replacement | Composable block ownership | Wrap/delegate current args; unsupported ownership stays native |
| Zero sparse calls | Valid optimization-usage outcome | Finish successfully and report counters |
| Whole-block hook/custom forward | Exact arithmetic ownership uncertain | Native Exact fallback |
| Failed arithmetic gate | Computed invariant violation | Hard failure |
| Missing lifecycle/tampered metadata/cyclic preprocessing | Broken internal contract | Hard failure |
| Malformed affine indices/strides | Unsafe fused indexing | Hard failure |

## Numerical details

Comfy's attention wrapper marks recursion before invoking the override. Sol-H3 preserves that marker on inherited dense calls.

For ordinary packed H3 attention, prefix queries that must stay dense are recomputed through the inherited provider (or the original Comfy fallback after provider demotion). The sparse kernel receives BF16 Q/K/V and native scale.

For VDN v3 local calls, preprocessing occurs on the full post-RoPE packed sequence before VDN gathers the restricted domain. This keeps coordinate-dependent transforms such as Untwist well-defined. SOL then sees only requested Q plus the exact restricted K/V rows.

For Flow mixed-grid external calls, Sol-H3 validates the explicit external contract and uses the packed leading non-video/global prefix as the exact sink. It does not interpret the mixed video rows as a regular native target lattice.

The rectangular all-selected arithmetic gate compares against independent BF16 SDPA and includes both Q and K/V shape in its calibration identity. Arithmetic failure is never converted into a compatibility fallback.

## Optimization decisions

| Component | Decision |
|---|---|
| Native fused QKV projection | Retained |
| Native Q/K norm + RoPE | Retained |
| Native SwiGLU/down projection | Retained |
| Native residual/gate math | Retained |
| Exact affine fusion | Implemented with native-rounding parity checks |
| Rectangular Sana SM120 attention | Implemented in packaged real Sana/CuTe source |
| VDN v2 square kernel bridge | Historical compatibility only; current Sol-H3 uses v3 direct rectangular routing |
| VDN dead square-Q gather for current provider | Removed by PR #11 v3/lazy-v2 dispatch |
| Flow mixed-grid whole-sequence SOL | Enabled only for explicit validated API-2 topology |
| Core Comfy Block Sparse Attention | Separate integration/ownership path |
| Learned FastH3/DMD2/VSA acceleration | Separate learned-model concern; not required here |
| Distributed execution / parallel VAE | Outside this MODEL patch |

## Evidence and remaining limits

Real RTX PRO 6000 evidence exists for:

- Exact Runtime and a matched timing A/B;
- packaged Sana source verification and direct `cute_sm120` execution;
- rectangular SM120 GPU tests;
- production native-grid VDN rectangular execution with 1:1 requested/kernel Q rows;
- production Spectrum/VDN/Untwist/Flow stack completion on the pre-mixed-SOL revision.

Current CPU/native CI additionally validates:

- VDN PR #8 -> PR #11 sequential application;
- provider API v3;
- absence of v2 square allocation when v3 is present;
- real ModelPatcher VDN v3 -> Sol-H3 routing;
- explicit mixed-grid SOL routing;
- mixed-grid -> native SOL resumption;
- malformed external-contract fallback.

Still outstanding are:

- GPU timing of the VDN v3 allocation removal;
- production GPU execution of the newly enabled mixed-grid SOL route;
- decoded-media comparison for that mixed route;
- a matched warmed end-to-end performance A/B for the current full stack.

See [RECTANGULAR](RECTANGULAR.md) for kernel geometry and [VALIDATION](VALIDATION.md) for commands/counters.
