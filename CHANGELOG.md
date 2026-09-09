# Changelog

## v0.1.3 — 2026-09-09

Restores the conservative one-evaluation dense SOL warmup after a controlled same-seed video comparison exposed a startup trajectory discontinuity when SOL approximation was enabled from the first sigma-1.0 evaluation.

### Changed

- `Sol-H3 SOL Attention (Experimental)` defaults `dense_evaluations` back to `1`.
- Programmatic `Config(backend="sol")` uses the same `dense_evaluations=1` default.
- `dense_evaluations=0` remains supported as an explicit maximum-speed mode. With Spectrum it can avoid one actual transformer NFE by eliminating the initial `dense -> sol` history boundary.
- `dense_layers=2` remains unchanged.
- Existing saved workflows keep their serialized value; workflows created or saved under v0.1.2 can therefore remain at `0` until changed explicitly.

### Quality evidence

A subsequent controlled same-seed comparison established a concrete failure mode for the SOL-first policy. With `dense_evaluations=0`, the opening motion showed an abrupt pose/orientation change with heavy early smearing before settling into the opposite heading. With `dense_evaluations=1`, the corresponding opening motion remained a continuous turn.

This is evidence that SOL from the first denoiser evaluation **can** destabilize the initial trajectory. It does not establish the frequency of the artifact across seeds, prompts, references, resolutions or model variants. The default is restored to `1` on quality-risk grounds.

### Spectrum / performance boundary

The v0.1.2 scheduling analysis remains valid: `dense_evaluations=1` creates a real `dense -> sol` numerical-backend transition, and Spectrum correctly invalidates incompatible forecasting history across it. The first would-be forecast can therefore become an additional actual transformer NFE to establish a SOL-side anchor.

The previous controlled hot evidence remains the measured speed trade-off:

| Run | SOL | `dense_evaluations` | Logical | Actual | Forecast | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

Do not weaken Spectrum's history/receipt invariant to recover that NFE while retaining a dense-to-SOL transition. The extra actual evaluation is the correct consequence of crossing numerical attention backends mid-trajectory.

See `docs/DENSE_EVALUATIONS.md` for the current default, migration behavior, speed/quality trade-off and telemetry guidance.

## v0.1.2 — 2026-09-09

Changes the default SOL trajectory policy from one full dense denoiser evaluation to SOL from the first real denoiser evaluation.

### Changed

- `Sol-H3 SOL Attention (Experimental)` now defaults `dense_evaluations` to `0` instead of `1`.
- Programmatic `Config(backend="sol")` uses the same `dense_evaluations=0` default.
- `dense_layers=2` is unchanged; the first two H3 blocks remain dense inside each otherwise-SOL denoiser evaluation.
- Existing saved workflows keep their serialized `dense_evaluations` value. Users can set `dense_evaluations=1` explicitly to retain the previous conservative trajectory warmup.
- Spectrum backend-history safety is unchanged. Real numerical-route changes still invalidate incompatible history; the new default simply avoids creating the initial `dense -> sol` transition.

### Controlled evidence

Same-seed, same-reference, same-resolution, same-prompt, same-sampler, same-workflow hot runs with VDN disabled:

| Run | SOL | `dense_evaluations` | Logical | Actual | Forecast | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

`metrics_00323` restores exact 25A/15F topology parity with the no-SOL control and reports backend history in `phase=sol` from the first call with `numerical_backend_transitions=0`. In this controlled hot pair, SOL is 42.28 s (11.3%) faster in H3 sampler wall time and 40.24 s (9.6%) faster end-to-end. These are deployment-specific measurements, not universal SOL percentages.

The `dense_evaluations=0` output is visibly different from the `dense_evaluations=1` same-seed video, as expected for approximate attention affecting the trajectory from sigma 1.0, but manual inspection did not establish either as better or worse. This is not statistical perceptual-equivalence evidence.

See `docs/DENSE_EVALUATIONS.md` for the scheduling rationale, timing interpretation, migration behavior, and the distinction between `dense_evaluations` and `dense_layers`.

## v0.1.1 — 2026-09-09

Patch release for native-Windows installation/provenance behavior reported in issue #4.

### Fixed

- Uses canonical POSIX manifest keys when verifying the vendored Sana source tree, fixing the Windows `Packaged Sana source file set mismatch` caused by `Path` backslash rendering.
- Pins vendored source files to LF checkout and accepts only Git-style CRLF-to-LF normalization in addition to exact packaged bytes; arbitrary tampering still fails closed.
- Adds Windows provenance CI for path semantics, CRLF normalization and tamper rejection.
- Scopes the CuTe/Triton runtime toolchain dependencies to Linux, matching the supported/validated custom-kernel platforms.
- Reports the Linux/WSL2 requirement explicitly when native Windows cannot provide the required `cute_sm120` backend.
- Fails Exact Runtime closed on native Windows before importing/executing the Triton affine kernel and delegates to the untouched native H3 block (`exact:native_windows_unvalidated`).
- Adds `docs/WINDOWS.md` with the supported-platform boundary and AIMDO isolation procedure.

### Scope and remaining boundary

RTX 5090 is SM120 hardware, but NVIDIA's current CUTLASS CuTe DSL does not support native Windows. The real SOL `cute_sm120` kernel therefore still requires Linux/WSL2; v0.1.1 does not claim native-Windows SOL acceleration.

The issue #4 v0.1.0 request reached `sparse_calls=0` / `sol_backend=null` and later failed inside `comfy_aimdo.malloc_graph_pop`, while Exact Runtime had executed (`exact_blocks=600`). v0.1.1 does not claim that Sol-H3 caused the AIMDO access violation. Instead, native Windows now automatically runs neither Sol-H3 custom kernel path: SOL falls back dense because CuTe is unavailable and Exact Runtime delegates to native H3. A remaining AIMDO crash after upgrading is therefore separable from Sol-H3 kernel execution.

The Linux/WSL SM120 production kernel contract and previously validated rectangular/VDN/Flow/Spectrum behavior are unchanged.

## v0.1.0 — 2026-09-09

Initial production-validated release of ComfyUI-Sol-H3.

### Highlights

- Packages Sana Sol-Attn from pinned `xmarre/Sana` revision `2936c47637380842aaa4a4488fac5006cc542b70` and executes the real CuTe `cute_sm120` backend on SM120.
- Adds rectangular attention support with independent Q and K/V lengths: `Q [B,Tq,H,128]`, `K/V [B,Tkv,H,128]`.
- Adds exact MiniMax-H3 affine/runtime optimization.
- Adds composable attention ownership/fallback semantics instead of blanket incompatibility gates.
- Integrates with VDN provider API v3, Spectrum numerical-backend history, Untwist preprocessing, Diff-Aid and Flow's explicit mixed-grid API-2 contract.
- Preserves VDN's restricted K/V domain, learned softmax gate, linear complement, output projection and global/anchor ownership.
- Adds real-SM120 arithmetic calibration and fail-closed routing/history behavior.
- Adds zero-copy BTHD input handling for suitable innermost-contiguous strided layouts.

### Production routing results

Final validated Spectrum schedule:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

The SOL-bypassed control is `13 actual + 5 forecast`; the extra SOL actual is the intentional first low-stage `dense -> sol` backend transition.

Representative native VDN API-v3 stages:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

Representative Flow mixed-grid route:

```text
external_mixed_sol_calls           144
external_mixed_q_rows         6,270,480
external_mixed_kernel_q_rows  6,270,480
compatibility_fallbacks              {}
```

The historical VDN v2 square compatibility bridge expanded Q kernel work by roughly `4.4x–5.4x` in affected stages. API v3 removes that expansion.

### Zero-copy BTHD result

Exact mixed production layout:

```text
shape      [1, 43545, 56, 128]
Q/V stride [7168, 21504, 128, 1]
K stride   [7168,  7168, 128, 1]
```

Seven-run isolated real-SM120 medians:

| Path | CUDA median | Host-wall median |
|---|---:|---:|
| strided zero-copy | **46.768 ms** | **42.338 ms** |
| pre-contiguous kernel | 46.941 ms | 42.376 ms |
| old copy + kernel | 47.727 ms | 43.136 ms |

The strided kernel is effectively parity with pre-contiguous execution while removing the old materialization cost. The old bridge materialized `1,248,522,240` bytes per representative mixed call; across 144 calls the zero-copy path avoids about **167.44 GiB** of redundant Q/V materialization and approximately **0.11 s** of direct copy overhead.

This is intentionally a micro-optimization claim, not an end-to-end speedup claim.

### Timing evidence

Historical Exact-only matched A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | **247.30 s** | **298.69 s** | 17.18 GB |

A same-process hot SOL run with the final 14/4 schedule measured `274.87 s` end-to-end / `229.13 s` sampler. The SOL-bypassed control measured `287.51 s` / `240.55 s` with a different 13/5 schedule. Because routing/content/cache state and NFE count are not a controlled A/B, v0.1.0 does **not** claim a large whole-workflow SOL percentage from those numbers.

### Validation

Validated on NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120), PyTorch `2.10.0+cu130`, CUDA 13.0.

CPU/native CI covers package/source provenance, real Comfy ModelPatcher integration, KJ behavior, Spectrum history/receipts, VDN #8 -> #11 composition, provider API v3/lazy v2 fallback, Diff-Aid wrapper orders, Flow marked-layout and mixed-grid history, mixed -> native resumption, direct benchmark invocation and zero-copy layout contracts.

### Companion pins

```text
Spectrum #104  9c682c07f4c5ea9de601cda234755a1561b59f59
Untwist #9     cf428e204f42354ce9a9582dd956906f75a52974
VDN #8         b6f0755c4172ec5c17386c56998f454e78b2a2d4
VDN #11        5b63dc670229d419a6350b64f7ceda609dbc8194
Flow v0.3.2    fe0ef8752b92081b5a85bc9b39ad8e2a7037d591
Diff-Aid       ba9d9efbcf7e64c755e068cb76547d8cc85481eb
Sana           2936c47637380842aaa4a4488fac5006cc542b70
```

VDN #11 remains tied to the still-unreleased VDN #8 overlay and is therefore documented as a pinned companion rather than a mainline VDN release.

### Known boundaries

- Production GPU validation is Linux/WSL SM120. Native Windows kernel execution is not yet validated.
- Arbitrary mixed/external layouts are not inferred; only the explicit validated Flow API-2 contract can use the mixed SOL route.
- Unknown replacement/history topology fails closed to actual/inherited attention rather than being broadly allowlisted.
- Sparse routing success is not itself a decoded-media quality or end-to-end speed claim.
- Optional SageAttention remains an inherited dense provider and is not bundled by Sol-H3.
