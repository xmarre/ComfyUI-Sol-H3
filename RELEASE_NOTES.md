# ComfyUI-Sol-H3 v0.1.0

Initial production-validated release of native MiniMax-H3 Exact Runtime and composable Sana Sol-Attn for ComfyUI.

## What ships

- The real Sana Sol-Attn source pinned from `xmarre/Sana` revision `2936c47637380842aaa4a4488fac5006cc542b70`.
- Real CuTe `cute_sm120` execution on Blackwell SM120.
- Rectangular attention: `Q [B,Tq,H,128]`, `K/V [B,Tkv,H,128]`.
- Exact native MiniMax-H3 affine/runtime optimization.
- VDN provider API v3 with requested-Q-only execution and unchanged restricted K/V semantics.
- Flow API-2 mixed-grid SOL routing while VDN retains its learned gate.
- Spectrum numerical-backend history/receipt coordination.
- Untwist preprocessing exactly once and audited Diff-Aid/Flow replacement composition.
- Layout-sensitive arithmetic calibration and fail-closed fallback/history behavior.
- Zero-copy BTHD input handling for suitable strided layouts.

## Production validation

Validated on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120)** with PyTorch `2.10.0+cu130` / CUDA 13.0.

Final Spectrum topology:

```text
sampler_logical_calls       18
transformer_actual_nfe      14
spectrum_forecast_calls      4

low:    8 actual / 2 forecast
high:   4 actual / 2 forecast
probe:  2 actual / 0 forecast
```

The SOL-bypassed control is `13 actual + 5 forecast`; the one additional SOL actual is the intentional initial `dense -> sol` backend transition and remains a safety anchor.

VDN API-v3 native routes are 1:1 requested/kernel Q rows with no square expansion:

| Stage | Rectangular SOL calls | Requested Q rows | Kernel Q rows | Square expansion |
|---|---:|---:|---:|---:|
| Native low | 1,584 | 3,744,000 | 3,744,000 | 0 |
| Native high | 1,056 | 4,972,800 | 4,972,800 | 0 |
| Later native high | 1,248 | 5,967,360 | 5,967,360 | 0 |

The historical VDN v2 compatibility bridge expanded Q work by roughly **4.4–5.4x** in affected stages. API v3 removes that kernel-row expansion without changing VDN's trained attention-domain ownership.

Representative Flow mixed-grid route:

```text
external_mixed_sol_calls           144
external_mixed_q_rows         6,270,480
external_mixed_kernel_q_rows  6,270,480
compatibility_fallbacks              {}
```

## Zero-copy BTHD result

Exact production mixed layout:

```text
shape      [1, 43545, 56, 128]
Q/V stride [7168, 21504, 128, 1]
K stride   [7168,  7168, 128, 1]
```

Seven-run isolated real-SM120 medians with identical Q/K/V values:

| Path | CUDA median | Host-wall median |
|---|---:|---:|
| strided zero-copy | **46.768 ms** | **42.338 ms** |
| pre-contiguous kernel | 46.941 ms | 42.376 ms |
| old copy + kernel | 47.727 ms | 43.136 ms |

The strided kernel is effectively parity with pre-contiguous execution; the historical copies add about `0.786 ms` CUDA / `0.761 ms` wall per representative mixed call. The old bridge materialized `1,248,522,240` bytes per mixed call. Across 144 calls, zero-copy avoids about **167.44 GiB** of redundant Q/V materialization and roughly **0.11 s** of direct copy overhead.

This is a micro-optimization result, not a claim of a large end-to-end speedup.

## Timing evidence

Historical matched Exact-only A/B:

| Exact Runtime | Sampler | End-to-end | Peak VRAM |
|---|---:|---:|---:|
| off | 263.56 s | 312.57 s | 17.18 GB |
| on | **247.30 s** | **298.69 s** | 17.18 GB |

A same-process hot SOL run with the final 14/4 topology measured `274.87 s` end-to-end / `229.13 s` sampler, versus `287.51 s` / `240.55 s` for the 13/5 bypass control. Because NFE count, sparse routing/content and cache state are not a controlled A/B, this release deliberately does **not** convert those figures into a large SOL percentage claim.

## Companion interoperability

The reviewed companion pins are:

```text
Spectrum #104  9c682c07f4c5ea9de601cda234755a1561b59f59
Untwist #9     cf428e204f42354ce9a9582dd956906f75a52974
VDN #8         b6f0755c4172ec5c17386c56998f454e78b2a2d4
VDN #11        5b63dc670229d419a6350b64f7ceda609dbc8194
Flow v0.3.2    fe0ef8752b92081b5a85bc9b39ad8e2a7037d591
Diff-Aid       ba9d9efbcf7e64c755e068cb76547d8cc85481eb
```

VDN #11 applies after the still-unreleased VDN #8 audio-fidelity overlay, so that integration remains a pinned companion stack rather than a mainline VDN release at the time of Sol-H3 v0.1.0.

## Boundaries

- Production GPU validation is Linux/WSL on SM120; native Windows kernel execution is not yet validated.
- Only Flow's explicit validated API-2 mixed contract is recognized; arbitrary mixed/external layouts are not inferred.
- Unknown replacement/history topology fails closed to actual/inherited attention.
- Optional SageAttention is an inherited dense provider and is not bundled.
- Successful sparse routing is not by itself a decoded-media quality or whole-workflow speed claim.

See `README.md`, `docs/VALIDATION.md`, `docs/RECTANGULAR.md` and `CHANGELOG.md` for the full implementation and evidence record.
