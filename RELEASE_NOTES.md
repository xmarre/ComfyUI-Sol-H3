# ComfyUI-Sol-H3 v0.1.3

Patch release restoring the conservative one-evaluation dense SOL warmup after a controlled same-seed video comparison exposed a startup trajectory discontinuity when SOL approximation was enabled from the first sigma-1.0 denoiser evaluation.

## Default restored

`Sol-H3 SOL Attention (Experimental)` now defaults to:

```text
dense_evaluations = 1
dense_layers      = 2
```

`dense_evaluations=1` keeps the first complete denoiser evaluation on inherited dense attention before switching to SOL. `dense_layers=2` remains a separate per-evaluation leading-layer policy and does not add a transformer NFE.

`dense_evaluations=0` remains supported as an explicit maximum-speed mode. This release changes the default because of a demonstrated quality risk; it does not remove the SOL-first path.

## Startup quality evidence

A subsequent controlled same-seed video comparison established a concrete failure mode for `dense_evaluations=0`:

- SOL from the first denoiser evaluation produced an abrupt opening pose/orientation change with heavy early motion smearing before settling into the opposite heading.
- One initial dense evaluation preserved a continuous opening turn in the corresponding comparison.

This demonstrates that SOL-first execution **can** destabilize the initial trajectory when approximate attention acts from sigma 1.0. The available pair does not establish the frequency of the artifact across seeds, prompts, references, resolutions or model variants, so the release does not claim that every `dense_evaluations=0` run is affected.

## Spectrum and NFE trade-off

The v0.1.2 scheduling analysis remains correct. A one-evaluation dense warmup creates a real numerical-backend transition:

```text
first actual evaluation: dense
next numerical route:    SOL
```

Spectrum correctly invalidates incompatible forecasting history across that `dense -> sol` boundary. The first would-be forecast can therefore become an additional actual transformer NFE to establish a SOL-side anchor.

The previous controlled hot evidence remains the measured performance trade-off:

| Run | SOL | `dense_evaluations` | Logical | Actual | Forecast | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

`dense_evaluations=0` avoids the initial backend-history transition and restored `25 actual / 15 forecast` topology parity with the no-SOL control in that test. Relative to the otherwise-matched `dense_evaluations=1` SOL run, it removed one actual NFE and measured 20.80 s (5.9%) lower H3 sampler wall time and 20.37 s (5.1%) lower end-to-end wall time. Arithmetic-gate/calibration time also varied, so the structural result is the removed NFE; the entire wall-time delta must not be assigned to that NFE alone.

Do not weaken Spectrum's history/receipt safety to recover the NFE while retaining a dense-to-SOL transition. The extra actual call is the correct safety consequence of changing numerical attention backends mid-trajectory.

## Migration

Existing saved workflows keep their serialized `dense_evaluations` value. Workflows created or saved under v0.1.2 can therefore remain at `0` after upgrading until the node value is changed explicitly.

Recommended policy:

```text
dense_evaluations = 1   # default: conservative startup trajectory

dense_evaluations = 0   # opt-in: maximum-speed SOL-first path
                         # may save one Spectrum actual NFE
                         # may cause visible startup discontinuity
```

## Documentation and tests

- Restored `Config(backend="sol")` and the ComfyUI node default to `dense_evaluations=1`.
- Added regression coverage for the restored default, the SOL-first opt-in and Flow continuation semantics.
- Reworked `docs/DENSE_EVALUATIONS.md` around the speed/quality trade-off and saved-workflow migration behavior.
- Updated README and changelog to keep the v0.1.2 timing result as historical performance evidence rather than a quality-equivalence claim.

---

# ComfyUI-Sol-H3 v0.1.2

Patch release changing the default SOL trajectory policy so new workflows start SOL on the first real denoiser evaluation instead of burning a full dense evaluation before switching numerical backends.

## Default change

`Sol-H3 SOL Attention (Experimental)` now defaults to:

```text
dense_evaluations = 0
dense_layers      = 2
```

The previous `dense_evaluations=1` default created an initial `dense -> sol` numerical-backend transition. Spectrum correctly invalidates forecasting history across that transition, so the first would-be forecast became an additional actual transformer NFE solely to establish a SOL-side anchor.

v0.1.2 does not weaken Spectrum's history/receipt safety. It removes that unnecessary transition from the default policy. Existing saved workflows retain their serialized `dense_evaluations` value, and `dense_evaluations=1` remains available as an explicit conservative trajectory warmup.

`dense_layers=2` is unchanged. It keeps the first two H3 blocks dense inside each otherwise-SOL denoiser evaluation and does not add a full transformer NFE.

## Controlled hot evidence

Same seed, references, resolution, prompt, sampler, sampling settings and workflow structure; VDN disabled in all three runs:

| Run | SOL | `dense_evaluations` | Logical | Actual | Forecast | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

The new default therefore restores exact NFE/forecast topology parity with the no-SOL control:

```text
40 logical
25 actual NFE
15 Spectrum forecasts

low:    15 actual / 9 forecast
high:    8 actual / 6 forecast
probe:   2 actual / 0 forecast
```

`metrics_00323` starts backend history directly in `phase=sol` and reports `numerical_backend_transitions=0`.

Against the topology-matched no-SOL control, the tested SOL run is 42.28 s (11.3%) faster in H3 sampler wall time and 40.24 s (9.6%) faster end-to-end. This is a controlled measurement for this MiniMax-H3/RTX PRO 6000 deployment instance, not a universal Sol-Attn percentage.

Against the previous `dense_evaluations=1` SOL run, sampler wall falls by 20.80 s (5.9%) and end-to-end wall by 20.37 s (5.1%). Arithmetic-gate/calibration time also varied between hot runs, so the defensible structural gain is the restored forecast replacing one full actual NFE; the entire wall-time delta should not be assigned to that NFE alone.

## Quality boundary

The same-seed `dense_evaluations=0` output is visibly different from the `dense_evaluations=1` output, which is expected when approximate SOL attention affects the trajectory from sigma 1.0. Manual inspection did not establish either video as better or worse.

That supports making zero the operational default for the tested stack, but it is not statistical perceptual-equivalence evidence. Users who want the previous conservative policy can set `dense_evaluations=1` explicitly.

## Documentation and tests

- Added `docs/DENSE_EVALUATIONS.md` with the backend-history rationale, timing evidence, quality boundary, migration behavior and the `dense_evaluations`/`dense_layers` distinction.
- Updated the README and changelog to distinguish historical v0.1.0/v0.1.1 warmup behavior from the v0.1.2 default.
- Added regression coverage verifying that new SOL configs and the ComfyUI node default to `dense_evaluations=0`, while explicit one-evaluation warmup and Flow continuation handling remain supported.

## Telemetry caveat

The existing `dense_warmup` telemetry field counts attention calls kept dense by either `dense_evaluations` or `dense_layers`. It can therefore remain nonzero with `dense_evaluations=0`. Use the configured `dense_evaluations`, backend-history `phase`, `numerical_backend_transitions`, and actual/forecast topology to diagnose whole-evaluation warmup behavior.

---

# ComfyUI-Sol-H3 v0.1.1

Patch release addressing the native-Windows failure reported in issue #4 without claiming unsupported native-Windows SOL kernel execution.

## Fixed

- Fixed a real cross-platform provenance bug: vendored Sana file names are now compared with canonical `/` manifest paths instead of platform-native `Path` strings. A valid Windows checkout no longer fails every nested entry with `Packaged Sana source file set mismatch`.
- Pinned the vendored source snapshot to LF checkout via `.gitattributes`.
- Provenance hashing now accepts only Git-style CRLF-to-LF transport normalization in addition to exact bytes; any other content change still fails closed.
- Added `windows-latest` provenance CI covering Windows path semantics, CRLF normalization and tamper rejection.
- Scoped CuTe/Triton runtime dependencies to Linux, matching the platform actually supported by the packaged kernel path.
- Native Windows now reports an explicit `Linux/WSL2` requirement when the SM120 CuTe backend is unavailable instead of presenting the fallback as a generic backend-selection problem.
- Exact Runtime now fails closed on native Windows before importing or executing its Triton affine kernel and delegates to the untouched native H3 block. This removes the unvalidated Sol-H3 Exact/Triton path from ComfyUI's native-Windows AIMDO malloc-graph lifecycle.

## Native Windows boundary

RTX 5090 is SM120 hardware and is architecturally eligible for the packaged kernel. The current NVIDIA CUTLASS CuTe DSL runtime does not support Windows, so the real `cute_sm120` SOL kernel still requires Linux/WSL2. This release fixes Windows installation/provenance behavior and diagnostics; it does **not** claim native-Windows SOL acceleration.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

See `docs/WINDOWS.md` for the exact boundary and troubleshooting procedure.

## Issue #4 allocator crash

The attached failing request reported `sol_backend=null`, `sol_source_tree_verified=false` and `sparse_calls=0`, so no SOL sparse kernel executed before the later `comfy_aimdo` `malloc_graph_pop` access violation. The same v0.1.0 request had `exact_fusion=true` and `exact_blocks=600`, so Exact Runtime was the only Sol-H3 custom kernel path that actually executed.

v0.1.1 does not claim that Sol-H3 caused the AIMDO failure. Instead, native Windows now automatically delegates Exact Runtime to native H3, while SOL remains a dense fallback because CuTe is unavailable. If `comfy_aimdo` still fails after upgrading to v0.1.1, the failure is reproducible with both Sol-H3 custom kernel paths absent and should be investigated in the ComfyUI/AIMDO path separately.

## Regression scope

The Linux/WSL SM120 production kernel contract, rectangular Q/KV execution, VDN API-v3 route, Flow mixed-grid route, Spectrum history semantics and zero-copy BTHD behavior are unchanged from v0.1.0.
