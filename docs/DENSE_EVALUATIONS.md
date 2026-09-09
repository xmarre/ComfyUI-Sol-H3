# SOL trajectory warmup (`dense_evaluations`)

## Current default

Starting with **v0.1.3**, `Sol-H3 SOL Attention (Experimental)` defaults to:

```text
dense_evaluations = 1
dense_layers      = 2
```

`dense_evaluations=1` keeps the first complete denoiser evaluation on the inherited dense attention backend and enables SOL afterward. `dense_layers=2` is separate: the first two H3 blocks of each otherwise-SOL evaluation remain dense. The per-layer policy does **not** add a transformer NFE.

Existing saved workflows keep their serialized `dense_evaluations` value. In particular, workflows created or saved with the v0.1.2 default can remain at `0` after upgrading until the value is changed explicitly. The v0.1.3 default affects newly created nodes and programmatic `Config(backend="sol")` users.

## Why v0.1.3 restores one dense evaluation

v0.1.2 changed the default from `1` to `0` after controlled timing showed a real Spectrum interaction:

```text
dense_evaluations=1:
first actual evaluation: dense
next numerical route:    SOL

Spectrum action:
invalidate incompatible dense history
establish a SOL-side actual anchor
```

That `dense -> sol` numerical-backend transition can convert the first would-be Spectrum forecast into an additional actual transformer NFE. With `dense_evaluations=0`, backend history starts directly in `phase=sol`, so that extra actual NFE is avoided.

The scheduling result remains valid. The default is being restored for **quality**, not because the NFE analysis was wrong.

A subsequent controlled same-seed video comparison exposed a concrete startup failure mode when approximate SOL attention acts from the first sigma-1.0 evaluation. With `dense_evaluations=0`, the generated subject showed an abrupt early pose/orientation transition and heavy motion smearing before settling into the opposite heading. With `dense_evaluations=1`, the corresponding opening motion was a continuous turn.

This pair establishes that `dense_evaluations=0` **can** destabilize the initial trajectory. It does not establish how frequently the artifact occurs across seeds, prompts, references, resolutions or model variants. Because the failure is visible and occurs at the trajectory start, v0.1.3 uses the conservative one-evaluation warmup as the default.

## Speed / quality trade-off

The v0.1.2 controlled hot timing remains useful evidence for the aggressive mode. Same seed, references, resolution, prompt, sampler, sampling settings and workflow structure; VDN disabled:

| Run | SOL | `dense_evaluations` | Logical | Actual NFE | Forecasts | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

`metrics_00323` restored exact NFE/forecast topology parity with the no-SOL control:

```text
40 logical
25 actual NFE
15 Spectrum forecasts

low:    15 actual / 9 forecast
high:    8 actual / 6 forecast
probe:   2 actual / 0 forecast
```

It also started backend history directly in `phase=sol` and reported `numerical_backend_transitions=0`.

Relative to `metrics_00321`, the SOL-first run removed one actual NFE and measured 20.80 s (5.9%) lower H3 sampler wall time and 20.37 s (5.1%) lower end-to-end time. Arithmetic-gate/calibration time also varied, so the structural result is the removed NFE; the entire wall-time delta must not be attributed to that NFE alone.

Against the topology-matched no-SOL control, the tested `dense_evaluations=0` SOL run measured 42.28 s (11.3%) lower sampler wall and 40.24 s (9.6%) lower end-to-end wall. That remains deployment-specific timing evidence, not a universal Sol-Attn percentage and not a quality-equivalence claim.

The supported choices are therefore:

```text
dense_evaluations = 1   # default: conservative startup trajectory

dense_evaluations = 0   # opt-in: maximum-speed SOL-first trajectory
                         # may save one Spectrum actual NFE
                         # may cause visible startup discontinuity
```

Do not weaken Spectrum's backend-history invalidation to recover the NFE while keeping a dense-to-SOL transition. The additional actual call is the correct safety consequence of changing numerical attention backends mid-trajectory.

## `dense_layers` is separate

`dense_evaluations` and `dense_layers` control different things:

- `dense_evaluations`: complete denoiser evaluations that stay on inherited dense attention before SOL is allowed.
- `dense_layers`: leading H3 blocks that remain dense inside every otherwise-SOL denoiser evaluation.

v0.1.3 changes only the first default. `dense_layers=2` remains unchanged.

The legacy `dense_warmup` telemetry field counts dense attention calls caused by either policy, so it cannot be interpreted as a count of full dense denoiser evaluations. For trajectory-history diagnosis, use the configured `dense_evaluations`, backend-history `phase`, `numerical_backend_transitions`, and actual/forecast topology.

## Compatibility and safety

- Spectrum's numerical-backend history/receipt checks remain unchanged and fail closed across real route changes.
- `dense_evaluations=0` remains fully supported as an explicit performance policy; this release changes the default and documents its quality risk rather than banning the mode.
- Flow progressive high-stage continuation handling remains supported and prevents a trajectory-start one-evaluation warmup from being spuriously restarted at a known continuation boundary.
- Unknown or malformed routing still executes actual/inherited attention rather than forecasting across an unproven numerical route.
- Saved workflows are not rewritten on upgrade. Audit the serialized node value if a workflow was created under v0.1.2 and startup continuity matters.
