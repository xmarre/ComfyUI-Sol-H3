# SOL trajectory warmup (`dense_evaluations`)

## Current default

Starting with **v0.1.2**, `Sol-H3 SOL Attention (Experimental)` defaults to:

```text
dense_evaluations = 0
dense_layers      = 2
```

`dense_evaluations=0` means SOL is the numerical attention backend from the first real denoiser evaluation. `dense_layers=2` is unchanged: the first two H3 blocks of each otherwise-SOL evaluation remain dense. That per-layer policy does **not** add a transformer NFE.

Existing saved workflows keep their serialized `dense_evaluations` value. The default change affects newly created nodes and programmatic `Config(backend="sol")` users.

## Why the default changed

The v0.1.0/v0.1.1 default was `dense_evaluations=1`. With Spectrum enabled, that creates a real numerical-backend boundary:

```text
first actual evaluation: dense
next numerical route:    SOL
```

Spectrum correctly treats `dense -> sol` as incompatible forecasting history. The would-be first forecast therefore becomes an additional actual transformer evaluation so that Spectrum can establish a SOL-side anchor. The extra NFE is a consequence of the configured transition; it is not an intrinsic requirement of Sol-Attn.

The fix is **not** to weaken Spectrum's backend-history invariant. It is to avoid creating the unnecessary trajectory-level backend transition by default.

With `dense_evaluations=0`, backend history starts directly in `phase=sol`, so the first actual evaluation is already a valid SOL anchor and the normal Spectrum forecast topology is retained.

## Controlled MiniMax-H3 evidence

The default change was evaluated in a same-seed, same-reference, same-resolution, same-prompt, same-sampler, same-workflow hot comparison with VDN disabled. Only the named component/policy changed.

| Run | SOL | `dense_evaluations` | Logical | Actual NFE | Forecasts | H3 sampler | End-to-end |
|---|---|---:|---:|---:|---:|---:|---:|
| `metrics_00321` | on | 1 | 40 | 26 | 14 | 353.36 s | 400.94 s |
| `metrics_00322` | off | — | 40 | 25 | 15 | 374.84 s | 420.81 s |
| `metrics_00323` | on | **0** | **40** | **25** | **15** | **332.56 s** | **380.57 s** |

`metrics_00323` therefore restores exact NFE/forecast topology parity with the no-SOL control:

```text
40 logical
25 actual NFE
15 Spectrum forecasts

low:    15 actual / 9 forecast
high:    8 actual / 6 forecast
probe:   2 actual / 0 forecast
```

The run also reports `phase=sol` from the first backend-history diagnostic and `numerical_backend_transitions=0`.

Relative to the otherwise-matched `dense_evaluations=1` SOL run (`metrics_00321`), the H3 sampler decreased by 20.80 s (5.9%) and end-to-end wall time by 20.37 s (5.1%). The first progressive chunk fell from 149.13 s to 130.61 s. Not all of that wall-time delta should be assigned to the removed NFE because arithmetic-gate/calibration time also varied materially between hot runs; the topology change itself is the defensible structural gain.

The topology-matched SOL-vs-no-SOL comparison is cleaner:

```text
metrics_00323 SOL, 25A/15F:     332.56 s sampler / 380.57 s end-to-end
metrics_00322 no SOL, 25A/15F:  374.84 s sampler / 420.81 s end-to-end
```

That is a 42.28 s (11.3%) sampler reduction and a 40.24 s (9.6%) end-to-end reduction in this controlled hot pair. It remains one deployment-instance measurement, not a universal SOL percentage.

## Quality boundary

The same-seed `dense_evaluations=0` video was visibly different from the `dense_evaluations=1` output, as expected when approximate SOL attention is allowed to affect the trajectory from sigma 1.0. Manual inspection did **not** establish either output as better or worse.

That is evidence against an obvious regression in this tested case, but it is not a statistical perceptual-equivalence result. Users who prefer the older conservative trajectory warmup can set:

```text
dense_evaluations = 1
```

and retain the previous behavior, including the Spectrum history boundary and possible extra actual NFE.

## `dense_layers` is separate

`dense_evaluations` and `dense_layers` control different things:

- `dense_evaluations`: whole denoiser evaluations that stay on the inherited dense attention backend before SOL is allowed.
- `dense_layers`: leading H3 blocks that remain dense inside every otherwise-SOL denoiser evaluation.

v0.1.2 changes only the first default. `dense_layers=2` remains unchanged.

The legacy `dense_warmup` telemetry field currently counts dense attention calls caused by either policy, so it can remain nonzero with `dense_evaluations=0`. For trajectory-history diagnosis, use the configured `dense_evaluations`, backend-history `phase`, `numerical_backend_transitions`, and the actual/forecast topology rather than interpreting `dense_warmup` as a count of full dense denoiser evaluations.

## Compatibility and safety

- Spectrum's numerical-backend history/receipt checks remain unchanged and fail closed across real route changes.
- `dense_evaluations > 0` remains supported as an explicit user policy.
- Flow progressive high-stage continuation handling for an explicit one-evaluation warmup remains supported; it prevents a trajectory-start warmup from being spuriously restarted at a known continuation boundary.
- Unknown/malformed routing still executes actual/inherited attention rather than forecasting across an unproven numerical route.
