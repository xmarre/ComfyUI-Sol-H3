# Production arithmetic-validation implementation notes

This document records the implementation status and evidence for
`PRODUCTION_ARITHMETIC_VALIDATION_LIFETIME_AND_PERFORMANCE_DESIGN.md` at design
revision `0b56156a2b2de0942777ea01eff53d1947ae0e90`. It is an implementation
record, not a performance or release claim.

## Protected base stack

The implementation is stacked on, and does not rewrite or repurpose, these
production-development PRs:

| Component | Preserved PR head | Implementation stack |
| --- | --- | --- |
| Flow | #49 `85b953c2cdf8304dbb7da283a9131a3722ff5386` | #51 / `mirror/arithmetic-validation-flow-20260918`; final contract checkpoint `69d27d813c85dc13437ff8721d0bc6dbec47fe97` |
| Sol-H3 | #15 `93b3e03f2b7b579aaf55fa0f87f55083b259e25c` | #18 / `mirror/arithmetic-validation-runtime-20260918`; final code checkpoint `b46aa1d5150782dc1f6d93c82b73459ef250a029` |
| VDN-H3-Plus | #19 `1bc9f9cd0f685a28f84664b50951df8241501249` | #22 / `mirror/arithmetic-validation-vdn-20260918`; final component-attribution checkpoint `7547f0959754a32deb0feadd11f400dbb3b315e0` |

Continuum Plus #23/#24 and VDN-H3-Plus #8 are outside this patch scope and are
not modified by these implementation stacks.

The implementation preserves the existing numerical acceptance thresholds,
full-sink all-selected oracle, exact-prefix ownership, provider/history policy,
VDN API-4 heterogeneous path, mapped query-position transport, and the
partitioned single-union attention operator.

## Primary evidence

The timing investigation uses only the section-20 evidence pair:

- released Target Input continuation: `metrics_00504_.json` and
  `Pasted text(20260917-214202).txt`;
- partitioned exact-prefix continuation: `metrics_00511_.json` and
  `Pasted text(20260918-020148).txt`.

The supplied evidence supports the following accounting:

| Quantity | 00504 | 00511 |
| --- | ---: | ---: |
| sampler wall | 278.306608945 s | 366.975348317 s |
| model-call intervals | 260.683242041 s | 345.492942404 s |
| enclosing remainder | 17.623366904 s | 21.482405913 s |
| recorded ordinary arithmetic-gate wall | 0.285471621 s | approximately 42.794904 s |
| untimed partitioned gates | 0 | 11 |

The approximately 42.51 s ordinary-gate interval increase is not treated as
removable arithmetic work. CuTe compilation and preprocessing initialization
can occur inside that interval, and the existing process-local executable cache
has a different lifetime from arithmetic proof acceptance. The 00511 final high
actual remains approximately 42.24 s after the ordinary gate shapes have already
been encountered, so a material non-gate difference also remains unresolved.

The four evidence files do not establish a matched cold-state experiment,
installed binary/compiler provenance, or decoded-media acceptance. No speedup or
root-cause conclusion is derived from them.

## Implemented lifetime model

Sol-H3 now owns one `Request`-local `RuntimeLease` and one bounded
`ArithmeticValidationState`.

The runtime lease verifies the packaged source tree once per OUTER_SAMPLE
request and records the manifest/source generation, implementation generation,
device identity, Python/PyTorch/CUDA environment, installed Triton,
nvidia-cutlass-dsl, cuda-python and apache-tvm-ffi versions, and distinct
ordinary and partitioned runtime/compiler identities. Device identity includes
both the OS process ID and a random per-process generation nonce, so same-process
campaign evidence does not rely on a recyclable PID alone. On CUDA it also
records the device name, CUDA UUID when exposed by PyTorch, SM, memory size,
multiprocessor count and driver version when available. Partitioned
`_sm120_union` no longer rehashes the source tree on
every sparse subcall.

Successful arithmetic proof entries remain request-local. The service is
bounded by count and bytes, supports per-key in-flight ownership, wakes waiters
after success or failure, rejects stale publication across validation-generation
changes, and retains no Q/K/V, dense reference, model weight, or output tensor.
A new OUTER_SAMPLE creates a fresh proof lifetime even if the process-global
compiled executable remains resident.

The process-global CuTe executable cache remains separate from the request-local
proof service. New executable-key insertion does not invalidate unrelated proof
entries, but destructive cache mutation is generation-tracked by the transparent
attribution wrapper. Clearing, deleting or replacing an existing executable
advances the live compiler namespace; ordinary and partitioned routes detect the
new namespace on their next call and invalidate Request-local arithmetic proof
entries before building a new key. Replacing the cache object is rewrapped and
also changes the namespace identity. A fresh request still revalidates its first
unseen arithmetic contract while being able to report an executable-cache hit.

Partitioned arithmetic keys deliberately remain conservative and still include
their physical map/group ownership digest. The design permits removing that
identity only after an SM120 replay demonstrates that the all-selected proof is
independent of map values while all structural map validation remains intact.
That proof has not yet been produced, so Stage C narrowing is not implemented.

## Compiler-attribution deviation

The first attribution checkpoint instrumented the vendored
`sol_attn.interface` directly. Hosted contract tests rejected that approach:
the public `sol_attn` signature changed, reviewed source/probe blob identities
changed, and the generated vendor manifest/patch record no longer matched the
established source contract. CPU-contract run `35302113686` captured those
failures.

The selected implementation therefore leaves the reviewed Sana Sol-Attn source
bytes, public API, manifest, compile keys and executable-cache policy unchanged.
`sol_h3/compiler_attribution.py` transparently wraps the existing compile
dictionary, compile lock, `_compile_sm120`, preprocessing entrypoint and compiled
callables while preserving existing entries; it does not introduce a competing
executable cache. A `ContextVar` scopes receipts to the active Sol call. This
records host-side cache hit/miss/race, compile-lock wait, compile-body time,
preprocessing/JIT-inclusive wall time and compiled dispatch enqueue wall time.
The wrapper also generation-tags destructive executable replacement/eviction so
request-local arithmetic acceptance cannot survive a changed compiled runtime.

The compiler-attribution implementation first passed the full Sol CPU-contract
matrix at checkpoint `51f590b476b344e25138bd99376ee3c8912ac785`.
The final Sol code checkpoint
`b46aa1d5150782dc1f6d93c82b73459ef250a029` passed the complete
CPU-contract workflow in run `35347207776`, including the full test suite,
native interop and Windows provenance. VDN checkpoint
`7547f0959754a32deb0feadd11f400dbb3b315e0` passed CI run
`35345618924`. Flow checkpoint
`69d27d813c85dc13437ff8721d0bc6dbec47fe97`, pinned to those exact Sol/VDN
contracts, passed CI run `35347426118`, including the cross-repository source
contracts and Python 3.10-3.13 matrix. These are structural results, not SM120
performance or decoded-media evidence. The reviewed vendored Sana Sol-Attn
source remains unchanged.

## CUDA diagnostics

CUDA attribution is opt-in through:

```bash
export SOL_H3_CUDA_DIAGNOSTICS=1
```

When disabled, production adds no diagnostic synchronization. When enabled,
each Request owns a bounded recorder with at most 128 detailed samples. Gate
misses are preferred; production calls are sampled densely at first and then at
powers of two. The recorder stores CUDA Event objects and scalar/context
metadata only.

For an arithmetic gate, diagnostic mode performs and separately records an
initial stream drain before the normal gate wall timer. It records CUDA-event
spans for:

- preprocessing;
- the complete all-selected call;
- compiled dispatch;
- dense reference;
- error reductions.

Production samples record the complete sparse production call plus preprocessing
and compiled dispatch. Pending CUDA events are drained at the existing H3
model-evaluation boundary, with a final no-op-compatible drain at Request
teardown. Diagnostic synchronization wall time is reported separately. CUDA-event
spans that include host-side compiler delay can contain device idle time and are
not claimed to be exclusive kernel occupancy.

`tools/check_arithmetic_validation_diagnostics.py` validates a saved ComfyUI
log and fails closed if the diagnostic receipt is missing, unresolved, exceeds
its bound, lacks required gate/production spans, or violates a requested
compiler-hit/miss condition.

### Bounded same-input replay

The decisive same-input replay is separately opt-in:

```bash
SOL_H3_CUDA_DIAGNOSTICS=1 SOL_H3_REPLAY_DIAGNOSTICS=1 python main.py
```

For at most one ordinary low contract, one ordinary continuation-high contract,
and one mapped partitioned suffix contract per Request, Sol-H3 clones Q/K/V
after preprocessing/gathering while preserving the compiler-relevant tensor
shape/stride geometry. Shared Q/K/V storage is cloned as one replacement storage
where applicable. Bias and mapped-descriptor inputs are cloned as well.

Each captured contract runs exactly three diagnostic-only arms:

1. fresh arithmetic-proof state at the first observed executable state;
2. a second fresh arithmetic-proof state with the executable cache retained;
3. the same proof state as arm 2, proving retained proof reuse without a third
   arithmetic gate.

Replay calls the Sol arithmetic operator directly and discards every replay
output. It does not re-enter the transformer, Flow provider, VDN runtime,
Spectrum history, or BSA path, and it restores Python/CPU/CUDA RNG state.
Replay snapshots are not retained by the Request after the replay call returns.
The diagnostic can warm the executable cache and therefore must not be used as a
production timing arm; its host wall remains explicit in the Request summary.

A true first-executable claim is evidence-driven rather than assumed. When a
fresh-process campaign is intended to begin from an executable miss, validate it
with `--require-replay-cold-miss`. If the first replay arm reports no compile
miss, the executable was already primed and that run is not relabeled as cold.

Stage C key narrowing is deliberately still blocked. Same-input replay does not
prove map-value independence; physical map/group identity remains in the
partitioned arithmetic key until a separate descriptor-mutation proof establishes
the required quotient safely.


### Partitioned suffix sparse-route discriminator

The mapped partitioned-suffix replay additionally compares the exact production
sparse output against the weighted dense reference on the same detached Q/K/V,
key-bias and mapped-neighbor snapshot. The report publishes finite/max/mean/relative-L2
error metrics under `production_vs_dense_*`; these measurements are diagnostic
evidence only and do not alter the live model output.

A separate explicit causal arm can replace only partitioned local **suffix**
attention calls that would otherwise be sparse with weighted dense attention:

```bash
export SOL_H3_FORCE_DENSE_PARTITIONED_SUFFIX_DIAGNOSTIC=1
```

The override does not change prefix-query forced-dense ownership and does not
replace normal dense warmup calls. Sol summaries publish
`partitioned_diagnostic_dense_suffix_calls` so a run cannot be mistaken for
ordinary production. This arm is intended only to determine whether the
prefix-dense/suffix-sparse routing transition contributes materially to a
partitioned continuation defect; it is not a production fallback or a proposed
permanent dense route.

## Cross-repository attribution

Flow #51 adds request, stage, and model-evaluation correlation identifiers and
derives stage wall/model/remainder accounting from the existing metrics event
stream. Missing correlation produces `null` accounting rather than invented
zero time. The partitioned stage runtime also exposes bounded host-component
accumulators to the VDN path.

VDN #22 records host-side ranges for partitioned API-4 preprocessing, gather,
partitioned softmax, weight residency, softmax epilogue, variable-grid linear
features/statistics/scans/output-gate/state-gather/output, the fixed-epsilon
scalar extraction, and final projection. The fixed-epsilon `.item()` is
measured explicitly because on CUDA it can expose a synchronization cost that
was previously hidden outside the linear-output timer. The recorder is optional
and duck-typed; ordinary released VDN execution is unchanged.

When Sol CUDA diagnostics are enabled, VDN also contributes bounded CUDA-event
spans for those partitioned components through the same Request-owned recorder.
These spans identify nesting and device intervals; they are not additive
exclusive occupancy. Host wall ranges can overlap asynchronous device work and
likewise must not be summed as exclusive time.

No VDN batching or heterogeneous-algorithm rewrite is included. The design
requires the primed attribution result to select the largest avoidable component
before Phase 2 is allowed.

## Required hardware campaign

Promotion requires one frozen workflow/prompt/seed/reference/model/adapter/
patch/decoder environment and the three semantic arms defined by the design:

1. released Target Input;
2. preserved partitioned implementation;
3. fixed partitioned implementation.

For each implementation arm, preserve and record repository SHA/dirty state,
loaded module path and digest, model/adapter hashes, wrapper order, PyTorch/CUDA/
CuTe/Triton/driver versions, device identity, compiler-cache state, workflow hash,
seed and media inputs.

Run a fresh-process cold case first. Do not delete a user's shared compiler
cache; use an isolated cache location where the relevant compiler supports it
and record whether the isolated disk cache begins cold or retained. Then force a
second sampler execution in the same process with the executable cache retained.
That primed run must use a fresh Sol Request rather than ComfyUI graph-output
reuse. Finally run a supported numerical-invalidation and geometry/bias mutation
case and require revalidation of the changed contract.

For ordinary CUDA-attribution arms, start ComfyUI from its root with diagnostics
enabled:

```bash
SOL_H3_CUDA_DIAGNOSTICS=1 python main.py
```

For the bounded same-input diagnostic arm, enable replay as well:

```bash
SOL_H3_CUDA_DIAGNOSTICS=1 SOL_H3_REPLAY_DIAGNOSTICS=1 python main.py
```

Save Flow metrics and a complete **single-prompt log segment** for every
campaign run; each run segment must contain exactly one final
`Prompt executed in ... seconds` receipt while retaining every Sol Request
summary produced by that prompt. For same-process cold/primed/invalidation
sequences, also preserve the uncut master process log as provenance. Validate a
cold request segment that is expected to compile with:

```bash
python custom_nodes/ComfyUI-Sol-H3/tools/check_arithmetic_validation_diagnostics.py \
  --log /path/to/cold-comfy.log \
  --require-compile-miss
```

Validate a same-process primed request, when its exact executable keys were
already compiled, with:

```bash
python custom_nodes/ComfyUI-Sol-H3/tools/check_arithmetic_validation_diagnostics.py \
  --log /path/to/primed-comfy.log \
  --require-no-compile-miss
```

Use the explicit `--request-index` option when a saved process log contains
multiple Sol Request summaries. A compile miss in the primed arm is evidence of
a new executable key and must be investigated rather than relabeled as
validation overhead.

The three same-input replay targets do not belong to one Sol Request in the
progressive benchmark. Flow low/probe/high sampler invocations cross separate
Sol OUTER_SAMPLE lifetimes; the exact 00511 log contains six Sol summaries for
its two chunks. Therefore a whole-prompt replay check must resolve targets
across the process log rather than selecting one `--request-index`.

The checker uses the mapped `partitioned_suffix` replay's
`flow_request_id` as the continuation correlation identity. It then requires
`ordinary_continuation_high` from the high stage with that same Flow request
ID, while `ordinary_low` must come from a low stage under a different Flow
request ID. Ambiguous or missing correlation fails closed. This corrects the
earlier single-summary command without changing replay execution or any
production lifetime.

A fresh-process diagnostic that is expected to demonstrate first executable
compilation can use:

```bash
python custom_nodes/ComfyUI-Sol-H3/tools/check_arithmetic_validation_diagnostics.py \
  --log /path/to/replay-comfy.log \
  --all-requests \
  --require-replay-target ordinary_low \
  --require-replay-target ordinary_continuation_high \
  --require-replay-target partitioned_suffix \
  --require-replay-cold-miss
```

For a partitioned performance-attribution run, require Flow correlation, complete
low/probe/high accounting, VDN host-component receipts, and correlated VDN CUDA
samples:

```bash
python custom_nodes/MiniMax-H3-Flow-Aligned-Regenerate/tools/check_partitioned_runtime_evidence.py \
  --metrics /path/to/partitioned-metrics.json \
  --log /path/to/partitioned-comfy.log \
  --expected-logical 9 --expected-actual 7 --expected-forecast 2 \
  --require-performance-accounting
```

The 9/7/2 values describe the latest partitioned chunk in the frozen benchmark;
whole-run 18/14/4 accounting remains a separate acceptance check.

Use at least three interleaved paired repetitions of the primed fixed-partitioned
and matched full-target control when deciding performance. Record sampler/E2E and
low/probe/high walls, actual/forecast counts, model calls, proof and compiler
hit/miss rates, gate components, VDN component receipts, memory residency and
device clocks/power/workload state.

## Promotion boundary

Structural tests, source-contract CI, proof-cache hit rates, or hot-cache timing
alone cannot promote this path. Promotion requires all of the following:

- no numerical or compatibility regression in the protected Sol/Flow/VDN
  contracts;
- matched cold, primed, and invalidated SM120 evidence;
- a repeatable net sampler and end-to-end advantage over the matched released
  Target Input control;
- exact-prefix and provider/history receipts remaining intact;
- decoded video acceptance;
- decoded audio acceptance, including the continuation seam.

Until those gates pass, the partitioned path remains experimental and the
released Target Input fallback remains available.
