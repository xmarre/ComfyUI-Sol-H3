# Production arithmetic validation lifetime and progressive continuation performance

Status: implementation design; performance promotion is blocked on the measurements specified below. This document does not establish a hardware-validated fix. No production runtime change accompanies it.

## 1. Decision and problem

MiniMax-H3 partitioned exact-prefix continuation preserves target-grid protected context while generating its suffix on a smaller grid. The observed implementation is slower than the released full-target continuation. The goal is a net generation-time advantage, including cold costs, without changing the physical attention operator, learned VDN complement, forecast schedule, or exact-prefix contract.

**Do not implement a process-global or model-wide successful-validation cache as the primary fix.** The available measurements do not show expensive repeated validation of an identical ordinary key across sampler lifetimes. They show expensive first encounters of **15 distinct ordinary shape contracts**, plus 11 untimed partitioned checks. CuTe compilation and Triton preprocessing initialization occur inside the gate timer. The existing compiled-kernel cache already survives sampler requests. Skipping an arithmetic gate cannot remove compilation that the production call still needs.

The selected implementation is staged:

1. Add bounded, correctly nested attribution for compilation, preprocessing, arithmetic validation, production execution and source verification. Establish matched cold/primed controls before attributing the regression to arithmetic.
2. Give ordinary and partitioned execution one request-owned verified runtime lease and one bounded arithmetic-validation service, preserving their distinct numerical modes and current history ownership. Remove partitioned per-subcall source-tree hashing. Retain full representative Q/K/V validation and current acceptance limits. Reduce partitioned duplicate arithmetic checks only when their actual all-selected proof keys are equal; structural map validation remains per call.
3. Measure the remaining cost. If heterogeneous VDN work dominates, batch equal-grid frame operations under the exact algorithm specified in section 12. If ordinary high-stage work dominates, first resolve its residency/provider/kernel-time difference under matched conditions. Do not optimize the heterogeneous low stage to explain an ordinary high-stage slowdown.
4. Promote only after the complete cold, primed and invalidated campaign demonstrates net advantage and decoded-media acceptance. Keep released Target Input available throughout.

Sol owns this design because it owns the arithmetic gate, runtime lease and compiler invocation. Flow owns stage-level accounting and comparison; VDN owns heterogeneous learned computation. Continuum text/audio transport is outside the patch scope.

## 2. Audited repository state and topology

Live API and Git reads during this investigation returned the following state. These are repository heads, **not proof of installed runtime provenance**.

| Repository / PR | Head | Base SHA / branch | Head branch |
|---|---|---|---|
| xmarre/MiniMax-H3-Flow-Aligned-Regenerate #49 | `85b953c2cdf8304dbb7da283a9131a3722ff5386` | `b659b311fa548c7e3275db0e6f0a05c037dac730` / main | `mirror/exact-prefix-progressive-v2-20260917` |
| xmarre/ComfyUI-Sol-H3 #15 | `93b3e03f2b7b579aaf55fa0f87f55083b259e25c` | `0208ddbaaa8a94d70cbf29003680a24d6404fc66` / main | `mirror/exact-prefix-partitioned-attention-20260917` |
| xmarre/ComfyUI-VDN-H3-Plus #19 | `1bc9f9cd0f685a28f84664b50951df8241501249` | `d244ae4cc635123826e243aac335b4719c865670` / main | `mirror/exact-prefix-partitioned-vdn-20260917` |
| xmarre/ComfyUI-H3-Continuum-Plus #23 | `abe65a82194bed5308a284c800cef5bd1ef565c4` | `ccf50cddf83124707b866e4df3956e85d41dda4b` / production mirror | `mirror/production-physical-text-transport-20260913` |
| Continuum Plus #24 | `e164609f6907ecd2a30da559cc54f19d2e5ab98c` | #23 head | `mirror/production-phase-aware-audio-20260913` |

Flow/Sol/VDN heads are respectively 65/19/24 commits ahead of their main bases, with no base-only commits in the fetched comparisons. All three PRs are open drafts. Reviews and inline review comments were empty. Their discussion includes skipped automatic reviews; that is not review approval. Flow discussion also preserves superseded import/metrics failures, not new production requirements.

Reported checks for these heads succeeded: Flow source-contracts and Python 3.10-3.13; Sol test/native-interop/windows-provenance; VDN pinned Comfy oracle/current-main smoke/legacy migration. Continuum #23/#24 test 3.10-3.13 and publisher-toolchain succeeded. These are structural checks, not timing or media evidence.

Related open work was inspected: Flow #50 is stacked on #49; Sol #16/#17 and VDN #20/#21 are separate Keyless work; Sol #11-#13, Flow #34-#46 diagnostic stacks and VDN #15-#17 preserve earlier experiments. VDN #8 remains separate audio-fidelity work; #12 concerns AIMDO guard placement. Do not merge, rebase, repurpose or replace these branches to perform this task. Continuum main was `a5b8943844594545301b20d01af5d9e3fa38ae29`; neither main nor a PR title identifies an installed overlay.

No AGENTS.md was found in the three primary checked-out trees. Continuum's PR-head AGENTS.md requires public wrapper contracts, exact-state preservation, no global class replacements and no compatibility allowlists; its text/audio work must remain isolated.

Development topology: dedicated Sol design mirror `mirror/arithmetic-validation-design-20260918`, based on Sol #15. An initial design checkpoint was committed as `f0251e0065b8b0b9454696be5e26b88647917eac`. Implementation must use fresh implementation mirrors and checkpoint on GitHub before risky/destructive changes, hostile diagnostics, long tests and after substantial progress. Preserve the existing three PR identities and stacked dependencies.

## 3. Evidence and runtime provenance

Primary evidence, read in full:

| Run | Log | Metrics |
|---|---|---|
| 00504, released Target Input continuation | `Pasted text(20260917-214202).txt` | `metrics_00504_.json` |
| 00511, partitioned exact-prefix continuation | `Pasted text(20260918-020148).txt` | `metrics_00511_.json` |

The metrics SHA-256 values are `33f8a062e3d1b3cf29571a998b76186c649aba840c50794d8a454135dc323ec8` and `c4b5d19554468f6e841aa7bd00e577f5d59d075c2aa39af695bf90ce7949a522`, respectively. Log text was read through a text interface that normalizes line endings; section 20 gives normalized hashes. Do not call those original-byte hashes.

00510 is not used in the timing attribution. No decoded 00504/00511 media, workflow JSON or startup provenance manifest is present in these four evidence files. Therefore this comparison establishes an observed regression, not a fully matched experiment or quality result.

| Item | Verified from the primary artifacts | Remaining uncertainty |
|---|---|---|
| ComfyUI commit/version; PyTorch/CUDA; comfy-kitchen version | Logs begin at `got prompt`; startup versions are absent | Exact versions/commits unknown |
| GPU / architecture | BF16 CUDA execution and `cute_sm120` route; audited loader requires SM120 | Exact GPU name, driver, clocks and power state absent; user-reported hardware is not an artifact receipt |
| Installed Flow/Sol/VDN/Continuum commits | Runtime route signatures agree with the audited development stack | Exact loaded-file hashes, dirty state and overlay SHAs absent |
| Sol source | 00511 reports Sana `2936c47637380842aaa4a4488fac5006cc542b70`, mapped-neighbor-v4 contract, config fingerprint `60b982185c153e150311119e76a858c3eda22be07da0fd06359651f5dd67f898`; summaries report verified source | No compiled binary identity, CuTe/CUTLASS/Triton version or compiler-cache receipt |
| KJ/Sage | Terminal provider name is KJ `DiffusionModelLoaderKJ...attention_override_sage` | Name does not identify selected Sage binary/version or every runtime dispatch |
| VDN | 00511 applies 50 blocks, grouped backend, radius 1, chunk 5, both anchors, vdn_solve, streamed INT8 ConvRot branch, retained buffers; API 4 active | No branch checkpoint byte hash |
| LoRA/model patches | 00511 default/turbo adapters strength 1; runtime DoRA loader reports 200 model hooks; Spectrum reports 615 patches, 611 keys, 563 recognized LoRA, 51 unknown patches | No complete model/LoRA filenames, tensor/content fingerprints or numerical mutation generation |
| DiffAid | 00511 strength 0.5, blocks 1/13/25/37/50, sigma 0..0.95, conditioning-only | Installed revision absent |
| Spectrum | Actual/forecast receipts and profile diagnostics; video blend .5/audio blend 0 | Installed revision absent |
| Untwist | Not uniquely established by these logs | Must enumerate actual registered preprocess transforms and state |
| Continuum | Both logs show chunk 1 text compiler v2, chunk 2 v3; corresponding text hashes match across the two logs | Cannot equate this with the current #23/#24 source or infer the audio overlay |

00504's first node activity is already-cached trajectory/handoff work; Spectrum profile lookups are all hits. 00511 re-applies VDN, runtime LoRA, DiffAid, Sol and Spectrum and has two profile misses. Model profile base UUIDs differ. Equal patch counts and sensitivity scalars do not prove equal weights or identical allocator/compiler state.

Before hardware work, capture the loaded module `__file__` and file digest, repository SHA/dirty diff/overlay ordering, runtime package versions, device UUID/SM/driver, compiler options, selected attention implementation, model/checkpoint/adapter hashes, wrapper order/configuration and workflow/media/seed hashes. Check repository and installed files independently. A branch name is insufficient.

## 4. Wall-time accounting

Times below are seconds. Sampler/model timers are host wall intervals from Flow; they are not isolated CUDA kernel measurements. `flow_predict_wrapper` reads the timestep scalar before recording elapsed time, and called implementations also synchronize. Do not add gate intervals to model-call times: gates are already nested inside them.

### 4.1 Complete top-level ledger

| Quantity | 00504 | 00511 | Difference |
|---|---:|---:|---:|
| E2E, rounded log value | 329.700 | 457.220 | +127.520 |
| First sampler | 104.382158 | 142.796484 | +38.414326 |
| Continuation sampler | 173.924451 | 224.178864 | +50.254414 |
| Combined sampler | 278.306609 | 366.975348 | +88.668739 (+31.860%) |
| E2E minus sampler | 51.393391 | 90.244652 | +38.851261 |
| Actual model-call intervals | 256.080806 | 342.169370 | +86.088564 |
| Forecast model-call intervals | 4.602436 | 3.323572 | -1.278864 |
| Sampler minus all model-call intervals | 17.623367 | 21.482406 | +3.859039 |

The non-sampler remainder includes setup, decoding, assembly, encoding and other node work; these logs do not time every component. It must remain an explicit unresolved bucket. Node memory-trim `duration_ms` measures trimming, not the enclosed node's execution.

### 4.2 Stages and intended topology

| Run/chunk | Low L/A/F; wall | Probe L/A/F; wall | High L/A/F; wall | Transfer wall |
|---|---|---|---|---:|
| 00504 first | 5/4/1; 43.928455 | 1/1/0; 11.456164 | 3/2/1; 48.399646 | .572415 |
| 00504 continuation | Full-target single stage: **8/6/2**, sampler 173.924451 | none | none | none |
| 00511 first | 5/4/1; 63.905499 | 1/1/0; 12.228165 | 3/2/1; 66.033861 | .604522 |
| 00511 continuation | 5/4/1; 107.686153 | 1/1/0; 26.279481 | 3/2/1; 89.445123 | .730187 |

00504 totals **17L/13A/4F**, 00511 **18L/14A/4F**. No accidental excess NFE is demonstrated in 00511. The control nevertheless has one fewer actual call because it has no continuation handoff probe. Its cost cannot be normalized away when evaluating net benefit.

First sampler stage sums leave .025478 s and .024437 s of unassigned enclosing overhead. The partitioned continuation leaves .037920 s. Learned-upscale time is nested in transfer: .543464 s for 00504 first and .694283 s for 00511 continuation. Guidance/setup/callback/history costs are partly inside model intervals and partly in the sampler remainder; no exclusive sum for them exists.

### 4.3 Every model call, in order within each stage

`A` = actual, `F` = forecast. These are Flow model-call wall times, including nested attention/VDN/patch work.

| Run/chunk/stage | Calls, seconds |
|---|---|
| 00504 first low | A10.359506, A9.647071, F.442744, A9.664969, A9.651959 |
| 00504 first probe | A10.158557 |
| 00504 first high | A22.043973, F.995091, A21.720449 |
| 00504 continuation single | A30.305252, A26.614135, F1.215804, A26.360440, A26.613266, F1.948797, A26.458130, A26.483099 |
| 00511 first low | A12.281669, A24.851608, F0.456137, A10.579650, A10.390465 |
| 00511 first probe | A10.769987 |
| 00511 first high | A37.809367, F1.059658, A22.905417 |
| 00511 continuation low | A24.143754, A35.761065, F0.535812, A21.772811, A21.704510 |
| 00511 continuation probe | A24.929898 |
| 00511 continuation high | A42.032320, F1.271965, A42.236850 |

Use the JSON events for full precision. The stage ledger and aggregate totals are calculated directly from those events.

### 4.4 Gate ledger: measured intervals versus inferred causes

| Lifetime | Ordinary gates | 00504 gate wall | 00511 gate wall | 00511 loader wall |
|---|---:|---:|---:|---:|
| First low | 5 | .041161 | 14.514625 | .025442 |
| First probe | 0 | 0 | 0 | 0 |
| First high | 5 | .121444 | 14.022638 | .016969 |
| Continuation low | 11 partitioned in 00511 | not applicable | **not recorded** | not recorded |
| Continuation probe | 0 | not applicable | 0 | not recorded |
| Continuation target/high | 5 | .122867 | 14.257642 | .002762 |
| Timed ordinary sum | 15 | **.285472** | **42.794905** | **.045173** |

00504 loader sum is .011326 s. `kernel_loader_s` measures import/provenance/wrapper loading, not CuTe compile time. Partitioned gates omit both timers entirely.

The five ordinary first-low (Tq,Tkv) pairs are (2024,12144), (2530,14674), (2530,15180), (2530,13156), (506,10626). First-high pairs are (4224,18194), (5280,23474), (5280,24530), (5280,20306), (1056,15026). Continuation high/target pairs are (4224,18311), (5280,23591), (5280,24647), (5280,20423), (1056,15143). All have 56 heads, width 128, BF16 and matching logged BTHD strides. **No ordinary pair repeats across these lifetimes.** Across the two runs the corresponding geometries do match; that does not establish identical compiler process/cache state.

Partitioned checks have five distinct shape pairs: (1518,19697), (2530,16947), (2530,15847), (2530,13823), (506,11293). The middle pair appears seven times because the current validation identity also includes physical map/group semantics. All 11 report key-measure bias and mapped ABI. Logs omit their strides and canonical bias keys, so shape equality alone cannot certify cache equivalence.

The ordinary gate interval increase is **42.509433 s**, or **47.94%** of the sampler regression. This is interval attribution, not proof that removing validation saves 42.5 s. Compile/initialization and earlier queued GPU work can be inside the interval. Dense-reference dominance is unproven.

Optimistically subtracting every visible 00511 ordinary gate leaves **324.180443 s**, still **45.873834 s** above 00504 combined sampler. For continuation it leaves **209.921222 s**, still **35.996772 s** above the control. These are counterfactual accounting bounds, not predicted runtimes. Partitioned gate time is unknown; a roughly 14 s spike in its second low actual is suggestive, not a measurement of 11 gates. No defensible 50-60 s arithmetic-only saving follows from these artifacts.

### 4.5 Cold state, setup and residual

The nearly shape-independent ~2.8 s ordinary gate costs in 00511, versus milliseconds in 00504, are consistent with first-use compilation. The source invokes CuTe compilation on unseen keys inside the gate and Triton preprocessing uses autotuning. This is the leading explanation for the gate discrepancy, with moderate confidence pending direct timing. A process restart, cache miss or dependency change is not explicitly recorded.

Spectrum profile lookup times sum to .021575 s in 00504 and 1.453007 s in 00511. Two 00511 miss records report build_s 1.927878 and .938956. Do not add these blindly: hit records retain historical build_s, and lookup/build nesting needs the installed Spectrum source. Neither reading plausibly explains tens of seconds by itself.

At comparable forecast boundaries, 00511 reports higher allocated/reserved CUDA memory. Maximum reported forecast peaks are approximately 60,575/84,736 MiB in 00504 and 62,894/88,704 MiB in 00511. These are sampled forecast peaks, not run-wide peaks or proof of paging. 00511 initially reports 41.16 GiB available for the VDN branch, chooses streaming and retained buffers. Model residency, allocator stalls, other GPU work, power and thermal state remain unmeasured.

00511 continuation's second high actual takes **42.236850 s**, after the first actual has encountered the five ordinary gate keys. The control's later target actuals take 26.36-26.61 s. This is evidence of a substantial non-gate difference; it is not explained by request-local calibration. Flow removes partitioned stage options in `finally`; VDN's `partitioned_aware` delegates to its original forward when that option is absent. Thus a source hypothesis that high deliberately continues using heterogeneous linear execution is contradicted by the audited dispatch. Installed-runtime leakage remains testable through receipts.

## 5. Architectural invariants

- Low lifetime: authoritative target-grid prefix with target physical positions/RoPE plus actual source-grid generated suffix with source positions/RoPE. Never resize the prefix into H3 low-grid context.
- Learned 3D transfer may resize context internally; discard its generated prefix and restore the original authoritative target prefix. Preserve exact final prefix and explicit exact probe; high must begin with an actual evaluation.
- Preserve Spectrum histories, provider object/semantic identity and backend-history-v1 transitions. Arithmetic reuse is never a history receipt or forecast entitlement.
- Production Sol remains the real SM120 mapped rectangular, **single-union** operator with mapped-neighbor ABI v4 and physical key-measure bias. Independently thresholded partitions plus LSE merge are not interchangeable with this sparse operator.
- VDN API 4 grouped softmax, raw pre-QK-norm/pre-RoPE linear inputs, learned state/gates/norm/projection and center-aligned FP32 cross-grid temporal interpolation remain intact.
- Keep request/owner/layout/provider validation, generic preprocess contracts, Sage/Sol/Spectrum combinations, managed runtime buffers and the existing fallback. No global VDN monkeypatch or user-facing allowlist.

## 6. Verified source facts and ownership audit

Source anchors below refer to the exact heads in section 2.

| File / function | Proven behavior and lifetime |
|---|---|
| Sol `runtime.py:SamplingWrapper.__call__`, `Request` | Constructs a new Request on every OUTER_SAMPLE; sets/resets ContextVar in try/finally. `sparse_verified`, kernel wrapper, gates and descriptor caches are fresh. Successful sets are discarded with the request, not explicitly cleared between blocks. Nested requests restore the previous token. |
| `runtime.py:install` | Clones ModelPatcher; installs immutable SamplingWrapper/DiffusionWrapper and block wrappers. Wrapper lifetime exceeds request lifetime, but currently owns no successful-proof cache. |
| `sparse.py:attention` | Validates shapes, BF16/device, mapped/weighted descriptors; lazily loads kernel; keys successful gates by device, dtype, Q/K/V shapes+strides, calibration identity and mapped ABI presence. Ordinary mapped values are deliberately excluded. |
| `sparse.py:attention`, `_dense_reference`, `error_metrics` | Gate uses full-KV sink, dense SDPA, finite/mean/relative/peak reductions; then production executes separately with its actual sink. `.item()` calls synchronize repeatedly. Gate start is before the all-selected kernel; no initial stream drain isolates earlier work. |
| `partitioned_request.py:partitioned_request_attention` | Request-owned dynamically created `partitioned_sparse_verified`. Key contains Q/K/V shapes/strides plus a digest including Flow semantics, kind, geometry, sink/exact end and mapped map/descriptor identity. Gate has no wall timing. Device/dtype fixed by validation, not explicitly included in this key. |
| `partitioned_request.py:_sm120_union` | Calls `verify_source()` on **every** sparse subcall, including calibration, then prepare/allocate/lookup/compile/launch. Supports simultaneous mapped metadata and bias directly; public `sol_attn` currently rejects that combination. Do not route it through the public wrapper unchanged. |
| `_vendor/sol_attn/interface.py:_sol_attn_cute`, `_compile_sm120` | Module-local `_compiled` dict and `_compile_lock` survive Requests. Compile key contains device index/SM, batch, Q/KV rows, heads, splits, strides, bias/mapped structural mode. Sink bounds and scale are runtime arguments. Full-sink calibration and production use the same compiled specialization. Partitioned uses its own ABI-prefixed key in the same dict. |
| `_vendor/sol_attn/preprocess.py:prepare` | Triton reductions and threshold work precede CuTe lookup; autotuning/JIT may occur here. Both gate and production recompute input-dependent summaries. |
| `provenance.py:verify_source` | Checks manifest identity, complete vendor file set and hashes (allowing CRLF transport normalization). It proves on-disk packaged provenance at that check, not an arbitrary later monkeypatch or compiled binary identity. |
| `mapped_neighbors.py` | Request-owned CPU wire/descriptor LRU (64) and device descriptors (64 / 4 MiB), with creation events/waits/record_stream. Bounds/owner/map contracts are separate from arithmetic. |
| `partitioned_request.py:_key_bias` | Request-owned 64-entry/4 MiB cache keyed by semantic identity, range, log measure and device; event-protected device metadata. |
| `weighted_measure.py:register`, `prepare`, `preprocess_digest` | Existing model-local owner generation and request-local validated plans; current ownership revalidated on hits. Stateful identity is weakref-generation based. It is not a general model/LoRA mutation tracker or a successful arithmetic cache. |
| Flow `partitioned_scheduler.py:_partitioned_stage_contract` | Owns one runtime object for a low/probe lifetime; removes options in finally. Provider runtime state is a non-dict leaf so copied options preserve its identity. |
| Flow `partitioned_transformer.py:_stage_partitioned_attention_override` | Stable semantic terminal-provider identity, current preprocess closures rebound each call. Do not substitute arithmetic key identity here. |
| VDN `partitioned_runtime.py:_partitioned_vdn_forward` | Every block gathers grouped queries/KV, handles prefix/global/anchor dense work, calls Sol single-union suffix work, and runs the learned heterogeneous complement. |

The existing gate proves **sampled all-selected arithmetic agreement** at the observed input/layout. It does not prove sparse approximation quality, mapped-neighbor routing correctness, arbitrary future Q/K/V numerical behavior, or audiovisual quality. All-selected sink forces every block exact, so descriptor values do not determine the selection outcome in this check. Structural descriptors still require independent validation for the subsequent sparse call.

Current limits: mean_abs <= .002, rel_l2 <= .005, finite tensors, max_abs <= max(.5, 4 * reference_peak_abs). Preserve them. Do not weaken thresholds to obtain cache hits.

## 7. Root-cause conclusions and confidence

1. **High confidence:** gate-timed work accounts for a large measured interval increase; the timer conflates arithmetic and cold initialization. The control really had much cheaper gates.
2. **High confidence:** within-run repeated ordinary validation of equivalent keys is not established; all 15 shape pairs differ. A persistent proof cache alone cannot explain or fix their cold cost.
3. **Moderate confidence:** cold CuTe/Triton initialization is the leading explanation for the ~2.8 s gate pattern. Required evidence: compile misses and exclusive host durations, preprocessing initialization, isolated device events and a primed replay.
4. **High confidence:** a residual difference exists outside recorded gates, including an ordinary high actual. Its causal split among memory/residency, kernel sparsity/inputs, inherited provider work and other runtime differences is unresolved.
5. **High confidence, cost magnitude unknown:** per-subcall vendor hashing and per-frame heterogeneous operations are avoidable source-level overhead candidates. A bounded local CPU probe found verify_source median ~2.85 ms over 100 warm reads; this is not a production-machine estimate.

The design is implementation-ready for attribution and bounded semantics-preserving changes. It cannot honestly certify the complete root cause or a speedup magnitude without the missing hardware evidence.

## 8. Hypotheses that must not be recycled

| Hypothesis / approach | Classification and scope |
|---|---|
| Accidental extra NFE in 00511 | Falsified by counters for this run. Intended probe difference from 00504 remains. |
| Repeated sparse-phase provider churn | Falsified for 00511: 2 creations/4 reuses; only legitimate phase transitions. Earlier failures are superseded. |
| Square-Q expansion | Falsified for these receipts: requested == kernel Q rows, expansion zero. |
| Cheap gates in 00504 were exaggerated | Falsified: .285472 s total recorded ordinary gates. |
| All 42.8 s is dense-reference work | Inconclusive and unsupported; compile/prepare/synchronization are nested. |
| Persistent successful-proof cache saves every visible gate interval | Rejected by source: unseen production keys still compile. |
| Unchanged gate-free high-stage cost | Contradicted by the 42.236850 s last high actual; cause unresolved. |
| 00510 non-sampler anomaly explains 00511 sampler regression | Unsupported; 00510 is excluded from attribution. |
| Sol provider/history weakening restores speed safely | Rejected: changes numerical-history semantics and targets a superseded defect. |
| Resize protected prefix or restore deprecated Mixed-Grid | Rejected: violates required physical semantics. |
| Independent sparse partitions + LSE merge | Superseded prototype; dense algebraic equivalence does not preserve single-union sparse thresholds. |
| Continuum prompt/audio changes as performance fix | Out of scope; no causal evidence supports it. |
| Existing tests/green CI prove quality/performance | False inference. Structural success is already established; decoded acceptance remains separate. |

## 9. Solution options

| Option | Correctness / invalidation | Expected effect | Complexity / visibility | Decision |
|---|---|---|---|---|
| Persistent model/process success cache | Requires complete live mutation identity, compiler generation and data-sensitive acceptance policy; clone/closure mutation can invalidate it | Saves warm arithmetic only, not unseen production compilation; observed control gates total .285 s | High, difficult misses; unsafe if keyed by names/UUID alone | Do not select for initial fix |
| Request-owned bounded validation service | Retains current safety boundary; separates arithmetic key from structural proof | Removes only demonstrably redundant checks; bounds memory | Moderate, explicit miss reasons | Select |
| Preflight calibration | Synthetic values cannot certify live QKV; real values require the forward | Moves cold time; may retain large buffers or add work | Moderate/high; easy to hide costs outside sampler | Reject mandatory preflight; diagnostic compiler warming only, counted in E2E |
| Sampled heads/rows | Can miss tails, exceptional values and a distinct full-shape specialization | Less reference work, uncertain usefulness | High validation burden | Reject as replacement; supplemental oracle only |
| Fuse metric reductions | Preserve FP32 accumulation and exactly the acceptance semantics | Fewer allocations and host syncs; likely small once compiled | Moderate; component timer makes benefit visible | Implement only if measured worthwhile |
| Reuse calibration output as production | Full-sink output differs from sparse output | Invalid shortcut | Low code, unacceptable semantics | Reject except a separately proven identical all-selected operation |
| Reuse prepared summaries within one call | QKV/bias/scale must be immutable; summaries do not depend on sink selection | Avoid repeated prepare during gate+production | Moderate lifetime/aliasing risk | Optional measured follow-up; never cross-call result cache |
| Extend existing executable cache receipts | Preserves specialization and source ownership | Makes cold/warm difference visible; prevents accidental duplicate compilation | Low/moderate | Select; do not add a competing compiled dictionary |
| Request-pinned source-verification lease | Same provenance check at lease entry, pinned loaded implementation | Removes thousands of repeated file reads | Low/moderate; retain explicit generation/invalidation | Select |
| Batch equal-grid VDN work | Must retain interpolation, boundary taps, dtypes and recurrence | Targets steady low/probe cost | Moderate/high; compare with existing oracle | Second phase, triggered by measured component cost |

## 10. Validation ownership, key and invalidation model

### 10.1 Selected lifetime

The narrowest useful safe reuse boundary supported by the current contracts is the **active Request and its pinned runtime generation**. Extend Request with `validation_state` and `runtime_lease`; do not place numerical acceptance in module globals, shared mutable SamplingWrapper fields, `model_options` leaves or an unqualified ModelPatcher attachment.

A Request's lease pins: verified vendor manifest digest, loaded module/callable identities, backend/profile ABI, numerical implementation generation, device context identity and compiler environment. Lease creation runs source verification once. Partitioned and ordinary paths acquire the same request lease lazily; ordinary `state.kernel` may remain a compatibility facade. Do not retain QKV, dense references, outputs or model weights in this lease.

Model clone, LoRA, object-patch and provider changes at a subsequent OUTER_SAMPLE naturally start a new acceptance lifetime. A genuine numerical/provider transition inside a Request clears affected acceptance entries through a **separate** validation-generation hook; it does not change HistoryPolicy identity or its scheduling behavior. On unknown mutable preprocess state, preserve normal execution with conservative validation reuse restricted to the current safe scope. Do not reject a previously supported wrapper solely because its persistent identity cannot be proven.

Immutable device/dtype/source/ABI contract violations remain errors. Mid-request supported provider changes remain normal transitions with renewed validation. Native runtime teardown destroys proof entries; no successful proof survives failed request completion.

### 10.2 Key derivation

Use a versioned `ArithmeticKeyV1` built at the canonical tensors actually passed to the kernel. Keep a separate `PhysicalContract` checked on every production invocation. Existing keys must not simply be copied to a global dict.

| Component | Arithmetic key / boundary | Reason |
|---|---|---|
| Source manifest and loaded implementation generation | Required via lease | Source/binary changes alter arithmetic; Sana upstream revision alone omits packaged patches |
| Compiled executable identity and ABI/options | Required generation; validate after a replacement/recompile | A cleared/replaced executable cache must not inherit unqualified acceptance |
| Device/context/SM | Required | `cuda:0` alone is not a context generation; no cross-device proof reuse |
| PyTorch/CUDA/CuTe/Triton numerical environment | Lease provenance; new environment/new lease | Dense reference and compiled implementation can change |
| Dtype, B/H/D, Q/KV sizes, all tensor strides/layout/alignment class | Required | CuTe specialization, memory access and tails depend on them; validate actual tensors |
| Scale | Required exact scalar encoding | Changes softmax arithmetic; do not assume all callers use 1/sqrt(128) |
| Mapped ABI enabled/version | Required | Distinct compiled structural path |
| Mapped values / group index / owner token | Keep in physical proof, not automatically arithmetic equivalence | Full-sink check does not certify routing values. First implementation may keep conservative existing partition identity; quotient only after tests prove identical bias/layout/ABI |
| Bias enabled, canonical bias range and exact FP32 log-measure value | Required | All-selected softmax still depends on bias. Include converted reference-mask dtype; arbitrary bias needs immutable content identity or conservative miss |
| Sink/tau/threshold/splits | Retain current Config fingerprint conservatively; splits/profile required | Full-sink gate forces selection, but production routing still needs validation; no settings change may silently inherit a more permissive policy |
| VDN grouping/external semantics | Per-call physical ownership; arithmetic mode discriminator | Different numerical bias/layout requires new gate; same shape alone does not prove equal domain transport |
| Terminal dense provider | History/physical identity unchanged; validation generation on numerical change | Gate uses PyTorch SDPA rather than this provider; provider history must not be conflated with the oracle |
| Model/LoRA/patch state | Request boundary plus in-request mutation generation | Gate is sample-based; no universal model-independent future-input certificate exists |
| QKV preprocess transform | Current physical preprocessing + numerical generation | Fresh closure object identity alone may churn; function name alone misses mutable semantics |
| QKV values / timestep | Not hashed for existing request-level sampled policy | They vary every layer/step; the existing policy is representative validation, not a theorem for all inputs. Retain fail-closed arithmetic failures and media gates |
| Autograd / torch.compile / capture | Preserve current eligibility checks; never bypass because of hit | Partitioned path rejects them; ordinary cache changes must not widen support implicitly |

A narrowed partitioned arithmetic key may omit map/group identity **only** after proving the full-sink selection is independent of map values, preserving all physical wire validation, and canonicalizing the exact bias range/value. Shape-pair repetitions alone are insufficient. Initially retain current acceptance frequency while adding telemetry; switch to the narrower key in an independently tested commit.

### 10.3 Bounds and concurrency

Use one request-owned LRU with at most 256 successful entries and a 256 KiB canonical-key/record payload budget. Entries contain scalar metadata and gate metrics only. Oversized keys bypass reuse and validate normally. Evict least-recently-used success; eviction produces a future miss, never a permissive fallback. Log aggregate evictions and at most 32 examples per Request. Existing device descriptor/bias caches keep their independent 64-entry/4 MiB limits and CUDA event discipline.

Use per-key in-flight state guarded by a lock; GPU work must occur outside the lock. Publish success only after all reductions complete and generation is rechecked. Exceptions/cancellation remove in-flight state and wake waiters without success. A stale generation at completion cannot publish into the new generation. Reentrant same-key execution must bypass shared reuse, not wait on itself. A hit contains no live output tensor, so it requires no CUDA event wait for QKV; device descriptor caches still do. ContextVars isolate Requests, but copied contexts can expose the same object to workers, hence explicit locking.

Keep the existing process-local executable cache separate. First phase adds receipts and generation identity without changing its eviction behavior. A broad executable-cache eviction policy is not part of this fix; it needs safe in-flight binary lifetime management and may create new cold regressions.

### 10.4 Why wider reuse is not yet safe or useful enough

A frozen Config, a model UUID, a weakref provider token or unchanged shape is not a complete numerical snapshot. Comfy ModelPatcher clones copy patch UUIDs and object/wrapper state while sharing underlying model storage; attachments may also be shared without an explicit clone hook. Runtime LoRA hooks and stateful preprocess closures can mutate independently. The existing weighted-measure weakref-generation machinery prevents identity-address reuse but does not certify immutable closure contents.

If future measured warm arithmetic cost justifies persistent reuse, its narrow candidate is an explicitly immutable **model execution epoch plus compiled-runtime generation**, shared only through a clone-aware lease. Every participating mutable component must supply a trustworthy generation, or requests retain current validation. Do not invent a finite hash of arbitrary Python state and call stale reuse impossible. This extension is not an implementation requirement or a performance claim of this design.

## 11. Ordinary and partitioned handling

Both routes use the same lease/service and gate metrics. Keep separate mode tags for ordinary unweighted, ordinary weighted and partitioned single-union; they do not share successful entries until canonical inputs, reference policy and executable profiles are proven equal.

Ordinary validation still invokes full-sink kernel + dense SDPA + existing errors, followed by its sparse call and any required prefix-query recomputation. Partitioned validation still includes log-measure bias and mapped ABI together in one operation. Do not accidentally route simultaneous bias+map through `interface.sol_attn`, whose current public guard rejects it.

No shared acceptance cache owns Spectrum state, mapped descriptor tensors, VDN scratch, Flow providers, dense-provider disable sets or native/exact-fusion checks. The new service replaces only duplicated arithmetic acceptance bookkeeping, not those independent safety mechanisms.

## 12. Residual cost and exact implementation plan

### 12.1 Geometry and lower-cost opportunity

Continuation has 62 video latent frames: 12 protected prefix frames at 1056 token rows/frame and 50 suffix frames at 506. Thus partitioned video rows are **37,972** (12,672 prefix + 25,300 suffix), compared with **65,472** full-target rows. Including 6,695 non-video rows, total rows are **44,667 versus 72,167**, a 38.11% reduction. Relative to fictitious all-low rows 38,067, the exact prefix adds 6,600 rows. Those rows are required work.

The measured steady partitioned-low actuals are 21.77/21.70 s, versus about 26.5 s full target: a much smaller reduction than row count, with more than linear attention-domain effects. Prefix/global/anchor queries remain dense; suffix local domains include required context. The low/probe execution also runs per-frame convolutions, statistics and readout in Python. VDN costs are not separately timed in the old artifacts.

Sol partitioned `_sm120_union` hashes the source tree for each sparse subcall: the continuation low summary reports 1650 sparse calls, plus 11 arithmetic invocations. This repeated host I/O is not a calibration-only cost. It must be removed through the verified Request lease before considering a kernel redesign.

The explicit probe costs 26.279481 s enclosing wall, including 24.929898 s model wall. Keep it. Transfer costs .730187 s and is not the main target. High consumes 89.445123 s; its second actual remains slow without a new gate.

### 12.2 Per-repository patch sequence

| Order | Repository / exact locations | Required change and boundary |
|---|---|---|
| A | Sol `sparse.py:attention/error_metrics`; `partitioned_request.py:partitioned_request_attention/_sm120_union`; vendor `interface.py:_compile_sm120/_sol_attn_cute`; `runtime.py:SamplingWrapper` | Add bounded attribution, compiler-key/generation receipts, partition gate timing and Request summary. Initially preserve calls, keys and failures. Vendor edits require regenerated packaged manifest and maintained source patch artifacts. |
| B | Sol new `validation.py`; `runtime.py:Request`; `sparse.py:load_kernel`; `partitioned_request.py:_sm120_union`; `provenance.py` | Introduce pinned request runtime lease and bounded success service. Verify source once per lease; pin loaded implementation identity. Preserve first-unseen full gate. Unit-test invalidation/failure before routing both callsites through it. |
| C | Sol validation key builder + partitioned callsite | Separately prove and remove only duplicate map/group-dependent arithmetic keys whose bias/layout/profile is identical. Keep all map/owner/history checks. No universal cross-request cache. |
| D | Flow `h3_flow_regenerate/runtime.py:flow_predict_wrapper/flow_sample_wrapper`, `partitioned_scheduler.py`, `metrics.py`; `partitioned_runtime_gate.py` / `tools/check_partitioned_runtime_evidence.py` | Correlate request/stage/evaluation IDs and summaries with actual/forecast events; retain exact-prefix/provider validators. Missing timing is unknown, not zero. Add offline exclusive accounting. No sampler or prompt-routing change. |
| E | VDN `partitioned_runtime.py:_partitioned_vdn_forward`, `partitioned_linear.py:_variable_features/_core_readout/partitioned_linear_readout`; released `branch.py` and runtime buffer lease | Add sampled ranges for gather/preprocess/softmax/linear/weights. Optimize heterogeneous work only under the phase-2 trigger below. Preserve original implementation as numerical oracle during testing. |
| None | Continuum #23/#24, VDN #8, diagnostic/Keyless stacks | Record installed provenance and keep fixed; do not modify for performance attribution. |

Phase-2 trigger: after priming and attribution, if partitioned continuation remains slower than its matched control beyond paired-run variability, select the largest measured avoidable component. This is a required decision gate, not permission to declare phase 1 sufficient.

For VDN heterogeneous linear dominance, use this concrete algorithm:

1. Partition frame indices into equal-grid runs; batch spatial depthwise convolution for each grid using frame as batch. Preserve row order and original dtypes.
2. Assemble temporal taps in original tap order. Same-grid taps use indexed views of spatial outputs; cross-grid taps use the current FP32 bilinear/align_corners=False mapping, casting back at exactly the existing point. Cache a mapped neighbor only within that invocation and exact source/destination grid pair. Preserve end zero-padding and skip_ends.
3. Batch frame statistics and frame readout by equal token count with exact original measure multipliers, accumulation dtype and frame order. Scatter results back before the unchanged bidirectional recurrence and alpha/text-state computation. Do not reorder the recurrence or merge prefix and suffix into an average grid.
4. Keep non-fused epilogue semantics until a separate numerical comparison proves any fused replacement. Avoid per-block GPU scalar extraction for a fixed epsilon only after matching its current dtype rounding exactly; replacing it blindly with Python 1e-6 changes the effective value.
5. Use VDN execution-owned scratch/resources. No process-global buffers, no retained raw QKV across evaluations, no mutation of the shared base branch backend.

If ordinary high-stage time dominates, compare exclusive attention/linear/GEMM timings and H2D/residency after the handoff versus full-target control at the same actual coordinate and physical layout. Gather provider/backend receipts and memory/clock samples. If kernels themselves slow down, use the bounded same-input replay below to distinguish input-driven sparsity from residency/environment. Do not infer causation from total rows or change attention thresholds.

## 13. Instrumentation contract

Telemetry must describe non-overlapping components and identify nesting. Always-on counters are CPU-only; no unconditional device-wide synchronization per attention call.

Required summary fields: request ID, stage/evaluation ID, owner generation, route/mode, canonical arithmetic-key digest, compiler-key digest, hit/miss counts, miss reasons (`new_request`, `new_geometry`, `new_bias`, `new_runtime`, `numerical_transition`, `evicted`, `unrepresentable_identity`), validation attempts/failures, compilation hits/misses, bounded evictions, gate total and untimed count. A hit never claims that the current sparse result was compared with dense.

On misses, host timers separate source verification, prepare/JIT/autotune-inclusive wall, compile-lock wait, compile body, all-selected dispatch, reference dispatch, error-reduction/host-wait wall and total gate interval. Production uses a different counter. Do not name asynchronous dispatch wall `kernel_wall`.

An opt-in bounded diagnostic records CUDA events around prepare, all-selected execution, dense reference, reductions, production kernel, VDN gather and linear components. Drain only at an existing end-of-evaluation/stage boundary. Use at most 128 detailed samples per Request, including first unseen keys and representative later evaluations; summarize overflow counts. Device event spans overlapping host compile may contain idle time, so they are not exclusive kernel occupancy. A bounded profiler trace resolves that ambiguity. Logging and hash construction must not scale with tensor contents on hot hits.

For clean gate attribution, diagnostic mode records an initial stream-drain duration separately before timing the gate. Production must not add that synchronization. Error reductions currently call `.item()` repeatedly; any fused replacement must report the same metrics/limits and reject nonfinite values identically.

The decisive bounded replay captures one ordinary low, ordinary continuation-high and partitioned suffix Q/K/V contract from live execution **after preprocessing and gathering**, cloning before VDN scratch reuse. Replay all-selected+reference twice: first executable use and primed executable use, each with a fresh validation state; then repeat with retained validation state. Record compile/prepare/reference/reduction/prod costs separately. Preserve RNG, provider/history/VDN state, stream ordering and any BSA pool state if present; release snapshots after replay. Never inject replay output into the sampling trajectory. This separates executable priming from proof reuse without extra H3 evaluations.

Preflight warming, diagnostic replay and logging time must remain visible outside production timing; do not subtract them from total cost while claiming E2E improvement. Benchmark runs use low-overhead mode after diagnostic causes are resolved.

## 14. Tests

Use existing suites and targeted additions; no long unrelated full-stack campaign is needed for the design document.

- Sol `tests/test_sparse.py`, `test_partitioned.py`, `test_partitioned_history.py`, `test_vdn_v4_runtime.py`, `test_runtime.py`, mapped-neighbor and provenance suites: preserve full-sink then production order, arithmetic failure, true first miss, same-request hit, device/stride/scale/bias/ABI mutation misses, failed-publication cleanup and request destruction.
- New service tests: bounded LRU/bytes, oversized key bypass, owner-generation changes, concurrent first use, waiter exception, reentrancy, generation change during validation, no GPU tensor retention and no stale acceptance after executable replacement.
- Map tests must mutate one descriptor/range/owner with unchanged Q/KV shape: invalid physical metadata still fails on arithmetic hit. Equal all-selected arithmetic must not accept an invalid sparse physical contract.
- Source lease tests: ordinary+partitioned share one verification per Request, new Request rechecks, loaded implementation replacement invalidates, manifest mismatch fails, existing Windows CRLF provenance behavior stays valid.
- Compile telemetry tests use injected fake compilation and controlled clocks: first gate reports compilation, its production call reports hit, fresh Request with retained executable reports compile hit plus validation miss. This distinguishes the two lifetimes without claiming GPU performance.
- Flow runtime gate tests retain provider binding count = logical low+probe calls and transformer count = actual low+probe calls. Require 18/14/4 only for the frozen benchmark workflow; do not hardcode it into generic runtime policy.
- VDN phase 2 compares existing heterogeneous implementation with batching on unequal grids, tiny/odd shapes, prefix boundary taps, all-same-grid reduction, disabled short-conv features, text state, anchor ends, retained/streamed buffers, BF16/FP32 accumulation and failure cleanup. Test input immutability and raw pre-RoPE ownership.
- Cross-repo tests use pinned exact implementation mirror commits and Comfy's generated custom-node loader namespace. Preserve public wrapper composition, Sage only, Sage+Sol, Sage+Spectrum and Sage+Sol+Spectrum. Current CI source pins must be deliberately updated, not silently redirected to mutable branches.

Numerical tensor tests establish implementation equivalence within declared BF16 tolerances; they do not establish perceptual output quality.

## 15. Matched hardware campaign

Freeze one workflow, prompt, seed, reference media, model/LoRA/patch stack, geometry, sampler/scheduler, conditioning, transfer/guidance settings, decoder and driver/dependencies. Save their hashes. Keep Continuum text/audio revision identical in every arm. The sole workflow change between controls is released Target Input versus partitioned handoff; the sole implementation change between partitioned arms is the candidate fix.

Arms: released Target Input semantics, current partitioned implementation, fixed partitioned implementation. Use the same audited source stack when comparing handoff components; also retain the exact released baseline as a separately identified control if source versions differ.

For each arm measure:

1. Fresh process with explicitly documented on-disk compiler-cache state. Report cold-disk and retained-disk conditions separately if both are used. Do not delete a user's shared caches; use isolated cache directories for the experiment where supported.
2. Primed repeat in the same process and same model/runtime, forcing sampler execution rather than Comfy graph-output reuse. Keep compiler cache; explicitly record whether validation state is fresh.
3. Deliberate numerical invalidation, such as supported LoRA strength/preprocess generation change, and a geometry/bias mutation case. Require revalidation where the contract changes. Never mutate weights in place without the runtime's normal lifecycle.

Use at least three paired repetitions of the measured primed condition in interleaved order, and record each cold result. Broaden only if paired variability prevents a conclusion. Record wall E2E/sampler/low/probe/high, actual/forecast count and per-call times, gate components, executable/proof hit rates, peak allocated/reserved and device residency, clocks/power/other workload, decoded video and audio. Preserve audio lock/restore and exact-prefix receipts.

Correctness gates: expected 18L/14A/4F for partitioned benchmark, stable 2/4 provider creation/reuse, no sparse-phase churn, first high actual, API 4, mapped rectangular, requested Q == kernel Q, square expansion zero, exact target prefix throughout/after transfer/final return, every genuinely unseen arithmetic contract validated and deliberately invalidated key rechecked. Also require no stale physical receipt on a cache hit.

Performance gate: fixed partitioned must show a repeatable net advantage over the matched full-target control in total sampler and E2E, not merely lower validation counters. Report distribution and setup costs. Cold behavior must not become pathological; if extra compilation makes cold partitioned slower, document the amortization and do not promote it as an unconditional speedup. Default/fallback choice needs explicit product justification.

Decoded acceptance: inspect motion, continuity, prefix seam, prompt adherence, texture artifacts and audio seam/intelligibility on matched outputs. Numerical exact-prefix checks are necessary but insufficient. Maintain draft status until both timing and decoded acceptance pass.

## 16. Failure behavior and compatibility

A failed arithmetic check publishes no success and preserves existing error behavior. Missing/unknown cache identity means normal conservative validation, not a new compatibility rejection. Failed optional preflight uses the existing released Target Input fallback where that fallback is already supported; never switch operators silently after a corrupt physical contract.

If instrumentation cannot establish exact compilation/binary provenance, mark it unknown and disable widened reuse. Preserve ordinary fallback behavior for unavailable optional kernels. Keep partitioned autograd/compile/capture restrictions and all supported public wrapper contracts. No new allowlist, threshold relaxation, global provider patch or hidden sample-count change is authorized.

## 17. Rollout and promotion

Commit attribution first, then the request lease/service, then any narrower partitioned key, each on implementation mirrors with GitHub checkpoints. Run targeted tests per boundary. Freeze cross-repo SHAs for hardware evaluation. Keep diagnostic traces outside normal runtime paths and keep the released fallback loadable.

No promotion based solely on hot-cache timing, CPU mocks, CI or theoretical FLOP reduction. If the matched cold/primed campaign shows no net advantage, retain the experimental path and the fallback; document the result. Do not present this design as proof that validation caching solves the regression.

## 18. Explicit unresolved questions

- Exact installed source, dependency/compiler versions, model/LoRA hashes and cold-state provenance for 00504/00511 are unavailable in the primary artifacts.
- Exclusive compilation/autotune, dense-reference, all-selected, metric and synchronization costs are unrecorded; partitioned gates have no duration at all.
- The 42.24 s gate-free ordinary high actual needs matched residency/provider/kernel evidence.
- VDN variable-grid linear, gather/metadata, sparse union and repeated source-verification shares are not timed on the production machine.
- Full workflow/reference/seed equality and decoded 00504/00511 acceptance cannot be established from the supplied logs/metrics.

These limit causal certainty and promotion, not the implementability of the prescribed attribution/lease changes. No missing values have been replaced with timings from another run.

## 19. Recheck before implementation

Re-fetch all main/head/base SHAs, open related PRs, reviews, CI and repository instructions. Confirm source still matches each function/contract described here. Inspect installed overlays and loaded files separately. Recheck compiler key structure, error thresholds, actual Request lifecycle, ModelPatcher clone semantics, mapped+biased dispatch, VDN scratch ownership and Spectrum provider identity before editing.

Justified deviations are allowed when live source or new measurements require them. Record the changed assumption, evidence, selected alternative and affected validation gates in this document; do not silently change operator semantics or topology.

## 20. Relevant out-of-tree artifacts and reproduction

Original evidence remains under `/ComfyUI-Sol-H3/` in the user's persistent files:

| Artifact | Stable file ID | Disposition |
|---|---|---|
| `metrics_00504_.json` | `libfile_6bc028cf2f6c8191997d3287bd6cd393` | Preserve; inspect for baseline accounting |
| `Pasted text(20260917-214202).txt` | `libfile_8f7d5c028aa881918266cc63ca0d0568` | Preserve; inspect gate/setup receipts |
| `metrics_00511_.json` | `libfile_77cc239e39348191888731159fa7003e` | Preserve; inspect partitioned stages/counters |
| `Pasted text(20260918-020148).txt` | `libfile_c516c34a80b8819197e794bbd332bc3b` | Preserve; inspect all six Sol lifetimes |

Investigation copies: `/workspace/scratch/770592e214ac/evidence/metrics_00504_.json`, `metrics_00511_.json`, `run00504.log`, `run00511.log`. Logs are LF-normalized UTF-8 text with SHA-256 `708e55490a3f92faaf85ce90793ae5e2d0b13fe98b32130dab1aa08372184823` and `75ef126090c3288d7e7565841ec6a0a4e1fd7da3d18d9fe282c5ee15ca7af381`. Scratch copies may be discarded after comparison; reconstruct them from the named originals if absent. Do not substitute 00510 or another media file.

No production patch, captured QKV, compiler binary, CUDA trace, benchmark result or generated dataset was created by this design task. The local CPU source-hash timing is reproducible and is not a promotion artifact.

Minimal accounting reproduction, from a directory containing a selected metrics/log pair:

```python
import json
from pathlib import Path

metrics = json.loads(Path('metrics_00511_.json').read_text())
log = Path('run00511.log').read_text()
sampler_s = sum(e['fields']['elapsed_ms'] / 1000 for e in metrics['events']
                if e['kind'] == 'sampler_wall')
model_s = sum(e['fields']['elapsed_ms'] / 1000 for e in metrics['events']
              if e['kind'] == 'model_call')
lifetimes = [json.loads(line.split('Sol-H3 ', 1)[1])
             for line in log.splitlines() if 'Sol-H3 {' in line]
gates = [g for lifetime in lifetimes for g in lifetime['arithmetic_gates']]
print(sampler_s, model_s, sampler_s - model_s)
print(sum(g['gate_wall_s'] for g in gates if 'gate_wall_s' in g))
print('untimed gates:', sum('gate_wall_s' not in g for g in gates))
```

Expected 00511 values: sampler 366.975348317 s, model intervals 345.492942404 s, remainder 21.482405913 s, ordinary gate total approximately 42.794905 s, 11 untimed gates. For 00504: sampler 278.306608945 s, model intervals 260.683242041 s, remainder 17.623366904 s, gates .285471621 s, zero untimed gates.

The supplied Sol-Attn paper (2607.24027v1) describes online thresholding and approximate contributions from unselected blocks; it supports keeping sparse operator semantics distinct from a dense all-selected oracle. Sol Engine (2606.23743v2) describes instance-specific optimization. Neither defines this code's validation key/lifetime or proves a caching speedup. DMD2 (2405.14867v2), VSA (2505.13389v5) and Spectrum (2603.01623v1) provide background, not provenance or timing for these runs. Their training/forecast mechanisms are not replacement fixes for this regression.
