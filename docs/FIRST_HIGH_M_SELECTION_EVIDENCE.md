# First-high Experiment E: saved-witness interpretation and M selection

Status: **evidence checkpoint / bounded follow-up selection**, not a production-fix authorization.

Authority: `docs/FIRST_HIGH_SOL_LOCAL_ROOT_CAUSE_AND_FIX_DESIGN.md` at design commit `0b8715faa0a82c730f5aaf0e44b1185e64291e49`.

This checkpoint records the successful real-SM120 Experiment E result, corrects one diagnostic interpretation defect discovered from the saved witness, and applies the authoritative decision table to select the next bounded media arm. R and W remain complete and must not be repeated. E itself does not need another H3 execution for this interpretation.

## 1. Successful E result

Capture: `234ed062128e43ed8d5ec63e27517b22`.

The completed E report is valid:

- `execution_valid=true`
- `entry_state_exact=true`
- `completed_first_call_only=true`
- `arithmetic_conformant=true`
- complete Core cleanup
- exactly 700 backend receipts, 14 per block:
  - 528 `vdn_local_sol_all_selected_e`
  - 22 `vdn_dense_warmup`
  - 50 `vdn_global_native`
  - 100 `vdn_anchor_native`
- one logical / one actual / zero forecast H3 calls
- no extra learned-upscaler call
- packaged backend `cute_sm120`, source tree verified
- Sana source revision `2936c47637380842aaa4a4488fac5006cc542b70`
- SM120 compute capability `(12, 0)`
- all-selected SM120 -> native SDPA comparison passes for all three witnesses
- independent KC/VC/threshold preparation passes
- independent route trace matches the executed sparse route
- diagnostic specializations match ordinary execution
- supplied E media is clean; raw and pre-guidance E media are identical for this call.

The durable full tensor evidence remains on the workstation at:

`/home/toor/ComfyUI/output/h3_first_high_sol_local_e/234ed062128e43ed8d5ec63e27517b22-all-selected-e.pt`

with SHA-256:

`e25ff53bb3e6f7d6c1780ba9a1bb44288be1b3e095e87166a77b152971e8a6d5`.

For bounded offline transport, block 2 / group 10 was extracted without regenerating E. The transported shard has SHA-256:

`7abacbb03407a7486329ce05bcb4ed2a2778f9eb090c6f08ec890a658bf3f9ec`.

Its wrapper points back to the authoritative full evidence SHA above and identifies evidence index 42, block 2, group 10.

## 2. Correction: `frozen_route_mixed_arithmetic_conformant=false` is a diagnostic false negative

The completed report labels all three frozen-route witnesses non-conformant, but the failure is not evidence of a fused mixed-arithmetic defect.

The current Flow validator applies the same absolute gate (`mean_abs <= 0.002`, `rel_l2 <= 0.005`) to:

1. final output,
2. a reconstructed numerator scaled to the reference row maximum,
3. denominator,
4. LSE.

For all three witnesses, output, denominator and LSE satisfy the baseline limits. The only failing quantity is the reconstructed numerator's **absolute** mean error (~0.052–0.062). That numerator is not output-scaled: the Sol witness reconstructs it as `BF16 kernel output * FP32 kernel denominator`. Applying the output-space absolute limit `0.002` to that differently scaled quantity is dimensionally invalid.

The authoritative design also explicitly required a separate BF16-emulated mixed comparison. The current witness only computes the FP32 frozen-route reference.

The SM120 implementation performs mixed online-softmax state in FP32, but casts the probability fragment to BF16 before the PV MMA and casts the final output accumulator to BF16 before storing. An apples-to-apples BF16-emulated offline replay of block 2 / group 10 using the exact saved Q/K/V, KC/VC and frozen executed routes gives:

| comparison | mean abs | rel L2 | max abs | p99 abs |
|---|---:|---:|---:|---:|
| production sparse vs FP32 frozen mixed reference | 0.0013647134 | 0.0016879705 | 0.06147766 | 0.00833821 |
| production sparse vs FP32 reference cast only at final output | 0.0002166541 | 0.0010314837 | 0.0625 | 0.0078125 |
| production sparse vs BF16-probability-PV + BF16-output emulation | **0.0001663658** | **0.0008874494** | 0.0625 | 0.00390625 |
| all-selected SM120 vs native SDPA baseline | 0.0002487788 | 0.0011001661 | 0.0625 | 0.0078125 |

The mixed operator is therefore at least as numerically well behaved as the already accepted all-selected/native baseline under matching BF16 execution semantics. The prior `production_sparse_conformant_to_independent_witness=false` conclusion is a validator/reporting defect, not a causal mixed-kernel failure.

No production kernel change follows from this correction. A future diagnostic cleanup should report numerator error in a scale-aware way and add the separately required BF16-emulated reference, but E does not need to be rerun to establish the present causal branch.

## 3. Group-10 policy evidence

Group 10 saved geometry:

- Q rows: 1024
- restricted K/V rows: 11293
- 16 Q64 tiles x 56 heads x 177 K64 blocks
- current selected exact block-pairs: 88,393 / 158,592 = 55.7361%
- prefix/sink rows: 3101 -> sink blocks `[0, 49)`
- `query_positions_in_restricted_kv`: contiguous 9245..10268.

The saved query positions map the 16 Q64 tiles to restricted-K coordinates around blocks 144..160. For example, Q tile 0 represents K blocks 144/145 and therefore its mapped local-neighbor set is 143..146; Q tile 15 maps to 159/160 with neighbor set 158..161.

The current packaged ordinal rule instead compares local Q-tile ordinals 0..15 directly against K blocks. Those ordinal neighbors lie entirely inside prefix/sink blocks 0..48, which are already forced exact. In this rectangular gathered domain the intended ordinal local-neighbor forcing therefore adds no physical-local protection for these group-10 video queries.

Applying the design's mapped-neighbor rule to the **existing route only additively** gives:

- mapped-neighbor pairs: 3,584
- already selected by existing route: 2,999
- missing mapped-neighbor pairs: **585**
- missing fraction of mapped-neighbor pairs: **16.32%**
- added exact work relative to the current 88,393 selected pairs: **+0.662%**

The omission is strongest near the group boundaries: Q tile 0 misses 36.6% of mapped-neighbor pairs and Q tile 15 misses 33.9%; the middle tiles still contain repeated omissions.

## 4. M is material; T is rejected by the saved witness

Using the saved group-10 tensors, frozen existing routes, exact token contributions for selected blocks and the Sol zeroth-order KC/VC approximation for rejected blocks, the BF16-emulated mixed reference was compared with the saved native-SDPA restricted-support output.

| policy on the same Q/K/V | selected pairs | mean abs vs native | rel L2 vs native | max abs | p99 abs |
|---|---:|---:|---:|---:|---:|
| current `diag` sparse route | 88,393 | 0.02579051 | 0.05495828 | 8.46875 | 0.25390625 |
| **M: add missing mapped physical neighbors only** | 88,978 | **0.02509269** | **0.05211410** | **7.796875** | **0.25** |
| T: full-covariance threshold, same tau/sink/domain | 67,538 | 0.05666611 | 0.08742280 | 8.46875 | 0.453125 |

Relative to the current sparse route, M improves:

- mean absolute error by **2.71%**,
- relative L2 error by **5.18%**,
- maximum absolute error by **7.93%**,
- while increasing exact selected work by only **0.662%**.

The missing mapped-neighbor exact-vs-approx correction is not numerically negligible: applying only those 585 route additions changes the mixed output by rel-L2 ~0.0101 against the current mixed output. In the current-denominator scale, the mapped-neighbor numerator correction has rel-L2 ~0.0187 against the current normalized numerator/output state.

The T estimator was emulated from the actual packaged exact-threshold contract: BF16 pooled query, FP32 pooled-key mean, BF16 pooled-key second moment with FP32 projected dot, tau=1, same sink and same ordinal forcing. The equivalent diagonal emulation reproduces the saved packaged diagonal threshold to max abs `2.86e-6` / mean abs `3.67e-7`, establishing that the offline threshold arithmetic matches the deployed preparation contract closely. T raises the threshold enough to remove 20,950 currently exact pairs while adding only 95, reducing density to 42.586%. Its group-10 error is materially worse, so T is not the evidence-selected arm.

## 5. Decision-table result

The corrected causal state is:

- E valid and media clean.
- all-selected SM120 arithmetic conforms to native SDPA.
- KC/VC/threshold preparation conforms.
- executed route trace conforms to the independent route rule.
- mixed SM120 arithmetic conforms once compared under matching BF16 execution semantics; the previous frozen numerator gate was a false negative.
- the rectangular ordinal-neighbor coordinate mismatch is present in the actual saved witness.
- that mismatch leaves real mapped self/near K blocks rejected.
- forcing only those missing mapped neighbors exact produces measurable same-input error reduction with a very small exact-work increase.
- full-covariance threshold T materially worsens this witness.

Therefore the authoritative decision table selects **M** as the next bounded causal media arm.

This is **not** authorization to promote a production fix. It authorizes exactly one controlled M media call after implementation/review.

## 6. Exact M arm to implement

M must preserve all E/R invariants and change only the selected-set augmentation for eligible Sol-local calls:

1. Keep original diagonal threshold, tau, sink, restricted VDN support, complement, dense layers, global/anchor providers, Q/K/V, adapters, schedule and Spectrum state unchanged.
2. Consume VDN-owned `query_positions_in_restricted_kv` derived from the exact gathered plan. Do not infer physical positions from local Q ordinals.
3. For each Q64 tile, identify the K64 blocks containing its represented query positions, then union the immediate K-block neighbors `-1, 0, +1`, clamped to the restricted K-block domain.
4. Add those blocks to the existing exact route. **Never remove an existing threshold/sink/ordinary selection.**
5. Do not expand the restricted K/V domain and do not reintroduce square-Q expansion.
6. Preserve 1 logical / 1 actual / 0 forecast and exactly 528 Sol local + 22 native dense local + 50 native global + 100 native anchor backend receipts.
7. Record per-call added block pairs, reason (`mapped_neighbor`), original/effective selected counts, exact-work increase and bounded route evidence.
8. Fail closed if the VDN->Sol query-position capability is absent, ambiguous, inconsistent with the gathered plan, or would mutate an existing route decision.
9. Keep M diagnostic-only and separately identified in history/provenance. Do not silently reinterpret provider API v3.
10. Decode the same first-high raw/pre-guidance media and compare against preserved R/W/E. User review of the complete clip remains the media gate.

Only if M media is clean with matched invariants and retained acceleration does the design authorize a narrow production candidate: versioned VDN->Sol query-position contract plus additive SM120 forced-route mapping. Any performance cleanup that removes accidental old ordinal neighbors is separate work and is not part of the first proven fix.

## 7. No additional E/R/W work

Do not rerun R, W or E for this selection. The saved tensors are sufficient for the arithmetic/policy decision above. Do not run C. Do not run T. Do not change threshold policy, support width, complement, dense warmup, scheduler, Spectrum, state transport or H3 call count.
