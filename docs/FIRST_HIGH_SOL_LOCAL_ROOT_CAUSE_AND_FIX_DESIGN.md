# First-high Sol-local causal design

Status: investigation checkpoint; not yet implementation authority. No production change is authorized.

Controlled R/W evidence isolates the removed Sol-local operator for capture `234ed062128e43ed8d5ec63e27517b22`. Both retrieved W reports have valid invariants and matching entry hashes. Native restricted-window attention is clean according to the completed controlled media assessment; full support is not required. Preserve R/W and Flow #43, VDN #15, Sol #11 unchanged.

Audited source baseline: Sol `fff274559fe08f9b8ede6affe05c5487b2f8747e`, VDN `7d2d146c658e0ca4a90aa8a90b3ab35d78a2bbed`, Flow `b41a20c8b15e17fcb97329e2bbbb38e058bc3506`.

Findings requiring completion of provenance/history audit:

- VDN supplies requested Q and exactly global-prefix plus window K/V to provider v3. Sol calls the packaged SM120 rectangular kernel without recomputing prefix Q.
- Making the whole local K/V domain a sink selects every valid block; `has_approx` is false and all contributions use the real exact branch. This is a mathematically dense local-domain oracle, subject to BF16/implementation error, not bitwise identity.
- `threshold=diag` is a diagonal-covariance threshold approximation. `exact_kernel=rounded-affine-v1` is unrelated H3 affine modulation fusion.
- K/V summaries, denominator multiplicities and tail lengths use the local K/V domain. No full-packed-domain complement mass is visible in the inspected Sol call path.
- The unchanged shared selector forces `abs(q_block-kv_block)<=1`. Requested Q and gathered local K/V have different origins. This is an unresolved inherited square-coordinate routing policy, not a proved artifact cause. The rectangular patch updates lengths and threshold statistics but not this selector.

Next design work: verify installed bytes, inspect exact upstream revision and PR #1 history, specify one all-selected first-high intervention plus bounded same-input operator witnesses and conditional follow-up. Do not repeat R/W or choose broad Sol disable, full attention, Flow transport changes, dense warmup, added H3 calls or weighted Mixed-Grid research.
