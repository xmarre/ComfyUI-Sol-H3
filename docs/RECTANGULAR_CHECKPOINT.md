# Rectangular SM120 recovery record

Original PR #1 head: c543f4f017c0ddb276ff28148a7e9be291057b75.
Base main: 5db282ca836416a32cf114346b946fe75136e4f1.
Mirror: mirror/rectangular-sm120-20260908.

Remote checkpoints precede broad review/testing:
- c9d2f677b79c1d576e49b08327e52ca7ba2256f1: kernel/bridge implementation.
- c85cf9a0b694b9cc257874dd8199515d9cabbbba: focused contracts, provenance, GPU probe.

Review traced Q loads/grid/tail/route-column divisor, threshold indexing, K/V
centroids/statistics, KV route traversal/masses, sink bounds, exact-score masks,
output descriptors and LSE stores. The concrete remaining empirical risk is GPU
compilation/execution and changed sparse query grouping; see RECTANGULAR.md.
The aggregate gate was preserved. No companion PR was modified.

GitHub connector supplied the source and remote Git tree/commit/ref operations.
The initial local snapshot matched all 95 remote Git blob hashes. Local snapshot
commit IDs are diff baselines only; remote commits use real remote ancestry.
Final PR branch consolidation must use neutral main as its sole parent and the
validated mirror tree, preserving one implementation commit.

No rectangular GPU execution, warmed timing, or media-quality result is claimed.
