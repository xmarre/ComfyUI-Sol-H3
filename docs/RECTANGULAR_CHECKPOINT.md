# Rectangular SM120 development checkpoint

Parent PR #1 head: c543f4f017c0ddb276ff28148a7e9be291057b75; base main: 5db282ca836416a32cf114346b946fe75136e4f1.
Mirror: mirror/rectangular-sm120-20260908. PR branch must remain one implementation commit.

Implemented independent Q/KV interface, preprocessing, output and cache geometry;
SM120 q_len derives from Q. VDN v2 uses requested Q and restricted KV directly.
Aggregate arithmetic gate unchanged; calibration selects all KV, cache includes both shapes.
Original Sana hashes retained; functional patch saved at tools/rectangular_sm120.patch.

NOT YET REVIEWED OR VALIDATED. Remaining: focused CPU/GPU tests, reproducible vendor
patch application and provenance documentation, current VDN #8/#11 interoperability,
GPU probe with matched warmed square-expanded baseline, review, CI, final consolidation.
No GPU execution or performance claim exists for this change.

Source retrieved with GitHub connector; local baseline snapshot has identical Git blob
hashes for all 95 files. Direct Git transport unavailable. Remote checkpoints use GitHub
Git tree/commit/ref APIs preserving the real remote parent; local snapshot commit is
only a diff baseline and must never become a remote PR parent.
