# Weighted dense exact-prefix attention

Heterogeneous exact-prefix continuation carries a target-grid prefix and a
source-grid suffix through VDN's grouped attention. Prefix keys have a physical
measure of `source_rows / target_rows`, represented as an additive log bias.
Global queries, protected-prefix queries and dense warmup evaluations retain
dense attention over their complete grouped K/V domains.

A non-null attention mask excludes PyTorch's CUDA FlashAttention backend. These
weighted dense calls use the packaged SM120 weighted union with the exact sink
covering every K/V block. Every key is selected; no compressed block contributes.
The union retains one softmax normalizer, the original query/key rows, native
attention scale and VDN window/global/anchor ownership. The FP32 bias input is
rounded through the query dtype to preserve the preceding native SDPA mask
rounding. Sparse calls retain their existing FP32 key measure and routing.

Before completing the first weighted dense call for each request, layout and
measure identity, Sol compares its result against the independent native Core
SDPA dispatcher using the existing arithmetic thresholds. Failure propagates
without publishing a successful receipt. Native SDPA remains the sparse
all-selected oracle and the production dispatcher for unit-measure dense calls.
The metadata-only verification cache holds at most 64 entries per request;
evicted identities require validation again. It retains no activation tensors
and is discarded with the native sampling request.

Partitioned dense and sparse kernel invocations also share packaged-source
verification within that request. This source check is independent of each
device/layout arithmetic gate. See [source ownership and receipts](REQUEST_SOURCE_VERIFICATION.md).

Completion receipts use `dense_sm120_forced` or `dense_sm120_warmup` and kernel
identity `sm120-weighted-all-selected-v1`. History validation binds the new
identity to the active request's completed call, a real negative key measure and
the corresponding prefix range. Old native dense receipts remain supported.
Changing kernel, mode, geometry or key measure still changes numerical history.

Runtime summaries publish `partitioned_weighted_dense_calls`,
`partitioned_weighted_dense_q_rows` and `partitioned_weighted_dense_gate_entries`.
`partitioned_core_dense_calls` and `partitioned_torch_dense_calls` continue to
count executed native dispatcher calls, including independent references.
Arithmetic-gate receipts identify `all_keys_selected=true` and native query-dtype
bias rounding. Gate wall time includes compilation and queued GPU work, so it
does not measure only the reference cost.

CPU tests verify dispatch, dense-domain ownership, native bias rounding,
independent references, request isolation, cache bounds, failure propagation and
history identity. The SM120 tests execute the production request and compare
against both native SDPA and an explicit FP64 softmax oracle. Those tests require
a real SM120 device. GPU latency, peak allocation and generated video/audio
quality need hardware qualification; CPU success does not establish them.
