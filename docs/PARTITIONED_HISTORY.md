# Partitioned exact-prefix history

Sol supplies Spectrum with a numerical history identity for Flow's partitioned
exact-prefix transformer. A valid identity lets Spectrum consider a forecast;
it does not force one or change the configured sampler schedule.

The heterogeneous path retains its existing identity checks. Flow's equal-grid
control is recognized only when its API-1 contract is canonical: source and
target patch grids match on both axes, all key measures are zero, prefix/suffix
ranges cover the current video rows, exact query ownership is declared, and the
semantic digest matches the canonical metadata. The replacement classifier also
binds those fields to the actual Flow closure's latent geometry and inherited
block replacement. Equal row products with different physical grids are rejected.

The current layout signature, row segments and VDN API-4 external binding remain
part of validation. Stale layouts, mismatched VDN digests and malformed controls
remain opaque, which requires actual model execution. Full layout signatures,
inherited attention providers, dense warmup and backend transitions retain their
existing identity boundaries. Attention completion receipts must still belong
to the active Sol request; the control does not qualify unknown routes.

Partitioned v2 completion receipts contain stable numerical fields. Evaluation
numbers belong to request-owned completion proof, checked against the active
forward before Spectrum accepts an anchor. Repeating the same route preserves
history; changes to geometry, measure, mapping, kernel or execution mode still
invalidate it. Proof is cleared on the next actual evaluation, so an earlier
completion cannot authorize an unexecuted attention call in the current forward.

The classifier reads scalar metadata without modifying the Flow plan, layout,
options, model or tensors. It adds no transformer evaluations, attention shadows,
sampler lifetimes or GPU synchronization. Histories remain run-scoped. The actual
transformer and attention arithmetic are unchanged, but using a forecast can
change the generated trajectory. CPU contract tests establish recognition and
rejection behavior; hardware timings and rendered video/audio must qualify the
resulting forecast-enabled workflow before production promotion.
