# Production mapped-neighbor query-position contract

Status: architecture checkpoint; final specification and validation matrix in progress. Design only; production source is unchanged.

## Verified evidence

The M report and durable JSON were inspected. Durable JSON SHA-256 matches 1914990091be84bf01c382820b1b12e7f73a9886db114ce0f820fced4122b0ba. The 528 reports sum to 226499197 original + 2817011 added = 229316208 effective exact pairs (+1.243717875%). Execution, arithmetic, entry state, first-call-only and cleanup gates pass. Topology is 1 logical/1 actual/0 forecast, zero upscaler calls; 700 backend receipts. metrics_00483_ records 48136.404779 ms for the model call. media_clean is null; clean complete-clip review is supplied by the investigation handoff, not inferred from JSON.

## Source and runtime

Main SHAs: Sol f82ff2693be37dbad3438a30eb389d77136c0276; VDN 76b31323f9e09019b435237dcd8bad1e05476ce1; Flow 970396db839ae7ab431b9718859f6d48a2e5019b. All are ancestors of current M heads: b95ad7b3bc7028465547b22fc61300cda53eb110 / 6ca09ec37cd2dcfad02b790573a79b0462a662f3 / 9b1c2c1bb32e17592a72f5e8dbc8798b9785c1cb. W/E/M reviews and review threads are empty; reported head checks pass. Independent VDN #12 has changes requested and no check runs; do not apply it merely because it is open. Independent Flow #34 has failed/cancelled tests.

Installed preflight labels differ: Sol c1b7136cfda9f110192bcca97307cb70e25037e9; VDN 10255747eb225cb20b1aff10ec7c2e39cb080002 dirty; Flow c0e324da8be0294c61888f8bf5238a5249a3bc6c. Relevant fetched Sol/VDN runtime, interop/provider, retained-plan, hybrid and M implementation bytes match the loaded-source hashes. ComfyUI is 0.35.0, core e3c077bd8a31eb6a9c0efa64a65b341337032dbd dirty. Verify files and active callables, not Git labels alone.

## Chosen architecture direction

Add vdn_softmax_provider_v4; preserve v1/v2/v3 signatures. VDN owns exact restricted-domain query mapping and deterministic geometry identity. Sol owns derivation/validation of additive K64 forcing and its packaged kernel metadata. No query expansion, K/V widening, complement change or threshold change.

Use an immutable mapping contract derived from the same frame lists as the VDN gather. Current window_bounds has monotone clamped bounds, so equal-window groups are contiguous frame runs; endpoint anchor-row exclusion does not create interior holes. Arbitrary future bounds passed to the generic builder do not inherit that proof. Require exact representability and native fallback for unsupported maps; never use a min/max hull over holes.

Use runtime int32 per-Q64 interval metadata in the SM120 ABI, ORed after existing selector/sink decisions and before ballot/approximation masking. Descriptor values must not enter JIT cache keys; only structural enablement/layout does. No selector monkeypatch, per-geometry compiled specialization, QxK mask, tensor-to-host sync per local call or import-time CUDA work.

History identity must distinguish the mapped numerical policy and deterministic plan geometry. Completed receipts must be stable across evaluations; do not include evaluation tokens, timings or activation-dependent selected counts in Spectrum-compared receipts. Dynamic fallback must fail closed to actual-only. Diagnostic M label normalization is not the production contract.

Next: finalize exact mapping schema, cache ownership and history preflight integration; verify vendor regeneration path; commit file-level plan, compatibility table, evidence gates and implementation handoff.
