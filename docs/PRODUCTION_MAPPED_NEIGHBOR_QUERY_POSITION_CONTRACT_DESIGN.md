# Production mapped-neighbor query-position contract

Status: investigation checkpoint; not yet an implementation specification. No production code change is authorized by this document.

Scope: versioned VDN-owned query positions in the restricted gathered K/V domain; additive mapped K64 neighbors while retaining every existing selector decision and the VDN complement. Preserve W/E/M diagnostic PR topology. No threshold/arithmetic redesign or removal of ordinal neighbors.

Live source audit started 2026-09-16. Main SHAs: Sol-H3 f82ff2693be37dbad3438a30eb389d77136c0276; VDN-H3-Plus 76b31323f9e09019b435237dcd8bad1e05476ce1; Flow 970396db839ae7ab431b9718859f6d48a2e5019b.

Open stacks: Sol #9 -> #11 -> #12 -> #13; VDN #14 -> #8 -> #15 -> #16 -> #17, plus independent #12; Flow #33 -> #35 -> #36 -> #41 -> #43 -> #44 -> #45, with #30 and other overlays requiring installed-manifest verification.

Live M heads: Sol b95ad7b3bc7028465547b22fc61300cda53eb110; VDN 6ca09ec37cd2dcfad02b790573a79b0462a662f3; Flow 9b1c2c1bb32e17592a72f5e8dbc8798b9785c1cb.

Evidence resolved for inspection: M_report_00001.json; 234ed062128e43ed8d5ec63e27517b22-mapped-neighbor-m.json; preflight_provenance_manifest_00007.json; executioncontractreport_00017.json; metrics_00483_.json. Source and machine-evidence audit is in progress; runtime claims in the handoff have not yet been independently checked.
