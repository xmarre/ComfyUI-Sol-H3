## Unreleased: domain-uniform Flow stream history

Spectrum backend history recognizes Flow's opt-in target-band
`target_band_context=domain_uniform_v1` block replacement.
`PARTITIONED_DOMAIN_STREAM_API = 1` advertises the capability to paired Flow.
The replacement evaluates two uniform-grid hidden streams per block. Each stream
must carry a canonical equal-grid partition contract, a layout signature ending
with its domain-stream leaf, a matching stream leaf, and a dense-query head
inside its temporal extent. The identity records the policy, native layout and
both streams' digests, row counts and signatures. Any inconsistency leaves the
replacement unrecognized, so history stays opaque rather than reusing another
identity. Existing classifiers, receipts and request routing are unchanged.

## Unreleased: dense partitioned sink measure

Dense partitioned requests may extend a key-measure range from row 0 through
the packed non-video sink and into video keys. `PARTITIONED_SINK_MEASURE_API = 1`
advertises this to paired Flow/VDN. Partial sink overlap and sink-covering sparse
requests remain rejected. The range remains part of validation and completion
identity; receipt ownership remains request/evaluation scoped.

Use with [Flow #97](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/97)
and [VDN #40](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/40) for the opt-in
`target_query_sink_measure` comparison. Existing Sol requests retain their
weighting and dispatch. This capability reuses the weighted dense kernel and
does not establish rendered boundary quality or GPU performance.

# ComfyUI-Sol-H3 v0.1.9

Recognize Flow's target-grid native partition carrier in Spectrum backend history.

## Spectrum history

- Partitioned history identifies Flow's exact-prefix replacement when the native
  (pre-partition) sequence is the uniform target grid, declared by
  `native_carrier_grid="target"` with a matching integer
  `native_carrier_rows_per_frame`. The partition remains
  `[target-grid head | source-grid tail]`.
- Without this, Flow's opt-in target-band continuation would leave the history
  policy opaque and Spectrum would execute every scheduled forecast as an actual
  model call.
- Contracts without the field keep the reduced-grid carrier checks. Inconsistent
  carrier or row declarations remain opaque.
- `sol_h3.partitioned_history.PARTITIONED_NATIVE_CARRIER_GRIDS` lists the
  recognized carriers for preflight capability checks.

The packaged kernel, routing, attention measures and provenance are unchanged.

## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.10](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.10) | [#96](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/96) |
| Sol-H3 | [v0.1.9](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.9) | [#39](https://github.com/xmarre/ComfyUI-Sol-H3/pull/39) |
| VDN-H3-Plus | [v1.5.8](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.8) | [#38](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/38) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | [#39](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/40) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
For INT8 fused MLP runtime adapters, retain that ComfyUI Patcher PR overlay until
the repair is available upstream. The independent Core #16720 optimization is
not included in this release set.

---

# ComfyUI-Sol-H3 v0.1.8

Preserve completed continuation history and reuse independently checked attention
work without changing the packaged kernel or its provenance.

## Attention and request lifetime

- Qualify equal-grid exact-prefix history using geometry, measure, semantic
  identity and actual completion receipts, restoring compatible Spectrum
  forecasting.
- Validate packaged source once per native partitioned request. Changed verifier
  or source identity, nested requests and subsequent requests validate separately;
  failed validation clears trust.
- Reuse the packaged dynamic tensor ABI across rectangular partitioned subcalls,
  retaining device, architecture, bias/map specialization, dtype/rank and
  broadcast-stride identity. Rebind current tensor and scalar arguments at launch.
- Execute independently checked weighted dense attention through the existing
  all-selected union. Unit-measure dense calls retain Core SDPA.
- Preserve SM120/SM121 support, grouped domains, physical key measure, numerical
  history boundaries and hard arithmetic/CUDA failure propagation.

The coordinated same-grid profile has reported audiovisual acceptance. Existing
source and arithmetic tests cover request ownership and dispatch; they do not
establish a general generated-quality or speed claim.

## Coordinated release set

Update the coordinated components together. Every release links this same
version set and identifies its implementation PRs.

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.9](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.9) | [#89](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/89), [#93](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/93) |
| Sol-H3 | [v0.1.8](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.8) | [#37](https://github.com/xmarre/ComfyUI-Sol-H3/pull/37) |
| VDN-H3-Plus | [v1.5.7](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.7) | [#33](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/33), [#34](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/34), [#35](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/35), [#36](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/36), [#37](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/37) |
| H3 Continuum-Plus | [v3.4.5](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.5) | [#37](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/37), [#38](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/38) |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | [#16](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/pull/16) |

[Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28)
is the unchanged companion. Separate Keyless, audio-training and rejected
decoded-geometry experiments are outside this release set.

The tested Core adapter repair is
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783).
It remains an upstream review item, with upstream workflow approval and merge
controlled by Comfy-Org maintainers. For INT8 fused MLP runtime adapters,
retain that ComfyUI Patcher PR overlay until the repair is available upstream.
The independent Core #16720 optimization is not included in this release set.

---

# ComfyUI-Sol-H3 v0.1.7

v0.1.7 releases the SM121 / NVIDIA GB10 support implemented by [PR #38](https://github.com/xmarre/ComfyUI-Sol-H3/pull/38) and requested by [issue #10](https://github.com/xmarre/ComfyUI-Sol-H3/issues/10).

## SM121 execution support

Compute capability `(12, 1)` now selects the existing packaged Sana `cute_sm120` CuTe backend, matching the SM120-family implementation path without falsifying the device capability reported by PyTorch. The production gates are opened consistently in the SOL loader, ordinary runtime shape checks, partitioned execution, and rectangular K/V sizing.

The change remains inside the existing vendored-source contract: the backend mapping is packaged as a reviewed Sana patch, the manifest/provenance identities are updated, and source verification continues to fail closed rather than accepting an untracked runtime substitution.

## Validation

PR #38 supplied direct GB10 execution evidence:

- a BF16 `1×64×4×128` `load_kernel` call selected `cute_sm120` and returned finite output;
- a 64×64 / five-frame Comfy prompt recorded `sol_backend=cute_sm120` and `sparse_calls=100`.

The implementation head also passed the repository CPU/provenance/interoperability workflow before merge, including the public API contract for `(12, 1) -> cute_sm120`. The merged main commit passed the same hosted workflow before this release metadata commit.

This establishes functional SM121 admission and real-GB10 execution. It does **not** establish an independent maintainer GB10 benchmark, a universal speed claim, or exact SM120/SM121 performance parity.

## Scope and compatibility

- Supported custom-kernel execution targets are Linux/WSL2 on SM120 and SM121.
- Native Windows behavior is unchanged: SOL and Exact Runtime continue to fail closed to their native/inherited paths.
- Existing SM120 behavior and the `cute_sm120` kernel contract are unchanged.
- The still-open Sol #35 diagnostic and Sol #37 continuation-performance overlay are **not** part of v0.1.7; they remain separate maintainer overlays rebased on this release line.


---

# ComfyUI-Sol-H3 v0.1.6

Coordinated production release with [ComfyUI-VDN-H3-Plus v1.5.6](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.6), [MiniMax H3 Flow-Aligned Regenerate v0.3.6](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.6), and [H3 Continuum v3.4.4](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.4). [Spectrum MiniMax H3 v0.2.28](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.28) remains the unchanged Spectrum companion.

Production consolidation PRs: [Sol-H3 #32](https://github.com/xmarre/ComfyUI-Sol-H3/pull/32), [VDN-H3-Plus #32](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/32), [Flow #73](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/73), and [Continuum #34](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/34). The validated source lines consolidated by those PRs are Sol #15, VDN #30, Flow #70 and Continuum #33.

## Exact-prefix partitioned SOL execution

v0.1.6 completes Sol-H3's production side of the newer exact-prefix progressive stack. The partitioned route accepts the heterogeneous prefix/suffix execution contract produced by Flow and VDN while preserving the existing mapped-neighbor provider-v4 ownership model and the real packaged Sana Sol-Attn SM120 CuTe backend.

The production contract keeps VDN's restricted K/V support, mapped physical query positions, learned softmax gate, output projection and learned linear complement intact. It does not reconstruct full K/V, expand Q to a square domain, or move VDN-owned semantics into Sol. Unsupported or stale contracts continue to fail closed to the supplied native path rather than silently changing attention arithmetic.

The release also keeps backend-history ownership explicit so Spectrum can distinguish real numerical-route transitions from stable partitioned execution. No additional H3 transformer NFE is introduced by the partitioned SOL routing itself.

## Coordinated stack

The paired releases finish the surrounding path:

- **VDN-H3-Plus v1.5.6** carries the heterogeneous exact-prefix VDN contract and adds sampler-admission VRAM eviction for retained-buffer runs without changing VDN arithmetic.
- **Flow v0.3.6** promotes the hardware-validated fast source-uniform exact-prefix continuation path with the 16-tick sampler-owned audio overlap and no duplicate shadow low/probe lifetimes.
- **H3 Continuum v3.4.4** adds exact carried-audio phase transport, structural terminal lead-out/padding, and a dedicated fresh post-prefix prompt bridge so prior speech and terminal control prose do not leak into newly generated audio.

## Production validation

Sol #15's exact head passed the CPU-contract workflow before release consolidation. The coordinated components were then exercised on RTX PRO 6000 Blackwell / SM120 hardware through the same development stack.

Flow run 00575 validated the production #70 path at three continuation sampler lifetimes / two history boundaries with six source-uniform transformer calls, zero exact-partitioned duplicate calls, repaired boundary audio and decoded-media parity with the accepted width-16 run. VDN's final sampler-admission fix restored healthy high-stage allocator behavior without changing arithmetic. Continuum's final prompt/audio-boundary fix was confirmed by two clean 00603/00604 renders: no reference-image restage, no chunk-2 speech/gibberish bleed, and no vocalized terminal-control prose.

Important scope note: the later 00603/00604 Continuum confirmation runs also carried a separate Flow diagnostic overlay. That diagnostic remains unreleased. Flow v0.3.6 is the independently hardware-validated #70 production path.

## Release scope

This release intentionally does **not** merge or promote the Keyless research PR family, historical selector/arithmetic diagnostics, duplicated shadow-path experiments, or other superseded A/B branches. Those branches remain evidence only. v0.1.6 is the consolidated production tree from Sol #15 plus release metadata.


---

# ComfyUI-Sol-H3 v0.1.5

Coordinated production release with [ComfyUI-VDN-H3-Plus v1.5.5](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.5) and [MiniMax H3 Flow-Aligned Regenerate v0.3.5](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.5).

Implementation PRs: [Sol-H3 #14](https://github.com/xmarre/ComfyUI-Sol-H3/pull/14), [VDN-H3-Plus #18](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/18), and Flow [#33](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/33) + [#48](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/48).

## Motivation: the first-high artifact

The real production stack could produce a reproducible **first-high visual corruption/artifact** when VDN retained grouped local attention was routed through Sol-H3 sparse attention after the progressive low-to-high handoff.

The controlled same-input investigation isolated the failure without changing the sampler entry state:

- replay R reproduced the broken first-high output;
- native W-window was clean while preserving VDN's restricted local K/V support and learned linear complement;
- W-full-support was also clean;
- global and anchor attention remained native.

This ruled out both “VDN needs full K/V support” and “the VDN learned complement must be removed” as necessary fixes. The remaining causal boundary was the **Sol-local sparse route**.

VDN local attention is not an ordinary square sequence. For each grouped call it gathers requested Q rows separately and constructs the restricted K/V domain as `[global_rows, permitted_window_rows]`. A requested query's physical row in that gathered K/V domain therefore differs, in general, from its local Q ordinal.

The old Sol exact-neighbor selector still used the ordinary aligned-coordinate rule:

```text
abs(q_block - kv_block) <= 1
```

That is correct for aligned/square attention but not for these grouped rectangular calls. In the preserved failing group-10 geometry, the requested queries physically map into K positions `9245..10268` while their local Q block ordinals are only `0..15`. Sol was therefore protecting the wrong local K neighborhood exactly.

Diagnostic M proved the correction on the same input: preserve the existing sparse approximation and VDN restricted support, but add exact neighbors around every query's **mapped physical K/V position**. The diagnostic implementation itself used temporary selector instrumentation and descriptor-value specialization, so it was evidence rather than the production design.

## Resolution: mapped-neighbor provider v4

v0.1.5 turns that evidence into the real production kernel path.

The paired VDN-H3-Plus v1.5.5 provider-v4 contract now transports an immutable, owner-bound affine mapping from each requested Q row to its exact position in the already-gathered restricted K/V domain. Sol validates the full mapping and ownership contract, compiles it into bounded `[start_block, end_block)` K64 intervals per Q64 tile, keeps descriptor data request-local, and sends the small descriptor to the packaged SM120 CuTe kernel as runtime metadata.

Inside the kernel the correction is additive:

```text
new_exact = old_exact OR mapped_neighbor
```

The existing route is otherwise preserved: threshold routing, sinks, the ordinary ordinal-neighbor selector, Q/K/V values and row order, restricted K/V support, scale/tau, KC/VC preprocessing, VDN's learned softmax gate/output projection/complement, global and anchor calls, and dense-warmup policy.

The production implementation deliberately does **not** carry forward the diagnostic shortcuts: no square-Q expansion, no full-K/V reconstruction, no QxK mask, no selector monkeypatch, no per-map JIT specialization, and no per-local-call GPU-to-CPU synchronization. Unsupported, stale or ambiguous maps use VDN's supplied native restricted-domain callback; arithmetic, CUDA and OOM failures remain real failures rather than being hidden as capability fallback.

This is the production fix for the captured artifact: **VDN transports the physical coordinate it owns, and Sol preserves that physical neighborhood exactly inside the actual sparse SM120 kernel.**

## Coordinated Flow / Continuum path

The companion Flow v0.3.5 release makes **Progressive Handoff (Target Input)** the standard target-input/Continuum path and retires Mixed-Grid from the production acceptance matrix. Exact protected continuation stays on the target grid; the validated four-audio-tick guided overlap is used during sampling while caller-owned exact video/audio values are restored at output.

The validated production stack for this coordinated release is:

```text
MiniMax H3 Flow-Aligned Regenerate v0.3.5
ComfyUI-Sol-H3                    v0.1.5
ComfyUI-VDN-H3-Plus              v1.5.5
```

The retired Mixed-Grid weighted companion work is not a release prerequisite, and VDN PR #8's separate audio-fidelity/training experiment is not part of this coordinated release.

## Production validation

The production route passed all release gates on real RTX PRO 6000 / SM120 hardware.

### Same-input mapped correctness

- route mismatch count: `0`;
- preserved E/M accounting: `88393` original + `585` mapped = `88978` effective block pairs;
- frozen-route arithmetic gate: `rel_l2=0.0016882352`, `max_abs=0.0614815`;
- changing descriptor values created no new CuTe specialization keys.

### Matched historical-M performance

Five-repeat matched median:

```text
production mapped: 1.109472036 ms
historical M:      1.105535984 ms
ratio:             1.0035603114x
median delta:      +0.356031%
acceptance budget: <= +5%
result:            clear_pass
```

### Controlled first-high replay

The production replay preserved the exact captured first-high inputs and produced:

- `1 logical / 1 actual / 0 forecast`;
- zero learned-upscaler calls;
- exactly 700 completed Sol receipts: 528 mapped local, 22 dense warmup, 50 native global, 100 native anchor;
- exact mapped topology/geometry with no mapped-local or kernel-unavailable fallback;
- clean raw, pre-guidance and final decoded first-high media.

### Representative two-chunk trajectory

Run `00494` completed:

```text
17 logical calls
13 actual H3 NFE
4 Spectrum forecasts

low:    4 actual / 1 forecast
probe:  1 actual / 0 forecast
high:   2 actual / 1 forecast
later:  6 actual / 2 forecast
```

Mapped local calls were `1,584` low, `1,056` high and `3,120` later. Requested Q rows always equalled kernel Q rows, square expansion remained zero, and all Mixed-Grid/weighted-Mixed-Grid counters were zero.

The 14-second decoded output showed no first-high corruption, frame shift, zoom-out, top-edge reveal, flash, grid artifact or physical AV seam; the chunk boundary was perceptually seamless.

00494 evidence hashes:

- runtime log: `699c17b6d85d186039b9179c4b99edb596e88fdda37d3f8814917c50b6905ca8`;
- metrics JSON: `c25a90af61d83c1430c3f29b9e99de7b0adb22f60d714e2d1392a269e6f0802e`;
- final MP4: `1ffb5ebff47417f2f9354d3ae7cdfa32b6b6e292cd661efb9ddc2fd818d66ab2`.

## Review hardening

The final review additionally fixed two retained compatibility edge cases:

- empty weighted exact-K ranges now clamp `sink_start` to the current K-row count;
- preprocessing cache identity now incorporates bounded semantics of code-reachable globals/helpers instead of using code-only identity when mutable `LOAD_GLOBAL` state is visible; unsupported mutable state receives deliberately volatile identity.

These review fixes are covered by the exact-head hosted CI lanes and do not change the mapped-neighbor production arithmetic validated above.

---

Previous release notes through v0.1.4 are preserved verbatim in `docs/RELEASE_NOTES_v0.1.4_AND_EARLIER.md`.
