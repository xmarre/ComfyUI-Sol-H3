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
