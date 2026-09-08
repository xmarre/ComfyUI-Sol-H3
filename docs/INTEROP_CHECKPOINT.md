# Interoperability checkpoint — 2026-09-08

This is unfinished work, saved at the user's urgent checkpoint request. Continue existing Sol-H3 PR #1; DO NOT create a replacement Sol-H3 PR or merge it. PR #1 remains draft on `feature/native-sol-h3`, one implementation commit `049939c4ef9f08d05c1557b52c602f0b9a75eaaa` above main `5db282ca836416a32cf114346b946fe75136e4f1`. Its branch has NOT yet been rewritten. Work is on a mirror branch.

## Durable checkpoints

| Repository | Branch | Saved implementation SHA |
|---|---|---|
| xmarre/ComfyUI-Sol-H3 | mirror/native-sol-h3-interop | b282f7a254c80297359a6a78f9f39f9d071e658f |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | feature/attention-backend-history | 51e08537dd8206905ca3adffa0afc3542e18faad |
| xmarre/ComfyUI-VDN-H3-Plus | feature/sol-softmax-provider | e2fac486179bd0badb82f931911fea3d630b72e2 |
| xmarre/ComfyUI-Untwisting-RoPE | feature/attention-preprocess-contract | eaea259c7f1af691fc8de144cde013cae1b27268 |

These were published with the GitHub app's Git tree/commit/ref APIs. Direct anonymous Git clone/fetch works; `git push` fails because this environment has no HTTPS credentials. Do not request new credentials: the connected GitHub app can publish. Local commits have different SHAs but identical trees; preserve them before aligning local refs with remote checkpoint commits. Companion PRs have NOT yet been opened.

## What was changed

- SOL runtime now chains arbitrary optimized-attention dense providers and preserves `_inside_attn_wrapper`. Kernel arithmetic verification uses independent BF16 SDPA; prefix queries use inherited dense attention.
- Removed provider-name bans, sparse/forecast rejection helpers, fatal zero-sparse completion checks, and exactly-one-sparse-call/block constraints. Current topology/flags/device/dtype/ownership determine per-call native fallback.
- Actual warmup increments at the first executed block, not diffusion-wrapper entry. Forecast-only wrapper calls do not consume warmup. Request ContextVars retain scoped cleanup.
- Compatible Exact/SOL duplicate applications merge through one keyed wrapper owner. Existing immutable BlockPatch callbacks use active merged configuration; clone branches retain separate model options. Conflicting SOL policy settings explicitly reject ambiguous merging.
- Exact execution falls back for whole-block hooks/custom forward, unsupported devices/capture, and unaudited native source. The affine arithmetic gate still fails on an actual parity violation.
- VDN gets `vdn_softmax_provider_v1` restricted-subcall dispatch. Global/anchor and rectangular local calls preserve native SDPA; Flex preserves its masked operator. Square aligned local domains can reach SOL. Gate, projection, local KV selection and linear complement math are unchanged. VDN advertises a current-layout history descriptor.
- Spectrum gets `attention_backend_history_v1` preflight policies and `attention_backend_receipts_v1` actual routing receipts, consumed before last-block anchor capture. Transitions reset stage histories/controllers, invalidate offline archives, and force fresh actuals. Rollback snapshots include backend identity. Opaque routing runs actual-only per call; it does not abort the sampler. Core BSA without a history contract currently uses this conservative actual-only path.
- Untwist's override exposes `attention_preprocess_v1=(pure_transform,inherited_provider)`. SOL applies these transforms once before sparse attention and calls only the dense leaf for prefix recomputation. Legacy Untwist without the contract uses inherited fallback.
- Added `tools/attention_probe.py` for synthetic SM120 PyTorch/Sage/Sage3 + SOL checks. It has not run on GPU.

## Evidence and exact test commands

No GPU was available. No full model weights, media generation, GPU SOL kernel or GPU Sage execution occurred. All kernel substitutions in CPU tests are explicit and cannot establish media quality or performance.

Workspace was `/workspace/scratch/70e7b1afd707`; source directories: `sol-h3`, `spectrum`, `vdn`, `untwist`, `ComfyUI`, `kj`, `flow`, `continuum`, `diffaid`, `dora`. Do not name the ComfyUI checkout `comfy`: it caused a Python namespace collision with its own top-level `utils` folder during pytest collection.

Environment: Python 3.12, torch 2.14.0+cpu, Triton 3.6.0, comfy-kitchen 0.2.33, comfy-aimdo 0.5.2; pytest/ruff and CPU Comfy dependencies installed. CUDA tests skip. Original SOL baseline: 45 passed, 19 skipped before installing Triton.

Latest runs:

```bash
# From sol-h3
COMFYUI_PATH=/workspace/scratch/70e7b1afd707/ComfyUI \
KJNODES_PATH=/workspace/scratch/70e7b1afd707/kj python -m pytest -q
# 58 passed, 18 skipped (includes offline SM120 compilation).
# Real ModelPatcher ownership and actual KJ Sage/Sage3 Python wrappers are tested;
# unavailable Sage CUDA kernels are substituted with CPU SDPA.

# From spectrum
COMFYUI_PATH=/workspace/scratch/70e7b1afd707/ComfyUI python -m pytest -q
# 908 passed, 13 skipped.

# From vdn (use the renamed current ComfyUI path when rerunning)
COMFYUI_ROOT=/workspace/scratch/70e7b1afd707/ComfyUI \
PYTHONPATH=/workspace/scratch/70e7b1afd707/ComfyUI python -m pytest -q
# Last completed run, before directory rename: 133 passed, 12 skipped.
# Skips include unavailable official oracle checkout / GPU-dependent paths.

# From untwist
python -m pytest -q
# 40 passed.
```

Full output retained locally in `/tmp/sol-validated.txt`, `/tmp/spectrum-validated.txt`, `/tmp/vdn-validated.txt`. Spectrum's untouched baseline against current core had 11 failures from outdated `_CountingBlock` fixtures. Fixtures now accept core's optional `attention` argument and isolate the native options copy, since current H3 adds layout/block-index fields. The separate import failures were the checkout-name collision, resolved by renaming the local checkout to ComfyUI; no production Comfy code was modified.

## Audited sources

- ComfyUI `efa6c8f804bff78b46a0fd458ebd2e47bba07a30`: H3 model, optimized attention containers/wrap_attn, ModelPatcher keyed wrapper append/remove/clone semantics, BlockSparseAttention producer.
- KJNodes `c9869eade9920a1b949de07c4a197156006bcceb`: H3 loader and get_sage_func including Sage3.
- VDN base `3516368a09de0c4bc785faff434ea9d5a9479cb0`.
- Spectrum base `a360f64fbfa54681ded100a64ded86a5713ddf17`.
- Flow `fe0ef8752b92081b5a85bc9b39ad8e2a7037d591`.
- Continuum `a5b8943844594545301b20d01af5d9e3fa38ae29`.
- Untwist base `8ab3f38a621076de9f25cf64c60f52a9b5457827`.
- DiffAid `ba9d9efbcf7e64c755e068cb76547d8cc85481eb`.
- DoRA source was inspected; retrieve its exact local SHA before recording a final audit.

The user referenced an attached runtime log showing Sage rejection, Spectrum gating and duplicate Exact rejection, but only the five PDF papers were actually attached/materialized. We have the user's error summary, not the full log/workflow. Do not claim the full log was inspected or the full stack reproduced. Get that missing evidence before claiming reproduction.

## Required work before finalizing

1. Review the new runtime/contracts critically. Passing these tests is not proof that every requested cross-product works. Add integrated VDN+SOL+Spectrum tests using real VDN dispatch and Spectrum capture, including full coverage, APIs 1/2, return to native grid, repeated chunks and both wrapper orders. Current tests cover these mainly as component contracts.
2. Audit Spectrum reset/retry/rollback/offline-replay paths, mixed conditioning subcalls, stage banks, residual feedback and policy disappearance. Verify no old auxiliary state or preflight identity can authorize an incompatible forecast. Test actual capture ordering with real wrappers. Opaque Flow/DiffAid block replacements currently force actual-only forecasting through HistoryPolicy; decide whether explicit preflight contracts can preserve more forecasts.
3. VDN local SOL is eligible only for square aligned query/key domains. Normal windows containing global prefix KV are rectangular and correctly fall back. Do not claim VDN-local speedup without actual eligible calls. Flex currently always uses its native masked implementation.
4. Legacy Untwist uses native fallback; the new preprocessing companion is needed for sparse execution with its K transform. Audit transform shape/dtype/device and provider recursion/fallback behavior further.
5. Full compatibility-gate classification table and README/AUDIT/VALIDATION rewrite are outstanding. Existing documents and PR description still describe the old restrictive implementation and are stale for this mirror branch. Do not present them as updated.
6. Run final lint/compile and exact tests after further edits. Some earlier lint passed, but final full lint across all companion changes has not been completed. Ensure CI invokes the real Comfy/KJ integration tests with checkout paths.
7. Document exact RTX PRO 6000 commands using tools/attention_probe.py, existing test_gpu.py and the pinned Sana PYTHONPATH. GPU/media/performance A/B remains outstanding. Preserve model/LoRA/seed/prompt/sigmas/references/quantization, Continuum boundaries and progressive geometry.
8. Open focused DRAFT companion PRs for VDN, Spectrum and the small Untwist protocol, with truthful limitations and test results. Link from existing Sol-H3 PR #1. No replacement Sol-H3 PR.
9. Refresh all remote heads before final publication. Consolidate final validated mirror tree onto feature/native-sol-h3 as one implementation commit above its neutral main, preserving original author and draft status. Use lease/concurrency checks; do not overwrite concurrent user changes. GitHub commit creation does not expose author parameters; preserve original authorship if using a different publishing mechanism.
10. Report PR #1 head, companion links, exact test results, GPU combinations actually executed (currently none), and remaining media/performance tests.

This checkpoint preserves the unfinished implementation. It is not release-ready and must not be marked complete merely because the CPU suites pass.
