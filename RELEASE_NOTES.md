# ComfyUI-Sol-H3 v0.1.1

Patch release addressing the native-Windows failure reported in issue #4 without claiming unsupported native-Windows SOL kernel execution.

## Fixed

- Fixed a real cross-platform provenance bug: vendored Sana file names are now compared with canonical `/` manifest paths instead of platform-native `Path` strings. A valid Windows checkout no longer fails every nested entry with `Packaged Sana source file set mismatch`.
- Pinned the vendored source snapshot to LF checkout via `.gitattributes`.
- Provenance hashing now accepts only Git-style CRLF-to-LF transport normalization in addition to exact bytes; any other content change still fails closed.
- Added `windows-latest` provenance CI covering Windows path semantics, CRLF normalization and tamper rejection.
- Scoped CuTe/Triton runtime dependencies to Linux, matching the platform actually supported by the packaged kernel path.
- Native Windows now reports an explicit `Linux/WSL2` requirement when the SM120 CuTe backend is unavailable instead of presenting the fallback as a generic backend-selection problem.
- Exact Runtime now fails closed on native Windows before importing or executing its Triton affine kernel and delegates to the untouched native H3 block. This removes the unvalidated Sol-H3 Exact/Triton path from ComfyUI's native-Windows AIMDO malloc-graph lifecycle.

## Native Windows boundary

RTX 5090 is SM120 hardware and is architecturally eligible for the packaged kernel. The current NVIDIA CUTLASS CuTe DSL runtime does not support Windows, so the real `cute_sm120` SOL kernel still requires Linux/WSL2. This release fixes Windows installation/provenance behavior and diagnostics; it does **not** claim native-Windows SOL acceleration.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

See `docs/WINDOWS.md` for the exact boundary and troubleshooting procedure.

## Issue #4 allocator crash

The attached failing request reported `sol_backend=null`, `sol_source_tree_verified=false` and `sparse_calls=0`, so no SOL sparse kernel executed before the later `comfy_aimdo` `malloc_graph_pop` access violation. The same v0.1.0 request had `exact_fusion=true` and `exact_blocks=600`, so Exact Runtime was the only Sol-H3 custom kernel path that actually executed.

v0.1.1 does not claim that Sol-H3 caused the AIMDO failure. Instead, native Windows now automatically delegates Exact Runtime to native H3, while SOL remains a dense fallback because CuTe is unavailable. If `comfy_aimdo` still fails after upgrading to v0.1.1, the failure is reproducible with both Sol-H3 custom kernel paths absent and should be investigated in the ComfyUI/AIMDO path separately.

## Regression scope

The Linux/WSL SM120 production kernel contract, rectangular Q/KV execution, VDN API-v3 route, Flow mixed-grid route, Spectrum history semantics and zero-copy BTHD behavior are unchanged from v0.1.0.
