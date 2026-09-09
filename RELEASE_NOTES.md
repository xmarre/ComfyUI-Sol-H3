# ComfyUI-Sol-H3 v0.1.1

Patch release addressing the native-Windows failure reported in issue #4 without claiming unsupported native-Windows SOL kernel execution.

## Fixed

- Fixed a real cross-platform provenance bug: vendored Sana file names are now compared with canonical `/` manifest paths instead of platform-native `Path` strings. A valid Windows checkout no longer fails every nested entry with `Packaged Sana source file set mismatch`.
- Pinned the vendored source snapshot to LF checkout via `.gitattributes`.
- Provenance hashing now accepts only Git-style CRLF-to-LF transport normalization in addition to exact bytes; any other content change still fails closed.
- Added `windows-latest` provenance CI covering Windows path semantics, CRLF normalization and tamper rejection.
- Scoped CuTe runtime dependencies (`nvidia-cutlass-dsl`, `cuda-python`, `apache-tvm-ffi`) to Linux, matching the platform actually supported by NVIDIA's current CuTe DSL runtime.
- Native Windows now reports an explicit `Linux/WSL2` requirement when the SM120 CuTe backend is unavailable instead of presenting the fallback as a generic backend-selection problem.

## Native Windows boundary

RTX 5090 is SM120 hardware and is architecturally eligible for the packaged kernel. The current NVIDIA CUTLASS CuTe DSL runtime does not support Windows, so the real `cute_sm120` SOL kernel still requires Linux/WSL2. This release fixes Windows installation/provenance behavior and diagnostics; it does **not** claim native-Windows SOL acceleration.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

See `docs/WINDOWS.md` for the exact boundary and troubleshooting procedure.

## Issue #4 allocator crash

The attached failing request reported `sol_backend=null`, `sol_source_tree_verified=false` and `sparse_calls=0`, so no SOL sparse kernel executed before the later `comfy_aimdo` `malloc_graph_pop` access violation. The same request had `exact_fusion=true` and `exact_blocks=600`.

This patch therefore does not misattribute or claim to fix the AIMDO failure. Native-Windows Exact Runtime plus ComfyUI's AIMDO malloc-graph compiler remains unvalidated. For isolation, rerun on native Windows with `exact_fusion=false`; if the AIMDO failure persists, it is outside the SOL sparse-kernel path.

## Regression scope

The Linux/WSL SM120 production kernel contract, rectangular Q/KV execution, VDN API-v3 route, Flow mixed-grid route, Spectrum history semantics and zero-copy BTHD behavior are unchanged from v0.1.0.
