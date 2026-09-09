# Native Windows status

## SOL attention

RTX 5090 is an SM120 GPU, so the hardware architecture is eligible for the packaged Sana Sol-Attn path. The current NVIDIA CUTLASS CuTe DSL runtime is the limiting component on native Windows: NVIDIA documents Windows support as unsupported and the CuTe DSL quick-start currently lists Linux x86_64/aarch64 as the supported platforms.

For the real `cute_sm120` SOL kernel, use Linux or WSL2. Native Windows is not a supported SOL-kernel execution target in this release. When CuTe is unavailable on native Windows, Sol-H3 falls back locally to the inherited dense attention provider and reports an explicit `native Windows ... requires Linux/WSL2` compatibility reason.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

## Issue #4: `Packaged Sana source file set mismatch`

The original v0.1.0 provenance verifier used `str(Path.relative_to(...))` to compare on-disk vendored paths with manifest keys. That produces backslashes on Windows while the manifest uses canonical forward-slash paths, so a valid Windows checkout could reject the complete vendored source tree before SOL initialization.

v0.1.1 fixes that bug by using POSIX manifest keys on every platform. It also:

- pins the vendored snapshot to LF checkout in `.gitattributes`;
- accepts only Git-style CRLF-to-LF transport normalization when validating canonical packaged hashes;
- keeps every other file-content difference fail-closed;
- scopes CuTe runtime dependencies to Linux so native-Windows installation does not try to install an unsupported kernel toolchain;
- runs provenance regression tests on `windows-latest` CI.

This fixes the false provenance failure. It does **not** claim native-Windows CuTe execution support.

## `comfy_aimdo` / allocator crashes

An error such as:

```text
OSError: exception: access violation reading 0x0000000000000000
...
comfy_aimdo.malloc_graph ... malloc_graph_pop
RuntimeError: aimdo memory compile error
```

is a separate execution path from Sol-Attn. In issue #4 the final Sol-H3 telemetry showed:

```text
sol_backend: null
sol_source_tree_verified: false
sparse_calls: 0
exact_blocks: 600
```

Therefore no SOL sparse kernel executed in that failing request. The log does not establish that the AIMDO crash was caused by SOL attention.

The same request had `exact_fusion=true`, so Exact Runtime did execute. Native-Windows Exact Runtime together with ComfyUI's AIMDO malloc-graph compiler has not been production-validated by this project. To isolate an AIMDO crash on native Windows, first rerun with `exact_fusion=false` on the SOL node. If the crash persists with both SOL sparse execution absent and Exact Runtime disabled, investigate the ComfyUI/AIMDO path separately. If it disappears, report the new log so Exact Runtime/AIMDO compatibility can be isolated without conflating it with SOL.
