# Native Windows status

## SOL attention

RTX 5090 is an SM120 GPU, so the hardware architecture is eligible for the packaged Sana Sol-Attn path. The current NVIDIA CUTLASS CuTe DSL runtime is the limiting component on native Windows: NVIDIA documents Windows support as unsupported and the CuTe DSL quick-start currently lists Linux x86_64/aarch64 as the supported platforms.

For the real `cute_sm120` SOL kernel, use Linux or WSL2. Native Windows is not a supported SOL-kernel execution target in this release. When CuTe is unavailable on native Windows, Sol-H3 falls back locally to the inherited dense attention provider and reports an explicit `native Windows ... requires Linux/WSL2` compatibility reason.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

## Exact Runtime

The Exact Runtime affine path uses a Triton kernel. It is production-validated on Linux/WSL, not native Windows. v0.1.1 therefore fails this optimization closed on `sys.platform == "win32"`: `ineligible_reason()` returns `native_windows_unvalidated` before the Triton kernel is imported or executed, and the normal untouched MiniMax-H3 block runs instead.

This is intentional. A third-party native-Windows Triton build plus ComfyUI's AIMDO malloc-graph compiler is not an execution contract validated by this project. WSL2/Linux continues to use the existing Exact Runtime path unchanged.

## Issue #4: `Packaged Sana source file set mismatch`

The original v0.1.0 provenance verifier used `str(Path.relative_to(...))` to compare on-disk vendored paths with manifest keys. That produces backslashes on Windows while the manifest uses canonical forward-slash paths, so a valid Windows checkout could reject the complete vendored source tree before SOL initialization.

v0.1.1 fixes that bug by using POSIX manifest keys on every platform. It also:

- pins the vendored snapshot to LF checkout in `.gitattributes`;
- accepts only Git-style CRLF-to-LF transport normalization when validating canonical packaged hashes;
- keeps every other file-content difference fail-closed;
- scopes CuTe/Triton runtime dependencies to Linux so native-Windows installation does not try to install an unsupported/unvalidated custom-kernel toolchain;
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

is a separate execution path from Sol-Attn. In issue #4 the final v0.1.0 Sol-H3 telemetry showed:

```text
sol_backend: null
sol_source_tree_verified: false
sparse_calls: 0
exact_blocks: 600
```

Therefore no SOL sparse kernel executed in that failing request. The same request had `exact_fusion=true`, so Exact Runtime was the only Sol-H3 custom kernel path that actually executed. The log does not establish that either Sol-H3 or AIMDO was the root cause.

v0.1.1 removes that ambiguity on native Windows: SOL cannot execute CuTe and falls back dense, while Exact Runtime automatically delegates to native H3 before importing or executing its Triton affine kernel. If `comfy_aimdo` still crashes after upgrading to v0.1.1, capture the new log. A failure with `sparse_calls=0` and `compatibility_fallbacks` containing `exact:native_windows_unvalidated` is evidence that neither Sol-H3 custom kernel path executed, so the remaining crash should be investigated in the ComfyUI/AIMDO path separately.
