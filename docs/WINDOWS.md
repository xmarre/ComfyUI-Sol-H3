# Native Windows status

Sol-H3 can be installed on native Windows, but its custom SOL and Exact Runtime kernel paths are not supported execution targets there. The nodes remain usable because both paths fail closed to native/inherited execution rather than requiring workflow changes.

## SOL attention

The packaged SOL path uses NVIDIA CUTLASS CuTe DSL for the `cute_sm120` backend. NVIDIA currently documents CuTe DSL Windows support as unsupported and lists Linux x86_64/aarch64 as the supported platforms.

For real SOL custom-kernel execution, use Linux or WSL2 on supported SM120 hardware. On native Windows, Sol-H3 delegates SOL attention to the inherited dense attention provider and records an explicit compatibility fallback instead of attempting an unsupported CuTe execution path.

Official NVIDIA references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html

## Exact Runtime

The Exact Runtime affine path uses a Triton kernel and is production-validated on Linux/WSL2, not native Windows. On `sys.platform == "win32"`, Exact Runtime is therefore marked ineligible before its Triton kernel is imported or executed. The untouched MiniMax-H3 block runs instead and telemetry records:

```text
exact:native_windows_unvalidated
```

This keeps native-Windows execution on the supported MiniMax-H3 path rather than depending on an unvalidated third-party Windows Triton runtime.

## Installation behavior

The custom-kernel runtime dependencies are scoped to Linux. Native-Windows installation therefore does not attempt to install the CuTe/Triton/CUDA-Python/TVM-FFI stack used by the Linux/WSL2 custom-kernel path.

The vendored Sana source verifier is platform-independent:

- manifest paths are canonical POSIX-style paths on every operating system;
- the repository pins vendored source files to LF checkout;
- provenance hashing accepts exact packaged bytes or Git-style CRLF-to-LF transport normalization;
- every other file-content difference still fails closed.

## Troubleshooting

### `Packaged Sana source file set mismatch`

A current checkout should not fail solely because Windows renders filesystem separators differently. If this error appears, update the node checkout first and verify that the vendored files have not been locally modified. The verifier reports missing and unexpected paths to make actual source-tree drift distinguishable from platform formatting.

### `comfy_aimdo` or allocator errors

Allocator failures are not, by themselves, evidence that a Sol-H3 custom kernel executed. Use Sol-H3 telemetry to identify the actual route.

On native Windows, the expected state is:

```text
sparse_calls: 0
SOL: inherited dense fallback
Exact Runtime: exact:native_windows_unvalidated
```

If an allocator or `comfy_aimdo` failure occurs with that state, neither Sol-H3 custom kernel path executed and the allocator failure should be investigated in the ComfyUI/AIMDO path separately.

If telemetry instead shows a Sol-H3 custom kernel executing on native Windows, include the complete telemetry and environment details in a bug report because that is outside the supported platform contract.
