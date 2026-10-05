# Reuse the partitioned kernel's dynamic CuTe ABI

Partitioned Sol previously requested a separate compiled function for each
concrete Q/K/V shape and stride tuple. The packaged `to_cute_tensor` already
calls `mark_layout_dynamic(leading_dim=tensor.ndim-1)` on every argument.
All extents and positive non-leading strides are runtime values; changing a
rectangular VDN subcall's row count or positive projection stride does not
change that compiled tensor type.

NVIDIA documents the dynamic-layout contract in
[CUTLASS 4.3.2](https://docs.nvidia.com/cutlass/4.3.2/media/docs/pythonDSL/cute_dsl_general/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-layout-dynamic)
and the [current framework guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html#mark-the-tensor-s-layout-as-dynamic-with-mark-layout-dynamic).
The leading unit stride and broadcast zero strides remain static. The node
uses fixed 16-byte alignment and 64-bit dynamic strides through its unchanged
packaged converter.

The partitioned cache key retains device, architecture, request ABI, batch/head
contract, bias/map specializations and each argument's rank, dtype and broadcast
stride pattern. All ten tensor arguments participate. Arguments whose last
stride is not one retain rejection. Scalar scale and sink ranges are typed
runtime parameters of the unchanged SM120 kernel. Each call binds its current
Q/K/V, output, K/V summaries, threshold, bias, mapped descriptor and LSE; cached
functions do not own those tensors. Compilation uses the existing shared lock
and publishes a function only after successful compilation.

Physical arithmetic qualification remains independent of machine-function
reuse. Its complete shape/stride and semantic-measure identities, independent
Core reference, failure propagation and numerical thresholds are unchanged.
Source validation remains request-owned. Vendor kernel, converter, preprocessing
bytes and their source manifest are unchanged. The ordinary public Sana cache
is untouched; this change applies to `_sm120_union` continuation requests.

Three summary fields distinguish compilation from request validation:

- `partitioned_kernel_compile_calls`: actual compilation attempts in this native request.
- `partitioned_kernel_compile_cache_hits`: launches using an existing compiled function.
- `partitioned_kernel_compile_wall_s`: CPU wall time around compilation, including failures.

The timer excludes preprocessing, kernel execution, independent arithmetic
references and explicit GPU synchronization. Reused functions can originate
in earlier requests; arithmetic checks still run in the current request.
Cold versus warm process state matters when comparing captures.

The host acceptance matrix covers changing rectangular extents, interleaved
positive-stride views, live data/output/scalar rebinding and distinct bias/map
specializations. Nine launches request three functions instead of nine. Negative
cases cover broadcast, rank, dtype and invalid leading stride. Existing weighted
tests retain requalification for changed physical layout and semantic measure.

`tests/test_partitioned_dynamic_compile.py` also contains real SM120 tests:
each weighted/unweighted function executes multiple rectangles and interleaved
views against an independent FP32 attention oracle, with the existing arithmetic
thresholds, while asserting a single compilation. Those tests skip on CPU.
Host reuse does not establish GPU numerical acceptance, GPU latency or total
prompt speedup. 01041 predates this change and does not isolate compilation
time; its roughly 2.8 s gate durations include compilation and queued work.

Use the existing ComfyUI Patcher Sol #37 overlay. Optional #35 remains an
independent preceding diagnostic overlay. No new user configuration is needed.
