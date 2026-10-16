# BF16 attention memory architecture

Status: implemented candidate; CUDA execution, sampler memory/time and decoded
quality acceptance are open. Historical release measurements do not qualify
this producer or the query-origin kernel ABI.

## Allocation findings

The source audit compares Sol-H3 v0.1.10 (`5a5e57b`),
[ComfyUI `ba994879`](https://github.com/Comfy-Org/ComfyUI/blob/ba994879b88d0ebd9950b1e5c1e923d928f216ba/comfy/ldm/minimax/model.py),
[Core sparse attention](https://github.com/Comfy-Org/ComfyUI/blob/ba994879b88d0ebd9950b1e5c1e923d928f216ba/comfy_extras/nodes_sparse_attention.py),
and [comfy-kitchen `b614102f`](https://github.com/Comfy-Org/comfy-kitchen/blob/b614102f8f0fcbba00d7fe570c31eeeda4f0061d/comfy_kitchen/backends/cuda/sage_attention/sol_attn.cu).
Relevant companion snapshots are VDN-H3-Plus `1e19b986`, Flow `89925af6`, and
Spectrum `806fe498`. These are source comparisons, not a matched GPU campaign.

### 1. Callback cycles retain block arguments

The released block bridge installs temporary callbacks in a copied transformer
options dictionary. Those callbacks close over the same dictionary, the block
arguments and the Request. The resulting cycle keeps the hidden tensor, RoPE
and other argument-owned objects alive until Python cyclic collection.

The defect is reproduced on CPU using PyTorch storage weak references with
cyclic GC disabled: the original bridge retains the underlying hidden storage
after the call; explicit GC releases it. The corrected bridge restores inherited
provider entries in `finally`, so storage expires immediately on successful and
failed calls. It does not invoke GC or flush the CUDA allocator.

For 100,000 rows, a BF16 `[T,5376]` hidden allocation is **1.001 GiB**. Native H3
updates its residual in place, so multiple blocks can refer to the same storage
within one evaluation. Do not multiply this amount by the number of blocks.
Distinct evaluations, physical streams and retained companion arguments require
separate storage accounting. The observed GPU peak contribution remains unmeasured.

### 2. First-use arithmetic qualification can dominate peak memory

The old ordinary gate holds complete all-selected and dense-reference outputs.
Its metric function also holds FP32 candidate, reference, difference and absolute
difference arrays. Computing the reference absolute peak creates another FP32
temporary. None of these arrays is needed for the subsequent sparse output.

Let `S = T * H * 128 * 2`, the size of one BF16 sequence. At the reference-peak
reduction, the listed allocations account for `3S` packed input + `2S` outputs +
`10S` FP32 arrays, or **15S**. At 100,000 rows and 56 heads this is **20.027 GiB**,
before model weights, hidden streams and allocator/backend overhead. This is a
source-derived allocation count, not a measured CUDA peak.

Qualification now visits every query row in chunks against all original keys.
Unweighted references use at most 4096 query rows; weighted references use 64
because a masked SDPA backend may allocate score matrices. Candidate/reference
outputs are consumed immediately. Metrics use at most 262,144 elements per tile,
FP32 differences and FP64 scalar accumulation, then transfer five statistics
together. Array scratch is bounded to roughly 8 MiB plus reduction internals.
Each streamed query origin is qualified once per request/layout; sharing a
compile shape does not skip later chunks of the initial full sequence. Later
blocks with the same origin/layout reuse that proof, as native layout proofs do.
The existing finite, mean-absolute, relative-L2 and catastrophic-peak limits are
unchanged. This is full query coverage, not sampled qualification.

This bound applies to `sparse.attention`, including its VDN rectangular/mapped
calls. Partitioned request qualification retains its existing output architecture;
its shared error statistics receive the bounded scratch improvement.

### 3. Full QKV is one packed allocation

Native H3 performs `qkv_proj -> split whole Q/K/V slabs -> per-head norm/RoPE ->
optimized_attention -> out_proj`. With RoPE, Q/K normalization and rotation run
in place. The ordinary Sol bridge transposes views without copying them.
Three input views therefore account for one `[T,3*H*128]` BF16 backing allocation,
not three independent packed allocations. The legacy `materialized_qkv_bytes=0`
reports bridge copies; it never meant that native QKV used no memory.

At 100,000 rows / 56 heads:

| Listed allocation | GiB |
| --- | ---: |
| One BF16 Q, K, V or attention output | 1.3351 |
| Native packed QKV | 4.0054 |
| Native packed QKV plus attention output | 5.3406 |
| K/V block summaries combined | 0.0417 |
| Full ordinary LSE | 0.0209 |
| FP32 routing threshold | 0.00033 |

The summaries and ordinary LSE do not explain a multi-GiB difference by
themselves. LSE remains necessary for partitioned normalization/merging.

### 4. Core MSA is a different numerical and storage contract

Core's H3 producer projects 4096 rows at a time into INT8 carriers. Its workspace
contains compressed Q/K/V, scales, summaries, selected-block indices and optional
token augmentation. In the audited workspace plan, Q/K/V carriers consume about
half their BF16 equivalents; the selected-block index term is quadratic in block
count (`2 * B * H * ceil(T/64)^2` bytes). Token augmentation also allocates a
row-major V carrier. Compare the actual `token_aug` setting when measuring.

The chunked Core producer uses previous-evaluation K-centering/V-scaling
statistics after bootstrap. Sol-H3 uses current BF16 centroids, exact BF16 tile
retrieval and its vendored Sana routing arithmetic. Core MSA is a practical
memory comparator, not a numerical oracle. Its INT8 precision, routing, token
augmentation and warmup settings must be reported separately.

### 5. Compilation and reserved memory

Sol's global cache stores compiled programs; Requests store kernel functions,
proof metadata and bounded measure/mapping data. Full QKV and prepared K/V are
not stored in those caches. Offline CuTe/DLPack storage-lifetime checks on DSL
4.7.1 and 4.8.0 released all original CPU-backed input storages while the compiled
program stayed alive. This rules out that specific Python/DLPack retention path
in those checks; real-device compiler/driver residency remains a measurement item.

PyTorch reserved bytes are reported separately from live allocated bytes. CUDA
driver modules, context allocations and allocations outside PyTorch may contribute
to device-wide usage. Neither a large reserved pool nor `nvidia-smi` alone proves
that a live Sol tensor leaked. Production performs no `empty_cache()` calls.

## Streamed native producer

`qkv_chunk_size=4096` enables a two-pass producer for eligible long native H3
calls; `0` keeps native full QKV production. The node input is optional for saved
workflow compatibility. Streaming is selected above eight chunks (32,768 rows
at the default) to leave room for qualification/prefix temporaries and amortize
projection overhead.

1. Call the original `qkv_proj` on bounded row slices. Apply Core's original
   in-place RMSNorm/partial split-half RoPE. Copy only K and V to independent
   full-length BF16 carriers and release every packed chunk.
2. Allocate the final attention output. Re-run the same module on each row slice,
   retain only that chunk's Q view, and attend against the full BF16 K/V.
3. Compute K/V summaries once per invocation and reuse them for qualification
   and query chunks. Query thresholds use the original global K statistics.
4. Copy each attention result into the final output and release the chunk.
   Release K/V and their summaries before the original output projection.

Noncausal Sol needs global key statistics and arbitrary exact K/V tiles. Keeping
BF16 K/V avoids lossy compression or repeated projection of selected key tiles.
Recomputing Q through the existing projection preserves its quantized/module
owner. It adds one projection pass; a sampler-time benefit is **not established**.

The SM120/SM121 kernel receives a runtime global query-block origin. Routing
uses `local_query_block + origin` for the original diagonal/neighbor rule, while
Q/output/LSE addressing and supplied mapped descriptors remain local. Chunk
starts are 64-aligned; only the last chunk may end in a partial tile. Ordinary,
weighted, rectangular and partitioned callers default to origin zero.

For 56 heads and 4096-row chunks, a source-derived steady attention storage
comparison (excluding summaries/allocator/GEMM/hidden storage) is:

| Rows | Native QKV + output | BF16 K/V + output + packed chunk + chunk output |
| ---: | ---: | ---: |
| 32,768 | 1.7500 GiB | 1.5312 GiB (below production streaming cutoff) |
| 65,536 | 3.5000 GiB | 2.8438 GiB |
| 100,000 | 5.3406 GiB | 4.2242 GiB |
| 200,000 | 10.6812 GiB | 8.2296 GiB |

These are array-size estimates. Qualification adds two bounded outputs and metric
scratch. Output projection, MLPs, model residency or other integrations can own
the actual sampler peak. Only an end-to-end run can establish the net saving.

## Ownership and interoperability

The producer enters through the existing block `attention` argument. It does not
mutate module forwards, clone models or construct a second weight producer.
It requires an audited native Attention/DiTBlock forward, BF16, RoPE and
SM120/SM121 inference. Unknown source, autograd, graph capture, arbitrary hooks or compiled
module calls retain native execution. A VDN-owned forward may opt into bounded
QKV projection while its QKV module has only VDN's exact
`vdn_h3.apply._PostForwardLoRA` post-hooks. These hooks apply a row-local
additive low-rank residual (and optional bias), so slicing changes the batch
geometry without changing which tokens participate in the adapter. Block/attention
hooks, projection pre-hooks, unknown projection post-hooks, global hooks and
compiled module calls still fail closed. The special admission does not enable
the native two-pass producer with hooks.

VDN forwards consume raw Q/K/V for their learned linear branch. With the
cooperating VDN-H3-Plus bounded-projection API v1 installed, VDN's *native
forward* can instead receive independent full-length BF16 Q/K/V carriers filled
from bounded native projection chunks. Its original learned linear branch,
post-RoPE preprocessing, grouped-window/gated softmax and output projection
still execute in VDN. Independent V storage avoids its packed-QKV preservation
clone. Sol requires the explicit VDN forward capability marker and does not
modify or replace the VDN attention operator. Without a compatible VDN overlay,
it continues to use VDN's native producer and records the skip reason.

At 100,000 rows and 56 heads, native packed QKV plus VDN V clone uses
5.3406 GiB before softmax/output. Independent Q/K/V carriers plus one 4096-row
packed BF16 chunk use approximately 4.169 GiB, a 1.17 GiB reduction in those
particular listed allocations. These sizes are source-derived; actual CUDA peaks
and latency remain unmeasured. The VDN path uses **one** bounded projection pass,
not the native Sol two-pass Q streamer. Its counters are
`streamed_vdn_projection_calls` and `streamed_vdn_projection_rows`, separate from
`streamed_blocks`.

### Runtime performance finding, October 2026

The supplied two-chunk Continuum/Flow/VDN run telemetry reports
`qkv_chunk_size=4096` but `streamed_vdn_projection_calls=0` and
`streamed_skips={"module_hooks_or_compile":300}` per six-model-call stage.
VDN `lora_mode=bypass` had installed 209 module post-forward hooks. Thus the
observed prior sampler memory decrease **cannot** establish that bounded QKV
ran; Sol was using the unbounded native projection. This candidate narrows
admission to exactly the VDN row-wise hook while retaining strict rejection of
unknown/stateful hooks. It must produce nonzero projection counters on the
actual CUDA workflow before claiming any QKV memory saving.

The additive hook itself runs on each projection chunk, and the INT8 convrot
base can be re-launched on many small slices. This increases kernel launches
and may change GEMM shape-dependent rounding. A larger chunk reduces launches
but raises the peak packed temporary allocation. No faster default is asserted
without GPU evidence. The targeted script
`tools/vdn_qkv_projection_benchmark.py` runs the real quantized Comfy
projection with VDN's post-forward adapter and compares full, 4096, 8192 and
16384-row arms at typical low/high sequence lengths, reporting warmed CUDA
latency, allocator peaks, retention and representative output parity.


VDN's Flow-aware partitioned forward can use the same carrier producer when
the VDN object-patch wrapper advertises bounded-projection API 2. This is a
QKV allocation-only change: physical mixed/equal-grid domains, protected
prefix weighting, grouped attention, diagnostics and pre-/post-RoPE raw
linear reads continue in their original owners. The wrapper must explicitly
publish API 2 or Sol rejects the partitioned admission. Other foreign physical
producers, weighted measures and inherited preprocessing remain on their
established path;
the native producer must not bypass them. Runtime projection weight/bias hooks
and explicit full-precision/cast overrides also retain native execution: repeating a dynamic full-weight adapter patch per
chunk can be both stateful and prohibitively expensive. Merged weights and
ordinary native INT8 convrot projections use the original projection module.

Dense warmup and prefix rows keep their inherited dense provider. Sparse policy,
exact conditioning sinks, weighted normalization, mapped-neighbor ABI and
partitioned normalization are preserved. Config fingerprints include producer
choice/chunk size; actual streamed calls emit `sol_streamed` or
`dense_warmup_streamed`. Spectrum accepts these distinct receipts. Warmup NFE
counting remains block/evaluation based; attention-call counters can increase
because one streamed block dispatches multiple query chunks.

Every invocation owns its stream context and prepared K/V. Context tokens and
workspace references are cleared in `finally`. Production allocates/consumes on
the current CUDA stream and keeps no tensor workspace across blocks, evaluations
or requests. No cross-request workspace pool or process-wide proof cache is added.

## Reproducible CUDA acceptance

Install this implementation through the ComfyUI Patcher PR overlay and restart
ComfyUI. Keep the remaining overlay stack, model, references, LoRAs, seeds,
resolution, duration, sampler, precision, attention settings and residency fixed.

### Operator and projection tests

From the Sol node checkout in the ComfyUI environment:

```bash
python -m pytest -q tests/test_streamed_qkv.py tests/test_weighted_sm120_gpu.py tests/test_zero_copy_bthd.py tests/test_partitioned_dynamic_compile.py
python tools/streamed_attention_benchmark.py --comfyui-path /path/to/ComfyUI --projection int8-convrot --rows 32768 65536 100000 --output /path/to/results/streamed-int8.json
python tools/streamed_attention_benchmark.py --comfyui-path /path/to/ComfyUI --projection bf16 --rows 32768 65536 100000 --output /path/to/results/streamed-bf16.json
python tools/vdn_qkv_projection_benchmark.py --comfyui-path /path/to/ComfyUI --vdn-path /path/to/ComfyUI/custom_nodes/comfyui-vdn-h3-plus --rows 54744 107630 --chunks 4096 8192 16384 --output /path/to/results/vdn-bounded-qkv.json
```

Repeat with `--dense` for inherited dense warmup and `--order streamed-first` to
check ordering/cache effects. Run in fresh processes for cold-process timing.
The runner uses actual Core H3 projection/norm/RoPE/output modules and the Sol
bridge with synthetic weights/activations. It reports first-request and warmed
allocated/reserved peaks, incremental allocation, output-release/request-cleanup
retention, CUDA and wall latency, and whole-output arithmetic error. CPU copies
for comparison occur after GPU timing/memory accounting. The `0` arm isolates
native production with the candidate's bounded gate; it is not a reproduction of
the old release's unbounded validation peak.

Query slicing on identical Q/K/V must be bitwise equal to unsliced Sol, including
irregular/rectangular, all-selected, sparse, weighted and mapped cases. Full native
versus streamed projection uses the existing production arithmetic limits; these
are fixed before observing candidate GPU outputs. A failure stays a failure.

### Real sampler and decoded media

Set `SOL_H3_MEMORY_PROFILE=1` in the environment used to launch ComfyUI. Each
Sol-owned sample emits one `Sol-H3 memory` JSON record. Profiling synchronizes only
at request boundaries, resets process-wide CUDA allocator peaks once, attributes
new observed peaks to stage checkpoints, and records per-evaluation live memory.
The peak stage is the first checkpoint observing that peak, not an allocation
stack trace. Run without concurrent GPU jobs or other peak-resetting profilers.

Compare `qkv_chunk_size=0` and `4096` for at least three increasing sequence
lengths, including the problematic high-resolution case. Record a first process
run and at least three warmed identical runs per arm, then reverse arm order.
Compare allocated and reserved peaks separately. Require `streamed_blocks > 0`
before attributing a change to native projection; inspect `streamed_skips` for
delegated owners. Record actual/forecast NFE counts and inherited dense backend.

Exercise native single-chunk, VDN (with and without the companion bounded
projection VDN overlay), Flow continuation, mixed-grid/partitioned and Spectrum
on/off configurations. Require `streamed_vdn_projection_calls > 0` before
attributing a VDN workload's memory change to bounded QKV projection. VDN-owned physical Flow partitioned execution can now use this same bounded
projection path; require nonzero `streamed_vdn_projection_calls` on the
actual low/probe/high stages. Compare the same continuation with `qkv_chunk_size=0` and `4096`, verifying exact
audio/video prefix ownership, full attention receipts, runtime retention and
decoded quality on a real SM120/SM121 device. Repeat generations
with prompt/reference changes; check post-request live memory against equivalent
output/model residency. Observe device-wide usage separately if it disagrees
with PyTorch's counters. Disable profiling for final timing comparisons.

Retain workflow JSON, complete logs and identically encoded decoded media. Compare
motion, temporal coherence, reference fidelity, audio and continuation boundaries.
Synthetic operator parity or lower allocated memory does not establish quality.

Release requires real-device arithmetic passes, a substantial high-resolution
sampler peak reduction, stable repeated-request live allocation, and acceptable
sampler time and decoded quality in the supported deployment matrix. Core MSA
must be measured with the same model/residency/geometry and its different
precision/routing settings recorded. Any remaining memory gap must be measured;
there is currently no measured parity or speed claim.
