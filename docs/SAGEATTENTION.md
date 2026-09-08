# SageAttention installation and repair

SageAttention is optional. Sol-H3 can run without it; Sage is only an inherited dense-attention provider for dense warmup/prefix/native fallback work. A broken Sage installation must not be treated as a Sol-H3 kernel failure.

## Blackwell / SM120 recommendation

For NVIDIA Blackwell `sm120` (RTX 50-series and RTX PRO 6000 Blackwell), use the upstream `sageattn` automatic dispatcher after building SageAttention from source. The upstream SageAttention 2.2.0 code explicitly routes `sm120` to its CUDA FP8/SageAttention2++ path and states that the Triton kernel is currently not usable on `sm120`.

Therefore, in KJNodes use **`auto`** for SageAttention on SM120. Do not select `sageattn_qk_int8_pv_fp16_triton` on SM120 as the normal production setting.

The source-build instructions below are pinned to upstream commit:

```text
d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
```

That revision's root package is SageAttention 2.2.0 and its build supports compute capabilities `12.0` and `12.1`. Upstream requires Python >=3.9, PyTorch >=2.3, Triton >=3.0, and CUDA >=12.8 for Blackwell.

## Clean install / repair on Linux or WSL

Run these commands in the same Python environment that launches ComfyUI. For the documented production environment this is `comfy312`.

First check the active environment and toolchain:

```bash
conda activate comfy312
which python
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
PY
which nvcc
nvcc --version
/usr/bin/g++ --version
```

For SM120, the capability check must report `(12, 0)` and the CUDA toolkit used for the build must be >=12.8.

Install the normal build prerequisites if needed:

```bash
sudo apt update
sudo apt install -y build-essential git ninja-build
python -m pip install -U packaging ninja
```

Then remove any incompatible binary/wheel install and rebuild the official package from source against the machine's own C++ runtime and CUDA toolkit:

```bash
conda activate comfy312
python -m pip uninstall -y sageattention
rm -rf /tmp/SageAttention

git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
git -C /tmp/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

cd /tmp/SageAttention
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=12.0
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export EXT_PARALLEL=4
export MAX_JOBS=16

python -m pip install --no-build-isolation --no-cache-dir -v .
```

If `/usr/local/cuda` is not the toolkit used by `nvcc`, set `CUDA_HOME` to the real CUDA toolkit root before building. Do not fix this by globally injecting another `libstdc++.so.6` through `LD_LIBRARY_PATH`, `LD_PRELOAD`, or `ctypes`; rebuild SageAttention against the environment/toolchain that will actually run ComfyUI.

### Why this repairs `GLIBCXX_3.4.32` failures

`GLIBCXX_3.4.32` was introduced by GCC 13.2. A SageAttention binary compiled on a newer distribution can therefore fail to load in an older WSL/Ubuntu or Conda runtime even though CUDA itself is fine. Building SageAttention locally with the same host compiler/runtime used by ComfyUI removes that binary-ABI mismatch instead of masking it with a second C++ runtime.

## Verify the SageAttention installation

First verify that both compiled extensions load:

```bash
conda activate comfy312
python - <<'PY'
import sageattention
import sageattention._fused
import sageattention._qattn_sm89
print('sageattention:', sageattention.__file__)
print('compiled extensions: OK')
PY
```

Then execute the upstream automatic SM120 path:

```bash
python - <<'PY'
import torch
from sageattention import sageattn

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
q = torch.randn(1, 8, 2048, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
with torch.no_grad():
    out = sageattn(q, k, v, tensor_layout='HND', is_causal=False)
torch.cuda.synchronize()
print('output:', tuple(out.shape), out.dtype, out.device)
assert out.shape == q.shape
assert torch.isfinite(out).all()
print('sageattn SM120 execution: OK')
PY
```

## Optional SageAttention3

SageAttention3 is a separate Blackwell package (`sageattn3`). It is not installed by the SageAttention 2.2.0 root package. If you want to test the KJNodes `sage3` mode as a separate A/B, build it from the same pinned checkout:

```bash
conda activate comfy312
cd /tmp/SageAttention/sageattention3_blackwell
python -m pip install --no-build-isolation --no-cache-dir -v .
python - <<'PY'
from sageattn3 import sageattn3_blackwell
print('sageattn3 import: OK', sageattn3_blackwell)
PY
```

Upstream currently documents SageAttention3 with stricter requirements (`python>=3.13`, `torch>=2.8`, CUDA >=12.8). Treat it as an optional separate experiment; SageAttention2/2++ via `sageattn` is the normal Blackwell path documented here.

## Sol-H3 validation after Sage repair

From the Sol-H3 checkout, run the packaged kernel checks first:

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/ComfyUI-Sol-H3
python -m pip install -r requirements.txt
python -m pip check

python - <<'PY'
import torch
from sol_h3.provenance import verify_source
from sol_h3.sparse import load_kernel
print(verify_source()['revision'])
kernel = load_kernel(torch.device('cuda:0'))
print('Sol-H3 backend:', kernel.backend_name)
assert kernel.backend_name == 'cute_sm120'
PY

python -m pytest -q tests/test_gpu.py
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512
```

The Sage probe must show real Sol-H3 sparse execution and prefix parity. If Sage itself still cannot load, current Sol-H3 records `dense_provider_failures` and demotes that optional dense provider to the original Comfy attention for the current request; this allows Sol-H3's own CuTe/Sana kernel to remain testable. That fallback is diagnostic resilience, not evidence that SageAttention itself is healthy.

For the full integration and production acceptance matrix, continue with [VALIDATION.md](VALIDATION.md).
