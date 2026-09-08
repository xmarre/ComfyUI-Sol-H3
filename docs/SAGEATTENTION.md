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

First check the active environment and GPU:

```bash
conda activate comfy312
which python
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
PY
/usr/bin/g++ --version
```

For SM120, the capability check must report `(12, 0)`.

### Important: PyTorch CUDA is not the CUDA compiler toolkit

`torch.version.cuda == 13.0` only tells you which CUDA runtime PyTorch was built against. It does **not** mean the system has the CUDA toolkit or `nvcc` installed. SageAttention builds native CUDA extensions and therefore requires an actual CUDA toolkit with `nvcc`.

Check it explicitly:

```bash
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo 'nvcc is not installed'
fi
```

Do not derive `CUDA_HOME` from `command -v nvcc` until this check succeeds. If `nvcc` is absent, expressions based on an empty command result can incorrectly produce `CUDA_HOME=.` and the Sage build then fails with `./bin/nvcc: No such file or directory`.

### WSL: install the CUDA toolkit without installing a Linux NVIDIA driver

On WSL2, NVIDIA explicitly warns not to install driver-bearing CUDA meta-packages because the NVIDIA driver comes from Windows and is exposed into WSL. Use the NVIDIA **WSL-Ubuntu toolkit** package only.

For the documented PyTorch `2.10.0+cu130` environment, install CUDA Toolkit 13.0 from NVIDIA's WSL repository:

```bash
sudo apt update
sudo apt install -y cuda-toolkit-13-0
```

Do **not** use Ubuntu's `nvidia-cuda-toolkit`, and do not install the `cuda`, `cuda-13-0`, or `cuda-drivers` meta-packages on WSL.

If `apt` says `cuda-toolkit-13-0` cannot be found, add/repair NVIDIA's WSL repository keyring first:

```bash
wget -O /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-0
```

Then verify the exact toolkit path:

```bash
/usr/local/cuda-13.0/bin/nvcc --version
```

The toolkit package normally installs `/usr/local/cuda-13.0`; use that exact versioned path for the Sage build instead of relying on a shell `PATH` or `/usr/local/cuda` symlink.

An unrelated APT repository signature warning, for example from the GitHub CLI repository, does not explain a missing `nvcc` if the NVIDIA WSL repository itself is reachable. It should still be repaired separately, but it is not a reason to install Ubuntu's `nvidia-cuda-toolkit` as a workaround.

### Normal build prerequisites

```bash
sudo apt install -y build-essential git ninja-build libgomp1
python -m pip install -U packaging ninja
```

### Rebuild SageAttention 2.2.0 from source

Remove any incompatible binary/wheel install and rebuild the official package against the Python environment, host compiler and CUDA toolkit that actually run ComfyUI:

```bash
conda activate comfy312
python -m pip uninstall -y sageattention
rm -rf /tmp/SageAttention

git clone https://github.com/thu-ml/SageAttention.git /tmp/SageAttention
git -C /tmp/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

cd /tmp/SageAttention
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export EXT_PARALLEL=4
export MAX_JOBS=16

printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
python -m pip install --no-build-isolation --no-cache-dir -v .
```

If you intentionally use another supported toolkit version, set `CUDA_HOME` to that toolkit's actual root and verify `$CUDA_HOME/bin/nvcc --version` before invoking pip. For Blackwell, upstream requires CUDA >=12.8. Matching the toolkit major/minor to the active PyTorch build is the least surprising choice when possible.

Do not fix a Sage binary ABI failure by globally injecting another `libstdc++.so.6` through `LD_LIBRARY_PATH`, `LD_PRELOAD`, or `ctypes`; rebuild SageAttention against the environment/toolchain that will actually run ComfyUI.

### Why this repairs `GLIBCXX_3.4.32` failures

`GLIBCXX_3.4.32` was introduced by GCC 13.2. A SageAttention binary compiled on a newer distribution can therefore fail to load in an older WSL/Ubuntu or Conda runtime even though CUDA itself is fine. Building SageAttention locally with the same host compiler/runtime used by ComfyUI removes that binary-ABI mismatch instead of masking it with a second C++ runtime.

## Verify the SageAttention installation

First verify that the compiled extensions load:

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

SageAttention3 is a separate Blackwell package (`sageattn3`) and is not installed by the SageAttention 2.2.0 root package. Upstream currently documents **Python >=3.13, PyTorch >=2.8 and CUDA >=12.8** for SageAttention3. The documented production `comfy312` environment therefore does not meet its Python requirement. Do not install SageAttention3 into that environment merely to satisfy Sol-H3 testing; it is not required. Test KJNodes `sage3` only in a separate environment that satisfies SageAttention3's upstream requirements.

## Sol-H3 validation after Sage repair

From the Sol-H3 checkout, run the packaged kernel checks first:

```bash
conda activate comfy312
cd /home/toor/ComfyUI/custom_nodes/comfyui-sol-h3
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
python tools/attention_probe.py --backend pytorch --tokens 4096 --prefix 512 --heads 8
python tools/attention_probe.py --backend sage --tokens 4096 --prefix 512 --heads 8
```

The Sage probe must show real Sol-H3 sparse execution and prefix parity. If Sage itself still cannot load, current Sol-H3 records `dense_provider_failures` and demotes that optional dense provider to the original Comfy attention for the current request; this allows Sol-H3's own CuTe/Sana kernel to remain testable. That fallback is diagnostic resilience, not evidence that SageAttention itself is healthy.

For the full integration and production acceptance matrix, continue with [VALIDATION.md](VALIDATION.md).
