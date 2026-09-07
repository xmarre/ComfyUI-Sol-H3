import pytest
import torch

from sol_h3.exact import affine, native_affine

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("width", [128, 257, 3072])
def test_affine_native_bitwise(dtype, indexed, width):
    torch.manual_seed(991)
    h = torch.randn(67, width, device="cuda", dtype=dtype)
    # Strided chunk views match AdalnProj.chunk's actual output.
    params = torch.randn(7, 6 * width, device="cuda", dtype=torch.float32)
    shift, scale = params.chunk(6, dim=-1)[:2]
    row = torch.arange(62, device="cuda") % 7 if indexed else 3
    segments = [(0, 5, 1), (5, 67, row)]
    want = h.clone()
    for a, b, idx in segments:
        native_affine(want[a:b], shift, scale, idx)
    got = affine(h.clone(), shift, scale, segments, set())
    assert torch.equal(got, want)
