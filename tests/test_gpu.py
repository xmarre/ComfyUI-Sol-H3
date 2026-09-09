import pytest
import torch

from sol_h3.exact import affine, native_affine

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("width", [128, 257, 5376])
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


@pytest.mark.parametrize('rows,prefix', [(65, 1), (130, 65), (4096, 512)])
def test_sana_sm120_real_kernel(rows, prefix):
    from sol_h3.contracts import Config
    from sol_h3.runtime import Request
    from sol_h3.sparse import attention
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip('SM120 required')
    torch.manual_seed(173)
    q, k, v = (torch.randn(1, 2, rows, 128, device='cuda', dtype=torch.bfloat16) for _ in range(3))
    cfg = Config(exact=False, backend='sol')
    state = Request(cfg)
    result = attention(q, k, v, prefix, cfg, state).reshape(1, rows, 2, 128)
    torch.cuda.synchronize()
    assert state.kernel.backend_name == 'cute_sm120'
    assert state.kernel.source_tree_verified
    assert state.sparse_calls == 1
    assert torch.isfinite(result).all()
    want = torch.nn.functional.scaled_dot_product_attention(q[:, :, :prefix], k, v).transpose(1, 2)
    torch.testing.assert_close(result[:, :prefix], want, rtol=0, atol=0)
