"""Offline CuTe ABI compilation; these tests never launch a GPU kernel."""
import pytest


@pytest.mark.parametrize("arch", ["sm_120", "sm_121"])
@pytest.mark.parametrize("weighted,mapped", [(False, False), (True, False), (False, True), (True, True)])
def test_query_origin_cute_abi_compiles_offline(monkeypatch, arch, weighted, mapped):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
    from sol_h3._vendor.sol_attn.sm120 import make_kernel

    monkeypatch.setenv("CUTE_DSL_ARCH", arch)

    def tensor(dtype, shape):
        return make_fake_compact_tensor(dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16)

    # Q has the original packed projection stride; K/V are compact full-sequence
    # carriers. Include irregular lengths and enabled optional runtime tensors.
    q = make_fake_tensor(cutlass.BFloat16, (1, 65, 56, 128), (7168, 21504, 128, 1), assumed_align=16)
    k = tensor(cutlass.BFloat16, (1, 193, 56, 128))
    o = tensor(cutlass.BFloat16, (1, 65, 56, 128))
    kc = tensor(cutlass.BFloat16, (1, 4, 56, 128))
    threshold = tensor(cutlass.Float32, (1, 2, 56))
    bias = tensor(cutlass.Float32, (193,)) if weighted else threshold
    mapping = tensor(cutlass.Int32, (2, 2)) if mapped else threshold
    lse = tensor(cutlass.Float32, (1, 65, 56))
    compiled = cute.compile(
        make_kernel(key_bias_enabled=weighted, mapped_neighbors_enabled=mapped),
        q, k, k, o, kc, kc, threshold, bias, mapping, lse,
        128 ** -0.5, 0, 2, 1, stream=make_fake_stream(),
        options=f"--enable-tvm-ffi --gpu-arch {arch}",
    )
    assert compiled is not None
