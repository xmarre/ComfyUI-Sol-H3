"""Compile for SM120 offline; this does NOT execute or validate a GPU kernel."""
import pytest

triton = pytest.importorskip("triton")


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("indexed", [False, True])
def test_offline_sm120_compilation(dtype, indexed):
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from sol_h3.kernels import affine_kernel

    constants = {"WIDTH": 5376, "SS": 32256, "SC": 1, "CS": 32256, "CC": 1,
                 "INDEXED": indexed, "BLOCK": 256}
    signature = {"H": f"*{dtype}", "SHIFT": "*fp32", "SCALE": "*fp32", "INDEX": "*i64",
                 "N": "i32", "HS": "i64", "ROW": "i32"}
    source = ASTSource(affine_kernel, signature=signature, constexprs=constants)
    compiled = triton.compile(source, target=GPUTarget("cuda", 120, 32), options={"enable_fp_fusion": False})
    ptx = compiled.asm["ptx"]
    assert "sm_120" in ptx
    assert "fma.rn" not in ptx
    if dtype == "bf16":
        assert "cvt.rn.bf16.f32" in ptx
