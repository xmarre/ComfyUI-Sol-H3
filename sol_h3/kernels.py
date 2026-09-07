"""Original Triton affine implementation preserving native materialization rounding."""
import triton
import triton.language as tl


@triton.jit
def affine_kernel(H, SHIFT, SCALE, INDEX, N: tl.constexpr, WIDTH: tl.constexpr,
                  HS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr,
                  CS: tl.constexpr, CC: tl.constexpr, ROW: tl.constexpr,
                  INDEXED: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // WIDTH, offset % WIDTH
    valid = offset < N * WIDTH
    if INDEXED:
        mod = tl.load(INDEX + row, row < N, other=0)
    else:
        mod = ROW
    dtype = H.dtype.element_ty
    scale = tl.load(SCALE + mod * CS + col * CC, valid, other=0).to(dtype).to(tl.float32)
    shift = tl.load(SHIFT + mod * SS + col * SC, valid, other=0).to(dtype).to(tl.float32)
    h = tl.load(H + row * HS + col, valid, other=0).to(tl.float32)
    # Native: scale.to(dtype); (1 + scale) -> dtype; h.mul_ -> dtype;
    # h.add_ -> dtype. Do not contract mul/add or elide either rounding.
    factor = (1.0 + scale).to(dtype).to(tl.float32)
    product = (h * factor).to(dtype).to(tl.float32)
    tl.store(H + row * HS + col, product + shift, valid)
