import triton
import triton.language as tl


@triton.jit
def scalar_kernel(output, source, WIDTH: tl.constexpr):
    value = tl.load(source)
    tl.store(output, value + WIDTH)

