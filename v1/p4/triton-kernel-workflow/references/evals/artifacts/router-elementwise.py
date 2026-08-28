import triton
import triton.language as tl


@triton.jit
def add_kernel(output, left, right, size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    tl.store(output + offsets, tl.load(left + offsets, mask=mask) + tl.load(right + offsets, mask=mask), mask=mask)

