import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output, source, rows, width: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    first_row = tl.program_id(0)
    row_stride = tl.num_programs(0)
    for row in range(first_row, rows, row_stride):
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(source + row * width + offsets, mask=offsets < width, other=-float("inf"))
        values = values - tl.max(values, axis=0)
        numerators = tl.exp(values)
        result = numerators / tl.sum(numerators, axis=0)
        tl.store(output + row * width + offsets, result, mask=offsets < width)

