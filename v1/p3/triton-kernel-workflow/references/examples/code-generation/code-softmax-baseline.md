# Softmax Tuning-Handoff Baseline

Load this example only for a softmax-style reduction with `optimization_intent=handoff-to-tuning`. It illustrates the required structure; adapt names, strides, masks, dtype, and block sizes to the actual specification.

```python
@triton.jit
def softmax_baseline(
    x_ptr,
    y_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=-float("inf"))
        row_max = tl.maximum(row_max, tl.max(x, axis=1))

    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=-float("inf"))
        row_sum += tl.sum(tl.exp(x - row_max[:, None]), axis=1)

    for n_start in range(0, N, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=-float("inf"))
        y = tl.exp(x - row_max[:, None]) / row_sum[:, None]
        tl.store(y_ptr + rows[:, None] * N + cols[None, :], y, mask=mask)
```

The Step 4 specification for this structure includes three ordered `reduce_loop_passN` entries and an `optimization_surface` similar to:

```json
{
  "intent": "handoff-to-tuning",
  "operator_pattern": "softmax-style",
  "baseline_form": "chunked-reduction-loop",
  "reduction_axis": {
    "name": "N",
    "extent": 4096,
    "block_parameter": "BLOCK_N",
    "block_candidates": [256, 512, 1024]
  },
  "passes": ["max", "sum", "normalize-store"],
  "parallel_block_parameter": "BLOCK_M",
  "parallel_block_candidates": [4, 8, 16],
  "autotune": {
    "num_stages_candidates": [1, 3],
    "num_warps_candidates": [1],
    "nram_model": "lifetime-aware"
  }
}
```

The numeric values illustrate the reported `(4096, 4096)` case and are not universal defaults. For another shape, generate valid candidates from its real extent and platform properties. Later strategies may eliminate loops, reuse loads, or select another configuration only after compilation, accuracy, and comparable measurement.

