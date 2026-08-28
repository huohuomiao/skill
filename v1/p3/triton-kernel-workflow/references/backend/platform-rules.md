# Platform Rules

This is the only source of device, Grid, on-chip memory, dtype, primitive, and device-math constraints for the target accelerator. It does not decide local versus Worker execution; that decision belongs only to `../contracts/execution-backend.md` and `EnvConfig/config.json`.

Read only the sections needed by the current task.

## Runtime and device adaptation

For the MLU target:

- Replace CUDA device literals with their MLU equivalents: `cuda` → `mlu`, `is_cuda` → `is_mlu`, and `torch.device("cuda")` → `torch.device("mlu")`.
- Use `torch.mlu.synchronize()` around timed device work.
- Do not treat a CPU fallback, a PyTorch reference implementation, or skipping the Triton kernel as a valid repair.

## Device properties

Read physical core and NRAM information from the backend driver instead of hard-coding it:

```python
import torch
from triton.backends.mlu import driver

props = driver.BangUtils().get_device_properties(torch.mlu.current_device())
total_core_num = props["cluster_num"] * props["core_num_per_cluster"]
max_nram_size = props["max_nram_size"]
```

An existing project may keep a verified `torch.mlu.get_device_properties(...)` path. Do not mix device-property APIs inside one generated file.

## Grid and persistent kernels

- Check every Grid dimension against the installed backend limit; older toolchains commonly limit one dimension to `65535`.
- A compressed one-dimensional Grid may use `min(logical_grid, total_core_num // num_warps)`.
- After Grid compression, cover every logical task with `tl.num_programs(axis=0)` and a fixed-stride loop.
- A multidimensional Grid may be flattened when the kernel reconstructs every logical index exactly.
- If increasing a block to reduce Grid conflicts with decreasing it to fit NRAM, prefer a persistent kernel instead of oscillating between settings.

```python
@triton.jit
def persistent_kernel(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    for block_start in range(pid * BLOCK_SIZE, size, programs * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        value = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, value, mask=mask)
```

## NRAM and tuning

- Use the runtime `max_nram_size`; 512 KiB is only a conservative fallback when properties cannot be read.
- On NRAM exhaustion, first reduce non-reduction block sizes, then reduce `num_stages`, then reassess `num_warps`.
- When NRAM utilization is low and parallelism or memory traffic is the bottleneck, enlarge high-priority block axes gradually.
- Prefer powers of two or multiples of 32 for block candidates, then retain a candidate only after real accuracy and performance measurements.
- Obtain `num_warps` and `num_stages` choices from target-platform measurements rather than CUDA defaults.
- Estimate NRAM from simultaneously live values and proven pipeline buffers. Do not apply an unconditional full-tile multiplier for `num_stages` or sum intermediates whose lifetimes do not overlap.
- Treat a plausible but uncertain NRAM estimate as a compile candidate. A real compiler resource error may eliminate it; static uncertainty alone must not be reported as a measured failure.
- For I/O-bound row reductions, preserve at least one small non-reduction block candidate when output ownership remains valid. This leaves room for pipelining without asserting that the smallest block is fastest.

## Dtype, primitives, and device math

- Read `{skill_root}/references/backend/supported-primitives.md` before using a primitive or dtype combination whose support is uncertain.
- Read `{skill_root}/references/backend/math-functions.md` before generating a Libdevice call.
- Use `{skill_root}/references/strategies/device-math.md` only when a matching device-math optimization is applicable.
- Do not widen or narrow a dtype merely to silence a compiler error unless equivalence and the accuracy contract are preserved.

## Backend optimization constraints

- A fixed-step counting `while` loop may become a `for` loop only when its value sequence and post-loop semantics are provably identical.
- A three-dimensional tile reduced along its middle dimension may be lowered to two dimensions only when shape and index equivalence are proven.
- Absorb wrapper `transpose/permute + contiguous` into kernel indexing only when the same layout semantics are preserved.

## Verification invariants

1. Device literals and synchronization APIs match the target platform.
2. Primitive and dtype combinations are supported.
3. Grid compression still covers all logical work.
4. The kernel compiles without Grid or NRAM resource errors.
5. Accuracy uses the original reference and tolerance.
6. Performance comparisons use identical inputs, timing method, hardware, and backend.
7. Failed modifications preserve real stdout/stderr and never become success claims.
