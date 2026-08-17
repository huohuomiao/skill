## 多维 Grid 规约：等价展平

**场景**：多个 program 处理同一输出行的不同 N 块，原实现已经用 atomic 合并部分和。展平只能改变 PID 编码，不得改变规约合并语义。

## 初始二维 Grid

```python
@triton.jit
def row_sum_partial(a_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets_n < N
    values = tl.load(a_ptr + pid_m * N + offsets_n, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(out_ptr + pid_m, partial)

def run(a):
    M, N = a.shape
    BLOCK_N = 256
    out = torch.zeros(M, device="cuda", dtype=torch.float32)
    grid = (M, triton.cdiv(N, BLOCK_N))
    row_sum_partial[grid](a, out, M, N, BLOCK_N=BLOCK_N, num_warps=4)
    return out
```

输出使用加法单位元初始化，是因为原 kernel 明确使用 `tl.atomic_add`。max/min/按位规约必须使用各自正确单位元，不能统一零初始化。

## 普通一维展平候选

```python
@triton.jit
def row_sum_partial(a_ptr, out_ptr, M, N, BLOCK_N: tl.constexpr):
    flat_pid = tl.program_id(0)
    blocks_n = tl.cdiv(N, BLOCK_N)
    pid_m = flat_pid // blocks_n
    pid_n = flat_pid % blocks_n
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets_n < N
    values = tl.load(a_ptr + pid_m * N + offsets_n, mask=mask, other=0.0)
    partial = tl.sum(values, axis=0)
    tl.atomic_add(out_ptr + pid_m, partial)

def run(a):
    M, N = a.shape
    BLOCK_N = 256
    out = torch.zeros(M, device="cuda", dtype=torch.float32)
    blocks_n = triton.cdiv(N, BLOCK_N)
    grid = (M * blocks_n,)
    row_sum_partial[grid](a, out, M, N, BLOCK_N=BLOCK_N, num_warps=4)
    return out
```

## 判断与约束

1. 二维 Grid 在 CUDA 上有效，展平不是强制优化。
2. 展平 Grid 仍为 `M * blocks_n`，不限制到 SM 数；每个原逻辑 program 恰好执行一次。
3. 原实现若使用普通 store 且每个输出只由一个 program 负责，展平后必须继续保持独占写，不能擅自引入 atomic。
4. 若值得测试 persistent，只在本轮记录候选；后续 `gen-autotune-config` 才生成 `tl.num_programs(0)` grid-stride loop并为该架构独立调参，Grid 上限由该 config 的 registers/shared-memory occupancy 得到。
5. atomic 顺序可能造成浮点微小差异，沿用用户原精度阈值并与普通二维基线实测比较。无稳定性能提升则保留原二维 Grid。
