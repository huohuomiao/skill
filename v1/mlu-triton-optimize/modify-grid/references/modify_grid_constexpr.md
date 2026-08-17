## 情况 D：Grid 全为 1，生成普通 tiled 基线

**场景**：原始 kernel 用单 program 处理完整张量。先生成全逻辑 Grid 的普通 CUDA tiled 版本；不要直接 persistent 化。

## 初始代码

```python
@triton.jit
def matrix_transpose(a_ptr, b_ptr, M: tl.constexpr, N: tl.constexpr):
    offsets = tl.arange(0, M * N)
    i = offsets // N
    j = offsets % N
    value = tl.load(a_ptr + offsets)
    tl.store(b_ptr + j * M + i, value)

def run():
    M, N = 256, 512
    a = torch.randn(M * N, device="cuda")
    b = torch.empty(N * M, device="cuda")
    matrix_transpose[(1,)](a, b, M, N)
    return b
```

完整张量 `tl.arange` 可能造成极高寄存器压力或超出 Triton tile 约束。正确的第一候选是普通分块并行。

## 修改后的普通 Grid

```python
@triton.jit
def matrix_transpose(a_ptr, b_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = M * N
    mask = offsets < total_elements
    value = tl.load(a_ptr + offsets, mask=mask)
    i = offsets // N
    j = offsets % N
    tl.store(b_ptr + j * M + i, value, mask=mask)

def run():
    M, N = 256, 512
    a = torch.randn(M * N, device="cuda")
    b = torch.empty(N * M, device="cuda")
    grid = lambda meta: (triton.cdiv(M * N, meta["BLOCK_SIZE"]),)
    matrix_transpose[grid](a, b, M, N, BLOCK_SIZE=256, num_warps=4)
    return b
```

## 要点

1. `M`、`N` 可保留 constexpr 或改为运行时参数，取决于原调用契约和编译缓存需求；不要无理由强制一种形式。
2. `grid` 启动全部 `cdiv(M*N, BLOCK_SIZE)` 个逻辑 program，不按 SM 数封顶。
3. BLOCK_SIZE 与 `num_warps` 通过编译资源和实测选优；RTX 3090 常见 warps 候选为 2/4/8。
4. 若普通 tiled 版本正确且 persistent 有合理收益，本轮只记录方案；后续 `gen-autotune-config` 为其独立调参、计算 occupancy，并用 `tl.num_programs(0)` 循环完整覆盖任务。
5. RTX 3090 不得使用 FP8、TMA、thread-block cluster 或 Hopper 专属转置路径。
