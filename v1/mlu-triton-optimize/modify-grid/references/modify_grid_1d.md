## 情况 B：普通一维 Grid

**场景**：原始 Grid 已是一维，例如 `grid=(triton.cdiv(N, BLOCK_SIZE),)`。

## 初始代码

```python
@triton.jit
def vector_scale(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask)
    tl.store(b_ptr + offsets, a * 2.0, mask=mask)

def run():
    N = 65536
    BLOCK_SIZE = 256
    a = torch.randn(N, device="cuda")
    b = torch.empty_like(a)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_scale[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return b
```

## 判断

1. `logical_grid = cdiv(N, BLOCK_SIZE)`，每个 PID 只处理对应块。
2. Grid 覆盖所有逻辑块；这是 CUDA 上正确的普通 launch。
3. kernel 没有 grid-stride loop，因此给 Grid 添加 `min(..., sm_count)` 会直接漏算。

**默认结论**：无需修改。单维 Grid 不等于“缺少硬件约束”，也不需要限制到 RTX 3090 的 SM 数。

## 可选 Persistent 候选

只有主策略 Step 3 的全部准入条件通过时，才另建候选：

```python
@triton.jit
def vector_scale_persistent(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    total_blocks = tl.cdiv(N, BLOCK_SIZE)
    for block_id in range(pid, total_blocks, programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        a = tl.load(a_ptr + offsets, mask=mask)
        tl.store(b_ptr + offsets, a * 2.0, mask=mask)
```

以下 persistent 代码仅是交给后续 `gen-autotune-config` 的候选模板；当前 `modify-grid` 轮不得写入输出。后续为本家族独立选 config 并编译，从 `share/gpu` 的 NCU 分析结果取得 `active_blocks_per_sm`，然后动态读取 `sm_count`：

```python
props = torch.cuda.get_device_properties(torch.cuda.current_device())
resident_programs = props.multi_processor_count * active_blocks_per_sm
total_blocks = triton.cdiv(N, BLOCK_SIZE)
grid = (min(total_blocks, resident_programs),)
vector_scale_persistent[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
```

`active_blocks_per_sm` 不是常量猜测，必须由该已编译 config 的 registers、shared memory 和 threads 联合约束得出。普通版本与 persistent 版本做同输入 A/B；无稳定提升就保留初始普通 Grid。
