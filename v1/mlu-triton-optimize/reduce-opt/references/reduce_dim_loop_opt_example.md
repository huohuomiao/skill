# 归约轴循环优化 - 详细示例

## 示例1：已有 heuristic 的合并

### 原始代码
```python
@triton.heuristics({"NUM_WARPS": lambda args: 4, "BLOCK_SIZE": lambda args: 128})
@triton.jit
def sum_kernel(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BLOCK_SIZE: tl.constexpr):
    output_idx = tl.program_id(0)
    acc = 0.0
    for start in range(0, reduce_size, BLOCK_SIZE):       # 归约轴循环
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < reduce_size
        x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, acc)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...
    sum_kernel[grid](x, out, reduce_size, stride_reduce, stride_out)
    return out

x = torch.randn(64, 8192, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

### 优化后代码
```python
# 合并原有 heuristic：保留 NUM_WARPS，覆盖 BLOCK_SIZE
@triton.heuristics({
    "NUM_WARPS": lambda args: 4,
    "BLOCK_SIZE": lambda args: triton.next_power_of_2(args["reduce_size"])
})
@triton.jit
def sum_kernel(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BLOCK_SIZE: tl.constexpr):
    output_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)                    # 一次性加载全轴
    mask = offsets < reduce_size
    x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, result)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...
    sum_kernel[grid](x, out, reduce_size, stride_reduce, stride_out)
    return out

x = torch.randn(64, 8192, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

### 示例说明：
1. 归约轴识别与分块参数提取：循环 `for start in range(0, reduce_size, BLOCK_SIZE)` 表明归约轴大小为 `reduce_size`，分块参数名为 `BLOCK_SIZE`。
2. 归约轴大小提取与阈值判断：测试数据中 `x.shape = (64, 8192)`，`dim=1`，则归约轴大小 `reduce_size = 8192 ≤ 16384 = MAX_REDUCE_DIM`，满足优化条件。
3. heuristics 合并：保留 `NUM_WARPS` 配置，将 `BLOCK_SIZE` 覆盖为 `lambda args: triton.next_power_of_2(args["reduce_size"])`，以覆盖完整归约轴并满足 block 长度约束。
4. Kernel 函数体优化：移除循环，`tl.arange(0, BLOCK_SIZE)` 在运行时生成全轴偏移，一次性加载并归约。

## 示例2：存在 @triton.autotune 时的处理

### 原始代码
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['reduce_size'],
)
@triton.jit
def sum_kernel_autotuned(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BLOCK_SIZE: tl.constexpr):
    output_idx = tl.program_id(0)
    acc = 0.0
    for start in range(0, reduce_size, BLOCK_SIZE):       # 归约轴循环
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < reduce_size
        x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, acc)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...                                                   # stride 计算等
    sum_kernel_autotuned[grid](x, out, reduce_size, stride_reduce, stride_out)
    return out

x = torch.randn(16, 4096, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

### 优化后代码
```python
# 优化 kernel：
# 1. autotune 配置中移除分块参数名 BLOCK_SIZE 的条目
# 2. 添加 heuristic 将 BLOCK_SIZE 设置为覆盖 reduce_size 的 2 的幂
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),   # 移除了 BLOCK_SIZE
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['reduce_size'],
)
@triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["reduce_size"])})
@triton.jit
def sum_kernel_autotuned(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BLOCK_SIZE: tl.constexpr):
    output_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)                    # 循环消除，向量化全轴
    mask = offsets < reduce_size
    x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, result)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...                                                   # stride 计算等
    sum_kernel_autotuned[grid](x, out, reduce_size, stride_reduce, stride_out)
    return out

x = torch.randn(16, 4096, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

### 示例说明：
1. 归约轴识别与分块参数名提取：循环 `for start in range(0, reduce_size, BLOCK_SIZE)` 表明归约轴大小为 `reduce_size`，分块参数名为 `BLOCK_SIZE`。
2. 归约轴大小提取与阈值判断：测试数据中 `x.shape = (16, 4096)`，`dim=1`，则归约轴大小 `reduce_size = 4096 ≤ 16384 = MAX_REDUCE_DIM`，触发优化。
3. 处理 `@triton.autotune` 装饰器：原始 kernel 的 autotune 配置中包含对 `BLOCK_SIZE` 的调优项；优化 kernel 的 autotune 配置中，从每个 `triton.Config` 的字典里移除 `BLOCK_SIZE` 条目（变为空字典 `{}`），仅保留其他可调参数（如 `num_warps`），保留 autotune 的其他参数（如 `key=['reduce_size']`）不变，装饰器顺序与原始一致（`@triton.autotune` 在上，`@triton.heuristics` 在下）。
4. heuristics 合并：新增 `@triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["reduce_size"])})`，将分块参数设置为覆盖归约轴的 2 的幂并保留 mask。
5. Kernel 函数体优化：消除 `for` 循环，直接向量化加载全轴并归约。

## 示例3：无规约轴循环，完全向量化
```python
@triton.jit
def sum_vectorized(x_ptr, out_ptr, reduce_size, stride_reduce, BLOCK_SIZE: tl.constexpr):
    output_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)                   # 一次性向量化全轴
    x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce)
    result = tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, result)
```

### 示例说明：
**无 `for` 循环**，直接使用 `tl.arange(0, BLOCK_SIZE)` 生成归约轴偏移，归约操作由 `tl.sum` 完成。已为最优向量化形式，**无需优化**。

## 示例4：循环仅用于遍历输出位置，而非归约轴

```python
@triton.jit
def sum_kernel(inp, out, M, N: tl.constexpr, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    for task_idx in range(pid, total_tasks, num_programs):  # 输出遍历循环，非归约轴循环
        ...
        n_offset = tl.arange(0, N)                        # 归约轴一次性向量化
        inp_vals = tl.load(inp + offset, mask=mask)
        sum_val = tl.sum(inp_vals, axis=1)                # 沿 N 轴归约
        tl.store(out + out_offset, sum_val, mask=out_mask)
```

### 示例说明：
1. 归约轴识别：`tl.sum(..., axis=1)` 表明归约轴是第二个维度，其大小由 `N` 表示，`n_offset = tl.arange(0, N)` 实现了一次性向量化加载。
2. 循环分析：外层循环 `for task_idx` 用于划分输出块（`M`、`K` 维度），`n_offset` 在每次迭代中固定不变，因此**不是归约轴循环**。
3. 优化结论：该 kernel 已消除归约轴循环，**无需优化**。

## 示例5：归约轴大小 > MAX_REDUCE_DIM，不触发优化

```python
@triton.jit
def sum_kernel_blocked(x_ptr, out_ptr, reduce_size, stride_reduce, stride_out, BK: tl.constexpr):
    output_idx = tl.program_id(0)
    acc = 0.0
    for start in range(0, reduce_size, BK):               # 归约轴循环
        offsets = start + tl.arange(0, BK)
        mask = offsets < reduce_size
        x = tl.load(x_ptr + output_idx * stride_out + offsets * stride_reduce, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    tl.store(out_ptr + output_idx, acc)

def wrapper(x, dim):
    reduce_size = x.shape[dim]
    ...                                                   # stride 计算等
    sum_kernel_blocked[grid](x, out, reduce_size, stride_reduce, stride_out, BK=128)
    return out

# 测试数据：输入形状 (32, 32768)，dim=1，归约轴大小 = 32768 (> MAX_REDUCE_DIM)
x = torch.randn(32, 32768, device='cuda', dtype=torch.float32)
out = wrapper(x, dim=1)
```

### 示例说明：
1. 归约轴大小提取：从测试数据规格可知输入形状 `x.shape = (32, 32768)`，`dim=1`，则归约轴大小 `reduce_size = 32768`。
2. 阈值比较：`32768 > 16384 = MAX_REDUCE_DIM`，不满足优化条件。
