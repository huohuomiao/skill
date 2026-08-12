# retiling 优化

分块分析与改写属于通用 Triton 逻辑；CUDA shared memory、寄存器、occupancy 和运行时约束读取 `.claude/skills/share/gpu/references/platform-rules.md`。

## 职责概述

该策略针对 Triton Kernel 当前的分块方案做修正：

1. 保证各个轴都被 `block_size` 覆盖且 `block_size` 有效参与张量构建
2. 归约轴必须指定 `block_size`，且要么在 kernel 内 loop tile，要么在编译资源允许时由 `@triton.heuristics` 设为该轴大小

## 步骤

### 步骤1：Kernel Info 提取

参考 `.claude/skills/mlu-triton-optimize/kernel-info/strategy.md`，从 triton kernel 中提取轴信息。

### 步骤2：`block_size` 全覆盖

查看从步骤1获取的轴信息中，若存在 `block_size` 为 `null` 的轴，则赋予该轴 `block_size`，并且使其通过 `tl.arange(0, BLOCK_SIZE)` 方式参与构建该维度上的张量。

详细参考 `references/template_parallel_retiling.md`。

**重要提示**：若 kernel 存在 persistent loop(即针对并行轴分块的多批次处理)，当引入新的并行轴分块，需保证 persistent loop 迭代参数的正确性。pesistent loop 迭代参数模板如下：

`单并行轴`：

```python
@triton.jit
def xxx_kernel(
    ...
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_blocks = (M + BLOCK_M -1) // BLOCK_M
    # persistent loop
    for flat_pid in range(pid, total_blocks, num_programs):
    ...
```

`多并行轴`：

```python
@triton.jit
def xxx_kernel(
    ...
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks_m = (M + BLOCK_M -1) // BLOCK_M
    num_blocks_n = (N + BLOCK_N -1) // BLOCK_N
    total_blocks = num_blocks_m * num_blocks_n
    # persistent loop
    for flat_pid in range(pid, total_blocks, num_programs):
    ...
```
### 步骤3：归约轴规范化

检查归约轴是否在 kernel 内被 loop tile，如果没有，使用 @triton.heuristics 固定该拆分块大小为轴的大小。

示例如下：

```python
@triton.jit
def xxx_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ...
):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    result = tl.min(x != 0, axis=1)
    ...
```

若 `xxx_kernel` 中的 `BLOCK_N` 为 reduce 轴，且未在 kernel 中被 loop tile，只有 `N` 已被证明是 Triton 接受的 2 的幂 block 长度且资源可编译时，才可固定 `BLOCK_N=N`；否则使用 `triton.next_power_of_2(N)` 或继续分块循环，并保留 `offsets < N` mask。满足条件时示例如下：

```python
@triton.heuristics({"BLOCK_N": lambda args: args["N"]})
@triton.jit
def xxx_kernel(
    x_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ...
):
    pid_m = tl.program_id(0)
    m_offset = pid_m * BLOCK_M
    m_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = tl.arange(0, BLOCK_N)
    x_index = m_idx[:, None] * stride_x0 + n_idx[None, :] * stride_x1
    mask_x = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    x = tl.load(x_ptr + x_index, mask=mask_x, other=0.0)
    result = tl.min(x != 0, axis=1)
    ...
```
**重要提示**：若 kernel 有 `@triton.autotune`，且 config 配置项中存在该 `block_size` 的配置，则将该 `block_size` 从 autotune config 配置项中移除。

### 步骤4：验证结果，输出代码

若上面对 kernel 有任何修改，则需运行修改后代码，声明如下：

- 若编译或运行显示寄存器/shared-memory 超限、spilling 或 occupancy 过低，优先尝试调小分块参数，并按 `.claude/skills/share/gpu/references/platform-rules.md` 联合评估 `num_warps`、`num_stages`
- 本优化策略不用保证性能，通过即可
- 优化后 kernel 必须通过精度测试，误差应在测试代码设定的阈值范围内，且与原始 kernel 数学等价
- 若精度不正确或执行报错，根据错误信息进行修复，最多尝试 3 次；如果 3 次修复后仍无法通过精度测试，则回退至原始代码并记录原始性能。调试过程中的中间文件保存在本地

以上没有问题，则输出修改后代码。
