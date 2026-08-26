## 情况 D 扩展：规约操作（Reduction）

**场景**：多维 grid 用于规约操作，多个 program 处理同一输出位置的不同数据块，需要原子操作合并结果。

**初始代码**：
```python
import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_reduce(
    a_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # 计算该块的行和
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n in range(n_start, min(n_start + BLOCK_N, N)):
        for m in range(m_start, min(m_start + BLOCK_M, M)):
            val = tl.load(a_ptr + m * N + n)
            row_sum[m - m_start] += val

    # 写回输出（多个 program 可能写同一行）
    for m in range(m_start, min(m_start + BLOCK_M, M)):
        tl.store(out_ptr + m, row_sum[m - m_start])

def run():
    M, N = 2048, 4096
    BLOCK_M, BLOCK_N = 128, 256

    a = torch.randn(M * N, device='mlu')
    out = torch.zeros(M, device='mlu')

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    row_sum_reduce[grid](a , out , M, N, BLOCK_M, BLOCK_N)
```

## 优化分析步骤 (Modified Analysis)

**第一步：提取原始 Grid 结构**
1.  **定位调用**：找到 `@triton.jit` 装饰的 `row_sum_reduce` 和调用语句 `row_sum_reduce[grid](...)`。
2.  **提取表达式**：`grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))`。
3.  **识别形式**：变量引用。
4.  **拆解维度**：多维度 `dims = ["triton.cdiv(M, BLOCK_M)", "triton.cdiv(N, BLOCK_N)"]`，`has_lambda = False`。
**第二步：判断是否需要优化 (Decision)**
1.  **硬件接口**：代码中未调用 `driver.BangUtils()`，未计算核心总数。
2.  **Grid 约束**：Grid 为多维且未包含 `min` 约束。
* **结论**：**需要执行优化**。

**第三步：生成推荐 Grid 表达式**
1.  **判定情况**：属于 **情况 D（多维度规约场景）**。
2.  **计算总量**：`total_blocks = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)`。
3.  **应用约束**：引入 `MAX_GRID_SIZE`（基于 `TOTAL_CORE_NUM // num_warps`）。
4.  **推荐表达式**：`grid = lambda meta: (min(total_blocks, MAX_GRID_SIZE),)`。

**第四步：改写 Kernel 代码**

**4.1 Wrapper 函数改写**
- 导入必要的包 `from triton.backends.mlu import driver `
- 获取硬件核心数：`_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())`
    `TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")`
- 计算原始二维 grid 的维度：
  - `blocks_m = (M + BLOCK_M - 1) // BLOCK_M`
  - `blocks_n = (N + BLOCK_N - 1) // BLOCK_N`
- 计算总块数：`total_blocks = blocks_m * blocks_n`
- 根据`num_warps` 计算`MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps`
- 替换 grid 表达式：`grid_size = min(total_blocks, MAX_GRID_SIZE)`
- 将 `total_blocks` 作为参数传入 kernel

**4.2 Kernel 函数体改写**
- 获取一维 program ID：`pid = tl.program_id(0)`
- 获取总 program 数：`num_programs = tl.num_programs(0)`
- 预计算块数维度：`blocks_m = (M + BLOCK_M - 1) // BLOCK_M; blocks_n = (N + BLOCK_N - 1) // BLOCK_N`
- 引入步长 for 循环：`for flat_pid in range(pid, total_blocks, num_programs)`
- **多维索引还原**（关键步骤）：
  - `pid_n = flat_pid % blocks_n`
  - `pid_m = flat_pid // blocks_n`
- 计算块的起始位置：`m_start = pid_m * BLOCK_M; n_start = pid_n * BLOCK_N`
- 循环内保持原有的块处理逻辑

**4.3 规约场景特殊处理**
- 这是规约操作，多个 program 可能处理同一输出位置的不同数据块
- 在 for 循环**之前**初始化累加器（已在循环内初始化）
- 在 for 循环**内部**累积部分结果
- 在 for 循环**之后**使用**原子操作**写回：`tl.atomic_add(out_ptr + m, row_sum[m - m_start])`
- Wrapper 函数中输出 tensor 需要**零初始化**：`out = torch.zeros(M, device='mlu')`


## 修改后的完整代码

```python
import torch
import triton
import triton.language as tl
from triton.backends.mlu import driver

@triton.jit
def row_sum_reduce(
    a_ptr, out_ptr,
    M, N, # 改为普通参数
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    total_blocks: tl.constexpr,
):
    # 4.2.1 获取一维 program ID 和总 program 数
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    # 4.2.4 内部使用整数整除替代 triton.cdiv
    blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    # 4.2.2 引入 Persistent 步长循环
    for flat_pid in range(pid, total_blocks, num_jobs):
        # 4.2.3 多维索引还原 (还原为逻辑上的二维 grid)
        pid_n = flat_pid % blocks_n
        pid_m = flat_pid // blocks_n
        m_start = pid_m * BLOCK_M
        n_start = pid_n * BLOCK_N
        # 4.4 规约操作：循环内部累积块结果
        # 注意：此处为简便保持了原示例的标量循环，实际可进一步向量化
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        # 边界处理：使用 min 确保不越界
        current_m_limit = min(m_start + BLOCK_M, M)
        current_n_limit = min(n_start + BLOCK_N, N)
        for n in range(n_start, current_n_limit):
            for m in range(m_start, current_m_limit):
                val = tl.load(a_ptr + m * N + n)
                row_sum[m - m_start] += val
        # 4.4 核心优化：使用原子操作合并不同 program 产生的同一行的局部和
        for m in range(m_start, current_m_limit):
            tl.atomic_add(out_ptr + m, row_sum[m - m_start])

def run():
    M, N = 2048, 4096
    BLOCK_M, BLOCK_N = 128, 256
    num_warps = 1 # 默认为 1
    a = torch.randn(M * N, device='mlu')
    # 4.4 必须零初始化，原子加法依赖初始值为 0
    out = torch.zeros(M, device='mlu')
    # 4.1.1 插入核心数获取逻辑 (针对寒武纪 MLU)
    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")
    # 4.1.2 引入 Union 架构约束
    MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps
    # 计算总任务块数
    blocks_m = (M + BLOCK_M - 1) // BLOCK_M
    blocks_n = (N + BLOCK_N - 1) // BLOCK_N
    total_blocks = blocks_m * blocks_n
    # 4.1.3 推荐 Grid 表达式：Lambda 格式 + 核心数上限限制
    grid = lambda meta: (min(total_blocks, MAX_GRID_SIZE),)
    # Launch Kernel
    row_sum_reduce[grid](
        a, out,
        M, N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        total_blocks=total_blocks,
        num_warps=num_warps
    )
    return out
```
