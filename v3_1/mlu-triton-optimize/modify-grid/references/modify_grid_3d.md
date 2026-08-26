# Triton Kernel Grid 优化 - 完整示例

本文档包含展示如何将 Triton kernel 的多维度 Grid，折叠维度相乘再加核心数上限，grid 改写为一维、不超过硬件物理核心数的形式。

## 初始代码
```python

import torch
import triton
import triton.language as tl

# 示例核函数：原三维 grid 写法
@triton.jit
def original_kernel(
    a_ptr, b_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)
    # 假设每个维度是块的数量
    offset = pid_n * M * K + pid_m * K + pid_k
    # 处理逻辑略（读取、计算、写回）
    ...
# Host 调用示例
def run():
    # 假设每维块数
    M, N, K = 8, 4, 16
    block_size = 128
    a = torch.randn( M* N * K * 128, device='mlu')  # device示例，改成实际支持设备
    b = torch.empty_like(a)
    grid = (M,N,K)
    original_kernel[grid](
        a , b ,
        M, N, K,block_size
    )

```

---

## 优化分析步骤 (Modified Analysis)

**第一步：提取原始 Grid 结构**
1.  **定位调用**：找到 `@triton.jit` 装饰的 `original_kernel` 和调用语句 `original_kernel[grid](...)`。
2.  **提取表达式**：回溯查找得到 `grid = (N, M, K)`。
3.  **识别形式**：变量引用。
4.  **拆解维度**：三维度 `dims = ["N", "M", "K"]`，`has_lambda = False`。

**第二步：判断是否需要优化 (Decision)**
1.  **硬件接口**：代码中未调用 `driver.BangUtils()` 获取硬件核心乘积。
2.  **Grid 约束**：Grid 为三维且未包含 `min` 函数与核心数限制。
* **结论**：两项未同时满足，**需要执行优化**。

**第三步：生成推荐 Grid 表达式**
1.  **判定情况**：属于 **情况 D（多维度 grid）**。
2.  **计算总量**：将所有维度相乘 `total_blocks = N * M * K`。
3.  **应用约束**：引入 `MAX_GRID_SIZE`（基于 `TOTAL_CORE_NUM // num_warps`）。
4.  **推荐表达式**：`grid = lambda meta: (min(total_blocks, MAX_GRID_SIZE),)`。
**第四步：执行 Persistent 优化 (Optimization)**

* **4.1 Wrapper 改写**：
    * 导入必要的包 `from triton.backends.mlu import driver`
    * 插入寒武纪特有核心数获取逻辑 `driver.BangUtils()`。
    * 引入 `num_warps` 并计算 `MAX_GRID_SIZE`。
    * 将各维度（N, M, K）及总块数 `total_blocks` 传给 Kernel。
* **4.2 Kernel 改写**：
    * **4.2.1/4.2.2**：获取一维 `pid` 和 `num_jobs`，建立 `for flat_pid in range(pid, total_blocks, num_jobs)` 循环。
    * **4.2.3 多维索引还原**：在循环内部将 `flat_pid` 还原为 `pid_n`, `pid_m`, `pid_k`。
    * **4.2.4**：内部逻辑保持不变。

---


## 修改后的代码

```python

import torch
import triton
import triton.language as tl
# 导入必要的包
from triton.backends.mlu import driver
# 改写后的 kernel，采用一维 grid 和循环遍历所有块
@triton.jit
def modified_kernel(
    a_ptr, b_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    total_blocks: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs()
    # 为处理更多数据块，步长循环
    for flat_pid in range(pid, total_blocks, num_programs):
        # 多维索引还原
        blocks_per_k = K
        blocks_per_m = M * K
        pid_n = flat_pid // blocks_per_m
        pid_m = (flat_pid % blocks_per_m) // blocks_per_k
        pid_k = flat_pid % blocks_per_k
        offset = pid_n * M * K + pid_m * K + pid_k
        # 处理逻辑略（如加载、计算、存储）
    ...
# Host 调用示例
def run():
    # 假设每维块数
    M, N, K = 8, 4, 16
    total_blocks = M * N * K

    a = torch.randn(total_blocks * 128, device='mlu')  # device示例，改成实际支持设备
    b = torch.empty_like(a)
    core_num = torch.mlu.get_device_properties().multi_processor_count
    grid_size = min(total_blocks, core_num)
    # 调用改写的 kernel，就只有一维 grid
    modified_kernel[(grid_size,)](
        a , b ,
        M, N, K, total_blocks
    )

```
