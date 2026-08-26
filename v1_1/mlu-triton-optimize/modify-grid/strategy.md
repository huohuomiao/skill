# Triton Kernel Grid 优化 (grid_modify_optimization)

Grid 展平与持久化循环属于通用 Triton 变换；MLU 核心数查询、Grid 上限和设备替换统一读取 `.claude/skills/share/mlu/references/platform-rules.md`。

## 职责概述

将 Triton kernel 的 grid 改写为 一维、不超过硬件物理核心数 的形式，原来由多维 grid 并行承载的任务改由 kernel 内部 for 循环分批处理。

## 步骤

### Step 1：提取原始 Grid 结构

需要从输入代码中找到 kernel 的调用位置，读取其 grid 表达式。具体做法：

#### 1.1 定位 kernel 定义与调用

1. 找到所有被 `@triton.jit` 装饰的函数，记录函数名。
2. 在代码中搜索形如 `kernel_name[grid_expr](...)` 的调用语句。方括号 `[...]` 内的内容就是 grid 表达式。
3. 若存在多个 kernel 调用，取**第一个**。
4. 若只有 kernel 定义、没有调用语句，则视为**无法提取 grid**，直接进入第三步使用默认值。

#### 1.2 识别 Grid 的写法形式

Grid 存在以下几种常见写法，需要逐一识别：

| 写法形式 | 特征 | 提取方式 |
|---|---|---|
| **内联元组** | `kernel[(128, 1, 1)](...)` | 直接从方括号内提取元组 |
| **变量引用** | `kernel[grid](...)` | 回溯查找 `grid = ...` 的赋值语句，取右值 |
| **lambda 表达式** | `grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]),)` | 提取完整 lambda 表达式，标记 `has_lambda = True` |
| **复合表达式** | `grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))` | 提取完整元组 |

#### 1.3 拆解各维度

拿到 grid 表达式字符串后，将其拆解为各维度的子表达式列表：

- 如果是 lambda 形式：先提取 `lambda meta:` 后面的 body 部分（通常是一个元组），再拆解元组元素。
- 如果是普通元组：拆解时需要**正确处理嵌套括号**——按顶层逗号分割，忽略函数调用内部的逗号。例如 `(triton.cdiv(M, BM), N)` 应拆为 `["triton.cdiv(M, BM)", "N"]` 两个维度，而不是三个。
- 如果不是元组形式（单个表达式）：视为一维 grid，列表中只有一个元素。

最终得到：
- **dims**：维度表达式列表，如 `["triton.cdiv(M, BM)", "triton.cdiv(N, BN)"]`
- **has_lambda**：布尔值，标记原始是否为 lambda 格式

### Step 2 ：生成推荐 Grid 表达式

根据第二步提取到的维度信息，按照以下决策树生成推荐的一维 grid 表达式：

#### 情况 A：无法提取到 grid

接使用默认表达式:
```python
num_warps = 1 # 默认为1
_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")
MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps
```

**加载参考**：`references/modify_grid_without_grid.md`

#### 情况 B：所有维度都是常数 1

（如 `grid=(1,)` 或 `grid=(1,1,1)`）

这说明原始 kernel 在单个 program 中处理全部数据，未做任何并行拆分。此时需要引入 BLOCK_SIZE 来拆分工作：

1. 扫描 kernel 函数签名，查找所有 `参数名: tl.constexpr` 形式的参数。
2. 若找到了 constexpr 参数（假设为 `B, M, N`）：用它们的乘积作为总元素数，生成 `min(triton.cdiv(B * M * N, BLOCK_SIZE), MAX_GRID_SIZE)`。
3. 若没有 constexpr 参数：使用通用形式 `min(triton.cdiv(n_elements, BLOCK_SIZE), MAX_GRID_SIZE)`。

**加载参考**：`references/modify_grid_constexpr.md`
#### 情况 C：单维度 grid

直接对该维度表达式加上 core_num 上限：`min(该维度表达式, MAX_GRID_SIZE)`

**加载参考**：`references/modify_grid_1d.md`
#### 情况 D：多维度 grid

将所有维度表达式用 `*` 连接合并为一维总量，再取上限：`min(dim0 * dim1 * ... * dimN, MAX_GRID_SIZE)`

**加载参考**：`references/modify_grid_3d.md`
#### Lambda 保持

如果原始 grid 使用了 lambda 格式，最终推荐表达式需要包裹为 `lambda meta: (推荐表达式,)` 的形式。
**重要**：当原始 grid 是 lambda 形式时，需要：
1. 从 lambda 体中提取出实际的维度表达式（去掉外层元组括号）
2. 用该表达式替换 `min()` 的第一个参数
3. 最终形式：`grid = lambda meta: (min(原始维度表达式, MAX_GRID_SIZE),)`

例如：
- 原始：`grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)`
- 提取维度：`triton.cdiv(n_elements, meta['BLOCK_SIZE'])`
- 优化后：`grid = lambda meta: (min(triton.cdiv(n_elements, meta['BLOCK_SIZE']), MAX_GRID_SIZE),)`

### Step 3：改写Kernel 代码
这一步骤将代码 **Persistent Kernel** 化：将 Triton kernel 的 grid 改写为 **一维、不超过硬件物理核心数** 的形式，原来由多维 grid 并行承载的任务改由 kernel 内部 for 循环分批处理,从而最小化 Kernel 启动与上下文切换开销，最大化硬件计算资源利用率。

#### 3.1 Wrapper 函数改写

1.  **导入需要包**：
```python
from triton.backends.mlu import driver
```
2.  **插入核心数获取逻辑**：
```python
_devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")
```
3.  **引入 Union 架构约束**：
    * 提取代码中是否存在 `num_warps`， 如果不存在则引入`num_warps = 1`
    * 根据原代码`num_warps` 调整最大并行度：`MAX_GRID_SIZE = TOTAL_CORE_NUM // num_warps`,
4.  **替换 Grid**：
    * 将原始多维 Grid 替换为一维：`grid = lambda meta: (min(total_blocks, MAX_GRID_SIZE),)`。

#### 3.2 Kernel 函数体改写 — 通用转换规则

##### 3.2.1 获取一维 program ID 和总 program 数

原始代码中可能通过 `tl.program_id(0)`, `tl.program_id(1)`, `tl.program_id(2)` 分别获取各轴的 program ID。改写后 grid 只有一维，因此：
- 用 `pid = tl.program_id(0)` 获取唯一的一维 ID
- 用 `num_jobs = tl.num_programs(0)` 获取总 program 数
##### 3.2.2 引入 Persistent 步长循环

由于 grid 缩小了（最多 core_num 个 program），每个 program 需要处理多个数据块。引入 for 循环，步长为总 program 数：

- 循环范围：`for flat_pid in range(pid, total_blocks, num_jobs)`
- 其中 `total_blocks` 是原始 grid 所有维度的乘积，代表总任务数

##### 3.2.3 多维索引还原

如果原始是多维 grid（如 `grid=(A, B, C)`），在循环内部需要将一维线性 `flat_pid` 还原为多维索引。使用整除和取模运算：

- 对于原始三维 grid `(A, B, C)`：
  - 最内层维度 C 的索引：`c_i = flat_pid % C`
  - 中间维度 B 的索引：`b_i = (flat_pid // C) % B`
  - 最外层维度 A 的索引：`a_i = flat_pid // (B * C)`
- 其中 A、B、C 从 kernel 函数参数列表传入

还原出多维索引后，kernel 内部原有的地址计算和数据处理逻辑**保持不变**。

##### 3.2.4 整除运算替代 triton.cdiv

如果 kernel 内部需要计算 ceil_div（向上整除），**禁止**调用 `triton.cdiv`（它是 host 函数）。必须用纯整数运算替代：`(a + b - 1) // b`。例如原始的 `triton.cdiv(N, BLOCK_N)` 改为 `(N + BLOCK_N - 1) // BLOCK_N`。

#### 3.3 非规约场景的改写方式

适用于：elementwise 操作（如 add、relu、scale）、softmax 在非规约轴上的拆分等。

特点：循环内每次迭代处理一个独立的数据块，各 program、各迭代之间完全无依赖。

改写要点：
1. 将原始的单次数据块处理逻辑**整体包入** for 循环体内。
2. 用步长方式遍历：初始块位置 = `pid * BLOCK_SIZE`，步长 = `num_jobs * BLOCK_SIZE`。
3. 边界用 mask 处理：`mask = offsets < n_elements`。
4. **不需要**任何额外的边界检查 if 语句或原子操作。

#### 3.4 规约场景的改写方式
适用于：求和（sum）、求均值（mean）、求最大值（max）等规约操作发生在被拆分轴上的情况。

特点：多个 program 可能处理同一输出位置的不同数据块，必须正确合并部分结果。

改写要点：
1. 在 for 循环**之前**初始化累加器（如用 `tl.zeros` 创建局部累加变量）。
2. 在 for 循环**内部**累积每个数据块的部分结果到累加器。
3. 在 for 循环**之后**，对累加器做最终规约（如 `tl.sum`），然后使用**原子操作**（如 `tl.atomic_add`）将结果写回全局输出。原子操作保证多个 program 对同一输出位置并发写入时的正确性。
4. wrapper 函数中，输出 tensor 需要用**零初始化**（因为多个 program 会通过原子操作累加到同一位置）。

**加载参考**：`references/modify_grid_reduce.md`

#### 3.5 grid=(1,1,1) 单 program 场景的改写方式

当原始 grid 全为 1 时，kernel 在单个 program 内处理完整张量（通常使用 `tl.arange` 一次加载所有数据），没有任何并行。

改写要点：
1. 将原始维度参数从 `tl.constexpr` 改为普通 int 参数（因为它们是运行时的张量维度）。
2. 新增 `BLOCK_SIZE: tl.constexpr` 参数，用于向量化访问。
3. 将原始的全量加载改为**分块加载**：用 `tl.arange(0, BLOCK_SIZE)` 做向量化读写，配合 mask 处理边界。
4. 引入 for 循环，步长为 `num_jobs * BLOCK_SIZE`。
5. 将高维索引展平为线性索引，总元素数 = 各维度的乘积。

### Step 4：验证生成代码，输出最终结果

运行生成的代码，保证其能够正确执行，且精度测试通过。如果有错误，根据错误信息进行调试修改。输出最终代码。

#### 代码自检优化

- 识别kernel函数索引逻辑，将复杂冗余索引计算简化。（通常用于pointwise算子多维缩影）

## 硬性约束（优化改写时必须遵守）

### 禁止项

| 禁止内容 | 原因 |
|---|---|
| kernel 内部调用 `triton.cdiv`、`torch.*`、Python 标准库函数 | 这些是 host 侧函数，kernel 内部不可用 |
| kernel 循环内使用 `continue` 或 `return` | Triton 编译器不支持，会报 `unsupported AST node type: Continue` |
| 步长循环内添加边界检查 if 语句 | 循环条件已隐式保证，多余检查会引入无用分支 |
| 修改原始函数名称 | 必须保留用户原始的 kernel 和 wrapper 函数名 |
| 使用 `.repeat()` / `.tile()` 做 tensor 形状扩展 | Triton 不支持，应使用 `tl.broadcast_to` |
| launch 传参时禁止使用指针传递，即`.date_ptr()`| 会导致类型解析失败 (Type Mismatch)

### 类型约束

- `tl.zeros`、`tl.ones`、`tl.full` 的 shape 参数必须是**元组**，如 `(BLOCK_SIZE,)` 而非 `BLOCK_SIZE`，否则报 `TypeError: 'int' object is not iterable`。
- `tl.arange` 的参数必须是**编译时常量**（`tl.constexpr`），不能使用运行时变量。
- `tl.atomic_add` 等原子操作的指针类型和值类型必须匹配：标量指针对标量值，数组指针对数组值。

### 保守原则

如果无法确保改写后代码的正确性（如规约语义难以确认、索引关系复杂），**不进行任何修改**，直接返回原始代码。逻辑正确性始终优先于优化。
**注意** Wrapper 函数中launch kernel时不要指针传递，直接对象传递即可

###

### 硬件适配

所有输出代码中的 `cuda` 替换为 `mlu`，`CUDA` 替换为 `MLU`（适配寒武纪硬件）。
