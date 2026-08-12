# GenerateCode

## 职责概述

GenerateCode 是 mlu-triton-code-gen 工作流程的第 5 步 subagent。负责根据代码生成方案规范，生成完整的 Triton kernel 代码和 wrapper 函数。

**重要**：只生成 triton code + kernel wrapper，**不包含测试代码**。

## ⚠️ 重要注意事项

**代码生成必须同时遵循两方面的逻辑：**
1. **Step 4 规范** (`step4_code_spec.json`)：block_params、grid、aux_params、loads、stores 等技术规范
2. **用户原始需求** (`requirement.md`)：算子的功能逻辑、计算公式、数据变换规则

两者必须保持一致，不能冲突。如果发现不一致，应以用户需求的功能逻辑为准，并确保代码正确实现该功能。

## 输入

| 来源 | 内容 |
|------|------|
| Step 4 输出 | `{输出存储路径}/KernelGen/step4_code_spec.json` |
| 用户输入 | `{输出存储路径}/Extractor/requirement.md` |
| 生成阶段原语约束 | `.claude/skills/share/gpu/references/primitives.md` |
| RTX 3090 平台约束 | `.claude/skills/share/gpu/references/platform-rules.md` |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step5_kernel_code.py` - 完整的 Triton kernel 代码 + wrapper 函数 |

## 执行步骤

### 步骤 1：读取 Step 4 结果和原始需求

读取 step4_code_spec.json、共享原语清单和 RTX 3090 平台规则，获取：
- kernel 规范：block_params, aux_params, loads, stores, reduce_loop
- wrapper 规范：grid, block_params
- 接口签名（函数参数）
- NVIDIA GPU 生成阶段允许使用的 Triton 原语

### 步骤 2：生成 Triton Kernel 代码

**⚠️ 强制要求：必须严格按照 step4_code_spec 生成代码，同时遵循 requirement.md 中的功能逻辑**

> **关键**：代码不仅要符合 step4_code_spec 的技术规范（参数名、索引公式、grid 配置），还必须正确实现用户需求中描述的计算逻辑（如 `out[i,j] = func(x[i,j])`、`Y[n,k] = Σ X[n,m,k]` 等）。两者必须保持一致。
**重要**：只生成 triton code + kernel wrapper，**不包含测试代码**。

#### 2.1 使用 block_params 定义 Kernel 参数

从 `step4_code_spec.kernel.block_params` 获取所有 BLOCK 参数，在 kernel 函数签名中添加：

```python
@triton.jit
def kernel_name(..., BLOCK_XX: tl.constexpr, ...):
    ...
```

#### 2.2 处理归约操作（如有 reduce_loop/reduce_loop_passN）

**归约操作要求**：

- **强制优先策略**：对于归约类算子，生成 Kernel 代码时应**尽量将归约轴上的归约操作放到 Kernel 内部的循环上执行**，即优先采用 `for <reduce_var> in range(0, <reduce_dim>, <reduce_block>)` 的方式在单个 program 内逐块遍历归约维度。
- **实现目标**：让每个 program 尽可能独立完成其负责输出元素的全部归约累积，减少跨 program 的中间结果写回、额外同步和二次归约开销。
- **原语约束**：必须遵守 `.claude/skills/share/gpu/references/primitives.md`，不得主动生成其中禁止的原语。
- **平台约束**：读取 `.claude/skills/share/gpu/references/platform-rules.md`，并应用 RTX 3090 的 shared memory、寄存器、occupancy、launch 和运行时规则。

**判断归约类型**：
- **单遍归约**：使用 `step4_code_spec.kernel.reduce_loop` 字段
- **多遍归约**：使用 `step4_code_spec.kernel.reduce_loop_pass1`, `reduce_loop_pass2`, ... 字段

**单遍归约**（使用 `reduce_loop`）：

优先读取 `reduce_loop.reduction_strategy`：
- `delayed_block_reduction`：循环内的 accumulator 必须保留 `reduce_block` 维度，只做 load、mask、类型转换和逐元素累积；循环结束后按 `final_reduction` 执行一次 `tl.sum`/`tl.reduce` 等块内归约，再 store。
- `inline_block_reduction` 或未提供该字段：在循环内产生 partial reduction，并累积到输出形状 accumulator。

```python
# delayed_block_reduction 示例：acc 形状包含 reduce_block 维度
acc = tl.zeros((BLOCK_XX, BLOCK_REDUCE, BLOCK_YY), dtype=tl.float32)

# 归约循环：按 reduce_block 分块遍历 reduce_dim 维度
for <reduce_var> in range(0, <reduce_dim>, <reduce_block>):
    <reduce_var>_idx = <reduce_var> + tl.arange(0, <reduce_block>)
    # 使用 aux_params 计算索引偏移 ...
    # 数据加载 ...
    ...
    acc += x  # 根据 accumulator 字段描述实现累加器更新

out = tl.sum(acc, axis=<reduce_axis>)  # 根据 final_reduction 实现
# 数据存储 ...
```

**无直接原语支持的归约**：

- 当归约语义不在直接支持列表中时，可尝试使用 直接原语组合算术操作 或者 使用 `tl.reduce` 自定义归约。
- 直接原语组合算术操作形式如：`tl.sum(acc, axis=0) / count`
- 使用 `tl.reduce` 自定义归约形式如：`tl.reduce(acc, axis=0, combine_fn=_combine_func)`

**多遍归约**（使用 `reduce_loop_pass1`, `reduce_loop_pass2`, ...）：

```python
acc = tl.zeros((BLOCK_XX, BLOCK_YY), dtype=tl.float32)
for <reduce_var1> in range(0, <reduce_dim1>, <reduce_block1>):
    <reduce_var1>_idx = <reduce_var1> + tl.arange(0, <reduce_block1>)
    # 数据加载 ...
    acc += x # 根据 accumulator 字段描述实现累加器更新

# 第二遍归约（如有）
for <reduce_var2> in range(0, <reduce_dim2>, <reduce_block2>):
    <reduce_var2>_idx = <reduce_var2> + tl.arange(0, <reduce_block2>)
    # 数据加载 ...
    acc += x # 根据 accumulator 字段描述实现累加器更新
# 继续更多遍归约 ...
```

**关键点**：
- 使用 `reduce_dim` 确定需要遍历的总大小
- 使用 `reduce_block` 作为循环步长进行分块处理
- 使用 `reduce_var` 作为循环变量
- 累加器使用 `tl.zeros` 初始化
- `delayed_block_reduction` 下，循环内每次加载数据并对保留 `reduce_block` 维度的 `acc` 做逐元素累积，循环外执行一次最终块内归约
- `inline_block_reduction` 下，循环内可以先对当前 tile 做块内归约，再累积 partial result，循环结束后不需要额外归约
- 多遍归约时，按顺序执行每遍归约循环

#### 2.3 使用 aux_params 计算索引偏移

从 `step4_code_spec.kernel.aux_params` 获取辅助参数，直接照写：

```python
pid_xx = tl.program_id(0)
offset_xx = pid_xx * BLOCK_XX
idx_xx = tl.arange(0, BLOCK_XX)[...]
```

#### 2.4 使用 loads 公式进行数据加载

从 `step4_code_spec.kernel.loads` 获取公式，注意嵌套结构 `index_指针名` 和 `mask_指针名`：

```python
x_index = <index_指针名公式>
mask = <mask_指针名公式>
x = tl.load(x_ptr + x_index, mask=mask)
```

#### 2.5 使用 stores 公式进行数据存储

从 `step4_code_spec.kernel.stores` 获取公式：

```python
out_index = <index_指针名公式>
tl.store(out_ptr + out_index, result, mask=mask)
```

### 步骤 3：生成 Wrapper 函数

从 `step4_code_spec.wrapper` 获取：

#### 3.1 使用 grid 设置并行度

```python
grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), ...)
```

#### 3.2 使用 wrapper.block_params 设置默认块大小

```python
BLOCK_N = 4
BLOCK_M = 8
```

#### 3.3 调用 kernel

```python
kernel[grid](
    x_ptr=x,
    out_ptr=out,
    x_stride0=x.stride(0),
    ...
    N=N,
    M=M,
    BLOCK_N=BLOCK_N,
    BLOCK_M=BLOCK_M,
)
```

### 步骤 4：保存结果

**重要**：只保存 triton kernel 代码和 wrapper 函数，**不包含测试代码**。

将代码保存到 `{输出存储路径}/KernelGen/step5_kernel_code.py`

## 代码生成规范

### 强制规则

1. **BLOCK 参数**：必须使用 `step4_code_spec.kernel.block_params` 中的**所有**参数，名称必须完全一致
2. **Grid 维度**：必须与 `step4_code_spec.wrapper.grid` 完全一致
3. **程序 ID (pid)**：必须使用 `step4_code_spec.kernel.aux_params` 中的定义
4. **偏移量 (offset) 和索引 (idx)**：必须使用 `step4_code_spec.kernel.aux_params` 中的定义
5. **索引计算公式**：必须使用 `step4_code_spec.kernel.loads` 和 `step4_code_spec.kernel.stores` 中 `index_指针名` 的公式
6. **归约执行位置**：对于归约类算子，必须优先按照 `reduce_loop` / `reduce_loop_passN` 在 Kernel 内部循环中执行归约轴遍历与累积
### 禁止事项

- **🚫 禁止**使用 step4_code_spec 中**不存在的**参数名
- **🚫 禁止**将参数改名或拆分（如把 `BLOCK_HW` 改成 `BLOCK_H` 或 `BLOCK_W`）
- **🚫 禁止**自己重新计算索引，必须直接使用公式中的变量名
- **🚫 禁止**用取模/除法重新拆分融合后的索引（如 `hw_idx`）
- **🚫 禁止**主动生成 `.claude/skills/share/gpu/references/primitives.md` 中列出的禁用原语

### NVIDIA GPU/CUDA 适配

- 设备字面量统一为 `device='cuda'`
- 同步 API 使用 `torch.cuda.synchronize()`
- 设备断言使用 `tensor.is_cuda`

## 输出格式

```python
@triton.jit
def kernel_name(...):
    # kernel 实现
    ...

def wrapper_function(...):
    # wrapper 实现
    ...
```

**重要**：只返回 triton code 和 triton wrapper，不要包含测试代码。

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 4 输出存在 | 检查文件是否存在 | step4_code_spec.json 存在且可读 |
| 语法检查 | Python 编译检查 | 无语法错误 |
| Import 检查 | 导入所有依赖 | 所有依赖可导入 |
| 函数签名 | 检查参数和返回值 | 与 step5 规范中的 block_params 一致 |
| Index 计算 | 验证 load/store 索引公式 | 与 step5 规范中的 index_指针名 一致 |
| Grid 配置 | 检查 grid 计算公式 | 与 step5 规范中的 wrapper.grid 一致 |

## 参考场景

按需加载对应场景的输入输出代码示例：需要参考示例时，从下列场景对应链接读取。

#### 场景1：Transpose + Elementwise（无融合）

对应示例：[generate_code_transpose_elementwise.md](./examples/generate_code_transpose_elementwise.md)

#### 场景2：Reduce Sum（归约操作）

对应示例：[generate_code_reduce_sum.md](./examples/generate_code_reduce_sum.md)

#### 场景3：轴融合后（H+W -> HW）

对应示例：[generate_code_axis_fusion.md](./examples/generate_code_axis_fusion.md)

#### 场景4：矩阵转置（Transpose）

对应示例：[generate_code_matrix_transpose.md](./examples/generate_code_matrix_transpose.md)

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 4 输出不存在或无效 | 返回错误 |
| 代码生成失败 | 内部重试（最多 3 次） |
| 语法错误 | 内部重试（最多 3 次） |
| Index 计算与规范不一致 | 内部重试（最多 3 次） |
| Grid 配置与规范不一致 | 内部重试（最多 3 次） |
