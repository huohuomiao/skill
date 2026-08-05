# GenerateSpec

## 职责概述

GenerateSpec 是 mlu-triton-code-gen 工作流程的第 4 步 subagent。负责根据前一步的轴融合结果，生成推荐的标准 Triton Kernel 代码规范。

## 输入

| 来源 | 内容 |
|------|------|
| Step 3 输出 | `{输出存储路径}/KernelGen/step3_axis_fusion.json` |
| 生成阶段原语约束 | `.claude/skills/mlu-triton-code-gen/ref/mlu_supported_primitives.md` |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step4_code_spec.json` - 代码生成方案规范 |

## 执行步骤

### 步骤 1：读取 Step 3 结果和原始需求

读取 step3_axis_fusion.json 和 `ref/mlu_supported_primitives.md`，获取：
- compute_formula
- compute_note（description + torch_impl，用于生成 kernel.compute.note 并辅助理解计算逻辑）
- io_block_mapping（包含 block_name, axis_size, contiguity, reduce_dim —— 形状/连续性已凝结于此，直接使用）
- fusion_note（如有）
- 接口签名
- MLU 生成阶段允许使用的 Triton 原语

### 步骤 2：生成代码规范

使用 LLM 生成推荐的 Triton Kernel 代码规范：

**设计内容**：
- **kernel 部分**：
  - block_params：BLOCK 参数定义，值为各轴 size 数组
  - aux_params：辅助计算参数定义（如 pid_xx, offset, tl.arange，仅保留公共计算信息
  - loads：各指针的 index 和 mask 计算公式，
  - stores：各指针的 index 和 mask 计算公式，
  - reduce_loop/reduce_loop_passN：归约维度循环（如有），**详见下方拆分规则**

- **wrapper 部分**：
  - grid：Grid 计算公式，如 `triton.cdiv(N, BLOCK_N)`
  - block_params：BLOCK 参数默认值（整数形式）

**输出格式**：
```json
{
"compute_formula": "计算公式",
    "compute_note": {
        "description": "算子计算逻辑的自然语言描述",
        "torch_impl": "对应的 torch 参考实现"
    },
    "fusion_note": "融合说明（如有）",
    "kernel": {
        "block_params": {"BLOCK_XX": [维度大小数组], ...},
        "aux_params": {"变量名": "计算公式", ...},
        "loads": {
            "指针名1": {
                "index_指针名1": "index计算公式（包含扩维操作如[:, None]或[None, :]）",
                "mask_指针名1": "mask计算公式（可选，用于边界检查）"
            },
            ...
        },
        "stores": {
            "指针名2": {
                "index_指针名2": "index计算公式（包含扩维操作）",
                "mask_指针名2": "mask计算公式（可选）"
            },
            ...
        },
        // 单遍归约时使用（一般情况）：
        "reduce_loop": {
            "reduce_dim": "归约维度名",
            "reduce_var": "循环变量名",
            "reduce_block": "BLOCK_XX",
            "accumulator": "累加方式",
            "reduction_strategy": "inline_block_reduction|delayed_block_reduction",
            "accumulator_shape": "累加器形状说明",
            "final_reduction": "循环结束后的最终块内归约公式（仅 delayed_block_reduction 必填）"
        },
        // 多遍归约时使用（按需添加），内部字段必须与"reduce_loop"相同：
        // "reduce_loop_pass1": { ... },
        // "reduce_loop_pass2": { ... },
        "compute": {
            "formula": "核心计算公式",
            "note": "计算逻辑说明"
        }
    },
    "wrapper": {
        "grid": "grid计算公式",
        "block_params": {"BLOCK_XX": 默认块大小, ...}
    }
}
```

⚠️ **字段说明**：

输出必须严格按照以下字段，结合以上JSON代码格式，不得额外添加或删除字段
| 字段 | 必填 | 说明 |
|------|------|------|
| `compute_formula` | ✅ 必填 | 原始计算公式，如 `Y[n,k] = Σ X[n,m,k]` |
| `compute_note` | ✅ 必填 | 从 step1 透传的算子语义，包含 `description`（自然语言描述）和 `torch_impl`（torch 参考实现） |
| `fusion_note` | 可选 | 融合说明，如 "融合: H + W -> HW"，无融合则不填 |
| `kernel.block_params` | ✅ 必填 | BLOCK 参数定义，值为各轴 size 数组（列表形式） |
| `kernel.aux_params` | ✅ 必填 | 辅助计算参数定义（如 pid_xx, offset, tl.arange） |
| `kernel.loads` | ✅ 必填 | 各输入指针的 index 和 mask 计算公式 |
| `kernel.stores` | ✅ 必填 | 各输出指针的 index 和 mask 计算公式 |
| `kernel.reduce_loop` | 仅归约时 | **单遍归约**：归约维度循环信息，包含 reduce_dim, reduce_var, reduce_block, accumulator；归约实现策略字段 `reduction_strategy` 必须填写，详见“归约实现策略选择规则”<br/>**多遍归约**：字段名称采用 `kernel.reduce_loop_pass1`, `kernel.reduce_loop_pass2`, ... 分别描述每遍归约的信息<br/>**无归约操作时**：**不包含此字段** |
| `kernel.compute` | ✅ 必填 | 核心计算逻辑，包含 formula 和 note 两个子字段 |
| `wrapper.grid` | ✅ 必填 | Grid 计算公式 |
| `wrapper.block_params` | ✅ 必填 | BLOCK 参数默认值（整数形式） |

**关键规则**：
- **必须遵守** `.claude/skills/mlu-triton-code-gen/ref/mlu_supported_primitives.md` 中的生成阶段原语约束；不得在 spec 中推荐禁止生成的原语。
- **aux_params** 只保留公共计算信息，如：
  - `pid_xx = tl.program_id(x)`
  - `xx_offset = pid_xx * BLOCK_XX`
  - `xx_idx = tl.arange(0, BLOCK_XX)`
- **index_指针名** 中包含具体的扩维操作，每个指针可以有不同的扩维逻辑
- **mask_指针名** 用于边界检查，每个指针可以有独立的 mask 计算逻辑
- **多遍归约判断**：当算子的计算公式`compute`中解析到**需要多次遍历归约轴**时，使用多遍归约。此时使用 `reduce_loop_pass1`, `reduce_loop_pass2`, ... 字段替换 `reduce_loop` 字段，分别描述每遍归约的信息
- **归约轴放置原则**：对于归约类算子，`reduce_dim` 对应的工作应优先编码为 Kernel 内部的 `reduce_loop` / `reduce_loop_passN` 循环；尽量不要把归约轴直接映射为额外的 grid/program 并行维度

**归约实现策略选择规则**：
对每个 `reduce_loop` / `reduce_loop_passN`，必须先判断归约应采用哪种实现策略，并在字段中明确表达。

`reduce_loop` / `reduce_loop_passN` 子字段要求：

| 子字段 | 必填条件 | 说明 |
|--------|----------|------|
| `reduce_dim` | 必填 | 被遍历的归约维度名 |
| `reduce_var` | 必填 | 归约循环变量名 |
| `reduce_block` | 必填 | 归约维度的块大小参数 |
| `accumulator` | 必填 | 循环内累加器更新方式 |
| `reduction_strategy` | 必填 | `delayed_block_reduction` 或 `inline_block_reduction` |
| `accumulator_shape` | `delayed_block_reduction` 时必填 | 说明 accumulator 保留 reduce_block 维度后的形状 |
| `final_reduction` | `delayed_block_reduction` 时必填 | 循环外最终块内归约公式 |

| 策略 | 适用场景 | 生成规范要求 |
|------|----------|--------------|
| `delayed_block_reduction` | 简单可结合归约（sum/prod/mean），循环遍历 reduce_dim 的多个 block，每次 load 得到仍包含 `reduce_block` 维度的 tile；循环内部不依赖已经归约成输出形状的结果 | **优先选择**。循环内只做 load、mask、类型转换和 elementwise 累积，累加器保留 `reduce_block` 维度；循环结束后只执行一次 `tl.sum`/`tl.reduce` 等块内归约得到输出形状，再 store |
| `inline_block_reduction` | 每次循环内必须立即得到该 reduce block 的归约值，或后续计算依赖每个 chunk 的归约结果，或保留 reduce_block 维度会导致累加器过大/NRAM 风险 | 在循环内执行 `tl.sum`/`tl.max`/`tl.min` 等块内归约，并将 partial result 累积到输出形状的 accumulator |

**MLU 归约原语选择**：
- 查看 `ref/mlu_supported_primitives.md` 优先使用对应直接原语。
- 如果没有直接原语支持的归约，可尝试使用 直接原语组合算术操作 或者 使用 `tl.reduce` 自定义归约。
- 使用直接原语组合算术操作时，示例：mean 操作可写为 `tl.sum(...) / count`。
- 使用 `tl.reduce` 自定义归约时候，需要定义 combine 函数，combine 内只能使用允许的算术、比较、逻辑和 `tl.where`。例如：将 `final_reduction` 写为 `tl.reduce(acc, axis=..., combine_fn=combine函数)`，并在 `compute.note` 或 `reduce_loop.final_reduction` 中说明需要生成 `@triton.jit 乘法 combine 函数。

**默认偏好**：
- 根据归约计算方式判断如 `sum(dim=...)` 等归约操作是否可以在 **Kernel 内部循环中保留 reduce_block 维度做逐元素累积，并把最终块内归约放到循环结束后**，优先生成 `delayed_block_reduction`。
- 选择 `delayed_block_reduction` 时， `accumulator` 应描述为类似“保持 reduce_block 维度的逐元素累积”的形式，例如 `acc += x`；不要写成单次迭代规约的形式 `acc += tl.sum(x, axis=...)
`。
- `delayed_block_reduction` 必须补充：
  - `accumulator_shape`：说明 accumulator 形状中包含 reduce_block 维度，例如 `(BLOCK_N, BLOCK_M, BLOCK_K)` 或 `(BLOCK_M, BLOCK_K)`。
  - `final_reduction`：说明循环外最终归约，例如 `out = tl.sum(acc, axis=block_reduce_axis)`。
- 除非确有资源限制、算法依赖每个 chunk 的 partial reduction，或上游逻辑已明确要求多遍归约，否则不建议设计成“多个 program 分别处理归约轴片段，再通过 Kernel 外部或附加阶段合并”的方案。

**归约循环字段拆分规则**：
根据 `reduce_loop.accumulator` 的复杂程度决定是否需要拆分为多遍归约：

| accumulator 复杂度 | 示例 | 处理方式 |
|-------------------|------|---------|
| **简单累加** | `acc += x`, `acc = acc * x` | 使用单 `reduce_loop` |
| **多步归约** | 归约公式包含多个不同的归约操作（如 `softmax` 需要 `exp`、`sum`、`div`） | 必须拆分，每个独立操作一个 `reduce_loop_passN` |

**拆分原则**：
- 当 `accumulator` 包含多个不同操作（如乘法后再加法、调用函数等），难以在单次循环内完成时，拆分为多遍归约
- 多遍归约时，使用 `reduce_loop_pass1`, `reduce_loop_pass2`, ... 分别描述每遍归约的信息
- 拆分后的每遍归约应该只包含单一类型的归约操作

### 步骤 3：保存结果

将分析结果保存到 `{输出存储路径}/KernelGen/step4_code_spec.json`

## 参考场景

按需加载对应场景的输入输出格式示例：需要参考示例时，从下列场景对应链接读取。

#### 场景1：Reduce Sum

对应示例：[generate_spec_reduce_sum.md](./examples/generate_spec_reduce_sum.md)

#### 场景2：Transpose + Elementwise Add（无融合）

对应示例：[generate_spec_trans_add.md](./examples/generate_spec_trans_add.md)

#### 场景3：轴融合后（H+W -> HW）

对应示例：[generate_spec_axis_fusion.md](./examples/generate_spec_axis_fusion.md)

#### 场景4：矩阵转置（Transpose）

对应示例：[generate_spec_matrix_trans.md](./examples/generate_spec_matrix_trans.md)

**说明**：
- 当计算公式包含转置操作（如 `X.T[m,n]`）时，需要在 `compute` 字段中指定 `formula: "tl.trans(x)"`
- 转置场景下，load 时使用转置前的索引顺序（n 行 m 列）
- 转置场景下的 mask 计算需要注意维度的对应关系

## 核心规则

### Grid 规范

⚠️ **Grid 个数必须与 Output 的拆分轴个数完全一致**
- Grid 的维度数量 = Output 张量的拆分轴数量（从 `io_block_mapping` 中 output ptr 的 `block_name` 键值对个数确定）
- 每个 grid 维度对应一个拆分轴，使用 `triton.cdiv(轴大小, BLOCK_轴)` 计算
- **写法是固定的**：`(triton.cdiv(轴大小1, BLOCK_轴1), triton.cdiv(轴大小2, BLOCK_轴2), ...)`

⚠️ **Grid 顺序必须与 pid 获取顺序一致**
- Grid 第 0 维对应 `tl.program_id(0)`，第 1 维对应 `tl.program_id(1)`

### Block 参数规范

- `kernel.block_params`：值为**列表形式**，对应各测试用例的轴大小
- `wrapper.block_params`：值为**整数形式**，推荐使用的默认块大小

### 融合轴处理

当输入包含融合轴（如 `HW`, `NM`）时：
- block_params 使用融合后的轴名和融合后的维度值
- grid 维度基于融合后的轴数量
- aux_params 使用融合后的变量名（如 `hw_idx`）
- **重要**：在 loads/stores 公式中，融合后的索引使用**原始 stride 的最后一维**

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 3 输出存在 | 检查文件是否存在 | step3_axis_fusion.json 存在且可读 |
| compute_formula 存在 | 解析 JSON 格式 | 包含 compute_formula 字段 |
| compute_note 透传 | 解析 JSON 格式 | 包含 compute_note.description 和 compute_note.torch_impl |
| kernel 规范完整 | 检查 kernel 字段 | 包含 block_params, aux_params, loads, stores |
| loads/stores 结构正确 | 检查 loads/stores 字段 | 每个指针包含 index_指针名 和 mask_指针名 字段 |
| wrapper 规范完整 | 检查 wrapper 字段 | 包含 grid, block_params |
| Grid 维度正确 | 检查 grid 公式 | Grid 维度数量 = 输出张量的拆分轴数量 |

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 3 输出不存在或无效 | 返回错误 |
| 输出 JSON 格式无效 | 内部重试（最多 3 次） |
| kernel 规范缺失关键字段 | 内部重试（最多 3 次） |
| wrapper 规范缺失关键字段 | 内部重试（最多 3 次） |
| Grid 维度与输出轴不匹配 | 内部重试（最多 3 次） |
