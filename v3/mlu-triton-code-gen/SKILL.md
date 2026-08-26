---
name: mlu-triton-code-gen
description: MLU Triton Kernel 代码生成专家，负责根据算子需求生成完整的 Triton kernel 实现。
---

# mlu-triton-code-gen

## 多阶段流程

| Stage | 名称              | 执行方式                             | 进入条件                                    |
| ----- | ----------------- | ------------------------------------ | ------------------------------------------- |
| 0     | 输入类型检查      | 直接读取 requirement.md              | START（起点）                               |
| 1     | ExtractBaseInfo   | 调度 Subagent                        | Stage 0 判断为非 Triton 输入                |
| 2     | TraceBlockMapping | 调度 Subagent                        | Stage 1 成功                                |
| 3     | AxisFusion        | 调度 Subagent                        | Stage 2 成功                                |
| 4     | GenerateSpec      | 调度 Subagent                        | Stage 3 成功                                |
| 5     | GenerateCode      | 调度 Subagent / 直接复制             | Stage 0 判断为 Triton 输入，或 Stage 4 成功 |
| 6     | GenTestCode       | 调度 Subagent                        | Stage 5 成功                                |
| 7     | 代码验证          | 调用 Skill（mlu-triton-code-review） | Stage 6 成功                                |
| 8     | 检查回退机制      | 条件跳转                             | Stage 7 返回验证失败                        |
| 9     | 输出结果          | 直接输出                             | Stage 7 返回验证通过                        |

### Stage 1 二态路由

| 检测结果           | 含义                        | 下一步                                    |
| ------------------ | --------------------------- | ----------------------------------------- |
| `is_triton: true`  | Triton 代码输入（快速路径） | 跳过 Stage 1-4，直接复制 original_code.py |
| `is_triton: false` | 需求描述输入（正常流程）    | 执行 Stage 1-4 后进入 Stage 5             |

### 快速路径说明
- 从 `requirement.md` 读取 "输入类型" 字段，若为 `triton` 则走快速路径
- 快速路径将跳过 Stage 1-4，直接使用 `{输出存储路径}/Extractor/original_code.py` 作为 Stage 5 的输出
- 后续流程（Stage 5及之后）与正常路径一致

## 概述

该 SKILL 是 MLU Triton Kernel 代码生成专家，负责根据算子需求生成符合 MLU 硬件架构的 Triton kernel 代码。

**核心目标**：根据输入的需求文档，生成可在 MLU 设备上正确执行的 Triton kernel 代码，包含**方案设计、代码生成、测试生成与首轮执行、最终验证**几个关键阶段。

## 运行环境选择规则

涉及真实运行、精度测试、性能测试时，必须遵守 `.claude/skills/mlu-triton-main/SKILL.md` 中的运行环境选择规则。

- 先以 EnvConfig 的环境检查结果为准：本地执行环境顺序执行 `get_device_info.py` 和 `test_env_code.py` 都成功时，优先在本地执行动态命令。
- 若本地执行环境任一环境检查脚本失败，则在当前 `JOB_ID` 下通过 `.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 提交 Worker Task，并在 Worker 上执行同一套环境检查。
- 每次 Worker Task 调用必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再进行下一步；禁止 `&` 后台、禁止并发提交多个 Worker Task。
- 禁止为了测试另建 Job；禁止手写 HTTP 请求绕过提交脚本。

**工作流程**：

1. **方案设计**：分析需求文档，确定拆分轴、块索引映射、轴融合优化，生成代码规范方案
2. **代码生成**：根据方案生成 Triton kernel 代码和 wrapper 函数
3. **测试代码生成**：生成完整的测试代码（包含数据生成、精度验证、性能测试），本阶段不执行
4. **执行与最终验证**：由 mlu-triton-code-review 统一执行测试并进行动态检查和修复，确保代码正确性和精度

## 用法

```bash
/mlu-triton-code-gen <requirement>
```

用户传递输入的参数`requirement`是需求文档文件路径

## 默认设置

**输出存储路径**默认设置为：`output_dir/`

生成失败时默认返回错误信息

## 工作流程

该 SKILL 的生成流程分为以下 **4 个阶段**：

### 阶段 0：输入类型检查（Step 0）

读取 `{输出存储路径}/Extractor/requirement.md`中的 "输入类型" 字段 。

- **Triton 输入（快速路径）**：跳过步骤 1-4，直接使用 `{输出存储路径}/Extractor/original_code.py`
- **非 Triton 输入**：继续执行步骤 1-4，按正常流程生成代码

### 阶段 1：方案设计（Steps 1-4）

解析需求文档，进行结构化信息与拆分轴一体化提取、块索引映射分析、轴融合优化，最终生成代码规范方案。

**注意**：仅在非 Triton 输入时执行。
### 阶段 2：代码生成（Step 5）

根据方案生成 Triton kernel 代码和 wrapper 函数，或直接使用 original_code.py。

### 阶段 3：测试代码生成（Step 6）

生成完整的测试代码，包含数据生成、精度验证和性能测试；本阶段不执行该文件，避免与 Step 7 重复运行。

### 阶段 4：最终代码验证（Step 7）

调用 mlu-triton-code-review 进行动态检查和修复。

---

**详细步骤**：

### 步骤 0：检查输入类型（快速路径）

在执行步骤 1-4 之前，读取需求文档中的输入类型字段，判断是否为 Triton 代码输入。

**读取位置**（按优先级）：

1. **方式一：从 requirement.md 读取**（推荐）
   - 读取 `{输出存储路径}/Extractor/requirement.md`
   - 查找 `## 输入类型` 或 `## Input Type` 字段后的值
   - 如果值为 `triton`，则表示输入为完整的 Triton kernel 代码

2. **方式二：从调用参数传入**
   - mlu-triton-main 主 Skill 在调用时可通过参数显式传入 `input_type` 字段

**判断条件**：
- 如果 `输入类型: triton` 或 `input_type: triton`，则表示输入为完整的 Triton kernel 代码

**快速路径处理**：

- **是 Triton 输入**：直接跳到步骤 5，读取 `{输出存储路径}/Extractor/original_code.py` 作为 GenerateCode 的输出，**跳过步骤 1-4**
- **非 Triton 输入**：继续执行步骤 1-4，按正常流程生成代码

### 步骤 1：解析需求文档（结构化信息 + 拆分轴一体化提取）

解析输入的需求文档文件路径，读取 `{输出存储路径}/Extractor/requirement.md` 获取算子需求信息，一次性提取结构化信息（含计算公式、compute_note、io_shapes 的 axis/shape/contiguity、reduce_axes）并保存到文件。

**注意**：只有当输入类型为非 Triton 时才执行此步骤。

| 步骤   | Subagent          | 说明                                       | 关键输出                                                                                            | 输出文件                                                                                                                 |
| ------ | ----------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Step 1 | `ExtractBaseInfo` | 从需求文档中一体化提取计算信息与拆分轴信息 | op_name, compute_type, compute_formula, compute_note, io_shapes(axis/shape/contiguity), reduce_axes | `{输出存储路径}/KernelGen/step1_base_info.json`<br/>`{输出存储路径}/KernelGen/step1_io_shapes.json`（仅 io_shapes 部分） |

#### 执行 Step 1 - ExtractBaseInfo

**重点要求**：分发给 subagent 执行 ExtractBaseInfo 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/ExtractBaseInfo.md` 下的**操作说明**，按照规范要求充当 ExtractBaseInfo 角色，完成 {算子名} 算子的结构化信息 + 拆分轴一体化提取任务。

## 用户输入
- 需求文档路径：{{requirement_path}}
- 存储路径：`{{output_dir}}/KernelGen`

## 重要约束
- **只能读取**以下文件：
  - 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/ExtractBaseInfo.md`
  - 需求文档：`{{requirement_path}}`
- **禁止读取**其他文件或其他目录下的文件
- 读取文件时必须使用绝对路径

请严格按照任务文档要求执行，输出必须包含 compute_note、axis、shape、contiguity、reduce_axes。
"""
)
```

### 步骤 2：追踪块索引映射

根据 Step 1 的输出，分析输入到输出的映射关系。

**注意**：只有当输入类型为非 Triton 时才执行此步骤。

| 步骤   | Subagent            | 说明                 | 关键输出                          | 输出文件                                            |
| ------ | ------------------- | -------------------- | --------------------------------- | --------------------------------------------------- |
| Step 2 | `TraceBlockMapping` | 追踪输入依赖与块索引 | compute_formula, io_block_mapping | `{输出存储路径}/KernelGen/step2_block_mapping.json` |

#### 执行 Step 2 - TraceBlockMapping

**重点要求**：分发给 subagent 执行 TraceBlockMapping 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/TraceBlockMapping.md` 下的**操作说明**，按照规范要求充当 TraceBlockMapping 角色，完成 {算子名} 算子的块索引追踪任务。

## 用户输入
- 存储路径：`{{output_dir}}/KernelGen`
- Step 1 输出：{{output_dir}}/KernelGen/step1_base_info.json（包含 compute_formula, compute_note, io_shapes(axis/shape/contiguity), reduce_axes）

## 重要约束
- **只能读取**以下文件：
  - 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/TraceBlockMapping.md`
  - Step 1 输出：`{{output_dir}}/KernelGen/step1_base_info.json`
- **禁止读取**其他文件（如原始需求文档、其他步骤的输出文件等）
- 读取文件时必须使用绝对路径

请严格按照任务文档要求执行。
"""
)
```

### 步骤 3：轴融合优化

根据 Step 2 的输出，判断拆分轴是否有可以融合的情况。

**注意**：只有当输入类型为非 Triton 时才执行此步骤。

| 步骤   | Subagent     | 说明           | 关键输出                      | 输出文件                                          |
| ------ | ------------ | -------------- | ----------------------------- | ------------------------------------------------- |
| Step 3 | `AxisFusion` | 轴融合优化方案 | fusion_note, io_block_mapping | `{输出存储路径}/KernelGen/step3_axis_fusion.json` |

#### 执行 Step 3 - AxisFusion

**重点要求**：分发给 subagent 执行 AxisFusion 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/AxisFusion.md` 下的**操作说明**，按照规范要求充当 AxisFusion 角色，完成 {算子名} 算子的轴融合优化任务。

## 用户输入
- 存储路径：`{{output_dir}}/KernelGen`
- Step 2 输出：{{output_dir}}/KernelGen/step2_block_mapping.json（包含 compute_formula, compute_note, io_block_mapping）

## 重要约束
- **只能读取**以下文件：

- 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/AxisFusion.md`
  - Step 2 输出：`{{output_dir}}/KernelGen/step2_block_mapping.json`
- **禁止读取**其他文件（如原始需求文档、其他步骤的输出文件等）
- 读取文件时必须使用绝对路径

请严格按照任务文档要求执行。
"""
)
```

### 步骤 4：生成代码规范方案

根据 Step 3 的输出，生成代码生成方案。

**注意**：只有当输入类型为非 Triton 时才执行此步骤。

| 步骤   | Subagent       | 说明             | 关键输出                  | 输出文件                                        |
| ------ | -------------- | ---------------- | ------------------------- | ----------------------------------------------- |
| Step 4 | `GenerateSpec` | 生成代码生成方案 | kernel spec, wrapper spec | `{输出存储路径}/KernelGen/step4_code_spec.json` |

#### 执行 Step 4 - GenerateSpec

**重点要求**：分发给 subagent 执行 GenerateSpec 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/GenerateSpec.md` 下的**操作说明**，按照规范要求充当 GenerateSpec 角色，完成 {算子名} 算子的代码生成方案任务。

## 用户输入
- 存储路径：`{{output_dir}}/KernelGen`
- Step 3 输出：{{output_dir}}/KernelGen/step3_axis_fusion.json（包含 compute_formula, compute_note, io_block_mapping, fusion_note）
## 重要约束
- **只能读取**以下文件：
  - 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/GenerateSpec.md`
  - MLU 生成阶段原语约束：`.claude/skills/share/mlu/references/primitives.md`
  - MLU 平台规则：`.claude/skills/share/mlu/references/platform-rules.md`
  - Step 3 输出：`{{output_dir}}/KernelGen/step3_axis_fusion.json`
- **禁止读取**其他文件（如其他步骤的输出文件等）
- 读取文件时必须使用绝对路径

请严格按照任务文档要求执行。
"""
)
```

### 步骤 5：生成代码

根据输入类型选择不同的处理方式：

#### 情况 1：Triton 输入（快速路径）

- **触发条件**：步骤 0 判断为 Triton 输入（`is_triton: true` 或 `input_type: triton`）
- **处理方式**：直接读取 `{输出存储路径}/Extractor/original_code.py` 作为输出
- **输出文件**：将代码复制到 `{输出存储路径}/KernelGen/step5_kernel_code.py`

#### 情况 2：非 Triton 输入（正常流程）

- **触发条件**：步骤 0 判断为非 Triton 输入
- **处理方式**：调用 `GenerateCode` subagent，根据 Step 4 的输出生成 Triton kernel 代码和 wrapper 函数
- **输出文件**：`{输出存储路径}/KernelGen/step5_kernel_code.py`

| 步骤   | Subagent       | 说明                                                   | 关键输出    | 输出文件                                        |
| ------ | -------------- | ------------------------------------------------------ | ----------- | ----------------------------------------------- |
| Step 5 | `GenerateCode` | 生成 Triton kernel 代码（或直接使用 original_code.py） | triton_code | `{输出存储路径}/KernelGen/step5_kernel_code.py` |

#### 执行 Step 5 - GenerateCode

**重点要求**：分发给 subagent 执行 GenerateCode 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/GenerateCode.md` 下的**操作说明**，按照规范要求充当 GenerateCode 角色，完成 {算子名} 算子的 Triton kernel 代码生成任务。

## 用户输入
- 存储路径：`{{output_dir}}/KernelGen`
- Step 4 输出：{{output_dir}}/KernelGen/step4_code_spec.json（包含 kernel spec, wrapper spec）

## 重要约束
- **只能读取**以下文件：
  - 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/GenerateCode.md`
  - MLU 生成阶段原语约束：`.claude/skills/share/mlu/references/primitives.md`
  - MLU 平台规则：`.claude/skills/share/mlu/references/platform-rules.md`
  - Step 4 输出：`{{output_dir}}/KernelGen/step4_code_spec.json`
  - 原始需求文档：`{{output_dir}}/Extractor/requirement.md`
- **禁止读取**其他文件（如其他步骤的输出文件等）
- 读取文件时必须使用绝对路径

请严格按照任务文档要求执行。

**重要**：只生成 triton code + kernel wrapper，不包含测试代码。
"""
)
```

### 步骤 6：生成测试代码
使用 `GenTestCode` subagent，根据需求文档和 Step 5 生成的 triton code，生成完整的测试代码。

**输出**：

- 完整的可执行 `.py` 文件，包含 triton code + 精度测试 + 性能测试

| 步骤   | Subagent      | 说明         | 关键输出  | 输出文件                                      |
| ------ | ------------- | ------------ | --------- | --------------------------------------------- |
| Step 6 | `GenTestCode` | 生成测试代码 | test_code | `{输出存储路径}/KernelGen/step6_test_code.py` |

#### 执行 Step 6 - GenTestCode

**重点要求**：分发给 subagent 执行 GenTestCode 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
## 任务文档
读取路径 `.claude/skills/mlu-triton-code-gen/subagents/GenTestCode.md` 下的**操作说明**，按照规范要求充当 GenTestCode 角色，完成 {算子名} 算子的测试代码生成任务。

## 用户输入
- 输出根目录：`{{output_dir}}`
- 存储路径：`{{output_dir}}/KernelGen`
- Extractor 输出：`{{output_dir}}/Extractor/requirement.md`
- io_shapes：`{{output_dir}}/KernelGen/step1_io_shapes.json`（输入输出形状/连续性信息，用于生成测试输入）
- Step 5 输出：`{{output_dir}}/KernelGen/step5_kernel_code.py`（包含 triton code 和 wrapper）
## 重要约束
- **只能读取**以下文件：
  - 操作说明：`.claude/skills/mlu-triton-code-gen/subagents/GenTestCode.md`
  - MLU 平台规则：`.claude/skills/share/mlu/references/platform-rules.md`
  - Extractor 输出：`{{output_dir}}/Extractor/requirement.md`
  - io_shapes：`{{output_dir}}/KernelGen/step1_io_shapes.json`
  - Step 5 输出：`{{output_dir}}/KernelGen/step5_kernel_code.py`
- **禁止读取**其他文件（包括 step2/3/4 的中间产物 step2_block_mapping.json / step3_axis_fusion.json / step4_code_spec.json，以及原始 step1_base_info.json）
- 读取文件时必须使用绝对路径

请严格按照**操作说明**执行。

**重要**：输出完整的 .py 文件，包含：
1. Triton kernel 代码
2. Wrapper 函数
3. 数据生成函数
4. 精度验证函数（必须按照操作说明文档中的格式实现）
5. 性能测试函数（必须按照操作说明文档中的格式实现）

本步骤只负责生成测试代码，不执行该文件；运行与精度验证由后续 `mlu-triton-code-review` 完成。
"""
)
```

### 步骤 7：代码验证

使用 `mlu-triton-code-review` Skill 验证待交付代码，并获取最终代码文件路径。

**Step 7 输入路由**：

- 直接验证 `{输出存储路径}/KernelGen/step6_test_code.py`


```python
# 调用 mlu-triton-code-review 进行验证
Skill(
    skill="mlu-triton-code-review",
    args="{step7_input_code_path}"
)
```

验证内容：

- 功能正确性验证
- 精度验证

**获取最终代码路径**：

- `mlu-triton-code-review` 的输出契约：输入 `xxx.py` → 同目录产出 `xxx_fix.py`（单个最终产物，迭代在其内部完成，不对外暴露"迭代次数"）
- 因此 `final_code_path = step7_input_code_path` 去掉 `.py` 后追加 `_fix.py`
- `final_code_path` 为 `{output_dir}/KernelGen/step6_test_code_fix.py`
- 将此路径作为整个代码生成流程的最终代码文件路径，用于步骤 9 的输出

### 步骤 8：检查回退机制

每个步骤验证失败时的回退处理：

**快速路径场景（Triton 输入）**：

| 失败场景        | 处理方式                                 | 回退目标              |
| --------------- | ---------------------------------------- | --------------------- |
| Step 5 验证失败 | 重新执行 Step 5（使用 original_code.py） | Step 5 (GenerateCode) |
| Step 6 验证失败 | 重新执行 Step 6                          | Step 6 (GenTestCode)  |
| 最终验证失败    | 回到步骤 0 重新判断                      | Step 0 (输入类型检查) |

**正常路径场景（非 Triton 输入）**：

| 失败场景        | 处理方式             | 回退目标                   |
| --------------- | -------------------- | -------------------------- |
| Step 2 验证失败 | 重新执行 Step 2      | Step 2 (TraceBlockMapping) |
| Step 3 验证失败 | 重新执行 Step 3      | Step 3 (AxisFusion)        |
| Step 4 验证失败 | 重新执行 Step 4      | Step 4 (GenerateSpec)      |
| Step 5 验证失败 | 重新执行 Step 5      | Step 5 (GenerateCode)      |
| Step 6 验证失败 | 重新执行 Step 6      | Step 6 (GenTestCode)       |
| 最终验证失败    | 回到 Step 1 重新开始 | Step 1 (ExtractBaseInfo)   |

**回退限制**：每个步骤最多回退 3 次，超过限制则报告失败

### 步骤 9：输出结果

根据步骤 7 获取的 `final_code_path`，将最终代码复制/保存到输出目录：

- **最终代码文件（交接给下游的 fix 版本代码）**：`{输出存储路径}/KernelGen/triton_code_fix.py`（从步骤 7 返回的 `step6_test_code_fix.py` 复制并改名得到）
- **测试报告**：`{输出存储路径}/KernelGen/triton_report.md`

## 输出格式

生成完成后，输出以下信息：

```
## Triton Kernel 生成结果

### 算子信息
- 源需求：<requirement_path>
- 算子名称：<算子名>

### 生成流程记录
0. [输入类型检查] - 输入类型：Triton / 非 Triton（快速路径：跳过步骤 1-4）
1. [ExtractBaseInfo] - 执行结果：成功/失败/跳过
2. [TraceBlockMapping] - 执行结果：成功/失败/跳过
3. [AxisFusion] - 执行结果：成功/失败/跳过
4. [GenerateSpec] - 执行结果：成功/失败/跳过
5. [GenerateCode] - 执行结果：成功/失败
6. [GenTestCode] - 执行结果：成功/失败
7. [mlu-triton-code-review] - 执行结果：成功/失败
### 验证结果
- 功能正确性：通过/失败
- 精度验证：通过/失败
- 性能结果：具体性能数据

### 最终代码文件路径
{mlu-triton-code-review 返回的 final_code_path}

### 生成代码
（输出完整的 Triton kernel 代码 + wrapper 函数）
```

## 与其他 SKILL 的关系

- **依赖**：依赖 `Extractor` SKILL 输出的需求文档
- **验证**：使用 `mlu-triton-code-review` SKILL 进行代码验证
- **后续**：验证通过后可使用 `mlu-triton-optimize` SKILL 进行性能优化
