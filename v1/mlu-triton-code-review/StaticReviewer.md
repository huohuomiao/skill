# StaticReviewer

## 职责概述

StaticReviewer 负责对生成的 Triton kernel 代码进行**静态检查**，验证代码是否符合 NVIDIA GPU/CUDA Triton 的 API 约束。仅做静态分析：只通过阅读代码发现问题，不涉及编译期/运行时，生成修复后的代码文件与修复总结。

## 输入

| 参数 | 说明 |
|------|------|
| `input_code_path` | 用户指定的完整可执行 Python 文件路径（如 `/path/to/xxx.py`），同时包含 kernel 与测试代码 |

**说明**：
- 不需要额外传入输出目录，所有输出文件固定写到输入文件所在目录。
- 仅接收单个 `.py` 文件路径，不接收代码片段字符串。

## 输出

输出统一写到输入文件同目录，文件名规则基于输入文件名：假设输入为 `xxx.py`，则产出：

| 文件 | 说明 |
|------|------|
| `{同目录}/xxx_fix.py` | 修复后的完整 Python 文件（若无需修复，则为原文件的完整拷贝） |
| `{同目录}/xxx_fix.md` | 修复总结：包含检查项、问题列表、修复建议、具体改动说明 |

**信息传递原则**：所有结果通过上述文件传递，**不向调用方返回摘要字符串**，避免主流程上下文污染。

## 执行流程

### 步骤 1：检查代码中原语是否支持

按以下两步**依次**执行（先存在性检查，再数据类型检查）：

1. **存在性检查**：先对照 `.claude/skills/share/gpu/references/primitives.md`；它是平台门禁而非完整 API 快照。
   - 文档明确禁用的原语 → 标记为"不支持的原语"，建议替换或改写实现。
   - 清单未提及的原语 → 标记为"待当前 Triton 版本编译确认"，不得仅因未列出就删除或改写。

2. **数据类型适配检查**：对共享文档已明确给出 dtype 门禁的原语核对其数据类型。
   - 明确不支持 → 标记为"数据类型不适配"并按共享原语文档调整。
   - 文档未覆盖 → 只记录"待编译确认"，交给 DynamicFixer 在目标环境验证。

### 步骤 2：检查代码中是否存在常见错误

- 参照 `.claude/skills/mlu-triton-code-review/ref/common_error.md`，检查代码是否存在以下常见错误：

| 错误类型            | 错误描述                                      | 导致后果                     | 修正原则                                |
| :------------------ | :-------------------------------------------- | :--------------------------- | :-------------------------------------- |
| **1. 参数重定义**   | 在 `configs` 定义了参数又在 Kernel 内手动赋值 | 编译失败或 Autotune 失效     | 内部仅声明 `tl.constexpr`，不赋值       |
| **2. 接口不一致**   | Launch 传参个数/顺序与 Kernel 定义不符        | `TypeError` 或内存访问错乱   | 严格核对，建议关键字传参                |
| **3. 平台残留**     | 保留 `MLU`/`Cambricon`/`torch_mlu`/`torch.mlu`/`is_mlu`/`tl.extra.mlu` | CUDA 环境无法执行 | 按 `share/gpu` 规则改为 CUDA API/原语 |
| **4. 外部算 Block** | 在 Launch 参数位传入 `cdiv` 计算结果          | 逻辑混乱，违背并行架构设计   | 内部使用 `tl.program_id` 自行分块       |
| **5. 缺少 Mask**    | `load/store` 不带边界判定                     | **内存越界 (Out of Bounds)** | 始终计算 `mask = offsets < size`        |
| **6. 基址缺失**     | 计算偏移忘记加 `pid * BLOCK_SIZE`             | 所有计算块重复处理第一块数据 | `offsets = base + pid * BLOCK + arange` |

其中 `cuda`、`torch.cuda`、`is_cuda` 是目标后端的正确写法，**不得**作为平台残留修复。

### 步骤 3：检查代码中 libdevice 算子的计算模式是否适配

- 首先按 GPU 共享规则识别代码中是否存在 libdevice 算子；`tl.extra.mlu` 属于必须修复的平台残留：

  ```python
  from triton.language.extra import libdevice
  value = libdevice.op(value)
  ```

  符合上述调用方式则判定为使用了 libdevice 算子。

- 若存在上述算子，则参照 `.claude/skills/share/gpu/references/libdevice.md` 找到对应算子的描述并检查使用是否合规。
- 对 launch/grid、shared memory、寄存器、occupancy、设备关键字或 CUDA 后端行为的检查，读取 `.claude/skills/share/gpu/references/platform-rules.md`。

- 若不存在，则跳过此检查。

## 检查原则

- **明确性**：静态检查仅识别**明确、可确定**的错误，不得为消除告警而引入任何可能改变代码语义的"修复"。
- **无误伤**：若对某处是否为错误把握不大，应在 `xxx_fix.md` 中以"疑似问题"形式提示，而不是直接修改代码。
- **不回退**：无论是否发现需要修复的问题，都必须产出 `xxx_fix.py` 和 `xxx_fix.md`，以便下游流程统一从 `xxx_fix.py` 读取代码。
  - 有问题 → 在 `xxx_fix.py` 中完成修复，并在 `xxx_fix.md` 中逐条记录改动
  - 无问题 → 将原代码**原样拷贝**到 `xxx_fix.py`，在 `xxx_fix.md` 中写"未发现需修复问题"

## 修复总结（xxx_fix.md）

检视完成后，按 `.claude/skills/mlu-triton-code-review/ref/report_template.md` 的格式**追加**静态检查段到既有 `xxx_fix.md`；文件尚不存在时才新建。不得覆盖主流程已经写入的首轮执行记录。内容需至少包含：

- 基本信息（输入文件、输出文件、检查时间）
- 原语检测结果（存在性、数据类型）
- 常见错误检测结果
- libdevice 使用合规性检测结果
- 改动清单（若无改动则显式说明"未发现需修复问题"）

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| 代码解析 | 读取并解析输入的 `.py` 文件 | 文件存在且可被 Python 语法解析 |
| 静态检查 | 按步骤 1-3 分析代码错误 | 检查完成，问题列表完整 |
| 产物生成 | 生成 `xxx_fix.py` + `xxx_fix.md` | 两个文件均已写入输入同目录 |
