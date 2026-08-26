---
name: mlu-triton-code-review
description: Triton Kernel 代码验证工具，负责对生成的 Triton kernel 代码进行动态检查和修复，直到获得正确的精度值
---

# mlu-triton-code-review

## 概述

该 SKILL 是 MLU Triton Kernel 代码验证修复工具，对输入的 Python 文件（同时包含 kernel 与测试代码）进行**执行优先**的验证：先跑一次原代码，通过则直接完成；失败才依次进入静态检查 → 动态修复的迭代流程，直至得到可正确执行且精度达标的代码。

**核心目标**：让输入的 Triton kernel 代码在 MLU 上正确运行并通过精度验证。

**工作原则**：
- **执行优先**：测试能通过就不做任何静态/动态检查，避免误伤
- **文件传递信息**：所有中间产物和总结通过文件输出，**不返回摘要字符串到上下文**
- **统一输入输出契约**：输入为 `.py` 文件路径，输出固定为同目录下的 `xxx_fix.py` + `xxx_fix.md`
- **运行环境选择规则**：动态执行必须遵守 `.claude/skills/mlu-triton-main/SKILL.md` 中的规则；本地执行环境有可用 MLU/Triton-MLU 工具链时优先本地执行，否则在当前 `JOB_ID` 下通过 `submit_task_to_worker.py` 提交 Worker Task，并以真实日志/结果判定

## 用法

```bash
/mlu-triton-code-review <input_code_path>
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `input_code_path` | 用户指定的完整可执行 Python 文件路径（如 `/path/to/xxx.py`），必须同时包含 Triton kernel 和可运行的测试代码 |

**仅接收文件路径**，不接收代码片段字符串；输出目录由输入路径自动推导，无需用户指定。

## 输出

所有输出统一写入**输入文件所在目录**。假设输入为 `xxx.py`：

| 文件 | 说明 |
|------|------|
| `{同目录}/xxx_fix.py` | 最终可执行、通过测试的 Python 文件；若原代码直接通过，则为原代码的原样拷贝 |
| `{同目录}/xxx_fix.md` | 修复总结（执行记录、静态/动态检查发现、每轮改动） |

## 红线（修复时严禁的做法）

在任何修复迭代中，**禁止**出现以下"替代式修复"——它们会让测试通过但背离 Triton on MLU 的目标：

1. ❌ 将 Triton kernel 改为 CPU 实现
2. ❌ 用纯 PyTorch 算子替代原 Triton kernel 的计算
3. ❌ 把 Triton kernel 写成标量（逐元素循环）执行，绕过 tile 并行语义

一旦迭代中出现上述迹象，必须立刻回退该轮修改，改走其他修复思路。

## 工作流程

> 总体思想：**执行优先，按需检查，迭代修复**。
### 步骤 1：首轮执行原代码

**执行契约**：首轮执行必须先确认当前工作流的 EnvConfig 产物，不允许直接猜测本地执行环境或 Worker。

- 优先从 `input_code_path` 所在目录向上查找 `{output_dir}/EnvConfig/config.md`，读取其中的 `execution_backend`
- `execution_backend=local`：直接在本地执行环境执行
- `execution_backend=worker`：通过 `submit_task_to_worker.py` 提交 Worker Task
- EnvConfig 缺失或无法判断后端时，必须先回到 `mlu-triton-main` 的 EnvConfig 规则完成环境确认；不要把环境错误当作 kernel 错误修复

本地执行：

```bash
python <input_code_path>
```

Worker 执行（必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再判断结果；禁止 `&` 后台、禁止并发提交多个 Worker Task）：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <绝对路径> \
    --command "python <input_code_path>" \
    --timeout-sec 1800 \
    --task-type accuracy
```

**结果分类**：
- `0` → 执行成功且精度断言通过 → 进入步骤 2a
- `1` → 业务错误（Traceback / 精度不达标）→ 进入步骤 2b
- `2` → 基础设施错误（输入路径不存在 / Worker 不可达等）→ **不要修改 kernel 代码**，先修 EnvConfig 或路径，终止本 Skill 并提示用户

### 步骤 2a：原代码通过 —— 直接产出并结束

1. 将原文件**原样拷贝**为 `{同目录}/xxx_fix.py`
2. 在 `{同目录}/xxx_fix.md` 中写入"原代码已通过测试，无需修改"，并附首轮执行日志摘要
3. 任务完成，退出流程

### 步骤 2b：原代码失败 —— 进入静态检查

**重点要求**：分发给 subagent 执行`输入代码静态检查`任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
    ## 任务文档
    根据 .claude/skills/mlu-triton-code-review/StaticReviewer.md 中的规范要求，充当 StaticReviewer 角色，
    对输入的 Python 文件执行静态检查任务。

    ## 输入
    input_code_path: {input_code_path}

    严格按照任务文档执行：仅修复明确的错误；无论是否发现问题，都必须产出
    {input_code_path 去除 .py}_fix.py 和 {input_code_path 去除 .py}_fix.md。
    """
)
```
StaticReviewer 执行完后，同目录已存在 `xxx_fix.py` 和 `xxx_fix.md`，后续步骤统一从 `xxx_fix.py` 继续。

### 步骤 3：动态修复（执行 + 迭代修复）

**重点要求**：分发给 subagent 执行`静态修复后代码的动态修复`任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
    ## 任务文档
    根据 .claude/skills/mlu-triton-code-review/DynamicFixer.md 中的规范要求，充当 DynamicFixer 角色，
    对经过静态检查后的 Python 文件执行动态修复任务（执行驱动 + 按错误分类迭代修复）。

    ## 输入
    fixed_code_path: {xxx_fix.py 的绝对路径}

    严格按照任务文档执行：
    - 必须遵守 mlu-triton-main 主 Skill 的运行环境选择规则：本地执行环境 MLU 可用时直接 `python xxx.py`，否则通过 `.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 提交 Worker Task
    - 迭代时直接覆盖 xxx_fix.py，并在 xxx_fix.md 追加迭代记录
    - 遵守红线：严禁 CPU / PyTorch 替代 / 标量 kernel 修复
    - 达到终止条件（通过 / 同类错误连续 2 次 / 最大 5 次迭代）即结束
    - 所有结果通过文件传递，不向调用方返回摘要字符串
    """
)
```

DynamicFixer 执行完后，`xxx_fix.py` 为最终代码、`xxx_fix.md` 已在静态检查段之后追加动态修复全部迭代记录与结论。主流程**不再**直接运行代码，也不再自行解析报错。

### 流程总览

```
输入 .py
  ↓
[步骤1] 执行原代码
  ├─ 通过 → [步骤2a] 原样拷贝为 xxx_fix.py，xxx_fix.md 写"无修改"，结束
  └─ 失败 ↓
[步骤2b] 静态检查（StaticReviewer 子代理） → 生成 xxx_fix.py + xxx_fix.md
  ↓
[步骤3] 动态修复（DynamicFixer 子代理）
         内部循环：执行 xxx_fix.py ↔ 按错误分类改写，
         终止条件：通过 / 同类错误连续 2 次 / 达到最大迭代 5 次
  ↓
最终产物：xxx_fix.py + xxx_fix.md（含静态检查 + 动态修复记录）
```

## xxx_fix.md 的分段写入约定

`xxx_fix.md` 由主流程和两个子代理**分段追加**完成，具体格式由对应文档规范：

| 段落 | 写入方 | 参考文档 |
|------|-------|---------|
| 基本信息 / 首轮执行结果（步骤 1） | 主流程（本 SKILL） | 本文档 |
| 静态检查发现（步骤 2b） | StaticReviewer | `StaticReviewer.md` + `ref/report_template.md` |
| 动态修复迭代 / 最终精度 / 结论（步骤 3） | DynamicFixer | `DynamicFixer.md` |

主流程仅负责在步骤 1 / 2a 写入"基本信息 + 首轮执行结果"段，随后把文件交给子代理追加各自段落，**不再**规定各段具体字段——避免与子文档重复，以子文档为准。
