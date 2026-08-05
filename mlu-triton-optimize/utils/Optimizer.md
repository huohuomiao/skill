# Optimizer

## 职责概述

作为指定优化策略的包装器，对策略文档进行读取，执行。最重要地，对优化策略提出诸多约束，保证其可以正确的执行。

## 核心原则

⚠️⚠️⚠️ **非常重要**：优化策略要严格按照策略文档中的工作流程进行，所有调优必须在对应策略文档中有说明，不得做出越界行为。

⛔ **禁止**：凭空捏造，按自己想法进行盲目调优。

## 工作原则

- **保证有输出**：无论优化结果如何，都必须有输出代码 `triton_optimized.py` 和 调优过程 `triton_optimized.md`

## 执行契约（benchmark / 精度验证的统一入口）

所有涉及 Triton 代码的执行必须先确认当前工作流的 EnvConfig 产物，不允许策略自己猜测本地执行环境或 Worker。

### 执行后端选择

执行 benchmark / 精度验证前，先从当前策略 `工作目录` 向上推断 `{output_dir}`，并读取：

```text
{output_dir}/EnvConfig/config.md
```

后端判定：

- `execution_backend=local`：直接在本地执行环境执行代码
- `execution_backend=worker`：通过 `submit_task_to_worker.py` 提交 Worker Task
- EnvConfig 缺失或无法判断后端：记录环境错误并终止当前策略，**不要修改 kernel 代码**

本地执行：

```bash
python xxx.py
```

Worker 兜底执行（必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再判断结果；禁止 `&` 后台、禁止并发提交多个 Worker Task）：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <当前工作目录绝对路径> --command "python xxx.py" \
    --task-type {accuracy|performance}

```

**Worker 模式结果分类**（以 `submit_task_to_worker.py` 退出码为准）：
- `0` → 执行成功
- `1` → 业务错误（Traceback / 精度不达标）
- `2` → 基础设施错误（输入路径不存在 / Worker 不可达等）→ **不要修改 kernel 代码**，先修 EnvConfig 或路径，终止本 Skill 并提示用户

- 禁止为了测试或 benchmark 另建 Job。
- 判断优化是否有效时，必须以实际执行环境的真实 stdout/stderr/result 为准。

## Optimizer 包装器输入说明

| 参数 | 说明 |
|------|------|
| 策略名称 | 当前执行的优化策略 |
| 策略文档路径 | 当前策略的对应文档 |
| 工作目录 | 当前策略的工作目录 |

## 优化策略输入输出路径说明

- 输入 Triton 代码：`{工作目录}/input.py`
- 输出 Triton 代码：`{工作目录}/triton_optimized.py`
- 输出优化策略报告：`{工作目录}/triton_optimized.md`

## 优化策略执行步骤

### 步骤 1：读取策略文档

使用 Read 工具读取优化策略的操作指引

```bash
Read {策略文档路径}
```

### 步骤 2：执行优化策略

根据输入指定的 `策略名称`，按照策略 `strategy.md` 中的操作指引，严格按步骤执行优化。

### 步骤 3：输出优化结果文件

将优化策略输出的 Triton 代码写入 `{工作目录}/triton_optimized.py`；将本次策略的优化报告写入优化报告文件 `{工作目录}/triton_optimized.md`。

每个策略的报告块格式如下：

```markdown
## 策略：{策略名称}
- 状态：{成功/失败}

### 优化说明
{简要描述本策略做了什么修改、修改了哪些部分}

### 精度结果
- 精度是否通过：{accuracy_pass}
- 绝对误差容限 (atol)：{accuracy_atol}
- 相对误差容限 (rtol)：{accuracy_rtol}

### 性能测量
| 指标 | PyTorch | 优化前 Triton | 优化后 Triton |
|------|---------|-------------|-------------|
| 耗时 (ms) | {torch_ms} | {original_triton_ms} | {opt_triton_ms} |
| 带宽 (GB/s) | {torch_bandwidth} | {original_triton_bandwidth} | {opt_triton_bandwidth} |

### 加速比
- 优化后 vs 优化前：{speedup_opt_vs_original}x
- 优化后 vs PyTorch：{speedup_opt_vs_torch}x

### 最终代码
- 文件：`{工作目录}/triton_optimized.py`
```

**注意事项**：
- 若策略使用原始代码，报告中需标注状态为`失败`，性能数据使用原始代码的数据（opt 值 = original 值，加速比 = 1.0）
- 每个策略的报告块以 `---` 分隔线开头，便于主 agent 解析

---

## 约束

### 1. 调试规范

- 运行代码若出错，请根据错误信息进行调试修改，尝试修复次数最多为3

### 2. 文件权限

- **只能读取**以下文件：
  1. 当前 Triton 代码路径指定的文件
  2. 当前策略的 `strategy.md`
  3. 当前策略的 references 目录下的参考文档（若存在）
  4. 当前工作目录向上推断出的 `{output_dir}/EnvConfig/config.md`
- **可以写入**以下文件：
  1. 当前工作目录下的文件
- **禁止读取**以下文件：
  1. 其他优化策略目录下的文件
  2. 其他工作目录下的文件
- 读取文件时必须使用绝对路径

