# Triton Kernel 性能分析

## 职责概述

使用性能分析工具对目标 Triton 代码进行性能分析，解析性能指标，并生成深度优化建议。

MLU 专属采集实现位于 `.claude/skills/share/mlu/perf-analyzer/`，本文件只保留可复用的分析与报告流程。

## 执行契约

所有涉及 Triton 代码或 cnperf 命令的执行必须先确认当前工作流的 EnvConfig 产物，不允许直接猜测本地执行环境或 Worker。

执行前从本轮 perf-analyzer 的 `output_dir` 向上推断工作流 `{output_dir}`，读取 `{output_dir}/EnvConfig/config.md` 中的 `execution_backend`：

- `execution_backend=local`：直接在本地执行环境执行 `bash analyzer.sh ...`
- `execution_backend=worker`：通过 `submit_task_to_worker.py` 提交 Worker Task 执行同一条 `bash analyzer.sh ...` 命令
- EnvConfig 缺失或无法判断后端：记录环境错误并终止 perf-analyzer，**不要修改 kernel 代码**

Worker 执行示例（必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再判断结果；禁止 `&` 后台、禁止并发提交多个 Worker Task）：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <仓库根目录的绝对路径> \
    --command "bash .claude/skills/share/mlu/perf-analyzer/analyzer.sh <output_dir> <output_dir>/run_once.py" \
    --timeout-sec 1800 \
    --task-type custom

```

**Worker 模式结果分类**（以 `submit_task_to_worker.py` 退出码为准）：
- `0` → 执行成功
- `1` → 业务错误（Traceback / 精度不达标）
- `2` → 基础设施错误（输入路径不存在 / Worker 不可达等）→ **不要修改 kernel 代码**，先修 EnvConfig 或路径，终止本 Skill 并提示用户

- 禁止为了测试或 benchmark 另建 Job。
- 判断优化是否有效时，必须以实际执行环境的真实 stdout/stderr/result 为准。

## 输入输出说明

**输入**：
- `input_file`：可直接运行的 Python 代码文件，包含 Triton kernel 定义、wrapper 函数、精度及性能测试
- `output_dir`：生成文件存储目录

**输出**：
- `{output_dir}/report.md`：对 `input_file` 的性能分析报告

**要求**：
- 修改后的代码能够正确执行，跑通精度测试，若性能下降则回退为初始代码

## 现有深度优化策略

|      策略      |     策略名称     |                            说明                             |
|---------------|-----------------|
------------------------------------------------------------|
| Libdevice 优化 | `libdevice-opt` | 使用 Cambricon libdevice 高效算子替换低效计算模式（fast_* 优先） |
| Config 微调    | `config-tuner`  | 通过调整 block size，num_stages，num_warps 等参数进行性能调优   |
| 除法指令优化 | `div-to-mul` | 将除法指令修改为乘倒数 |
## 性能分析步骤

### Step 1：使用性能分析脚本生成性能数据

1. 首先简化代码，保证 Triton kernel 只执行一次。

提取 Triton 代码中精度性能测试的输入规模，包括 `Shape` 以及 `Dtype` 信息:

例如：
```python
Shape = (128 * 1024 * 1024,)
Dtype = torch.float32
```

删除 `performance_test`，只执行 `accuracy_test`，随后将代码存储为`{output_dir}/run_once.py`

2. 运行脚本获取性能数据

```bash
bash .claude/skills/share/mlu/perf-analyzer/analyzer.sh {output_dir} {output_dir}/run_once.py
rm {output_dir}/run_once.py
```

上述命令也必须按本文件“执行契约”选择后端：local 直接执行，worker 通过 `submit_task_to_worker.py --task-type custom` 执行。

### Step 2：性能数据解析

#### 2.1：NRAM 使用率

**症状**：NRAM 使用率不满 90%

**NRAM 使用率低的可能原因**：

  - BLOCK SIZE 参数过小

**优化建议**：

  - 优化策略 `config-tuner`
  - 优化策略 `libdevice-opt`
  - 优化策略 `div-to-mul`

### Step 3：生成性能分析报告

根据以上信息，生成性能分析报告，并存储在 `{output_dir}/report.md`。

性能分析报告模版见：[`report_template.md`](references/report_template.md)
