# Triton Kernel 性能分析

## 职责概述

使用性能分析工具对目标 Triton 代码进行性能分析，解析性能指标，并生成深度优化建议。

MLU 专属采集实现位于 `{skill_root}/scripts/profiling/`，本文件只保留可复用的分析与报告流程。

## 执行契约

Read `{output_dir}/EnvConfig/config.json` and `{output_dir}/Optimizer/tuning_state.json`, then run profiling commands exactly through `{skill_root}/references/contracts/execution-backend.md`. Check the global budget first. For Worker profiling pass `--budget-state` and a unique `--budget-label`; exit `4` stops analysis without retry or advice. Backend selection, Worker submission, timeout, and result classification are not repeated here. An infrastructure failure stops analysis without modifying the kernel. Use only real stdout/stderr/result as evidence. All real hardware profiling and benchmarks are serial.

## 输入输出说明

**输入**：
- `input_file`：可直接运行的 Python 代码文件，包含 Triton kernel 定义、wrapper 函数、精度及性能测试
- `output_dir`：生成文件存储目录
- `optimization_mode=max-performance`

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
bash {skill_root}/scripts/profiling/collect-profile.sh {output_dir} {output_dir}/run_once.py
rm {output_dir}/run_once.py
```

上述命令必须按本文件“执行契约”运行。

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

性能分析报告模板见：`{skill_root}/references/templates/performance-report.md`
