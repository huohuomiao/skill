# Triton Kernel 性能分析

## 职责概述

使用性能分析工具对目标 Triton 代码进行性能分析，解析性能指标，并生成深度优化建议。

NVIDIA CUDA 采集实现位于 `.claude/skills/share/gpu/perf-analyzer/`，本文件只保留可复用的分析与报告流程。

`analyzer.sh` 负责统一入口，NCU 结果解析由 `.claude/skills/share/gpu/perf-analyzer/analyzer_ncu.py` 完成；具体 section、metric 名称和版本兼容逻辑以该共享实现为准，本文件不复制一份易过期的 metric 清单。

## 执行契约

所有涉及 Triton 代码或 NCU 命令的执行必须先确认当前工作流的 EnvConfig 产物，不允许直接猜测本地执行环境或 Worker。

执行前从本轮 perf-analyzer 的 `output_dir` 向上推断工作流 `{output_dir}`，读取 `{output_dir}/EnvConfig/config.md` 中的 `execution_backend`：

- `execution_backend=local`：直接在本地执行环境设置 `NCU_KERNEL_NAME` 后执行 `bash analyzer.sh ...`
- `execution_backend=worker`：通过 `submit_task_to_worker.py` 提交 Worker Task 执行同一条带 `NCU_KERNEL_NAME` 的 `bash analyzer.sh ...` 命令
- EnvConfig 缺失或无法判断后端：记录环境错误并终止 perf-analyzer，**不要修改 kernel 代码**

Worker 执行示例（必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再判断结果；禁止 `&` 后台、禁止并发提交多个 Worker Task）：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <仓库根目录的绝对路径> \
    --command "NCU_KERNEL_NAME='regex:.*<triton_kernel_name>.*' bash .claude/skills/share/gpu/perf-analyzer/analyzer.sh <output_dir> <output_dir>/run_once.py" \
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
|---------------|-----------------|-------------------------------------------------------------|
| Libdevice 优化 | `libdevice-opt` | 使用官方 CUDA libdevice 或标准 `tl` 原语替换低效计算模式 |
| Config 微调    | `config-tuner`  | 通过调整 block size，num_stages，num_warps 等参数进行性能调优   |
| 架构与配置联合搜索 | `gen-autotune-config` | 普通/persistent/split-K 各自调参后全局择优；用于架构级资源瓶颈 |
| 除法指令优化 | `div-to-mul` | 将除法指令修改为乘倒数 |
## 性能分析步骤

### Step 1：使用性能分析脚本生成性能数据

1. 首先简化代码，保证待分析的 Triton kernel 只执行一次，并记录其 `@triton.jit` Python 函数名。多 kernel 流水线必须逐个采集，不能把第一个 CUDA launch 当作目标 kernel。

提取 Triton 代码中精度性能测试的输入规模，包括 `Shape` 以及 `Dtype` 信息:

例如：
```python
Shape = (128 * 1024 * 1024,)
Dtype = torch.float32
```

删除 `performance_test`，只执行 `accuracy_test`，随后将代码存储为`{output_dir}/run_once.py`

2. 运行脚本获取性能数据

```bash
NCU_KERNEL_NAME='regex:.*<triton_kernel_name>.*' \
    bash .claude/skills/share/gpu/perf-analyzer/analyzer.sh {output_dir} {output_dir}/run_once.py
rm {output_dir}/run_once.py
```

`analyzer.sh` 会以 NCU `function` 名匹配并只采集第一个匹配 launch；若没有设置 `NCU_KERNEL_NAME` 则直接失败，避免误采集 PyTorch reference 或输入初始化 kernel。

上述命令也必须按本文件“执行契约”选择后端：local 直接执行，worker 通过 `submit_task_to_worker.py --task-type custom` 执行。

### Step 2：性能数据解析

#### 2.1：寄存器、Shared Memory 与 Occupancy

**症状**：寄存器 spilling、shared-memory 压力过高、理论/实测 occupancy 偏低，或内存/计算吞吐未达到合理水平

**可能原因**：

  - BLOCK_SIZE、`num_warps`、`num_stages` 组合导致并行度不足或每个 program 资源占用过高
  - 普通 kernel 被错误限制为 SM 数量；persistent kernel 未按编译后资源占用计算可驻留 program 数

解析时至少区分以下信息，缺失则在报告中标为“未采集”，不得补造数值：

- 编译资源：registers/thread、shared memory/block、threads/block、local spilling；
- 并行度：theoretical occupancy、achieved occupancy、active warps/SM，以及 registers/shared memory/blocks 中实际 limiter；
- 内存：DRAM/L2 吞吐、load/store 效率以及主要 stall 原因；
- 计算：SM 吞吐、Tensor Core/FP32 利用情况和指令混合；
- launch：实际 Grid、block/warp 配置、架构家族（ordinary/persistent/split-K）与 kernel 总耗时。

诊断顺序：先确认精度、输入规格和 kernel 对应关系，再判断瓶颈，最后给出一个可验证的策略建议。低 occupancy 不自动等于性能问题；若 DRAM 或 Tensor Core 已接近相关硬件上限，应避免只为提高 occupancy 缩小 tile。Matmul 额外计算 `2*M*N*K/time`，峰值比例必须注明采用的**稠密** FP16/TF32 峰值来源，不能拿稀疏峰值或其它 GPU 数据代入。

**架构级强路由**：若 persistent matmul 出现 local spill 或接近 255 registers/thread，并且 register limiter 只允许约 1 block/SM或理论 occupancy 约不高于 25%，同时吞吐仍明显未饱和，则首选 `gen-autotune-config`，报告写 `architecture_reselect_required=true`、`force_non_persistent=true`。必须先独立重调普通完整 Grid；K 足够大且并行不足时写 `consider_split_k=true`。此时不得先用 `config-tuner` 在原 persistent 家族内消耗所有深度优化轮次。

未触发上述组合证据时，资源压力可先由 `config-tuner` 小范围联合调整 BLOCK/warps/stages。单独看到 occupancy 低、单独看到寄存器高，均不足以断言 persistent 必然更慢；最终仍由同一 RTX 3090 上的正确性和统一计时决定。

**优化建议**：

  - 优化策略 `config-tuner`
  - 优化策略 `gen-autotune-config`
  - 优化策略 `libdevice-opt`
  - 优化策略 `div-to-mul`

### Step 3：生成性能分析报告

根据以上信息，生成性能分析报告，并存储在 `{output_dir}/report.md`。

性能分析报告模版见：[`report_template.md`](references/report_template.md)
