---
name: mlu-triton-optimize
description: NVIDIA CUDA GPU（重点适配 RTX 3090 / sm_86）上的 Triton 算子性能优化专家；沿用 mlu-triton-optimize 名称以兼容原调用，当用户要求 GPU、CUDA、RTX 3090 或 Triton 算子优化时调用
---

# mlu-triton-optimize

## 概述

该 SKILL 是 NVIDIA CUDA GPU 上的 Triton 算子性能优化专家，重点适配 RTX 3090（sm_86）。目录名、Skill 名和 `/mlu-triton-optimize` 命令仅作为旧版兼容标识保留，不代表 MLU 执行语义。

执行任何策略前先读取 `.claude/skills/share/gpu/references/platform-rules.md`；设备探测、硬件上限、CUDA 特性门控和 profiler 入口均以 `share/gpu` 为唯一事实源，禁止在本 Skill 内硬编码硬件细节。

**核心目标**：在保证精度无损的前提下，将指定的 Triton 算子性能提升到最优。

## 用法

```bash
/mlu-triton-optimize <triton_code> [output_dir]
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `triton_code` | 包含 Triton kernel 和测试代码的完整代码（文件路径或代码片段） |
| `output_dir` | 输出结果存储文件夹路径（可选，默认为当前目录下的 `output/`） |

**输入形式支持**：
   - 文件路径：指向 `.py` 文件的路径
   - 代码片段：直接提供的 Python 代码文本

## 工作流程

### 步骤 1：输入代码检查

检查输入 Triton 代码是否完整，必须包含以下四个部分，否则终止优化：

   - Triton Kernel：`@triton.jit` 装饰的函数
   - wrapper函数：包含 grid 配置和参数传递逻辑
   - 精度测试代码
   - 性能测试代码

### 步骤 2：开箱性能优化（Out-of-Box Performance Optimization）

开箱性能优化：基于 Triton CUDA 优化经验，采用结构化匹配方式，将 Triton 代码优化为 NVIDIA GPU 友好的代码，追求用最小代价获得性能良好的 RTX 3090 Triton 代码。

开箱性能优化不过分追求单步收益，但每个策略仍须在统一测试阶段验证；无收益或回退的候选必须恢复该策略输入，不能把回退带入下一步。

以下为所有的开箱性能优化策略：
| 序号 | 策略 | 策略名称 | 说明  | 是否必须执行 |
|--------|------|-------------|------|------------------|
| 1 | 分块优化 | `retiling` | 分析 Triton kernel 代码，优化归约轴和并行轴的分块方案 | 否 |
| 2 | 归约优化 | `reduce-opt` | 分析 Triton kernel 代码，对包含 reduce 类算子的 kernel 进行优化 | 否 |
| 3 | Grid 优化 | `modify-grid` | 保留普通 CUDA launch 并按需展平；仅对实测有益的 persistent 候选按编译后 occupancy 限流 | 是 |
| 4 | 索引计算简化 | `index-computation-simplify` | 消除 load/store 地址计算中的冗余计算 | 否 |
| 5 | 自动调优配置生成 | `gen-autotune-config` | 分析可调优的配置轴，生成带有单一最优配置项 autotune 的代码 | 是 |

#### 2.1 顺序执行各优化策略

严格按照序号，串行执行上述所有开箱性能优化策略，单个优化策略执行步骤如下：

1. 开始 `当前优化策略`

2. 创建 `当前优化策略` 工作目录：`mkdir -p {output_dir}/Optimizer/{当前策略序号}_{当前策略名称}/`

3. 复制输入代码：将 `上一轮优化策略` 输出的 `triton_optimized.py` 复制到 `当前优化策略` 工作目录并改名为 `input.py` (第一个策略使用步骤 1 解析的输入代码)：

   ```bash
   cp {上一轮优化策略输出代码路径} {output_dir}/Optimizer/{当前优化策略序号}_{当前策略名称}/input.py
   ```

4. **重点要求**：分发给 subagent 执行 `当前优化策略`（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
   ## 任务文档
   根据 .claude/skills/mlu-triton-optimize/utils/Optimizer.md 中的说明，作为优化器，执行 {`本轮优化策略`}。

   ## 用户输入
   策略名称：{当前策略名称}
   策略文档路径：.claude/skills/mlu-triton-optimize/{`当前优化策略`名称}/strategy.md
   工作目录：{`当前优化策略` 工作目录}

   严格按照任务文档要求执行。
   """
)
```

**subagent 输入输出说明**：

   - 优化策略输入代码：`{工作目录}/input.py`（主流程在调用前复制）
   - 优化策略输出代码：`{工作目录}/triton_optimized.py`（subagent 生成）
   - 优化策略优化报告：`{工作目录}/triton_optimized.md`（subagent 生成）

**输出校验与重试**：

每个 subagent 执行完毕后，主流程必须检查其工作目录是否包含以下两个输出文件：

   - `{工作目录}/triton_optimized.py`（优化后代码）
   - `{工作目录}/triton_optimized.md`（优化报告）

若任一文件缺失，则重新调用 `当前优化策略` 的 subagent，最多重试 2 次。若重试后仍未生成，则标记该策略为失败，将 `{当前优化策略工作目录}/input.py` 复制为 `{当前优化策略工作目录}/triton_optimized.py`，并生成最小 `triton_optimized.md`，明确 `status=failed`、`accuracy_pass=N/A`、`performance=N/A` 与真实失败原因，保证汇总链不断。

6. 第 5 步验证通过后，继续进行下一轮优化策略，直至所有优化策略顺序执行完毕。

#### 2.2 汇总优化结果

开箱优化所有策略执行完毕后，读取每个策略的优化报告 `{output_dir}/Optimizer/{策略序号}_{策略名称}/triton_optimized.md`，提取以下信息并记录到`优化策略执行进度清单`：

   - **状态**：成功 / 失败
   - **精度是否通过**：`accuracy_pass`
   - **性能数据**：`opt_triton_ms`、`speedup_opt_vs_original`、`speedup_opt_vs_torch`

提取完毕将`优化策略执行进度清单`写入 `{output_dir}/Optimizer/triton_oob_optimized.md`。

**生成开箱优化代码**：

   - 开箱优化的产物为所有开箱优化策略顺序执行完毕之后的输出代码，即最后一项优化策略输出代码
   - 若最后一项优化策略输出代码缺失，则向前依次查找最近一个存在输出代码的优化策略
   - 若所有优化策略均无输出代码，则使用步骤 1 的原始输入代码

将开箱优化代码复制到 `{output_dir}/Optimizer/triton_oob_optimized.py` 。

### 步骤 3：深度性能优化（Advanced Performance Optimization）

深度性能优化：基于实测 perf 数据对开箱优化后的 Triton kernel 进行微调，通过不断迭代优化达到极致性能。

深度性能优化专注于性能提升，要求每一步修改都必须有性能的提升。

深度性能优化先通过 perf-analyzer 收集目标 Triton 代码的 perf 数据，并获取调优建议，并按照调优建议调用以下可用优化策略：

|      策略      |     策略名称     |                            说明                            |
|---------------|-----------------|------------------------------------------------------------|
| Libdevice 优化 | `libdevice-opt` | 使用官方 `triton.language.extra.libdevice` 或标准 `tl` 原语替换低效计算模式 |
| Config 微调    | `config-tuner`  | 通过调整 block size，num_stages，num_warps 等参数进行性能调优   |
| 除法指令优化 | `div-to-mul` | 将除法指令修改为乘倒数 |

#### 3.1 创建工作目录

创建深度性能优化总的工作目录：`mkdir -p {output_dir}/Optimizer/Advanced_Optimization/`

#### 3.2 迭代优化

1. 开始第 `i` 轮迭代优化

2. 创建 `本轮迭代优化` 工作目录：`mkdir -p {output_dir}/Optimizer/Advanced_Optimization/iter_{i}/`
3. 复制输入代码：将 `上一轮迭代优化` 输出的 `triton_optimized.py` 复制到 `本轮迭代优化` 工作目录并统一改名为 `input.py` (首轮迭代使用 `{output_dir}/Optimizer/triton_oob_optimized.py`)：

   ```bash
   cp {上一轮迭代输出代码路径} {output_dir}/Optimizer/Advanced_Optimization/iter_{i}/input.py
   ```

4. **重点要求**：分发给 subagent 执行 `本轮迭代优化` 的性能分析 perf-analyzer 任务（禁止主流程接管此任务）:

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
   ## 任务文档
   根据 .claude/skills/mlu-triton-optimize/perf-analyzer/strategy.md 中的描述，完成性能分析任务。

   ## 用户输入
   性能分析名称：perf-analyzer
   性能分析文档路径：.claude/skills/mlu-triton-optimize/perf-analyzer/strategy.md
   input_file：{output_dir}/Optimizer/Advanced_Optimization/iter_{i}/input.py
   output_dir：{output_dir}/Optimizer/Advanced_Optimization/iter_{i}/

   严格按照任务文档要求执行。
   """
)
```

5. 选取 `本轮优化策略`

读取 perf-analyzer 的性能分析结果，从优化建议中选中 `本轮优化策略`。

当同时给出多个优化建议，只选中其中一个作为 `本轮优化策略`。（优先选择之前轮次中未被选中的优化策略）

6. **重点要求**：分发给 subagent 执行 `本轮优化策略`（禁止主流程接管此任务）:

先解析策略文档路径：

- `libdevice-opt`：`.claude/skills/share/gpu/optimize/libdevice-opt.md`
- 其他策略：`.claude/skills/mlu-triton-optimize/{本轮优化策略名称}/strategy.md`

所有 CUDA GPU 策略还必须读取 `.claude/skills/share/gpu/references/platform-rules.md`。RTX 3090 禁用 FP8、TMA、thread-block cluster 以及仅 Hopper 可用的路径。

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
   ## 任务文档
   根据 .claude/skills/mlu-triton-optimize/utils/Optimizer.md 中的说明，作为优化器，执行 {`本轮优化策略`}。

   ## 用户输入
   策略名称：{`本轮优化策略`名称}
   策略文档路径：{按上述规则解析后的策略文档路径}
   工作目录：{`本轮迭代优化` 工作目录}

   严格按照任务文档要求执行。
   """
)
```

7. 确保当前 subagent 执行完毕
查看 `本轮迭代优化` 工作目录下是否包含以下两个输出文件：

- `{output_dir}/Optimizer/Advanced_Optimization/iter_{i}/triton_optimized.py`（优化后代码）
- `{output_dir}/Optimizer/Advanced_Optimization/iter_{i}/triton_optimized.md`（优化报告）

若任一文件缺失，则重新调用 `本轮优化策略` 的 subagent，最多重试 2 次。若重试后仍未生成，则标记该策略为失败，将 `{本轮迭代优化工作目录}/input.py` 复制为 `{本轮迭代优化工作目录}/triton_optimized.py`。

8. 第 7 步验证通过后，继续第 `i+1` 轮迭代，**直至达成任一优化终止条件**

**迭代优化终止条件**：

   - 连续 3 次迭代优化性能无提升
   - 本轮性能分析报告中无任何优化建议

#### 3.3 汇总优化结果

深度优化迭代完毕后，按顺序读取`{output_dir}/Optimizer/Advanced_Optimization` 下所有迭代轮次的优化报告，提取以下信息并记录到`优化策略执行进度清单`：

   - **状态**：成功 / 失败
   - **精度是否通过**：`accuracy_pass`
   - **性能数据**：`opt_triton_ms`、`speedup_opt_vs_original`、`speedup_opt_vs_torch`

提取完毕将`优化策略执行进度清单`写入 `{output_dir}/Optimizer/triton_advanced_optimized.md`。

**生成深度优化代码**：

   - **最后一轮迭代**的输出代码即为深度优化代码
   - 若最后一轮迭代输出代码缺失，则前一轮次的输出代码即为深度优化代码
   - 若所有轮次优化均无输出代码，则使用步骤 1 的原始输入代码

将深度优化代码复制到 `{output_dir}/Optimizer/triton_advanced_optimized.py` 。

### 步骤 4：优化输出

1. 确定最终代码：
   - 若步骤 3 有执行，使用 `{output_dir}/Optimizer/triton_advanced_optimized.py`
   - 若步骤 3 被跳过，使用 `{output_dir}/Optimizer/triton_oob_optimized.py`
   - 若上述代码均缺失，则使用步骤 1 的原始输入代码
2. 将最终代码复制到 `{output_dir}/Optimizer/triton_optimized.py` 中作为输出代码
3. 读取各阶段汇总报告（`triton_oob_optimized.md`、`triton_advanced_optimized.md`），合并生成全局汇总报告 `{output_dir}/Optimizer/triton_optimized.md`


## 报告格式

```
## 优化结果报告

### 算子信息
- 源文件：<file_path>
- 算子名称：<算子名>

### 开箱优化（OOB）执行记录
1. [策略名称] - 执行结果：成功 / 失败
2. ...

### 深度优化（Perf）执行记录
迭代 1：
1. [策略名称] - 执行结果：成功 / 失败
2. ...
迭代 2：
...

### 性能对比
| PyTorch耗时 | 优化后耗时 | PyTorch带宽 | 优化后带宽 | 相对PyTorch加速比 |
|-------------|-----------|-------------|-----------|------------------|
| ...ms | ...ms | ...GB/s | ...GB/s | ...x |

### 最终代码路径
{output_dir}/Optimizer/triton_optimized.py

### 优化报告
{output_dir}/Optimizer/triton_optimized.md
```
