---
name: mlu-bangc-optimize
description: 面向 MLU590 的 BANG C/CNRT 算子性能优化工作流。用于优化完整 .mlu 实现，依次执行片上分块、归约、任务规模、索引和离线配置搜索，再依据 notifier、CNPerf 与 CNCC/MLISA 证据做受控迭代；只保留在真实 MLU590 上精度通过且无性能回退的候选。
---

# mlu-bangc-optimize

## 概述

优化一个完整、可独立编译运行的 BANG C `.mlu` 算子文件。流程保持“输入检查与 baseline → 五个开箱策略 → 性能证据驱动迭代 → 汇总输出”的 v1 结构。平台事实、编译环境、原语与采集工具统一读取 `.claude/skills/share/mlu/`；本 Skill 不复制或臆测硬件常量。

没有真实 MLU590 执行证据时可以静态分析，但不得声称性能提升，不得填写推测的耗时、带宽、片上容量、Core/Cluster 数、架构 flag、CNPerf 指标或 MLISA 结论。

## 用法与交接

```text
/mlu-bangc-optimize <bangc_code_path> [output_dir]
```

| 参数 | 说明 |
| --- | --- |
| `bangc_code_path` | 完整 `.mlu` 文件；主链路通常传入 `KernelGen/bangc_code_fix.mlu` |
| `output_dir` | 可选；默认当前工作流输出目录 |

输入必须包含：

1. BANG C Kernel 入口。
2. Host CNRT wrapper、任务规模、function type 与 Queue launch。
3. 可自动失败的 correctness 测试和独立 reference。
4. notifier 或其他经确认的可重复 device 计时。
5. CNRT 调用、launch 和 Queue 完成检查。

固定最终产物：

```text
{output_dir}/Optimizer/bangc_optimized.mlu
{output_dir}/Optimizer/bangc_optimized.md
```

主流程随后将选中源码复制为 `bangc_final.mlu`。

## 全局执行契约

### 环境与编译

从当前策略工作目录向上读取最近的 `{output_dir}/EnvConfig/config.md`。动态编译、运行、精度、notifier 与 profiling 必须沿用其中同一个 `execution_backend` 和完整 `cncc` 命令。

- 架构参数只使用 EnvConfig、显式配置、`cncc --help` 或当前官方构建脚本确认值。
- 片上容量、对齐、任务规模和 function type 限制只使用设备探测、头文件、编译器或共享平台规则确认值。
- `bang.h` 的位置不假定为 `${NEUWARE_HOME}/include`；由 EnvConfig 验证 `cncc` 实际解析路径。
- 使用 C++ 标准库的 harness 必须沿用已验证的 CNRT/C++/math/thread 链接项。
- baseline 与 candidate 使用相同 compiler、flags、include/lib 顺序、输入、Queue、warmup、repeat 和计时范围。编译 flag 本身是实验变量时单独记录。

### Baseline

进入首个策略前保存只读 baseline：

- 源码哈希与完整编译命令。
- MLU 型号、驱动、NeuWare/CNToolkit、`cncc` 与目标 flag。
- Shape、dtype、stride/layout、seed、容差、Queue、warmup、repeat。
- `accuracy_pass`、最大绝对/相对误差、NaN/Inf mismatch。
- notifier 的样本、median、p20、p80；未取得时写 `N/A`。
- 编译器片上资源诊断、CNPerf 原始输出和 CNCC/MLISA 产物；不可用写 `N/A`。

性能字段统一为：

```text
host_reference_ms
original_bangc_ms
opt_bangc_ms
```

Host reference 时间仅用于背景，不作为设备 Kernel keep 门禁。

### 候选保留门禁

候选只有同时满足以下条件才能进入下一策略：

1. 使用相同构建契约编译成功。
2. 所有 correctness 用例通过，未修改 reference、输入或容差。
3. CNRT、launch、Queue、搬运与资源生命周期无新增失败。
4. 真实 MLU590 上以相同 notifier 口径测量，`opt_bangc_ms` 无回退且收益超出预先确定的噪声阈值。
5. 变更属于本策略，一轮只改变一个可归因因素。

目标设备或可靠计时不可用时，性能状态为 `not_measured`，最终选择原始或最后已验证 best-so-far；不能把静态候选标为性能赢家。

## 步骤 1：输入与 baseline

1. 检查 `.mlu` 的 Kernel、CNRT launch、correctness、benchmark 与错误处理。
2. 读取共享平台、原语与数学文档。
3. 按 EnvConfig 编译运行 baseline。
4. 失败属于代码时先调用 `mlu-bangc-code-review`；环境失败则停止。
5. 固化 baseline 源码、报告和机器可读结果，策略不得覆盖。

## 步骤 2：开箱优化

严格串行执行：

| 序号 | 策略目录 | BANG C 目标 |
| ---: | --- | --- |
| 1 | `retiling` | GDRAM↔NRAM/WRAM/SRAM tile、向量长度、buffer 复用与流水 |
| 2 | `reduce-opt` | 片上局部归约、跨 Task 合并、workspace/合法原子与 layout 融合 |
| 3 | `modify-grid` | `cnrtDim3_t`、function type、task ID 映射与完整覆盖 |
| 4 | `index-computation-simplify` | 指针/stride/线性化、循环不变量和 div/mod 简化 |
| 5 | `gen-autotune-config` | 外部 `cncc` 编译运行候选，冻结单一最佳配置 |

每轮目录保持 v1 拓扑：

```text
{output_dir}/Optimizer/{序号}_{策略名}/
├── input.mlu
├── candidate.mlu
├── bangc_optimized.mlu
├── bangc_optimized.md
└── result.json
```

调用策略 subagent：

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
读取 .claude/skills/mlu-bangc-optimize/utils/Optimizer.md，执行指定策略。
策略名称：{strategy_name}
策略文档：.claude/skills/mlu-bangc-optimize/{strategy_name}/strategy.md
工作目录：{strategy_workdir}
严格遵守相同编译契约、精度门禁、真实 MLU590 benchmark 与 no-regression 回退。
""",
)
```

缺少任一固定产物最多重试两次；仍失败则把 `input.mlu` 逐字复制为 `bangc_optimized.mlu`，生成最小失败报告。完成后汇总为 `bangc_oob_optimized.mlu/.md`。

## 步骤 3：性能分析驱动迭代

仅在真实 MLU590、baseline 正确且性能工具链可用时进入。每轮：

1. 上一轮 best-so-far 复制为 `input.mlu`。
2. 按 `perf-analyzer/strategy.md` 采集 notifier、CNPerf 原始报告和可用的 CNCC/MLISA 产物。
3. 依据证据只选一个策略：
   - `libdevice-opt`：读取共享 `optimize/libdevice-opt.md`。
   - `config-tuner`：调整 tile、任务规模/function type、向量长度、buffer/流水配置。
   - `div-to-mul`：仅处理语义允许且证据显示为热点的除法。
4. 生成独立候选，执行统一门禁；失败或无收益则回退。
5. 只有 keep 后才重新 profile。

终止条件：连续三轮无超噪声收益；报告无可执行建议；建议均被正确性/资源/no-regression 门禁回退；达到用户预算。

汇总所有有效轮次，选择最低设备 median 的 best-so-far，而非自动选择最后一轮。输出 `bangc_advanced_optimized.mlu/.md`。

## 步骤 4：最终输出

1. 深度优化存在通过门禁的更优 best-so-far：选择 `bangc_advanced_optimized.mlu`。
2. 深度优化未执行或无更优候选：选择 `bangc_oob_optimized.mlu`。
3. 无可靠性能测量或链路不确定：选择原始输入/最后已验证版本。
4. 将选中版本逐字复制为 `bangc_optimized.mlu`，汇总为 `bangc_optimized.md`。

## 报告最小字段

```markdown
## BANG C 优化结果

### 环境
- execution_backend:
- device / driver / NeuWare:
- cncc / compile_command / arch_flag:

### 正确性
- accuracy_pass:
- atol / rtol:
- max_abs_error / max_rel_error:

### 性能
| 版本 | device median | p20 | p80 | 带宽 | 相对 baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | | | | | 1.0x |
| selected | | | | | |

- host_reference_ms:
- original_bangc_ms:
- opt_bangc_ms:

### 资源与证据
- NRAM/SRAM/WRAM: [compiler/CNPerf evidence | N/A]
- CNPerf: [raw report path | N/A]
- MLISA/intermediates: [paths | N/A]

### 决策
- selected_strategy:
- decision: keep | revert | not_measured
- final_code: {output_dir}/Optimizer/bangc_optimized.mlu
```

未知值写 `N/A`/`null` 并解释原因，禁止填 0 或经验估计。
