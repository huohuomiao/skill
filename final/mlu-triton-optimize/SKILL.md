---
name: mlu-triton-optimize
description: "Optimize, resume, budget, and select the best validated Triton kernel on Cambricon MLU. Use when a complete reviewed Triton operator needs correctness, balanced, or max-performance mode with static strategy routing, real MLU evidence, and bounded Subagent/Worker attempts."
---

# mlu-triton-optimize

## 目标

在保持精度的前提下优化包含 Kernel、Wrapper、精度测试和性能测试的 MLU Triton 文件。先用确定性脚本生成策略计划，再只执行被选中的策略；任何优化动作都必须受同一个优化阶段预算约束。

详细的路由信号、预算字段、状态机和命令读取 [references/optimization-control.md](references/optimization-control.md)。

## 输入

```text
/mlu-triton-optimize <triton_code> [output_dir] [mode] [budget_file]
```

| 参数 | 说明 |
|---|---|
| `triton_code` | 完整 `.py` 文件路径；代码片段应先落盘 |
| `output_dir` | 输出根目录，默认当前目录下的 `output` |
| `mode` | `correctness`、`balanced` 或 `max-performance`，默认 `balanced` |
| `budget_file` | 可选 JSON 文件，只覆盖指定预算字段 |

## 模式

| 模式 | OOB 策略 | 深度优化 | 适用场景 |
|---|---|---|---|
| `correctness` | 不执行 | 不执行 | 只需要正确代码，或单独调用本 Skill 但不希望改写代码 |
| `balanced` | 只执行静态路由命中的策略 | 不执行 | 默认模式，控制时间与 Token |
| `max-performance` | 只执行静态路由命中的策略 | 在预算内执行 | 明确要求极致性能或深度调优 |

用户显式模式优先。未指定时使用 `balanced`。不得因为存在性能测试代码而自动升级为 `max-performance`。

## 不变量

- 精度未通过的候选代码不得成为下一阶段输入或最终代码。
- 不得执行 `optimization_plan.json` 中 `selected=false` 的 OOB 策略。
- 不得执行 `advanced.candidates` 之外的高级策略。
- 每次 Subagent 或 Worker 调用前都必须先执行预算 `check`；不允许时立即停止新增动作。
- Worker 预算只约束优化阶段，EnvConfig 和 Code Gen 的 Worker 调用不计入此状态文件。
- 高级候选只有在精度通过且提升达到 `min_improvement_pct` 时才重置 patience。
- 最终选择 `optimization_state.json.best.artifact_path` 中存在且精度通过的最佳候选；否则回退最近一个精度通过的 OOB 产物，再否则回退原始输入。
- 禁止用 CPU、PyTorch 或标量 Kernel 替换 Triton Kernel 来通过测试。
- 已存在 `optimization_plan.json` / `optimization_state.json` 时禁止再次初始化；中断恢复必须验证输入哈希、模式、预算和策略路由一致，并保留累计用量、最佳候选、patience 与历史。

## 步骤 1：输入检查

确认输入文件存在且至少包含：

- `@triton.jit` Kernel。
- Kernel launch 和 Wrapper。
- 精度测试。
- 性能测试。

缺少任一部分时停止，不生成性能结论。代码真实运行遵守 `mlu-triton-main` 的 EnvConfig 后端选择规则。

## 步骤 2：生成优化计划与预算状态

首次进入、两个状态文件都不存在时，创建 `{output_dir}/Optimizer` 并运行：

```bash
python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <triton_code绝对路径> \
  --output-dir <output_dir>/Optimizer \
  --mode <mode> \
  [--budget-file <budget_file绝对路径>]
```

若外层 `run_manifest.json` 表明 `optimizer` 是中断/失败后重开，且两个状态文件都已存在，则改为：

```bash
python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <triton_code绝对路径> \
  --output-dir <output_dir>/Optimizer \
  --mode <mode> \
  [--budget-file <budget_file绝对路径>] \
  --resume
```

`--resume` 只在现有 plan/state 的输入 SHA-256、模式、预算限制、策略路由和版本全部兼容时成功，且不会改写状态。只有一个状态文件、校验失败或不带 `--resume` 覆盖已有状态都必须停止；不得通过删除/重建状态绕过全局预算。

必须得到：

- `{output_dir}/Optimizer/optimization_plan.json`
- `{output_dir}/Optimizer/optimization_state.json`

若 `manual_review_required=true`，说明脚本未识别到完整 Kernel 或 launch。停止优化并报告，不允许退化为“全部策略都跑”。

`correctness` 模式直接把原始输入复制到 `Optimizer/triton_optimized.py`，并生成说明“未执行优化”的 `Optimizer/triton_optimized.md`，然后结束。

## 步骤 3：按需执行 OOB 策略

从 `optimization_plan.json.oob_strategies` 读取 `selected=true` 的项目，按原顺序串行执行。未选策略只记入汇总报告，不创建策略工作目录。

每个选中策略执行以下流程：

1. 检查预算：

   ```bash
   python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py check \
     --state <output_dir>/Optimizer/optimization_state.json \
     --phase oob
   ```

   退出码 `3` 表示预算终止；记录 `stop_reason` 并跳到步骤 5。

2. 创建 `{output_dir}/Optimizer/{order}_{strategy}/`，把上一轮通过精度的代码复制为 `input.py`。

3. 记录一次 Subagent 调用：

   ```bash
   python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py record \
     --state <output_dir>/Optimizer/optimization_state.json \
     --event subagent --phase oob --strategy <strategy>
   ```

4. 分发给 Subagent：

   ```python
   agent = spawn_agent(
       agent_type="default",
       message=f"""
       根据 .claude/skills/mlu-triton-optimize/utils/Optimizer.md 执行单个优化策略。

       策略名称：{strategy}
       策略文档：{strategy_path}
       工作目录：{strategy_workdir}
       优化计划：{output_dir}/Optimizer/optimization_plan.json
       预算状态：{output_dir}/Optimizer/optimization_state.json
       优化阶段：oob

       只执行指定策略；若需要 Worker，调用前必须按 Optimizer.md 记录 Worker 预算。
       """
   )
   ```

   `libdevice-opt` 文档仍位于 `.claude/skills/share/mlu/optimize/libdevice-opt.md`；其他 OOB 策略位于本 Skill 对应目录。

5. 校验 `triton_optimized.py` 和 `triton_optimized.md`。缺失时可重试，但每次重试仍需重新 `check` 并记录 Subagent；预算不允许时不得重试。

6. 从报告取得 `accuracy_pass`、`original_triton_ms` 和 `opt_triton_ms`，记录策略结果：

   ```bash
   python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py record \
     --state <output_dir>/Optimizer/optimization_state.json \
     --event strategy --phase oob --strategy <strategy> \
     --accuracy-pass <true|false|unknown> \
     [--baseline-ms <original_triton_ms>] \
     [--candidate-ms <opt_triton_ms>] \
     [--artifact-path <triton_optimized.py绝对路径>]
   ```

7. 只有 `accuracy_pass=true` 才把候选作为下一策略输入；否则继续使用上一轮通过精度的代码。

所有选中策略完成后，将最近一个通过精度的候选复制为 `Optimizer/triton_oob_optimized.py`，并将计划中的“执行/跳过原因”写入 `Optimizer/triton_oob_optimized.md`。

## 步骤 4：预算内深度优化

仅当 `mode=max-performance` 且 `optimization_plan.json.advanced.enabled=true` 执行。

每轮流程：

1. 对 `phase=advanced` 执行预算 `check`。退出码 `3` 立即结束。
2. 记录一次 perf-analyzer Subagent 调用，运行性能分析。
3. 将性能分析建议与 `advanced.candidates` 取交集；交集为空时以 `no_applicable_advanced_strategy` 结束。
4. 每轮只选择一个未优先尝试的候选策略。
5. 再次检查预算、记录优化 Subagent 调用并执行策略。
6. 测试精度和性能；任何缺失数据按未提升处理，不得推测。
7. 记录 `strategy` 事件，再记录本轮 `advanced_iteration`：

   ```bash
   python .claude/skills/mlu-triton-optimize/scripts/optimization_control.py record \
     --state <state_path> --event advanced_iteration --phase advanced \
     --strategy <strategy> --accuracy-pass <true|false|unknown> \
     [--baseline-ms <baseline_ms>] [--candidate-ms <candidate_ms>] \
     [--artifact-path <candidate_path>]
   ```

8. 精度通过且提升达到阈值时进入下一轮；否则回退本轮输入。达到最大轮数、patience、墙钟时间、Subagent/Worker/策略次数任一上限即停止。

深度优化结束后，把状态中的最佳有效候选复制为 `Optimizer/triton_advanced_optimized.py`，并生成 `Optimizer/triton_advanced_optimized.md`。

## 步骤 5：最终选择与报告

最终代码选择顺序：

1. `optimization_state.json.best.artifact_path` 指向的有效最佳候选。
2. `Optimizer/triton_advanced_optimized.py`。
3. `Optimizer/triton_oob_optimized.py`。
4. 原始输入代码。

复制为 `{output_dir}/Optimizer/triton_optimized.py`。生成 `{output_dir}/Optimizer/triton_optimized.md`，至少包含：

- 模式和输入文件。
- 预算限制、实际用量、停止原因。
- 所有 OOB 策略的执行/跳过状态及原因。
- 深度优化每轮结果。
- 最佳候选、精度与性能证据。
- 最终代码路径。

缺失数据必须写 `N/A（原因：...）`，禁止伪造。

外层 Main 只有在以上四个最终产物均通过交接验证，且最终候选已经通过 L1/L2 与当前
`run_context` 下的 L3 精度/性能验证后，才允许使用三个 `--validation-level` 参数把
`optimizer` 标记为 `complete` 并写入内容寻址缓存。若整个外层阶段由缓存恢复，不得再次运行本 Skill，也不得增加预算用量。

## 默认预算

| 字段 | correctness | balanced | max-performance |
|---|---:|---:|---:|
| `max_wall_time_sec` | 0 | 1800 | 7200 |
| `max_subagent_calls` | 0 | 8 | 16 |
| `max_worker_calls` | 0 | 8 | 16 |
| `max_strategy_attempts` | 0 | 8 | 12 |
| `max_advanced_iterations` | 0 | 0 | 3 |
| `advanced_patience` | 0 | 0 | 2 |
| `min_improvement_pct` | 2.0 | 2.0 | 2.0 |

预算只允许显式 JSON 覆盖，不从自然语言猜测数值。所有预算都是硬上限；不得通过重启状态文件规避预算。
