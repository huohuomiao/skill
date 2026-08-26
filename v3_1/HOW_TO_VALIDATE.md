# v3_1 验证说明

v3_1 的验证分为离线门禁、优化控制器命令验证、预算恢复负向验证、实际产物检查和 MLU 集成回归。完整设计见 [STAGE3_OPTIMIZATION_MANUAL.md](STAGE3_OPTIMIZATION_MANUAL.md)。

## 1. 环境

使用 Python 3.9 或更高版本。离线验证只使用标准库，不需要 PyYAML、pytest 或 jsonschema。

所有命令从 v3_1 根目录执行，并使用 `-B` 避免生成 `__pycache__`：

```bash
python -B validation/validate.py all
```

系统中的 `python` 若不是实际解释器，请替换成可用解释器的绝对路径。

## 2. L1 静态检查

```bash
python -B validation/validate.py l1
```

L1 检查：

- 四个 Skill 的 frontmatter、名称和描述。
- Python AST、Markdown 围栏、文件非空和硬编码路径。
- v2 的 P0 契约修复是否仍然成立。
- 三种优化模式是否存在。
- Main 是否支持 correctness 跳过优化。
- Optimizer 是否强制计划、状态、预算和最佳候选选择。
- `optimization_control.py` 是否存在且语法正确。
- Step 1～4、Optimization Plan、Optimization State Schema 是否可解析。

## 3. L2 离线行为检查

```bash
python -B validation/validate.py l2
```

L2 会直接加载控制器函数，但不启动 Agent、不连接 Worker、不需要 MLU。覆盖：

- correctness 不选 OOB、不启用高级优化。
- balanced 对 Reduce 命中 `retiling` 和 `reduce-opt`。
- 单维且显式限核的 Grid 跳过 `modify-grid`。
- max-performance 对数学函数、Block 参数和除法产生高级候选。
- 非 Reduce 算子跳过 `reduce-opt`。
- 未限核 Grid 命中 `modify-grid`。
- 不完整 Triton 输入触发 `manual_review_required`。
- Subagent 达到上限后停止。
- 连续低收益触发 patience。
- 达到提升阈值后 patience 归零。
- 首次 plan 可以初始化，普通 plan 不能覆盖已有状态。
- `plan --resume` 保留已消耗预算、patience 和 history。
- 输入哈希变化或 plan/state 不完整时恢复失败。
- 正向和负向 JSON fixture。

## 4. 日常最小验证

每次修改 Skill、控制器、策略或预算后至少运行：

```bash
python -B validation/validate.py all
```

`[PASS] L1` 和 `[PASS] L2` 必须同时出现。

## 5. 手工检查策略计划

对任意完整 Triton 文件生成计划：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> \
  --output-dir <临时输出目录>/Optimizer \
  --mode balanced
```

检查：

- `optimization_plan.json` 的 `selected_oob_strategies` 是否符合算子特征。
- `optimization_state.json.usage` 是否全部从 0 开始。
- `manual_review_required` 是否为 false。
- plan 和 state 的 mode、limits 是否完全一致。

检查预算：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py check \
  --state <临时输出目录>/Optimizer/optimization_state.json \
  --phase oob
```

记录一次模拟 Subagent：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py record \
  --state <临时输出目录>/Optimizer/optimization_state.json \
  --event subagent --phase oob --strategy reduce-opt
```

验证预算恢复：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> \
  --output-dir <同一个临时输出目录>/Optimizer \
  --mode balanced \
  --resume
```

检查：

- 输出包含 `"resumed": true`。
- `optimization_state.json.usage`、`advanced_no_improvement`、`best`、`stop_reason`、`history` 与恢复前完全一致。
- 对已有状态再次执行不带 `--resume` 的普通 plan，退出码必须为 `2`。
- 修改输入、模式或预算后执行 `--resume`，退出码必须为 `2`。

## 6. 验证真实工作流产物

```bash
python -B validation/validate.py artifacts \
  --output-dir <output_mlu_triton_main>
```

完整交付检查：

```bash
python -B validation/validate.py artifacts \
  --output-dir <output_mlu_triton_main> \
  --require-complete
```

检查包括：

- KernelGen Step 1～4 JSON Schema。
- Optimization Plan 和 State Schema。
- Plan/State 的 mode 和 limits 一致。
- `selected_oob_strategies` 与 OOB 明细一致。
- Code Gen IO Shapes 一致。
- 完整代码、报告和 summary 非空。
- correctness 模式允许没有 Optimizer 目录。

## 7. L3 本地 MLU 验证

环境检查：

```bash
python -B validation/run_mlu_integration.py
```

算子精度和性能检查：

```bash
python -B validation/run_mlu_integration.py \
  --operator <triton_final.py>
```

## 8. Worker 模式

先把整套 v3_1 部署到目标项目 `.claude/skills`，在同一个 `JOB_ID` 中检查环境：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
  --task-type custom \
  --workdir <目标项目根目录绝对路径> \
  --timeout-sec 600 \
  --command "python .claude/skills/share/mlu/runtime/get_device_info.py && python .claude/skills/share/mlu/runtime/test_env_code.py"
```

优化阶段的 Worker Task 必须在提交前执行预算 `check` 和 `record --event worker`。EnvConfig 与 Code Gen 的 Worker 调用不计入优化状态。

## 9. 三模式回归矩阵

| 用例 | correctness | balanced | max-performance |
|---|---|---|---|
| Elementwise | 直接交付 Code Gen | 不应运行 Reduce 策略 | 高级候选取决于数学、除法和 Block 信号 |
| Reduce | 直接交付 Code Gen | 应命中 retiling/reduce-opt | 应在预算内进入 perf 分析 |
| 已限核单维 Grid | 不优化 | 应跳过 modify-grid | 应跳过 modify-grid |
| 未限核 Grid | 不优化 | 应命中 modify-grid | 应命中 modify-grid |
| exp/log | 不优化 | 只执行命中的 OOB | 应产生 libdevice-opt 候选 |
| 除法 | 不优化 | 只执行命中的 OOB | 应产生 div-to-mul 候选 |

每个用例记录：

- accuracy_pass、atol、rtol、max_diff。
- 总耗时。
- Subagent、Worker 和策略尝试次数。
- selected/skipped 策略及原因。
- stop_reason。
- v3/v3_1 最终性能，以及预算恢复前后的 usage。

## 10. 验收标准

- L1、L2 全部通过。
- correctness 不创建 Optimizer 产物，最终代码来自 KernelGen。
- balanced 不执行任何未选策略，也不进入高级优化。
- max-performance 不超过预算中的任一上限。
- 重试/中断恢复不清零或回退任何预算计数。
- 已有状态不能被普通 plan 覆盖，不兼容状态不能被 `--resume` 接受。
- 精度失败候选不会成为最终代码。
- 缺失性能数据不会被当作提升。
- 实际产物通过 `artifacts --require-complete`。
- 至少完成一个 Elementwise 和一个 Reduce 的真实 MLU 回归。
