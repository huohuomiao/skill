# Optimization Control Contract

本文件定义 `optimization_control.py` 的机器契约。只有在修改策略路由、预算或状态记录时读取；执行单个策略时只需读取计划中的本策略条目和当前状态。

## 1. 路由输入信号

| 信号 | 检测内容 | 影响 |
|---|---|---|
| `has_reduction` | `tl/torch.sum|max|min|argmax|argmin|cumsum|cumprod` | `retiling`、`reduce-opt` |
| `block_params` | `NAME: tl.constexpr` | `retiling`、`gen-autotune-config`、`config-tuner` |
| `has_grid` | Grid 定义或 Kernel launch | `modify-grid` |
| `has_multidimensional_grid` | 非零 `program_id` 轴或多维 Grid | `modify-grid` |
| `has_core_cap` | 明确的物理核心数上限变量 | 可能跳过 `modify-grid` |
| `has_index_div_or_mod` | `//` 或 `%` | `index-computation-simplify` |
| `has_autotune` | `@triton.autotune` | 已存在时跳过生成 Autotune |
| `has_libdevice_candidate` | 常用数学函数或已有 MLU Libdevice | `libdevice-opt` |
| `has_tensor_division` | 非整数除法 `/` | `div-to-mul` |

Reduce、数学、除法、索引和 Block 信号只扫描 `@triton.jit` 函数体；Grid、Launch 和物理核心数上限扫描完整文件。这样测试代码中的 Reference 表达式不会误触发 Kernel 优化。

路由是保守的候选筛选，不等价于优化一定生效。策略内部仍需检查适用条件和精度。

## 2. 模式路由

- `correctness`：所有 `oob_strategies.selected=false`，`advanced.enabled=false`。
- `balanced`：按信号选择 OOB，`advanced.enabled=false`。
- `max-performance`：按信号选择 OOB，`advanced.enabled=true`，高级候选由信号过滤。

如果没有识别到 `@triton.jit` 或 Kernel launch，计划标记 `manual_review_required=true`。调用方必须停止，不得用“全策略执行”兜底。

## 3. optimization_plan.json

稳定字段：

```json
{
  "version": 1,
  "mode": "balanced",
  "input_path": "/abs/path/kernel.py",
  "input_sha256": "...",
  "manual_review_required": false,
  "limits": {},
  "signals": {},
  "oob_strategies": [
    {"order": 1, "name": "retiling", "selected": true, "reason": "..."}
  ],
  "selected_oob_strategies": ["retiling"],
  "advanced": {
    "enabled": false,
    "reason": "balanced mode does not run advanced optimization",
    "candidates": []
  }
}
```

同一次运行不得手工编辑计划。输入文件改变时应重新开始一次新的优化运行，而不是沿用旧计划。

## 4. optimization_state.json

状态包含：

- `limits`：硬上限。
- `usage`：Subagent、Worker、策略尝试和高级迭代计数。
- `advanced_no_improvement`：连续未达到提升阈值的高级迭代数。
- `best`：最佳有效候选的耗时、路径和策略。
- `stop_reason`：第一次触发的终止原因。
- `history`：最多保留最近 200 个事件。

状态由脚本原子写入。不得由 Agent 直接改 JSON。

## 5. 预算检查顺序

每个新动作之前按顺序检查：

1. 已有 `stop_reason`。
2. 墙钟时间。
3. Subagent 次数。
4. Worker 次数。
5. 策略尝试次数。
6. 高级阶段最大轮数。
7. 高级阶段 patience。

任一项达到上限即停止。采用 `usage >= limit`，因此 limit 为 0 表示不允许第一次动作。

## 6. 事件计数时机

| 事件 | 记录时机 |
|---|---|
| `subagent` | 启动 Subagent 之前 |
| `worker` | 提交 Worker Task 之前 |
| `strategy` | 一次策略尝试得到结果后；失败和重试都计数 |
| `advanced_iteration` | 一轮高级优化完成后 |

如果动作在记录后因基础设施失败，计数仍然保留，因为资源已经消耗。

## 7. 高级候选接受规则

计算：

```text
improvement_pct = (baseline_ms - candidate_ms) / baseline_ms * 100
```

只有同时满足以下条件才视为有效提升：

- `accuracy_pass=true`
- baseline 与 candidate 都是实际测量值
- `baseline_ms > 0`
- `improvement_pct >= min_improvement_pct`

否则增加 `advanced_no_improvement` 并回退本轮代码。缺失性能数据不是有效提升。

## 8. 停止原因

脚本生成的预算停止原因包括：

- `max_wall_time_reached`
- `max_subagent_calls_reached`
- `max_worker_calls_reached`
- `max_strategy_attempts_reached`
- `max_advanced_iterations_reached`
- `advanced_patience_exhausted`

流程层可以补充：

- `manual_review_required`
- `no_selected_oob_strategy`
- `no_applicable_advanced_strategy`
- `accuracy_failure`
- `infrastructure_failure`

补充原因写入最终报告，不直接改写状态文件中的脚本字段。

## 9. 预算覆盖文件

覆盖文件只包含需要修改的字段：

```json
{
  "max_wall_time_sec": 3600,
  "max_subagent_calls": 10,
  "max_worker_calls": 8,
  "max_strategy_attempts": 8,
  "max_advanced_iterations": 2,
  "advanced_patience": 1,
  "min_improvement_pct": 3.0
}
```

未知字段、负数、非数字或应为整数却提供小数时，`plan` 命令退出码为 `2`。
