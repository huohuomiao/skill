# 阶段 2：按需优化与全局预算说明书

## 1. 版本目标

v3 在 v2 的 P0 修复和 L1/L2/L3 验证体系之上，解决两个问题：

1. 不再对每个算子固定运行全部优化策略。
2. 不再允许深度优化、重试、Subagent 或 Worker 调用无限增长。

核心变化是把优化拆成“模式选择 → 静态路由 → 预算控制 → 有证据的候选选择”。路由和预算由标准库脚本确定，Agent 只负责执行计划中允许的策略。

本阶段没有合并 Code Gen Subagent，也没有实现跨运行缓存；这两项属于后续阶段。

## 2. 总体架构

```text
KernelGen/triton_code_fix.py
          │
          ▼
选择 optimization_mode
          │
          ├─ correctness ──────────────► triton_final.py
          │                              不进入 Optimizer
          │
          └─ balanced / max-performance
                     │
                     ▼
          optimization_control.py plan
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
optimization_plan.json  optimization_state.json
          │                     │
          └──────────┬──────────┘
                     ▼
           只执行 selected OOB
                     │
          max-performance 才进入高级优化
                     │
                     ▼
          最佳有效候选 / OOB / 原代码
                     │
                     ▼
       Optimizer/triton_optimized.py
```

## 3. 三种模式

### 3.1 correctness

用途：只要求生成正确代码、快速交付、调试 Code Gen，或暂时不关心性能。

行为：

- Main 完成 EnvConfig、Extractor、Code Gen 和 Code Review。
- 跳过整个 Optimizer 阶段。
- `triton_final.py` 直接复制 `KernelGen/triton_code_fix.py`。
- `summary.md` 中 Optimize 字段写 `N/A（correctness 模式未执行优化）`。
- 不创建伪造的 `Optimizer/triton_optimized.py`。

如果直接调用 `mlu-triton-optimize` 并指定 correctness，则该 Skill 会生成一个原样代码副本及“未优化”说明，以保持 standalone 调用契约。

### 3.2 balanced

用途：默认开发模式，在有限时间内获得常规 MLU 优化收益。

行为：

- 运行静态路由器。
- 只执行 `selected=true` 的 OOB 策略。
- 不运行 perf-analyzer 和高级优化。
- 受 30 分钟、8 次 Subagent、8 次 Worker、8 次策略尝试上限约束。

### 3.3 max-performance

用途：用户明确要求极致性能、深度优化或性能冲榜。

行为：

- 先执行静态命中的 OOB 策略。
- 再运行 perf-analyzer。
- 只从静态高级候选和 perf 建议的交集中选择策略。
- 最多 3 轮高级优化。
- 连续 2 轮未达到 2% 提升即停止。
- 整个优化阶段最长 2 小时。

### 3.4 Main 的模式判定

优先级：

1. 用户显式传入 `optimization_mode`。
2. “只保证正确、不要优化、快速生成”映射到 correctness。
3. “极致性能、深度优化、最大性能”映射到 max-performance。
4. 其他情况使用 balanced。

不得因为代码包含 benchmark 就自动升级到 max-performance。

## 4. 静态策略路由

控制器位置：

```text
mlu-triton-optimize/scripts/optimization_control.py
```

Reduce、数学函数、除法、索引除模和 Block 参数只从 `@triton.jit` 函数体提取，测试代码或 PyTorch Reference 中的除法不会触发 Kernel 优化。Grid 和核心数上限从 Wrapper/Launch 侧提取。

生成计划：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <完整Triton文件.py> \
  --output-dir <output_dir>/Optimizer \
  --mode balanced
```

### 4.1 OOB 路由规则

| 策略 | 选择条件 | 常见跳过原因 |
|---|---|---|
| `retiling` | 检测到归约，或存在两个及以上 Block 参数 | 无归约且只有一个 Block 参数 |
| `reduce-opt` | 检测到 Reduce 原语 | 未检测到归约 |
| `modify-grid` | 存在 Grid，且为多维或没有明确物理核心数上限 | 没有 Grid，或已经是单维限核 Grid |
| `index-computation-simplify` | 检测到整数除法 `//` 或取模 `%` | 地址计算没有除模 |
| `gen-autotune-config` | 存在 `tl.constexpr` Block 参数且没有 `@triton.autotune` | 无可调参数或已存在 Autotune |

策略被选中只表示“值得分析”，不表示一定修改代码。策略内部仍需验证实际适用性、精度和性能。

### 4.2 高级策略候选

只有 max-performance 会生成高级候选：

| 策略 | 静态候选条件 |
|---|---|
| `libdevice-opt` | 检测到 exp、log、sqrt、rsqrt、sin、cos、sigmoid、erf 或已有 MLU Libdevice 调用 |
| `config-tuner` | 检测到 `tl.constexpr` Block 参数 |
| `div-to-mul` | 检测到非整数除法 `/` |

真正执行时还必须与 perf-analyzer 的建议取交集。交集为空时停止高级优化，不用随机选择策略。

### 4.3 保守失败方式

如果路由器没有识别到 `@triton.jit` 或 Kernel launch：

```json
"manual_review_required": true
```

流程必须停止并报告原因。禁止用“无法识别，所以全部策略都跑”作为兜底。

## 5. 默认全局预算

预算范围只覆盖 Optimizer 阶段，不包含 EnvConfig、Extractor、Code Gen 和 Code Review 的消耗。

| 字段 | correctness | balanced | max-performance |
|---|---:|---:|---:|
| `max_wall_time_sec` | 0 | 1800 | 7200 |
| `max_subagent_calls` | 0 | 8 | 16 |
| `max_worker_calls` | 0 | 8 | 16 |
| `max_strategy_attempts` | 0 | 8 | 12 |
| `max_advanced_iterations` | 0 | 0 | 3 |
| `advanced_patience` | 0 | 0 | 2 |
| `min_improvement_pct` | 2.0 | 2.0 | 2.0 |

预算是硬上限，采用 `usage >= limit` 判定。因此上限为 0 时不允许第一次动作。

### 5.1 预算优先级

1. 用户提供的预算 JSON 中显式字段。
2. 当前模式默认值。

自然语言中的“多跑几次”“尽量快”等描述不会自动转换成数字。

### 5.2 自定义预算

示例 `budget.json`：

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

调用：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input kernel.py \
  --output-dir output/Optimizer \
  --mode max-performance \
  --budget-file budget.json
```

未知字段、负数、非数字，或次数型字段使用小数时，命令退出码为 2。

## 6. 优化计划

输出：

```text
Optimizer/optimization_plan.json
```

主要字段：

| 字段 | 含义 |
|---|---|
| `mode` | 当前模式 |
| `input_sha256` | 输入代码哈希 |
| `manual_review_required` | 是否必须人工确认输入完整性 |
| `limits` | 本次运行硬预算 |
| `signals` | 静态检测信号 |
| `oob_strategies` | 所有 OOB 策略的选择结果和原因 |
| `selected_oob_strategies` | 实际允许执行的 OOB 顺序 |
| `advanced.enabled` | 是否允许深度优化 |
| `advanced.candidates` | 允许的高级策略集合 |

同一次运行不得手工编辑计划。输入文件改变时，应创建新的优化运行，而不是沿用旧计划。

## 7. 预算状态

输出：

```text
Optimizer/optimization_state.json
```

主要字段：

| 字段 | 含义 |
|---|---|
| `usage.subagent_calls` | 已启动的优化 Subagent 数 |
| `usage.worker_calls` | 已提交的优化 Worker Task 数 |
| `usage.strategy_attempts` | 已得到结果的策略尝试数，失败和重试都计数 |
| `usage.advanced_iterations` | 已完成的高级优化轮数 |
| `advanced_no_improvement` | 连续未达到阈值的高级轮数 |
| `best` | 当前最佳有效候选、耗时和策略 |
| `stop_reason` | 第一次触发的预算停止原因 |
| `history` | 最近 200 个事件 |

脚本采用临时文件加原子替换写入状态。Agent 不应直接修改 JSON。

## 8. 预算命令

### 8.1 检查是否允许新动作

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py check \
  --state output/Optimizer/optimization_state.json \
  --phase oob
```

或：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py check \
  --state output/Optimizer/optimization_state.json \
  --phase advanced
```

退出码：

| 退出码 | 含义 |
|---|---|
| `0` | 允许执行 |
| `3` | 预算或 patience 已停止 |
| `2` | 状态文件或参数错误 |

### 8.2 记录 Subagent

必须在启动前记录：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py record \
  --state output/Optimizer/optimization_state.json \
  --event subagent --phase oob --strategy reduce-opt
```

### 8.3 记录 Worker

必须在提交 Worker 前记录：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py record \
  --state output/Optimizer/optimization_state.json \
  --event worker --phase advanced --strategy config-tuner
```

### 8.4 记录策略结果

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py record \
  --state output/Optimizer/optimization_state.json \
  --event strategy --phase advanced --strategy config-tuner \
  --accuracy-pass true \
  --baseline-ms 0.150 \
  --candidate-ms 0.140 \
  --artifact-path /abs/path/triton_optimized.py
```

### 8.5 记录高级迭代

每轮结束后记录，用于更新 patience：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py record \
  --state output/Optimizer/optimization_state.json \
  --event advanced_iteration --phase advanced \
  --strategy config-tuner \
  --accuracy-pass true \
  --baseline-ms 0.150 \
  --candidate-ms 0.140 \
  --artifact-path /abs/path/triton_optimized.py
```

## 9. 候选接受和回退

### 9.1 OOB

- 精度通过：可以作为下一策略输入。
- 精度失败或未知：回退上一轮通过精度的代码。
- OOB 不使用最小提升阈值，因为部分结构变换可能是后续调优前置条件。

### 9.2 高级优化

提升计算：

```text
improvement_pct = (baseline_ms - candidate_ms) / baseline_ms * 100
```

候选必须同时满足：

- `accuracy_pass=true`
- baseline 和 candidate 都来自实际测量
- baseline 大于 0
- 提升不低于 `min_improvement_pct`

否则回退本轮输入，并增加 `advanced_no_improvement`。

### 9.3 最终代码优先级

1. 状态中的最佳有效候选。
2. `triton_advanced_optimized.py`。
3. `triton_oob_optimized.py`。
4. 原始输入代码。

任何精度失败代码都不得进入此序列。

## 10. 停止条件

预算停止原因：

- `max_wall_time_reached`
- `max_subagent_calls_reached`
- `max_worker_calls_reached`
- `max_strategy_attempts_reached`
- `max_advanced_iterations_reached`
- `advanced_patience_exhausted`

流程停止原因还可能包括：

- `manual_review_required`
- `no_selected_oob_strategy`
- `no_applicable_advanced_strategy`
- `accuracy_failure`
- `infrastructure_failure`

达到停止条件后仍须产出当前最佳有效代码和报告，不能把“预算停止”误报为成功达到最优性能。

## 11. 重试规则

- 输出缺失时允许重试，但重试前重新检查预算。
- 每次重试重新计一次 Subagent 和策略尝试。
- Worker 基础设施失败已经消耗调用次数，不回滚计数。
- 不得删除 `optimization_state.json` 重新获取预算。
- 输入代码发生变化时，应该建立新的输出目录和新计划。

## 12. 报告要求

`Optimizer/triton_optimized.md` 必须包含：

- 模式。
- 输入文件和哈希。
- 默认预算或预算覆盖来源。
- 计划选择的 OOB 策略和跳过原因。
- 实际执行次数。
- 每个候选的精度与性能。
- 高级优化每轮提升。
- `stop_reason`。
- 最终候选路径及选择原因。

缺失数据统一写 `N/A（原因：...）`，不得推算或伪造。

## 13. 验证方法

### 13.1 离线验证

```bash
python -B validation/validate.py all
```

覆盖内容包括：

- 三种模式。
- Reduce 和非 Reduce 路由。
- 已限核单维 Grid 跳过 `modify-grid`。
- 未限核 Grid 命中 `modify-grid`。
- Libdevice、Config、除法高级候选。
- 输入不完整触发人工确认。
- Subagent 硬上限。
- patience 停止。
- 有效提升重置 patience。
- 计划与状态 Schema。

### 13.2 验证实际产物

```bash
python -B validation/validate.py artifacts \
  --output-dir <output_mlu_triton_main> \
  --require-complete
```

balanced 和 max-performance 会额外校验计划、状态、模式和预算一致性。correctness 允许不存在 Optimizer 目录。

### 13.3 MLU 回归

至少选择：

- Elementwise：确认跳过 Reduce 策略。
- Reduce Sum/Max：确认命中 Reduce 策略。
- 已限核单维 Grid：确认跳过 Grid 改写。
- 含 exp/log：确认 max-performance 产生 Libdevice 候选。
- 含除法：确认产生 div-to-mul 候选。

对每个用例记录 v2/v3 的精度、总耗时、Subagent 数、Worker 数和最终性能。

## 14. 部署与回滚

部署时将 v3 内容作为一套整体 Skill 安装，不能只替换 `mlu-triton-optimize/SKILL.md`，否则控制脚本、Schema 和 Main 模式契约不匹配。

建议步骤：

1. 保留当前 v2 副本。
2. 在隔离项目中部署 v3。
3. 运行 L1/L2。
4. 运行代表性 MLU 用例。
5. 比较 v2/v3 指标后再切换默认版本。

回滚时恢复整套 v2，不要只回滚状态脚本。

## 15. 已知边界

- 路由器基于静态语法信号，不证明某策略一定有收益。
- 动态构造的 Grid 或被封装的数学表达式可能需要人工确认。
- v3 尚未实现跨运行缓存和受影响阶段增量重跑。
- v3 尚未合并 Code Gen 的多个串行 Subagent。
- 实际性能结论仍必须来自相同硬件、工具链和测试输入上的真实测量。
