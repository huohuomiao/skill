# v2 与 v3 结构变化对照

## 1. 总览

v3 从 v2 完整复制后独立开发。v2 的 P0 修复、JSON Schema、L1/L2/L3 验证入口全部保留。

v3 新增按需优化路由、三种模式和全局优化预算；Main、Optimizer 包装器、Perf Analyzer 和验证体系相应升级。

文件哈希对比结果：v2 有 74 个文件，v3 有 81 个文件；新增 7 个、删除 0 个、修改 8 个，其余 66 个文件内容完全一致。

## 2. 文件变化

| 路径 | v2 | v3 | 说明 |
|---|---|---|---|
| `mlu-triton-main/SKILL.md` | 固定强制优化 | 已修改 | 增加 correctness、balanced、max-performance；correctness 可跳过优化 |
| `mlu-triton-optimize/SKILL.md` | 固定执行全部 OOB，深度优化无统一预算 | 已重构 | 先生成计划，只运行命中策略，所有动作受全局预算约束 |
| `mlu-triton-optimize/utils/Optimizer.md` | 只选择 local/worker | 已修改 | 增加策略许可检查和 Worker 预算记账 |
| `mlu-triton-optimize/perf-analyzer/strategy.md` | 无全局预算 | 已修改 | Perf Worker 运行前检查并记录预算；修复原有 Markdown 表格 |
| `mlu-triton-optimize/scripts/optimization_control.py` | 无 | 新增 | 静态路由、默认预算、计划、状态、check 和 record CLI |
| `mlu-triton-optimize/references/optimization-control.md` | 无 | 新增 | 控制器机器契约和状态语义 |
| `validation/validate.py` | v2 的 P0/L1/L2 检查 | 已扩展 | 增加模式、路由、预算、patience、计划/状态一致性检查 |
| `validation/cases/behavior_cases.json` | 7 个核心行为用例 | 已扩展 | 增加 correctness、balanced 和预算硬停止场景 |
| `validation/contracts/optimization_plan.schema.json` | 无 | 新增 | 优化计划 Schema |
| `validation/contracts/optimization_state.schema.json` | 无 | 新增 | 全局预算状态 Schema |
| `validation/fixtures/valid/Optimizer/*.json` | 无 | 新增 2 个 | 计划与状态正向样例 |
| `HOW_TO_VALIDATE.md` | v2 验证说明 | 已重写 | 增加阶段 2 控制器、三模式和预算验证 |
| `STAGE2_OPTIMIZATION_MANUAL.md` | 无 | 新增 | 阶段 2 详细说明书 |
| `STRUCTURE_COMPARISON.md` | v1/v2 对照 | 已重写 | 本文档改为 v2/v3 对照 |

## 3. 目录结构

### v2 新增部分

```text
v2/
├── validation/
│   ├── validate.py
│   ├── run_mlu_integration.py
│   ├── contracts/                  # Step 1～4 Schema
│   ├── cases/
│   └── fixtures/
├── HOW_TO_VALIDATE.md
└── STRUCTURE_COMPARISON.md
```

### v3 新增或扩展部分

```text
v3/
├── mlu-triton-optimize/
│   ├── SKILL.md                    # 模式、按需执行和预算工作流
│   ├── scripts/
│   │   └── optimization_control.py
│   ├── references/
│   │   └── optimization-control.md
│   ├── utils/Optimizer.md          # Worker 预算契约
│   └── perf-analyzer/strategy.md   # Perf 预算契约
├── validation/
│   ├── validate.py                 # 增加阶段 2 测试
│   ├── contracts/
│   │   ├── optimization_plan.schema.json
│   │   └── optimization_state.schema.json
│   ├── cases/behavior_cases.json
│   └── fixtures/valid/Optimizer/
│       ├── optimization_plan.json
│       └── optimization_state.json
├── HOW_TO_VALIDATE.md
├── STAGE2_OPTIMIZATION_MANUAL.md
└── STRUCTURE_COMPARISON.md
```

## 4. 工作流变化

| 场景 | v2 | v3 |
|---|---|---|
| 默认模式 | 固定全流程优化 | balanced |
| 只要求正确代码 | 仍强制运行 Optimizer | correctness 跳过 Optimizer |
| OOB 策略 | 固定串行执行 5 个 | 只执行静态命中的策略 |
| 高级优化 | 进入循环直到连续 3 次无提升 | 仅 max-performance，最多 3 轮且受 patience/总预算控制 |
| Subagent 上限 | 无全局上限 | 模式预算硬上限 |
| Worker 上限 | 无优化阶段统一上限 | 模式预算硬上限 |
| 墙钟时间 | 无优化阶段总上限 | balanced 1800 秒，max-performance 7200 秒 |
| 候选接受 | 主要依赖每个策略报告 | 精度必过；高级候选还需达到提升阈值 |
| 最终选择 | Advanced/OOB 路径规则 | 状态中的最佳有效候选优先 |
| 停止证据 | 分散在报告文本 | `optimization_state.json.stop_reason` 和 history |

## 5. 输出结构变化

非 correctness 模式新增：

```text
{output_dir}/Optimizer/
├── optimization_plan.json
├── optimization_state.json
├── {order}_{selected_strategy}/   # 只为选中策略创建
├── triton_oob_optimized.py
├── triton_oob_optimized.md
├── triton_advanced_optimized.py   # 仅 max-performance 且有高级结果时
├── triton_advanced_optimized.md
├── triton_optimized.py
└── triton_optimized.md
```

correctness 模式不要求创建 Optimizer 目录，`triton_final.py` 直接来自 `KernelGen/triton_code_fix.py`。

## 6. 保持不变

- 四个 Skill 名称保持不变。
- EnvConfig 的 local/worker 判定和 Worker 接口保持不变。
- Code Gen Step 1～7 中间产物名称保持不变。
- Code Review 的 `xxx.py -> xxx_fix.py + xxx_fix.md` 契约保持不变。
- OOB 和高级策略的具体实现文档保持不变。
- v2 的 P0 修复保持不变。
- 实际精度和性能结论仍必须来自真实 MLU 执行。

## 7. 尚未进入 v3 的后续优化

- Code Gen Subagent 合并。
- 环境和平台规则进一步去重。
- 输入、Skill 版本和硬件指纹缓存。
- 断点续跑及受影响阶段增量失效。
