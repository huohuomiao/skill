# p3 · 模式、静态路由与全局预算

p3 从 p2.1 独立演进，实现阶段 3 的模式选择、优化前静态策略路由和统一调优预算。p2.1 的归约生成修正、两次 Code Gen 调度、Step 1–6 产物和最终候选选择契约继续保留。

## 三种模式

- `correctness`：完成生成、编译和精度验证；跳过性能调优，不产生延迟或加速比结论。
- `balanced`：默认模式；建立真实基线，只执行静态命中的 OOB 策略，跳过深度优化。
- `max-performance`：在 `balanced` 路径之后，允许静态命中的深度策略在硬预算内迭代。

`optimization_mode` 在环境阶段写入 `run_manifest.json` 并保持不可变。`correctness` 最终直接交付精度验证通过的 Code Gen 产物。

## 确定性静态路由

新增 `plan-strategies.py` 和 `strategy-plan.schema.json`。调优前扫描 Triton AST，生成 `Optimizer/strategy_plan.json`：

- 没有归约、复杂索引、可调分块参数、设备数学模式或除法时，跳过对应策略。
- 已有有界 persistent Grid 覆盖时，跳过 `modify-grid`。
- `balanced` 固定关闭深度策略；`correctness` 关闭全部性能策略。
- 实际硬件编译、精度、profile 和 benchmark 保持串行。

工作流只调度计划中 `decision=apply` 的策略，`skip` 项只记录确定性原因，不创建工作目录、不启动优化 Agent。

## 硬性全局预算

新增 `manage-tuning-budget.py`、`tuning-state.schema.json` 和本地执行时间守卫。每次调优的限额固定为：

- 深度优化最多 3 轮。
- 性能调优阶段最多 16 次 Worker 提交。
- 从基线测量前开始计时，总墙钟时间最多 1800 秒（30 分钟）。

Worker 提交入口在 POST 前原子预留调用次数，并将超时压缩到剩余墙钟时间。本地动态命令由 `run-budgeted-local.py` 限制在同一时间预算内。任一限额到达后停止新调度，保留已完成候选，再进入确定性最终选择。

## 明确不在 p3 实现的内容

p3 不引入噪声感知性能选择，不增加分位数、方差、置信区间或新的候选得分。`optimization-candidate.schema.json` 和 `select-best-candidate.py` 保持 p2.1 语义：只在精度通过、硬件与 benchmark 口径可比的候选中选择最小 `latency_ms`，完全相同时按 `candidate_id` 稳定排序。

## 离线验证

- p3 契约回归：9 项通过，覆盖模式、路由、3/16/1800 限额、本地/Worker 拦截和 manifest 模式不可变。
- L1：通过，4.705 秒，低于 30 秒门禁。
- L2：28 个固定场景全部通过，3.764 秒。
- L1+L2 总入口：通过，8.612 秒，无预算或门禁错误。
- 本轮未执行 L3 真实硬件回归，因此不声明实际精度通过率或性能收益。
