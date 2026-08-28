# p3 大模型真实环境测试方法

本方法用于验证“大模型使用 p3 Skill 后是否真的按契约完成算子开发”，不是只验证脚本能否启动。每次测试使用一个全新的模型任务和独立输出目录，避免沿用旧对话结论或旧产物。

## 1. 环境前提

测试机或当前 Worker Job 必须具备：

- 可用的目标加速卡，`cnmon` 能识别真实设备。
- 与目标设备匹配的 PyTorch、Triton 和工具链。
- `torch.mlu.is_available()` 返回真。
- 允许编译、精度测试和 benchmark；性能测试期间没有其他负载干扰。
- 若本机没有设备，设置当前已有的 `JOB_ID`，并保证 Worker 提交服务可用；不要为测试自动创建新 Job。

先在 Skill 根目录外执行真实预检：

```bash
python <p3-skill-root>/scripts/environment/inspect-device.py
python <p3-skill-root>/scripts/environment/verify-runtime.py
```

两条命令都必须以退出码 `0` 完成。任何一条失败都先修复环境，不要让模型在无设备环境中继续并推断性能。

随后执行低成本门禁：

```bash
python <p3-skill-root>/scripts/validation/run-validation.py \
  --level l2 \
  --report-dir <absolute-report-directory>
```

只有 L1、L2 都通过，才进入真实模型测试。

## 2. 安装并隔离模型任务

将 `p3/triton-kernel-workflow` 安装或链接为模型可发现的 `triton-kernel-workflow` Skill。为每个测试新建一个模型任务，不提供预期答案、历史失败分析或人工优化方案，只提供原始算子需求、模式和输出目录。

建议最少执行三组：

| 用例 | 模式 | 主要检查 |
| --- | --- | --- |
| Vector Add，elementwise | `correctness` | 生成、编译、精度通过，性能阶段必须跳过 |
| 最后一维 Softmax，reduction | `balanced` | 保留可调归约面，只运行静态命中的 OOB 策略 |
| Transpose 或 layout 变换 | `max-performance` | 静态路由正确，允许深度优化但严格受全局预算限制 |

每组使用不同的空输出目录，例如 `runs/vector-add-correctness`、`runs/softmax-balanced` 和 `runs/transpose-max`。

## 3. 推荐模型提示词

以下提示词可直接交给一个全新的大模型任务；替换尖括号内容即可：

```text
使用 $triton-kernel-workflow 完成一个完整的 Triton 算子开发任务。

optimization_mode=<correctness|balanced|max-performance>
output_dir=<absolute-empty-output-directory>

需求：<只描述算子语义、输入 shape/dtype、输出、精度容差和需要覆盖的测试 shape。不要提供实现方案。>

必须使用当前真实设备或当前已有 Worker Job 完成环境检查、编译和精度验证。只有真实 benchmark 输出才能形成性能结论；环境或基础设施失败时停止并保留原始 stdout、stderr 和退出码。最终返回 run_manifest.json、triton_final.py 和 summary.md 的绝对路径。
```

Softmax 代表需求可写为：

```text
实现最后一维 Softmax。覆盖 float32 的 [128, 2048]、[32, 8192] 和非 2 次幂宽度 [17, 1000]；结果与 PyTorch 参考实现比较，给出明确 rtol/atol。代码中必须包含可独立执行的 accuracy_test 和 performance_test。
```

不要在提示词中告诉模型应该命中哪些优化策略；策略选择本身就是被测行为。

## 4. 验收证据

不要只检查模型的自然语言回复。逐项核对输出文件：

1. `EnvConfig/config.json` 来自真实探测，设备型号、工具链版本和 `execution_backend` 不是 `unknown` 或模拟值。
2. `run_manifest.json` 的阶段顺序正确，所有 `completed` 阶段都有非空产物；`correctness` 的 `performance-tuning` 为 `skipped`。
3. `KernelGen/triton_code_fix.py` 能在同一后端重新执行，覆盖的 shape/dtype 全部通过精度测试。
4. `balanced` 和 `max-performance` 在调优前生成 `Optimizer/strategy_plan.json`，只执行 `decision=apply` 的策略。
5. `Optimizer/tuning_state.json` 的固定上限为 3 个深度轮次、16 次 Worker 调用和 1800 秒；真实硬件命令串行执行。
6. `Optimizer/best_so_far.json` 选择精度通过、硬件和 benchmark 口径一致且 `latency_ms` 最小的候选；不能因为它最后运行就选择它。
7. `triton_final.py` 与 Manifest 记录的最终检查点一致。没有真实测量时，`summary.md` 不得声明加速比或性能收益。

判定一次模型测试通过的最低条件是：环境证据真实、编译成功、全部要求的精度用例通过、模式与路由正确、预算没有越界、最终产物来源正确。性能提升不是每个用例都必须出现；没有适用策略时，正确保留已通过精度的最佳候选同样是合法结果。

## 5. 发布前 L3 回归

把上述三类算子的编译、精度和性能命令写入符合 `references/schemas/integration-suite.schema.json` 的 suite JSON，并补充 Worker 正常提交与失败恢复命令。然后运行：

```bash
python <p3-skill-root>/scripts/validation/run-validation.py \
  --level l3 \
  --report-dir <absolute-report-directory> \
  --integration-suite <absolute-integration-suite-json>
```

L3 会先重新执行 L1、L2，再串行执行 elementwise、reduction、layout 的真实编译、精度和性能命令。最终报告中 `hardware_evidence=true`、所有检查为 `pass`，才可以声明 p3 完成真实硬件集成回归。
