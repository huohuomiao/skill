# v4 验证说明

v4 验证分为：离线门禁、缓存/续跑 smoke test、回归比较器验证、真实产物检查和 MLU 持续回归。设计依据见 [STAGE4_CACHE_RESUME_REGRESSION_MANUAL.md](STAGE4_CACHE_RESUME_REGRESSION_MANUAL.md)。

## 1. 环境

使用 Python 3.9 或更高版本。离线工具只使用标准库，不需要 MLU、PyYAML、pytest 或 jsonschema。

所有命令从 v4 根目录执行，建议使用 `-B`：

```bash
python -B validation/validate.py all
```

## 2. 每次修改必须运行

```bash
python -B validation/validate.py l1
python -B validation/validate.py l2
```

或一次运行：

```bash
python -B validation/validate.py all
```

通过条件是同时出现 `[PASS] L1` 和 `[PASS] L2`。

L1 覆盖：

- 四个 Skill frontmatter、Python AST、Markdown 围栏、路径和文件非空。
- v2 P0 契约与 v3 模式/预算契约未回退。
- v4 run controller、stage source config、regression comparator 存在。
- Main 恢复规则、Code Gen 检查点、Optimizer `--resume` 契约存在。
- 所有 JSON Schema 可解析。

L2 覆盖：

- v3 按需路由、预算硬上限、patience 和最佳候选。
- EnvConfig 不缓存，运行上下文必须绑定。
- Code Gen 内部检查点在中断后保留。
- 相同请求/上下文跨输出目录命中 extractor、kernel_gen 缓存。
- 产物篡改和指纹变化向下游失效。
- 路径穿越被拒绝。
- Optimizer 恢复保留已消耗预算，普通 plan 拒绝覆盖。
- 回归正例、精度/性能负例、硬件/工具链不可比样例。

## 3. 手工验证缓存与续跑

使用一个临时输出目录：

```bash
python -B mlu-triton-main/scripts/run_control.py init \
  --output-dir <temp-output> --input "reduce sum" --mode balanced

python -B mlu-triton-main/scripts/run_control.py next \
  --manifest <temp-output>/run_manifest.json
```

第一次 `next` 应返回 `env_config/run`。真实完成 EnvConfig 后必须生成并绑定上下文：

```bash
python -B mlu-triton-main/scripts/run_control.py complete \
  --manifest <manifest> --stage env_config
python -B mlu-triton-main/scripts/run_control.py bind-context \
  --manifest <manifest> --context-file <temp-output>/EnvConfig/run_context.json
```

模拟中断只能在阶段已标记 `running` 后执行：

```bash
python -B mlu-triton-main/scripts/run_control.py resume --manifest <manifest>
```

检查：

- `running` 变为 `pending`，attempts 不归零。
- 有效检查点仍返回 `reusable=true`。
- 已完成阶段只有在产物哈希有效时跳过。
- 第二个相同输入、相同 `run_context`、同一 `cache_dir` 的运行返回 `restore`。

## 4. 验证 Optimizer 预算恢复

首次：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <temp-output>/Optimizer --mode balanced
```

恢复：

```bash
python -B mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <temp-output>/Optimizer --mode balanced --resume
```

通过条件：

- `--resume` 输出 `"resumed": true`。
- `optimization_state.json.usage`、`best`、`history` 和 `advanced_no_improvement` 完全保留。
- 去掉 `--resume` 再运行普通 plan 时退出码为 2，不覆盖状态。
- 修改输入、模式或预算后 `--resume` 退出码为 2。

## 5. 验证持续回归比较器

正例：

```bash
python -B validation/regression.py compare \
  --baseline validation/fixtures/valid/Regression/baseline.json \
  --current validation/fixtures/valid/Regression/current_pass.json \
  --report-json <temp>/pass.json --report-md <temp>/pass.md
```

期望退出码 0，报告 `passed=true`。

负例：

```bash
python -B validation/regression.py compare \
  --baseline validation/fixtures/valid/Regression/baseline.json \
  --current validation/fixtures/invalid/Regression/current_fail.json \
  --report-json <temp>/fail.json --report-md <temp>/fail.md
```

期望退出码 1，至少包含 `accuracy_failed` 和 `latency_regression_exceeded`。

不可比上下文：

```bash
python -B validation/regression.py compare \
  --baseline validation/fixtures/valid/Regression/baseline.json \
  --current validation/fixtures/invalid/Regression/current_context_mismatch.json \
  --report-json <temp>/mismatch.json --report-md <temp>/mismatch.md
```

默认策略下期望退出码 1、`context_compatible=false`，性能不得显示为 pass。

## 6. 验证真实工作流产物

增量检查：

```bash
python -B validation/validate.py artifacts --output-dir <output_dir>
```

完整交付检查：

```bash
python -B validation/validate.py artifacts \
  --output-dir <output_dir> --require-complete
```

v4 在 v3 检查基础上新增：

- `run_context.json` Schema。
- `run_manifest.json` Schema。
- `regression_result.json` Schema，且硬件/工具链与 EnvConfig 一致。
- `finalize=complete`。
- 完整输出仍需代码、报告、精度/性能字段和模式一致性。

## 7. 真实 MLU 验证

环境检查：

```bash
python -B validation/run_mlu_integration.py
```

单算子：

```bash
python -B validation/run_mlu_integration.py --operator <triton_final.py>
```

Nightly 至少覆盖一个 Elementwise 和一个 Reduce，并输出符合 `regression_result.schema.json` 的 current JSON，再运行第 5 节比较器。

## 8. CI 分层

| 层级 | 触发 | 命令/范围 | 阻断条件 |
|---|---|---|---|
| PR | 每次 Skill 修改 | `validation/validate.py all` | 任一 L1/L2 失败 |
| Nightly | 每晚或环境可用时 | 代表性真实 MLU 用例 + regression compare | 精度失败、性能/成本超阈值、上下文不可比 |
| Release | 发布前 | 全硬件/工具链矩阵 + artifacts complete | 任一门禁失败或未审计 not_comparable |

## 9. 最终验收

- L1、L2 全部通过。
- 正向回归退出 0，负向和上下文不匹配退出 1。
- 至少一次相同输入的第二运行产生真实 cache hit。
- 人工破坏输出产物后，清单自动失效当前阶段和下游。
- Code Gen 中断可从有效 Step 检查点继续。
- Optimizer 中断不重置预算。
- 真实 MLU Elementwise、Reduce 回归通过。

只有最后一项必须在可用 MLU 环境执行；离线通过不能替代动态精度和性能结论。
