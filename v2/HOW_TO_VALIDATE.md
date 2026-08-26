# v2 验证说明

本文说明如何验证 `v2` 的 Skill 修改。除 L3 外，所有检查都不需要 MLU，也不需要 PyYAML、pytest 或 jsonschema。

## 1. 准备 Python

要求 Python 3.9 或更高版本。验证器只使用标准库。

在 `v2` 根目录执行。为避免产生 `__pycache__`，命令统一使用 `-B`：

```bash
python -B validation/validate.py all
```

如果系统中的 `python` 不是实际解释器，请将命令中的 `python` 替换成可用解释器的绝对路径。

## 2. L1：静态检查

```bash
python -B validation/validate.py l1
```

L1 检查内容：

- 四个 `SKILL.md` 的 frontmatter、名称和描述。
- 核心脚本与共享规则是否存在。
- `analyzer_rep.py` 是否已经删除。
- Python 文件能否被 AST 解析；不会生成 `.pyc`。
- Markdown 代码围栏是否闭合。
- `.claude/skills/...` 引用映射到源码树后是否存在。
- P0 契约不变量：Code Review 单参数、测试只在 Review 阶段执行、深度优化产物优先、EnvConfig 路径正确。
- JSON Schema 是否存在且为合法 JSON。

成功输出示例：

```text
[PASS] L1: <检查数量> check(s)
```

## 3. L2：离线行为契约

```bash
python -B validation/validate.py l2
```

L2 不启动 Agent，不执行 MLU 代码，主要验证：

- 自然语言需求进入完整 Code Gen 路径。
- Triton 代码输入走快速路径并跳过 Stage 1～4。
- 原代码通过时不启动静态、动态修复。
- 基础设施错误不得修改 Kernel。
- DynamicFixer 具有明确停止条件。
- 深度优化产物优先于 OOB 产物。
- 本地环境失败时复用当前 Job 的 Worker。
- Step 1～4 的有效 fixture 符合 Schema。
- 缺字段的负向 fixture 能被拒绝。
- `step1_base_info.io_shapes` 与 `step1_io_shapes.json` 一致。

行为用例位于 `validation/cases/behavior_cases.json`。增加或修改路由时，应同步增加对应场景，避免只验证文案格式。

## 4. L1 + L2：日常推荐命令

每次修改 Skill 后，至少执行：

```bash
python -B validation/validate.py all
```

退出码含义：

| 退出码 | 含义 |
|---|---|
| `0` | 全部检查通过 |
| `1` | 存在契约、Schema、路径或行为检查失败 |

## 5. 验证真实工作流产物

完成一次算子开发后，可以检查实际输出目录：

```bash
python -B validation/validate.py artifacts --output-dir <output_mlu_triton_main>
```

要求最终完整产物时增加：

```bash
python -B validation/validate.py artifacts --output-dir <output_mlu_triton_main> --require-complete
```

检查内容包括：

- 已产生的 Step 1～4 JSON 是否符合对应 Schema。
- Step 1 两份 IO Shapes 是否一致。
- Triton 快速路径是否存在 `Extractor/original_code.py`。
- EnvConfig 是否记录 `local` 或 `worker`。
- KernelGen、Optimizer、最终代码和总结是否非空。
- `summary.md` 是否包含精度、误差容限、Code Gen 和 Optimize 证据字段。

## 6. L3：本地 MLU 集成验证

在具备 MLU、PyTorch-MLU 和 Triton-MLU 的环境中运行：

```bash
python -B validation/run_mlu_integration.py
```

该命令按顺序执行：

1. `share/mlu/runtime/get_device_info.py`
2. `share/mlu/runtime/test_env_code.py`

环境通过后，可以继续验证一个完整算子文件：

```bash
python -B validation/run_mlu_integration.py --operator <triton_code_fix.py>
```

默认环境检查超时 600 秒，算子超时 1800 秒。可以通过以下参数调整：

```bash
python -B validation/run_mlu_integration.py \
  --operator <triton_code_fix.py> \
  --env-timeout-sec 600 \
  --operator-timeout-sec 1800
```

## 7. Worker 模式验证

本地没有可用 MLU 时，应先把 `v2` 部署到目标项目的 `.claude/skills`，然后在同一个 `JOB_ID` 下使用现有提交脚本。不要创建新 Job：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
  --task-type custom \
  --workdir <目标项目根目录绝对路径> \
  --timeout-sec 600 \
  --command "python .claude/skills/share/mlu/runtime/get_device_info.py && python .claude/skills/share/mlu/runtime/test_env_code.py"
```

环境检查通过后，再以前台同步方式提交代表性算子的精度或性能测试。成功与否以提交脚本退出码及 `stdout.log`、`stderr.log`、`result.json` 为准。

## 8. 建议的变更验证矩阵

| 修改范围 | L1 | L2 | 产物检查 | L3 MLU |
|---|---:|---:|---:|---:|
| 文档、路径、调用契约 | 必须 | 必须 | 可选 | 可选 |
| Step 1～4 JSON 字段 | 必须 | 必须 | 必须 | 可选 |
| Code Review 修复逻辑 | 必须 | 必须 | 必须 | 必须 |
| 环境或 Worker 逻辑 | 必须 | 必须 | 可选 | 必须 |
| Kernel 生成或优化策略 | 必须 | 必须 | 必须 | 必须 |

## 9. 本版本验收顺序

建议按以下顺序完成 v2 验收：

1. `python -B validation/validate.py all`
2. 用一个自然语言 Reduce 算子验证完整路径。
3. 用一个已有 Triton 算子验证快速路径。
4. 对两个输出目录执行 `artifacts --require-complete`。
5. 在同一型号 MLU 上比较 v1 与 v2 的精度和性能结果。

L1/L2 通过只能证明结构和行为契约正确，不能替代真实 MLU 精度及性能验证。
