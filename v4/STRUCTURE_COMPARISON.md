# v3 与 v4 结构变化对照

## 1. 总览

v4 从 v3 完整复制后独立开发，v3 的 P0 修复、三种优化模式、按需策略路由和全局预算全部保留。

哈希对比结果：v3 有 81 个文件，v4 有 97 个文件；新增 16 个、删除 0 个、修改 8 个，其余 73 个文件内容完全一致。比较时排除了 `__pycache__` 和 `.pyc`。

v4 的主变化是：

- Main 新增内容寻址缓存和外层阶段状态机。
- Code Gen 新增 Step 1～7 哈希检查点。
- Optimizer 新增不会重置全局预算的 `--resume`。
- EnvConfig 新增稳定硬件/工具链上下文。
- 验证体系新增持续回归比较、策略阈值和正负 fixtures。

## 2. 修改文件

| 路径 | v3 | v4 |
|---|---|---|
| `mlu-triton-main/SKILL.md` | 顺序执行五阶段 | 增加 init/next/start/complete/restore/resume 事务和恢复总结 |
| `mlu-triton-main/subagents/EnvConfig.md` | 输出 config 与 runtime_info | 新增 `run_context.json`，稳定标识真实硬件和工具链 |
| `mlu-triton-code-gen/SKILL.md` | 阶段中断后按原回退逻辑重跑 | Step 1～7 保存/验证内部检查点 |
| `mlu-triton-optimize/SKILL.md` | 每次进入步骤 2 生成 plan/state | 区分首次初始化和 `--resume`，禁止覆盖预算 |
| `mlu-triton-optimize/scripts/optimization_control.py` | `plan` 总会重建状态 | 已有状态时默认拒绝；`--resume` 校验兼容性且不改写 |
| `validation/validate.py` | v3 路由/预算验证 | 新增缓存、恢复、损坏、路径安全、预算续跑和回归正负测试 |
| `HOW_TO_VALIDATE.md` | v3 模式与预算验证 | 重写为 v4 离线、缓存、续跑、回归、MLU 分层验证 |
| `STRUCTURE_COMPARISON.md` | v2/v3 对照 | 重写为 v3/v4 文件、目录和工作流对照 |

## 3. 新增文件

| 路径 | 作用 |
|---|---|
| `mlu-triton-main/scripts/run_control.py` | 标准库运行控制器、原子清单、缓存、检查点与恢复 |
| `mlu-triton-main/references/stage-sources.json` | 阶段 DAG、相关源边界、模式和必需产物 |
| `mlu-triton-main/references/run-control.md` | Main 的精简机器操作契约 |
| `validation/regression.py` | 规范化结果比较器，输出 JSON/Markdown 和 CI 退出码 |
| `validation/regression_policy.json` | 正确性、性能、耗时、Token、Subagent 默认阈值 |
| `validation/contracts/run_context.schema.json` | 硬件/工具链上下文 Schema |
| `validation/contracts/run_manifest.schema.json` | 可续跑清单 Schema |
| `validation/contracts/regression_result.schema.json` | 单次回归输入 Schema |
| `validation/contracts/regression_report.schema.json` | 比较输出 Schema |
| `validation/contracts/regression_policy.schema.json` | 回归阈值策略 Schema |
| `validation/fixtures/valid/RunControl/run_context.json` | 运行上下文正向 fixture |
| `validation/fixtures/valid/Regression/baseline.json` | 持续回归基线 fixture |
| `validation/fixtures/valid/Regression/current_pass.json` | 阈值内正向 fixture |
| `validation/fixtures/invalid/Regression/current_fail.json` | 精度/性能/成本回退 fixture |
| `validation/fixtures/invalid/Regression/current_context_mismatch.json` | 硬件/工具链不可比 fixture |
| `STAGE4_CACHE_RESUME_REGRESSION_MANUAL.md` | 阶段 4 完整设计、使用、迁移和验收说明书 |

## 4. 目录结构变化

### v3 相关结构

```text
v3/
├── mlu-triton-main/
│   ├── SKILL.md
│   └── subagents/
├── mlu-triton-code-gen/
├── mlu-triton-optimize/
│   ├── scripts/optimization_control.py
│   └── references/optimization-control.md
├── validation/
│   ├── validate.py
│   ├── run_mlu_integration.py
│   ├── contracts/                 # Code Gen + Optimization Schema
│   └── fixtures/
├── STAGE2_OPTIMIZATION_MANUAL.md
├── HOW_TO_VALIDATE.md
└── STRUCTURE_COMPARISON.md
```

### v4 新增或扩展结构

```text
v4/
├── mlu-triton-main/
│   ├── SKILL.md                   # 外层阶段事务
│   ├── scripts/
│   │   └── run_control.py         # 缓存/续跑控制器
│   ├── references/
│   │   ├── run-control.md
│   │   └── stage-sources.json
│   └── subagents/EnvConfig.md     # run_context 契约
├── mlu-triton-code-gen/SKILL.md   # Step 检查点
├── mlu-triton-optimize/
│   ├── SKILL.md                   # 预算恢复契约
│   └── scripts/optimization_control.py
├── validation/
│   ├── validate.py                # v4 离线负向验证
│   ├── regression.py
│   ├── regression_policy.json
│   ├── contracts/
│   │   ├── run_context.schema.json
│   │   ├── run_manifest.schema.json
│   │   ├── regression_result.schema.json
│   │   ├── regression_report.schema.json
│   │   └── regression_policy.schema.json
│   └── fixtures/
│       ├── valid/RunControl/
│       ├── valid/Regression/
│       └── invalid/Regression/
├── STAGE2_OPTIMIZATION_MANUAL.md  # 保留 v3 优化设计
├── STAGE4_CACHE_RESUME_REGRESSION_MANUAL.md
├── HOW_TO_VALIDATE.md
└── STRUCTURE_COMPARISON.md
```

## 5. 工作流变化

| 场景 | v3 | v4 |
|---|---|---|
| 新运行 | 直接进入 EnvConfig | 先创建 `run_manifest.json` |
| 相同需求重跑 | 所有阶段重新执行 | EnvConfig 重检，其他有效阶段可按指纹恢复 |
| Skill 修改 | 人工判断重跑范围 | 相关阶段指纹变化，自动失效本阶段及下游 |
| Code Gen 中断 | 按回退表重跑 | 有效 Step 检查点可继续 |
| Optimizer 中断 | 误调用 plan 可能重置状态 | 普通 plan 拒绝覆盖，`--resume` 保留预算 |
| 输出被人工修改 | 可能继续使用 | SHA-256 不符，自动撤销阶段信任 |
| 硬件变化 | 报告依赖人工识别 | `hardware_key` 不同，运行态缓存不命中 |
| 工具链变化 | 报告依赖人工识别 | `toolchain_key` 不同，性能不可比较 |
| 回归验证 | 人工比较 summary | 标准 JSON + 阈值策略 + CI 退出码 |
| 最终 summary | 模式、预算和性能 | 另含 attempts、cache hit、恢复/失效原因和上下文 |

## 6. 运行产物变化

v4 的每次完整运行新增：

```text
{output_dir}/
├── run_manifest.json
├── EnvConfig/
│   ├── config.md
│   ├── runtime_info.txt
│   └── run_context.json
├── Extractor/
├── KernelGen/
├── Optimizer/                     # correctness 模式不生成
├── triton_final.py
├── summary.md
└── regression_result.json         # Main 直接产出的持续回归输入
```

共享缓存默认位于：

```text
{output_dir的同级}/.mlu-triton-cache/<stage>/<fingerprint>/
├── metadata.json
└── artifacts/...
```

缓存不是最终交付物，不应复制进算子输出目录，也不由控制器自动删除。

## 7. 保持不变

- 四个 Skill 名称和基本职责不变。
- Code Gen Step 1～7 的业务顺序和已有产物名称不变。
- Code Review 的单文件输入及 `xxx.py -> xxx_fix.py` 契约不变。
- v3 的 correctness、balanced、max-performance 模式不变。
- v3 的 OOB 静态路由、深度候选和全局预算字段不变。
- Worker 仍必须在当前 `JOB_ID` 内串行、阻塞执行。
- 真实精度和性能结论仍必须来自可用 MLU 环境。

## 8. 兼容性

- v4 可以读取 v4 自己创建的 schema version 1 清单。
- v3 输出没有运行清单，不能直接作为可信缓存完成状态。
- v3 的优化 plan/state 结构仍保留；只有 v4 控制脚本提供安全恢复入口。
- 回归基线按 `hardware_key + toolchain_key` 隔离，不跨上下文比较。
