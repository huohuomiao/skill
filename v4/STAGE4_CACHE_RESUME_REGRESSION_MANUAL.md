# 阶段 4 详细说明书：缓存、断点续跑和持续回归

## 1. 版本目标

v4 在 v3 的“按需优化 + 全局预算”之上解决三个剩余成本问题：

1. 相同输入、相同相关 Skill、相同运行环境重复执行，浪费开发时间、Worker 配额和 Token。
2. Code Gen 或 Optimizer 中断后只能从头开始，已生成且有效的产物不能安全复用。
3. Skill 修改后只能靠人工观察，难以判断正确性、性能、耗时和 Token 是否回退。

v4 的交付不是一个隐式提示词约定，而是三个可执行控制面：

- `mlu-triton-main/scripts/run_control.py`：外层阶段清单、内容寻址缓存、断点恢复和失效传播。
- `mlu-triton-optimize/scripts/optimization_control.py plan --resume`：保留 v3 全局优化预算的安全恢复入口。
- `validation/regression.py`：同硬件/工具链下的持续回归比较器。

全部控制脚本只依赖 Python 标准库，CPU 环境即可执行契约验证。

## 2. 设计边界

### 2.1 v4 会做什么

- 对输入、模式、预算、相关 Skill 源、上游指纹和运行上下文计算确定性 SHA-256。
- 复用完整成功阶段的不可变缓存。
- 在 Code Gen Step 1～7 保存带哈希的内部检查点。
- 中断后保留 Optimizer 的计划、累计预算、最佳候选、patience 和历史。
- 当输入、Skill 或上下文变化时，只失效受影响阶段及其下游。
- 对精度、延迟、总耗时、Token 和 Subagent 调用数执行阈值回归。

### 2.2 v4 不会做什么

- 不缓存 EnvConfig 的“设备当前可用”结论；每个新运行都必须真实探测。
- 不在硬件或工具链不一致时比较性能。
- 不自动清理共享缓存，避免误删其他运行仍在使用的条目。
- 不把离线 fixture 结果当作真实 MLU 精度/性能结论。
- 不允许多个主流程同时写同一个 `run_manifest.json`。

## 3. 总体结构

```text
用户输入
  |
  v
run_manifest.json
  |
  +-- env_config --真实执行，不缓存--> run_context.json
  |
  +-- extractor --------内容寻址缓存--------+
  |
  +-- kernel_gen -------完整阶段缓存---------+-- Step 1..7 内部检查点
  |
  +-- optimizer --------完整阶段缓存---------+-- optimization_state.json 持续预算
  |
  +-- finalize ---------本次重新生成---------> summary.md
  |
  v
regression result -> baseline compare -> JSON + Markdown 门禁报告
```

外层阶段顺序为：

```text
env_config -> extractor -> kernel_gen -> optimizer -> finalize
```

`correctness` 模式跳过 `optimizer`，`finalize` 直接依赖 `kernel_gen` 的有效结果。

## 4. 核心文件

| 文件 | 作用 |
|---|---|
| `mlu-triton-main/scripts/run_control.py` | init、next、start、complete、restore、resume、invalidate、检查点 |
| `mlu-triton-main/references/stage-sources.json` | 阶段 DAG、模式、缓存能力、源边界、必需产物 |
| `mlu-triton-main/references/run-control.md` | Main 可直接执行的运行契约 |
| `{output_dir}/run_manifest.json` | 单次运行状态、指纹、产物哈希、attempts 和事件审计 |
| `{output_dir}/EnvConfig/run_context.json` | 稳定硬件与工具链标识 |
| `.mlu-triton-cache/<stage>/<fingerprint>/` | 不可变共享缓存条目 |
| `{output_dir}/regression_result.json` | Main finalize 从真实证据生成的规范化当前结果 |
| `validation/regression.py` | 基线比较和 CI 退出码 |
| `validation/regression_policy.json` | 默认阈值策略 |
| `validation/contracts/*.schema.json` | v4 运行与回归 JSON 契约 |

## 5. 新运行生命周期

### 5.1 初始化

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py init \
  --output-dir <output_dir> \
  --input <输入文件路径或原始需求文本> \
  --mode balanced \
  [--budget-file <budget.json>] \
  [--cache-dir <共享缓存目录>]
```

行为：

- 创建 `{output_dir}/run_manifest.json`。
- 输入是现存文件时记录路径与文件内容 SHA-256；否则按原始文本计算 SHA-256。
- 预算文件只记录内容哈希，任何修改都会使相关阶段指纹变化。
- 已存在清单时拒绝初始化，防止覆盖恢复证据。
- 默认缓存目录是 `output_dir` 的同级 `.mlu-triton-cache`。

### 5.2 查询下一动作

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py next \
  --manifest <output_dir>/run_manifest.json
```

返回值只有四类：

| action | 主流程行为 |
|---|---|
| `run` | 对返回阶段执行 `start`，再调用对应 Skill |
| `restore` | 执行 `restore`，不要调用对应 Skill |
| `blocked` | 按 reason 修复前置条件，禁止越过阶段 |
| `done` | 所有阶段已成功、缓存命中或按模式跳过 |

### 5.3 阶段事务

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py start \
  --manifest <manifest> --stage <stage>

python -B .claude/skills/mlu-triton-main/scripts/run_control.py complete \
  --manifest <manifest> --stage <stage>
```

`complete` 之前控制器会检查本阶段所有必需产物存在、非空并计算 SHA-256，同时按阶段配置收集已存在的可选中间 JSON、代码和报告。检查失败时不会写完成状态。

执行阶段失败时：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py fail \
  --manifest <manifest> --stage <stage> --reason <真实失败原因>
```

失败只记录状态和原因，不删除目录、不回滚预算、不覆盖调试证据。

## 6. 运行上下文

EnvConfig 新增 `{output_dir}/EnvConfig/run_context.json`：

```json
{
  "schema_version": 1,
  "execution_backend": "local",
  "hardware_key": "MLU370-X8:cluster-16:memory-48GB",
  "toolchain_key": "neuware-1.18:triton-3.0:torch-2.4",
  "device_info": {},
  "toolchain_info": {}
}
```

EnvConfig 完成后绑定上下文：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py bind-context \
  --manifest <manifest> \
  --context-file <output_dir>/EnvConfig/run_context.json
```

约束：

- 三个字符串字段必须来自真实探测且非空。
- key 不得包含时间戳、task_id、随机目录等不稳定字段。
- Worker 模式记录 Worker 硬件和工具链，不记录调用端 CPU 信息。
- 未绑定上下文时，`kernel_gen` 返回 `blocked/run_context_not_bound`。
- 上下文变化会失效 `kernel_gen`、`optimizer`、`finalize`，但不要求重新抽取相同需求。

## 7. 内容寻址缓存

### 7.1 指纹组成

每个阶段的指纹是以下规范化内容的 SHA-256：

- 输入文件内容或原始需求文本哈希；
- 对模式敏感的阶段，`optimization_mode`；
- 对预算敏感的阶段，显式预算文件内容哈希；
- `stage-sources.json` 为本阶段声明的 Skill、脚本和共享规则哈希；
- 全部上游阶段指纹；
- 对运行态阶段，规范化 `run_context` 哈希。

相关源边界示例：

- 修改 `Extractor.md`：失效 extractor 及下游，不重新做 EnvConfig。
- 修改 Code Gen、Code Review 或 MLU 平台规则：失效 kernel_gen 及下游。
- 修改某个 Optimize 策略或控制器：只失效 optimizer 和 finalize。
- 只切换优化模式/预算：Extractor 与 Code Gen 仍可复用，Optimizer 和 finalize 使用新指纹。
- 只修改验证文档：不影响生产阶段缓存。

### 7.2 缓存布局

```text
.mlu-triton-cache/
├── extractor/<fingerprint>/
│   ├── metadata.json
│   └── artifacts/Extractor/requirement.md
├── kernel_gen/<fingerprint>/
│   ├── metadata.json
│   └── artifacts/KernelGen/...
└── optimizer/<fingerprint>/
    ├── metadata.json
    └── artifacts/Optimizer/...
```

缓存写入规则：

- `<stage>/<fingerprint>` 一旦存在即视为不可变。
- 写入时复制产物、复算哈希，最后原子写入 `metadata.json`。
- 已有条目若损坏，控制器阻断并报告；不会静默覆盖同一 key。

缓存恢复规则：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py restore \
  --manifest <manifest> --stage <stage>
```

- `next` 必须先返回同一阶段的 `restore`。
- 先验证 metadata 身份和缓存内每个文件哈希。
- 使用临时文件复制，再原子替换输出文件。
- 恢复后阶段状态是 `cached`，summary 必须披露 cache hit。
- 绝对产物路径、`..` 路径或 output/cache 根目录之外的路径一律拒绝。

### 7.3 为什么不缓存 EnvConfig 和 finalize

- EnvConfig 是设备健康检查，旧结果不能证明设备当前可用。
- finalize 成本低，而且 summary 必须反映本次 attempts、cache hit 和失效事件。

## 8. Code Gen 内部检查点

外层 `kernel_gen` 可能包含多个 Subagent 和动态验证，是最容易因中断浪费时间的阶段。v4 在它处于 `running` 时记录 Step 检查点：

| name | 产物 |
|---|---|
| `step1` | `step1_base_info.json`、`step1_io_shapes.json` |
| `step2` | `step2_block_mapping.json` |
| `step3` | `step3_axis_fusion.json` |
| `step4` | `step4_code_spec.json` |
| `step5` | `step5_kernel_code.py` |
| `step6` | `step6_test_code.py` |
| `step7` | `step6_test_code_fix.py` |

保存：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py checkpoint-save \
  --manifest <manifest> --stage kernel_gen --name step3 \
  --artifact KernelGen/step3_axis_fusion.json
```

恢复前验证：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py checkpoint-status \
  --manifest <manifest> --stage kernel_gen --name step3
```

返回 `reusable=true` 才能跳过 Step。检查点包含当前外层阶段指纹和产物哈希；输入或 Skill 源变化时会被清除。Triton 输入快速路径不创建被跳过步骤的伪检查点。

## 9. 中断续跑

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py resume \
  --manifest <output_dir>/run_manifest.json
```

恢复算法：

1. 重新计算所有阶段指纹。
2. 从最早变化的阶段开始失效全部下游。
3. 对 `complete` / `cached` 阶段复算输出产物哈希。
4. 把 `running` / `failed` 阶段重新打开为 `pending`。
5. 保留同一指纹下有效的 Code Gen 内部检查点。
6. 返回下一步 `run`、`restore`、`blocked` 或 `done`。

人工要求强制重做时：

```bash
python -B .claude/skills/mlu-triton-main/scripts/run_control.py invalidate \
  --manifest <manifest> --stage kernel_gen --reason manual_rebuild
```

`invalidate` 不删除文件；它只撤销信任，防止破坏性清理。

## 10. Optimizer 的预算续跑

v3 的 `optimization_state.json` 是优化阶段唯一全局预算。v4 禁止用普通 `plan` 覆盖已有状态。

首次初始化：

```bash
python -B .claude/skills/mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <output_dir>/Optimizer --mode balanced
```

中断恢复：

```bash
python -B .claude/skills/mlu-triton-optimize/scripts/optimization_control.py plan \
  --input <kernel.py> --output-dir <output_dir>/Optimizer --mode balanced --resume
```

`--resume` 验证：

- plan/state 同时存在；
- 输入 SHA-256 相同；
- mode 和 limits 相同；
- OOB 选中列表和 advanced 路由相同；
- state 指向当前 plan。

成功时不改写任何状态。验证失败时退出码为 2，要求外层显式失效或换新输出目录。它不会自动删除已有结果。

## 11. 持续回归数据契约

Main 的 finalize 必须生成 `{output_dir}/regression_result.json`。回归输入采用规范化 JSON，而不是让比较器从自由文本报告猜字段：

```json
{
  "schema_version": 1,
  "run_id": "nightly-2026-08-25",
  "hardware_key": "MLU370-X8:cluster-16:memory-48GB",
  "toolchain_key": "neuware-1.18:triton-3.0:torch-2.4",
  "cases": [
    {
      "case_id": "reduce_sum_fp32_4096",
      "status": "completed",
      "accuracy": {"pass": true, "atol": 0.0001, "rtol": 0.0001, "max_diff": 0.00001},
      "performance": {"latency_ms": 1.03},
      "resources": {
        "wall_time_sec": 650.0,
        "token_count": 11000,
        "subagent_calls": 4,
        "worker_calls": 3
      }
    }
  ]
}
```

数据采集原则：

- `accuracy`、`latency_ms` 必须来自真实 MLU 运行。
- `wall_time_sec` 来自任务开始/结束时间。
- `token_count` 来自平台用量记录；平台不可提供时不要伪造 0，应在上游采集契约中标记缺失并调整策略。
- `subagent_calls`、`worker_calls` 从清单和优化状态审计记录汇总。
- `case_id` 在基线和当前运行间必须稳定且唯一。

## 12. 回归策略和判定

默认 `validation/regression_policy.json`：

| 规则 | 默认值 |
|---|---:|
| 硬件/工具链必须一致 | true |
| 精度必须通过 | true |
| 延迟指标必须存在 | true |
| 耗时/Token/调用数必须存在 | true |
| 最大延迟回退 | 5% |
| 最大总耗时回退 | 20% |
| 最大 Token 回退 | 20% |
| 最大 Subagent 调用增加 | 1 |
| 最大 Worker 调用增加 | 1 |

执行：

```bash
python -B validation/regression.py compare \
  --baseline <baseline.json> \
  --current <current.json> \
  --policy validation/regression_policy.json \
  --report-json <regression_report.json> \
  --report-md <regression_report.md>
```

退出码：

| 退出码 | 含义 |
|---:|---|
| 0 | 所有门禁通过 |
| 1 | 输入合法，但存在回归 |
| 2 | 输入、策略或输出文件错误，无法判定 |

判定细节：

- 当前用例缺失、未完成或精度失败：失败。
- 延迟回退超过阈值：失败。
- 总耗时、Token、Subagent 或 Worker 调用缺失/超过各自阈值：失败；若平台确实无法提供某项，必须显式修改项目策略，不能静默按 0 处理。
- 硬件或工具链不同：性能标为不可比；默认策略同时产生全局阻断，不能当作通过。
- 当前新增但没有基线的用例：`not_comparable` 警告，不冒充性能通过。

## 13. 持续回归分层

### PR 门禁：每次 Skill 修改

无需 MLU，目标在数分钟内完成：

```bash
python -B validation/validate.py all
```

覆盖语法、契约、Schema、路由、预算、缓存、恢复、损坏拒绝和回归比较器正负样例。

### Nightly：代表性 MLU 用例

至少包含：

- Elementwise：基础 load/store 与数学函数各一例。
- Reduce：单轴与多轴各一例。
- Transpose/非连续访问：至少一例。
- `correctness`、`balanced`、`max-performance` 各有代表用例。

步骤：真实 EnvConfig → 运行用例 → 生成规范化 current JSON → 与固定基线比较 → 保存 JSON/Markdown 报告。

### Release：完整矩阵

按支持的 MLU 型号、NeuWare/Triton/PyTorch 组合分别维护基线。不同上下文绝不共用性能基线。发布门禁还需运行 `artifacts --require-complete` 并人工审计所有 `not_comparable`。

## 14. 基线治理

- 基线文件应纳入版本控制或不可变制品库。
- 基线更新必须有真实 MLU 报告、变更原因和审核人，不允许回归失败后自动覆盖。
- 新工具链建立新 baseline key；不要修改旧 key 来强行可比。
- 性能数据建议用稳定统计值生成，保持相同预热、重复次数和负载。
- 回归报告与当前结果一起保留，便于追踪哪条阈值触发。

## 15. 故障处理矩阵

| 现象 | 原因 | 处理 |
|---|---|---|
| `manifest already exists` | 对已有运行再次 init | 使用 `status` 或 `resume` |
| `run_context_not_bound` | EnvConfig 后未绑定上下文 | 生成并绑定 `run_context.json` |
| `artifact_hash_mismatch` | 已完成产物被修改 | 让控制器失效并恢复缓存或重跑 |
| `cache_hash_mismatch` | 共享缓存损坏 | 停止使用该缓存目录，保留证据后由管理员处理 |
| `stage_already_running` | 未声明中断就重复开始 | 中断后先执行 `resume` |
| `run_resume_before_retry` | 阶段处于 failed | 执行 `resume`，不要手改 JSON |
| `checkpoint_hash_mismatch` | 内部产物被修改 | 该检查点会被拒绝，从对应 Step 重做 |
| Optimizer 提示已有状态 | 普通 plan 试图覆盖预算 | 使用 `plan --resume` |
| Optimizer incompatible | 输入/模式/预算/路由变化 | 外层 invalidate optimizer 或使用新输出目录 |
| context mismatch | 基线与当前硬件/工具链不同 | 选择匹配基线，不比较性能 |

## 16. 安全与并发

- 清单和报告使用同目录临时文件 + `os.replace` 原子写入。
- 缓存恢复逐文件校验，拒绝绝对路径、`..` 和越界路径。
- 缓存不执行自动删除；需要回收时由管理员按精确路径和保留策略处理。
- 单个清单采用单写者模型；并行回归应为每个用例使用独立 `output_dir`。
- 缓存可由不同输出目录共享，但只有完全相同指纹才会命中。

## 17. 从 v3 迁移

1. 将 v4 整体部署为目标项目的 `.claude/skills`，不要只覆盖一个脚本。
2. 新运行使用新 `output_dir` 并执行 `init`。
3. 旧 v3 输出没有 `run_manifest.json`，不能自动声明为可信缓存命中。
4. 若要保留旧结果，只把它作为人工参考或回归 baseline；不要手工构造 `complete` 状态。
5. 第一次 v4 运行会建立缓存；第二次相同输入/上下文才验证跨运行命中。
6. 为每个真实硬件/工具链组合建立独立回归基线。

## 18. 验收标准

阶段 4 完成需同时满足：

- L1/L2 离线验证全部通过。
- 相同请求和上下文的第二次运行能恢复 extractor、kernel_gen 或 optimizer 缓存。
- 修改/损坏已完成产物会使当前阶段及下游失效。
- Code Gen 中断后有效内部检查点仍可复用。
- Optimizer `--resume` 保留已消耗预算，普通 plan 拒绝覆盖。
- 回归正例退出 0，精度/性能负例退出 1，非法输入退出 2。
- 不同硬件/工具链不会生成可比较的性能通过结论。
- 至少一个 Elementwise 和一个 Reduce 用例完成真实 MLU nightly 回归后，才能宣称动态验收完成。

当前仓库的离线验证能够确认控制逻辑和契约，但不能代替最后一项真实 MLU 验收。
