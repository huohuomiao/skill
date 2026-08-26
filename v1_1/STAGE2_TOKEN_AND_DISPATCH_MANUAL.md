# 阶段 2 详细说明书：降低 Token 和代理调度开销

## 1. 版本定位

`v1_1` 是从原始 `v1` 冻结副本上制作的阶段 2 独立实现，目标是缩短算子生成链路、减少
重复加载同一语义上下文，并保持下游文件契约不变。

本版解决：

1. Code Gen 将同一算子的需求、映射和规范在六个代理间重复传递；
2. Code Review 首轮失败后由静态、动态两个代理重复加载代码与平台规则；
3. 代理消息包含大段重复操作说明，缺少可执行的调度预算与度量口径；
4. 合并代理后容易误删中间产物，导致 Optimize 或人工排查无法复用原接口。

本版不解决：

- P0 契约修复和废弃脚本删除；这些已在 `v2` 实现，但 `v1_1` 是从 `v1` 分出的独立对照
  分支，不自动包含 `v2` 修改；
- 优化策略按需路由和全局优化预算；当前路线中它们属于阶段 3；
- 跨运行缓存、断点续跑和持续回归；它们属于阶段 4；
- 主机侧 tokenizer 的真实计费采集。仓库无法访问宿主模型计费数据，因此只提供可复现的
  静态上下文代理指标。

## 2. 改造前后

### v1 普通路径

```text
EnvConfig → Extractor
  → ExtractBaseInfo
  → TraceBlockMapping
  → AxisFusion
  → GenerateSpec
  → GenerateCode
  → GenTestCode
  → 首轮执行
      ├─ 通过：0 个 Review 代理
      └─ 失败：StaticReviewer → DynamicFixer
```

### v1_1 普通路径

```text
EnvConfig → Extractor
  → DesignKernel（一次生成 Step 1-4 的五个 JSON 检查点）
  → BuildKernel（一次生成 Step 5 kernel + Step 6 完整测试）
  → 首轮执行
      ├─ 通过：0 个 Review 代理
      └─ 失败：ReviewAndFix（静态检查 + 动态修复共用一个上下文）
```

环境识别与需求提取仍保持两个代理。二者职责、失败类型和执行权限不同，合并带来的 Token
收益有限，却会让“环境不可用”和“需求解析失败”难以区分，所以阶段 2 不动它们。

## 3. 调度预算

以下次数包含 Main 的两个固定代理，不包含后续 Optimizer：

| 路由 | Review 结果 | v1 | v1_1 | 降幅 |
|---|---|---:|---:|---:|
| 普通需求 | 原代码直接通过 | 8 | 4 | 50% |
| 普通需求 | 进入修复代理 | 10 | 5 | 50% |
| Triton 快速路径 | 原代码直接通过 | 3 | 3 | 0% |
| Triton 快速路径 | 进入修复代理 | 5 | 4 | 20% |

快速路径在 `v1` 已跳过 Step 1-4，原本只有一个测试生成代理。因此阶段 2 对它主要减少 Review
失败链重复加载，不宣称 50% 调度收益。

## 4. Token 代理指标

### 4.1 口径

`dispatch_metrics.py` 对每次代理调用会读取的稳定角色文档和共享规则文件求字节数；同一个
文件若被两个代理分别加载则计算两次。这能稳定反映“仓库内重复上下文”，无需第三方库。

该指标不包含：

- 宿主系统提示词和 tokenizer 细节；
- 每个算子不同的 requirement、JSON、代码和运行日志；
- 未改动的 EnvConfig/Extractor 两个代理；
- 模型输出 Token。

因此它适合作为版本回归门禁，不能直接等同于 API 账单 Token。若运行平台能导出每次代理
的 `input_tokens` / `output_tokens`，应在真实用例矩阵中额外记录。

### 4.2 当前测量结果

| 路由 | Review 结果 | v1 静态上下文 | v1_1 静态上下文 | 降幅 |
|---|---|---:|---:|---:|
| 普通需求 | 直接通过 | 96,018 B | 30,731 B | 67.99% |
| 普通需求 | 进入修复 | 201,873 B | 86,174 B | 57.31% |
| Triton 快速路径 | 直接通过 | 17,299 B | 12,836 B | 25.80% |
| Triton 快速路径 | 进入修复 | 123,154 B | 68,279 B | 44.56% |

普通路径门禁是：代理调度至少下降 50%，静态调度上下文至少下降 50%。两种普通路径均已
通过。

## 5. Code Gen 合并策略

### 5.1 DesignKernel

`DesignKernel` 一次读取 requirement、产物字段契约、MLU 原语表和平台规则，在同一上下文内
顺序完成：

1. 结构化信息与 io shape；
2. block mapping；
3. axis fusion；
4. kernel/wrapper spec。

合并不代表取消检查点。它仍逐个写出：

- `step1_base_info.json`
- `step1_io_shapes.json`
- `step2_block_mapping.json`
- `step3_axis_fusion.json`
- `step4_code_spec.json`

每一步写下一步前执行一致性闸门。字段定义集中到
`mlu-triton-code-gen/references/artifact-contracts.md`，避免六个角色分别维护重复模板。

### 5.2 BuildKernel

`BuildKernel` 在一个上下文中生成 `step5_kernel_code.py` 与 `step6_test_code.py`。测试生成不再
由另一个代理重新加载 kernel 和平台规则。

普通路径只读取 requirement、Step 1 shape、Step 4 spec 和两份共享规则；禁止读取 Step 2/3
中间文件。Triton 快速路径只读取 requirement、`original_code.py` 和共享规则。

Step 6 必须原样保留 Step 5 kernel/wrapper，并追加输入构造、PyTorch truth、精度验证和
`triton.testing.do_bench()` 性能测试。构建代理不执行代码，真实运行只有 Code Review 负责。

## 6. Code Review 合并策略

Code Review 保留“执行优先”：

- 原代码直接通过：不启动任何代理，原样生成 fix 文件和报告；
- 业务失败：只启动一个 `ReviewAndFix`；
- 环境/Worker/路径错误：停止，不修改 kernel。

`ReviewAndFix` 在一个上下文中先做保守静态检查，再使用固定 EnvConfig 后端真实执行并进行
最多五轮修复。通过、连续两次相同错误、五轮耗尽或基础设施错误都会终止。

CPU 替代、纯 PyTorch 替代和标量逐元素 kernel 仍是红线。

## 7. 文件传递与上下文隔离

代理消息只包含：角色文档路径、输入产物路径、输出目录、路由枚举和共享规则路径。消息中
不复制 requirement 正文、代码、JSON、平台规则或执行日志。

主流程只消费状态和路径；大结果落盘。这同时降低：

- 调度提示词长度；
- 代理回复污染主上下文；
- 多次转述造成的字段漂移；
- 同一代码在 Review 两个角色间重复加载。

## 8. 兼容性

保持不变：

- Main 的 EnvConfig → Extractor 顺序；
- Step 1-7 文件名与核心字段；
- Code Review 的 `xxx.py → xxx_fix.py + xxx_fix.md` 契约；
- `triton_code_fix.py`、`triton_report.md` 下游接口；
- 本地优先、Worker 前台同步、同一 `JOB_ID` 的环境规则；
- Optimizer 的入口与现有策略实现。

新增 `dispatch_metrics.json` 作为非侵入式观测产物；旧下游可忽略它。

## 9. 失败、重试与权衡

- Design 或 Build 的输出闸门失败时，只在当前代理上下文内修正一次；仍失败则停止，不自动
  重调代理。这样失败路径也不会悄悄突破调度预算。
- 禁止失败后回退到旧六代理链，否则调度上限失效。
- 合并后单次代理上下文更长，但消除了四次中间调度和重复规则加载；测量结果显示总稳定
  上下文明显下降。
- Step 1-4 的故障重跑粒度从单步变为设计组，Step 5-6 从单步变为构建组。阶段 4 的检查点/
  续跑能力用于进一步解决这一权衡，`v1_1` 本身不实现跨运行恢复。

## 10. 部署建议

`v1_1` 是阶段 2 隔离验证版。若要形成包含全部阶段的生产版本，不应直接把 `v1_1` 覆盖到
`v3_1` 或 `v4`；应把本版的 Code Gen/Review 合并和调度契约移植到已包含 P0 修复的分支，
再运行双方验证套件。

回滚时整套恢复到原 `v1`；不要只恢复 Skill 入口而保留新角色文档或相反，否则调度契约与
实现不一致。

## 11. 主要文件

| 文件 | 作用 |
|---|---|
| `mlu-triton-code-gen/SKILL.md` | 两代理 Code Gen 总调度 |
| `mlu-triton-code-gen/subagents/DesignKernel.md` | Step 1-4 合并角色 |
| `mlu-triton-code-gen/subagents/BuildKernel.md` | Step 5-6 合并角色 |
| `mlu-triton-code-gen/references/artifact-contracts.md` | 兼容产物字段真源 |
| `mlu-triton-code-gen/references/dispatch-contract.json` | v1/v1_1 调度与上下文口径 |
| `mlu-triton-code-gen/scripts/dispatch_metrics.py` | 可执行度量工具 |
| `mlu-triton-code-review/SKILL.md` | 执行优先、失败时单代理调度 |
| `mlu-triton-code-review/ReviewAndFix.md` | 静态 + 动态合并角色 |
| `validation/validate.py` | 标准库离线验证 |
