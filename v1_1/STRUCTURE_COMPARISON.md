# v1 与 v1_1 结构变化对照

## 1. 变更范围

`v1_1` 从 `v1` 完整复制后独立修改。四个 Skill 名称、Main 的两代理初始化、Step 1-7 产物
文件名、Worker 提交接口和 Optimizer 目录保持不变。

本版修改两个活跃 Skill 入口，新增三个合并角色、两个契约、一个度量脚本、一套离线验证和
四份交付文档。六个旧 Code Gen 角色与两个旧 Review 角色未删除，作为度量和回滚审计基线，
但不再由活跃入口调度。

最终文件统计：`v1` 59 个文件，`v1_1` 71 个文件；新增 12 个、修改 2 个、删除 0 个，
其余 57 个原文件内容保持一致。

## 2. 文件级变化

| 路径 | v1 | v1_1 | 说明 |
|---|---|---|---|
| `mlu-triton-code-gen/SKILL.md` | 六个串行 Code Gen 代理 | 已修改 | 普通路径收敛为 Design + Build 两个代理 |
| `mlu-triton-code-review/SKILL.md` | 失败后 Static + Dynamic 两代理 | 已修改 | 失败后单个 ReviewAndFix 代理 |
| `mlu-triton-code-gen/subagents/DesignKernel.md` | 无 | 新增 | 合并 Step 1-4，保留五个 JSON 检查点 |
| `mlu-triton-code-gen/subagents/BuildKernel.md` | 无 | 新增 | 合并 Step 5-6，kernel 与测试共用上下文 |
| `mlu-triton-code-review/ReviewAndFix.md` | 无 | 新增 | 合并静态检查与动态修复 |
| `mlu-triton-code-gen/references/artifact-contracts.md` | 字段散落在多个角色文档 | 新增 | Step 1-4 字段真源与跨文件不变量 |
| `mlu-triton-code-gen/references/dispatch-contract.json` | 无 | 新增 | 路由、调度数和静态上下文度量口径 |
| `mlu-triton-code-gen/scripts/dispatch_metrics.py` | 无 | 新增 | 标准库调度/上下文度量工具 |
| `validation/expected_dispatch_metrics.json` | 无 | 新增 | 四种路径的离线期望值 |
| `validation/validate.py` | 无 | 新增 | 结构、语法、契约和行为检查 |
| `STAGE2_TOKEN_AND_DISPATCH_MANUAL.md` | 无 | 新增 | 阶段 2 详细说明书 |
| `HOW_TO_VALIDATE.md` | 无 | 新增 | 离线与 MLU 验证说明 |
| `STRUCTURE_COMPARISON.md` | 无 | 新增 | 本结构对照 |
| `VERSION_STAGE_MAP.md` | 无 | 新增 | 全版本与当前阶段映射 |

## 3. 目录结构对照

### v1 相关结构

```text
v1/
├─ mlu-triton-code-gen/
│  ├─ SKILL.md                  # 六个调度点
│  └─ subagents/
│     ├─ ExtractBaseInfo.md
│     ├─ TraceBlockMapping.md
│     ├─ AxisFusion.md
│     ├─ GenerateSpec.md
│     ├─ GenerateCode.md
│     └─ GenTestCode.md
└─ mlu-triton-code-review/
   ├─ SKILL.md                  # 失败时两个调度点
   ├─ StaticReviewer.md
   └─ DynamicFixer.md
```

### v1_1 相关结构

```text
v1_1/
├─ mlu-triton-code-gen/
│  ├─ SKILL.md                  # 两个调度点
│  ├─ scripts/
│  │  └─ dispatch_metrics.py
│  ├─ references/
│  │  ├─ artifact-contracts.md
│  │  └─ dispatch-contract.json
│  └─ subagents/
│     ├─ DesignKernel.md        # 活跃：Step 1-4
│     ├─ BuildKernel.md         # 活跃：Step 5-6
│     └─ <六个旧角色>.md         # 仅审计，不调度
├─ mlu-triton-code-review/
│  ├─ SKILL.md                  # 失败时一个调度点
│  ├─ ReviewAndFix.md           # 活跃
│  ├─ StaticReviewer.md         # 仅审计
│  └─ DynamicFixer.md           # 仅审计
├─ validation/
│  ├─ expected_dispatch_metrics.json
│  └─ validate.py
├─ STAGE2_TOKEN_AND_DISPATCH_MANUAL.md
├─ HOW_TO_VALIDATE.md
├─ STRUCTURE_COMPARISON.md
└─ VERSION_STAGE_MAP.md
```

## 4. 行为变化

| 场景 | v1 | v1_1 |
|---|---|---|
| 普通 Code Gen | 六个串行代理 | 两个串行代理 |
| Triton 快速路径 | 直接复制 + 一个测试代理 | 一个 Build 代理完成接入和测试 |
| Review 首轮通过 | 零代理 | 零代理 |
| Review 首轮失败 | 静态代理后再动态代理 | 单代理连续完成静态和动态 |
| 代理消息 | 多处重复操作说明和输入描述 | 只传路径、路由和输出目录 |
| Step 1-7 产物 | 全部存在 | 文件名与核心语义保持兼容 |
| 调度度量 | 无 | `dispatch_metrics.json` + 可执行脚本 |
| 离线门禁 | 无 | 结构、语法、路由、降幅 fixture |

## 5. 保持不变与未合入内容

保持不变：EnvConfig、Extractor、共享 MLU 规则、Optimizer 全目录、Worker 提交脚本和旧角色
内容。

因为本版直接从 `v1` 分支，`v1` 中的历史 P0 问题和废弃 `analyzer_rep.py` 也仍在本目录；
这些不是阶段 2 的新增问题，但意味着 `v1_1` 是调度优化对照版，不是替代 `v2/v3_1/v4`
的累计生产版。全阶段对应关系见 `VERSION_STAGE_MAP.md`。
