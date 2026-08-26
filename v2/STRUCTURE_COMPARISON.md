# v1 与 v2 结构变化对照

## 1. 变更范围

`v2` 从 `v1` 完整复制后独立修改。四个 Skill 名称、主工作流阶段、既有输出目录和 Worker 提交接口保持不变。

本版本只包含两类变化：

1. 修复已确认的 P0 契约错误。
2. 增加阶段 1 的 L1 静态检查、L2 离线行为契约和 L3 MLU 集成验证入口。

文件哈希对比结果：`v1` 59 个文件，`v2` 74 个文件；新增 16 个、删除 1 个、修改 4 个，其余 54 个文件内容完全一致。

## 2. 文件级变化

| 路径 | v1 | v2 | 变化说明 |
|---|---|---|---|
| `mlu-triton-code-gen/SKILL.md` | 原文件 | 已修改 | 统一由 Code Review 执行测试；Review 调用改为单文件参数；修复 Step 2 Markdown 表格 |
| `mlu-triton-optimize/SKILL.md` | 原文件 | 已修改 | 深度优化产物存在时优先选择，OOB 改为回退产物 |
| `mlu-triton-main/subagents/EnvConfig.md` | 原文件 | 已修改 | 两个环境脚本的位置修正为 `share/mlu/runtime` |
| `mlu-triton-optimize/kernel-info/references/reducemax-axis-example.md` | 代码围栏缺失 | 已修改 | 补齐 Markdown 代码围栏，使 L1 基线通过 |
| `mlu-triton-optimize/perf-analyzer/scripts/analyzer_rep.py` | 1 字节空脚本 | 已删除 | 明确删除废弃文件及删除后产生的空 `scripts/` 目录 |
| `validation/validate.py` | 无 | 新增 | L1、L2、真实产物检查的统一入口，只有标准库依赖 |
| `validation/run_mlu_integration.py` | 无 | 新增 | L3 本地 MLU 环境和可选算子验证入口 |
| `validation/contracts/*.schema.json` | 无 | 新增 5 个 | Step 1 BaseInfo、IO Shapes、Step 2 Mapping、Step 3 Fusion、Step 4 Spec 契约 |
| `validation/cases/behavior_cases.json` | 无 | 新增 | 7 个核心路由和失败处理场景 |
| `validation/fixtures/valid/KernelGen/*.json` | 无 | 新增 5 个 | Schema 正向测试样例 |
| `validation/fixtures/invalid/step1_missing_required.json` | 无 | 新增 | Schema 负向测试样例 |
| `HOW_TO_VALIDATE.md` | 无 | 新增 | v2 验证方法和变更验证矩阵 |
| `STRUCTURE_COMPARISON.md` | 无 | 新增 | 本结构变化对照文档 |

## 3. 目录结构对照

### v1

```text
v1/
├── mlu-triton-main/
├── mlu-triton-code-gen/
├── mlu-triton-code-review/
├── mlu-triton-optimize/
│   └── perf-analyzer/scripts/analyzer_rep.py
└── share/
```

### v2

```text
v2/
├── mlu-triton-main/
├── mlu-triton-code-gen/
├── mlu-triton-code-review/
├── mlu-triton-optimize/
├── share/
├── validation/
│   ├── validate.py
│   ├── run_mlu_integration.py
│   ├── contracts/
│   │   ├── step1_base_info.schema.json
│   │   ├── step1_io_shapes.schema.json
│   │   ├── step2_block_mapping.schema.json
│   │   ├── step3_axis_fusion.schema.json
│   │   └── step4_code_spec.schema.json
│   ├── cases/
│   │   └── behavior_cases.json
│   └── fixtures/
│       ├── valid/KernelGen/
│       │   ├── step1_base_info.json
│       │   ├── step1_io_shapes.json
│       │   ├── step2_block_mapping.json
│       │   ├── step3_axis_fusion.json
│       │   └── step4_code_spec.json
│       └── invalid/
│           └── step1_missing_required.json
├── HOW_TO_VALIDATE.md
└── STRUCTURE_COMPARISON.md
```

## 4. 行为变化对照

| 场景 | v1 行为 | v2 行为 |
|---|---|---|
| Step 6 测试生成 | 文档一处要求立即执行，另一处要求不执行 | Step 6 只生成，Step 7 Code Review 统一执行 |
| Code Review 调用 | Code Gen 传入文件路径和输出目录两个参数 | 只传 `input_code_path`，输出目录由 Review 从输入文件推导 |
| 深度优化最终选择 | 深度优化执行后错误选择 OOB 代码 | 优先选择存在的 advanced 代码，缺失时回退 OOB |
| EnvConfig 脚本定位 | 表格指向不存在的 Main scripts 目录 | 指向实际的 `share/mlu/runtime` 文件 |
| 空性能脚本 | 保留 1 字节 `analyzer_rep.py` | 删除 |
| Skill 修改验证 | 依赖人工阅读和完整 MLU 流程 | 增加 L1/L2 离线门禁、产物 Schema 检查和 L3 入口 |

## 5. 保持不变的内容

- Skill 名称和触发描述未改变。
- Main 的 EnvConfig、Extractor、Code Gen、Optimize、Summary 五段主流程未重构。
- Code Gen 的 Step 1～7 中间产物名称未改变。
- Code Review 的 `xxx.py -> xxx_fix.py + xxx_fix.md` 输出契约未改变。
- OOB 和 Advanced Optimization 的策略集合未改变。
- Worker 提交脚本、接口、退出码和日志判定方式未改变。
- 本版本尚未实施阶段 2 的文档去重、代理合并、按需策略路由和缓存。
