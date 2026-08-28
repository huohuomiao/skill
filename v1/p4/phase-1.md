# p1 · 阶段 1：三层验证体系

p1 从 v0 的 P0 契约基线演进而来。本阶段不改变算子开发流程，只为 Skill 修改建立由便宜到昂贵的验证门禁。

## 优化内容

- L1 静态检查：验证 Skill frontmatter、引用、Markdown 围栏、空文件与占位符、Python/JSON/Shell 语法、重复规则、文档体积阈值、输入输出命名契约和最终候选选择不变量。
- L2 离线行为评测：使用固定场景与模拟 JSON，验证输入路由、阶段门禁、文件读取白名单、Schema、最小回退和性能结论证据约束。
- L3 硬件集成回归：使用外部真实算子套件串行执行编译、精度、性能、Worker 提交和失败恢复；不附带虚假硬件结果。
- 分层总入口：L3 必须读取同一 Skill 指纹下通过的 L1/L2 报告；L1 超过 30 秒或 L1+L2 超过 5 分钟均视为门禁失败。

## 使用方式

```bash
python triton-kernel-workflow/scripts/validation/run-validation.py --level l1 --report-dir <report-dir>
python triton-kernel-workflow/scripts/validation/run-validation.py --level l2 --report-dir <report-dir>
python triton-kernel-workflow/scripts/validation/run-validation.py --level l3 --report-dir <report-dir> --integration-suite <suite.json>
```

L3 的 `suite.json` 必须符合 `references/schemas/integration-suite.schema.json`，并提供 elementwise、reduction、layout 三类代表算子及 Worker 集成命令。
