# Triton Kernel Workflow 结构调整方案

本次调整只改变 Skill 源码的组织方式，不改变原有执行流程、阶段顺序和运行产物。

## 调整后的 Skill 结构

```text
triton-kernel-workflow/
├── SKILL.md
├── references/
│   ├── workflows/        # 完整流程及各阶段编排
│   ├── roles/            # Subagent 角色说明
│   ├── strategies/       # 性能优化策略及策略专用参考
│   ├── backend/          # 设备规则、原语和数学函数
│   ├── contracts/        # 输入、输出、交接和回退契约
│   ├── diagnostics/      # 静态检查与运行错误处理
│   ├── templates/        # 报告模板
│   ├── examples/         # 按命中场景加载的代码生成和优化示例
│   ├── schemas/          # 环境、候选、状态摘要和评测 Schema
│   └── evals/            # L2 固定场景与模拟产物
└── scripts/
    ├── environment/      # 环境检查脚本
    ├── execution/        # 远端任务执行脚本
    ├── profiling/        # 性能分析脚本
    ├── state/            # 配置、状态摘要和最优候选的确定性更新
    └── validation/       # L1/L2/L3 分层验证
```

`SKILL.md` 作为唯一入口，负责模式选择和流程调度；详细规则按需从 `references/` 读取；可重复执行的确定性操作统一放在 `scripts/` 中。

## 执行流程

```text
环境检查
→ 需求抽取
→ 代码方案与生成
→ 测试代码生成
→ 静态与动态验证
→ 性能优化
→ 输出最终代码和报告
```

## 运行后的产物结构

```text
output_dir/
├── run_manifest.json
├── EnvConfig/
│   ├── config.json
│   ├── config.md
│   └── runtime_info.txt
├── Extractor/
│   └── requirement.md
├── KernelGen/
│   ├── step1_base_info.json
│   ├── step1_io_shapes.json
│   ├── step2_block_mapping.json
│   ├── step3_axis_fusion.json
│   ├── step4_code_spec.json
│   ├── step5_kernel_code.py
│   ├── step6_test_code.py
│   ├── triton_code_fix.py
│   └── triton_report.md
├── Optimizer/
│   ├── {n}_{strategy}/
│   ├── best_so_far.json
│   ├── best_so_far.py
│   ├── triton_optimized.py
│   └── triton_optimized.md
├── triton_final.py
└── summary.md
```

## 相比原结构的优势

1. **入口统一**：不再依赖多个顶层 Skill 互相调用，触发和调度逻辑更清晰。
2. **职责明确**：流程、角色、策略、设备知识和脚本分别管理，减少内容混放。
3. **流程保持稳定**：阶段顺序和产物交接契约不变，结构调整不会影响现有执行结果。
4. **降低路径耦合**：内部统一使用相对路径，减少目录调整时需要修改的硬编码引用。
5. **便于扩展后端**：目录和文件名不绑定特定设备，后续可以增加其他运行后端。
6. **便于维护策略**：新增或删除优化策略时，只需调整 `strategies/` 和对应流程配置。
7. **减少上下文占用**：主入口保持简洁，只有进入具体阶段时才加载对应参考文档。
8. **状态交接精简**：下游从 `run_manifest.json` 获取状态和产物路径，无需重复装载全部阶段报告。
9. **规则来源唯一**：执行规则与平台规则分别只有一个事实源，减少规则漂移和重复维护。
