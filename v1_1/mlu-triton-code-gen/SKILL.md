---
name: mlu-triton-code-gen
description: 为 MLU 生成、测试并验证 Triton Kernel。普通需求使用“设计 + 构建”两个串行代理，已有 Triton 代码走单代理快速路径；保持既有 Step 1-7 产物契约并降低 Token 与代理调度开销。
---

# mlu-triton-code-gen（v1_1）

## 目标与边界

本 Skill 接收 `{output_dir}/Extractor/requirement.md`，生成可执行测试文件，调用
`mlu-triton-code-review` 验证，最后交付：

- `{output_dir}/KernelGen/triton_code_fix.py`
- `{output_dir}/KernelGen/triton_report.md`
- `{output_dir}/KernelGen/dispatch_metrics.json`

阶段 2 只优化调度与上下文，不改变 Step 1-7 的文件名、JSON 语义、精度标准或 MLU
执行后端规则。旧的六个细粒度角色文档仅用于历史审计，正常流程不得再调度它们。

## 输入与路由

读取 `{output_dir}/Extractor/requirement.md` 的 `输入类型` / `Input Type`：

| 路由 | 条件 | 调度 |
|---|---|---|
| `normal` | 非 Triton 需求 | `DesignKernel` → `BuildKernel` |
| `triton_fast` | 输入类型为 `triton` | 跳过设计，仅调度 `BuildKernel` |

所有代理均为阻塞串行调用；只在消息中传递绝对文件路径、路由和输出目录，不复制需求正文、
平台规则或上一步 JSON 到消息中。

## Step 1-4：单次设计调度（仅 normal）

`DesignKernel` 在一次上下文中依次完成结构提取、块映射、轴融合和代码规范，仍写出全部
兼容产物：

1. `step1_base_info.json`
2. `step1_io_shapes.json`
3. `step2_block_mapping.json`
4. `step3_axis_fusion.json`
5. `step4_code_spec.json`

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
读取绝对路径 {skill_root}/mlu-triton-code-gen/subagents/DesignKernel.md，
按其中契约完成一次串行设计任务。

route: normal
requirement_path: {requirement_path}
kernelgen_dir: {output_dir}/KernelGen
artifact_contract: {skill_root}/mlu-triton-code-gen/references/artifact-contracts.md
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md

只通过上述路径读取上下文；结果全部写入 kernelgen_dir，不在回复中粘贴 JSON 正文。
"""
)
```

调度返回后，确认五个文件存在、JSON 可解析，且
`step1_io_shapes.json == step1_base_info.json["io_shapes"]`。若校验失败，DesignKernel 只在
当前上下文内自检修正一次；仍失败则停止，不进入构建，不自动重新调度代理。

## Step 5-6：单次构建调度

`BuildKernel` 在同一上下文中生成 kernel 与完整测试，避免把 kernel 代码再次传给第二个代理。

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
读取绝对路径 {skill_root}/mlu-triton-code-gen/subagents/BuildKernel.md，
按其中契约完成一次串行构建任务。

route: {route}
requirement_path: {requirement_path}
original_code_path: {output_dir}/Extractor/original_code.py
kernelgen_dir: {output_dir}/KernelGen
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md

normal 路由读取 kernelgen_dir 下的 step1_io_shapes.json 与 step4_code_spec.json；
triton_fast 路由读取 original_code_path。结果写入 step5_kernel_code.py 与
step6_test_code.py，不执行测试，也不在回复中粘贴代码正文。
"""
)
```

调度返回后必须确认：

- `step5_kernel_code.py` 与 `step6_test_code.py` 存在且非空；
- 两个文件均可通过 Python AST 解析；
- `step6_test_code.py` 原样包含 Step 5 的 kernel/wrapper；
- 测试包含输入构造、PyTorch 参考实现、`torch.allclose` 和
  `triton.testing.do_bench`。

代理在当前上下文内自检修正一次；仍失败则停止，不自动重新调度，也不得回退到六代理旧链路。

## Step 7：执行优先验证

调用文件契约，不传代码正文：

```python
Skill(
    skill="mlu-triton-code-review",
    args="{output_dir}/KernelGen/step6_test_code.py"
)
```

Code Review 首轮直接执行成功时不调度代理；失败时最多调度一个 `ReviewAndFix` 代理。动态
执行必须读取 `{output_dir}/EnvConfig/config.md`：`local` 在本地同步执行，`worker` 使用
`mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 前台同步提交。基础设施错误不得当作
kernel 错误修复。

验证结束后必须存在：

- `{output_dir}/KernelGen/step6_test_code_fix.py`
- `{output_dir}/KernelGen/step6_test_code_fix.md`

将前者复制为 `{output_dir}/KernelGen/triton_code_fix.py`，将后者复制或整理为
`{output_dir}/KernelGen/triton_report.md`。若报告结论为未收敛，不得声称生成成功。

## 调度度量

根据实际路由和 Code Review 是否进入修复代理，执行：

```bash
python <skill_root>/mlu-triton-code-gen/scripts/dispatch_metrics.py analyze \
  --route normal \
  --outcome direct-pass \
  --output <output_dir>/KernelGen/dispatch_metrics.json
```

`--route` 可取 `normal` / `triton-fast`，`--outcome` 可取 `direct-pass` / `repair`。
该报告给出精确代理次数以及静态调度上下文字节代理指标；它不是运行时 tokenizer 的精确
Token 计费值，具体口径见 `references/dispatch-contract.json`。

## 不变量

- 不并发调度代理，不后台提交 Worker。
- 普通路径固定两个 Code Gen 代理；快速路径固定一个。
- 代理消息只传路径和短枚举值，所有大内容通过文件交接。
- 不让 Build 读取 Step 2/3 中间文件；它只依赖需求、Step 1 的形状与 Step 4 规范。
- 不在本 Skill 重复平台规则、原语表或测试模板；需要时由负责代理按给定路径读取一次。
- 任何修复都不得退化为 CPU、纯 PyTorch 替代或标量逐元素 kernel。

## 输出摘要

最终只返回路径与状态，不回填大段代码：

```text
route: normal | triton_fast
design: completed | skipped | failed
build: completed | failed
review: direct_pass | repaired | failed
final_code_path: <absolute path>
report_path: <absolute path>
dispatch_metrics_path: <absolute path>
```
