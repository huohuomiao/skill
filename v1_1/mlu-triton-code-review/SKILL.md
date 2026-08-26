---
name: mlu-triton-code-review
description: 执行优先验证 MLU Triton 完整测试文件；直接通过时零代理，失败时用一个 ReviewAndFix 代理完成静态检查和最多五轮动态修复。
---

# mlu-triton-code-review（v1_1）

## 契约

输入仅接受一个绝对 `.py` 文件路径，不接受代码片段或额外输出目录。假设输入为 `xxx.py`，
输出固定写在同目录：

| 文件 | 含义 |
|---|---|
| `xxx_fix.py` | 最终候选代码；直接通过时为原文件原样拷贝 |
| `xxx_fix.md` | 首轮执行、静态检查、修复迭代和最终结论 |

所有大内容通过文件传递，不把代码、日志或报告全文返回调用方。

## 1. 确认执行后端

从输入目录向上定位 `{output_dir}/EnvConfig/config.md`，读取 `execution_backend`：

- `local`：在输入文件所在目录同步执行 `python <input_code_path>`；
- `worker`：使用 `mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 前台同步提交，
  `--timeout-sec 1800 --task-type accuracy`；
- 缺失或未知：停止并返回环境契约错误，不修改 kernel。

禁止并发 Worker、后台 `&`、另建 Job 或绕过提交脚本。

退出码分类：`0` 为执行和精度断言通过；`1` 为可修复的业务/精度错误；`2` 或明确设备、
Worker、路径不可用为基础设施错误。基础设施错误只记录，不进入代码修复。

## 2. 执行优先

先执行原文件一次：

- 通过：原样复制为 `xxx_fix.py`，在 `xxx_fix.md` 写入后端、命令、退出码和“无需修改”，
  立即结束；本路径不得调度代理。
- 业务失败：记录首轮日志摘要后进入单代理修复。
- 基础设施失败：写报告并停止，不创建虚假的通过产物。

## 3. 单代理静态 + 动态修复

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
读取绝对路径 {skill_root}/mlu-triton-code-review/ReviewAndFix.md，按其契约在一个上下文内
完成静态检查、生成 fix 文件、真实执行和最多五轮动态修复。

input_code_path: {input_code_path}
initial_log_path: {initial_log_path}
env_config_path: {output_dir}/EnvConfig/config.md
primitives_path: {skill_root}/share/mlu/references/primitives.md
platform_rules_path: {skill_root}/share/mlu/references/platform-rules.md
libdevice_path: {skill_root}/share/mlu/references/libdevice.md
common_error_path: {skill_root}/mlu-triton-code-review/ref/common_error.md
troubleshooting_path: {skill_root}/mlu-triton-code-review/ref/troubleshooting.md
report_template_path: {skill_root}/mlu-triton-code-review/ref/report_template.md
worker_submit_path: {skill_root}/mlu-triton-main/subagents/scripts/submit_task_to_worker.py

只传递文件路径。结果写入与输入同目录的 xxx_fix.py 和 xxx_fix.md；回复仅给状态和路径。
"""
)
```

代理结束后，主流程只检查两个输出存在、Python AST 可解析，以及报告中存在明确结论；不再
重复执行或解析错误。报告为“未收敛/环境失败”时，向上游返回失败。

## 修复红线

- 禁止把 Triton kernel 改为 CPU 实现。
- 禁止用纯 PyTorch 计算替代 Triton kernel。
- 禁止把 tile 并行 kernel 退化为标量逐元素循环。
- 每轮必须基于真实 stderr、Traceback 或精度数值做最小改动。
- 动态循环达到任一条件即停止：通过；连续两次完全相同错误；累计五轮修复。

## 调度上限

| 路径 | v1 | v1_1 |
|---|---:|---:|
| 原代码直接通过 | 0 | 0 |
| 原代码失败且需修复 | 2 | 1 |

旧的两个角色文档保留用于审计，但本 Skill 不再引用或调度它们。
