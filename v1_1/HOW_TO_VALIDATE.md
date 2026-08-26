# v1_1 验证说明

本文验证阶段 2 的结构、调度预算、上下文代理指标和真实 MLU 行为。除最后的集成验证外，
均不需要 MLU、PyYAML、pytest 或网络。

## 1. 准备 Python

在 `v1_1` 根目录使用任意 Python 3.9+。推荐加 `-B`，避免生成 `__pycache__`：

```powershell
python -B validation/validate.py
```

若 Windows 的 `python.exe` 是 Microsoft Store 占位符，请改用已安装的真实 Python 绝对路径：

```powershell
& 'C:\path\to\python.exe' -B validation\validate.py
```

当前 Codex 工作区可使用其依赖运行时，但该路径只属于本机，不应写入 Skill 源码。

## 2. 一键离线验证

```powershell
python -B validation/validate.py
```

预期：退出码 `0`、`status` 为 `PASS`、`errors` 为空。验证内容包括：

- 全部 `SKILL.md` frontmatter；
- 全部 Python AST 与 JSON 解析；
- 新角色、产物契约、调度契约和度量脚本存在；
- Main 恰好 2 个代理调用点，Code Gen 恰好 2 个，Code Review 恰好 1 个；
- 活跃 Skill 不再引用旧六个 Code Gen 角色和旧两个 Review 角色；
- Step 1-6 兼容产物名、精度和性能测试契约存在；
- 四种路由/Review 组合的调度次数与上下文降幅符合 fixture。

当前基线预期：

| 场景 | 调度 v1→v1_1 | 调度降幅 | 静态上下文降幅 |
|---|---:|---:|---:|
| normal/direct-pass | 8→4 | 50% | 67.99% |
| normal/repair | 10→5 | 50% | 57.31% |
| triton-fast/direct-pass | 3→3 | 0% | 25.80% |
| triton-fast/repair | 5→4 | 20% | 44.56% |

## 3. 单独检查调度度量

```powershell
python -B mlu-triton-code-gen/scripts/dispatch_metrics.py analyze --route normal --outcome direct-pass
python -B mlu-triton-code-gen/scripts/dispatch_metrics.py analyze --route normal --outcome repair
python -B mlu-triton-code-gen/scripts/dispatch_metrics.py analyze --route triton-fast --outcome direct-pass
python -B mlu-triton-code-gen/scripts/dispatch_metrics.py analyze --route triton-fast --outcome repair
```

要模拟实际工作流落盘：

```powershell
python -B mlu-triton-code-gen/scripts/dispatch_metrics.py analyze `
  --route normal `
  --outcome direct-pass `
  --output output_dir/KernelGen/dispatch_metrics.json
```

检查 JSON 的 `measurement_note.excluded`。如果有人把 `static_context_bytes` 描述成实际计费
Token，验证评审应判失败；真实 Token 必须由宿主平台遥测提供。

## 4. 人工结构审查

执行：

```powershell
rg -n "spawn_agent\(" mlu-triton-main mlu-triton-code-gen mlu-triton-code-review
rg -n "DesignKernel|BuildKernel|ReviewAndFix" mlu-triton-code-gen mlu-triton-code-review
```

预期活跃入口：

- Main：EnvConfig、Extractor 两处；
- Code Gen：DesignKernel、BuildKernel 两处；
- Code Review：ReviewAndFix 一处。

旧角色文档仍可由 `rg --files` 找到，但只能出现在历史基线和度量契约中，不能出现在两个活跃
Skill 的调度代码中。

## 5. 产物契约验证

对一次普通需求运行检查：

```powershell
$out = 'D:\absolute\path\to\output_dir'
Get-Item "$out\KernelGen\step1_base_info.json"
Get-Item "$out\KernelGen\step1_io_shapes.json"
Get-Item "$out\KernelGen\step2_block_mapping.json"
Get-Item "$out\KernelGen\step3_axis_fusion.json"
Get-Item "$out\KernelGen\step4_code_spec.json"
Get-Item "$out\KernelGen\step5_kernel_code.py"
Get-Item "$out\KernelGen\step6_test_code.py"
Get-Item "$out\KernelGen\step6_test_code_fix.py"
Get-Item "$out\KernelGen\triton_code_fix.py"
Get-Item "$out\KernelGen\triton_report.md"
Get-Item "$out\KernelGen\dispatch_metrics.json"
```

再用 Python 检查 Step 1 同源约束：

```powershell
python -c "import json,pathlib; p=pathlib.Path(r'D:\absolute\path\to\output_dir\KernelGen'); a=json.loads((p/'step1_base_info.json').read_text(encoding='utf-8')); b=json.loads((p/'step1_io_shapes.json').read_text(encoding='utf-8')); assert a['io_shapes']==b"
```

Triton 快速路径应不存在伪造的 Step 1-4 文件；Build 直接从 `original_code.py` 生成 Step 5/6。

## 6. 真实 MLU 验证矩阵

至少运行以下四类用例：

| 用例 | 输入 | 预期代理路径 | 关键断言 |
|---|---|---|---|
| A | 简单 elementwise 需求 | Main 2 + Design + Build；Review 直接通过 | 总代理 4，精度通过 |
| B | reduction/softmax 需求 | Main 2 + Design + Build；必要时 ReviewAndFix | 最多 5，归约 spec 与结果正确 |
| C | 已有且正确的 Triton 完整代码 | Main 2 + Build；Review 直接通过 | 总代理 3，Step 1-4 跳过 |
| D | 含一个明确可修错误的 Triton 代码 | Main 2 + Build + ReviewAndFix | 总代理 4，报告含真实修复日志 |

每个用例记录：

- 最终 `torch.allclose` 结果与 max diff；
- Triton/PyTorch 性能输出；
- 实际代理启动次数；
- 总耗时；
- 若宿主支持，逐代理 input/output Token；
- `dispatch_metrics.json` 与真实路由是否一致。

本地执行必须由 EnvConfig 确认 `execution_backend=local`。Worker 模式必须在同一 `JOB_ID`
下前台同步调用现有提交脚本，不另建 Job、不并发。

## 7. 负向验证

建议故意验证以下失败：

1. 删除 `step4_code_spec.json` 后启动 Build：必须失败，不得猜测 spec。
2. 让 Step 1 两份 io_shapes 不一致：设计闸门必须失败。
3. EnvConfig 缺失：Code Review 必须返回环境契约错误，不修改 kernel。
4. Worker 不可达：报告基础设施失败，不进入五轮盲修。
5. 连续两轮相同业务错误：ReviewAndFix 停止并写 `not_converged`。
6. 修改调度契约中的普通路径总数：`validation/validate.py` 必须非零退出。
7. 在活跃 Code Gen Skill 重新引用旧角色：离线验证必须失败。

## 8. 与 v1 的文件隔离检查

`v1_1` 不应反向修改 `v1`。可执行只读比较：

```powershell
$v1 = Resolve-Path '..\v1'
$v11 = Resolve-Path '.'
Compare-Object `
  (Get-ChildItem $v1 -Recurse -File | ForEach-Object { $_.FullName.Substring($v1.Path.Length + 1) }) `
  (Get-ChildItem $v11 -Recurse -File | ForEach-Object { $_.FullName.Substring($v11.Path.Length + 1) })
```

要确认原目录内容哈希未变，应在复制前后对 `v1` 分别生成清单并比较。当前交付的最终验证也
会执行这一步。

## 9. 验收顺序

1. 一键离线验证通过；
2. 人工审查三个活跃调度入口；
3. 普通路径 A/B 在 MLU 上通过；
4. 快速路径 C/D 在 MLU 上通过；
5. 对比 v1/v1_1 的精度、耗时、代理数和真实 Token；
6. 确认收益后，再把阶段 2 变更移植到包含 P0 修复的目标分支。
