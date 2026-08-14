---
name: mlu-bangc-code-review
description: 面向 MLU590 的 BANG C/CNRT 算子代码验证与最小修复工具。用于审查、使用 cncc 编译、在 CNRT 上运行并修复完整 .mlu 文件，检查精度、任务划分、片上存储、数据搬运、intrinsic、launch 与资源生命周期，输出 bangc_code_fix.mlu 和 bangc_report.md。
---

# mlu-bangc-code-review

## 目标

验证一个完整、可独立编译运行的 BANG C `.mlu` 文件。文件必须包含设备 Kernel、Host 侧 CNRT 启动代码、独立 reference 与可自动失败的精度测试。先编译和运行原文件；只有门禁失败时才进入静态检查和有界动态修复。

平台事实统一读取：

- `.claude/skills/share/mlu/references/platform-rules.md`
- `.claude/skills/share/mlu/references/primitives.md`
- `.claude/skills/share/mlu/references/libdevice.md`
- 当前工作流最近的 `{output_dir}/EnvConfig/config.md`

不得根据“MLU590”名称猜测 `cncc` 架构参数、片上容量、Cluster/Core 数、任务类型限制或 intrinsic 支持。未从 EnvConfig、已安装头文件、`cncc --help`、编译日志或真实执行取得的事实写 `N/A`。

## 调用与产物

```text
/mlu-bangc-code-review <input_code_path>
```

仅接受一个现有 `.mlu` 文件路径。输出固定写入输入所在目录：

| 文件 | 契约 |
| --- | --- |
| `bangc_code_fix.mlu` | 最终候选；原文件全门禁通过时逐字复制 |
| `bangc_report.md` | 环境、编译、运行、精度、静态发现和修复迭代证据 |

主链路通常以 `KernelGen/step6_test_code.mlu` 调用本 Skill，并直接接收 `KernelGen/bangc_code_fix.mlu` 与 `KernelGen/bangc_report.md`。不要生成 `*_fix_fix.mlu`。

## 不可违反的规则

- 保留自定义 BANG C Kernel；不得用 Host CPU、框架算子、CNNL 或预计算结果替代设备计算。CPU 实现仅可作为独立 reference。
- 不得删除测试、缩小 Shape、改变 dtype/stride/layout、放宽容差或让失败返回 0。
- 不得把环境、Worker、编译器或链接器缺失误判为 Kernel 错误。
- 保持 Host wrapper 公共接口、Kernel 数学语义、别名/in-place 契约与 Queue 顺序。
- 允许正确的标量控制与尾部处理；禁止把本应使用 MLU Kernel 的主体计算迁回 Host。
- 每轮只做一个可解释的最小修改；新增失败类型时回退该轮。

## 步骤 1：确定执行后端与编译契约

从输入目录向上读取最近的 `EnvConfig/config.md`，至少取得：

- `execution_backend=local|worker`
- 目标设备是否真实确认为 MLU590
- `cncc` 绝对路径和版本
- NeuWare/CNToolkit 根、运行库路径
- 已确认的完整编译/链接命令与可选架构参数
- Worker 上下文（若使用）

编译命令必须来自 EnvConfig。概念形式为：

```bash
<cncc> <source>.mlu -o <temporary_binary> <confirmed_compile_and_link_flags>
```

兼容性门禁：

- 不要求 `${NEUWARE_HOME}/include/bang.h` 必然存在；新版工具链可能由 `cncc` 从 Clang resource include 定位 `bang.h`。
- 使用 C++ 标准库、数学库或线程库的单文件测试必须带 EnvConfig 已验证的链接项；当前审计表明常见组合需要 `-lstdc++ -lm -lpthread`，但最终以真实链接命令为准。
- 不重定义已由当前 `cnrt.h` 提供的 `CNRT_CHECK`。旧代码中的 `CNRT_RET_SUCCESS` 仅在头文件确认存在时可用；当前已审计栈使用 `cnrtSuccess`/头文件宏，修复前先以安装头文件为证据。
- 架构 flag 只来自 EnvConfig、显式环境配置或 `cncc --help`/官方构建脚本，不凭设备营销名推导。

Worker 命令必须在当前 `JOB_ID` 下前台同步执行：

```bash
python .claude/skills/mlu-bangc-main/subagents/scripts/submit_task_to_worker.py \
  --workdir <absolute_workdir> \
  --command "<compile-or-run-command>" \
  --timeout-sec 1800 \
  --task-type accuracy
```

Worker wrapper 退出码 `2` 代表基础设施错误；本地命令按真实 stderr/工具输出分类。

## 步骤 2：完整性扫描

至少确认：

1. 存在 `__mlu_global__` 或当前工具链支持的 Kernel 入口限定符。
2. Host 侧存在与其匹配的 BANG C launch，包含任务规模、function type 与 Queue。
3. 存在设备分配、H2D/D2H、Queue 同步和资源释放路径。
4. 存在独立 reference，比较全部逻辑输出，失败时进程返回非零。
5. 没有可执行的其他后端替代路径。

完整性失败即进入静态检查，即使原 binary 返回 0。

## 步骤 3：编译并运行原文件

在临时构建目录中使用 EnvConfig 命令编译，保存完整命令、退出码、stdout/stderr。编译成功后运行 binary，保存：

| 门禁 | 通过条件 |
| --- | --- |
| 编译/链接 | `cncc` 返回 0 且生成预期 binary |
| CNRT/launch | 所有 CNRT 调用、Kernel 启动与 Queue 完成均成功 |
| 精度 | 所有固定用例在原容差内通过，整数/布尔精确比较 |
| 目标 | 实际运行设备确认为 MLU590 |

退出码 0 不自动等于精度通过；测试必须打印机器可解析结果或以非零退出编码失败。

## 步骤 4：快速通过

若完整性、编译、运行、精度与目标门禁全部通过：

1. 将输入逐字复制为 `bangc_code_fix.mlu`。
2. 生成 `bangc_report.md`，记录 `passed=true`、`blocked=false`、命令与证据。
3. 停止，不重构通过代码。

## 步骤 5：静态检查

代码门禁失败时按 `StaticReviewer.md` 分发静态检查。StaticReviewer 必须生成 `bangc_code_fix.mlu` 并追加 `bangc_report.md`，只修复源码中可证明的问题。

```python
agent = spawn_agent(
    agent_type="default",
    message=f"""
读取 .claude/skills/mlu-bangc-code-review/StaticReviewer.md。
审查 {input_code_path}，输出 {fixed_code_path} 并追加 {report_path}。
保留 reference、测试、容差、Host API 与自定义 BANG C Kernel。
""",
)
```

## 步骤 6：动态修复

按 `DynamicFixer.md` 分发 `bangc_code_fix.mlu`。每轮执行 `cncc → binary → accuracy`；直接更新同一文件并追加 `bangc_report.md`。达到以下任一条件停止：

- 全门禁通过
- 环境/基础设施阻塞
- 相同规范化失败连续出现两次
- 达到五轮

主流程不能因文件存在而宣称成功，必须读取最终状态字段。

## 报告所有权

按顺序追加：

1. 主流程：输入、EnvConfig、原始编译/运行/精度。
2. StaticReviewer：静态契约、发现和精确改动。
3. DynamicFixer：每轮命令、失败签名、修改与最终结果。

报告末尾必须包含：

```text
passed: true|false
blocked: true|false
target_verified: true|false
final_code_path: <absolute path>
```

只有 `passed=true`、`blocked=false`、`target_verified=true` 才能称为“已在 MLU590 验证”。
