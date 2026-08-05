# EnvConfig

## 职责概述

EnvConfig 负责 Triton 算子开发前的运行环境确认。Triton 代码的真实运行、
编译、精度测试和性能测试必须在实际可用的 MLU 环境中完成；如果当前环境
已经具备 MLU/Triton-MLU 工具链，则直接在本地执行命令，否则通过
`.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py`
提交 Worker Task 作为兜底。

本文中的“本地执行”指直接在当前环境运行命令；“Worker”指通过
`submit_task_to_worker.py` 提交任务的远端执行环境。

EnvConfig 产出人类可读的环境记录和设备信息原文；下游需要运行代码时，继续
遵守主流程 `.claude/skills/mlu-triton-main/SKILL.md` 中的运行环境选择规则。

## 运行环境选择规则

所有动态执行必须遵守主流程 `.claude/skills/mlu-triton-main/SKILL.md` 中的运行环境选择规则：

- EnvConfig 必须先直接在本地顺序执行 `get_device_info.py` 和 `test_env_code.py`。
- 只有两个脚本在本地都 exit code = 0，才判定 `execution_backend=local`，下游动态执行直接在本地运行命令。
- 只要任意一个脚本在本地失败，就必须在当前 `JOB_ID` 下通过 Worker Task 顺序执行同一套 `get_device_info.py` 和 `test_env_code.py`，禁止新建 Job。
- 只有两个脚本在 Worker 上都成功，才判定 `execution_backend=worker`；如果 Worker 检查也失败，必须停止流程并报告真实日志。
- 每次 Worker Task 调用必须前台同步执行，等待 `submit_task_to_worker.py` 退出后再进行下一步；禁止 `&` 后台、禁止并发提交多个 Worker Task。
- Worker Task 必须以脚本退出码和 Worker 写入的 `stdout.log`、`stderr.log`、`result.json` 作为真实结果。
- 禁止修改 Worker 全局环境或共享依赖。

## 输入

| 来源 | 内容 |
|------|------|
| 用户输入 | 输出存储路径（默认为 `output_dir`） |
| 辅助脚本 | `.claude/skills/mlu-triton-main/subagents/scripts/test_env_code.py`、`get_device_info.py`、`submit_task_to_worker.py` |
| 环境变量 | `JOB_ID` —— 当前 Job 的真实 ID，仅在需要 Worker Task 时由 `submit_task_to_worker.py` 读取 |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/EnvConfig/config.md` - 人类可读的运行环境记录 |
| 文件输出 | `{输出存储路径}/EnvConfig/runtime_info.txt` - 本地或 Worker 信息采集 stdout 原文 |
| 摘要返回 | 环境配置摘要（见[步骤 4](#步骤-4返回环境配置摘要)） |
## 执行步骤

### 步骤 1：本地探测

直接在本地执行：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/get_device_info.py
python .claude/skills/mlu-triton-main/subagents/scripts/test_env_code.py
```

**检查项**：
- `cnmon` 与 MLU 设备信息（由 `get_device_info.py` 采集）
- Triton 版本
- PyTorch 版本
- MLU 设备可用性
- Vector Add 功能测试
- 精度验证（最大误差 < 1e-5）

**结果判断**：
- ✅ **成功**（`get_device_info.py` 和 `test_env_code.py` exit code 都为 0）：本地 MLU 就绪，进入步骤 2a
- ❌ **失败**（任一脚本 exit code 非 0）：进入步骤 1b，使用 Worker Task 兜底

### 步骤 1b：Worker 自检（本地不可用时）

在 Worker 侧顺序执行 `.claude/skills/mlu-triton-main/subagents/scripts/get_device_info.py`
和 `.claude/skills/mlu-triton-main/subagents/scripts/test_env_code.py`，验证远端 MLU 设备和 Triton 环境：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --task-type custom \
    --workdir <仓库根目录的绝对路径> \
    --timeout-sec 600 \
    --command "python .claude/skills/mlu-triton-main/subagents/scripts/get_device_info.py && python .claude/skills/mlu-triton-main/subagents/scripts/test_env_code.py"
```

前台同步执行该命令，等待脚本退出后读取退出码再判断。
**结果判断**：
- ✅ **成功**（exit code = 0）：Worker 环境就绪，进入步骤 3
- ❌ **失败**（exit code = 1 / 2）：按[回退机制](#回退机制)退出，回读并打印 Worker 自检任务真实日志

### 步骤 2a：本地环境信息整理

将步骤 1 中两个本地脚本的 stdout 合并为环境信息原文，进入步骤 3。

### 步骤 3：生成环境记录

#### 3.1 生成 `{输出存储路径}/EnvConfig/config.md`

```markdown
# 环境配置

## 执行位置

- execution_backend: {local|worker}
- worker_submit_url: http://127.0.0.1:8086/run/v1/agent/submit-task（仅 Worker 模式）
- env_check_task_id: {Worker task_id 或 local}

## 环境检查结果

✓ {本地或 Worker} 环境自检通过
✓ {本地或 Worker} 环境信息采集完成

### 执行环境信息

> 以下内容是信息采集 stdout 原文，等价于在实际执行环境中
> 顺序执行 `get_device_info.py` 和 `test_env_code.py`。

\```
{环境信息采集 stdout 原文}
\```

## 时间戳

{timestamp}
```

#### 3.2 保存环境信息原文

把步骤 2 的 stdout 原文写入：
```text
{输出存储路径}/EnvConfig/runtime_info.txt
```

### 步骤 4：返回环境配置摘要

将结果以摘要形式返回给调用方：

```json
{
  "status": "ready",
  "execution_backend": "local 或 worker",
  "worker_submit_url": "http://127.0.0.1:8086/run/v1/agent/submit-task（仅 Worker 模式）",
  "env_check_task_id": "<Worker 模式下的环境检查任务 id；local 模式为 local>",
  "runtime_info_path": "{输出存储路径}/EnvConfig/runtime_info.txt",
  "config_path": "{输出存储路径}/EnvConfig/config.md"
}
```

字段说明：
- `status`: `ready` 表示实际执行环境自检与信息采集都通过
- `execution_backend`: `local` 表示本地 MLU 可用，可直接在本地执行命令；`worker` 表示通过 Worker Task 执行
- `env_check_task_id`: Worker 模式用于追溯 Worker 侧真实执行记录；local 模式为 `local`
- `runtime_info_path`: 指向环境信息采集 stdout 的本地副本

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| 本地设备信息 | 直接在本地执行 `python .../get_device_info.py` | 退出码 = 0 |
| 本地环境自检 | 直接在本地执行 `python .../test_env_code.py` | 退出码 = 0 |
| Worker 环境检查（本地不可用时） | 执行 `submit_task_to_worker.py --task-type custom ... --command "python .../get_device_info.py && python .../test_env_code.py"` | 脚本退出码 = 0 |
| 环境信息落盘 | 检查 `runtime_info.txt` 存在且非空 | 文件内容来自实际执行环境 stdout |

## 回退机制

| 场景 | 处理方式 |
|------|--------|
| 本地 `get_device_info.py` 失败 | 切换到 Worker Task 兜底 |
| 本地 `test_env_code.py` 失败 | 切换到 Worker Task 兜底 |
| Agent-Service 本地接口不可达（Worker 模式） | 退出执行，提示用户确保 Agent-Service 在 `http://127.0.0.1:8086` 可用 |
| Worker 环境检查失败 | 退出执行，回读并打印 Worker 任务 `stdout.log` / `stderr.log` 的真实失败原因 |

## 下游使用说明

后续步骤（`mlu-triton-code-review` 动态测试、`mlu-triton-optimize` benchmark 等）在需要真实运行 Triton 代码时，继续遵守主 Skill 的运行环境选择规则：本地 MLU 可用则直接在本地执行命令；本地不可用则通过 `.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 执行：
```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <绝对路径> \
    --command "<要跑的命令，如 python /abs/path/xxx.py>" \
    --task-type {compile|accuracy|performance|custom} \
    --timeout-sec <运行阶段超时秒数>
```

每次 Worker Task 必须前台同步执行，等待 `submit_task_to_worker.py` 返回退出码后再进入下一步；禁止 `&` 后台、禁止并发提交多个 Worker Task。每次提交必须显式传入 `--timeout-sec`，它表示 Worker lease 后才开始计时的运行阶段超时；Scheduler 会额外用当前 Job 剩余时间限制排队 + 运行的总截止时间。判断成功/失败时，以实际执行环境的退出码和 stdout/stderr/result 为准；Worker 模式下以脚本退出码（`0` = succeeded / `1` = failed / `2` = 基础设施错误）和 Worker 产物 `stdout.log`、`stderr.log`、`result.json` 为准。

## 相关脚本

| 脚本 | 位置 | 功能 |
|------|------|------|
| `test_env_code.py` | `.claude/skills/mlu-triton-main/subagents/scripts/` | Triton 环境自检脚本，可直接在本地执行；Worker 模式下作为 `submit_task_to_worker.py --command` 的参数执行 |
| `get_device_info.py` | `.claude/skills/mlu-triton-main/subagents/scripts/` | MLU 设备信息采集脚本，可直接在本地执行；Worker 模式下作为 `submit_task_to_worker.py --command` 的参数执行 |
| `submit_task_to_worker.py` | `.claude/skills/mlu-triton-main/subagents/scripts/` | Worker 提交 + 轮询 + 回读真实结果；仅在本地没有可用 MLU/工具链时使用 |