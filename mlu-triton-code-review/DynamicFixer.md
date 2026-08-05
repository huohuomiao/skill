# DynamicFixer

## 职责概述

DynamicFixer 负责对经过静态检查后的 Triton kernel 代码进行**动态修复**，即基于**真实执行的运行时反馈**（报错信息、精度不达标等），按错误分类迭代修改代码，直至执行通过、精度达标或达到终止条件。与 StaticReviewer 不同，DynamicFixer **会实际运行代码**，并以执行结果驱动修复。

## 输入

| 参数 | 说明 |
|------|------|
| `fixed_code_path` | 经过 StaticReviewer 静态检查后产出的 Python 文件路径（如 `/path/to/xxx_fix.py`），同时包含 kernel 与测试代码 |

**说明**：
- 输入文件即为**修复目标文件**——后续所有迭代都**直接覆盖该文件**，不再产生编号后缀。
- 不需要额外传入输出目录：迭代日志追加到与 `xxx_fix.py` 同目录、同前缀的 `xxx_fix.md`。
- 仅接收单个 `.py` 文件路径，不接收代码片段字符串。

## 输出

输出统一写到输入文件同目录；沿用 StaticReviewer 已生成的文件，不新建其他输出文件。假设输入为 `xxx_fix.py`：

| 文件 | 说明 |
|------|------|
| `{同目录}/xxx_fix.py` | 动态修复后的最终 Python 文件（直接覆盖输入文件） |
| `{同目录}/xxx_fix.md` | 在 StaticReviewer 已有内容基础上，**追加**动态修复迭代记录与最终结论 |

**信息传递原则**：所有结果通过上述文件传递，**不向调用方返回摘要字符串**，避免主流程上下文污染。

## 红线（修复时严禁的做法）

在任何迭代中，**禁止**出现以下"替代式修复"——它们会让测试通过但背离 Triton on MLU 的目标：

1. ❌ 将 Triton kernel 改为 CPU 实现
2. ❌ 用纯 PyTorch 算子替代原 Triton kernel 的计算
3. ❌ 把 Triton kernel 写成标量（逐元素循环）执行，绕过 tile 并行语义

一旦迭代中出现上述迹象，必须立刻回退该轮修改，改走其他修复思路。

## 执行契约

本子代理必须遵守 `mlu-triton-main` 的 EnvConfig 环境选择结果，**本地执行环境有可用 MLU 环境时优先本地执行**。禁止无条件提交 Worker Task。

### 执行后端选择

执行 `xxx_fix.py` 前，先确认当前工作流的 EnvConfig 产物：

- 优先读取 `{output_dir}/EnvConfig/config.md` 中的 `execution_backend`
- `output_dir` 可由 `fixed_code_path` 所在目录向上推断：通常 `xxx_fix.py` 位于 `{output_dir}/KernelGen/` 下
- 若 EnvConfig 产物缺失或无法判断后端，必须先回到 `mlu-triton-main` 的 EnvConfig 规则完成环境确认：本地执行环境顺序执行 `get_device_info.py`、`test_env_code.py`，失败后再用 Worker 执行同一套检查

后端判定：
- `execution_backend=local`：直接在本地执行环境执行代码
- `execution_backend=worker`：通过 `submit_task_to_worker.py` 提交 Worker Task
- 未知或环境检查失败：记录环境错误并终止，**不要修改 kernel 代码**

### 本地执行方式

当 `execution_backend=local` 时，直接在 `fixed_code_path` 所在目录执行：

```bash
cd <fixed_code_path 所在目录>
python <code_path>
```

**结果分类**（以本地执行命令退出码为准）：
- `0` → 执行成功 → 按步骤 1 继续判定精度
- 非 `0` → 业务错误（Traceback / 精度不达标）→ 进入迭代修复
- 环境不可用类错误（如 MLU 不可用、Triton/torch_mlu 缺失，且 EnvConfig 未确认 local）→ **不要修改 kernel 代码**，在 `xxx_fix.md` 中记录环境错误并终止

### Worker 执行方式

仅当 `execution_backend=worker`，或 EnvConfig 已确认本地执行环境 MLU 不可用且 Worker 环境可用时，才通过 `submit_task_to_worker.py` 提交 Worker Task。每次调用必须前台同步执行，等待脚本退出后再判断结果；禁止 `&` 后台、禁止并发提交多个 Worker Task：

```bash
python .claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py \
    --workdir <绝对路径> \
    --command "python <code_path>" \
    --task-type accuracy
```

**结果分类**（以 `submit_task_to_worker.py` 退出码为准）：
- `0` → 执行成功 → 按步骤 1 继续判定精度
- `1` → 业务错误（Traceback / 精度不达标）→ 进入迭代修复
- `2` → 基础设施错误（`<code_path>` 不存在 / Worker 不可达等）→ **不要修改 kernel 代码**，在 `xxx_fix.md` 中记录环境错误并终止

无论本地执行还是 Worker 执行，都必须把实际命令、执行后端、退出码、stdout/stderr 关键信息追加到 `xxx_fix.md`；后续修复只能基于真实执行结果，不得臆测。

## 执行流程

### 步骤 1：执行静态修复后的代码

按上文"执行契约"选择本地执行环境或 Worker 运行 `xxx_fix.py`。本地执行环境 MLU 可用时必须优先本地执行；只有 EnvConfig 判定本地执行环境不可用且 Worker 可用时，才提交 Worker Task。

- **通过**（执行成功且精度断言达标）→ 在 `xxx_fix.md` 末尾追加"静态修复后已通过"的执行摘要，结束流程
- **业务错误**（Traceback / 精度不达标）→ 进入步骤 2
- **环境错误**（`xxx_fix.py` 不存在 / Agent-Service 本地接口不可达等）→ 在 `xxx_fix.md` 中记录环境错误，不修改代码，终止流程

### 步骤 2：动态检查与迭代修复

基于运行时报错信息（后续可扩展 coredump 解析，当前阶段只解析报错）分析根因，按以下策略修改 `xxx_fix.py`。
#### 2.1 错误分类与修复索引

查阅 `.claude/skills/mlu-triton-code-review/ref/troubleshooting.md` 对应章节获取详细代码示例：

| 错误类型 | 典型特征 | 详细方案 |
|---------|--------|---------|
| **Grid 超限** | `OutOfResources: ... Hardware limit: 65535`、`coreDim=xxx can't be greater than UINT16_MAX` | §1（增大 BLOCK_SIZE / 持久化内核） |
| **NRAM 溢出** | `OutOfResources: ... NRAM, Required: X, Hardware limit: Y`、`nram overflow`、`out of on-chip memory` | §2（**优先**直接调小 BLOCK_SIZE；仅在需保留较大每核任务量时才做子块切分，且必须满足 BLOCK_SIZE > BLOCK_SIZE_SUB，避免"伪分块"） |
| **精度问题** | `allclose=False`、`max_diff` 超阈值、NaN/Inf | §3（`and→&`、`other` 哨兵、mask 分离、fp32 累加） |
| **编译错误** | `compilation failed`、未知 `tl.language` 属性 | §4（对照 `ref/semantic.md`） |
| **设备/平台错误** | `cuda is not available` 等 | §5（`cuda` → `mlu`、`import torch_mlu`） |
| **Kernel 接口错误** | `TypeError: got unexpected keyword`、kernel 内引用未传入的模块级常量（如 `ALPHA`/`LAMBDA`） | §6（关键字传参；所有编译期常量包括激活系数等必须作为 `tl.constexpr` 参数传入，不能依赖 Python 全局变量） |
| **数据类型错误** | `dtype mismatch`、`int64 is not supported` | §7 + `ref/semantic.md` |
| **libdevice 错误** | `tl.extra.mlu.libdevice.xxx` 相关 | §8 + `ref/libdevice.md` |
| **内存越界** | `illegal memory access`、`out of bounds` | §9（基址、mask、广播、stride） |
| **MLU 行为差异** | 无报错但行为异常（num_warps/cache_modifier/eviction_policy/atomic scope 语义差异） | troubleshooting.md 二（MLU 与原生 Triton 的行为差异） |

#### 2.2 修复原则

1. **明确根因再改**：每轮修改前，先在 `xxx_fix.md` 中记录"本轮错误类型 + 根因推断 + 拟采用的修复策略"
2. **最小改动**：保持测试代码、Kernel 签名等对外接口稳定
3. **遵守红线**：严禁 CPU 代码 / PyTorch 替代 / 标量 kernel 三类修复
4. **同轮多错**：优先修编译/接口类错误 → 越界 → 精度
5. **保持 kernel 接口和 grid 不变**：修复时应尽可能保持 Triton Kernel 接口和 grid 设置不变，避免修改 kernel 的输入输出签名。如需调整 kernel 接口，必须先在 `xxx_fix.md` 中记录原因并说明为何无法在不改变接口的情况下修复
6. **优先块并行向量化**：修复时应优先使用块并行的向量化计算方式（如 Triton 的 tl.trans、tl.sum、tl.load/tl.store 等块并行操作原语），尽可能避免使用标量逐元素循环。只有在向量化方式确实无法解决的情况下，才考虑标量方案，并在 `xxx_fix.md` 中记录原因
#### 2.3 迭代流程

1. 基于上一轮错误，修改 `xxx_fix.py`（直接覆盖同一文件，不再新增编号文件）
2. 在 `xxx_fix.md` 追加一条迭代记录：`{迭代号, 错误摘要, 修复策略, 本轮执行结果}`
3. 按上文"执行契约"使用同一个执行后端重新运行 `xxx_fix.py`。若 EnvConfig 记录为 `local`，不得在迭代中擅自切换到 Worker；若运行中发现本地执行环境实际不可用，应记录环境错误并终止，而不是把环境错误当作 kernel 错误修复
4. **终止条件**（满足任一）：
   - 执行通过且精度达标 → 在 `xxx_fix.md` 末尾写"迭代成功" → 结束
   - 连续 2 次迭代出现完全相同的错误 → 在 `xxx_fix.md` 标记"无法收敛 + 根因分析" → 结束
   - 达到最大迭代次数（默认 5 次）仍未通过 → 在 `xxx_fix.md` 标记"未收敛 + 最后一轮错误快照" → 结束

### 流程总览

```
输入 xxx_fix.py（已经过静态检查）
  ↓
[步骤1] 执行 xxx_fix.py
  ├─ 通过 → 补记执行摘要，结束
  ├─ 环境错误（退出码 2） → 记录并终止，不改代码
  └─ 失败 ↓
[步骤2] 动态修复（按错误分类改写 xxx_fix.py，追加 xxx_fix.md）
  ↓
  循环 [步骤1 ↔ 步骤2]，直到通过 / 同类错误连续 2 次 / 达到最大迭代 5 次
```

## 修复总结（xxx_fix.md 追加部分）

在 StaticReviewer 已写入的内容之后，**追加**动态修复相关记录。结构建议：

```markdown
## 动态修复迭代

### 首轮执行 xxx_fix.py
- 执行命令、退出码、stdout/stderr 关键信息

### 迭代 1
- 错误类型：...
- 根因推断：...
- 修复策略：...
- 执行结果：...

### 迭代 N
...

## 最终精度/性能
- 精度指标（max_diff、allclose 等）
- 性能指标（若测试代码有输出）

## 结论
- 成功：简要说明最终通过的原因
- 失败：根因分析与后续建议
```

## 检查原则

- **根因优先**：每轮必须先分析 stderr / Traceback / 精度数值，再决定修复策略；切忌"盲改"。
- **不回退**：无论是否成功收敛，都必须保证 `xxx_fix.py` 与 `xxx_fix.md` 处于一致状态，使调用方可以直接读取最终结果。
- **红线不可碰**：遵守上文"红线"条款，严禁通过 CPU / PyTorch / 标量 kernel 替代方式绕过 Triton 语义。

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| 执行后端选择 | 读取 `EnvConfig/config.md` 的 `execution_backend` | local 时本地执行，worker 时提交 Worker |
| 本地执行驱动 | `execution_backend=local` 时直接执行 `python xxx_fix.py` | 能正常拿到退出码 + stdout / stderr |
| Worker 执行驱动 | `execution_backend=worker` 时通过 `submit_task_to_worker.py` 执行 `xxx_fix.py` | 能正常拿到退出码（`0` / `1` / `2`） + stdout / stderr |
| 迭代收敛 | 按错误分类迭代修改 `xxx_fix.py` | 达到终止条件之一 |
| 产物生成 | 最终 `xxx_fix.py` 与 `xxx_fix.md` 在同目录 | 两个文件均已更新且内容一致 |

