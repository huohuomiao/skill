# TritonMLUMigration

## 职责概述

TritonMLUMigration 负责将社区中针对 GPU 实现的 Triton Kernel 迁移至寒武纪 MLU 平台，确保迁移后的 Kernel 在功能完整性方面满足要求。

## 输入

| 来源 | 内容 |
|------|------|
| Step 6 输出 | `{输出存储路径}/KernelGen/step6_test_code.py` |
| 用户输入 | 输出存储路径（默认为 `output_dir`） |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step7_migrated.py` - 迁移后的 Triton kernel 代码（适配 MLU 平台） |

## 迁移检查清单

- [ ] **Grid 超限检查**：计算 grid = triton.cdiv(size, BLOCK_SIZE)，确保 grid <= 65535
- [ ] **NRAM 超限检查**：尝试编译运行，确保无 NRAM 超限错误，如有则调小 BLOCK_SIZE
- [ ] **CUDA 到 MLU 适配**：替换 cuda→mlu、CUDA→MLU、is_cuda→is_mlu、torch.device('cuda')→torch.device('mlu')

## 迁移检查项指南

### Grid 超限检查

**问题说明**：受硬件架构固有特性限制，寒武纪平台对 Kernel 启动阶段的 grid 参数设置了明确的取值上限：最大值为 65535（即 2^16 - 1）；而在 GPU 平台中，该参数的最大值为 2^32 - 1。

**超限时解决方案**：

| 方案 | 说明 |
|------|------|
| 方案 1：调大 BLOCK_SIZE | 将 BLOCK_SIZE 调整为更大值（如 64），使 grid 满足限制 |
| 方案 2：持久化内核 | 将 grid 限制为物理核心数，任务拆分到 Kernel 内部循环处理 |

> **⚠️ 循环问题处理**：当尝试方案 1（调大 BLOCK_SIZE）后出现 NRAM 超限错误，需要调小 BLOCK_SIZE 时，可能陷入「grid 超限 → 调大 BLOCK_SIZE → NRAM 超限 → 调小 BLOCK_SIZE → grid 超限」的循环。此时**应切换至方案 2（持久化内核）**，而非继续在方案 1 中反复调整 BLOCK_SIZE。


#### 方案 1：剔除 BLOCK_SIZE 过小的调优配置（tuning config）

鉴于寒武纪 MLU 芯片拥有更大的片上存储空间，从性能优化的角度考量，采用相较于 GPU 更大的 BLOCK_SIZE 配置，更有助于充分释放硬件算力。例如，可将 BLOCK_SIZE 调整为 64，使 gridx 参数满足硬件限制要求。

```python
import torch
import torch_mlu
import triton
import triton.language as tl

@triton.jit
def persistent(inp_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < size
    x = tl.load(inp_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)
    BLOCK_SIZE = 64
size = 2**16 * 32
inp = torch.rand(size, device='mlu')
output = torch.empty_like(inp)
persistent[(triton.cdiv(size, BLOCK_SIZE), )](inp, output, size, BLOCK_SIZE)
```

#### 方案 2：采用持久化内核（persistent kernel）的实现方案

将 gridx 参数配置为物理核心的实际数量，同时将原由 gridx 维度承载的任务，拆分至 Kernel 内部循环中执行分批处理。

在寒武纪架构下，从性能优化角度出发，推荐采用持久化内核的实现方案。该方案有以下核心优势：
- 降低 Host 侧启动任务（launch task）的开销。
- 支持在 Kernel 内部拆分循环结构，为启用软流水优化（Soft Pipeline Optimization）提供便利条件，有助于进一步提升计算效率。

针对上述示例，可通过如下方式使能持久化内核（Persistent Kernel）优化：

```python
import torch
import torch_mlu
import triton
import triton.language as tl

@triton.jit
def use_gridy(inp_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE

    for block_start_offset in range(block_start, size, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        x = tl.load(inp_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

core_num = torch.mlu.get_device_properties().multi_processor_count
BLOCK_SIZE = 32
size = 2**16 * 32
inp = torch.rand(size, device='mlu')
output = torch.empty_like(inp)
grid = min(triton.cdiv(size, BLOCK_SIZE), core_num)
use_gridy[(grid, )](inp, output, size, BLOCK_SIZE)
```


### NRAM 超限检查

**问题说明**：由于 MLU 与 GPU 平台的后端编译机制存在差异，部分基于 Triton 框架开发的 GPU Kernel 在部署至 MLU 平台时，可能会出现 NRAM（本地内存）超限问题, 执行输入代码会触发如下错误： `OutOfResources: out of resource: grid size, Required: 65536, Hardware limit: 65535. Reducing block sizes or ``num_stages`` may help.`。\
**超限时解决方案**：**调小BLOCK_SIZE（块大小）** 参数即可解决。


### CUDA 到 MLU 适配

将代码中的 CUDA 相关内容替换为 MLU：

| 替换项 | 替换后 |
|--------|--------|
| `cuda` | `mlu` |
| `CUDA` | `MLU` |
| `is_cuda` | `is_mlu` |
| `torch.device('cuda')` | `torch.device('mlu')` |


按照上述指南完成代码修改后，需执行以下流程：

1. **保存代码**：存储检查后代码到 `{输出存储路径}/KernelGen/step7_migrated.py`
2. **执行测试脚本**：先读取 `{输出存储路径}/EnvConfig/config.md` 的 `execution_backend`；`local` 时直接运行 `python {输出存储路径}/KernelGen/step7_migrated.py`，`worker` 时通过 `.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 提交 Worker Task（必须前台同步执行，等待脚本退出后再读取结果；禁止 `&` 后台、禁止并发提交多个 Worker Task）；EnvConfig 缺失或未知时先补环境确认，不要修改 kernel 代码
3. **检查运行结果**：
   - 若无报错 → 验证通过
   - 若报 `NRAM` 超限错误 → 记录报错信息中的需求值，调小 BLOCK_SIZE 后返回步骤 1 重新执行
4. **循环迭代**：重复上述迁移检查项，直至无 NRAM 超限错误

> ⚠️ 注意：若在调整 BLOCK_SIZE 过程中出现「grid 超限 ↔ NRAM 超限」的循环，应切换至 Grid 超限的**方案 2（持久化内核）**，而非继续反复调整 BLOCK_SIZE。


## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 6 输出存在 | 检查文件是否存在 | step6_test_code.py 存在且可读 |
| Grid 参数检查 | 计算 grid 值 | grid <= 65535 或已采用持久化内核方案 |
| NRAM 使用检查 | 按主 Skill 运行环境选择规则尝试编译运行 | 无 NRAM 超限错误 |
| 代码语法检查 | Python 编译检查 | 无语法错误 |
| MLU 设备检查 | 检查 device 设置 | 所有 tensor 使用 'mlu' 设备 |
| 功能验证 | 执行精度测试 | 测试通过 |

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 6 输出不存在或无效 | 返回错误 |
| Grid 参数无法修复 | 优先尝试持久化内核方案，其次调大 BLOCK_SIZE |
| NRAM 超限 | 减小 BLOCK_SIZE，多次尝试找到合适值 |
| 迁移后代码无法运行 | 记录错误信息，返回原代码并标记问题 |
| 缺少必需的 kernel 函数 | 返回错误 |