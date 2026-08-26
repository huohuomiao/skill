# MLU Triton 平台规则

仅在目标平台为寒武纪 MLU 时读取本文件。通用 Triton 需求分析、轴分析、代码生成、审查和优化流程继续由四个主 Skill 负责。

## 目录

- 运行时与设备适配
- 设备属性与核心数
- Grid 与持久化 Kernel
- NRAM 与配置调优
- MLU 后端优化规则
- 迁移和验证清单

## 运行时与设备适配

- 将 CUDA 设备字面量替换为 MLU：`cuda` → `mlu`、`is_cuda` → `is_mlu`、`torch.device("cuda")` → `torch.device("mlu")`。
- 在测试与基准代码中使用 `torch.mlu.synchronize()` 完成计时同步。
- 不要把 CPU fallback、PyTorch reference 计算或跳过 Triton Kernel 当作修复结果。
- 动态执行前先读取 `{output_dir}/EnvConfig/config.md`：`local` 直接执行；`worker` 使用 `.claude/skills/mlu-triton-main/subagents/scripts/submit_task_to_worker.py` 前台同步提交。
- 环境探测只调用 `.claude/skills/share/mlu/runtime/get_device_info.py` 和 `.claude/skills/share/mlu/runtime/test_env_code.py`。

## 设备属性与核心数

需要 NRAM 或物理核心信息时，使用 MLU 后端驱动读取真实设备属性，不在通用 Skill 中硬编码：

```python
import torch
from triton.backends.mlu import driver

props = driver.BangUtils().get_device_properties(torch.mlu.current_device())
total_core_num = props["cluster_num"] * props["core_num_per_cluster"]
max_nram_size = props["max_nram_size"]
```

若旧代码使用 `torch.mlu.get_device_properties(...).multi_processor_count`，允许保留；新生成代码优先采用同一工程已验证可用的接口，避免在一个文件中混用两套设备属性 API。

## Grid 与持久化 Kernel

- 检查各维 Grid 是否超过 MLU 后端限制；旧版工具链常见单维上限为 `65535`。
- 普通一维 Grid 可写为 `min(logical_grid, total_core_num // num_warps)`。
- Grid 被压缩后，Kernel 必须用 `tl.num_programs(axis=0)` 和固定步长循环覆盖全部逻辑任务，禁止遗漏尾块。
- 多维 Grid 可展平为一维；Kernel 内根据线性任务编号恢复各维索引。
- 如果“增大 BLOCK_SIZE 降低 Grid”和“减小 BLOCK_SIZE 解决 NRAM”互相冲突，切换为持久化 Kernel，不要反复震荡调参。

持久化 Kernel 的通用形态：

```python
@triton.jit
def persistent_kernel(x_ptr, y_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    for block_start in range(pid * BLOCK_SIZE, size, programs * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < size
        value = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, value, mask=mask)
```

## NRAM 与配置调优

- 运行时读取 `max_nram_size`；仅在无法读取设备属性时把 512 KiB 当作保守参考值，不当作所有设备的固定上限。
- 遇到 `OutOfResources: ... NRAM` 时，优先减小非归约轴 BLOCK_SIZE，其次降低 `num_stages`，再评估 `num_warps`。
- NRAM 利用率偏低且性能受访存或并行度限制时，可逐步增大高优先级轴的 BLOCK_SIZE。
- Block size 优先选择 2 的幂或 32 的倍数，并以真实精度和性能结果决定是否保留。
- `num_warps`、`num_stages` 候选必须来自 MLU 实测；不要直接照搬 CUDA 默认值。
- 自动调优生成器需要同时约束 Grid、NRAM、核心数和测试数据规格。

## MLU 后端优化规则

- MLU Triton 对计数型 `for` 循环更容易启用流水；语义可证明等价时，可将固定步长 `while` 改写为 `for`。
- 三维 Tile 沿中间维归约可能效率较低；仅在形状和索引等价可证明时，尝试把非归约首维分块降为 1，将三维归约改成二维。
- Wrapper 中额外的 `transpose/permute + contiguous` 可能掩盖 Kernel 收益；能够在 Kernel 索引中等价吸收时再消除。
- 使用 MLU Libdevice 前读取 `.claude/skills/share/mlu/references/libdevice.md`；性能替换策略读取 `.claude/skills/share/mlu/optimize/libdevice-opt.md`。
- MLU 原语及 dtype 支持情况统一读取 `.claude/skills/share/mlu/references/primitives.md`。

## 迁移和验证清单

1. 检查设备字面量、同步 API 和平台残留。
2. 检查使用的 Triton 原语与 dtype 是否在共享原语清单中。
3. 检查 Grid 上限；使用持久化 Kernel 时验证所有逻辑块均被覆盖。
4. 真实编译运行，确认不存在 NRAM、Grid 或后端编译错误。
5. 使用原始 PyTorch reference 验证精度，不放宽用户提供的误差阈值。
6. 只有在相同输入、相同计时方法和相同执行后端下才比较性能。
7. 修改失败时保留真实 stdout/stderr，不伪造成功结果。
