# NVIDIA CUDA Triton / RTX 3090 平台规则

仅在目标后端为 NVIDIA CUDA 时读取本文件。四个主 Skill 继续负责需求分析、轴分析、代码生成、审查和优化；GPU 差异、设备探测和性能采集集中在 `share/gpu`，避免把 `sm_86` 常量散落到主流程。

## 1. 平台门禁

执行或声称“已在目标机验证”前，依次确认：

1. 官方 Triton wheel 的支持平台是 Linux；NVIDIA GPU 最低 Compute Capability（CC）为 8.0。RTX 3090 是 Ampere、CC 8.6，对应 CUDA target `sm_86`。
2. `nvidia-smi` 能看到所选 GPU，且设备名严格匹配 `NVIDIA GeForce RTX 3090`（允许省略 `NVIDIA` 前缀，不把 3090 Ti 当作 3090）。
3. `torch.cuda.is_available()` 为真，`torch.cuda.get_device_capability(i) == (8, 6)`；Triton active target 的 backend 为 `cuda`。
4. 运行 `share/gpu/runtime/test_env_code.py --device i`，让一个真实 Triton kernel 完成 JIT、launch、同步和数值校验。仅 import 成功不代表环境可用。
5. 驱动只需满足所安装 CUDA runtime/wheel 的兼容要求；不要从 `nvidia-smi` 显示的“CUDA Version”推断本机安装了同版本 Toolkit。记录 PyTorch 编译时 CUDA (`torch.version.cuda`) 与驱动版本。

非目标设备可以用于静态生成，但报告必须写明“未在 RTX 3090 上验证”，不得把 CPU、Triton interpreter 或其他 GPU 的结果冒充 3090 实测。

## 2. RTX 3090 / Compute Capability 8.6 硬件边界

下表用于门禁和解释 profiler 结果；运行时调度仍读取实际设备属性。

| 项目 | RTX 3090 / CC 8.6 边界 | 使用规则 |
| --- | ---: | --- |
| 架构 | Ampere, `sm_86` | 禁用 Hopper/Blackwell 专属路径 |
| 显存 | 24 GB GDDR6X（标称） | 可用显存必须动态读取，不能按 24 GB 分配 |
| warp size | 32 threads | `num_warps * 32` 是一个 Triton program 的 CUDA threads |
| 每 block 最大 threads | 1024 | 因而 `num_warps <= 32` 是硬上界；实际 Triton 候选通常更小 |
| 每 SM 最大 resident blocks | 16 | occupancy 还受 warps、registers、shared memory 共同约束 |
| 每 SM 最大 resident warps / threads | 48 / 1536 | 不能用常见的“64 warps/SM”假设 |
| 每 SM 32-bit registers | 64 K | 编译后的 registers/thread 决定可驻留 blocks |
| 每 block 32-bit registers | 64 K | 每 thread 最多 255 个 register |
| 每 SM shared memory | 100 KiB | 与 L1 使用统一数据缓存容量配置 |
| 每 block shared memory | 99 KiB | 超过 48 KiB 需要动态 shared-memory opt-in；Triton/驱动能否设置仍需实编译 |
| grid x 上限 | `2^31 - 1` | 普通一维 grid 通常远大于 SM 数，合法且应保留 |
| grid y/z 上限 | 65,535 | 超限时展平或分批，不能静默截断 |
| Tensor Core 输入 | TF32、BF16、FP16、INT8、INT4 | FP8 Tensor Core 不受 CC 8.6 支持 |

RTX 3090 官方规格还给出 10,496 CUDA cores；这不是 Triton grid 大小。调度单位是 SM 上的 thread block/program，不能用 CUDA core 数替代 `multi_processor_count`。

### 不属于 3090 的能力

- TMA、WGMMA、thread-block cluster、distributed shared memory 是 Hopper (`sm_90`) 或更新架构能力；不要在 3090 路径使用。
- FP8 Tensor Core 从 Ada (`sm_89`) / Hopper 等后续架构开始；3090 上即使某 Triton 版本暴露 float8 dtype，也不代表 `tl.dot` 有原生 FP8 Tensor Core 路径。
- Triton 的 `tl.range(..., warp_specialize=True)` 当前只支持 Blackwell 的特定 matmul 循环；3090 必须禁用。
- `tl.dot_scaled` 面向更新架构的 block-scaled FP4/FP8 路径，不是 3090 的移植方案。

## 3. 动态设备属性：唯一运行时真值

优先调用共享探测脚本：

```bash
python .claude/skills/share/gpu/runtime/get_device_info.py --device 0
python .claude/skills/share/gpu/runtime/test_env_code.py --device 0
```

Kernel launcher 只从当前进程实际选中的设备读取属性：

```python
import torch
import triton
from triton.runtime import driver

device = torch.cuda.current_device()
props = torch.cuda.get_device_properties(device)
cc = torch.cuda.get_device_capability(device)
assert cc >= (8, 0)
assert triton.runtime.driver.active.get_current_target().backend == "cuda"

sm_count = props.multi_processor_count
total_memory = props.total_memory

# Triton 官方教程使用后端属性计算 persistent occupancy；键名可能随版本变化。
backend_props = driver.active.utils.get_device_properties(device)
num_sms = backend_props["multiprocessor_count"]
registers_per_sm = backend_props["max_num_regs"]
shared_memory_per_sm = backend_props["max_shared_mem"]
warp_size = backend_props["warpSize"]
```

不要默认硬编码 `82` 个 SM、24 GiB 可用显存或固定 shared-memory 数。OEM 型号、显存占用、MIG/vGPU、可见设备映射以及 Triton 版本都可能改变可执行条件。`CUDA_VISIBLE_DEVICES` 会重新编号；输出同时记录 index、UUID 和 PCI bus id。

## 4. 普通 grid 与 persistent grid 必须分开

### 普通 kernel

普通 kernel 由 GPU 调度大量独立 programs。grid 表示逻辑 tile 数，不能压到 SM 数或 occupancy 数：

```python
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
```

尾块用 mask；二维/三维 grid 可直接使用，只需遵守各维 CUDA 上限。若展平，必须在 kernel 中无损恢复索引。把普通 grid 写成 `min(logical_grid, sm_count)` 会漏算，除非 kernel 同时使用 grid-stride loop 覆盖剩余任务。

### persistent kernel

persistent kernel 只启动有限 programs，每个 program 用 `tl.num_programs(0)` 跨步处理多个逻辑 tile：

```python
@triton.jit
def persistent_copy(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    for start in tl.range(pid * BLOCK, n, stride):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(y + offsets, tl.load(x + offsets, mask=mask), mask=mask)
```

只有这种 grid-stride 语义才允许压缩 grid。初始值可以是一 program/SM：

```python
num_programs = min(logical_tiles, props.multi_processor_count)
```

更完整的候选必须在 kernel `warmup` 后读取编译结果的 `kernel.n_regs` 与 `kernel.metadata.shared`，按 registers、shared memory、warps 和 16 resident-block 上限估算 active blocks/SM，再令：

```text
resident_by_regs  = floor(65536 / (n_regs * 32 * num_warps))
resident_by_smem  = floor(shared_per_sm / shared_per_block)  # shared=0 时不限制
resident_by_warps = floor(48 / num_warps)
active_blocks_sm  = min(16, resident_by_regs, resident_by_smem, resident_by_warps)
grid              = min(logical_tiles, sm_count * active_blocks_sm)
```

该式是 `sm_86` 的初筛，不替代 CUDA 的 allocation granularity 或 Nsight Compute 的理论 occupancy；最终以 `ncu` 的 LaunchStats/Occupancy 与端到端实测为准。persistent 可能因静态分工、尾波不均、缓存局部性或单 program 过长而变慢，必须与普通 grid 同输入比较。

## 5. `num_warps`、`num_stages` 与 tile

- `num_warps` 首轮用 `{1, 2, 4, 8}`；矩阵核通常从 `{4, 8}` 开始。不要因为硬件上限是 32 就把 16/32 当默认候选。
- 增大 `num_warps` 会增加 threads/block，并可能增大寄存器总量；既可能提升并行度，也可能降低 resident blocks。以编译资源和 profiler 为准。
- `num_stages` 是软件流水深度，不等于 CUDA occupancy。增大它可能隐藏 global-memory latency，也会增加 shared memory 和寄存器存活期。3090 从 `{1, 2, 3, 4}` 小集合实测，资源超限时先降 stages 或 tile。
- `tl.range(..., num_stages=k)` 会尝试流水化该循环中的多数 loads；launch 参数 `num_stages` 的作用范围不同。不要混为同一个旋钮。
- block/tile 通常取 2 的幂以满足 `tl.arange` 约束，但矩阵的 M/N/K tile 仍需结合 layout、mask、Tensor Core 形状、L2 局部性和 shape 分布调优。
- autotune key 必须覆盖真正改变最佳配置的 shape/stride/dtype；候选会多次执行 kernel，带原子更新或原地写入时配置 `reset_to_zero`/`restore_value` 或使用无副作用基准。

## 6. Registers 与 shared memory

资源处理顺序：

1. 先保存编译后的 `n_regs`、shared bytes、spill 信息和 launch 配置；不要从源代码变量个数推断。
2. launch 失败或 occupancy 受 shared memory 限制：减小 tile，降低 `num_stages`，缩短同时存活的块；不要无条件提高 `num_warps`。
3. registers/thread 偏高或出现 local-memory spill：缩短 live range、拆分不必要的中间块、降低展开/stages，再比较性能。仅追求高 occupancy 也可能伤害 ILP。
4. shared memory 超过 48 KiB/block 时视为高风险候选；即便低于 99 KiB 硬上限，也可能只允许一 block/SM。必须真实 launch 并用 NCU 验证。
5. 对 persistent kernel，用编译资源计算 resident programs；普通 kernel 不要据此裁剪逻辑 grid。

## 7. 精度与 dtype 门禁

- FP32 `tl.dot` 在 NVIDIA Tensor Core 上默认可采用 TF32；需要严格 FP32 语义时显式 `input_precision="ieee"`，允许 TF32 时显式记录 `input_precision="tf32"`，并使用与需求一致的容差。`allow_tf32` 已弃用。
- TF32 只改变 FP32 matrix-multiply 输入精度，不是可存储的 Triton dtype。不要把逐元素 FP32 运算称作 TF32。
- FP16/BF16 Tensor Core 可用；累加优先 FP32，输出再按接口 dtype 转换。BF16 的 CUDA 硬件要求 CC >= 8.0，3090 满足。
- FP64 语法可编译不等于适合 3090：CC 8.6 的非 Tensor FP32:FP64 吞吐比为 64:1。除非接口要求，否则不要把中间量提升为 FP64。
- FP8：3090 无原生 FP8 Tensor Core。默认拒绝 FP8 kernel 移植；若仅做存储/软件转换，必须明确“非原生”，验证 installed Triton 的 dtype/lowering，并与 FP16/BF16 基线比较。
- `tl.exp`、`tl.sqrt` 等部分内建函数是快速近似；严格精度使用 `tl.*_rn`（若 API 提供）或 `triton.language.extra.libdevice` 的已验证函数，并读取 `references/libdevice.md`。

## 8. CUDA 专属 API 门禁

| 能力 | 3090 策略 |
| --- | --- |
| `tl.load/store`, `make_block_ptr`, `advance` | 可用；以实际 Triton 版本编译测试 |
| `tl.dot` FP16/BF16/TF32/INT8 | 可用；shape/dtype/精度需实测 |
| `tl.dot_scaled`, FP8/FP4 Tensor Core | 禁用 |
| TMA tensor descriptor 加速 | 禁用；使用普通 pointer/block pointer 路径 |
| WGMMA / warp specialization / clusters | 禁用 |
| inline PTX | 仅 CUDA 分支、必须约束 `sm_86`，且提供等价测试；优先官方 Triton primitive |
| CUDA libdevice | 使用 `from triton.language.extra import libdevice`；先查共享 API 文档 |

## 9. 验证顺序

1. 对目标算子代码及可执行脚本静态搜索残留：`mlu`、`torch.mlu`、`triton.backends.mlu`、`cnmon`、`cnperf`、`NRAM`、`tl.extra.mlu` 必须为零；兼容 Skill 名和本检查清单中的字面量不计。
2. 运行设备采集脚本，保存 JSON；确认目标 device 的型号、CC、driver、memory、SM count。
3. 运行 vector-add smoke test；再运行原算子的 correctness/reference 测试，覆盖非整 tile、空维/小维、极值、NaN/Inf（若契约包含）。
4. 在相同 GPU、输入、dtype、warmup、repeat、同步方式下比较性能；报告中分开 kernel time 与端到端 time。
5. 使用 `share/gpu/perf-analyzer/analyzer.sh` 生成 `.ncu-rep`、raw CSV 和摘要；关注 occupancy、registers、shared memory、SM/DRAM throughput 与 tail waves。
6. 只有正确性通过且性能未回退时保留修改；失败时保留真实命令、stdout/stderr 和环境摘要。

## 10. 官方来源

- Triton 支持平台与硬件：<https://github.com/triton-lang/triton#compatibility>
- Triton occupancy / persistent 示例：<https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html>
- Triton `tl.dot` 精度：<https://triton-lang.org/main/python-api/generated/triton.language.dot.html>
- NVIDIA CC / Tensor Core / 资源表：<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- RTX 3090 官方规格：<https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090/>
- Nsight Compute CLI：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
