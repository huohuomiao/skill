# GenTestCode

## 职责概述

GenTestCode 负责根据需求文档和生成的 Triton kernel 代码，生成完整的测试代码。测试代码包括数据生成、精度验证和性能测试，用于验证 kernel 的正确性。

## 输入

| 来源 | 内容 |
|------|------|
| Extractor 输出 | `{输出存储路径}/Extractor/requirement.md` |
| io_shapes | `{输出存储路径}/KernelGen/step1_io_shapes.json` |
| Step 5 输出 | `{输出存储路径}/KernelGen/step5_kernel_code.py` |
| MLU 平台规则 | `.claude/skills/share/mlu/references/platform-rules.md` |
| 用户输入 | 输出存储路径（默认为 `output_dir`） |

## 输出

| 输出类型 | 说明 |
|---------|------|
| 文件输出 | `{输出存储路径}/KernelGen/step6_test_code.py` - 完整的测试代码（包含 triton code + 精度验证 + 性能测试） |

## 执行步骤

### 步骤 0：检查测试完整性（快速路径）

在执行步骤 1 之前，先检查 Step 5 输出的 triton 代码中是否已包含完整的测试代码。

#### 检查标准

**完整的测试代码必须同时满足以下条件**：

1. **精度测试**（至少满足一项）：
   - 包含 `accuracy_test` 函数或 `test_accuracy` 函数
   - 包含 `torch.allclose` 或 `torch.isclose` 调用
   - 包含 Triton 输出与 PyTorch/NumPy 参考实现的对比逻辑

2. **性能测试**（至少满足一项）：
   - 包含 `performance_test` 函数或 `test_performance` 函数
   - 包含 `triton.testing.do_bench` 调用
   - 包含运行时间测量或吞吐量计算逻辑

#### 判断结果

| 检查结果 | 处理方式 |
|---------|--------|
| **测试完整**（同时有精度+性能测试） | 直接使用原始代码，跳过步骤 1-3，将原始代码保存为 step6_test_code.py |
| **测试不完整**（缺少精度或性能测试） | 继续执行步骤 1-3，补充完整测试 |
| **无测试代码** | 继续执行步骤 1-3，生成完整测试 |

#### 注意事项

- 检查的是 **Step 5 输出的代码**（即 `{输出存储路径}/KernelGen/step5_kernel_code.py`）

### 步骤 1：解析需求文档、io_shapes 和 Triton 代码

从 Extractor 的需求文档中提取测试相关信息：
- 计算逻辑公式
- 输入/输出数据规格（shapes、dtypes、contiguous）
- 多组测试数据的规格
- 参数信息（axis、keepdim 等）

从 `step1_io_shapes.json` 中读取 io_shapes，用于生成测试输入：
- 该文件的顶层结构即为 io_shapes 对象（每个键是指针名，值含 type/axis/shape/contiguity）
- 根据 io_shapes 中的 shape 生成对应形状的测试张量
- 根据 io_shapes 中的 contiguity 决定是否需要生成非连续张量
- 根据 io_shapes 中的 axis 名称（如 N、M、K）生成对应的常量定义

从 Step 5 的 triton code 中提取：
- kernel 函数名
- wrapper 函数名
- 函数签名

### 步骤 2：处理不完整数据

**主要参考来源**：优先从 `step1_io_shapes.json` 获取输入输出规格信息。

如果 io_shapes 中某些数据规格为 `None` 或不完整，GenTestCode **自动生成合理的测试数据**：

1. **形状（shapes）**：优先从 io_shapes 中读取，若缺失则根据计算逻辑生成合理的形状
2. **数据类型（dtypes）**：若 io_shapes 中未指定，默认使用 float32
3. **连续性（contiguous）**：若 io_shapes 中未指定，默认生成全连续的数据

### 步骤 2.1：精度阈值设置

精度阈值的选择遵循以下**优先级规则**：

#### 规则 1（最高优先级）：客户已约定精度标准

如果 `requirement.md` 中明确给出了精度阈值（如 `atol`、`rtol`、`tolerance`、"精度要求"等字段），**必须严格使用客户指定的精度标准**，不得擅自放宽或收紧。

#### 规则 2（兜底）：客户未约定时，按计算公式选择

查看 `requirement.md` 中的"计算逻辑公式"，根据是否含有**累计类型（accumulation）计算**来决定：

| 场景 | 典型算子 | `atol` / `rtol` |
|------|---------|----------------|
| **不含累计类型计算** | elementwise（add / mul / relu / sigmoid 等）、copy、简单变换 | `1e-4` |
| **含累计类型计算** | reduce（sum / mean / max / min / argmax）、matmul、dot、conv、softmax、layernorm、cumsum 等 | `1e-3` |

**累计类型计算的判定要点**：公式中存在跨多个元素的求和 / 内积 / 逐步累积（循环累加）等操作，浮点累加误差会随参与元素个数放大，因此需要更宽松的阈值。
#### 落地要求

- 在生成的测试代码中，`torch.allclose(..., atol=X, rtol=Y)` 的 `atol` / `rtol` 必须等于按上述规则选出的值。
- 若同时受规则 1 约束，以客户指定为准。
- **注意**：不要擅自添加额外的精度测试项目，例如比特级精度测试。

### ⚠️ 特别提醒：禁止修改输入的 Triton Kernel 代码

**在生成测试代码时，必须保持 Step 5 输出的 Triton kernel 代码完全不变，禁止任何修改，包括但不限于：**
- 不能添加、删除或修改 kernel 函数中的任何代码
- 不能修改变量名、函数名、注释
- 不能改变算法逻辑
- 只能原样复制 Step 5 的 kernel 代码到测试文件中

### 步骤 3：生成完整测试代码

**⚠️ 重要要求：测试代码必须严格按照以下结构生成**

**输出格式**：完整的可执行 .py 文件，包含：
1. **Triton kernel 代码**（从 Step 5 获取）
2. **数据生成函数** - 根据 shapes、dtypes 创建测试张量
3. **参考实现函数** - 使用 PyTorch 原生操作实现算子逻辑
4. **精度验证函数** - 对比 Triton 和 PyTorch 结果
5. **性能测试函数** - 测量 Triton 和 PyTorch 性能

**⚠️ 强制要求：测试代码必须严格按照下面的测试模版生成**

**⚠️ 硬性要求：性能测试必须使用 triton.testing.do_bench() 进行测试**

#### 性能带宽计算要求

生成性能测试时，带宽（GB/s）必须按算子的**实际读写数据类型**估算，不得无脑使用 `2 * x.numel() * x.element_size()`。

- 优先从 `io_shapes`、wrapper 返回值和参考实现输出中识别所有输入/输出 tensor。
- `total_bytes` 应累计每个实际读/写 tensor 的 `numel() * element_size()`；输出 dtype 可能不同于输入 dtype，例如 FP8 量化中输入为 `float32`，输出 `y` 为 `float8`，`scale` 为 `float32`。
- 若输出 shape 含 `keepdim`、scale、indices、mask 等额外 tensor，必须按各自 shape 和 dtype 单独计入。
- 对于 in-place 算子，按实际读写口径计入被修改 tensor，通常至少包含一次读和一次写。
- 如果需求或 truth 脚本明确规定吞吐口径（例如只按输入 bytes 统计），必须在测试代码中用注释说明该口径，并保持 PyTorch 与 Triton 使用同一 `total_bytes`。
- PyTorch 和 Triton 的 GB/s 只能使用同一个 `gbps(ms)` 函数计算，确保耗时相同时带宽不会因为口径不同而不一致。

推荐生成一个小工具函数，根据实际 tensor 计算 bytes：

```python
def tensor_bytes(t):
    return t.numel() * t.element_size()
```

**测试代码模版**：
```python
import torch
import triton
import triton.language as tl

# Triton Kernel Code (from Step 5)
# ==========================================================
@triton.jit
def kernel_name(...):
    ...

def wrapper_function(...):
    ...

# ==========================================================
# Input Creation
# ==========================================================

def create_inputs():
    """创建测试输入数据"""
    x = torch.randn(SHAPE, dtype=DTYPE, device='mlu')
    return x

# ==========================================================
# Reference Implementation
# ==========================================================

def torch_reference(x):
    """PyTorch 参考实现"""
    return torch.operation(x, dim=axis, keepdim=keepdim)

# ==========================================================
# Accuracy test
# ==========================================================

def accuracy_test():
    print("\n" + "=" * 60)
    print("ACCURACY TEST")
    print("=" * 60)
    x = create_inputs()

    torch_out = torch_reference(x)
    triton_out = wrapper_function(x)

    max_diff = (torch_out - triton_out).abs().max().item()
    mean_diff = (torch_out - triton_out).abs().mean().item()
    # atol / rtol 的取值遵循步骤 2.1 规则：
    #   - 客户已约定 → 使用客户值
    #   - 无累计计算 → 1e-4 / 1e-4
    #   - 含累计计算 → 1e-3 / 1e-3
    allclose = torch.allclose(torch_out, triton_out, atol=ATOL, rtol=RTOL)

    print(f"  Max  absolute diff : {max_diff:.2e}")
    print(f"  Mean absolute diff : {mean_diff:.2e}")
    print(f"  torch.allclose     : {allclose}")
    print("  >>> PASSED" if allclose else "  >>> FAILED")

    return allclose

# ==========================================================
# Performance test
# ==========================================================

def performance_test():
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)
    x = create_inputs()

    quantiles = [0.5, 0.2, 0.8]

    # 按实际读写 tensor 的 dtype/shape 统计 bytes；不要假设输出 dtype 等于输入 dtype。
    # 若算子有多个输入/输出，请分别累加每个 tensor 的 numel() * element_size()。
    sample_torch_out = torch_reference(x)
    sample_triton_out = wrapper_function(x)
    torch.mlu.synchronize()

    def tensor_bytes(t):
        return t.numel() * t.element_size()

    def outputs_bytes(out):
        if isinstance(out, torch.Tensor):
            return tensor_bytes(out)
        return sum(tensor_bytes(t) for t in out if isinstance(t, torch.Tensor))

    total_bytes = tensor_bytes(x) + outputs_bytes(sample_triton_out)
    gbps = lambda ms: total_bytes / ms * 1e-6

    ms_torch, _, _ = triton.testing.do_bench(lambda: torch_reference(x), quantiles=quantiles)
    ms_triton, _, _ = triton.testing.do_bench(lambda: wrapper_function(x), quantiles=quantiles)

    print(f"  {'PyTorch  reference':<35s} {ms_torch:.4f} ms   {gbps(ms_torch):.2f} GB/s")
    print(f"  {'Triton   wrapper_function':<35s} {ms_triton:.4f} ms   {gbps(ms_triton):.2f} GB/s")
    print(f"  Speedup torch/triton : {ms_torch / ms_triton:.2f}x")

# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  {算子名} Test")
    print(f"  Shape: {SHAPE}, Device: mlu, dtype: {DTYPE}")
    print("=" * 60)

    # Run accuracy test
    accuracy_passed = accuracy_test()
    if not accuracy_passed:
        print("\nWARNING: Accuracy test failed!")

    # Run performance test
    performance_test()
```

### 步骤 4：处理 CUDA 到 MLU 的适配

仅在目标平台为 MLU 时读取 `.claude/skills/share/mlu/references/platform-rules.md`，按其中的设备字面量、同步 API、Grid、NRAM 和执行后端规则生成测试代码。不要在本通用测试流程中复制平台约束。

### 步骤 5：保存结果

将完整测试代码保存到 `{输出存储路径}/KernelGen/step6_test_code.py`

## 验证方式

| 检查项 | 验证方式 | 通过条件 |
|--------|--------|--------|
| Step 0 快速路径 | 检查 step5 代码中的测试完整性 | 同时有精度测试+性能测试则使用原始代码 |
| step1_io_shapes.json 存在 | 检查文件是否存在 | step1_io_shapes.json 存在且可读 |
| Step 5 输出存在 | 检查文件是否存在 | step5_kernel_code.py 存在且可读 |
| 语法检查 | Python 编译检查 | 无语法错误 |
| 测试函数完整 | 检查必需函数 | 包含 create_inputs, accuracy_test, performance_test |
| 输入规格一致性 | 对比测试代码中 SHAPE/DTYPE 与 step1_io_shapes.json（其次参考 requirement.md） | 每个维度、dtype 完全相等，无缩减/替换；未给出的字段按步骤 2 自行生成 |
| 数据生成 | 执行数据生成函数 | 生成正确的测试数据 |

## 回退机制

| 失败场景 | 处理方式 |
|---------|--------|
| Step 0 快速路径判断错误 | 降级到步骤 1-3，生成完整测试 |
| step1_io_shapes.json 不存在或无效 | 返回错误 |
| Step 5 输出不存在或无效 | 返回错误 |
| 数据规格不完整 | 内部重试（最多 3 次） |
| 测试代码生成失败 | 内部重试（最多 3 次） |
| 缺少必需的测试函数 | 内部重试（最多 3 次） |
