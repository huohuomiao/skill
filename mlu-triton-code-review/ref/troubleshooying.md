# Triton MLU 故障排查与修复指南

本文档汇总 Triton kernel 在 Cambricon MLU 上运行时常见的错误现象、根因与修复策略，供 `mlu-triton-code-review` 在迭代修复阶段查阅。

---

## 一、常见错误及修复策略

### 1. Grid 维度超限

**错误信息（示例）**：
```
OutOfResources: out of resource: grid size, Required: 65536, Hardware limit: 65535.
Reducing block sizes or `num_stages` may help.
coreDim=xxxx can't be greater than UINT16_MAX
```

**原因**：MLU 单维 grid 上限为 `2**16 - 1 = 65535`（GPU 为 `2**32 - 1`），社区 GPU kernel 在大 shape 下容易触发。

**修复方案**：

- **方案 1：剔除 BLOCK_SIZE 过小的调优配置**
  MLU 片上存储更大，倾向于使用比 GPU 更大的 BLOCK_SIZE。直接调大 BLOCK_SIZE 使 `cdiv(size, BLOCK_SIZE) ≤ 65535` 即可：
  ```python
  N = x.numel()
  min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
  BLOCK_SIZE = max(32768, min_block_size)
  ```

- **方案 2：持久化内核（persistent kernel，推荐）**
  将 grid 固定为物理核数，由 kernel 内部循环处理多个块。该方案降低 host 端 launch 开销，且便于启用软流水优化：
  ```python
  @triton.jit
  def persistent_kernel(inp_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
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
  grid = (min(triton.cdiv(size, BLOCK_SIZE), core_num),)
  ```
  **注意**当尝试方案1后导致nram超限，则考虑使用方案2进行修复。

---

### 2. 片上内存（NRAM）溢出

**错误信息（示例）**：
```
OutOfResources: out of resource: NRAM, Required: 2131648, Hardware limit: 1048576
nram overflow, requires xxxx bits while xxxx bits available
out of on-chip memory
```

**原因**：MLU 与 GPU 后端编译机制存在差异，单次 `tl.load` / 中间张量规模过大会超出 NRAM 可用空间。每个 tile 按 `BLOCK_SIZE × sizeof(dtype)` 占用 NRAM，kernel 中存在多个中间张量时（如 `x`、`elu_x`、`output`）会累加。

**修复方案（首选）—— 直接调小 BLOCK_SIZE**：
若 kernel 已经是"外层按 BLOCK_SIZE 分块"的标准写法，**优先直接调小 BLOCK_SIZE**（或在 autotune 配置中剔除过大候选）。例如将 `BLOCK_SIZE` 从 262144 降到 65536，单张量占用即可从 1MB 降到 256KB。

**⚠️ 反模式：在已有分块之上再加一次"伪分块"**

以下是一个**错误的修复**示例：原 kernel 已经按 `BLOCK_NM` 外层分块，修复时又引入 `BLOCK_NM_SUB` 做内循环，但调用时 `BLOCK_NM == BLOCK_NM_SUB`：
```python
# ❌ 反例：BLOCK_NM = BLOCK_NM_SUB = 65536，内循环只执行一次
BLOCK_NM = 65536
BLOCK_NM_SUB = 65536
num_sub_blocks = tl.cdiv(BLOCK_NM, BLOCK_NM_SUB)  # = 1
for sub_idx in range(num_sub_blocks):
    ...
```
这种写法只是徒增循环开销和代码复杂度，实际等价于"直接把 BLOCK_NM 调小到 65536"。**sub-tiling 只在需要保留较大的"每核任务量"（BLOCK_SIZE > BLOCK_SIZE_SUB）时才有意义**。

**修复方案（进阶）—— 使用子块切分（sub-tiling）**：
若希望保留较大的"每核任务量"以降低 launch 开销 / 启用软流水，才将 BLOCK_SIZE 与片上驻留规模解耦：
```python
@triton.jit
def kernel_func(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE

    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < n_elements

        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute(data)
        tl.store(output_ptr + offsets, result, mask=mask)
```

建议取值：`BLOCK_SIZE` 决定每核总任务量，`BLOCK_SIZE_SUB` 决定单次片上驻留规模（按 NRAM 容量向下取 2 的幂）；**必须满足 `BLOCK_SIZE > BLOCK_SIZE_SUB`**，否则退化为方案一。
---

### 3. 精度问题（NaN / Inf / 数值误差）

**错误现象**：执行成功但结果包含 NaN/Inf，或 `torch.allclose` 失败、`max_diff` 超阈值。

#### 原因 1：逻辑运算符使用错误

Triton 中对 tile 做逐元素布尔运算必须用位运算：
```python
# ❌ 错误：and/or 对 tile 语义不明确
mask = mask1 and mask2

# ✅ 正确
mask = mask1 & mask2
```

#### 原因 2：`tl.load` 的 `other` 取值不当

当被加载值会被再次用作索引或参与后续计算时，`other=0` 可能被误当作合法值使用：
```python
# ❌ 错误：other=0 可能被当作有效索引
indices = tl.load(index_ptr + offsets, mask=mask, other=0)

# ✅ 正确：用越界值（或非法哨兵值）区分
indices = tl.load(index_ptr + offsets, mask=mask, other=N)
```

#### 原因 3：load/store 的 mask 混用

`tl.store` 的 mask 应由**输出边界**决定；`tl.load` 的 mask 可以进一步叠加输入有效性过滤：
```python
# ❌ 错误：store 使用了带输入过滤的 mask，导致该写的位置没写
final_mask = out_mask & index_valid_mask
selected = tl.load(inp_ptr + inp_off, mask=final_mask, other=0.0)
tl.store(out_ptr + out_off, selected, mask=final_mask)

# ✅ 正确
selected = tl.load(inp_ptr + inp_off, mask=final_mask, other=0.0)
tl.store(out_ptr + out_off, selected, mask=out_mask)
```

#### 原因 4：归约/累加 dtype 不当

低精度（fp16/bf16）直接累加容易造成精度丢失：
```python
# ✅ 将累加器提升到 fp32
acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
for ...:
    acc += tl.load(...).to(tl.float32)
out = acc.to(tl.float16)
```

#### 原因 5：BLOCK_SIZE 过大导致数值不稳定

对于需要逐元素归一化、softmax 等算法，过大的 BLOCK_SIZE 可能叠加累积误差：
```python
# 可尝试：在保证性能的前提下，适当减小 BLOCK_SIZE
BLOCK_SIZE = 1024  # 从 4096 减至 1024
---

### 4. 编译错误

**错误信息（示例）**：
```
compilation failed
AttributeError: module 'triton.language' has no attribute 'xxx'
```

**检查项**：
1. Triton 原语是否被 MLU 后端支持（对照 `ref/semantic.md`）
2. 是否误用了 GPU 专属 API（如 `triton.runtime.driver.active.get_active_torch_device`）
3. 是否引用了环境中缺失的外部库

**修复方向**：
```python
# ❌ GPU 专属残留
DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ✅ MLU 直接使用
import torch
import torch_mlu  # 必须导入以注册 mlu 后端
DEVICE = "mlu"
```

---

### 5. 设备/平台错误

**典型现象**：`cuda is not available`、`RuntimeError: No HIP/CUDA GPUs are available`。

**修复**：
- 全量替换 `cuda` → `mlu`
- `tensor.cuda()` → `tensor.to("mlu")`
- `torch.cuda.xxx` → `torch.mlu.xxx`
- 确保测试代码中 `import torch_mlu` 在 `import torch` 之后

---

### 6. Kernel 接口不一致

**错误信息**：`TypeError: xxx got unexpected keyword argument`、参数个数不符、访存错乱。
**修复原则**：
- Launch 端建议使用**关键字传参**，避免位置错位
- 常量参数（BLOCK_SIZE 等）一律声明为 `tl.constexpr`
- 若使用 `@triton.autotune`，configs 中已定义的参数，kernel 内部**不要**再显式赋值

**常见坑：在 kernel 中引用模块级全局常量未声明为 `tl.constexpr`**

若 kernel 代码中使用了 Python 模块级常量（如激活函数系数 `ALPHA`、`LAMBDA`），这些变量 **不会**被 Triton 自动捕获——Triton jit 只识别函数参数。正确做法是将它们作为 kernel 参数传入，并标注为 `tl.constexpr`：

```python
# ❌ 错误：kernel 内部直接引用模块级全局变量
ALPHA = 1.6732632423543772
LAMBDA = 1.0507009873554805

@triton.jit
def selu_kernel(x_ptr, output_ptr, n, BLOCK: tl.constexpr):
    ...
    out = LAMBDA * tl.where(x > 0, x, ALPHA * (tl.exp(x) - 1))  # 编译失败或数值错误
```

```python
# ✅ 正确：作为 constexpr 参数传入
@triton.jit
def selu_kernel(x_ptr, output_ptr, n,
                ALPHA: tl.constexpr, LAMBDA: tl.constexpr,
                BLOCK: tl.constexpr):
    ...
    out = LAMBDA * tl.where(x > 0, x, ALPHA * (tl.exp(x) - 1))

# launch 时显式传入
selu_kernel[grid](x, out, n,
                  ALPHA=1.6732632423543772,
                  LAMBDA=1.0507009873554805,
                  BLOCK=1024)
```

只有标为 `tl.constexpr` 的数值才能在 kernel 内参与编译期表达式（如做 `tl.where` 的标量分支、与 `BLOCK` 一起算索引等）。普通 runtime 参数在 kernel 内是 scalar tensor，不能当作编译期常量使用。

---

### 7. 数据类型错误

**错误信息**：`dtype mismatch`、`unsupported dtype`、`int64 is not supported`。

**修复方向**：
- `tl.arange` 默认产出 int32，不要手动强转 int64
- MLU 对 fp64 / int64 的支持有限，若 semantic.md 标注为不支持，改用 fp32 / int32
- 对比 `ref/semantic.md` 确认该原语在目标 dtype 下是否可用

```python
# ❌ 可能退化为 scalar 或不支持
position = tl.arange(0, BLOCK_SIZE).to(tl.int64)

# ✅
position = tl.arange(0, BLOCK_SIZE).to(tl.int32)
```

---

### 8. libdevice 使用错误

**错误信息**：`tl.extra.mlu.libdevice.xxx` 相关报错。

**修复**：对照 `ref/libdevice.md` 核对：
- 调用路径是否为 `tl.extra.mlu.libdevice.<op>`
- 输入 dtype 是否在该算子的支持列表内
- 是否遗漏了舍入/饱和后缀（`_rn`、`_sat` 等）

---

### 9. 内存越界 / 非法访问

**错误信息**：`CNRT_ERROR`、`illegal memory access`、`out of bounds`。

**修复清单**（按顺序排查）：
1. `offsets` 计算是否包含 `pid * BLOCK_SIZE` 基址
2. `tl.load` / `tl.store` 是否都带 `mask = offsets < size`
3. 多维 tile 的 mask 是否正确广播：`mask = (row < M)[:, None] & (col < N)[None, :]`
4. 指针步长是否与 Tensor 的 stride 一致

---

## 二、MLU 与原生 Triton 的行为差异

以下语义在 MLU 上与 GPU 原生 Triton 不同，迁移代码时若未意识到，可能表现为"没报错但行为异常"。

### 1. `num_warps` 的语义变化

原生 Triton 中该参数表示 block 内启动的线程数。寒武纪架构下它与任务类型对齐：
- `num_warps=1` → **Block 任务**（当 gridX 为 4 的倍数且不超过设备 IPU 总数时，会自动启用 U1 以降低下发延时）
- `num_warps=4` → **Union1 任务**
- 其他取值暂不支持

### 2. `BLOCK_SIZE` 解除"2 的幂"约束

MLU 单核算力更高，需要更大的分块才能充分利用带宽。寒武纪 Triton **允许非 2 次幂的 BLOCK_SIZE**（例如 24 可能优于 16）。`tl.arange` 仍要求 2 次幂，但 autotune 的 `BLOCK_SIZE` 候选可自由选取。

### 3. `cache_modifier` 语义重解释

`tl.load` / `tl.store` 的 `cache_modifier` 参数在 MLU 上映射如下：
- 不设置：按硬件默认策略
- `.ca` / `.cg`：**开启** MLU 缓存
- `.wb` / `.cs` / `.wt`：**关闭** MLU 缓存

### 4. `eviction_policy` 不支持

该参数在 MLU 上**不生效**（不会报错，但也不起作用），不要依赖它调优。

### 5. Atomic memory sync scope

原生 Triton 的 atomic 原语支持 `gpu` / `cta` / `system` 三种 scope，**MLU 仅支持 `gpu`**，其他取值不生效。

---

## 三、调试与定位技巧

### 1. 分步验证

```python
# Step 1：确认能跑通
try:
    result = my_kernel(**inputs)
    print("[OK] kernel executed")
except Exception as e:
    print(f"[FAIL] {e}")
    raise

# Step 2：检查 NaN/Inf
print("has_nan:", torch.isnan(result).any().item())
print("has_inf:", torch.isinf(result).any().item())

# Step 3：比对参考实现
torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
```

### 2. 缩小问题规模

用最小可复现数据集定位：
```python
x = torch.randn(16, 16, device="mlu")
```
### 3. kernel 内打印

```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    tl.device_print("pid", pid)  # 仅用于调试
```

### 4. 与参考 CPU/GPU 实现对拍

若 MLU 结果可疑，先让同一份 PyTorch 参考实现在 CPU 上跑出 golden，再比对 MLU 输出，隔离是"算法错"还是"平台适配错"。

---

## 四、错误诊断流程

```
1. 运行测试脚本
   ↓
2. 识别错误类型
   ↓
3. 按类型查表并应用修复
   ├── 编译错误         → §4（检查原语、平台残留）
   ├── Grid 超限        → §1（固定核数或增大 BLOCK_SIZE）
   ├── NRAM 溢出        → §2（子块切分）
   ├── 内存越界         → §9（mask、基址、广播）
   ├── 设备错误         → §5（cuda → mlu）
   ├── 接口错误         → §6（关键字传参、constexpr）
   ├── dtype 错误       → §7（对照 semantic.md）
   ├── libdevice 错误   → §8（对照 libdevice.md）
   └── 精度问题         → §3（and→&, other, mask, 累加 dtype, BLOCK_SIZE）
   ↓
4. 保存 triton_fixed_{N}.py 并重新执行
   ↓
5. 若连续 2 次同类错误无进展 → 提前终止
```

