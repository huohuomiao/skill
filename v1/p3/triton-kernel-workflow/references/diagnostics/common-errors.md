# 常见错误

| 错误类型                | 错误描述                                      | 导致后果                       | 修正原则                                        |
| :---------------------- | :-------------------------------------------- | :----------------------------- | :---------------------------------------------- |
| **1. 参数重定义**       | 在 `configs` 定义了参数又在 Kernel 内手动赋值 | 编译失败或 Autotune 失效       | 内部仅声明 `tl.constexpr`，不赋值               |
| **2. 接口不一致**       | Launch 传参个数/顺序与 Kernel 定义不符        | `TypeError` 或内存访问错乱     | 严格核对，建议关键字传参                        |
| **3. 平台/设备不一致**  | 代码设备字面量与目标后端不一致                | 目标后端无法执行               | 按目标平台规则统一设备与同步 API                |
| **4. 外部算 Block**     | 在 Launch 参数位传入 `cdiv` 计算结果          | 逻辑混乱，违背并行架构设计     | 内部使用 `tl.program_id` 自行分块               |
| **5. 缺少 Mask**        | `load/store` 不带边界判定                     | **内存越界 (Out of Bounds)**   | 始终计算 `mask = offsets < size`                |
| **6. 基址缺失**         | 计算偏移忘记加 `pid * BLOCK_SIZE`             | 所有计算块重复处理第一块数据   | `offsets = base + pid * BLOCK + arange`         |
| **7. input device错误** | 输入不在 Kernel 所使用的加速设备上            | 指针设备不匹配或隐式拷贝       | 所有输入、输出和 reference 明确使用预期设备     |

## 错误示例：

1. Autotune 参数重复定义
   错误现象：在 triton.autotune 的 configs 中定义了某个参数（如 BLOCK_SIZE），但在 Kernel 函数内部又手动赋值或通过 tl.constexpr 重新定义。

后果：编译错误，提示参数多次定义；或者 Autotune 传入的值被内部硬编码覆盖，导致性能调优失效。

```python
# ❌ 错误示例
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 128})], # 已经定义了 BLOCK_SIZE
    key=['n_elements'],
)
@triton.jit
def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    BLOCK_SIZE = 256  # ❌ 错误：在函数内部重新赋值，这会覆盖 autotune 的配置
    ...
```

修正：移除内部赋值，完全信任 Autotune 传入的参数。



2. Kernel 接口与 Launch 函数不一致
   错误现象：JIT 函数定义的形参个数、顺序或类型，与 Python 端调用时（Launch）传入的实参不匹配。
   后果：运行时抛出 TypeError 或 Argument Error。最危险的情况是类型强制转换导致内存地址计算错误。

```python
# ❌ 错误示例
@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...

# Launch 调用时
# ❌ 错误：漏传了 n_elements，或者参数顺序反了
kernel[(grid,)](x_ptr, y_ptr, BLOCK_SIZE=128)
```

修正：严格核对形参列表。建议在 Launch 时对非指针参数使用显式关键字传参。

3. 平台关键字未转换
   错误现象：迁移代码后仍保留源平台的设备字面量、同步 API 或环境变量。

后果：目标后端无法识别设备，或在底层调用时触发驱动异常。平台具体替换规则必须从对应共享平台配置读取；MLU 目标读取 `{skill_root}/references/backend/platform-rules.md`。

```python
# ❌ 错误示例
x = torch.randn(1024, device=SOURCE_DEVICE)  # ❌ 源平台设备
x = torch.randn(1024, device=TARGET_DEVICE)  # ✅ 目标平台设备
```


4. 在 Launch 函数内计算任务划分（Block 数量）并传参
   错误现象：在 Python 端（Launch 函数）尝试计算 Grid 的拆分逻辑（例如有多少个 Tile），并将其作为普通参数传给 Kernel。

后果：代码冗余且易错。Triton 的核心逻辑是让每个线程块（Program）通过 tl.program_id 自主计算它负责哪一部分数据，而不是被动地接收“我是第几个块”的指令。

```python
# ❌ 错误示例
# 在 Launch 时计算了划分数量，传给了 Kernel
grid = lambda meta: (triton.cdiv(M, meta['TILE_M']), triton.cdiv(L, meta['TILE_N']))
batched_matmul_kernel[grid](
    A, B, C,
    N, M, K, L,
    ...,
    triton.cdiv(M, 32), # ❌ 错误：不应在参数位手动传划分结果
    triton.cdiv(L, 32),
)
```

✅ 正确做法：在 Kernel 内部使用 tl.program_id 和坐标重组
Kernel 应该只接收原始的矩阵维度（N, M, K, L），然后根据自身的 program_id 自行计算索引。

```python
@triton.jit
def batched_matmul_kernel(A, B, C, N, M, K, L, ...):
    # 获取当前程序块在 grid 中的坐标
    pid_m = tl.program_id(0) # 对应 M 轴的第几个 Tile
    pid_n = tl.program_id(1) # 对应 L 轴的第几个 Tile
    pid_batch = tl.program_id(2) # 对应 Batch 轴

    # 根据 pid 计算该 block 负责的起始行和列
    rm = pid_m * TILE_M + tl.arange(0, TILE_M)
    rn = pid_n * TILE_N + tl.arange(0, TILE_N)

    # 后续进行 load 和计算...
```



5. 忘记处理边界 Mask
   错误现象：在进行 tl.load 或 tl.store 时，直接加载整个 BLOCK_SIZE，没有考虑到 n_elements 可能不是 BLOCK_SIZE 的整数倍。

后果：内存越界访问（Out of Bounds）。这可能导致程序直接 Segfault，或者读取到垃圾数据导致计算精度异常。

```python
# ❌ 错误示例
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# ❌ 错误：如果 n_elements = 100, BLOCK_SIZE = 128，后面 28 个元素会越界
data = tl.load(ptr + offsets)
```

修正：必须传入 mask 参数。

```python
# ✅ 正确做法
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0) # 越界处填充 0
```

6. 指针运算缺少基础偏移
   错误现象：在计算 offsets 时，只使用了 tl.arange，忘记加上 pid * BLOCK_SIZE（当前的基地址偏移）。

后果：所有的 Program ID (pid) 都在处理同一块内存区域（通常是数组的最前面一部分），计算结果完全错误。

```python
# ❌ 错误示例
pid = tl.program_id(0)
# ❌ 错误：每个 pid 算出来的 offsets 都是一样的 [0, 1, ..., 127]
offsets = tl.arange(0, BLOCK_SIZE)
```

修正：确保包含 pid 的线性偏移计算。

```pyhton
# ✅ 正确做法
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```
