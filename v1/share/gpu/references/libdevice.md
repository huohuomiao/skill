# NVIDIA CUDA Triton Libdevice 参考

CUDA libdevice 是 NVIDIA 随 CUDA Toolkit 提供的 LLVM bitcode 设备函数库。Triton 的 NVIDIA backend 为其中一部分函数提供类型化 wrapper。它主要补足特殊数学、显式舍入和 CUDA intrinsic；它不是通用“必然更快”的激活函数库。

本文件面向 RTX 3090 (`sm_86`)。函数是否存在、具体 overload 和 lowering 以当前安装的 Triton 源码/文档及真实编译为准。

## 1. 官方 import 与命名空间

使用 Triton 官方教程中的 import：

```python
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.jit
def asin_kernel(x_ptr, y_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = libdevice.asin(x)
    tl.store(y_ptr + offsets, y, mask=mask)
```

规则：

- 正确调用是 `libdevice.name(...)`，由 `from triton.language.extra import libdevice` 引入。
- 不使用 `tl.extra.mlu.libdevice`，也不写 `fast_gelu`、`fast_silu`、`ultra_*` 等不存在于 NVIDIA CUDA libdevice wrapper 的名字。
- wrapper 只能在 `@triton.jit` 的编译上下文中处理 Triton tensor；不要在普通 Python host 代码中把它当 NumPy/PyTorch 函数调用。
- 默认让 Triton 选择随 NVIDIA backend 提供的 `libdevice.10.bc`。只有自定义工具链确有需要时才在 launch 传 `extern_libs={"libdevice": path}`，并记录 CUDA/Triton 版本。

## 2. 版本与 dtype 门禁

Triton 的 `triton.language.extra.libdevice` 是公共声明层，实际 CUDA wrapper 位于 NVIDIA backend。不同 Triton release 可能增加、删除或改名 wrapper，所以：

1. 静态生成前可在 host 侧查看 `dir(libdevice)`；这只证明 Python symbol 存在。
2. 用目标 dtype 编译一个最小 kernel；这才证明当前 wrapper 有对应 overload。
3. 在 RTX 3090 上真实 launch；这才证明 `sm_86` target 与所选 bitcode/toolchain 可运行。
4. 与 PyTorch/高精度 CPU reference 验证完整 domain 和特殊值。

大多数连续数学 wrapper 接受 `fp32` 和 `fp64`，不是原生 FP16/BF16 API。若输入是 FP16/BF16，显式提升到 FP32，再根据输出契约转换：

```python
x32 = x.to(tl.float32)
y32 = libdevice.log1p(x32)
y = y32.to(x.dtype)
```

不要假设 wrapper 自动选择 half/bfloat16 版本。RTX 3090 虽能执行 FP64，但 CC 8.6 的非 Tensor FP32:FP64 峰值吞吐比为 64:1；仅在接口或误差要求必须时使用 FP64。

## 3. 可用 API 类别

下面列出当前 Triton CUDA wrapper 的主要类别和代表名称。不是版本冻结清单；生成时只引用当前环境实际存在的名字。

### 3.1 基础、舍入与分类

| 目的 | 常见 wrapper | 说明 |
| --- | --- | --- |
| 绝对值 | `abs` | int32/int64/fp32/fp64 overload 依版本 |
| 取整 | `floor`, `ceil`, `trunc`, `round`, `rint`, `nearbyint` | tie 与当前 rounding mode 语义不同，不能互换 |
| 整数结果取整 | `llrint`, `llround` | 返回类型及溢出行为需测试 |
| 符号/分类 | `copysign`, `signbit`, `isnan`, `isinf`, `finitef`, `isfinited` | `finitef` 与 `isfinited` 是 dtype 区分的历史命名 |
| 相邻值 | `nextafter` | 适合边界测试，不等价于加 epsilon |
| 差/余数 | `fdim`, `fmod`, `remainder` | `fmod` 与 IEEE remainder 对负数不同 |
| 饱和 | `saturatef` | FP32，限制到 CUDA 定义范围 |

### 3.2 指数、对数、幂和根

| 目的 | 常见 wrapper | 选择注意 |
| --- | --- | --- |
| 指数 | `exp`, `exp2`, `exp10`, `expm1` | 小 `x` 的 `expm1(x)` 比 `exp(x)-1` 更稳健 |
| 对数 | `log`, `log2`, `log10`, `log1p`, `logb`, `ilogb` | 小 `x` 的 `log1p(x)` 比 `log(1+x)` 更稳健 |
| 幂 | `pow` | domain、负底数、整数指数、NaN/Inf 必测 |
| 平方根 | `sqrt`, `rsqrt`, `rsqrt_rn` | precise/fast 语义不要混用 |
| 立方根 | `cbrt`, `rcbrt` | 对负输入与 `pow(x, 1/3)` 行为不同 |
| 尺度分解 | `ldexp`, `scalbn` | 第二参数通常是整数 exponent |
| 范数 | `hypot`, `rhypot`, `norm3d`, `rnorm3d`, `norm4d`, `rnorm4d` | 可改善 overflow/underflow，但不承诺更快 |

### 3.3 三角与双曲函数

- 三角：`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`。
- π 缩放：`sinpi`, `cospi`；用于计算 `sin(pi*x)` / `cos(pi*x)` 时可减少显式乘法并改善某些 argument-reduction 情况。
- 双曲：`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`。

这些函数通常有 FP32/FP64 overload。大参数 argument reduction、接近定义域边界、signed zero 和 NaN 必须覆盖。`atan2(y, x)` 的参数顺序不能改成 `atan(y/x)`。

### 3.4 特殊函数

| 类别 | 常见 wrapper |
| --- | --- |
| 误差函数 | `erf`, `erfc`, `erfinv`, `erfcinv`, `erfcx` |
| 正态分布 | `normcdf`, `normcdfinv` |
| Gamma | `lgamma`, `tgamma` |
| Bessel | `j0`, `j1`, `jn`, `y0`, `y1`, `yn`, `cyl_bessel_i0`, `cyl_bessel_i1` |

这些函数是使用 libdevice 的主要理由之一，因为 Triton core language 未必提供对应 primitive。先确认 wrapper 在安装版本存在；特殊函数的 tail/domain 精度用参考实现验证，不用单一随机区间下结论。

### 3.5 显式舍入算术

CUDA wrapper 提供以下模式族（是否包含 FP32/FP64 overload 以当前版本为准）：

- 加法：`add_rn`, `add_rz`, `add_rd`, `add_ru`
- 减法：`sub_rn`, `sub_rz`, `sub_rd`, `sub_ru`
- 乘法：`mul_rn`, `mul_rz`, `mul_rd`, `mul_ru`
- 除法：`div_rn`, `div_rz`, `div_rd`, `div_ru`
- 倒数：`rcp_rn`, `rcp_rz`, `rcp_rd`, `rcp_ru`
- 平方根：`sqrt_rn`, `sqrt_rz`, `sqrt_rd`, `sqrt_ru`
- fused multiply-add：`fma_rn`, `fma_rz`, `fma_rd`, `fma_ru`，另有默认 `fma`

后缀含义：round-to-nearest (`rn`)、toward-zero (`rz`)、toward-negative-infinity (`rd`)、toward-positive-infinity (`ru`)。这些 API 用于明确数值语义，不是自动性能优化。若算法没有定向舍入要求，优先 Triton core 运算，让编译器正常融合。

### 3.6 数值转换与 bit reinterpret

wrapper 包含多组转换，例如：

- `float2int_*`, `float2uint_*`, `float2ll_*`, `float2ull_*`
- `double2float_*`, `double2int_*`, `double2uint_*`, `double2ll_*`, `double2ull_*`
- `int2float_*`, `uint2float_*`, `ll2float_*`, `ull2float_*`
- `int2double_rn`, `uint2double_rn`, `ll2double_*`, `ull2double_*`
- `float_as_int`, `float_as_uint`, `int_as_float`, `uint_as_float`
- `double_as_longlong`, `longlong_as_double`, `double2hiint`, `double2loint`, `hiloint2double`

数值 conversion 与 bit reinterpret 是两种语义。`float_as_int` 保留 bit pattern，而 `float2int_rn` 做数值舍入；禁止互换。返回 signed/unsigned 类型必须查看当前 wrapper 实现，尤其不要仅从函数名猜 Python/Triton dtype。

### 3.7 整数与位操作

常见 wrapper：

- 计数：`clz`, `popc`, `ffs`
- bit reverse/permute：`brev`, `byte_perm`
- 高位乘：`mulhi`, `mul24`
- average/add difference：`hadd`, `rhadd`, `sad`

这些函数有严格位宽和 signedness overload。对负数、移位、溢出和 64-bit 输入分别测试；不要沿用其他后端对低位整数的扩展假设。

## 4. CUDA fast intrinsic

Triton CUDA wrapper 当前可暴露以下 FP32 fast 类函数（安装版本可能不同）：

`fast_sinf`, `fast_cosf`, `fast_tanf`, `fast_expf`, `fast_exp10f`, `fast_logf`, `fast_log2f`, `fast_log10f`, `fast_powf`, `fast_dividef`。

这些名字映射 CUDA 的近似/快速路径，规则如下：

1. 仅按 wrapper 支持的 FP32 signature 调用；不要传 FP64 后期待自动降精度。
2. fast 与非 fast 的误差上界、特殊值和大参数行为不同。只有需求明确允许近似且 adversarial accuracy test 通过时才候选。
3. 不把 `tl.exp` 自动替换为 `fast_expf`：Triton 文档已说明部分 core math 本身是 fast/approximate，二者谁快必须看生成代码和实测。
4. 不存在统一的 `fast_sqrt`, `fast_rcp`, `fast_gelu`, `fast_silu`, `fast_sigmoid` CUDA Triton wrapper。不要由 CUDA intrinsic 的命名规律臆造 API。
5. NVIDIA CUDA libdevice/Triton wrapper 没有 MLU 风格 `ultra_*` 系列。

## 5. Core primitive 与 libdevice 的选择

| 需求 | 首选 | 原因 |
| --- | --- | --- |
| 常规逐元素 `exp/log/sqrt/sin/cos/erf` | 对应 `tl.*` | 语法简洁，编译器可选择目标 lowering |
| 需要明确 nearest rounding | `tl.div_rn`, `tl.sqrt_rn` 或已验证 libdevice `*_rn` | 明确精度语义 |
| core API 没有的特殊函数 | `libdevice.*` | 官方 CUDA 设备函数 |
| 需要稳定的 `log(1+x)`/`exp(x)-1` | `libdevice.log1p` / `expm1` | 避免相消；仍验证 dtype/domain |
| 用户允许 fast approximation | fast wrapper 或 core fast primitive，逐个基准 | 不做无证据全局替换 |
| 激活函数 GELU/SiLU/sigmoid | Triton 表达式/`tl.sigmoid` | libdevice 没有这些高层 wrapper |

同一表达式中不要无意义混用 core math 与 libdevice。先确定精度契约，然后选择最简单、可编译、可测的实现；只有测得收益才保留替换。

具体替换候选、精度/性能保留条件见 `../optimize/libdevice-opt.md`。高层 GELU/SiLU/sigmoid 没有对应 CUDA libdevice wrapper，不得臆造函数名。

## 6. 验证清单

1. import 只有 `from triton.language.extra import libdevice`。
2. 函数名存在于当前 Triton wrapper；没有 MLU、`ultra_*` 或臆造激活 API。
3. 输入 dtype 有确切 overload；FP16/BF16 显式提升到 FP32。
4. NaN、Inf、±0、subnormal、定义域边界和大参数按接口契约测试。
5. 与 baseline 使用相同 dtype、输入、warmup、repeat 和同步；同时报告 max abs/rel error 与性能分布。
6. 查看 PTX/SASS 或 NCU，确认替换确实改变目标指令/吞吐；没有证据则保持原实现。
7. 只在 correctness 通过且统计上没有回退时保留；误差或性能不合格立即回退。

## 7. 官方来源

- Triton libdevice 教程与 import：<https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html>
- Triton 公共 libdevice 声明：<https://github.com/triton-lang/triton/blob/main/python/triton/language/extra/libdevice.py>
- Triton NVIDIA CUDA wrapper 实现：<https://github.com/triton-lang/triton/blob/main/third_party/nvidia/language/cuda/libdevice.py>
- NVIDIA CUDA libdevice User's Guide：<https://docs.nvidia.com/cuda/libdevice-users-guide/index.html>
- CUDA 数学函数与精度说明：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix>
- CUDA CC 8.6 技术规格：<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
