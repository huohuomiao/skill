# MLU Triton Libdevice (tl.math) 完整算子支持文档

## 目录

- 基础数学、三角、反三角与双曲函数
- 指数对数与幂根函数
- 舍入、算术与类型转换
- 位运算与 MLU 特有优化
- 不支持算子与类型
- 使用建议

> **核心校验规则：**
> 1. **数据类型：** 若"支持类型"未标明 `fp64`，则该算子在 MLU 平台上**禁止**使用 `double/fp64`。
> 2. **命名空间：** 所有函数均在 `tl.math` 下调用。
> 3. **对齐逻辑：** 标注为 "None" 的类型表示 MLU Triton 与标准 Triton 行为完全对齐。

---
## 基础数学函数

**支持: 7 个 | 不支持: 0 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `abs` | `abs` | bf16, fp16, fp32, int16, int32, int64, int8 | MLU 不支持数据类型: fp64 |
| `copysign` | `copysign` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `fdim` | `fdim` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `fmod` | `fmod` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `hypot` | `hypot` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `nextafter` | `nextafter` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `remainder` | `remainder` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |

## 三角函数

**支持: 3 个 | 不支持: 0 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `cos` | `cos` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sin` | `sin` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `tan` | `tan` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |

## 反三角函数
**支持: 4 个 | 不支持: 0 个**
### ✅ 支持的算子
| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `acos` | `acos` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `asin` | `asin` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `atan` | `atan` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `atan2` | `atan2` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
## 双曲函数
**支持: 6 个 | 不支持: 0 个**
### ✅ 支持的算子
| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `acosh` | `acosh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `asinh` | `asinh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `atanh` | `atanh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `cosh` | `cosh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sinh` | `sinh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `tanh` | `tanh` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |

## 指数对数
**支持: 14 个 | 不支持: 2 个**
### ✅ 支持的算子
| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `exp` | `exp` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `exp10` | `exp10` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `exp2` | `exp2` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `expm1` | `expm1` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `fast_exp10f` | `fast_exp10f` | fp32 | - |
| `fast_expf` | `fast_expf` | fp32 | - |
| `fast_log10f` | `fast_log10f` | fp32 | - |
| `fast_log2f` | `fast_log2f` | fp32 | - |
| `ilogb` | `ilogb` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `log` | `log` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `log10` | `log10` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `log1p` | `log1p` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `log2` | `log2` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `logb` | `logb` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
### ❌ 不支持的算子
| 函数名 |
|:---|
| `fast_logf` |
| `ldexp` |


## 幂根函数

**支持: 5 个 | 不支持: 6 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `cbrt` | `cbrt` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `fast_powf` | `fast_powf` | fp32 | - |
| `pow` | `pow` | bf16, fp16, fp32, int32 | MLU 不支持数据类型: fp64 |
| `rsqrt` | `rsqrt` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sqrt` | `sqrt` | fp32 | MLU 不支持数据类型: fp64 |

### ❌ 不支持的算子

| 函数名 |
|:---|
| `rcbrt` |
| `rsqrt_rn` |
| `sqrt_rd` |
| `sqrt_rn` |
| `sqrt_ru` |
| `sqrt_rz` |


## 舍入取整

**支持: 8 个 | 不支持: 0 个**
## 舍入取整

**支持: 8 个 | 不支持: 0 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `ceil` | `ceil` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `floor` | `floor` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `llrint` | `llrint` | fp32 | MLU 不支持数据类型: fp64 |
| `llround` | `llround` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `nearbyint` | `nearbyint` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `rint` | `rint` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `round` | `round` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `trunc` | `trunc` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |


## 舍入算术

**支持: 17 个 | 不支持: 4 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `add_rd` | `add_rd` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `add_rn` | `add_rn` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `add_ru` | `add_ru` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `add_rz` | `add_rz` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `div_rd` | `div_rd` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `div_rn` | `div_rn` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `div_ru` | `div_ru` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `div_rz` | `div_rz` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `fma` | `fma` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `mul_rd` | `mul_rd` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `mul_rn` | `mul_rn` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `mul_ru` | `mul_ru` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `mul_rz` | `mul_rz` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sub_rd` | `sub_rd` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sub_rn` | `sub_rn` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sub_ru` | `sub_ru` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `sub_rz` | `sub_rz` | fp16, fp32 | MLU 不支持数据类型: fp64 |
### ❌ 不支持的算子

| 函数名 |
|:---|
| `fma_rd` |
| `fma_rn` |
| `fma_ru` |
| `fma_rz` |


## 类型转换

**支持: 18 个 | 不支持: 47 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `double2float_rn` | `double2float_rn` | fp64 | - |
| `double2ll_rz` | `double2ll_rz` | fp64 | - |
| `float2int_rd` | `float2int_rd` | fp32 | - |
| `float2int_rn` | `float2int_rn` | fp32 | - |
| `float2int_ru` | `float2int_ru` | fp32 | - |
| `float2int_rz` | `float2int_rz` | fp32 | - |
| `float2ll_rn` | `float2ll_rn` | fp32 | - |
| `float2ll_rz` | `float2ll_rz` | fp32 | - |
| `float2uint_rn` | `float2uint_rn` | fp32 | - |
| `float2uint_rz` | `float2uint_rz` | fp32 | - |
| `float2ull_rz` | `float2ull_rz` | fp32 | - |
| `int2float_rn` | `int2float_rn` | int32 | - |
| `int2float_rz` | `int2float_rz` | int32 | - |
| `ll2double_rn` | `ll2double_rn` | int64 | - |
| `ll2float_rn` | `ll2float_rn` | int64 | - |
| `ll2float_rz` | `ll2float_rz` | int64 | - |
| `uint2float_rn` | `uint2float_rn` | uint32 | - |
| `ull2float_rn` | `ull2float_rn` | uint64 | - |

### ❌ 不支持的算子

| 函数名 |
|:---|
| `double2float_rd` |
| `double2float_ru` |
| `double2float_rz` |
| `double2hiint` |
| `double2int_rd` |
| `double2int_rn` |
| `double2int_ru` |
| `double2int_rz` |
| `double2ll_rd` |
| `double2ll_rn` |
| `double2ll_ru` |
| `double2loint` |
| `double2uint_rd` |
| `double2uint_rn` |
| `double2uint_ru` |
| `double2uint_rz` |
| `double2ull_rd` |
| `double2ull_rn` |
| `double2ull_ru` |
| `double2ull_rz` |
| `float2ll_rd` |
| `float2ll_ru` |
| `float2uint_rd` |
| `float2uint_ru` |
| `float2ull_rd` |
| `float2ull_rn` |
| `float2ull_ru` |
| `hiloint2double` |
| `int2double_rn` |
| `int2float_rd` |
| `int2float_ru` |
| `ll2double_rd` |
| `ll2double_ru` |
| `ll2double_rz` |
| `ll2float_rd` |
| `ll2float_ru` |
| `uint2double_rn` |
| `uint2float_rd` |
| `uint2float_ru` |
| `uint2float_rz` |
| `ull2double_rd` |
| `ull2double_rn` |
| `ull2double_ru` |
| `ull2double_rz` |
| `ull2float_rd` |
| `ull2float_ru` |
| `ull2float_rz` |


## 位运算

**支持: 2 个 | 不支持: 1 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `clz` | `clz` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `popc` | `popc` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |

### ❌ 不支持的算子

| 函数名 |
|:---|
| `ffs` |


## MLU 特有优化

**支持: 1 个 | 不支持: 3 个**
### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `fast_dividef` | `fast_dividef` | bf16, fp16, fp32 | - |

### ❌ 不支持的算子

| 函数名 |
|:---|
| `fast_cosf` |
| `fast_sinf` |
| `fast_tanf` |


## 其他

**支持: 282 个 | 不支持: 31 个**

### ✅ 支持的算子

| 函数名 | MLU Triton 函数名 | 支持的数据类型 | 说明 |
|:---|:---|:---|:---|
| `Not supported` | `abs_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `add` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `add_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `accurate_sqrt` | fp32 | - |
| `Not supported` | `bfloat162byte` | bf16 | - |
| `Not supported` | `bfloat162float` | bf16 | - |
| `Not supported` | `bfloat162half` | bf16 | - |
| `Not supported` | `bfloat162int` | bf16 | - |
| `Not supported` | `bfloat162ll` | bf16 | - |
| `Not supported` | `bfloat162short` | bf16 | - |
| `Not supported` | `bfloat162ubyte` | bf16 | - |
| `Not supported` | `bfloat162uint` | bf16 | - |
| `Not supported` | `bfloat162ull` | bf16 | - |
| `Not supported` | `bfloat162ushort` | bf16 | - |
| `Not supported` | `bfloat162float8e4nv` | bf16 | - |
| `Not supported` | `bfloat162float8e4nv_sat` | bf16 | - |
| `Not supported` | `bfloat162float8e5` | bf16 | - |
| `Not supported` | `bfloat162float8e5_sat` | bf16 | - |
| `Not supported` | `bitwise_and` | int1, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `bitwise_not` | int1, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `bitwise_or` | int1, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `byte2bfloat16` | int8 | - |
| `Not supported` | `byte2float` | int8 | - |
| `Not supported` | `byte2half` | int8 | - |
| `Not supported` | `byte2int` | int8 | - |
| `Not supported` | `byte2ll` | int8 | - |
| `Not supported` | `byte2short` | int8 | - |
| `Not supported` | `byte2ubyte` | int8 | - |
| `Not supported` | `byte2uint` | int8 | - |
| `Not supported` | `byte2ull` | int8 | - |
| `Not supported` | `byte2ushort` | int8 | - |
| `Not supported` | `cast_f8e4_to_f32` | fp8e4nv | - |
| `Not supported` | `cos_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `ctz` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `cycle_sub_mul_exp` | fp32 | - |
| `Not supported` | `digamma` | fp32 | - |
| `Not supported` | `div` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `div_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `double2float` | fp64 | - |
| `Not supported` | `double2ll` | fp64 | - |
| `Not supported` | `eq` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `eq_order` | bf16, fp16, fp32 | - |
| `Not supported` | `eq_out` | int64, uint64 | - |
| `Not supported` | `eq_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_digamma` | fp32 | - |
| `Not supported` | `fast_erf` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_float2byte` | fp32 | - |
| `Not supported` | `fast_float2short` | fp32 | - |
| `Not supported` | `fast_float2ubyte` | fp32 | - |
| `Not supported` | `fast_float2ushort` | fp32 | - |
| `Not supported` | `fast_gelu` | bf16, fp16, fp32 | 使用tanh近似公式计算 |
| `Not supported` | `fast_gelu_v2` | bf16, fp16, fp32 | 使用erf近似公式计算 |
| `Not supported` | `fast_lgamma` | fp32 | - |
| `Not supported` | `fast_log` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_max` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_min` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_nan_max` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_nan_min` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_powi` | fp32, int32 | - |
| `Not supported` | `fast_rcp` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_sigmoid` | fp32 | - |
| `Not supported` | `fast_silu` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_silubp` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_sqrt` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_tanh` | bf16, fp16, fp32 | - |
| `Not supported` | `fast_trigamma` | fp32 | - |
| `Not supported` | `float2bfloat16` | fp32 | - |
| `Not supported` | `float2byte` | fp32 | - |
| `Not supported` | `float2byte_rz` | fp32 | - |
| `Not supported` | `float2byte_sat` | fp32 | - |
| `Not supported` | `float2byte_sat_rn` | fp32 | - |
| `Not supported` | `float2double` | fp32 | - |
| `Not supported` | `float2float8e4nv` | fp32 | - |
| `Not supported` | `float2float8e4nv_sat` | fp32 | - |
| `Not supported` | `float2float8e5` | fp32 | - |
| `Not supported` | `float2float8e5_sat` | fp32 | - |
| `Not supported` | `float2half` | fp32 | - |
| `Not supported` | `float2half_rn` | fp32 | - |
| `Not supported` | `float2half_rz` | fp32 | - |
| `Not supported` | `float2int` | fp32 | - |
| `Not supported` | `float2ll` | fp32 | - |
| `Not supported` | `float2short` | fp32 | - |
| `Not supported` | `float2short_rz` | fp32 | - |
| `Not supported` | `float2short_sat` | fp32 | - |
| `Not supported` | `float2short_sat_rn` | fp32 | - |
| `Not supported` | `float2ubyte` | fp32 | - |
| `Not supported` | `float2ubyte_rz` | fp32 | - |
| `Not supported` | `float2ubyte_sat` | fp32 | - |
| `Not supported` | `float2uint` | fp32 | - |
| `Not supported` | `float2ull` | fp32 | - |
| `Not supported` | `float2ushort` | fp32 | - |
| `Not supported` | `float2ushort_rz` | fp32 | - |
| `Not supported` | `float2ushort_sat` | fp32 | - |
| `Not supported` | `float8e4nv2bfloat16` | fp8e4nv | - |
| `Not supported` | `float8e4nv2float` | fp8e4nv | - |
| `Not supported` | `float8e4nv2half` | fp8e4nv | - |
| `Not supported` | `float8e52bfloat16` | fp8e5 | - |
| `Not supported` | `float8e52float` | fp8e5 | - |
| `Not supported` | `float8e52half` | fp8e5 | - |
| `Not supported` | `ge` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `ge_order` | bf16, fp16, fp32 | - |
| `Not supported` | `ge_out` | int64, uint64 | - |
| `Not supported` | `ge_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `gt` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `gt_order` | bf16, fp16, fp32 | - |
| `Not supported` | `gt_out` | int64, uint64 | - |
| `Not supported` | `gt_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `half2bfloat16` | fp16 | - |
| `Not supported` | `half2byte` | fp16 | - |
| `Not supported` | `half2byte_rz` | fp16 | - |
| `Not supported` | `half2float` | fp16 | - |
| `Not supported` | `half2int` | fp16 | - |
| `Not supported` | `half2int_rz` | fp16 | - |
| `Not supported` | `half2ll` | fp16 | - |
| `Not supported` | `half2ll_rz` | fp16 | - |
| `Not supported` | `half2short` | fp16 | - |
| `Not supported` | `half2short_rz` | fp16 | - |
| `Not supported` | `half2ubyte` | fp16 | - |
| `Not supported` | `half2ubyte_rz` | fp16 | - |
| `Not supported` | `half2uint` | fp16 | - |
| `Not supported` | `half2uint_rz` | fp16 | - |
| `Not supported` | `half2ull` | fp16 | - |
| `Not supported` | `half2ull_rz` | fp16 | - |
| `Not supported` | `half2ushort` | fp16 | - |
| `Not supported` | `half2ushort_rz` | fp16 | - |
| `Not supported` | `half2float8e4nv` | fp16 | - |
| `Not supported` | `half2float8e4nv_sat` | fp16 | - |
| `Not supported` | `half2float8e5` | fp16 | - |
| `Not supported` | `half2float8e5_sat` | fp16 | - |
| `Not supported` | `int2bfloat16` | int32 | - |
| `Not supported` | `int2byte` | int32 | - |
| `Not supported` | `int2float` | int32 | - |
| `Not supported` | `int2half` | int32 | - |
| `Not supported` | `int2half_rn` | int32 | - |
| `Not supported` | `int2ll` | int32 | - |
| `Not supported` | `int2short` | int32 | - |
| `Not supported` | `int2ubyte` | int32 | - |
| `Not supported` | `int2uint` | int32 | - |
| `Not supported` | `int2ull` | int32 | - |
| `Not supported` | `int2ushort` | int32 | - |
| `Not supported` | `le` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `le_order` | bf16, fp16, fp32 | - |
| `Not supported` | `le_out` | int64, uint64 | - |
| `Not supported` | `le_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `ll2bfloat16` | int64 | - |
| `Not supported` | `ll2byte` | int64 | - |
| `Not supported` | `ll2double` | int64 | - |
| `Not supported` | `ll2float` | int64 | - |
| `Not supported` | `ll2half` | int64 | - |
| `Not supported` | `ll2half_rn` | int64 | - |
| `Not supported` | `ll2int` | int64 | - |
| `Not supported` | `ll2short` | int64 | - |
| `Not supported` | `ll2ubyte` | int64 | - |
| `Not supported` | `ll2uint` | int64 | - |
| `Not supported` | `ll2ull` | int64 | - |
| `Not supported` | `ll2ushort` | int64 | - |
| `Not supported` | `lrint` | fp32 | - |
| `Not supported` | `lround` | bf16, fp16, fp32 | - |
| `Not supported` | `lt` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `lt_order` | bf16, fp16, fp32 | - |
| `Not supported` | `lt_out` | int64, uint64 | - |
| `Not supported` | `lt_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `max` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `min` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `mod` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `mul` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `mul_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `nan_max` | bf16, fp16, fp32 | - |
| `Not supported` | `nan_min` | bf16, fp16, fp32 | - |
| `Not supported` | `nan_sign` | bf16, fp16, fp32 | - |
| `Not supported` | `philox` | uint32 | 精度对齐tl.philox，性能更高 |
| `Not supported` | `philox_v2` | uint32 | 精度对齐CNNL，性能高于philox |
| `Not supported` | `philox_v3` | uint32 | 精度对齐CNNL，相较于philox_v2; 加入subsequenceLimit参数; 小幅度放开innerRounds的限制 |
| `Not supported` | `ne` | bf16, fp16, fp32, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `ne_order` | bf16, fp16, fp32 | - |
| `Not supported` | `ne_out` | int64, uint64 | - |
| `Not supported` | `ne_unorder` | bf16, fp16, fp32 | - |
| `Not supported` | `negate` | bf16, fp16, fp32, int16, int32, int64, int8 | - |
| `Not supported` | `negate_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `rsqrt_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `scalbln` | bf16, fp16, fp32, int64 | - |
| `Not supported` | `shift_left` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `shift_right_arithmetic` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `shift_right_logical` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `short2bfloat16` | int16 | - |
| `Not supported` | `short2byte` | int16 | - |
| `Not supported` | `short2float` | int16 | - |
| `Not supported` | `short2half` | int16 | - |
| `Not supported` | `short2half_rn` | int16 | - |
| `Not supported` | `short2int` | int16 | - |
| `Not supported` | `short2ll` | int16 | - |
| `Not supported` | `short2ubyte` | int16 | - |
| `Not supported` | `short2uint` | int16 | - |
| `Not supported` | `short2ull` | int16 | - |
| `Not supported` | `short2ushort` | int16 | - |
| `Not supported` | `sign` | bf16, fp16, fp32, int16, int32, int64, int8 | - |
| `Not supported` | `sign_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `sin_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `sqrt_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `sub` | int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `sub_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `tan_complex` | bf16, fp16, fp32 | - |
| `Not supported` | `trigamma` | fp32 | - |
| `Not supported` | `ubyte2bfloat16` | uint8 | - |
| `Not supported` | `ubyte2byte` | uint8 | - |
| `Not supported` | `ubyte2float` | uint8 | - |
| `Not supported` | `ubyte2half` | uint8 | - |
| `Not supported` | `ubyte2int` | uint8 | - |
| `Not supported` | `ubyte2ll` | uint8 | - |
| `Not supported` | `ubyte2short` | uint8 | - |
| `Not supported` | `ubyte2uint` | uint8 | - |
| `Not supported` | `ubyte2ull` | uint8 | - |
| `Not supported` | `ubyte2ushort` | uint8 | - |
| `Not supported` | `uint2bfloat16` | uint32 | - |
| `Not supported` | `uint2byte` | uint32 | - |
| `Not supported` | `uint2float` | uint32 | - |
| `Not supported` | `uint2half` | uint32 | - |
| `Not supported` | `uint2half_rn` | uint32 | - |
| `Not supported` | `uint2int` | uint32 | - |
| `Not supported` | `uint2ll` | uint32 | - |
| `Not supported` | `uint2short` | uint32 | - |
| `Not supported` | `uint2ubyte` | uint32 | - |
| `Not supported` | `uint2ull` | uint32 | - |
| `Not supported` | `uint2ushort` | uint32 | - |
| `Not supported` | `ull2bfloat16` | uint64 | - |
| `Not supported` | `ull2byte` | uint64 | - |
| `Not supported` | `ull2float` | uint64 | - |
| `Not supported` | `ull2half` | uint64 | - |
| `Not supported` | `ull2half_rn` | uint64 | - |
| `Not supported` | `ull2int` | uint64 | - |
| `Not supported` | `ull2ll` | uint64 | - |
| `Not supported` | `ull2short` | uint64 | - |
| `Not supported` | `ull2ubyte` | uint64 | - |
| `Not supported` | `ull2uint` | uint64 | - |
| `Not supported` | `ull2ushort` | uint64 | - |
| `Not supported` | `ultra_gelu` | bf16, fp16, fp32 | 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_pow` | bf16, fp16, fp32 | 精度低于fast_pow，但是性能高于fast_pow; 精度性能对齐CNNL的pow算子 |
| `Not supported` | `ultra_sigmoid` | fp32 | 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silu` | bf16, fp16, fp32 | 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silu_mul_float2bfloat16` | fp32 | 输入fp32，输出为bfloat16; 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silu_mul_float2half` | fp32 | 输入fp32，输出为fp16; 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silubp` | bf16, fp16, fp32 | 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silubp_mul_float2bfloat16` | fp32 | 输入fp32，输出为bfloat16; 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_silubp_mul_float2half` | fp32 | 输入fp32，输出为fp16; 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ultra_gelu_float2bfloat16` | fp32 | - |
| `Not supported` | `ultra_gelu_float2half` | fp32 | - |
| `Not supported` | `ultra_silu_float2bfloat16` | fp32 | - |
| `Not supported` | `ultra_silu_float2half` | fp32 | - |
| `Not supported` | `ultra_silubp_float2bfloat16` | fp32 | - |
| `Not supported` | `ultra_silubp_float2half` | fp32 | - |
| `Not supported` | `ultra_tanh` | bf16, fp16, fp32 | 使用激活表实现，性能高但是精度低 |
| `Not supported` | `ushort2bfloat16` | uint16 | - |
| `Not supported` | `ushort2byte` | uint16 | - |
| `Not supported` | `ushort2float` | uint16 | - |
| `Not supported` | `ushort2half` | uint16 | - |
| `Not supported` | `ushort2half_rn` | uint16 | - |
| `Not supported` | `ushort2int` | uint16 | - |
| `Not supported` | `ushort2ll` | uint16 | - |
| `Not supported` | `ushort2short` | uint16 | - |
| `Not supported` | `ushort2ubyte` | uint16 | - |
| `Not supported` | `ushort2uint` | uint16 | - |
| `Not supported` | `ushort2ull` | uint16 | - |
| `Not supported` | `xor` | int1, int16, int32, int64, int8, uint16, uint32, uint64, uint8 | - |
| `Not supported` | `zeta` | fp32 | - |
| `erf` | `erf` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `erfc` | `erfc` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `erfcinv` | `erfcinv` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `erfcx` | `erfcx` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `erfinv` | `erfinv` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `isfinited` | `isfinited` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `isinf` | `isinf` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `isnan` | `isnan` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `lgamma` | `lgamma` | fp32 | MLU 不支持数据类型: fp64 |
| `mulhi` | `mulhi` | int32, int64, uint32, uint64 | - |
| `normcdf` | `normcdf` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `rcp_rd` | `rcp_rd` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `rcp_rn` | `rcp_rn` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `rcp_ru` | `rcp_ru` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `rcp_rz` | `rcp_rz` | fp16, fp32 | MLU 不支持数据类型: fp64 |
| `scalbn` | `scalbn` | bf16, fp16, fp32, int32 | MLU 不支持数据类型: fp64 |
| `signbit` | `signbit` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |
| `tgamma` | `tgamma` | bf16, fp16, fp32 | MLU 不支持数据类型: fp64 |

### ❌ 不支持的算子

| 函数名 |
|:---|
| `brev` |
| `byte_perm` |
| `cospi` |
| `cyl_bessel_i0` |
| `cyl_bessel_i1` |
| `double_as_longlong` |
| `finitef` |
| `float_as_int` |
| `float_as_uint` |
| `hadd` |
| `int_as_float` |
| `j0` |
| `j1` |
| `jn` |
| `longlong_as_double` |
| `mul24` |
| `norm3d` |
| `norm4d` |
| `normcdfinv` |
| `rcp64h` |
| `rhadd` |
| `rhypot` |
| `rnorm3d` |
| `rnorm4d` |
| `sad` |
| `saturatef` |
| `sinpi` |
| `uint_as_float` |
| `y0` |
| `y1` |
| `yn` |
---
## 🚫 绝对禁区：不支持算子与类型

如果 Triton 代码中出现以下情况，必须报错：

### 1. **fp64 (Double) 调用下列函数**

大多数数学函数不支持 `fp64`，包括但不限于：
- 三角函数：`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- 双曲函数：`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- 指数对数：`exp`, `exp2`, `exp10`, `expm1`, `log`, `log2`, `log10`, `log1p`, `logb`
- 幂根函数：`pow`, `sqrt`, `rsqrt`, `cbrt`
- 舍入函数：`ceil`, `floor`, `trunc`, `rint`, `round`, `nearbyint`
- 其他：`abs`, `erf`, `erfc`

### 2. **MLU 平台缺失的函数**

MLU 平台不支持以下函数：
- **π 系列**：`sinpi`, `cospi`
- **几何范数**：`hypot`, `norm3d`, `norm4d`
- **贝塞尔函数**：`j0`, `j1`, `jn`, `y0`, `y1`, `yn`, `cyl_bessel_i0`, `cyl_bessel_i1`
- **饱和运算**：`saturatef`, `rhadd`
- **其他**：`brev`, `byte_perm`

### 3. **不支持的舍入转换**

大部分 `double` 的舍入转换均不支持，如：
- `double2float_rd`, `double2float_ru`, `double2float_rz`
- `double2int_rd`, `double2int_rn`, `double2int_ru`, `double2int_rz`
- `double2ll_rd`, `double2ll_rn`, `double2ll_ru`, `double2ll_rz`
- `double2uint_rd`, `double2uint_rn`, `double2uint_ru`, `double2uint_rz`

---

## 📝 使用建议

1. **类型选择**：优先使用 `fp32`, `fp16`, `bf16`，避免 `fp64`
2. **函数替代**：
   - 需要 π 相关计算时，手动乘以 π 常数
   - 需要范数计算时，使用基础数学函数组合实现
3. **性能优化**：优先使用 MLU 特有的 `ultra_*` 和 `fast_*` 系列函数
4. **随机数**：推荐使用 `philox_v2` 或 `philox_v3`
