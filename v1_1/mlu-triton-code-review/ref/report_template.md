# Triton 算子静态代码检视报告

## 基本信息
- **算子名称**：[名称]
- **文件路径**：[路径]

---
## 一、 Triton 原语检测

| 原语名称       | 使用位置/行号 | 数据类型 | 是否支持 int64 | 是否符合规范 | 备注/建议修改                                           |
| -------------- | ------------- | -------- | -------------- | ------------ | ------------------------------------------------------- |
| `arange`       | kernel.py:45  | int64    | No             | ❌            | 不支持 int64，建议改用 `tl.arange(..., dtype=tl.int32)` |
| `zeros_like`   | kernel.py:50  | int64    | Yes            | ✅            | -                                                       |
| `cdiv`         | kernel.py:60  | int64    | Yes            | ✅            | -                                                       |
| `dot`          | kernel.py:72  | int8     | No             | ⚠️            | 输出默认 fp32，可根据需求指定 bf16                      |
| `load`         | kernel.py:80  | int64    | Yes            | ✅            | 注意边界 mask 检查                                      |
| `store`        | kernel.py:85  | int64    | Yes            | ✅            | 必须加 mask 防止越界                                    |
| `atomic_add`   | kernel.py:95  | int32    | Yes            | ⚠️            | int64 不支持；低位宽可用 fp16/bf16/int8/int16 扩展      |
| `broadcast_to` | kernel.py:102 | int64    | Yes            | ✅            | -                                                       |
| `sigmoid`      | kernel.py:110 | bf16     | No             | ⚠️            | 平台扩展，需读取目标平台原语清单                        |
| `randint`      | kernel.py:120 | int64    | Yes            | ✅            | seed 高 16 位会截断                                     |

## 二、 Triton 常见错误检测


| 错误类型 | 错误描述 | 导致后果 | 修正原则 |
| :--- | :--- | :--- | :--- |
| **1. 参数重定义** | 在 `configs` 定义了参数又在 Kernel 内手动赋值 | 编译失败或 Autotune 失效 | 内部仅声明 `tl.constexpr`，不赋值 |
| **2. 接口不一致** | Launch 传参个数/顺序与 Kernel 定义不符 | `TypeError` 或内存访问错乱 | 严格核对，建议关键字传参 |
| **3. 平台残留** | 代码中保留源平台关键字或环境 | 目标后端无法识别设备 | 按目标平台共享规则替换 |
| **4. 外部算 Block** | 在 Launch 参数位传入 `cdiv` 计算结果 | 逻辑混乱，违背并行架构设计 | 内部使用 `tl.program_id` 自行分块 |
| **5. 缺少 Mask** | `load/store` 不带边界判定 | **内存越界 (Out of Bounds)** | 始终计算 `mask = offsets < size` |
| **6. 基址缺失** | 计算偏移忘记加 `pid * BLOCK_SIZE` | 所有计算块重复处理第一块数据 | `offsets = base + pid * BLOCK + arange` |

---

## 三、平台扩展数学库检测

目标平台没有扩展数学库时标记为 `N/A`。MLU 目标按照 `.claude/skills/share/mlu/references/libdevice.md` 检查 Libdevice。

| 检查项 | 状态 | 详情/错误位置 | 判定标准 |
| :--- | :--- | :--- | :--- |
| **数据类型兼容性** | ✅/❌ | | **禁区检查**：`abs`, `sin`, `exp`, `log`, `sqrt` 等算子是否违规使用了 **fp64**。 |
| **舍入模式变体** | ✅/❌ | | 算术运算（`add`, `sub`, `mul`, `div`）是否显式指定了变体（如 `add_rn`, `sub_rz`）。 |
| **量化饱和处理** | ✅/❌ | | 在 `float2byte` 或 `float2int` 转换处，是否按量化需求补全了 `_sat` 后缀。 |
| **快速/极致算子使用** | ✅/❌ | | 是否在推理场景中错误调用了 `Standard` 算子而非 `Ultra/Fast` 算子。 |
| **复数域语法** | ✅/❌ | | `tl.math.add_complex` 等复数算子名及参数类型是否匹配。 |
| **随机数版本** | ✅/❌ | | 是否根据需求选择了正确的版本（`philox_v2` 对齐精度，`v3` 追求性能）。 |

---
