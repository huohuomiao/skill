# Triton原语支持现状
- MLU Triton支持当前官方Triton版本大部分原语。

## 具体支持情况如下所示：


### 1. 基础块初始化 (Block Initialization)
| 原语名称 | 支持 int64 | 备注 |
| :--- | :---: | :--- |
| `arange` | No | |
| `cat`, `full`, `zeros`, `zeros_like` | **Yes** | |

### 2. 形状变换 (Shape Manipulation)
此类算子对数据类型不敏感，全线支持 `int64`。
* **涵盖原语**：`broadcast`, `broadcast_to`, `expand_dims`, `interleave`, `join`, `permute`, `ravel`, `reshape`, `split`, `trans`, `view`
* **int64 支持**：**Yes**

### 3. 代数与数学运算 (Linear Algebra & Math)
| 原语名称 | 支持 int64 | 备注 |
| :--- | :---: | :--- |
| `dot` | No | 输入 `int8/int16` 时默认输出 `fp32`；支持指定输出为 `bf16` |
| `abs`, `cdiv`, `div` | **Yes** | |
| `clamp`, `cos`, `sin`, `exp`, `log` 等超越函数 | No | 超越函数建议高精度需求时使用 Libdevice |
| `sigmoid`, `softmax` | No | MLU特有/扩展支持 |
| `sqrt`, `rsqrt`, `floor`, `fma` | No | |

### 4. 访存与原子操作 (Memory Operations)
| 原语名称 | 支持 int64 | 备注 |
| :--- | :---: | :--- |
| `load`, `store` | **Yes** | 非原子访存 |
| `make_block_ptr`, `advance` | **Yes** | 块指针操作 |
| **`atomic_xxx` (所有原子操作)** | **No** | **扩展支持**：`fp16`, `bf16`, `int8`, `int16` 的原子操作 |
### 5. 逻辑与比较运算 (Arith & Logic Ops)
此类算子支持数值计算和位运算，全线支持 `int64`。
* **涵盖原语**：`add`, `sub`, `mul`, `div`, `rem`, `pow`, `minimum`, `maximum`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `and`, `or`, `xor`, `not`
* **int64 支持**：**Yes**

### 6. 索引与搜寻 (Indexing & Scans)
| 原语名称 | 支持 int64 | 备注 |
| :--- | :---: | :--- |
| `flip`, `where`, `swizzle2d`, `gather` | **Yes** | |
| `argmax`, `argmin`, `max`, `min`, `sum` | **Yes** | 归约操作 (Reductions)；最大/最小值索引场景优先参考下方推荐 |
| `cumsum`, `cumprod`, `sort`, `histogram` | **Yes** | 扫描操作 (Scans) |
| `masked_select` | **Yes** | |

### 7. 辅助与调试功能 (Rand & Debug)
| 原语名称 | 支持 int64 | 备注 |
| :--- | :---: | :--- |
| `rand`, `randn`, `randint`, `randint4x` | **Yes** | `rand` 传入 `int64` seed 时会截断高 16 位 |
| `static_print`, `static_assert`, `device_print` | **Yes** | |
| `device_assert` | **Yes** | MLU Triton 暂不支持该原语的布尔判定（Yes 仅代表类型通路） |
| `static_range` | No | 迭代器 |

---

## 关键类型转换与扩展特性说明

1.  **位比对转换 (Bitcast) 扩展**：
    社区标准仅支持相同位宽的转换（如 `fp16` 到 `int16`）。**MLU Triton 支持不同位宽的 bitcast 转换**，例如可将 `float16` 转换为 `int32`。
    ```python
    # MLU扩展示例
    out = in.to(tl.int32, bitcast=True) 
    3.  **编译器提示 (Compiler Hints)**：
    * 支持 `multiple_of`, `max_contiguous`, `max_constancy`（不支持 `int64`）。
    * **注意**：`debug_barrier` 原语在MLU架构中虽可调用，但语义**不生效**。

4.  **块内最大/最小值及索引归约推荐**：
    * 当需要同时获取块内最大值/最小值及其索引时，推荐优先使用 `tl.max(x, dim=xx, return_indices=True)` 或 `tl.min(x, dim=xx, return_indices=True)`。
    * 当只需要获取最大值/最小值对应的索引时，也推荐优先使用上述 `return_indices=True` 形式，再按需求使用返回的索引结果。
    * 该写法可避免手写比较、`where` 和索引维护逻辑，通常更利于编译器识别归约语义并生成更合适的实现。