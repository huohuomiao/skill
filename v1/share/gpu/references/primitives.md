# NVIDIA CUDA Triton 原语与 RTX 3090 门禁

本文件不是某个 Triton 版本的完整 API 快照。生成代码时先以安装版本的官方文档、函数签名和真实编译结果为准，再应用这里的 `sm_86` 硬件门禁。不得沿用 MLU 扩展语义，也不得因 Python 层存在符号就认定 3090 有对应硬件指令。

## 快速结论

- RTX 3090 是 CUDA `sm_86`，满足官方 Triton 当前的 NVIDIA CC >= 8.0 门槛。
- 通用 pointer/block-pointer、逐元素、归约、scan、随机数、原子和调试原语可作为候选，但 dtype、shape 和版本必须编译测试。
- `tl.dot` 可利用 Ampere Tensor Core 的 FP16、BF16、TF32、INT8/INT4 路径；FP8 Tensor Core 不支持。
- TMA、WGMMA、thread-block cluster、Blackwell warp specialization、`tl.dot_scaled` 的新架构 block-scaled 路径不属于 RTX 3090。
- `device_assert` 仅在 `TRITON_DEBUG` 非 0 时生效；`debug_barrier` 是 block 内同步，不能当作 grid barrier。

## 1. 编程模型与创建

| 类别 | 常用官方原语 | 3090 规则 |
| --- | --- | --- |
| program 身份 | `program_id`, `num_programs` | 可用；axis 必须与 launch grid 维度一致 |
| 创建 | `arange`, `full`, `zeros`, `zeros_like`, `to_tensor` | 可用；`tl.arange` 的区间/长度约束由安装版本验证 |
| dtype | `cast`, `.to`, `bitcast=True` | 数值 cast 与 bitcast 分开；bitcast 必须等位宽，禁止沿用跨位宽扩展 |
| 编译期 | `constexpr`, `static_range`, `static_assert`, `static_print` | 可用；大规模展开会放大代码和 register pressure |
| 运行期循环 | `range` | 可用；`num_stages`、unroll 先小范围调优 |

不要把普通 Python 值、0-D Triton tensor 和 block tensor 混为一谈。shape、stride、tile、循环上界中需要编译期常量的参数显式标注 `tl.constexpr`；动态 shape 只用于允许的标量控制流或 pointer arithmetic。

## 2. Shape 与 layout 操作

官方语言层包含：

`broadcast`, `broadcast_to`, `expand_dims`, `unsqueeze`, `squeeze`, `interleave`, `join`, `permute`, `trans`, `ravel`, `reshape`, `split`, `view`。

门禁：

1. 这些操作通常改变编译期 block shape/layout，不等价于 PyTorch 的任意动态 view。
2. `view`/bitcast 必须满足元素数、位宽和 layout 约束；不要使用 MLU 的跨位宽 bitcast 假设。
3. 过大的逻辑 block 会直接增加每 program 的 SSA 值、registers 或 shared-memory 使用。即使 shape 操作本身“零拷贝”，也必须查看编译资源。
4. `permute`/`trans` 后的访存是否合并取决于最终地址，不取决于 API 名称；用 NCU 验证 sectors/throughput。

## 3. 访存与 pointer

| 原语 | 3090 状态 | 必查条件 |
| --- | --- | --- |
| `load`, `store` | 支持 | mask、`other`、边界、alignment、cache/eviction hint |
| `make_block_ptr`, `advance` | 支持候选 | shape/stride/order 合法，block shape 与 mask/boundary_check 正确 |
| `gather` | 版本相关候选 | 安装版本签名、索引范围、性能 |
| tensor descriptor load/store | 不作为 3090 路径 | NVIDIA TMA 硬件是 Hopper+；改用 pointer/block pointer |
| inline assembly | 高风险候选 | CUDA-only、约束/pack 正确、`sm_86` 可编译、提供等价测试 |

`tl.load/store` 的 mask 必须和 pointer block 可广播。masked load 的 `other` 应与后续语义匹配：sum 常用 0，max 常用 `-inf`，min 常用 `+inf`。越界地址即使结果随后被 `where` 丢弃也不安全，应在访存处 mask。

内存合并以 warp 内地址连续性为核心。优先让 `tl.arange` 的连续维映射到输入/输出的 stride-1 维；只有确有复用收益时才为转置、归约或 matmul 接受非连续访问。

## 4. 数学与逐元素原语

官方常用集合包括：

- 算术：`add`, `sub`, `mul`, `cdiv`, `minimum`, `maximum`, `clamp`, `abs`, `fma`。
- 除法/根号：`fdiv`, `div_rn`, `sqrt`, `sqrt_rn`, `rsqrt`。
- 指数/对数：`exp`, `exp2`, `log`, `log2`。
- 三角/特殊：`sin`, `cos`, `erf`, `sigmoid`, `softmax`。
- 位与整数：Python 运算符、`umulhi` 等安装版本暴露的原语。

精度门禁：

- `tl.exp`、`tl.sqrt` 等文档明确区分 fast/precise 的原语时，不能把二者视为逐位等价。严格需求优先 `*_rn` 或 CUDA libdevice，并建立 ULP/rtol/atol 测试。
- FP16/BF16 输入通常应在 FP32 中完成归约或非线性中间计算，再按接口转换；这不是无条件规则，最终以契约和性能为准。
- FP64 可用但 RTX 3090 的非 Tensor FP64 吞吐很弱；禁止为了“更安全”默默提升 dtype。
- NaN 传播、signed zero、无穷与 domain error（如 `log(x<0)`）是语义的一部分。替换 `where/minimum/maximum` 或 libdevice 前需专门测试。
- 更多函数使用 `from triton.language.extra import libdevice`，不得引用 `tl.extra.mlu` 或虚构的激活函数。

## 5. 归约、scan、排序和索引

| 类别 | 常用原语 | 检查项 |
| --- | --- | --- |
| 归约 | `sum`, `max`, `min`, `argmax`, `argmin`, `reduce_or`, `xor_sum` | axis、keep_dims/返回类型、NaN/tie 行为、累加 dtype |
| scan | `cumsum`, `cumprod`, `associative_scan` | axis、combine_fn 结合律、block size |
| 排序 | `sort`, `topk`, `bitonic_merge` | 安装版本限制、最后一维长度、稳定性/相等元素 |
| 索引 | `where`, `flip`, `swizzle2d`, `gather` | 两分支可能都求值；不要用 `where` 代替访存 mask |
| 直方图 | `histogram` | `num_bins` 和输入范围必须满足 API 约束 |

归约块通常要求 2 的幂 padding。padding 值必须是归约幺元；例如 max 的尾部不能补 0，否则全负输入会错。对 softmax 使用 `x - max(x)` 保证稳定性，并明确空行、全 `-inf`、NaN 的契约。

## 6. Matrix multiply / Tensor Core

### RTX 3090 可用路径

`tl.dot(a, b, acc, input_precision=..., out_dtype=...)` 是矩阵乘核心原语。`sm_86` 的 Tensor Core 支持 FP16、BF16、TF32、INT8 和 INT4 输入类别，但 Triton 接受的 dtype/shape 仍由当前函数签名和 lowering 决定。

- FP16/BF16：通常 FP32 accumulate；验证输出转换和容差。
- FP32 输入：`input_precision` 可选 `"tf32"`, `"tf32x3"`, `"ieee"`, `"bf16x3"`, `"bf16x6"`（以安装版本文档为准）。当前 NVIDIA 默认是 `"tf32"`；需要 IEEE 路径时显式指定。
- INT8：通常 INT32 accumulate；不要假设任意 int shape 都会走 Tensor Core。
- tile 的 K 对齐、layout、`num_warps` 和 `num_stages` 决定是否得到期望的 MMA lowering；以 TTGIR/PTX/NCU 为证据。

### RTX 3090 禁用路径

- float8 输入可能在 API 类型集合中出现，但 CC 8.6 的 Tensor Core 不支持 FP8。不得把编译通过等同于硬件加速。
- `tl.dot_scaled` 的 FP4/FP8 microscaling 和 Triton block-scaled 教程针对更新硬件，不生成 3090 配置。
- TMA tensor descriptors、WGMMA、warp-group、cluster launch control 和 Blackwell tcgen05 代码必须通过 CC gate 排除。
- `warp_specialize=True` 当前是 Blackwell 限定，不用于 Ampere。

## 7. 原子操作

官方 Triton API 包含 `atomic_add`, `atomic_and`, `atomic_cas`, `atomic_max`, `atomic_min`, `atomic_or`, `atomic_poll`, `atomic_xchg`, `atomic_xor`。API 存在不代表任意 dtype、address space、memory semantic 都支持。

RTX 3090 的 CUDA 硬件边界：

- `atomicAdd` 的 FP32、FP64、FP16 和 BF16 硬件门槛均已满足（FP16 需 CC 7.x+，BF16 需 CC 8.x+，FP64 需 CC 6.x+）。
- `float2`/`float4` vector `atomicAdd` 要求 CC 9.x+，3090 禁用。
- CAS、整数位原子与 min/max 仍需按位宽、signedness 和 global/shared 地址空间验证 Triton lowering。
- 原子顺序、scope 和 semantic 必须显式匹配算法；默认值不能代替跨 block 的全局同步。

原子会序列化冲突地址。正确性需要原子时先保证正确，再通过分层归约、分桶或减少热点优化；不要把非原子 load-modify-store 当作性能替换。

## 8. 随机数与调试

- 随机数：`randint`, `randint4x`, `rand`, `rand4x`, `randn`, `randn4x`, `philox` 等以当前 API 为准。seed/offset 决定可复现性；不同 Triton 版本或程序分块不承诺与 PyTorch 相同序列。
- `static_assert` 在编译期生效；用于 tile、stride、constexpr 条件。
- `device_assert(cond, msg, mask)` 需要环境变量 `TRITON_DEBUG` 非 0；cond 必须是 boolean tensor，msg 是字符串字面量。
- `device_print` 会显著扰动执行与时序，只用于小输入调试。
- `debug_barrier()` 只同步一个 CUDA block/Triton program 内的 threads，绝不是跨 program/grid barrier。
- Triton interpreter 可帮助查逻辑错误，但不能证明 CUDA lowering、race、资源占用或性能。

## 9. Compiler hints

`multiple_of`, `max_contiguous`, `max_constancy`, `assume` 是给编译器的事实，不是运行时检查。错误 hint 可能产生错误代码；只有能从 shape/stride/alignment 契约证明时使用，并添加违反边界附近的测试。
不要为了优化声称任意用户 pointer 对齐；allocator 的基础地址对齐不能证明切片后的 pointer 保持同一对齐。

## 10. 生成前检查表

1. active backend 是否 `cuda`，设备 CC 是否 `(8, 6)`，型号是否严格 RTX 3090。
2. 每个 primitive 是否存在于安装版本，而不是来自旧文档或其他后端。
3. dtype/shape/layout 是否满足该 primitive 的函数签名和 `sm_86` lowering。
4. 是否误用了 FP8、TMA、WGMMA、clusters、Blackwell warp specialization。
5. 普通 grid 是否保持完整逻辑 tile；persistent grid 是否有 grid-stride loop。
6. mask、padding 幺元、NaN/Inf、signed zero、integer overflow、atomic race 是否覆盖测试。
7. correctness 通过后再看性能；不以“编译成功”代替数值与 profiler 证据。

## 官方来源

- Triton Language API：<https://triton-lang.org/main/python-api/triton.language.html>
- `tl.dot`：<https://triton-lang.org/main/python-api/generated/triton.language.dot.html>
- `tl.range`：<https://triton-lang.org/main/python-api/generated/triton.language.range.html>
- `device_assert`：<https://triton-lang.org/main/python-api/generated/triton.language.device_assert.html>
- `debug_barrier`：<https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html>
- Triton block-scaled matmul：<https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html>
- CUDA CC 与 Tensor Core dtype 表：<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html>
- CUDA atomic functions：<https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomic-functions>
- NVIDIA TMA（Hopper+）说明：<https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html>
