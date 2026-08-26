# BANG C 原语与生成约束

本文件是 MLU590 BANG C 第二版的生成门禁，不是某一版 `bang.h` 的完整 API 清单。实际函数签名、dtype、长度和对齐限制必须以目标服务器的头文件、官方 sample 和 CNCC 编译结果为准。2026-08-15 MLU590-M9DG 审计证据只作为已验证基线；当前动态证据优先。

## 状态定义

| 状态 | 含义 | 生成规则 |
|---|---|---|
| 基础必需 | BANG C/CNRT 主链路依赖的稳定概念 | 可生成，但仍需真实编译 |
| 审计基线确认 | 已在 MLU590-M9DG + CNCC 5.6.2 上验证 | 可作为候选；版本/型号不匹配时降级为条件使用 |
| 条件使用 | 名称常见但签名/约束可能随 SDK 或 dtype 变化 | 只有源码、头文件或服务器审计确认后生成 |
| 禁止迁移残留 | 属于 Triton/CUDA 或无法映射 | 不得出现在最终 `.mlu` |

## Kernel 与地址空间

| 语法/原语 | 状态 | 用途与约束 |
|---|---|---|
| `__mlu_global__` | 基础必需 | BANG C kernel 入口；host 侧通过 BANG launch 语法调用 |
| `__mlu_entry__` | 条件使用 | 某些源码/工具链的入口形式；先由当前工程或头文件确认 |
| `__nram__` | 基础必需 | 核心私有片上 tile；统计所有同时存活 buffer 的总预算 |
| `__wram__` | 条件使用 | 特定计算/权重片上空间；不能无依据替代 NRAM |
| `__sram__` | 条件使用 | cluster 共享空间；必须明确同步与所有权 |

已审计容量见 `{BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md`。不得把该基线容量、保留空间或对齐值无条件写入新算子。

## 任务索引内建变量

| 名称 | 状态 | 规则 |
|---|---|---|
| `taskId`, `taskDim` | 基础必需 | 线性任务编号/总任务数；适合 task-stride 覆盖 |
| `taskIdX/Y/Z`, `taskDimX/Y/Z` | 条件使用 | 多维任务映射；使用前确认当前编译器拼写 |
| `clusterId`, `coreId`, `coreDim` | 条件使用 | 低层 cluster/core 映射；不可根据营销规格硬编码范围 |

生成代码必须证明：任务间写区间不重叠、所有逻辑 tile 被覆盖、尾块无越界。

## 数据搬运

| 原语 | 状态 | 规则 |
|---|---|---|
| `__memcpy(dst, src, bytes, GDRAM2NRAM)` | 基础必需 | host/device 全局输入搬到 NRAM；第四参数与地址空间一致 |
| `__memcpy(dst, src, bytes, NRAM2GDRAM)` | 基础必需 | NRAM 结果写回全局内存 |
| 其他方向（如 NRAM2NRAM、GDRAM2SRAM） | 条件使用 | 只在当前 `bang.h` 与算法需求确认后使用 |
| `__memcpy_async` | 条件使用 | 需要对应同步/流水证据；不可仅把同步调用改名 |

`bytes` 是字节数，不是元素数。统一使用 `count * sizeof(dtype)`，并处理尾块和 API 对齐要求。

## 向量计算

以下名称已出现在 BANG C 审计或官方 sample 证据中，但第二版不宣称支持所有 dtype/长度：

| 原语族 | 状态 | 建议用途 |
|---|---|---|
| `__bang_add` | 审计基线确认 | float 原型已确认为 `__bang_add(float *dst, const float *src0, const float *src1, uint32_t elem_count)`；其他 dtype/版本仍需探测 |
| `__bang_sub` | 条件使用 | NRAM 向量逐元素减法 |
| `__bang_add_scalar` | 条件使用 | 向量与标量相加 |
| `__bang_floor` | 条件使用 | 向量 floor |
| `__bang_sumpool` | 条件使用 | 特定布局/窗口的求和规约 |
| `__bang_write_value` | 条件使用 | 片上 buffer 填充值 |
| 其他 `__bang_*` | 条件使用 | 必须先检索当前 `bang.h`/samples 并做最小编译探测 |

生成 intrinsic 前逐项确认：

1. 参数顺序和原地/非原地语义。
2. 输入、输出和累加 dtype。
3. 长度单位与最小/对齐粒度。
4. 源、目的地址空间。
5. 输入输出 alias 是否允许。
6. 尾块处理是否会读写 padding 之外的数据。

无法确认时，优先生成语义清晰的 BANG C 循环作为正确性基线，并在 code-review/optimize 阶段以已验证 intrinsic 替换；不要编造函数名或签名。

## Reduction 与数学函数

- reduction 必须定义 axis、初值、空输入和累加 dtype。
- `__bang_sumpool` 等专用原语不是通用 `sum(axis=...)` 的无条件替代；布局与窗口不匹配时不得使用。
- 标量数学可使用当前 CNCC 支持的 C/C++ 数学函数；设备侧可用性仍需最小编译验证。
- fast/approximate 变体读取 `{BANGC_SKILL_ROOT}/share/mlu/references/libdevice.md` 和 `{BANGC_SKILL_ROOT}/share/mlu/optimize/libdevice-opt.md`，且必须通过用户阈值。
- 不把一个 dtype 编译成功外推为所有 dtype 均支持。

## dtype 规则

| 需求 dtype | 保守源码表示 | 备注 |
|---|---|---|
| float32 | `float` | 第二版 smoke 默认 dtype |
| float16 | `half` 或当前 `bang.h` 定义 | 必须由目标头文件确认 spelling 与 intrinsic 支持 |
| int8/uint8 | `int8_t`/`uint8_t` | 包含正确标准头并检查提升语义 |
| int16/uint16 | `int16_t`/`uint16_t` | 同上 |
| int32/uint32 | `int32_t`/`uint32_t` | 索引与数据 dtype 分开 |
| int64/uint64 | `int64_t`/`uint64_t` | 不假设所有向量 intrinsic 支持 64 位 |
| bfloat16、fp64、其他低位格式 | 待服务器确认 | 未确认前不得声称支持 |

混合类型计算必须显式说明提升、累加和输出转换，不能沿用 PyTorch 或 Triton 的隐式类型规则。

## Host 侧 CNRT 原语

以下属于 host runtime，不可在 device kernel 中调用。2026-08-15 基线的确切 Queue 拼写是 `cnrtQueueCreate`/`cnrtQueueSync`/`cnrtQueueDestroy`，不是后缀 `Queue` 的旧变体。

- `cnrtSetDevice`
- `cnrtQueueCreate` / `cnrtQueueSync` / `cnrtQueueDestroy`
- `cnrtMalloc` / `cnrtFree`
- `cnrtMemcpy` 与 H2D/D2H 方向枚举
- `cnrtDim3_t` / `cnrtFunctionType_t`
- `cnrtNotifierCreate` / `cnrtNotifierDestroy` / `cnrtPlaceNotifier` / `cnrtNotifierDuration` / `cnrtNotifierElapsedTime`（基线已验证，当前 SDK 仍需确认）

每个返回码都要检查；不要重新定义 SDK 可能已有的 `CNRT_CHECK`。

## 禁止迁移残留

最终 BANG C `.mlu` 不得包含：

- `import triton`、`triton.language`、`@triton.jit`
- `tl.load`、`tl.store`、`tl.arange`、`tl.program_id`、`tl.num_programs`、`tl.constexpr`
- Triton launch `kernel[grid](...)`、`num_warps`、`num_stages`、autotune decorator
- CUDA `__global__`（缺少 `mlu`）、`threadIdx`、`blockIdx`、`cudaMalloc`、`cudaMemcpy`、`nvcc`
- 用 PyTorch/CPU reference 直接返回结果而不启动 BANG C kernel

文档中的“原始 Triton/CUDA 输入”引用可以保留为迁移来源，但生成代码和实际命令必须是 BANG C/CNRT。

## 使用流程

1. 从 requirement 确定计算、布局和 dtype。
2. 从本文件选择基础语法；条件原语先查目标头文件/样例。
3. 生成最小正确实现，显式搬运和处理边界。
4. 用 CNCC 编译；保存命令与完整错误。
5. 在 MLU590 上与独立 reference 比较。
6. 只有编译与精度通过后，才引入更强 intrinsic、异步搬运或流水。

遇到未列出的原语时不得直接判定支持或不支持；将其标记为 `needs_current_environment_verification` 并用当前环境的最小 probe 校准。
