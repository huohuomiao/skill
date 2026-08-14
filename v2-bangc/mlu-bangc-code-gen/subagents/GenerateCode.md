# GenerateCode

## 任务

严格根据 `step4_code_spec.json` 生成 `{output_dir}/KernelGen/step5_kernel_code.mlu`。文件必须是单一、完整的原生 BANG C/C++ translation unit，包含 device kernel、必要 helper、CNRT launcher 和测试插入标记；下一阶段在同一文件中补齐 CPU reference、资源管理、测试和 benchmark。

## 输入

- `step4_code_spec.json`
- `step1_base_info.json`
- `step1_io_shapes.json`
- `step2_block_mapping.json`
- `step3_axis_fusion.json`
- `requirement.md`
- `.claude/skills/share/mlu/references/platform-rules.md`
- `.claude/skills/share/mlu/references/primitives.md`

若 spec 的 `ready_for_code` 不为 true 或 `unresolved` 非空，停止并返回 GenerateSpec，不生成猜测代码。

## 输出合同

`step5_kernel_code.mlu` 必须：

- 使用 UTF-8。
- 是 `cncc` 可解析的 BANG C/C++ 源码。
- 不含 Markdown fence、TODO、伪代码、省略号或占位表达式。
- 不引用 requirement 未提供的项目私有 header。
- 不包含 Python、JIT、动态代码生成或其它语言 wrapper。
- 不猜架构 flag、核心数或 NRAM/WRAM/SRAM 容量。
- 保持 kernel/launcher ABI 与 spec 完全一致。
- 只生成 kernel 与 launcher；测试由 GenTestCode 添加。

## 文件布局

按以下顺序组织：

```cpp
// 1. bang.h / cnrt.h / actual standard includes
// 2. Small host/device helpers and compile-time constants
// 3. Device helpers
// 4. __mlu_global__ kernels
// 5. Public CNRT launchers
// 6. Exact marker for GenTestCode
```

文件末尾必须有：

```cpp
// === BANGC_TEST_HARNESS_BEGIN ===
```

GenTestCode 在该标记后追加测试，不复制第二套 kernel/launcher。

## Headers 与错误辅助

只包含实际使用的头：

```cpp
#include <bang.h>
#include <cnrt.h>

#include <stdint.h>
#include <stddef.h>
```

host 代码使用 `std::vector`、math、随机数或打印时，GenTestCode 再加入相应标准头。

强制规则：

- 不要检查 `${NEUWARE_HOME}/include/bang.h`；目标 SDK 可能把它放在编译器资源目录，配置正确的 `cncc` 会解析。
- 不要 `#define CNRT_CHECK`。`cnrt.h` 已提供时直接使用，避免宏重定义。
- 不要使用 `CNRT_RET_SUCCESS`。若确需比较状态，只能使用目标 `cnrt.h` 已确认的符号；通常让 `CNRT_CHECK` 处理 host/test API。
- 不在 device code 中调用 host-only 标准库函数。

## Kernel 生成

### 入口与签名

默认按 spec 生成：

```cpp
__mlu_global__ void op_kernel(const float* x, float* y, int64_t n) {
  // exact implementation
}
```

- 只有上游/目标工具链已确认 `__mlu_entry__` 时才使用或保留该 qualifier；不能同时生成两个含义不明的入口。
- 参数顺序、const、dtype 与 aliasing 完全按 spec。
- `__restrict__` 仅在需求明确保证不重叠时添加。
- runtime shape 不能伪装为编译期常量。

### taskId/taskDim 索引

先扩宽 builtin 再参与乘法：

```cpp
const int64_t total_tiles = (n + kTileElems - 1) / kTileElems;
for (int64_t tile = static_cast<int64_t>(taskId);
     tile < total_tiles;
     tile += static_cast<int64_t>(taskDim)) {
  const int64_t begin = tile * static_cast<int64_t>(kTileElems);
  const int64_t remaining = n - begin;
  const int valid = remaining < kTileElems
      ? static_cast<int>(remaining)
      : kTileElems;
  // copy, compute, store
}
```

- launcher 保证 `taskDim>0`。
- 所有 shape、stride、tile 乘积使用上游规定的足够宽整数。
- `taskIdX/Y/Z`、`clusterId`、`coreId`、`coreDim` 只照写 spec 中有证据的定义。

### 片上 buffer

按 spec 精确声明：

```cpp
__nram__ float x_nram[kTileElems];
__nram__ float y_nram[kTileElems];
```

- 数组 extent 必须是编译期常量。
- 每个声明必须能回溯到 `resource_summary` 的 byte 公式。
- 不把 runtime VLA 放到 NRAM。
- `__wram__`/`__sram__` 只在 spec 启用并明确布局、容量与同步时生成。
- 不用数组越界、union cast 或截断 tile 来绕过资源错误。

### GDRAM 与片上搬运

严格照写 spec 的方向与 bytes：

```cpp
__memcpy(x_nram, x + begin,
         static_cast<size_t>(valid) * sizeof(float), GDRAM2NRAM);
// compute
__memcpy(y + begin, y_nram,
         static_cast<size_t>(valid) * sizeof(float), NRAM2GDRAM);
```

禁止：

- 把 `valid` 元素数直接作为 byte count。
- tail 时仍搬 `kTileElems*sizeof(T)` 导致 GDRAM 越界。
- 把 non-contiguous 多行当成单连续区域。
- 交换 GDRAM2NRAM/NRAM2GDRAM。
- 使用未经 spec 证明的异步 copy 和同步。

若 full-width intrinsic 需要 padding，先只搬有效 bytes，再在 NRAM 初始化 `[valid,kTileElems)`；写回仍只写有效 bytes。

### `__bang_*` 计算

每个 intrinsic 必须与 spec 的 exact call 一致。例如已确认的向量加法：

```cpp
__bang_add(out_nram, x_nram, y_nram, kTileElems);
```

同时实现 spec 指定的 tail 初始化或 scalar fallback。不得自行猜测：

- intrinsic 名称或重载。
- 参数顺序。
- 长度单位。
- 支持 dtype。
- alignment、最小长度或数值误差。

证据不足时直接生成标量片上循环：

```cpp
for (int i = 0; i < valid; ++i) {
  out_nram[i] = x_nram[i] + y_nram[i];
}
```

这是一条 correctness baseline，不得在报告中伪称 intrinsic 路径。

## Reduction

严格按 `reduction.strategy` 生成。

### NRAM accumulator baseline

```cpp
for (int i = 0; i < output_valid; ++i) {
  acc_nram[i] = identity;
}
for (int64_t r0 = 0; r0 < reduce_extent; r0 += kReduceTile) {
  const int valid_r = min(kReduceTile, reduce_extent - r0);
  // 按真实 stride 把当前 chunk 搬入 NRAM。
  // 使用已确认 intrinsic 或只遍历 valid_r 的标量循环。
  // 更新 acc_nram，不能读取未初始化 tail。
}
// 当前 task 唯一写回输出 tile。
```

### 多遍算法

softmax/normalization 等按 spec 的 `pass1/pass2/pass3` 顺序生成。需要重读输入、workspace 或第二 kernel 时必须显式实现；不得假设跨 task barrier。多 kernel 依赖同一 queue 的顺序，业务 launcher 不无条件 host sync。

## CNRT Launcher

业务 launcher 接受 device pointer 与 `cnrtQueue_t`，不隐藏 allocation/copy：

```cpp
inline bool launch_vector_add(const float* x,
                              const float* y,
                              float* out,
                              int64_t n,
                              cnrtQueue_t queue) {
  if (n < 0) return false;
  if (n == 0) return true;
  if (x == NULL || y == NULL || out == NULL || queue == NULL) return false;

  // 没有已验证并行度事实时，1 task + task-grid-stride 是正确性 fallback。
  const uint32_t task_count = 1;
  cnrtDim3_t dim = {task_count, 1, 1};
  cnrtFunctionType_t function_type = cnrtFuncTypeBlock;
  vector_add_kernel<<<dim, function_type, queue>>>(x, y, out, n);
  return true;
}
```

真实代码中的 task_count、function type 与 launch ABI必须来自 spec。EnvConfig 有已验证 task 数时按其公式使用；没有时不能按 MLU590 名称填核心数。

launcher 默认异步：

- 不调用 `cnrtQueueSync`。
- 不创建/销毁 queue。
- 不分配/释放 device memory。
- 多 kernel 全部提交到传入 queue。
- 异步执行错误由测试/调用者的 `cnrtQueueSync` 检查。

若 requirement 明确要求 host convenience wrapper，可另加一层，但 benchmark 必须仍能调用不含 allocation/H2D/D2H 的 launcher。

## 数值语义

按 spec 选择运算；不要为“看起来更快”自行替换：

- 除法改倒数乘。
- 标准运算改近似 intrinsic。
- float reduction 改低精度累加。
- 独立乘加改融合乘加。
- 高精度类型改低精度类型。

任何候选近似属于 optimize，除非 requirement 明确许可且 spec 已固定。

## 完整结构示例

以下展示代码组织；tile 和 intrinsic 仅在对应 spec 已确认时使用：

```cpp
#include <bang.h>
#include <cnrt.h>

#include <stddef.h>
#include <stdint.h>

namespace {
constexpr int kTileElems = 256;
}

__mlu_global__ void vector_add_kernel(const float* x,
                                      const float* y,
                                      float* out,
                                      int64_t n) {
  __nram__ float x_nram[kTileElems];
  __nram__ float y_nram[kTileElems];
  __nram__ float out_nram[kTileElems];

  const int64_t tiles = (n + kTileElems - 1) / kTileElems;
  for (int64_t tile = static_cast<int64_t>(taskId);
       tile < tiles;
       tile += static_cast<int64_t>(taskDim)) {
    const int64_t begin = tile * kTileElems;
    const int64_t remain = n - begin;
    const int valid = remain < kTileElems ? static_cast<int>(remain)
                                          : kTileElems;
    const size_t bytes = static_cast<size_t>(valid) * sizeof(float);
    __memcpy(x_nram, x + begin, bytes, GDRAM2NRAM);
    __memcpy(y_nram, y + begin, bytes, GDRAM2NRAM);

    for (int i = valid; i < kTileElems; ++i) {
      x_nram[i] = 0.0f;
      y_nram[i] = 0.0f;
    }
    __bang_add(out_nram, x_nram, y_nram, kTileElems);
    __memcpy(out + begin, out_nram, bytes, NRAM2GDRAM);
  }
}

inline bool launch_vector_add(const float* x,
                              const float* y,
                              float* out,
                              int64_t n,
                              cnrtQueue_t queue) {
  if (n < 0) return false;
  if (n == 0) return true;
  if (x == NULL || y == NULL || out == NULL || queue == NULL) return false;
  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t type = cnrtFuncTypeBlock;
  vector_add_kernel<<<dim, type, queue>>>(x, y, out, n);
  return true;
}

// === BANGC_TEST_HARNESS_BEGIN ===
```

示例的 `kTileElems`、`__bang_add` 与 dim 不能机械复制到无关算子；真实代码必须满足 spec 和目标 `bang.h`。

## 参考示例路由

仅按当前 pattern 读取：

- [generate_code_axis_fusion.md](./examples/generate_code_axis_fusion.md)
- [generate_code_matrix_transpose.md](./examples/generate_code_matrix_transpose.md)
- [generate_code_reduce_sum.md](./examples/generate_code_reduce_sum.md)
- [generate_code_transpose_elementwise.md](./examples/generate_code_transpose_elementwise.md)

示例必须整体理解，不能只复制 kernel 而遗漏 CNRT launcher、tail、资源公式或测试标记。

## 静态自检

### 结构

- [ ] include 完整且无私有缺失依赖。
- [ ] kernel/launcher 声明、定义、调用签名一致。
- [ ] 每个 kernel 有 host launch 路径。
- [ ] 文件末尾有唯一 harness 标记。

### BANG C

- [ ] kernel 使用 `__mlu_global__` 或 spec 已确认的 `__mlu_entry__`。
- [ ] task index 在乘法前扩为 64 位。
- [ ] `taskDim>0` 且覆盖循环与 spec 一致。
- [ ] 片上数组 extent 是编译期常量，bytes 与 spec 一致。
- [ ] 每次 `__memcpy` 的方向和 byte count 正确。
- [ ] tail 不越界、不读取未初始化 NRAM。
- [ ] `__bang_*` 名称/签名/dtype/长度有证据。
- [ ] 无未确认跨 task/cluster 同步。
- [ ] launcher 不隐藏 allocation、copy 或同步。

### CNRT/P0

- [ ] 未重定义 `CNRT_CHECK`。
- [ ] 未出现 `CNRT_RET_SUCCESS`。
- [ ] 未硬编码 arch、核心数、NRAM/WRAM/SRAM 容量。
- [ ] build 侧将链接 `cnrt/stdc++/m/pthread`。

### 文本

禁止出现 `TODO`、`TBD`、伪代码标记或省略号。注释也用完整描述，避免残留扫描误判。

## 失败处理

- spec 参数不一致：返回 GenerateSpec。
- intrinsic 或 launch API 无目标证据：使用 spec 的保守 fallback，不能猜签名。
- non-contiguous DMA 无合法表达：返回 AxisFusion 重做 segment 规划。
- NRAM/WRAM/SRAM 编译超限：返回 AxisFusion 缩小 tile，不截断数组或删除输出。
- 需要跨 task barrier：重做 mapping 或拆成同 queue 多 kernel。
- 需要未提供外部依赖：返回 Extractor/用户，不伪造 header。

本阶段不运行 `cncc`；真实构建和修复由 GenTestCode 后的 `mlu-bangc-code-review` 完成。
