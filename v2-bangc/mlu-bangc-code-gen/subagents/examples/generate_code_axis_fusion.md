# GenerateCode 示例：融合 H/W 的 NCHW → NHWC Sigmoid

本示例展示：融合逻辑输出域、`taskId/taskDim` 覆盖、gather 到 NRAM、连续输出写回和 CNRT launch。它不假设存在向量 sigmoid intrinsic；`expf` 是否由目标工具链支持必须由共享原语/真实编译确认。

```cpp
#include <bang.h>
#include <cnrt.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

namespace {
constexpr int kTileOut = 256;

inline bool checked_mul_i64(int64_t a, int64_t b, int64_t* out) {
  if (a < 0 || b < 0 || out == NULL) return false;
  if (a != 0 && b > INT64_MAX / a) return false;
  *out = a * b;
  return true;
}
}  // namespace

__mlu_global__ void nchw_to_nhwc_sigmoid_kernel(
    const float* x,
    float* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t total) {
  __nram__ float x_nram[kTileOut];
  __nram__ float out_nram[kTileOut];

  const int64_t tiles = (total + kTileOut - 1) / kTileOut;
  for (int64_t tile = static_cast<int64_t>(taskId);
       tile < tiles;
       tile += static_cast<int64_t>(taskDim)) {
    const int64_t begin = tile * static_cast<int64_t>(kTileOut);
    const int64_t remain = total - begin;
    const int valid = remain < kTileOut ? static_cast<int>(remain)
                                        : kTileOut;

    for (int i = 0; i < valid; ++i) {
      const int64_t linear = begin + i;
      const int64_t c = linear % C;
      const int64_t nhw = linear / C;
      const int64_t hw = nhw % (H * W);
      const int64_t n = nhw / (H * W);
      const int64_t h = hw / W;
      const int64_t w = hw - h * W;
      const int64_t input_offset = ((n * C + c) * H + h) * W + w;
      __memcpy(x_nram + i, x + input_offset, sizeof(float), GDRAM2NRAM);
    }

    for (int i = 0; i < valid; ++i) {
      out_nram[i] = 1.0f / (1.0f + expf(-x_nram[i]));
    }

    __memcpy(out + begin, out_nram,
             static_cast<size_t>(valid) * sizeof(float), NRAM2GDRAM);
  }
}

inline bool launch_nchw_to_nhwc_sigmoid(
    const float* x,
    float* out,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cnrtQueue_t queue) {
  if (N < 0 || C < 0 || H < 0 || W < 0) return false;
  if (N == 0 || C == 0 || H == 0 || W == 0) return true;
  if (x == NULL || out == NULL || queue == NULL) return false;

  int64_t total = 0;
  int64_t tmp = 0;
  if (!checked_mul_i64(N, C, &tmp) ||
      !checked_mul_i64(tmp, H, &tmp) ||
      !checked_mul_i64(tmp, W, &total)) {
    return false;
  }

  // 正确性优先 fallback；EnvConfig 给出已验证并行度后可按 spec 替换。
  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t type = cnrtFuncTypeBlock;
  nchw_to_nhwc_sigmoid_kernel<<<dim, type, queue>>>(
      x, out, N, C, H, W, total);
  return true;
}

// === BANGC_TEST_HARNESS_BEGIN ===
```

## 审查要点

- 输出线性顺序是 NHWC，input offset 明确恢复 NCHW 坐标。
- gather 每次只搬一个有效 float，不把非连续 input 伪装成连续 DMA；这是正确性 baseline，性能优化应由实测驱动。
- output 是连续段，一次写回 `valid*sizeof(float)`。
- `dim.x=1` 不是硬件最优结论，而是缺少已验证 task 数时的合法 grid-stride fallback。
- 未重定义 `CNRT_CHECK`，没有旧返回常量，也没有硬编码 arch/片上容量。
