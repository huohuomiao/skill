# GenerateCode 示例：Task-local Reduce Sum

一个 task tile 唯一拥有 `(n, k-range)` 输出，在 NRAM 中保留 K accumulator 并遍历完整 M。示例用显式片上循环，不猜测 reduction intrinsic 名称或签名。

```cpp
#include <bang.h>
#include <cnrt.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

namespace {
constexpr int kTileK = 64;
constexpr int kReduceTileM = 8;
}

__mlu_global__ void reduce_sum_m_kernel(
    const float* x,
    float* y,
    int64_t N,
    int64_t M,
    int64_t K,
    int64_t stride_x0,
    int64_t stride_x1,
    int64_t stride_x2,
    int64_t stride_y0) {
  __nram__ float x_tile[kReduceTileM * kTileK];
  __nram__ float acc[kTileK];

  const int64_t tiles_k = K / kTileK + (K % kTileK != 0);
  const int64_t total_tiles = N * tiles_k;

  for (int64_t tile = static_cast<int64_t>(taskId);
       tile < total_tiles;
       tile += static_cast<int64_t>(taskDim)) {
    const int64_t n = tile / tiles_k;
    const int64_t tile_k = tile - n * tiles_k;
    const int64_t k0 = tile_k * kTileK;
    const int valid_k = K - k0 < kTileK ? static_cast<int>(K - k0)
                                        : kTileK;

    for (int k = 0; k < valid_k; ++k) {
      acc[k] = 0.0f;
    }

    for (int64_t m0 = 0; m0 < M; m0 += kReduceTileM) {
      const int valid_m = M - m0 < kReduceTileM
          ? static_cast<int>(M - m0)
          : kReduceTileM;

      for (int r = 0; r < valid_m; ++r) {
        const int64_t input_offset =
            n * stride_x0 + (m0 + r) * stride_x1 + k0 * stride_x2;
        __memcpy(x_tile + r * kTileK,
                 x + input_offset,
                 static_cast<size_t>(valid_k) * sizeof(float),
                 GDRAM2NRAM);
      }

      for (int r = 0; r < valid_m; ++r) {
        for (int k = 0; k < valid_k; ++k) {
          acc[k] += x_tile[r * kTileK + k];
        }
      }
    }

    __memcpy(y + n * stride_y0 + k0,
             acc,
             static_cast<size_t>(valid_k) * sizeof(float),
             NRAM2GDRAM);
  }
}

inline bool launch_reduce_sum_m(
    const float* x,
    float* y,
    int64_t N,
    int64_t M,
    int64_t K,
    int64_t stride_x0,
    int64_t stride_x1,
    int64_t stride_x2,
    int64_t stride_y0,
    cnrtQueue_t queue) {
  if (N < 0 || M < 0 || K < 0) return false;
  if (N == 0 || K == 0) return true;
  if (x == NULL || y == NULL || queue == NULL) return false;
  if (stride_x2 != 1 || stride_x1 < K || stride_y0 < K) {
    return false;
  }
  if (M > 0 && stride_x1 > 0 && M > INT64_MAX / stride_x1) return false;
  if (stride_x0 < M * stride_x1) return false;

  const int64_t tiles_k = K / kTileK + (K % kTileK != 0);
  if (tiles_k <= 0 || N > INT64_MAX / tiles_k) return false;

  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t type = cnrtFuncTypeBlock;
  reduce_sum_m_kernel<<<dim, type, queue>>>(
      x, y, N, M, K, stride_x0, stride_x1, stride_x2, stride_y0);
  return true;
}

// === BANGC_TEST_HARNESS_BEGIN ===
```

## 审查要点

- `M==0` 时循环不执行，当前 task 写回 identity `0.0f`；这只适用于 requirement 定义相同 identity 的情况。
- 每个 M row 只搬连续 K segment；因此明确要求 `stride_x2==1`。其它布局必须生成不同 mapping。
- accumulator 只占 `kTileK`，不会无必要保留完整 reduction tile。
- CPU reference 应用 double 累加并按需求容差比较；不能通过扩大容差掩盖索引错误。
- 使用目标环境确认的 `__bang_*` reduction 前，必须验证 dtype、长度、identity 与累加顺序。
