# GenerateCode 示例：NRAM Tile 矩阵转置

本示例使用两个小型 NRAM tile。input/output 分别沿各自连续轴逐行 DMA；片上用显式循环转置，不猜测专用转置 intrinsic。

```cpp
#include <bang.h>
#include <cnrt.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

namespace {
constexpr int kTileM = 8;
constexpr int kTileN = 8;
}

__mlu_global__ void matrix_transpose_kernel(
    const float* x,
    float* y,
    int64_t M,
    int64_t N,
    int64_t stride_x0,
    int64_t stride_y0) {
  __nram__ float input_tile[kTileM * kTileN];
  __nram__ float output_tile[kTileN * kTileM];

  const int64_t tiles_m = M / kTileM + (M % kTileM != 0);
  const int64_t tiles_n = N / kTileN + (N % kTileN != 0);
  const int64_t total_tiles = tiles_m * tiles_n;

  for (int64_t tile = static_cast<int64_t>(taskId);
       tile < total_tiles;
       tile += static_cast<int64_t>(taskDim)) {
    const int64_t tile_m = tile / tiles_n;
    const int64_t tile_n = tile - tile_m * tiles_n;
    const int64_t m0 = tile_m * kTileM;
    const int64_t n0 = tile_n * kTileN;
    const int valid_m = M - m0 < kTileM ? static_cast<int>(M - m0)
                                        : kTileM;
    const int valid_n = N - n0 < kTileN ? static_cast<int>(N - n0)
                                        : kTileN;

    for (int r = 0; r < valid_m; ++r) {
      __memcpy(input_tile + r * kTileN,
               x + (m0 + r) * stride_x0 + n0,
               static_cast<size_t>(valid_n) * sizeof(float),
               GDRAM2NRAM);
    }

    for (int r = 0; r < valid_m; ++r) {
      for (int c = 0; c < valid_n; ++c) {
        output_tile[c * kTileM + r] = input_tile[r * kTileN + c];
      }
    }

    for (int c = 0; c < valid_n; ++c) {
      __memcpy(y + (n0 + c) * stride_y0 + m0,
               output_tile + c * kTileM,
               static_cast<size_t>(valid_m) * sizeof(float),
               NRAM2GDRAM);
    }
  }
}

inline bool launch_matrix_transpose(
    const float* x,
    float* y,
    int64_t M,
    int64_t N,
    int64_t stride_x0,
    int64_t stride_y0,
    cnrtQueue_t queue) {
  if (M < 0 || N < 0 || stride_x0 < N || stride_y0 < M) return false;
  if (M == 0 || N == 0) return true;
  if (x == NULL || y == NULL || queue == NULL) return false;

  const int64_t tiles_m = M / kTileM + (M % kTileM != 0);
  const int64_t tiles_n = N / kTileN + (N % kTileN != 0);
  if (tiles_m <= 0 || tiles_n <= 0 || tiles_m > INT64_MAX / tiles_n) {
    return false;
  }

  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t type = cnrtFuncTypeBlock;
  matrix_transpose_kernel<<<dim, type, queue>>>(
      x, y, M, N, stride_x0, stride_y0);
  return true;
}

// === BANGC_TEST_HARNESS_BEGIN ===
```

## 审查要点

- input 每次只读 `valid_n*sizeof(float)`；output 每次只写 `valid_m*sizeof(float)`。
- tail tile 不读取未初始化 NRAM，也不越过 GDRAM 行边界。
- `input_tile` 与 `output_tile` 的总 NRAM bytes 必须共同计入 spec/编译门禁。
- 若 runtime stride 的连续内轴不为 1，不能使用此示例；应返回 AxisFusion 生成分段/逐元素访问。
- task 通过 grid-stride 覆盖 tile，不需要跨 task 同步。
