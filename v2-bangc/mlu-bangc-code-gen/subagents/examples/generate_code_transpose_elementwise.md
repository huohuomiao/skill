# GenerateCode 示例：Transpose + Elementwise Add

本示例把 A tile 搬入 NRAM 后转置，再与连续 B tile 使用已确认的 `__bang_add` 相加。full-tile intrinsic 前显式清零 padding，GDRAM 只读写有效 bytes。

```cpp
#include <bang.h>
#include <cnrt.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

namespace {
constexpr int kTileN = 8;
constexpr int kTileM = 8;
constexpr int kTileElems = kTileN * kTileM;
}

__mlu_global__ void transpose_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int64_t M,
    int64_t N) {
  __nram__ float a_rows[kTileElems];
  __nram__ float a_transposed[kTileElems];
  __nram__ float b_tile[kTileElems];
  __nram__ float c_tile[kTileElems];

  const int64_t tiles_n = N / kTileN + (N % kTileN != 0);
  const int64_t tiles_m = M / kTileM + (M % kTileM != 0);
  const int64_t total_tiles = tiles_n * tiles_m;

  for (int64_t tile = static_cast<int64_t>(taskId);
       tile < total_tiles;
       tile += static_cast<int64_t>(taskDim)) {
    const int64_t tile_n = tile / tiles_m;
    const int64_t tile_m = tile - tile_n * tiles_m;
    const int64_t n0 = tile_n * kTileN;
    const int64_t m0 = tile_m * kTileM;
    const int valid_n = N - n0 < kTileN ? static_cast<int>(N - n0)
                                        : kTileN;
    const int valid_m = M - m0 < kTileM ? static_cast<int>(M - m0)
                                        : kTileM;

    for (int i = 0; i < kTileElems; ++i) {
      a_rows[i] = 0.0f;
      a_transposed[i] = 0.0f;
      b_tile[i] = 0.0f;
    }

    for (int m = 0; m < valid_m; ++m) {
      __memcpy(a_rows + m * kTileN,
               a + (m0 + m) * N + n0,
               static_cast<size_t>(valid_n) * sizeof(float),
               GDRAM2NRAM);
    }
    for (int n = 0; n < valid_n; ++n) {
      for (int m = 0; m < valid_m; ++m) {
        a_transposed[n * kTileM + m] = a_rows[m * kTileN + n];
      }
    }

    for (int n = 0; n < valid_n; ++n) {
      __memcpy(b_tile + n * kTileM,
               b + (n0 + n) * M + m0,
               static_cast<size_t>(valid_m) * sizeof(float),
               GDRAM2NRAM);
    }

    // 仅当目标 primitives 表与 bang.h 已确认 float32/长度/对齐时采用。
    __bang_add(c_tile, a_transposed, b_tile, kTileElems);

    for (int n = 0; n < valid_n; ++n) {
      __memcpy(c + (n0 + n) * M + m0,
               c_tile + n * kTileM,
               static_cast<size_t>(valid_m) * sizeof(float),
               NRAM2GDRAM);
    }
  }
}

inline bool launch_transpose_add(
    const float* a,
    const float* b,
    float* c,
    int64_t M,
    int64_t N,
    cnrtQueue_t queue) {
  if (M < 0 || N < 0) return false;
  if (M == 0 || N == 0) return true;
  if (a == NULL || b == NULL || c == NULL || queue == NULL) return false;

  const int64_t tiles_n = N / kTileN + (N % kTileN != 0);
  const int64_t tiles_m = M / kTileM + (M % kTileM != 0);
  if (tiles_n <= 0 || tiles_m <= 0 || tiles_n > INT64_MAX / tiles_m) {
    return false;
  }

  cnrtDim3_t dim = {1, 1, 1};
  cnrtFunctionType_t type = cnrtFuncTypeBlock;
  transpose_add_kernel<<<dim, type, queue>>>(a, b, c, M, N);
  return true;
}

// === BANGC_TEST_HARNESS_BEGIN ===
```

## 审查要点

- A 按 `[M,N]` 行搬入，B/C 按 `[N,M]` 行搬入/写回，地址公式没有混淆。
- padded NRAM lanes 在 `__bang_add` 前被清零；tail 不从 GDRAM 多读，也不向 GDRAM 多写。
- 若目标 `__bang_add` 不支持该 dtype/长度/对齐，GenerateCode 必须根据 spec 生成 `for valid lanes` 的标量 fallback，不能改用猜测的 intrinsic。
- 4 个 tile 的 NRAM bytes 全部进入资源检查。
- `dim.x=1` 是无设备并行事实时的正确性 fallback，后续可用实测配置替换。
