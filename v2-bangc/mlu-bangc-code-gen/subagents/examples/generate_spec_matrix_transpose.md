# GenerateSpec 示例：矩阵转置

## 场景

```text
X: [M,N] row-major
Y: [N,M] row-major
Y[n,m] = X[m,n]
```

## 输出规格

```json
{
  "schema_version": 1,
  "operator_name": "matrix_transpose",
  "translation_unit": {
    "language": "BANG C/C++",
    "output": "step5_kernel_code.mlu",
    "headers": ["bang.h", "cnrt.h", "stdint.h", "stddef.h"]
  },
  "kernels": [
    {
      "name": "matrix_transpose_kernel",
      "qualifier": "__mlu_global__",
      "parameters": [
        {"type": "const float*", "name": "x"},
        {"type": "float*", "name": "y"},
        {"type": "int64_t", "name": "M"},
        {"type": "int64_t", "name": "N"},
        {"type": "int64_t", "name": "stride_x0"},
        {"type": "int64_t", "name": "stride_y0"}
      ],
      "compile_time_constants": {"TILE_M": 8, "TILE_N": 8},
      "task_mapping": {
        "tiles_m": "ceil_div(M,TILE_M)",
        "tiles_n": "ceil_div(N,TILE_N)",
        "total_tiles": "tiles_m*tiles_n",
        "loop": "tile=taskId; tile<total_tiles; tile+=taskDim",
        "tile_m": "tile/tiles_n",
        "tile_n": "tile%tiles_n"
      },
      "onchip_buffers": [
        {"space": "__nram__", "type": "float", "name": "input_tile", "elements": "TILE_M*TILE_N", "bytes": "TILE_M*TILE_N*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "output_tile", "elements": "TILE_M*TILE_N", "bytes": "TILE_M*TILE_N*sizeof(float)"}
      ],
      "resource_summary": {
        "nram_bytes": "2*TILE_M*TILE_N*sizeof(float)",
        "wram_bytes": 0,
        "sram_bytes": 0,
        "capacity_status": "must_be_verified_by_EnvConfig_or_cncc"
      },
      "operations": [
        {
          "order": 1,
          "kind": "row_loads",
          "repeat": "r in [0,valid_m)",
          "copy": "__memcpy(input_tile+r*TILE_N, x+(m0+r)*stride_x0+n0, valid_n*sizeof(float), GDRAM2NRAM)"
        },
        {
          "order": 2,
          "kind": "nram_transpose",
          "formula": "output_tile[c*TILE_M+r]=input_tile[r*TILE_N+c]",
          "primitive": null,
          "fallback": "nested scalar loop over valid_m,valid_n"
        },
        {
          "order": 3,
          "kind": "row_stores",
          "repeat": "c in [0,valid_n)",
          "copy": "__memcpy(y+(n0+c)*stride_y0+m0, output_tile+c*TILE_M, valid_m*sizeof(float), NRAM2GDRAM)"
        }
      ],
      "tail": "valid_m=min(TILE_M,M-m0); valid_n=min(TILE_N,N-n0); every row copy uses its own valid byte count",
      "inter_task_sync": "none"
    }
  ],
  "launchers": [
    {
      "name": "launch_matrix_transpose",
      "task_count": "verified value from EnvConfig, otherwise 1",
      "dim": "cnrtDim3_t{task_count,1,1}",
      "function_type": "cnrtFuncTypeBlock",
      "queue": "queue",
      "zero_size": "return without launch"
    }
  ],
  "tests": {
    "correctness_cases": ["minimum", "full_tile", "two_axis_tail"],
    "cpu_reference": "independent nested loop Y[n*M+m]=X[m*N+n]",
    "benchmark": "CNRT notifier around launcher only"
  },
  "build": {
    "compiler": "cncc",
    "language_standard": "c++11",
    "libraries": ["cnrt", "stdc++", "m", "pthread"],
    "arch_flag": null,
    "arch_source": "EnvConfig only"
  },
  "unresolved": [],
  "ready_for_code": true
}
```

## 关键点

- input 与 output 各自在其连续轴上逐行 DMA；不能把整个二维 tile 当成一个连续 region。
- 两个 NRAM tile 的总 bytes 必须相加。
- 示例不猜测转置 intrinsic；目标环境确认存在且约束满足后才可替换 scalar loop。
- tail 行/列只读写有效区，不读取未初始化片上元素。
