# GenerateSpec 示例：Transpose + Elementwise Add

## 场景

```text
A: [M,N] row-major
B: [N,M] row-major
C: [N,M] row-major
C[n,m] = A[m,n] + B[n,m]
```

## 输出规格

```json
{
  "schema_version": 1,
  "operator_name": "transpose_add",
  "translation_unit": {
    "language": "BANG C/C++",
    "output": "step5_kernel_code.mlu",
    "headers": ["bang.h", "cnrt.h", "stdint.h", "stddef.h"]
  },
  "kernels": [
    {
      "name": "transpose_add_kernel",
      "qualifier": "__mlu_global__",
      "parameters": [
        {"type": "const float*", "name": "a"},
        {"type": "const float*", "name": "b"},
        {"type": "float*", "name": "c"},
        {"type": "int64_t", "name": "M"},
        {"type": "int64_t", "name": "N"}
      ],
      "compile_time_constants": {"TILE_N": 8, "TILE_M": 8},
      "task_mapping": {
        "tiles_n": "ceil_div(N,TILE_N)",
        "tiles_m": "ceil_div(M,TILE_M)",
        "total_tiles": "tiles_n*tiles_m",
        "loop": "tile=taskId; tile<total_tiles; tile+=taskDim",
        "tile_n": "tile/tiles_m",
        "tile_m": "tile%tiles_m"
      },
      "onchip_buffers": [
        {"space": "__nram__", "type": "float", "name": "a_rows", "elements": "TILE_M*TILE_N", "bytes": "TILE_M*TILE_N*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "a_transposed", "elements": "TILE_N*TILE_M", "bytes": "TILE_N*TILE_M*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "b_tile", "elements": "TILE_N*TILE_M", "bytes": "TILE_N*TILE_M*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "c_tile", "elements": "TILE_N*TILE_M", "bytes": "TILE_N*TILE_M*sizeof(float)"}
      ],
      "resource_summary": {
        "nram_bytes": "4*TILE_N*TILE_M*sizeof(float)",
        "wram_bytes": 0,
        "sram_bytes": 0,
        "capacity_status": "must_be_verified_by_EnvConfig_or_cncc"
      },
      "operations": [
        {
          "order": 1,
          "kind": "load_a_rows",
          "copy": "for m in valid_m: __memcpy(a_rows+m*TILE_N,a+(m0+m)*N+n0,valid_n*sizeof(float),GDRAM2NRAM)"
        },
        {
          "order": 2,
          "kind": "transpose_in_nram",
          "body": "a_transposed[n*TILE_M+m]=a_rows[m*TILE_N+n] for valid lanes; zero padded lanes"
        },
        {
          "order": 3,
          "kind": "load_b_rows",
          "copy": "for n in valid_n: __memcpy(b_tile+n*TILE_M,b+(n0+n)*M+m0,valid_m*sizeof(float),GDRAM2NRAM); zero padded lanes"
        },
        {
          "order": 4,
          "kind": "compute",
          "primitive": "__bang_add",
          "call": "__bang_add(c_tile,a_transposed,b_tile,TILE_N*TILE_M)",
          "evidence": "target primitives table and bang.h",
          "scalar_fallback": "c_tile[i]=a_transposed[i]+b_tile[i] for valid lanes"
        },
        {
          "order": 5,
          "kind": "store_c_rows",
          "copy": "for n in valid_n: __memcpy(c+(n0+n)*M+m0,c_tile+n*TILE_M,valid_m*sizeof(float),NRAM2GDRAM)"
        }
      ],
      "tail": "initialize all padded NRAM lanes before full-tile __bang_add; write only valid row bytes",
      "inter_task_sync": "none"
    }
  ],
  "launchers": [
    {
      "name": "launch_transpose_add",
      "task_count": "verified value from EnvConfig, otherwise 1",
      "dim": "cnrtDim3_t{task_count,1,1}",
      "function_type": "cnrtFuncTypeBlock",
      "queue": "queue",
      "zero_size": "return without launch"
    }
  ],
  "tests": {
    "cpu_reference": "independent C[n*M+m]=A[m*N+n]+B[n*M+m]",
    "correctness_cases": ["minimum", "full_tile", "two_axis_tail"],
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

- A 与 B/C 的连续方向不同，分别逐行 DMA 后在 NRAM 重排。
- 只有目标 `bang.h` 确认 `__bang_add` 的 float32、长度和对齐要求时启用；否则使用 scalar fallback。
- full-tile intrinsic 前初始化所有 padded lanes；GDRAM 仍只读写有效 bytes。
- 4 个片上 tile 的 byte 数必须全部计入资源检查。
