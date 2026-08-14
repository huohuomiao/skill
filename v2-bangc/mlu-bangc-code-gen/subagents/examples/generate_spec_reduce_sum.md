# GenerateSpec 示例：Reduce Sum

## 场景

```text
X: [N,M,K]，stride 由 runtime 传入，示例要求 K 轴连续
Y: [N,K] contiguous
Y[n,k] = sum_m X[n,m,k]
```

## 输出规格

```json
{
  "schema_version": 1,
  "operator_name": "reduce_sum_m",
  "translation_unit": {
    "language": "BANG C/C++",
    "output": "step5_kernel_code.mlu",
    "headers": ["bang.h", "cnrt.h", "stdint.h", "stddef.h"]
  },
  "kernels": [
    {
      "name": "reduce_sum_m_kernel",
      "qualifier": "__mlu_global__",
      "parameters": [
        {"type": "const float*", "name": "x"},
        {"type": "float*", "name": "y"},
        {"type": "int64_t", "name": "N"},
        {"type": "int64_t", "name": "M"},
        {"type": "int64_t", "name": "K"},
        {"type": "int64_t", "name": "stride_x0"},
        {"type": "int64_t", "name": "stride_x1"},
        {"type": "int64_t", "name": "stride_x2"}
      ],
      "compile_time_constants": {"TILE_K": 64, "REDUCE_TILE_M": 8},
      "task_mapping": {
        "tiles_k": "ceil_div(K,TILE_K)",
        "total_tiles": "N*tiles_k",
        "loop": "tile=taskId; tile<total_tiles; tile+=taskDim",
        "n": "tile/tiles_k",
        "tile_k": "tile%tiles_k",
        "k0": "tile_k*TILE_K",
        "valid_k": "min(TILE_K,K-k0)"
      },
      "onchip_buffers": [
        {"space": "__nram__", "type": "float", "name": "x_tile", "elements": "REDUCE_TILE_M*TILE_K", "bytes": "REDUCE_TILE_M*TILE_K*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "acc", "elements": "TILE_K", "bytes": "TILE_K*sizeof(float)"}
      ],
      "resource_summary": {
        "nram_bytes": "(REDUCE_TILE_M*TILE_K+TILE_K)*sizeof(float)",
        "wram_bytes": 0,
        "sram_bytes": 0,
        "capacity_status": "must_be_verified_by_EnvConfig_or_cncc"
      },
      "reduction": {
        "reduce_axis": "M",
        "extent": "M",
        "chunk": "REDUCE_TILE_M",
        "strategy": "nram_accumulate",
        "accumulator_dtype": "float",
        "identity": "0.0f",
        "load": "for each valid m row, copy valid_k contiguous elements from GDRAM to x_tile row",
        "accumulate": "for each valid m and k lane, acc[k]+=x_tile[m*TILE_K+k]",
        "primitive": null,
        "primitive_reason": "do not guess a reduction intrinsic signature",
        "scalar_fallback": "explicit nested loop",
        "final_store_owner": "current task"
      },
      "operations": [
        {"order": 1, "kind": "initialize", "body": "acc[0:valid_k]=0"},
        {
          "order": 2,
          "kind": "reduce_loop",
          "range": "m0=0; m0<M; m0+=REDUCE_TILE_M",
          "copy": "__memcpy(x_tile+r*TILE_K, x+n*stride_x0+(m0+r)*stride_x1+k0*stride_x2, valid_k*sizeof(float), GDRAM2NRAM)",
          "precondition": "stride_x2==1 for row DMA; otherwise return to AxisFusion for a strided plan"
        },
        {"order": 3, "kind": "store", "copy": "__memcpy(y+n*K+k0,acc,valid_k*sizeof(float),NRAM2GDRAM)"}
      ],
      "tail": "valid_m and valid_k bound every loop/copy; no padded GDRAM access",
      "inter_task_sync": "none"
    }
  ],
  "launchers": [
    {
      "name": "launch_reduce_sum_m",
      "task_count": "verified value from EnvConfig, otherwise 1",
      "dim": "cnrtDim3_t{task_count,1,1}",
      "function_type": "cnrtFuncTypeBlock",
      "queue": "queue",
      "empty_reduction": "write requirement identity without reading X",
      "zero_output": "return without launch"
    }
  ],
  "tests": {
    "cpu_reference": "double accumulator over M, then cast to float",
    "correctness_cases": ["M=0 if allowed", "M tail", "K tail", "noncontiguous M stride"],
    "comparison": "atol + rtol*abs(expected)",
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

- 一个 task tile 唯一拥有 `(n, k-range)` 输出并遍历完整 M；不需要跨 task reduction。
- `stride_x2==1` 才能对 K 片段逐行 DMA。其它 stride 必须生成逐元素/分段方案，不能忽略。
- 累加器只含输出 K tile，不保留整个 M 维，避免无必要的 NRAM 放大。
- 若目标原语表确认 reduction intrinsic，可在不改变累加 dtype/顺序合同的前提下替换；本示例不猜名称。
