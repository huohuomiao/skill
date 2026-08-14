# GenerateSpec 示例：融合 H/W 输出域

## 场景

```text
x:   [N,C,H,W] contiguous
out: [N,H,W,C] contiguous
out[n,h,w,c] = sigmoid(x[n,c,h,w])
融合 H 和 W 为 HW，但保留 input/output 不同的地址公式。
```

## 输入摘要

```json
{
  "compute_formula": "out[n,h,w,c] = 1 / (1 + exp(-x[n,c,h,w]))",
  "fusion_note": "H + W -> HW for logical task domain",
  "io_block_mapping": {
    "x": {
      "axis": ["N", "C", "H", "W"],
      "strides": ["C*H*W", "H*W", "W", 1]
    },
    "out": {
      "axis": ["N", "HW", "C"],
      "strides": ["HW*C", "C", 1]
    }
  }
}
```

## 输出规格

```json
{
  "schema_version": 1,
  "operator_name": "nchw_to_nhwc_sigmoid",
  "translation_unit": {
    "language": "BANG C/C++",
    "output": "step5_kernel_code.mlu",
    "headers": ["bang.h", "cnrt.h", "stdint.h", "stddef.h", "math.h"]
  },
  "kernels": [
    {
      "name": "nchw_to_nhwc_sigmoid_kernel",
      "qualifier": "__mlu_global__",
      "parameters": [
        {"type": "const float*", "name": "x"},
        {"type": "float*", "name": "out"},
        {"type": "int64_t", "name": "N"},
        {"type": "int64_t", "name": "C"},
        {"type": "int64_t", "name": "H"},
        {"type": "int64_t", "name": "W"}
      ],
      "compile_time_constants": {"TILE_OUT": 256},
      "task_mapping": {
        "total_elements": "N*H*W*C",
        "total_tiles": "ceil_div(total_elements,TILE_OUT)",
        "loop": "tile=taskId; tile<total_tiles; tile+=taskDim",
        "inverse_mapping": [
          "c=linear%C",
          "hw=(linear/C)%(H*W)",
          "n=linear/(C*H*W)",
          "h=hw/W",
          "w=hw%W"
        ]
      },
      "onchip_buffers": [
        {"space": "__nram__", "type": "float", "name": "x_nram", "elements": 256, "bytes": "256*sizeof(float)"},
        {"space": "__nram__", "type": "float", "name": "out_nram", "elements": 256, "bytes": "256*sizeof(float)"}
      ],
      "resource_summary": {
        "nram_bytes": "2*256*sizeof(float)",
        "wram_bytes": 0,
        "sram_bytes": 0,
        "capacity_status": "must_be_verified_by_EnvConfig_or_cncc"
      },
      "operations": [
        {
          "kind": "gather_to_nram",
          "source_offset": "((n*C+c)*H+h)*W+w",
          "method": "one valid scalar segment at a time because permutation breaks contiguous input order",
          "copy": "__memcpy(x_nram+i,x+source_offset,sizeof(float),GDRAM2NRAM)"
        },
        {
          "kind": "compute",
          "formula": "out_nram[i]=1.0f/(1.0f+expf(-x_nram[i]))",
          "primitive": null,
          "reason": "do not invent a vector sigmoid intrinsic; use target-supported scalar math baseline"
        },
        {
          "kind": "store",
          "copy": "__memcpy(out+begin,out_nram,valid*sizeof(float),NRAM2GDRAM)"
        }
      ],
      "tail": "gather and compute only valid lanes; store valid bytes"
    }
  ],
  "launchers": [
    {
      "name": "launch_nchw_to_nhwc_sigmoid",
      "dim": "cnrtDim3_t{task_count,1,1}",
      "task_count": "verified value from EnvConfig, otherwise 1",
      "function_type": "cnrtFuncTypeBlock",
      "queue": "queue",
      "zero_size": "return without launch"
    }
  ],
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

- 融合的是逻辑输出域 `HW`；input 的 NCHW offset 仍需从 `hw` 反解 h/w。
- out 为连续 NHWC，可一次写回 `valid*sizeof(float)`；input 在该遍历顺序下是 gather，不能伪装成连续 DMA。
- 示例使用很小的片上 byte 公式，但不声明它等于 MLU590 容量；真实编译仍是资源 gate。
- 未确认向量 sigmoid intrinsic 时使用标量 math；后续 optimize 可依据目标 `bang.h` 再替换。
