# GenerateSpec

## 任务

把基础信息、task/tile 映射和片上访存规划固化为无歧义的原生 BANG C/CNRT 实现规格 `step4_code_spec.json`。规格必须足以让 GenerateCode 机械落地，不允许保留“自行选择 tile”“必要时同步”“使用合适 intrinsic”等开放决策。

## 输入

- `step1_base_info.json`
- `step1_io_shapes.json`
- `step2_block_mapping.json`
- `step3_axis_fusion.json`
- `requirement.md`
- 最近的 `EnvConfig/config.md`
- `{BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md`
- `{BANGC_SKILL_ROOT}/share/mlu/references/primitives.md`

上游任一 `ready_*` 为 false、存在 blocking/open issues 或 JSON 不一致时，返回对应阶段，不生成猜测规格。

## 规格覆盖范围

必须明确：

1. translation unit 的标准头、`bang.h`、`cnrt.h` 与实际依赖。
2. CNRT 错误检查策略；禁止重定义 SDK 的 `CNRT_CHECK`。
3. 每个 device helper 的签名、数值语义和地址空间。
4. 每个 `__mlu_global__` kernel 的完整参数与 compile-time tile 常量。
5. `taskId/taskDim` 到逻辑 tile 的表达式和覆盖证明。
6. 每个 NRAM/WRAM/SRAM buffer 的声明、bytes、生命周期和容量证据状态。
7. 每个 GDRAM/片上 load、compute、store 的顺序、方向、offset、byte count、tail。
8. 每个 `__bang_*` 的目标头文件证据、dtype、长度、对齐和 scalar fallback。
9. CNRT launcher 的 queue、`cnrtDim3_t`、`cnrtFunctionType_t`、空输入与同步语义。
10. `cncc`、include/lib、链接库与已经确认的完整 arch flag。
11. CPU reference、测试矩阵、比较规则与 CNRT notifier benchmark。

## Kernel Schema

对每个 kernel 创建独立对象。下面只展示结构，真实值来自上游：

```json
{
  "name": "vector_add_kernel",
  "qualifier": "__mlu_global__",
  "parameters": [
    {"type": "const float*", "name": "x", "address_space": "GDRAM"},
    {"type": "const float*", "name": "y", "address_space": "GDRAM"},
    {"type": "float*", "name": "out", "address_space": "GDRAM"},
    {"type": "int64_t", "name": "n"}
  ],
  "compile_time_constants": {"TILE_ELEMS": 256},
  "task_mapping": {
    "total_tiles": "ceil_div(n, TILE_ELEMS)",
    "loop_init": "int64_t tile = int64_t(taskId)",
    "loop_condition": "tile < total_tiles",
    "loop_step": "int64_t(taskDim)",
    "tile_begin": "tile * TILE_ELEMS",
    "valid_elements": "min(TILE_ELEMS, n - tile_begin)"
  },
  "onchip_buffers": [
    {"space": "__nram__", "type": "float", "name": "x_nram", "elements": "TILE_ELEMS", "bytes": "TILE_ELEMS*sizeof(float)"},
    {"space": "__nram__", "type": "float", "name": "y_nram", "elements": "TILE_ELEMS", "bytes": "TILE_ELEMS*sizeof(float)"},
    {"space": "__nram__", "type": "float", "name": "out_nram", "elements": "TILE_ELEMS", "bytes": "TILE_ELEMS*sizeof(float)"}
  ],
  "resource_summary": {
    "nram_bytes": "3*TILE_ELEMS*sizeof(float)",
    "wram_bytes": 0,
    "sram_bytes": 0,
    "capacity_source": "EnvConfig or cncc compile gate",
    "capacity_status": "must_be_verified"
  },
  "operations": [
    {"order": 1, "kind": "copy", "direction": "GDRAM2NRAM", "source": "x + begin", "destination": "x_nram", "bytes": "valid*sizeof(float)"},
    {"order": 2, "kind": "copy", "direction": "GDRAM2NRAM", "source": "y + begin", "destination": "y_nram", "bytes": "valid*sizeof(float)"},
    {
      "order": 3,
      "kind": "compute",
      "primitive": "__bang_add",
      "call": "__bang_add(out_nram, x_nram, y_nram, TILE_ELEMS)",
      "evidence": "target primitives table and bang.h",
      "padded_lanes": "initialize x/y lanes [valid,TILE_ELEMS) to 0 before full-width call",
      "scalar_fallback": "for i in [0,valid): out_nram[i]=x_nram[i]+y_nram[i]"
    },
    {"order": 4, "kind": "copy", "direction": "NRAM2GDRAM", "source": "out_nram", "destination": "out + begin", "bytes": "valid*sizeof(float)"}
  ],
  "global_accesses": [
    {"tensor": "x", "mode": "read", "element_offset": "begin", "bytes": "valid*sizeof(float)"},
    {"tensor": "y", "mode": "read", "element_offset": "begin", "bytes": "valid*sizeof(float)"},
    {"tensor": "out", "mode": "write", "element_offset": "begin", "bytes": "valid*sizeof(float)"}
  ],
  "tail": "copy only valid bytes; initialize padded lanes only in NRAM; write only valid bytes",
  "inter_task_sync": "none",
  "atomics": []
}
```

`TILE_ELEMS=256` 是 schema 示例，不是通用默认。实际 spec 要从上游 tile plan 固定具体值并记录 byte 公式；没有容量数字时只能标记“待 cncc 编译门禁确认”，不能声称适配全部 MLU590。

`__restrict__` 只有在 requirement 明确禁止指针重叠时才可加入。允许原地时必须确保所有输入在覆盖前已搬入片上 buffer。

## 搬运 Schema

每次 `__memcpy` 必须记录：

```json
{
  "primitive": "__memcpy",
  "direction": "GDRAM2NRAM",
  "destination": "x_nram + local_offset",
  "source": "x + global_offset",
  "bytes": "valid_elements * sizeof(float)",
  "source_range": "[global_offset, global_offset + valid_elements)",
  "destination_range": "[local_offset, local_offset + valid_elements)",
  "preconditions": ["valid_elements >= 0", "ranges are in bounds"],
  "alignment_evidence": null,
  "fallback": "smaller contiguous segments or scalar access"
}
```

- 第三个参数始终是 bytes。
- 非连续 tensor 用循环产生多条连续 segment；不要构造一个越过 padding 的 DMA。
- 异步搬运只有在 `__memcpy_async`、同步和 buffer 生命周期均经共享资料确认后使用；通用基线默认同步 `__memcpy`。

## Reduction Schema

归约不沿用向量语言的块归约概念。每个 reduction loop 固定以下字段：

```json
{
  "reduce_axis": "K",
  "extent": "K",
  "chunk": "REDUCE_TILE",
  "accumulator_dtype": "float",
  "accumulator_buffer": "acc_nram",
  "identity": "0.0f",
  "strategy": "nram_accumulate | chunk_reduce_then_accumulate | multi_pass",
  "load": "GDRAM2NRAM current valid chunk",
  "tail_identity_fill": "0.0f",
  "primitive": null,
  "primitive_evidence": null,
  "scalar_fallback": "explicit loop over valid chunk",
  "final_store_owner": "current task"
}
```

选择规则：

- `nram_accumulate`：输出 tile accumulator 留在 NRAM，遍历全部 K chunks 后一次写回。
- `chunk_reduce_then_accumulate`：每个 chunk 先在 NRAM 内归约成 partial，再累加到较小 accumulator。
- `multi_pass`：softmax 等需要 max、exp/sum、normalize 的确定多遍算法；逐遍写明是否重载输入、保存中间值或使用 workspace。
- 一个输出跨 task 合并时必须给出 multi-kernel partial buffer/atomic 方案及确定性依据；不能依赖隐式全局同步。
- `__bang_sumpool` 等名称只有目标原语表确认时填写；否则 `primitive=null` 并使用 scalar baseline。

## WRAM/SRAM Schema

```json
{
  "wram": {
    "enabled": false,
    "buffers": [],
    "bytes": 0,
    "reason": "baseline has no verified weight-layout requirement"
  },
  "sram": {
    "enabled": false,
    "buffers": [],
    "bytes": 0,
    "participants": null,
    "synchronization": [],
    "reason": "baseline is task independent"
  }
}
```

启用时必须补齐生产者、消费者、布局、bytes、容量事实、参与 cluster/core 和真实同步原语；否则保持禁用。

## Launcher Schema

业务 launcher 接收 device pointers 与现有 queue，不进行分配、拷贝或无条件同步：

```json
{
  "name": "launch_vector_add",
  "return_type": "void",
  "parameters": ["const float* x", "const float* y", "float* out", "int64_t n", "cnrtQueue_t queue"],
  "preconditions": [
    {"condition": "n < 0", "action": "report invalid argument according to wrapper contract"},
    {"condition": "n == 0", "action": "return without launch"},
    {"condition": "x/y/out are non-null", "action": "required for n > 0"}
  ],
  "launch_calculation": [
    "num_tiles = ceil_div(n, TILE_ELEMS)",
    "task_count = min(num_tiles, verified_recommended_task_count) when EnvConfig supplies one",
    "otherwise task_count = 1 for a correctness-first grid-stride baseline"
  ],
  "dim": "cnrtDim3_t{uint32_t(task_count), 1, 1}",
  "function_type": "cnrtFuncTypeBlock",
  "launch": "vector_add_kernel<<<dim, function_type, queue>>>(x, y, out, n)",
  "synchronization": "none; caller observes asynchronous failure at cnrtQueueSync"
}
```

如果本地 SDK 的 launch ABI 不同，以 `platform-rules.md` 和已编译 sample 为准。不得生成未经确认的 launch-error API。

## 错误处理

- `.mlu` 直接包含 `<cnrt.h>`，优先使用其现有 `CNRT_CHECK` 处理 host/test API。
- 禁止再定义同名宏，避免 SDK 宏重定义。
- 禁止使用 `CNRT_RET_SUCCESS`。若代码确实要比较返回值，只使用 EnvConfig/当前 `cnrt.h` 已确认的符号；测试环境审计已知版本可用 `cnrtSuccess`，但仍应以目标头文件为准。
- kernel launch 的异步错误由随后 `cnrtQueueSync` 捕获；不要伪造不存在的“last error”函数。
- resource cleanup 必须覆盖失败路径。

## 数值与 intrinsic

- 使用与合同一致的运算和累加 dtype。
- 不自行把 `/` 改为倒数乘、把标准运算改为近似 intrinsic、把 float 累加改为 half。
- 对每个 `__bang_*` 在 spec 写出 exact call、dtype、长度单位、alignment、tail 和证据。
- 同一语义的 intrinsic 不可确认时，生成明确 C/BANG C 标量循环并把 intrinsic 留给 optimize。

## Test Schema

```json
{
  "cpu_reference": {
    "function": "reference_vector_add",
    "independent_algorithm": true,
    "accumulation_type": "float"
  },
  "correctness_cases": ["C01", "C02"],
  "comparison": {
    "floating": "abs(actual-expected) <= atol + rtol*abs(expected)",
    "nan": "per requirement",
    "integer": "exact"
  },
  "cnrt_checks": [
    "cnrtSetDevice",
    "queue create/sync/destroy",
    "malloc/free",
    "H2D/D2H memcpy",
    "notifier create/place/duration/destroy"
  ],
  "benchmark": {
    "method": "CNRT notifier",
    "warmup": 20,
    "iterations": 100,
    "scope": "launcher device work only",
    "case": "P01",
    "fields": ["host_reference_ms", "original_bangc_ms"]
  },
  "failure_exit_code": 1
}
```

warmup/iterations 可按需求固定为其它正整数，但生成代码与报告必须使用同一实际值。

## 顶层输出 Schema

```json
{
  "schema_version": 1,
  "operator_name": "...",
  "translation_unit": {
    "language": "BANG C/C++",
    "output": "step5_kernel_code.mlu",
    "headers": ["bang.h", "cnrt.h", "cstdint", "cstdio"],
    "external_libraries": ["cnrt", "stdc++", "m", "pthread"]
  },
  "build": {
    "compiler": "cncc",
    "language_standard": "c++11",
    "include_dirs": ["${NEUWARE_HOME}/include"],
    "library_dirs": ["${NEUWARE_HOME}/lib64"],
    "libraries": ["cnrt", "stdc++", "m", "pthread"],
    "arch_flag": null,
    "arch_source": "EnvConfig only",
    "command": "cncc step6_test_code.mlu -o step6_test_code -I${NEUWARE_HOME}/include -L${NEUWARE_HOME}/lib64 -lcnrt -lstdc++ -lm -lpthread -std=c++11"
  },
  "error_handling": {
    "helper": "CNRT_CHECK from cnrt.h; do not redefine",
    "async_check": "cnrtQueueSync in harness",
    "cleanup": "all allocated resources and notifiers"
  },
  "device_helpers": [],
  "kernels": [],
  "launchers": [],
  "tests": {},
  "provenance": {
    "base_info": "step1_base_info.json",
    "io_shapes": "step1_io_shapes.json",
    "mapping": "step2_block_mapping.json",
    "axis_plan": "step3_axis_fusion.json"
  },
  "unresolved": [],
  "ready_for_code": true
}
```

`arch_flag` 为 null 时命令不追加架构参数，并在报告注明使用编译器默认。不能填入由 MLU590 名称推断的值。

## 参考示例路由

| Pattern | Spec 示例 | Code 示例 |
|---|---|---|
| 连续逐元素/维度线性化 | [generate_spec_axis_fusion.md](./examples/generate_spec_axis_fusion.md) | [generate_code_axis_fusion.md](./examples/generate_code_axis_fusion.md) |
| NRAM tile transpose | [generate_spec_matrix_transpose.md](./examples/generate_spec_matrix_transpose.md) | [generate_code_matrix_transpose.md](./examples/generate_code_matrix_transpose.md) |
| task-local reduction | [generate_spec_reduce_sum.md](./examples/generate_spec_reduce_sum.md) | [generate_code_reduce_sum.md](./examples/generate_code_reduce_sum.md) |
| transpose + elementwise | [generate_spec_transpose_elementwise.md](./examples/generate_spec_transpose_elementwise.md) | [generate_code_transpose_elementwise.md](./examples/generate_code_transpose_elementwise.md) |

示例用于结构参考，不替代 requirement；不得复制示例算子名、shape、容差、tile 或 task 数。

## 一致性检查

- kernel 参数与 launcher 参数顺序/类型一致。
- task loop、launch dim 与 mapping JSON 一致。
- 每个片上声明的 bytes 与 resource summary 一致。
- 每个 copy 的方向、offset、bytes 和 tail 完整。
- 每个 intrinsic 有证据与 fallback。
- SRAM/WRAM 参与关系与同步（如有）完整。
- 所有 GDRAM access 有范围证明。
- 空输入和非法参数有 launcher 行为。
- tests 与 base_info 对齐。
- build 包含 `cnrt/stdc++/m/pthread`，arch 不猜测，未硬查 `include/bang.h`。
- 未重定义 `CNRT_CHECK`，未使用 `CNRT_RET_SUCCESS`。
- `unresolved=[]` 且 `ready_for_code=true` 才能交接；资源若仅待编译门禁确认，必须明确写 `capacity_status=must_be_verified`，不得写“已适配”。

## 失败处理

| 冲突 | 返回 |
|---|---|
| 数学/接口不完整 | ExtractBaseInfo / Extractor |
| task coverage 无证明 | TraceBlockMapping |
| stride/融合/DMA 无依据 | AxisFusion |
| tile byte 公式明显不成立 | AxisFusion 重选 tile |
| intrinsic 签名未确认 | 移除 intrinsic，改为 scalar baseline |
| 跨 task 同步无合法方案 | 重做 mapping 或拆为多 kernel |
| arch/capacity 事实缺失 | 不猜；使用 correctness-first fallback 或等待环境信息 |

禁止在规格中用 TODO、placeholder、省略号或未解析自然语言绕过失败。
