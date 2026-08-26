# ExtractBaseInfo

## 任务

把 `requirement.md` 转换为严格、可机读的 `step1_base_info.json`，并把其中的 `io_shapes` 同源写出为 `step1_io_shapes.json`。本阶段只规范化事实，不选择 task 数、tile 大小、片上存储布局或 intrinsic，不生成代码，也不改变用户数值合同。

ExtractBaseInfo 是 `mlu-bangc-code-gen` 的 Stage 1，仅在 `input_type=not_bangc` / `is_bangc=false` 时执行。

## 输入与输出

- 输入：`{output_dir}/Extractor/requirement.md`
- 输出：`{output_dir}/KernelGen/step1_base_info.json`
- 输出：`{output_dir}/KernelGen/step1_io_shapes.json`

完整读取 requirement，尤其是数学合同、接口、shape/stride、dtype、数值规则、测试用例、环境事实与未决问题。禁止读取其它阶段的候选实现来反推需求。

## 抽取顺序

1. 确认算子名、来源、输入类型与 `execution_backend`。
2. 建立全部参数清单；区分 input、output、inout、scalar、shape、stride 与 queue。
3. 同时记录 C/C++ 类型和逻辑 dtype，不把 `const float*` 与 `float*` 混为同一权限。
4. 记录每个 tensor 的符号 shape、实际测试 shape、元素 stride、layout、alignment 与 aliasing。
5. 将数学公式拆为逐元素域、归约域、广播关系和输出写入规则。
6. 记录 accumulation dtype、容差、NaN/Inf、舍入、确定性与近似 intrinsic 许可。
7. 对齐 correctness/performance 测试矩阵。
8. 记录 EnvConfig 已确认的 MLU/CNCC/NeuWare/arch 信息；未知字段写 `null`，不得按 MLU590 名称猜测。
9. 检查 blocking questions；非空时写 `ready_for_mapping=false`。

## `step1_base_info.json` Schema

```json
{
  "schema_version": 1,
  "operator": {
    "name": "affine_relu",
    "summary": "y[i] = max(a*x[i] + b, 0)",
    "input_kind": "natural_language",
    "source_path": null
  },
  "target": {
    "device_family": "MLU590",
    "device_name": null,
    "cncc_version": null,
    "neuware_version": null,
    "arch_flag": null,
    "execution_backend": "unavailable",
    "facts_source": "EnvConfig/config.md"
  },
  "interface": {
    "kernel_name": "affine_relu_kernel",
    "launcher_name": "launch_affine_relu",
    "launcher_return_type": "void",
    "queue_parameter": "queue",
    "parameters": [
      {
        "name": "x",
        "role": "input",
        "c_type": "const float*",
        "dtype": "float32",
        "symbolic_shape": ["N"],
        "test_shapes": [[1], [257]],
        "strides": [1],
        "stride_unit": "elements",
        "layout": "contiguous",
        "alignment_bytes": null,
        "nullable": false,
        "aliasing": "unspecified"
      },
      {
        "name": "y",
        "role": "output",
        "c_type": "float*",
        "dtype": "float32",
        "symbolic_shape": ["N"],
        "test_shapes": [[1], [257]],
        "strides": [1],
        "stride_unit": "elements",
        "layout": "contiguous",
        "alignment_bytes": null,
        "nullable": false,
        "aliasing": "unspecified"
      },
      {
        "name": "n",
        "role": "shape",
        "c_type": "int64_t",
        "value_domain": "n >= 0"
      },
      {
        "name": "a",
        "role": "scalar",
        "c_type": "float"
      },
      {
        "name": "b",
        "role": "scalar",
        "c_type": "float"
      },
      {
        "name": "queue",
        "role": "queue",
        "c_type": "cnrtQueue_t"
      }
    ],
    "zero_size_behavior": "launcher does not launch"
  },
  "math": {
    "pattern": "elementwise",
    "iteration_domain": ["0 <= i < N"],
    "expression": "y[i] = max(a * x[i] + b, 0.0f)",
    "parallel_axes": ["N"],
    "reduction_axes": [],
    "broadcasts": ["a and b are scalar"],
    "write_multiplicity": "exactly_once",
    "empty_reduction_identity": null,
    "read_footprints": {"x": "x[i]"}
  },
  "numerics": {
    "input_dtype": "float32",
    "accumulation_dtype": "float32",
    "output_dtype": "float32",
    "atol": 1e-6,
    "rtol": 1e-6,
    "nan_policy": "follow requirement",
    "inf_policy": "follow requirement",
    "determinism": "required",
    "approximate_intrinsics_allowed": false
  },
  "io_shapes": {
    "x": {
      "type": "input",
      "dtype": "float32",
      "axis": ["N"],
      "shape": [257],
      "strides": [1],
      "contiguity": [true]
    },
    "y": {
      "type": "output",
      "dtype": "float32",
      "axis": ["N"],
      "shape": [257],
      "strides": [1],
      "contiguity": [true]
    }
  },
  "tests": {
    "correctness": [
      {"id": "C01", "symbols": {"N": 1}, "values": "fixed_edge_values", "purpose": "minimum"},
      {"id": "C02", "symbols": {"N": 257}, "values": "seeded_random", "purpose": "tile_tail"}
    ],
    "performance": [
      {"id": "P01", "symbols": {"N": 16777216}, "values": "seeded_random"}
    ],
    "seed": 20260813
  },
  "existing_bangc": {
    "present": false,
    "source_path": null,
    "kernels": [],
    "launches": [],
    "memory_spaces": [],
    "intrinsics": [],
    "risks": []
  },
  "build": {
    "compiler": "cncc",
    "language_standard": "c++11",
    "arch_flag": null,
    "libraries": ["cnrt", "stdc++", "m", "pthread"]
  },
  "assumptions": [],
  "blocking_questions": [],
  "ready_for_mapping": true
}
```

示例只展示结构，不得把示例 shape、容差、dtype 或测试值复制到无关算子。

## `step1_io_shapes.json`

把 `step1_base_info.json.io_shapes` 对象原样写出，不加外层包装。两个文件必须同源生成，键、顺序和值完全一致，禁止单独修改其中一个。

## 参数规则

### 指针、queue 与别名

- 只读 GDRAM 输入使用 `const T*`，output/inout 使用 `T*`。
- launcher 默认接收 device pointer 和 `cnrtQueue_t`；host pointer、workspace 或同步语义以 requirement 为准。
- 只有 aliasing 合同明确保证不重叠时，后续阶段才可添加 `__restrict__`。
- 原地允许时必须写清读取发生在覆盖之前的条件。

### shape、stride 与连续性

- shape 使用符号轴名；测试矩阵保存具体数值。
- stride 默认以元素为单位；字节 stride 必须显式标单位。
- `contiguity[i]` 只有在 stride 关系可证明时为 true：最后轴 `stride==1`，其它轴满足相邻 shape/stride 乘积。
- slice、padding、广播、负 stride 与非连续布局必须保留真实 stride，不能只写一个布尔值。
- shape/stride 乘积可能超过 32 位时记录 64 位索引要求。

### dtype 与数值合同

- 不从变量名猜 dtype；接口类型与逻辑 dtype 同时保存。
- fp16/bf16 的实际 BANG C 类型与 include 只能来自 requirement 或共享原语表。
- reduction、matmul、normalization 必须记录 accumulation dtype 与空归约 identity。
- 近似 `__bang_*` 或其它快速路径默认不允许；若用户允许，仍需保存明确误差门限。

## 数学结构分类

`math.pattern` 使用：

```text
elementwise | reduction | transpose | broadcast | gather | scatter | matmul | normalization | composite
```

- 显式/隐式归约都必须进入 `reduction_axes`；softmax 即使输入输出同 shape 也不能留空。
- scatter 或多 task 写同一地址时，`write_multiplicity` 不得写 `exactly_once`，必须说明 atomic、分段唯一性或阻塞问题。
- `compute_note` 可在 `math` 中补充 `description` 与独立的 host reference 表达式。

## 已有 BANG C 字段

若 requirement 引用了原生源码，只在非快速路径分析中记录：

```json
{
  "existing_bangc": {
    "present": true,
    "source_path": ".../Extractor/original_code.mlu",
    "kernels": [
      {
        "name": "op_kernel",
        "qualifier": "__mlu_global__",
        "parameters": ["..."],
        "task_builtins": ["taskId", "taskDim"],
        "memory_spaces": ["__nram__"],
        "copies": ["GDRAM2NRAM", "NRAM2GDRAM"],
        "intrinsics": []
      }
    ],
    "launches": [
      {
        "kernel": "op_kernel",
        "dim_expression": "cnrtDim3_t{task_count,1,1}",
        "function_type": "cnrtFuncTypeBlock",
        "queue": "queue"
      }
    ],
    "risks": []
  }
}
```

源码无法证明的 alignment、容量、arch、aliasing 或数值承诺保持 `null`/`unspecified`。

## 验证

- 两个 JSON 能被标准 parser 解析。
- 顶层存在 `operator/target/interface/math/numerics/io_shapes/tests/existing_bangc/build`。
- 每个参数名唯一，role、C type 与 dtype 不冲突。
- 每个输出在数学表达式中被定义。
- reduction 有 axis、identity 与 accumulation dtype。
- test case 的符号、dtype、layout 与 stride 完整。
- `step1_io_shapes.json` 与 `io_shapes` 完全一致。
- `arch_flag` 或容量未知时保持 null，未出现猜测值。
- blocking questions 非空时 `ready_for_mapping=false`。
- 不含 task count、tile size、NRAM 数组大小或 intrinsic 选择等后续决策。

## 失败处理

关键信息缺失时仍可输出 JSON，但列出具体问题并设置：

```json
{
  "blocking_questions": ["K=0 时 reduction 输出是什么？"],
  "ready_for_mapping": false
}
```

不得用示例、经验值或硬件营销名补足会改变接口、数值或资源设计的信息。
