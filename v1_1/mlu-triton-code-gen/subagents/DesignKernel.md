# DesignKernel

## 职责

在一个连续上下文中完成原六阶段链路的 Step 1-4：结构提取、块索引映射、轴融合、代码
规范。你不是总调度器，不得创建其他代理，也不生成 Python 代码。

连续完成四步的目的，是让需求和平台规则只加载一次，同时仍保留每一步的落盘检查点。

## 输入

调用消息必须提供以下绝对路径：

| 参数 | 用途 |
|---|---|
| `requirement_path` | Extractor 生成的 `requirement.md` |
| `kernelgen_dir` | 五个 JSON 产物的目录 |
| `artifact_contract` | JSON 字段与一致性契约 |
| `primitives_path` | MLU 生成阶段原语清单 |
| `platform_rules_path` | MLU Grid、NRAM、设备和后端规则 |

只读取这些文件。不要加载旧的细粒度角色文档或示例目录；只有当前需求无法确定某个字段时，
才回到 `requirement_path` 查证，禁止凭空补值。

## 输出

依次原子写入：

1. `{kernelgen_dir}/step1_base_info.json`
2. `{kernelgen_dir}/step1_io_shapes.json`
3. `{kernelgen_dir}/step2_block_mapping.json`
4. `{kernelgen_dir}/step3_axis_fusion.json`
5. `{kernelgen_dir}/step4_code_spec.json`

所有 JSON 使用 UTF-8、合法双引号、无注释。只在回复中返回 `completed` 或 `failed` 和文件
路径，不粘贴 JSON 正文。

## 执行顺序

### 1. 结构提取

从需求中提取算子名、接口、计算公式、参考 PyTorch 语义、输入输出实际 shape、逻辑 axis、
物理连续性和显式/隐式归约轴。

- `compute_type` 只能是 `reduction`、`elementwise`、`matmul`、`normalization`、`others`。
- `shape` 存实际数值或需求中的符号值，不写 block size。
- `axis`、`shape`、`contiguity` 的长度对每个 tensor 必须相同。
- slice、transpose、非单位 stride 必须反映在 `contiguity`。
- softmax、layernorm 等即使需求没写 reduce，也要从公式识别隐式归约轴。
- `compute_note.description` 与 `compute_note.torch_impl` 必须表达同一语义。

先生成一个内存中的 `io_shapes` 对象，并将同一对象分别写入
`step1_base_info.json["io_shapes"]` 和 `step1_io_shapes.json`，禁止二次独立推导。

### 2. 块索引映射

根据 Step 1 内存对象为每个输入/输出指针生成 `io_block_mapping`：

- 非归约轴写入 `block_name` 与 `axis_size`；
- 归约轴写入 `reduce_dim` 与 `reduce_size`；
- 保留对应 tensor 的 `contiguity`；
- `compute_formula` 和 `compute_note` 必须从 Step 1 原样透传。

`block_name` 表示 kernel tile 参数，不等于实际 shape；`axis_size` / `reduce_size` 保留需求
中的真实候选尺寸。

### 3. 轴融合

遍历所有相邻逻辑轴对。只有该轴对在所有相关输入和输出上都连续，并且不跨越 transpose、
reduce 或会改变语义的 broadcast 边界时，才允许融合。

- 每个相邻轴对都要判断，不能因为发现一组可融合轴就提前结束。
- 融合后同步更新所有相关指针的 `block_name`、`axis_size` 和索引含义。
- 不可融合时写明阻断它的 tensor/边界。
- `compute_formula`、`compute_note` 和未改变的映射字段原样透传。

### 4. 代码规范

读取 `primitives_path` 和 `platform_rules_path` 一次，根据融合后的映射生成 kernel/wrapper
规范：

- `kernel.block_params`：每个 BLOCK 的候选尺寸数组；
- `kernel.aux_params`：`program_id`、offset、`tl.arange` 等公共索引；
- `kernel.loads` / `kernel.stores`：每个指针独立的 index 和边界 mask；
- `kernel.compute`：公式和说明；
- `wrapper.grid` 与 `wrapper.block_params`。

归约规则：

- 归约轴优先在 kernel 内部循环，不直接扩张为额外 grid 维；
- 简单可结合归约优先 `delayed_block_reduction`：循环中保留 reduce-block 维度，循环后只做
  一次块内归约；
- 每个 chunk 必须立即归约或资源风险明确时才用 `inline_block_reduction`；
- 多遍算法用 `reduce_loop_pass1`、`reduce_loop_pass2` 等，每遍只描述单一归约目标；
- 所有原语与 dtype 必须在 `primitives_path` 支持范围内；Grid、NRAM 和设备行为遵守
  `platform_rules_path`。

## 每步闸门

写下一步前执行以下检查。任一失败先在当前上下文内修正一次；仍失败则停止，不输出伪造的
后续文件。

| 闸门 | 条件 |
|---|---|
| Step 1 | 必填字段齐全；三数组等长；两个 io_shapes 完全相等 |
| Step 2 | 所有指针可追溯到 Step 1 tensor；语义字段原样透传 |
| Step 3 | fusion_note 有依据；映射与融合结果一致 |
| Step 4 | kernel/wrapper 必填字段齐全；原语、dtype、Grid 规则合规 |

精确字段结构以 `artifact_contract` 为唯一准绳；本文件不重复维护整份 JSON 模板。
