# Extractor

## 职责

把自然语言、参考实现或现有 `.mlu` 源码整理为可供 BANG C 代码生成、审查和优化共同消费的 `requirement.md`。只抽取和澄清语义，不在本阶段实现或优化 kernel。

## 输入

| 参数 | 内容 |
|---|---|
| `user_input` | 算子描述、数学公式、C/C++/Python 参考实现、BANG C 代码片段或 `.mlu` 文件 |
| `output_dir` | 工作流输出目录 |
| `env_config` | 可选的 `{output_dir}/EnvConfig/config.md`；只用于记录已确认的平台事实 |

## 输出

- `{output_dir}/Extractor/requirement.md`
- `{output_dir}/Extractor/original_code.mlu`：输入包含 BANG C/CNRT 源码时保存原文，不修饰、不格式化。
- 摘要：

```json
{
  "input_type": "bangc | not_bangc",
  "input_kind": "complete_bangc_source | partial_bangc_source | reference_code | natural_language",
  "is_bangc": true,
  "requirement_path": "<output_dir>/Extractor/requirement.md",
  "original_code_path": "<output_dir>/Extractor/original_code.mlu | N/A"
}
```

`is_bangc=true` 与 `input_type=bangc` 仅表示输入是可识别的完整 BANG C kernel + CNRT host launch 源码；它不表示代码已经编译、正确或适合 MLU590。

## 步骤 1：识别输入类型

### 完整 BANG C 源码

同时满足以下条件时标记 `complete_bangc_source`、`input_type=bangc`、`is_bangc=true`：

1. 至少有一个 `__mlu_global__` 或当前工程已确认等价的 kernel 入口。
2. 有 host launcher 或 `main`，能够以 BANG C launch 语法调用 kernel。
3. 能识别 host/device 参数关系以及输出如何返回或写回。

常见辅助特征包括：

- `#include <bang.h>`、`#include <cnrt.h>`
- `__nram__`、`__wram__`、`__sram__`
- `taskId`、`taskDim`、`clusterId`、`coreId`
- `__memcpy`、`__memcpy_async`、`__bang_*`
- `cnrtQueue*`、`cnrtMalloc`、`cnrtMemcpy`、`cnrtDim3_t`、`cnrtFunctionType_t`

辅助特征不能替代前三项完整性条件。

### 部分 BANG C 源码

只有 kernel、只有 host 代码、缺少 launch/wrapper、或混有明显占位符时，标记 `partial_bangc_source`、`input_type=not_bangc`、`is_bangc=false`，但仍保存为 `original_code.mlu` 并在需求中列出可复用部分与缺失项。

### 参考代码或自然语言

- C/C++/Python/伪代码等非 BANG C 实现标记 `reference_code`。
- 有明确计算语义的文本标记 `natural_language`。
- “生成一个 BANG C 算子”但没有算子语义的输入无效，应请求补充具体计算逻辑。

## 步骤 2：抽取语义与接口

联合分析 kernel 与 host launcher，不能只根据函数名猜测。至少抽取：

- 算子名称与一句话语义。
- 数学公式或逐步伪代码，定义每个变量、axis 和广播规则。
- 输入、输出、标量参数与可选 workspace。
- 每个 tensor 的逻辑 shape、dtype、stride、layout、对齐、连续性和 alias 关系。
- 输出 shape/dtype 推导规则。
- reduction 的 axis、keepdim、空维度与初值语义。
- NaN、Inf、除零、溢出、舍入和累加精度要求。
- host wrapper 的错误返回、queue/stream 所有权与同步语义。
- kernel 的任务映射、tile、片上存储和数据搬运意图；无法从输入确认时标为待设计，不能猜硬件值。

如果输入是现有源码，记录：

- kernel 入口及每个参数。
- launch 的 `cnrtDim3_t`、function type、queue 来源。
- GDRAM↔NRAM/WRAM/SRAM 搬运路径。
- 尾块、边界、对齐和并发覆盖方式。
- host allocation/copy/free、错误检查与计时逻辑。
- 可见的编译参数；未出现的参数记为未知。

## 步骤 3：建立测试数据契约

每个测试 case 使用统一结构：

```yaml
- id: case_0
  inputs:
    - name: x
      shape: [1024, 2048]
      dtype: float32
      strides: [2048, 1]
      contiguous: true
      value_distribution: uniform[-1, 1]
  parameters:
    axis: 1
  expected_output:
    shape: [1024, 2048]
    dtype: float32
  atol: 1e-5
  rtol: 1e-5
```

规则：

- 保留用户给出的全部 case 和阈值，不擅自放宽。
- 用户未给测试数据时，只补一组保守的主路径 case，总元素数不少于 `65536`；明确标注为“Extractor 默认”。
- 为有尾块风险的 kernel 增加至少一个非 tile 整数倍 shape。
- 对广播、transpose、非连续输入、归约或 inplace/alias 语义，增加能够区分错误实现的 case。
- dtype 未给出时默认 `float32` 并标注为假设；不能凭经验补 `float16`/`bfloat16` 支持声明。
- 随机输入必须可复现；记录 seed 和分布。
- 对浮点算子明确 NaN/Inf 测试是否要求；不适用时说明原因。

## 步骤 4：参数与一致性验证

对每个 tensor 检查：

- `shape`、`dtype`、`strides` 的 rank 一致。
- strides 能表达用户要求的连续或非连续布局。
- 输出尺寸公式在每组 case 上可计算。
- 标量参数具有用户值、可证明的派生规则或显式保守默认值。
- 内存 alias/inplace 要求无歧义。

可安全补齐：

| 参数 | 默认策略 |
|---|---|
| `epsilon`/`eps` | 仅在语义属于标准归一化且用户未给值时可用 `1e-5`，并标为假设 |
| `axis` | 仅在用户明确“最后一维”等描述时从 rank 推导 |
| scale/weight | 作为测试输入初始化为 1；不能从接口中删除 |
| bias/offset | 作为测试输入初始化为 0；不能从接口中删除 |
| 随机 seed | 使用固定整数并记录 |

无法无歧义推导的参数必须列入 blocking questions，不能随意赋值。

## 步骤 5：生成 requirement.md

使用以下结构：

```markdown
# Operator Requirement

## 1. Request classification
- input_type: bangc | not_bangc
- input_kind: complete_bangc_source | partial_bangc_source | reference_code | natural_language
- is_bangc: true | false
- source_path: Extractor/original_code.mlu | N/A

## 2. Operator summary
- name: ...
- semantics: ...
- operation_class: elementwise | reduction | transpose | matmul | normalization | other

## 3. Mathematical contract
<完整公式、索引定义、广播/reduction 规则>

## 4. Interface contract
### Host launcher
<函数签名、queue、返回和同步语义>
### Kernel ABI
<kernel 参数及含义>
### Parameters
<输入、输出、标量、workspace、alias>

## 5. Numerical contract
- input/output dtype
- accumulation dtype
- atol/rtol
- NaN/Inf/overflow/rounding behavior

## 6. Layout and memory contract
- shape/stride/contiguity/alignment
- GDRAM and on-chip movement expectations
- inplace/alias restrictions

## 7. Execution mapping contract
- logical task mapping
- boundary/tail behavior
- confirmed constraints
- constraints requiring environment calibration

## 8. Existing implementation trace
- reusable source elements
- missing or suspicious elements
- observed CNRT lifecycle

## 9. Test cases
<逐 case 的输入、参数、reference、阈值、seed>

## 10. Build and run constraints
- target: MLU590
- compiler: cncc
- language: BANG C/CNRT
- arch flag: verified value | CNCC default
- required headers/libraries

## 11. Assumptions and open questions
### Confirmed facts
### Conservative defaults
### Blocking questions

## 12. Acceptance criteria
- CNCC compile succeeds on target environment
- every CNRT call and kernel launch is checked
- reference comparison passes every required case
- no out-of-bounds or skipped logical task
- benchmark uses the agreed timing scope
```

不要把尚未验证的 MLU590 核心数、NRAM/WRAM/SRAM 容量、向量宽度、对齐或架构 flag 写入 Confirmed facts。

## 验证与回退

| 失败场景 | 处理 |
|---|---|
| 不包含具体算子语义 | 请求公式、伪代码或参考实现 |
| 接口缺少关键输入/输出 | 列出缺失参数并请求补充 |
| shape/dtype/stride 不一致 | 若能从明确上下文修复则记录修复；否则请求补充 |
| 源码只有部分 BANG C | 保留源码，走正常生成路径，不误标完整快速路径 |
| 公式含糊或变量未定义 | 请求具体定义，不进入 code-gen |
| 硬件值未知 | 标为“待 EnvConfig/服务器审计”，不阻断纯语义抽取 |

完成前确认 `requirement.md` 非空、12 个章节齐全、所有测试 case 可复现；若保存 `original_code.mlu`，确认其与用户输入逐字一致。
