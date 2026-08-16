---
name: mlu-bangc-code-gen
description: 面向 MLU590 的原生 BANG C/CNRT 算子代码生成 Skill。用于从结构化算子需求生成单文件 .mlu，实现 taskId/taskDim 映射、GDRAM 与 NRAM/WRAM/SRAM 搬运、BANG intrinsic 或保守标量计算、CNRT launch、CPU reference、精度测试与 notifier 基准，并调用 mlu-bangc-code-review 编译和修复；也支持为已有 BANG C 源码补齐测试链路。
---

# mlu-bangc-code-gen

## 目标

把 `Extractor/requirement.md` 转换为一个可由 `cncc` 独立构建和运行的原生 BANG C `.mlu` 文件。目标设备是 MLU590；具体型号、工具链版本、架构 flag、可用核心数和片上存储容量只能来自最近的 `EnvConfig/config.md` 与共享探测结果，禁止按产品名称猜测。

本 Skill 生成并验证初始 baseline，不负责穷举性能调优。保持原有分阶段编排与中间产物契约，只把语言、运行时、代码模板和测试链路替换为 BANG C/CNRT。

## 布局解析规则

所有 Skill 内部引用统一通过 `BANGC_SKILL_ROOT` 解析：

1. 如果调用方显式提供 `BANGC_SKILL_ROOT`，先将它解析为绝对路径，并验证 `share/mlu`、`mlu-bangc-main/SKILL.md`、`mlu-bangc-code-gen/SKILL.md`、`mlu-bangc-code-review/SKILL.md` 和 `mlu-bangc-optimize/SKILL.md` 均位于该根下。显式根无效时直接 blocked，不再回退猜测。
2. 未显式提供时，取当前已加载的 `SKILL.md` 所在目录的父目录，并以同样方式验证。
3. 验证失败时立即停止，报告已尝试的绝对路径；不把仓库根、`output_dir` 或当前工作目录猜为 Skill 根。

该规则同时兼容顶层为 `mlu-bangc-*`/`share` 的扁平开发布局，以及它们位于 `.claude/skills` 下的安装布局。文档中的 `{BANGC_SKILL_ROOT}/...` 必须先展开为已验证的绝对路径再读取或执行；运行 shell 示例前，将同一绝对值导出为环境变量 `BANGC_SKILL_ROOT`。

## 多阶段流程

| Stage | 名称 | 执行方式 | 进入条件 |
|---:|---|---|---|
| 0 | 输入类型检查 | 读取 `requirement.md` | START |
| 1 | ExtractBaseInfo | 调度 subagent | `is_bangc=false` |
| 2 | TraceBlockMapping | 调度 subagent | Stage 1 成功 |
| 3 | AxisFusion | 调度 subagent | Stage 2 成功 |
| 4 | GenerateSpec | 调度 subagent | Stage 3 成功 |
| 5 | GenerateCode | 调度 subagent或逐字复制 | 原生源码快速路径，或 Stage 4 成功 |
| 6 | GenTestCode | 调度 subagent | Stage 5 成功 |
| 7 | 代码验证 | 调用 `mlu-bangc-code-review` | Stage 6 成功 |
| 8 | 检查回退 | 条件跳转 | Stage 7 未通过 |
| 9 | 输出结果 | 规范化产物 | Stage 7 完成 |

## 输入与路由

调用参数：

| 参数 | 必需 | 含义 |
|---|---:|---|
| `requirement` | 是 | `Extractor/requirement.md` 的绝对或可解析路径 |
| `output_dir` | 是 | 主流程输出目录 |

从 requirement 或主 Skill 参数读取：

```text
input_type: bangc | not_bangc
is_bangc: true | false
```

- `bangc` / `is_bangc=true`：跳过 Stage 1–4，逐字节复制 `Extractor/original_code.mlu` 为 `KernelGen/step5_kernel_code.mlu`，不得借快速路径改写用户 kernel。
- `not_bangc` / `is_bangc=false`：执行 Stage 1–4 后生成新实现。
- 两个字段冲突、字段缺失但源码类型无法可靠判断、或快速路径文件不存在时停止并返回 Extractor；不得猜测。

还必须读取从 `output_dir` 向祖先查找到的最近一个 `EnvConfig/config.md`。其中 `execution_backend` 只能是 `local`、`worker` 或 `unavailable`，并决定 Stage 7 的真实编译运行位置。

平台事实只从以下共享资源读取：

```text
{BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md
{BANGC_SKILL_ROOT}/share/mlu/references/primitives.md
```

## 输出

所有文件写入 `{output_dir}/KernelGen/`：

```text
step1_base_info.json
step1_io_shapes.json
step2_block_mapping.json
step3_axis_fusion.json
step4_code_spec.json
step5_kernel_code.mlu
step6_test_code.mlu
bangc_code_fix.mlu
bangc_report.md
```

快速路径允许 Stage 1–4 的 JSON 标记为 `skipped`，但 Stage 5–7 和两个固定交接文件不能省略。

- `bangc_code_fix.mlu`：经 code-review 处理的完整候选，包含 device kernel、CNRT launcher、CPU reference、测试与 benchmark。
- `bangc_report.md`：环境、编译、精度、运行和 baseline 性能事实；无法执行的字段写 `unavailable` 或 `N/A（原因）`。

## 全局生成约束

1. 生成单一自包含 `.mlu` translation unit，不生成 Python/JIT 包装器、CMake 工程或多文件库。
2. 设备侧包含 `<bang.h>`，host runtime 包含 `<cnrt.h>`；不要假设 `bang.h` 固定在 `$NEUWARE_HOME/include`，让已配置的 `cncc` 查找其资源头。
3. kernel 入口使用目标工具链实际支持的 `__mlu_global__`；只有输入源码或本地头文件明确使用 `__mlu_entry__` 时才保留该形式。
4. baseline 优先用扁平 `taskId`/`taskDim` 覆盖逻辑 tile；`taskIdX/Y/Z`、`clusterId`、`coreId` 等只能在共享规则确认后使用。
5. GDRAM 与片上存储之间使用方向明确、字节数正确的 `__memcpy`；尾块不得越界，不能把元素个数误当字节数。
6. `__nram__`、`__wram__`、`__sram__` 的每个数组都要给出字节公式。容量或对齐未获环境证据时使用保守方案并留给真实编译验证，不写伪造的 MLU590 常数。
7. `__bang_*` 只使用共享原语表或目标环境头文件已确认的签名、dtype、长度与对齐约束；不确定时生成语义明确的标量/分块 baseline，不猜 intrinsic 名称。
8. device pointer 生命周期由测试程序管理；业务 launcher 接受 device pointer 与 `cnrtQueue_t`，不隐藏 H2D/D2H、分配和无条件同步。
9. host 侧使用 `cnrtSetDevice`、queue、allocation、copy、notifier 等 CNRT API，并检查所有返回值与 `cnrtQueueSync` 的异步错误。
10. 不重定义 SDK 的 `CNRT_CHECK`，不生成旧常量 `CNRT_RET_SUCCESS`。如需自定义辅助，使用不会冲突的名称并仅引用当前 `cnrt.h` 已确认的返回值。
11. 架构参数必须逐字采用 EnvConfig 已确认的 flag；EnvConfig 未给出时使用 `cncc` 默认并在报告写明，禁止由“MLU590”推断 `--bang-mlu-arch`。
12. 默认保持严格数值语义；近似 intrinsic、改变累加 dtype、倒数乘法等只能由需求允许且经过精度测试后采用。

## 分阶段执行

一次只调度一个 subagent，等待产物生成并完成交接检查后再继续。主 Skill 不接管子步骤的设计。

### Stage 1：ExtractBaseInfo

读取：

```text
{BANGC_SKILL_ROOT}/mlu-bangc-code-gen/subagents/ExtractBaseInfo.md
```

输入 `requirement.md`；输出 `step1_base_info.json` 和同源的 `step1_io_shapes.json`。抽取接口、C 类型/dtype、shape/stride/layout、数学合同、容差、测试矩阵、已有 BANG C 结构和未决问题，不设计 tile 或 task 数。

### Stage 2：TraceBlockMapping

读取 `subagents/TraceBlockMapping.md`。输入 Stage 1 产物，输出 `step2_block_mapping.json`。设计逻辑输出 tile 到 `taskId/taskDim` 的覆盖关系、GDRAM offset、tail、归约归属和整数宽度，并证明无遗漏、无意外重叠。

### Stage 3：AxisFusion

读取 `subagents/AxisFusion.md`。输入 Stage 1–2 产物，输出 `step3_axis_fusion.json`。分析连续轴线性化、DMA 段、NRAM tile、广播复用、转置重排以及有依据的 `__bang_*` 候选。未知对齐或容量不得强行向量化。

### Stage 4：GenerateSpec

读取 `subagents/GenerateSpec.md`。输入 Stage 1–3 产物，输出 `step4_code_spec.json`。固定 headers、kernel 签名、task mapping、片上 buffer、搬运、计算、store、CNRT launch、build、reference、测试和 notifier 计时规格。不得留下 TODO 或“视情况”。

### Stage 5：GenerateCode

正常路径读取 `subagents/GenerateCode.md`，生成 `step5_kernel_code.mlu`，至少包含：

- 标准头、`bang.h`、`cnrt.h`
- 必要且不与 SDK 冲突的错误辅助
- `__mlu_global__` kernel 与 device helper
- `taskId/taskDim` 覆盖循环
- NRAM/WRAM/SRAM 声明与 GDRAM 搬运
- CNRT queue launcher
- `// === BANGC_TEST_HARNESS_BEGIN ===` 标记

快速路径只复制 `original_code.mlu`。Stage 5 不运行 `cncc`。

### Stage 6：GenTestCode

读取 `subagents/GenTestCode.md`，生成 `step6_test_code.mlu`。保持 Stage 5 kernel 与 launcher 原文，补齐：

- 独立 CPU reference
- 确定性输入生成
- CNRT device allocation 与 H2D/D2H
- 多 shape/dtype/tail 正确性用例
- `atol + rtol * abs(expected)` 比较与失败诊断
- warmup 与 CNRT notifier kernel-only benchmark
- `host_reference_ms` 与 `original_bangc_ms`
- 任一失败时非零退出码

Stage 6 只生成源码；统一由 Stage 7 编译运行。

### Stage 7：调用 code-review

```python
Skill(
    skill="mlu-bangc-code-review",
    args="{absolute_output_dir}/KernelGen/step6_test_code.mlu"
)
```

code-review 负责静态检查、按 EnvConfig 选择 local/Worker、运行 `cncc`、执行 binary、修复并重新验证，并固定写入同一 `KernelGen/` 下的 `bangc_code_fix.mlu` 与 `bangc_report.md`。主 Skill 必须读取报告末尾的 `passed`、`blocked`、`target_verified`、`compile_pass`、`accuracy_pass`、`final_code_path`；不能仅凭文件存在判定成功。确认两个产物存在且非空后：

1. 将最终 fix 源码逐字节规范化为 `KernelGen/bangc_code_fix.mlu`。
2. 将真实 review 事实规范化为 `KernelGen/bangc_report.md`。
3. 原样保留 `passed`、`blocked`、`target_verified`、`compile_pass`、`accuracy_pass`、所有 gate、命令和诊断；不得把未运行或失败改为成功。

## 编译协议

使用 EnvConfig 中已经验证的 `cncc`、`neuware_root`、库路径和完整架构 flag。若 `neuware_root` 不是 `N/A`，执行编译命令前必须将其原样导出为 `NEUWARE_HOME`：

```bash
export NEUWARE_HOME="<EnvConfig.neuware_root>"
```

通用形式为：

```bash
cncc step6_test_code.mlu -o step6_test_code \
  -I"${NEUWARE_HOME}/include" -L"${NEUWARE_HOME}/lib64" \
  -lcnrt -lstdc++ -lm -lpthread -std=c++11
```

- 仅当 EnvConfig 给出已验证的完整 arch flag 时追加它。
- `-I.../include` 用于 `cnrt.h` 等 SDK 公共头；不得以 `include/bang.h` 是否存在作为 NeuWare 根的硬条件。
- 使用 C++ 标准库时保留 `-lstdc++ -lm -lpthread`，避免仅链接 `-lcnrt`。
- 源码实际需要其它库时才追加；禁止为绕过错误堆叠无依据 flag。
- 记录完整命令、stdout、stderr 与退出码。

## 正确性协议

- CPU reference 独立表达数学合同，不复制 task/tile 索引实现。
- 浮点归约默认使用更高精度 host 累加，最终按输出 dtype 比较；如合同规定固定顺序则遵守合同。
- 浮点比较默认采用 `abs(actual-expected) <= atol + rtol*abs(expected)`，并按需求处理 NaN、Inf 与 signed zero；整数精确比较。
- 失败打印 case id、首个失败逻辑坐标、expected、actual、abs/rel error，并返回非零。
- 每个 case 检查 CNRT allocation/copy/queue/notifier API，确保所有资源在失败路径也释放。

## 性能协议

1. correctness 全部通过后才 benchmark。
2. 输入、allocation 与 H2D 在计时外完成。
3. warmup 后 queue sync。
4. start notifier → 重复 launcher → end notifier → queue sync。
5. 用 `cnrtNotifierDuration` 的目标 SDK 实际单位换算并除以迭代次数。
6. 报告 warmup、iterations、shape、统计范围与实际逻辑 bytes。
7. `original_bangc_ms` 只代表生成 baseline 的 device work；`host_reference_ms` 单独报告，二者不得混成 kernel-only 指标。

## `bangc_report.md` 固定字段

```markdown
# MLU BANG C Code Generation Report

## Environment
- execution_backend: local | worker | unavailable
- passed: true | false
- blocked: true | false
- target_verified: true | false
- device: ...
- cncc: ...
- neuware: ...
- arch_flag: ... | N/A（未确认，使用编译器默认）
- build_command: ...

## Build
- compile_pass: true | false | unavailable
- link_pass: true | false | unavailable
- executable: ...
- compiler_diagnostics: ...

## Accuracy
- accuracy_pass: true | false | unavailable
- atol: ...
- rtol: ...
- max_abs_error: ...
- max_rel_error: ...
- test_cases: ...

## Runtime safety
- cnrt_api_errors: none | ... | unavailable
- queue_sync_errors: none | ... | unavailable
- memory_check: passed | failed | not_run（原因）

## Baseline performance
- timing_method: CNRT notifier | unavailable
- warmup: ...
- iterations: ...
- host_reference_ms: ... | N/A
- original_bangc_ms: ... | N/A
- original_bangc_gbps: ... | N/A
- scope: kernel_only | wrapper_device_work | unavailable

## Design
- kernel: ...
- launch_dim: ...
- function_type: ...
- mapping: ...
- nram_bytes: ...
- wram_bytes: ...
- sram_bytes: ...

## Artifacts
- generated: KernelGen/step6_test_code.mlu
- reviewed: KernelGen/bangc_code_fix.mlu

## Final status
- passed: true | false
- blocked: true | false
- target_verified: true | false
- compile_pass: true | false | unavailable
- accuracy_pass: true | false | unavailable
- final_code_path: <absolute path to KernelGen/bangc_code_fix.mlu> | N/A
```

## 回退规则

| 失败阶段 | 允许的回退 |
|---|---|
| Stage 0 类型冲突 | 返回 Extractor，不能猜输入类型 |
| Stage 1 信息缺失 | 返回 requirement/Extractor，不能猜接口 |
| Stage 2 覆盖证明失败 | 重做 task mapping |
| Stage 3 融合、DMA 或 intrinsic 依据不足 | 使用保守未融合、标量/tail 路径 |
| Stage 4 片上容量无法证明 | 缩小 tile 或等待环境事实，不写硬件常数 |
| Stage 5 结构不完整 | 根据 spec 重生成 |
| Stage 6 harness 不可构建 | 修复 harness，不删除失败用例 |
| code-review 不收敛 | 保留最后候选与真实失败报告，`passed=false` |
| 环境 unavailable | 输出静态候选，动态字段全部标 unavailable |

每阶段最多内部重试 3 次；不得通过缩小测试 shape、放宽容差或删除检查来伪造通过。

## 完成检查

- [ ] 九个固定产物存在；快速路径的四个 JSON 可明确标记 skipped。
- [ ] 非快速路径四个 JSON 可由标准 parser 解析并相互追踪。
- [ ] `.mlu` 只有一套 kernel/launcher，包含 CPU reference、测试与 notifier benchmark。
- [ ] 未硬编码或猜测 MLU590 arch、核心数、NRAM/WRAM/SRAM 容量。
- [ ] 未重定义 `CNRT_CHECK`，未使用 `CNRT_RET_SUCCESS`。
- [ ] 编译链接包含源码所需的 C++/math/thread 库。
- [ ] 精度或运行失败必然返回非零。
- [ ] `bangc_code_fix.mlu` 与 `bangc_report.md` 已形成主流程交接。

## 与其他 Skill 的关系

- 上游：`mlu-bangc-main` 与其 Extractor/EnvConfig。
- 验证：`mlu-bangc-code-review`。
- 后续：`mlu-bangc-optimize`，输出 `Optimizer/bangc_optimized.mlu` 与 `bangc_optimized.md`。
- 主流程最终交接：`bangc_final.mlu`。
