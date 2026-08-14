# MLU590 BANG C/CNRT 平台规则

仅在目标是 MLU590 且实现语言是 BANG C/CNRT 时读取本文件。语义抽取、代码生成、审查和优化仍由各自主 Skill 负责；本文件集中保存平台边界，避免在多个模块中重复或互相矛盾。

## 目录

- 环境与证据
- 源码与编译
- CNRT host 生命周期
- BANG C 执行映射
- 存储层次与搬运
- 边界、对齐与数值
- 性能测量与优化
- 第一版待校准项

## 环境与证据

1. 动态执行前读取 `{output_dir}/EnvConfig/config.md`。
2. `execution_backend=local` 时直接运行；`worker` 时通过 `.claude/skills/mlu-bangc-main/subagents/scripts/submit_task_to_worker.py` 前台同步提交。
3. 环境探测只使用 `.claude/skills/share/mlu/runtime/get_device_info.py` 与 `test_env_code.py`；smoke source 是同目录 `bangc_vector_add.mlu`。
4. 只有真实 MLU590 上的 CNCC 编译、binary 执行和 reference 对比全部成功，才能写 `target_verified=true`。
5. 静态审查不能替代编译；编译不能替代运行；运行成功不能替代精度；一次耗时不能替代稳定 benchmark。
6. 记录设备原始型号、CNCC/CNRT 版本、编译命令、环境变量来源和 stdout/stderr。未检测项写 `N/A` 或 `UNAVAILABLE`。

## 源码与编译

标准 BANG C/CNRT 单文件实现包含：

```cpp
#include <bang.h>
#include <cnrt.h>

__mlu_global__ void op_kernel(/* GDRAM pointers and scalars */) {
  // task mapping, explicit data movement, vector/scalar computation
}

int launch_or_main(/* host parameters */) {
  // device, queue, allocation, copies, launch, sync, validation, cleanup
}
```

基础编译形态：

```bash
cncc input.mlu -o input \
  -std=c++11 \
  -I"${NEUWARE_HOME}/include" \
  -L"${NEUWARE_HOME}/lib64" \
  -lcnrt -lstdc++ -lm -lpthread \
  ${BANGC_ARCH_FLAG:-} ${BANGC_CNCC_EXTRA_FLAGS:-}
```

规则：

- `NEUWARE_HOME`/`CNTOOLKIT_HOME` 是候选根，不把某台机器的绝对安装路径写入生成代码。
- `cnrt.h` 通常位于 SDK include；`bang.h` 可能位于 include，也可能由 CNCC 从 `lib/clang/<version>/include` 解析。根目录探测不得硬要求 `include/bang.h`。
- 使用 C++ 标准库、数学库或线程支持时显式链接 `-lstdc++ -lm -lpthread`；始终链接 `-lcnrt`。
- 不根据 `MLU590` 名称猜 `--bang-mlu-arch` 或 `--bang-arch`。优先使用完整且已验证的 `BANGC_ARCH_FLAG`；未提供时允许 CNCC 默认。
- `BANGC_CNCC_EXTRA_FLAGS` 只承载用户或环境已确认的附加参数。
- 基线与候选必须使用同一编译参数。调试构建和性能构建不得混为一组数据。
- 如需 `-S`、`-save-temps`、`-emit-llvm` 等中间产物参数，先检查当前 `cncc --help`；不支持时标 `UNAVAILABLE`，不得猜参数。

## CNRT host 生命周期

典型顺序：

1. `cnrtSetDevice` 选择目标设备。
2. `cnrtQueueCreate` 创建 queue。
3. `cnrtMalloc` 分配 device buffer。
4. `cnrtMemcpy(..., cnrtMemcpyHostToDev)` 上传输入。
5. 构造 `cnrtDim3_t` 与 `cnrtFunctionType_t`，在指定 queue 上 launch kernel。
6. `cnrtQueueSync` 等待完成并暴露异步错误。
7. `cnrtMemcpy(..., cnrtMemcpyDevToHost)` 取回输出。
8. 完成 reference 对比。
9. `cnrtFree` 与 `cnrtQueueDestroy` 清理资源。

要求：

- 检查每个返回 `cnrtRet_t` 的调用；错误日志至少包含调用位置与数值返回码。
- 不自定义名为 `CNRT_CHECK` 的宏，避免与 SDK 头文件冲突。使用项目私有名，按返回码非零判失败，不依赖版本特定的 `CNRT_RET_SUCCESS`/`cnrtSuccess` 拼写。
- 明确 queue 的创建、传入、同步和销毁所有权；库式 wrapper 不应擅自销毁调用方 queue。
- 只有 correctness 需要时同步；benchmark 使用 notifier/event 等设备计时并保持同一 queue。
- 任一失败路径都不得继续使用未初始化输出。尽可能释放已成功分配的资源。
- 不用 PyTorch/Triton 设备 tensor 或 Python JIT 作为 BANG C/CNRT 实现的运行依赖。

## BANG C 执行映射

- `taskId`/`taskDim` 及其 X/Y/Z 变体表达逻辑任务编号与任务总数；使用前确认当前编译器支持的具体内建变量。
- `clusterId`、`coreId`、`coreDim` 等只在确有需要且当前头文件/样例确认时使用。
- 把输出空间划分为互不重叠的逻辑任务。每个 task 必须覆盖完整 tile 或经边界处理的尾块。
- 当 launch task 数小于逻辑 tile 数时，使用 task-stride 循环：起点来自 `taskId`，步长来自 `taskDim`；证明没有遗漏和重复写。
- 不把 Triton 的 `tl.program_id`、`tl.num_programs`、`BLOCK_SIZE`、`num_warps` 或 `num_stages` 直接翻译成同名概念。重新表达为 BANG C task 数、tile 元素数、function type 与流水阶段。
- `cnrtFuncTypeBlock` 可作为最小 smoke test 的保守 function type；生成算子的 Block/Union 选择必须结合算法、SDK 支持与真机结果。
- taskDim/grid 各维上限、cluster/core 数不得硬编码；从环境、SDK 查询或服务器审计中取得。

## 存储层次与搬运

常见地址空间：

| 空间 | 典型角色 | 规则 |
|---|---|---|
| GDRAM | host 分配后供设备访问的全局数据 | 合并连续访问，避免不必要往返 |
| NRAM (`__nram__`) | 每个计算核心的主要片上 tile | 预算输入、输出、中间值、对齐和双缓冲总和 |
| WRAM (`__wram__`) | 特定计算的数据/权重暂存 | 仅在算法与 intrinsic 确有需要时使用 |
| SRAM (`__sram__`) | cluster 范围共享/交换 | 明确同步、所有权与冲突；不可当作普通 NRAM |

搬运规则：

- 使用 `__memcpy` 的 GDRAM2NRAM、NRAM2GDRAM 等方向时，源/目的地址空间和字节数必须匹配。
- 字节数统一以 `element_count * sizeof(dtype)` 推导，防止把元素数当字节数。
- 尾块不能让 DMA 越界。可使用安全的实际长度，或在证明 API 对齐约束后采用 padding/分段策略。
- `__memcpy_async`、流水和双缓冲只有在当前 SDK 支持并且同步依赖正确时使用。
- 不在第一版硬编码 MLU590 的 NRAM/WRAM/SRAM 容量；从环境事实或编译错误取得限制。
- tile 预算应包含所有同时存活的片上 buffer，而不是只统计一个输入。
- 减小 tile 可解决片上容量溢出但会增加任务/搬运开销；增大 tile 必须重新验证边界、容量和性能。

## 边界、对齐与数值

- 对任意 shape，证明 `0 <= global_index < logical_size` 后才读写。
- 多维 tensor 使用显式 stride 计算偏移；不能把非连续输入当作连续数组。
- 转置、广播和 reduction 同时检查输入越界、输出覆盖、重复写与 race。
- intrinsic 的长度、地址对齐和 dtype 约束以当前 `bang.h` 与官方样例为准；不凭名称推导。
- 低精度输入可用更高精度累加，但必须符合 requirement 的输出和舍入语义。
- 保持用户 `atol`/`rtol`。fast/approximate intrinsic 只有通过完整 case 后才可保留。
- reference 默认在 host 上以清晰标量/标准库逻辑实现；reference 与 kernel 不得共享同一错误索引公式。
- NaN/Inf/除零/溢出语义按 requirement 验证；不能通过过滤异常值制造通过结果。

## 原语和数学函数

- 生成或审查 BANG C intrinsic 前读取 `.claude/skills/share/mlu/references/primitives.md`。
- 数学函数、fast/approximate 变体和 dtype 支持读取 `.claude/skills/share/mlu/references/libdevice.md`。
- 性能替换策略读取 `.claude/skills/share/mlu/optimize/libdevice-opt.md`。
- reference 文件中的清单不是头文件替代品。若清单与当前 `bang.h`/CNCC 编译结果冲突，以当前目标环境证据为准并记录差异。

## 性能测量与优化

- correctness 通过后才测性能。
- kernel-only 计时优先使用同一 queue 上的 CNRT notifier；记录 warmup、重复次数、统计量和计时单位。
- host end-to-end 时间与 kernel-only 时间分开报告；不得用前者冒充后者。
- 基线和候选保持相同输入、queue、数据驻留、编译参数与计时范围。
- CNPerf 可用时读取 `.claude/skills/share/mlu/perf-analyzer/analyzer.sh` 的原始报告；解析器仅作辅助，原始报告是最终证据。
- 优化顺序通常为：正确任务映射 → 连续/合并搬运 → 合理 tile → 减少中间 buffer → 适用 intrinsic → 流水/双缓冲。每步独立复测。
- 如果性能未改善或精度退化，回退到上一份已验证 best-so-far。

## 第一版待服务器校准项

以下信息不得在第一版写死；服务器审计后再更新：

1. CNCC、CNRT、NeuWare/CNToolkit 与驱动的确切版本和安装布局。
2. MLU590 对应的确切 CNCC architecture flag 及是否必须显式传入。
3. cluster/core 数、taskDim 各维限制及 Block/Union function type 的适用范围。
4. NRAM/WRAM/SRAM 容量、保留空间、对齐粒度和 DMA 长度约束。
5. `bang.h` 中可用 `__bang_*`、异步搬运、同步原语及 dtype 签名。
6. CNRT queue/notifier/错误 API 的当前拼写与行为。
7. CNPerf CLI 版本、record/kernel 参数和输出格式。
8. CNCC 汇编、MLISA、IR 与临时产物参数和文件命名。
9. 实际链接所需库、host compiler ABI 与运行时库搜索路径。

服务器未确认前，选择保守、可编译探测、失败即报告的路径，不用营销规格填补证据空白。

## 交付检查

1. 无 Triton、`tl.*`、Python JIT 或 `torch.mlu` 运行依赖。
2. `.mlu` 同时包含真实 BANG C kernel 与 CNRT host 路径。
3. 所有 CNRT 调用、launch、同步和清理有错误处理。
4. GDRAM/NRAM/WRAM/SRAM 地址空间和搬运方向正确。
5. task mapping 覆盖全部逻辑元素且无越界/race。
6. 真实 CNCC 编译、MLU 运行和 reference 精度证据齐全。
7. 性能数据来自一致计时方法；不可用项显式标注。
