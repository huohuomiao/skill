# GenTestCode

## 任务

以 `step5_kernel_code.mlu` 为唯一 kernel 源，在同一 translation unit 中补齐 CPU reference、确定性数据生成、CNRT 资源管理、正确性测试、错误检查和 CNRT notifier benchmark，写出 `step6_test_code.mlu`。本阶段只生成，不执行；统一由 `mlu-bangc-code-review` 编译运行。

## 输入

- `step5_kernel_code.mlu`
- `step4_code_spec.json`（快速路径可不存在）
- `step1_base_info.json` 与 `step1_io_shapes.json`（快速路径可标记 skipped）
- `requirement.md`
- 最近的 `EnvConfig/config.md`
- `{BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md`

## 输出

```text
{output_dir}/KernelGen/step6_test_code.mlu
```

文件必须自包含、可执行。程序发现任何 CNRT API、queue sync、accuracy 或 benchmark 失败时返回非零，禁止只打印 warning 后返回成功。

## Stage 0：已有测试完整性

先检查 Stage 5 源码。只有同时满足下列条件才可逐字节复制为 Stage 6：

1. 有一个可执行 `main`。
2. 有独立 CPU reference，不能用设备输出自身作为 expected。
3. 有确定性输入与至少一个 tail/boundary case。
4. 有 `cnrtSetDevice`、queue、device allocation、H2D/D2H、queue sync 和清理。
5. 有 `atol + rtol * abs(expected)` 或需求规定的精确比较，失败返回非零。
6. 有 warmup 与 `cnrtNotifierCreate/PlaceNotifier/Duration` 性能测试。
7. 输出 `host_reference_ms` 与 `original_bangc_ms` 或等价可规范化字段。

缺任一项就按本文补齐。检查快速路径时不得修改用户 kernel/launcher；只允许在唯一 harness 标记后追加 host 测试。若没有安全插入位置，报告 GenerateCode/Extractor 失败，不盲目拼接或重排用户源码。

## 输出构建协议

使用 EnvConfig 已确认的命令与路径。通用命令形式：

```bash
cncc step6_test_code.mlu -o step6_test_code \
  -I"${NEUWARE_HOME}/include" -L"${NEUWARE_HOME}/lib64" \
  -lcnrt -lstdc++ -lm -lpthread -std=c++11
./step6_test_code
```

- EnvConfig 给出已验证的完整 arch flag 时逐字追加；否则不追加并记录使用 `cncc` 默认。禁止由设备营销名猜 flag。
- `bang.h` 可位于编译器资源目录；不要要求 `${NEUWARE_HOME}/include/bang.h` 存在。
- 使用 `std::vector`、`std::chrono`、math/thread 支持时必须保留 `-lstdc++ -lm -lpthread`，不能只链接 `-lcnrt`。
- 本阶段不运行上述命令；命令由 code-review 执行并记录 stdout/stderr/exit code。

## 文件合成规则

1. 读取完整 `step5_kernel_code.mlu`。
2. 保持 device kernel 与业务 launcher 原文不变；只补确实缺少的标准 host headers。
3. 在 `// === BANGC_TEST_HARNESS_BEGIN ===` 后写测试。
4. 不复制第二份 kernel/launcher。
5. 不引入未在 spec 声明的测试框架、Python 或第三方库。
6. 不重定义 `CNRT_CHECK`，不使用 `CNRT_RET_SUCCESS`。

推荐结构：

```cpp
// generated includes/helpers/kernels/launchers
// === BANGC_TEST_HARNESS_BEGIN ===
// deterministic input generator
// independent CPU reference
// comparison and diagnostic helpers
// CNRT allocation/copy/run helpers
// run_correctness_case(...)
// benchmark_case(...)
// main(): config -> correctness -> benchmark -> cleanup -> summary
```

## CNRT 错误处理

直接使用目标 `cnrt.h` 提供的 `CNRT_CHECK`：

```cpp
CNRT_CHECK(cnrtSetDevice(0));
CNRT_CHECK(cnrtQueueCreate(&queue));
CNRT_CHECK(cnrtMalloc(reinterpret_cast<void**>(&device_x), bytes));
CNRT_CHECK(cnrtMemcpy(device_x, host_x.data(), bytes,
                      cnrtMemcpyHostToDev));
```

- 禁止再次 `#define CNRT_CHECK`。
- 禁止用旧常量 `CNRT_RET_SUCCESS`。
- 如果目标 SDK 不提供该宏，code-review 必须从当前 `cnrt.h` 选择不冲突的辅助与当前成功枚举；GenerateCode 不提前猜。
- kernel launch 后的异步错误由 `cnrtQueueSync(queue)` 捕获。
- 所有 allocation、copy、queue、notifier API 都检查。
- 任一步失败都必须使程序以非零状态结束；清理路径不得泄漏已经创建的资源。

## 资源管理

每个 case 的顺序：

1. 检查 shape 元素数和 `count*sizeof(T)` 不溢出。
2. 创建 host input/reference/output。
3. `cnrtSetDevice`，创建 queue。
4. `cnrtMalloc` 所有 device buffers。
5. H2D copy。
6. 调用业务 launcher。
7. `cnrtQueueSync` 捕获执行错误。
8. D2H copy并再次 sync（按目标 API 同步语义）。
9. 比较并输出结构化结果。
10. destroy notifier、free device memory、destroy queue。

可使用小型 RAII wrapper，但析构中不能悄悄吞掉影响结论的错误。显式 cleanup 更容易与旧 SDK 兼容。零长度 case 不解引用空 pointer，也不 launch。

## CPU reference

reference 必须独立表达数学合同，不复用设备 task/tile/index helper：

```cpp
void reference_vector_add(const std::vector<float>& x,
                          const std::vector<float>& y,
                          std::vector<float>* out) {
  out->resize(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    (*out)[i] = x[i] + y[i];
  }
}
```

归约默认使用更高精度 host accumulator：

```cpp
double sum = 0.0;
for (int64_t k = 0; k < K; ++k) {
  sum += static_cast<double>(x[row * stride0 + k * stride1]);
}
expected[row] = static_cast<float>(sum);
```

若 requirement 规定固定累加顺序、溢出或舍入，reference 按合同实现并记录，不可同时宣称顺序无关。

用 `std::chrono` 可单独测量 CPU reference：

```cpp
const auto host_begin = std::chrono::steady_clock::now();
reference_op(inputs, &expected);
const auto host_end = std::chrono::steady_clock::now();
const double host_reference_ms =
    std::chrono::duration<double, std::milli>(host_end - host_begin).count();
```

该值不能混入 device notifier 区间。

## 输入生成

- 固定 seed，例如 requirement/test schema 的 `20260813`。
- 混合正负、非整数、小量级与大但合法值。
- 特殊值用例单独构造，避免随机分布偶然不覆盖。
- index 输入覆盖首尾、重复（若合法）、非排序与边界。
- performance 输入在 benchmark 前初始化、分配和拷贝。
- non-contiguous case 在 host buffer 中按真实 stride/padding 构造，不能只传紧凑数组。

## 测试矩阵

严格实现 Stage 1 correctness cases，至少覆盖需求允许的：

- 最小合法 shape。
- 小于一个 tile。
- 恰好整 tile。
- 非整 tile 的 tail。
- 代表性 shape。
- 支持的 layout/stride。
- 数值特殊值与 dtype 边界。
- zero-size/empty reduction（仅合同允许时）。

多个 dtype 必须分别测试；不能只测试 float32 就声明 fp16/bf16/int 通过。快速路径缺少 Stage 1 JSON 时，从 requirement 的明确用例生成同等测试矩阵；缺失关键信息则返回 Extractor，不能猜。

## 比较器

浮点默认：

```cpp
bool nearly_equal(float actual, float expected,
                  double atol, double rtol) {
  if (std::isnan(expected)) return std::isnan(actual);
  if (std::isinf(expected)) return actual == expected;
  const double a = static_cast<double>(actual);
  const double e = static_cast<double>(expected);
  const double diff = std::fabs(a - e);
  return diff <= atol + rtol * std::fabs(e);
}
```

同时统计：

- `max_abs_error`
- `max_rel_error`（expected 非零）
- mismatch count
- 首个 mismatch 的逻辑坐标、expected、actual、abs/rel error

整数/bitwise 合同精确比较。NaN/Inf/signed zero 按 requirement。失败行：

```text
ACCURACY_FAIL case=C04 index=[7,31] expected=1.25 actual=1.5 abs=0.25 rel=0.2
```

随后 case 与 `main` 必须失败，不能只打印警告。

成功 case 输出：

```text
CASE_RESULT id=C02 passed=true max_abs_error=0 max_rel_error=0
```

## CNRT notifier benchmark

correctness 全部通过后执行。

### 范围

输入/输出 device buffers 和 queue 在计时前准备。warmup：

```cpp
for (int i = 0; i < warmup; ++i) {
  if (!launch_op(device_args, queue)) return false;
}
CNRT_CHECK(cnrtQueueSync(queue));
```

计时：

```cpp
cnrtNotifier_t start;
cnrtNotifier_t end;
CNRT_CHECK(cnrtNotifierCreate(&start));
CNRT_CHECK(cnrtNotifierCreate(&end));
CNRT_CHECK(cnrtPlaceNotifier(start, queue));
for (int i = 0; i < iterations; ++i) {
  if (!launch_op(device_args, queue)) return false;
}
CNRT_CHECK(cnrtPlaceNotifier(end, queue));
CNRT_CHECK(cnrtQueueSync(queue));

float duration_raw = 0.0f;
CNRT_CHECK(cnrtNotifierDuration(start, end, &duration_raw));
```

`duration_raw` 的单位必须从当前 CNRT 文档/共享平台规则确认后换算。已确认返回微秒的环境可使用：

```cpp
const double original_bangc_ms =
    static_cast<double>(duration_raw) / 1000.0 / iterations;
```

如果单位未确认，输出 raw value 和单位 `unverified`，`original_bangc_ms=N/A`；不得凭变量名猜单位。

计时循环中禁止 allocation、H2D/D2H、初始化、CPU reference 或打印。wrapper 含多 kernel 时 notifier 覆盖整个同 queue device work，并在 scope 说明。

### 带宽

只有能定义逻辑流量时报告：

```text
effective_GBps = bytes_per_iteration / (original_bangc_ms * 1e-3) / 1e9
```

`bytes_per_iteration` 累加所有实际输入读与输出写的 dtype/shape bytes；额外 indices、mask、scale 或多输出分别计算。不要把缓存/物理 transaction 当成已测 DRAM bytes。

### 输出

```text
BENCHMARK_RESULT case=P01 warmup=20 iterations=100 host_reference_ms=12.34 original_bangc_ms=0.1234 effective_gbps=456.7 scope=launcher_device_work
```

notifier 或 queue sync 失败时返回非零且不输出成功 benchmark。

## `main` 流程

```cpp
int main(int argc, char** argv) {
  // parse only documented optional flags
  // print build/config metadata supplied by EnvConfig
  // cnrtSetDevice and create queue
  // run every correctness case
  // only after all pass, run benchmark
  // cleanup all resources
  // print final machine-readable summary
  return all_passed ? 0 : 1;
}
```

最终成功行：

```text
FINAL_RESULT compile_pass=true accuracy_pass=true benchmark_pass=true
```

程序内部的 `compile_pass=true` 只表示 binary 已构建并启动；完整编译事实以 code-review 记录的命令和退出码为准。

## 目标设备门禁

- 主流程 EnvConfig 与 code-review 必须在执行前确认目标是 MLU590。
- harness 不调用未经确认的 CNRT 设备属性 API来猜型号。
- 若 EnvConfig 提供经过验证的 device query 代码，可以逐字采用；否则设备名、arch、容量由外部报告记录。
- 其它设备上运行成功只能作为 portability 信息，不能写 `target_verified=true`。

## 禁止做法

- 用设备输出自身作为 expected。
- 只测一个整 tile shape。
- kernel 后不检查 queue sync。
- 用 host wall-clock 标成 CNRT notifier。
- 把 allocation/H2D/D2H 混入 kernel-only 指标。
- accuracy 失败后仍返回 0。
- 捕获错误后只打印 warning。
- benchmark 先于 correctness。
- 用其它 MLU 成功冒充 MLU590 实测。
- 重定义 `CNRT_CHECK` 或使用 `CNRT_RET_SUCCESS`。
- 按 MLU590 名称猜 arch、task 数或片上容量。

## 完成自检

- [ ] 只有一套 kernel/launcher。
- [ ] CPU reference 与 task/tile mapping 独立。
- [ ] requirement correctness cases 全部实现。
- [ ] 包含小值、tail、边界和 performance case。
- [ ] CNRT allocation/copy/queue/notifier 全部检查与清理。
- [ ] benchmark 使用同一 queue 的 CNRT notifier。
- [ ] 输出 `host_reference_ms` 与 `original_bangc_ms`；无法确认时写 N/A。
- [ ] correctness/runtime 失败必然非零退出。
- [ ] build 命令链接 `cnrt/stdc++/m/pthread`，arch 不猜测。
- [ ] 可直接交给 `mlu-bangc-code-review`。

## 失败处理

- CPU reference 语义不明确：返回 Extractor。
- 测试矩阵缺失 dtype/layout/stride：返回 ExtractBaseInfo。
- launcher 不可测试或缺 host launch：返回 GenerateCode。
- notifier 单位或 API 未确认：保留正确性测试，性能字段写 N/A 并报告阻塞原因。
- 环境 unavailable：仍生成完整源码，不执行、不写成功结论。
- 某特殊值无法构造：说明具体类型/API限制并返回 requirement，不静默删除 case。
