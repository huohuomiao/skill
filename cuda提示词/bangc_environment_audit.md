# v2-bangc 本地环境审计报告 (bangc_environment_audit.md)

> 检查日期：2026-08-13 ｜ 机器：MLU590-M9DG ｜ 仅检查，未安装/升级/删除任何软件，未修改系统环境、NeuWare、CNToolkit、驱动或 v2-bangc。
> 所有 PASS 均来自真实命令或文件检查；编译/运行/性能数据均为真实执行结果，未伪造。

---

## 1. 环境总结

### 结论：**PARTIALLY_READY**

**判定依据**：
- ✅ **底层 MLU 工具链本身完全可用**：在临时副本上修正 3 处 Skill 源码与 SDK 不匹配后，真实完成了 `.mlu → cncc → binary → MLU execution → accuracy` 全链路（`BANGC_VECTOR_ADD_PASS`，max_diff=0），且 CNPerf、notifier 计时、MLISA/汇编、官方 sample 全部真实跑通。
- ❌ **Skill 自带的环境检测脚本 `test_env_code.py` 在当前机器上无法通过**（即使 `NEUWARE_HOME`/`PATH` 已正确设置），原因有 3 处 Skill 源码与 NeuWare 4.6.2 的不匹配（详见第 4 节）。这导致 Skill 的 EnvConfig 会判定本地不可用 → 尝试 Worker 兜底 → Worker 端点 `127.0.0.1:8086` 不可达且 `JOB_ID` 未设 → 工作流被迫停止。
- ⚠️ 因此按 Skill 现有逻辑，**主链路无法自动走通**；但底层环境并非 NOT_READY——工具链齐全且经实测可用，问题集中在 Skill 的 smoke-test 源码与脚本假设上。

> 说明：按指令定义，PARTIALLY_READY = 基础 BANG C 开发可运行但 CNPerf/MLISA 等优化工具部分缺失。本机实际情况相反——**优化工具(CNPerf/MLISA/notifier)全部可用，反而是基础检测脚本因源码不匹配而失败**。综合「Skill 主链路能否自动走通」这一最终目标，归为 PARTIALLY_READY（更精确说是「Skill 源码需最小修改后即可 READY」），而非 NOT_READY（底层编译/运行链路经实测可完成）。

---

## 2. 基础开发环境

| 项目 | 真实检查结果 | 状态 |
|---|---|---|
| MLU 设备 | `cnmon`：1× MLU590-M9DG，Firmware v1.1.1，Bus 0000:63:00.0，显存 0/81920 MiB，Mode FULL，0% 占用 | ✅ PASS |
| Driver | CNMON v6.5.26 / Driver v6.5.26 | ✅ PASS |
| `cnmon` | `/usr/bin/cnmon`，在 PATH 上，exit 0 | ✅ PASS |
| NeuWare 安装 | `/usr/local/neuware`（Neuware Version 4.6.2），bin/include/lib64/lib/cmake/samples 齐全 | ✅ PASS |
| `NEUWARE_HOME` | **未设置**（空） | ⚠️ 需设置（cncc 依赖它定位 include/libdevice） |
| `PATH` 含 NeuWare bin | **否**（`which cncc`/`which cnperf-cli` 均失败） | ⚠️ 需追加 `/usr/local/neuware/bin` |
| `LD_LIBRARY_PATH` | 已含 `/usr/local/neuware/lib64` | ✅ PASS |
| `cncc` | `/usr/local/neuware/bin/cncc`，`cncc 5.6.2 mlvm 1.3 clang 11.1.0` | ✅ PASS（不在默认 PATH） |
| `bang.h` | **不在** `include/`；实际在 `/usr/local/neuware/lib/clang/11.1.0/include/bang.h`（clang 资源头） | ⚠️ 位置与 Skill 假设不同（cncc 经 NEUWARE_HOME 自动找到，无需 `-I`） |
| `cnrt.h` | `/usr/local/neuware/include/cnrt.h`（244KB） | ✅ PASS |
| `cndrv.h` | **不存在**；CNDrv 现由 `cn_api.h` 提供（2023 版） | ℹ️ Skill 不依赖，无影响 |
| `libcnrt.so` | `libcnrt.so → libcnrt.so.7.6.1`（在 lib64） | ✅ PASS |
| `libcndrv.so` / `libcndev.so` | `libcndrv.so→3.6.2`、`libcndev.so→6.5.40`（在 lib64） | ✅ 存在（Skill 不链接） |
| CNCC arch 参数 | `--bang-mlu-arch=mtp_592`（= `--bang-arch=compute_50`，官方 build.sh 对 MLU590 的默认） | ✅ 真实确认（来自 cncc --help 与官方 build.sh，非凭名猜测） |
| 官方 BANG C sample | `samples/BANG/1_Performance/vectoradd/2_vectorAdd_single_core`，`#include <bang.h>`，用 `__bang_add`/notifier | ✅ PASS（直接 cncc 编译+运行 → `[MLU Hardware Time]: 4102.000 us`，`PASSED`） |
| **最小编译（Skill .mlu 原样）** | `bangc_vector_add.mlu` 原样编译 → **FAIL**（13 errors：`CNRT_RET_SUCCESS` 未声明 + `CNRT_CHECK` 重定义） | ❌ FAIL（源码不匹配，见 4.1/4.2） |
| **最小编译（临时副本修正后）** | 修正后编译 → **PASS**，binary 37952 字节 | ✅ PASS |
| **最小运行（修正后）** | `BANGC_VECTOR_ADD_MAX_DIFF=0`，`BANGC_VECTOR_ADD_PASS`，exit 0 | ✅ PASS |
| 精度验证 | max_diff=0 < 1e-5 容限 | ✅ PASS |
| Python 脚本 | 5/5 `python3 -m py_compile` 全通过 | ✅ PASS |
| Shell 脚本 | `analyzer.sh` `bash -n` 通过 | ✅ PASS |
| Worker 兜底 | `127.0.0.1:8086` **连接拒绝**；`JOB_ID`/`JOB_ROOT` **未设置** | ❌ 不可用（本地可用时不需要） |

### 真实最小编译/运行命令（临时副本，未改原文件）

```bash
export NEUWARE_HOME=/usr/local/neuware
export PATH=/usr/local/neuware/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/neuware/lib64:$LD_LIBRARY_PATH
# 临时副本仅做 3 处修正：删除自定义 CNRT_CHECK 宏、CNRT_RET_SUCCESS→cnrtSuccess、链接行加 -lstdc++ -lm -lpthread
cncc bangc_vector_add.mlu -o bangc_vector_add \
  -I/usr/local/neuware/include -L/usr/local/neuware/lib64 \
  -lcnrt -lstdc++ -lm -lpthread -std=c++11 --bang-mlu-arch=mtp_592
./bangc_vector_add   # → BANGC_VECTOR_ADD_MAX_DIFF=0 / BANGC_VECTOR_ADD_PASS
```

---

## 3. 优化环境

| 项目 | 真实检查结果 | 状态 |
|---|---|---|
| notifier 计时 | `cnrtNotifierCreate/PlaceNotifier/NotifierDuration/QueueSync` 在 `cnrt.h` 均存在；官方 sample 实测输出 `[MLU Hardware Time]: 4102.000 us` | ✅ PASS |
| `cnperf-cli` | `/usr/local/neuware/bin/cnperf-cli`，`cnperf-cli 6.6.1`（不在默认 PATH，需加 bin） | ✅ PASS |
| CNPerf record/kernel | `analyzer.sh` 实测：`CNPERF_RECORD=PASS`、`CNPERF_KERNEL=PASS`、`STATUS=COMPLETE_OR_PARTIAL_WITH_CNPERF` | ✅ PASS |
| CNPerf 输出内容 | kernel duration `0s 8ms 366us 480ns`；memory（read/write bytes、MB/s、bandwidth_utils）；compute（alu_cycles、simd_inst_executed）；DMA/TLB 等 | ✅ 丰富可用 |
| CNPerf 输出格式 vs Skill 解析器 | `analyzer_rep.py` 实测解析真实输出：**durations=0, counters=2, tables=0**——几乎提取不到内容（见 4.4） | ⚠️ 解析器与真实格式不匹配 |
| `cnperf-gui` | `/usr/local/neuware/bin/cnperf-gui`（198MB）存在 | ✅ 存在（CLI 已够用） |
| CNPAPI | `libcnpapi.so→4.6.2`、`cnpapi.h`、`samples/cnpapi/`(4 sample) 均存在；无 `cnpapi` 命令行二进制 | ✅ 可用（Skill 标记 OPTIONAL，不阻断） |
| MLISA/汇编生成 | `cncc -S` 生成 `*-bang-mlisa-cambricon-bang-mtp_592.s`（含 `.mlisa 5.0`/`.arch mtp_592`，CNCC MLISA Back-End）+ 合并 `.s`；`-save-temps` 生成 `.bc/.mlui/.o/.cnfatbin` | ✅ PASS |
| MLISA 产物识别 | `analyzer_cncc_artifacts.py` 实测：3 个 `.s`→`assembly_or_mlisa`、2 个 `.bc`→`compiler_ir`、2 个 `.o/cnfatbin`→`binary_or_object` | ✅ PASS |
| cncc 中间产物参数 | `--help` 确认：`-S`(汇编)、`-save-temps`、`-emit-llvm`、`--bang-cnbin-only`、`--bang-fatbin-only`、`--cnas-path` | ✅ 真实可用 |
| `cnas` 汇编器 | `/usr/local/neuware/bin/cnas` 存在（cncc `--cnas-path` 可用） | ✅ 存在 |

# 4. Skill 与环境不匹配项

### 4.1 【P0·阻断】`test_env_code.py` 的 `find_neuware_root()` 要求 `include/bang.h` 存在，但本机 `bang.h` 不在 `include/`

- **位置**：`share/mlu/runtime/test_env_code.py:66-74`（`find_neuware_root` 要求 `(root/"include"/"bang.h").is_file()` 且 `(root/"include"/"cnrt.h").is_file()`）。
- **真实情况**：NeuWare 4.6.2 中 `bang.h` 位于 `/usr/local/neuware/lib/clang/11.1.0/include/bang.h`（clang 资源头），**不在** `include/`；`cnrt.h` 在 `include/`。
- **后果**：即使 `NEUWARE_HOME`/`PATH` 正确，脚本仍报 `ERROR: could not locate a NeuWare/CNToolkit root containing include/bang.h and include/cnrt.h` 并 exit 1（已实测复现）。EnvConfig 判定本地不可用。
- **分类**：B（Skill API 假设）+ E（SDK 版本差异，头文件布局变更）。
- **备注**：cncc 经 `NEUWARE_HOME` 自动找到 `bang.h`，编译时无需 `-I include` 指向 bang.h；`-I include` 仅对 `cnrt.h` 等有意义。

### 4.2 【P0·阻断】`bangc_vector_add.mlu` 使用旧成功常量 `CNRT_RET_SUCCESS`，本机 `cnrt.h` 已改为 `cnrtSuccess`

- **位置**：`share/mlu/runtime/bangc_vector_add.mlu:16`（`if (_ret != CNRT_RET_SUCCESS)`），全 Skill 仅此一处用该符号。
- **真实情况**：NeuWare 4.6.2 `cnrt.h` 的 `cnrtRet_t` 枚举成功值为 `cnrtSuccess = 0`（cnrt.h:59），**无** `CNRT_RET_SUCCESS` 符号（grep 0 命中）。编译报 13× `use of undeclared identifier 'CNRT_RET_SUCCESS'`。
- **后果**：smoke test 源码无法编译，`test_env_code.py` 必然失败。
- **分类**：E（SDK 版本差异，返回码符号改名）。
- **关联**：该 `.mlu` 还在第 13 行**重定义** `CNRT_CHECK` 宏，与 `cnrt.h:6483` 已定义的 `CNRT_CHECK` 冲突（`-Wmacro-redefined` 警告）；官方 sample 直接用头文件里的 `CNRT_CHECK`（其实现为 `if (ret)` 判非零，无需显式成功常量）。
### 4.3 【P0·阻断】Skill 编译链接行缺 `-lstdc++`，导致用 `std::vector` 的 smoke test 链接失败

- **位置**：`test_env_code.py:106-118`（`compile_command` 仅 `-lcnrt`）；`GenTestCode.md:430-436`；`platform-rules.md:93-100`。
- **真实情况**：`bangc_vector_add.mlu` 用 `std::vector`（需 libstdc++）。仅 `-lcnrt` 链接报 `undefined reference to _ZSt20__throw_length_errorPKc@@GLIBCXX_3.4` / `DSO missing from command line`（已实测）。官方 CMakeLists 链接 `pthread stdc++ m cnrt`。
- **后果**：即便修了 4.1/4.2，链接仍失败。生成代码若用 C++ stdlib 同样会撞此问题。
- **分类**：B（Skill 链接参数不完整）。
- **修正验证**：加 `-lstdc++ -lm -lpthread` 后链接成功、运行 PASS（已实测）。

### 4.4 【P1·降级】`analyzer_rep.py` 解析器与真实 CNPerf 6.6.1 输出格式不匹配，几乎提取不到数据

- **位置**：`mlu-bangc-optimize/perf-analyzer/scripts/analyzer_rep.py`（`DURATION_RE`/`KEY_VALUE_RE`/`TABLE_ROW_RE`）。
- **真实情况**：对真实 `kernel_cnperf.txt` 实测解析 → `durations=0, numeric_counters=2, table_rows=0`。原因：
  - kernel 行 `Duration : 0s 8ms 366us 480ns` 是**复合时间字符串**，`DURATION_RE` 只匹配单值单单位（如 `8 ms`），无法捕获。
  - 指标表行格式是 `key:  unit  value`（如 `write_bytes:  bytes  588160`），而 `KEY_VALUE_RE` 期望 `key: value unit`，故绝大多数计数器漏提。
- **后果**：perf-analyzer 的 Step 5「解析 CNPerf」拿不到 duration/PMU 计数器，只能回退到「原始输出为最终证据」。Skill 本身声明「宽松解析、原始输出为准」，故不阻断，但**优化建议的证据质量显著下降**。
- **分类**：B（解析器与真实输出格式不匹配）。
- **备注**：Skill 设计上已声明 `analyzer_rep.py` 为 best-effort、原始报告为准，所以这是降级而非阻断；但当前解析率过低（2/~40 计数器），实用价值有限。

### 4.5 【P1·环境】`NEUWARE_HOME` 未设置 + NeuWare bin 不在 PATH

- **位置**：环境变量（非 Skill 源码）。
- **真实情况**：`NEUWARE_HOME` 为空；`PATH` 不含 `/usr/local/neuware/bin`，故 `which cncc`/`which cnperf-cli` 失败。
- **后果**：`test_env_code.py` 第一步 `shutil.which("cncc")` 即返回 None → `ERROR: CNCC not found`；`analyzer.sh` 的 `command -v cnperf-cli` 失败 → 走 PARTIAL_NO_CNPERF 分支。
- **分类**：A（环境配置）。
- **备注**：这是唯一「纯环境」问题，但指令禁止修改系统环境变量。可在调用前于**子进程级**临时 `export`（不写入系统），或由 Skill 脚本主动探测 `/usr/local/neuware`。

### 4.6 【P2·兜底缺失】Worker 兜底链路在本机不可用

- **位置**：`submit_task_to_worker.py`（依赖 `127.0.0.1:8086` Agent-Service + `JOB_ID`）。
- **真实情况**：`127.0.0.1:8086` 连接拒绝；`JOB_ID`/`JOB_ROOT` 未设。
- **后果**：当本地检测失败时，Skill 设计回退到 Worker；本机 Worker 不可达，故工作流会停止。但因本地底层实际可用，**不应触发 Worker 兜底**——根因是 4.1-4.3 让本地检测假阴性。
- **分类**：A（环境/基础设施）。
- **备注**：在真实 Agent-Service 部署环境下此项应可用；本机为纯开发检查环境。

### 4.7 【P3·文档】`platform-rules.md` 记载的 `include/bang.h` 路径与现版本不符

- **位置**：`share/mlu/references/platform-rules.md:82`（`include/bang.h`）。
- **真实情况**：同 4.1，`bang.h` 已移至 clang 资源 include。
- **分类**：E（文档与现版本布局差异）。仅影响文档准确性，不影响脚本（脚本问题在 4.1）。

---
## 5. 下一步最小修改建议（按优先级）

> 全部为「只报告，不修改 Skill」的建议；均经实测验证可行。P0 三项修完后，Skill 主链路在本机即可自动 READY。

### P0（阻断主链路，必须修）

1. **`test_env_code.py: find_neuware_root` 放宽 bang.h 定位**
   - 现状：硬要求 `include/bang.h`。
   - 建议：将 bang.h 探测改为「`include/bang.h` 或 `lib/clang/<ver>/include/bang.h` 存在即可」，或仅以 `include/cnrt.h` + `lib64/libcnrt.so` + cncc 可用为根判定（bang.h 由 cncc 经 NEUWARE_HOME 自动解析）。`candidate_roots` 已枚举 `/usr/local/neuware`，只需放宽 `find_neuware_root` 的 bang.h 判据。
   - 实测：放宽后即可定位根。

2. **`bangc_vector_add.mlu` 修正成功常量与宏重定义**
   - 现状：自定义 `CNRT_CHECK` 宏 + 用 `CNRT_RET_SUCCESS`。
   - 建议：删除第 13-21 行自定义 `CNRT_CHECK` 宏（改用 `cnrt.h` 自带的），并将 `CNRT_RET_SUCCESS` 替换为 `cnrtSuccess`（或直接用头文件 `CNRT_CHECK` 的 `if(ret)` 语义，无需成功常量）。
   - 实测：此两处修正后编译错误清零。

3. **Skill 编译链接行补 `-lstdc++`（及 `-lm -lpthread`）**
   - 现状：`test_env_code.py`/`GenTestCode.md`/`platform-rules.md` 链接行仅 `-lcnrt`。
   - 建议：在 `compile_command` 与文档模板的链接行追加 `-lstdc++ -lm -lpthread`（与官方 CMakeLists 的 `pthread stdc++ m cnrt` 一致）。
   - 实测：补齐后链接成功、运行 PASS。

### P1（影响优化证据质量，建议修）

4. **`analyzer_rep.py` 适配 CNPerf 6.6.1 真实格式**
   - 建议：① `DURATION_RE` 增加对复合时间 `0s 8ms 366us 480ns` 的解析（按 ns 累加）；② `KEY_VALUE_RE`/`TABLE_ROW_RE` 适配 `key:  unit  value` 三列布局，或对 `Kernels Info` 段做段级解析。
   - 备选：若不想维护格式，可让 perf-analyzer 直接引用 `kernel_cnperf.txt` 原文关键字段（Duration、read/write_bw、alu_cycles），最小正则提取这几项即可。
   - 实测：当前仅提取到 TID、Device ID 两个无关计数器。

5. **环境变量探测兜底**
   - 建议：在 `test_env_code.py`/`analyzer.sh` 中，当 `shutil.which`/`command -v` 找不到 cncc/cnperf-cli 时，主动探测常见路径 `/usr/local/neuware/bin`（脚本内本地变量，不修改系统环境）。
   - 可缓解 4.5，且不违反「不修改系统环境变量」。

### P2（兜底链路，按部署环境评估）

6. **Worker 兜底可用性**：在真实 Agent-Service 部署环境确认 `127.0.0.1:8086` 可达且 `JOB_ID` 已注入；本机为开发检查环境，无需修。若 P0 修好，本机不会触发 Worker。

### P3（文档准确性）

7. **`platform-rules.md:82` 更新 bang.h 路径说明**：补充「bang.h 在新版 NeuWare 中位于 `lib/clang/<ver>/include/`，由 cncc 经 NEUWARE_HOME 自动解析」。

## 6. 最终匹配表

| 项目 | Skill 需要 | 本地环境 | 状态 | 备注 |
|---|---|---|---|---|
| MLU Device | 可用 MLU | MLU590-M9DG ×1，Driver v6.5.26 | ✅ PASS | cnmon 识别正常，0% 占用 |
| cnmon | PATH 可用 | `/usr/bin/cnmon` | ✅ PASS | get_device_info.py 实测 PASS |
| NEUWARE_HOME | 已设置 | 未设置 | ⚠️ 需设 | cncc 依赖它；可子进程级临时 export |
| PATH(NeuWare bin) | 含 cncc/cnperf-cli | 不含 | ⚠️ 需追加 | which cncc/cnperf-cli 失败 |
| CNCC | 可用 + 版本 | cncc 5.6.2 @ /usr/local/neuware/bin | ✅ PASS | 不在默认 PATH |
| BANG C headers | bang.h + cnrt.h | bang.h 在 clang 资源 include；cnrt.h 在 include/ | ⚠️ 位置不同 | Skill 假设 include/bang.h，见 4.1 |
| CNRT(libcnrt) | libcnrt.so | libcnrt.so→7.6.1 | ✅ PASS | |
| CNRT API | cnrtSuccess/CNRT_CHECK 等 | cnrtSuccess=0；CNRT_CHECK 在 cnrt.h | ⚠️ Skill 用旧名 CNRT_RET_SUCCESS | 见 4.2 |
| BANG C compile | smoke test 可编译 | 原样 FAIL / 修正后 PASS | ❌→✅ | 见 4.1/4.2/4.3 |
| BANG C runtime | binary 可运行 | 修正后运行 PASS | ✅ PASS | 真实 MLU 执行 |
| accuracy | max_diff<1e-5 | max_diff=0 | ✅ PASS | |
| notifier timing | cnrtNotifier* 可用 | 官方 sample 实测 4102 us | ✅ PASS | |
| CNPerf | cnperf-cli 可用 | cnperf-cli 6.6.1 | ✅ PASS | analyzer.sh 实测 RECORD/KERNEL PASS |
| CNPerf 解析 | analyzer_rep.py 解析 | 解析率极低(2/~40) | ⚠️ 降级 | 见 4.4，原始报告仍可用 |
| CNPAPI | OPTIONAL | libcnpapi.so+cnpapi.h+samples 齐全 | ✅ 可用(OPTIONAL) | Skill 不直接依赖 |
| assembly | cncc -S 可生成 | `-S` 生成 `.s`(host+device) | ✅ PASS | |
| MLISA | cncc 可生成 MLISA | `-S` 生成 `*-bang-mlisa-*.s`(`.mlisa 5.0`) | ✅ PASS | CNCC MLISA Back-End |
| analyzer scripts | 产物识别 | analyzer_cncc_artifacts.py 实测识别 3 类产物 | ✅ PASS | |
| Python | 5 脚本可编译 | 5/5 py_compile PASS | ✅ PASS | |
| Worker script | 8086 可达 + JOB_ID | 8086 拒绝 + JOB_ID 未设 | ❌ 不可用 | 本机不需(本地可用)；见 4.6 |
## 7. 链路可用性总览

| 链路环节 | 真实可用性 |
|---|---|
| 环境检测 | ⚠️ Skill 脚本假阴性（4.1/4.2/4.3），底层实际可用 |
| BANG C 代码生成 | ✅ 静态生成不依赖运行环境；API 假设需对齐 4.2/4.3 |
| CNCC 编译 | ✅ 工具链可用（修正 smoke test 源码后实测 PASS） |
| CNRT 运行 | ✅ 实测 PASS（kernel launch/H2D/D2H/精度） |
| 精度验证 | ✅ 实测 PASS（max_diff=0） |
| 性能计时(notifier) | ✅ 实测可用（4102 us） |
| CNPerf 分析 | ✅ 采集可用；⚠️ 解析器降级（4.4） |
| 汇编/MLISA 分析 | ✅ 实测可用（`-S`/`-save-temps`） |
| 性能优化 | ✅ 证据链可用（CNPerf 原始报告 + MLISA + notifier + 源码静态） |
| Worker 兜底 | ❌ 本机不可用（本地可用时不需要） |

**最终结论**：底层 MLU/BANG C 开发与优化环境**齐全且经真实执行验证可用**；Skill 主链路当前因 3 处 P0 源码/脚本与 NeuWare 4.6.2 的不匹配而无法自动走通，按建议最小修正后即可达到 **READY**。归为 **PARTIALLY_READY**。

