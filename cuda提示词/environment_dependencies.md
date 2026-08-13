# v2-bangc Skill 反向提取依赖表 (environment_dependencies.md)

> 生成方式：完整扫描 `mlu-bangc-main / mlu-bangc-code-gen / mlu-bangc-code-review / mlu-bangc-optimize / share/mlu` 下所有 `.md/.py/.sh/.mlu`，逐关键字 grep + 人工核对源码 + 真实命令验证。
> 「是否必须」列：必须 = 缺失会阻断 Skill 主链路；可选 = 仅特定场景使用；仅文档 = 只在 reference 文档提及、脚本不直接依赖。

## 1. 外部工具（命令行可执行文件）

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `cncc` | EnvConfig.md:58; test_env_code.py:131-140; GenTestCode.md:430; DynamicFixer.md:97; platform-rules.md:65,94; code-review/SKILL.md:146,181; perf-analyzer/strategy.md:84 | BANG C 编译器，`.mlu → binary` | 必须 |
| `cnmon` | get_device_info.py:100-107; analyzer.sh:49-50; EnvConfig.md:57,60; DynamicFixer.md:185; perf-analyzer/strategy.md:14,302 | MLU 设备状态/型号采集 | 必须 |
| `cnperf-cli` | analyzer.sh:61,72,75,80,93; perf-analyzer/strategy.md:123 | 性能 profiling（record/kernel） | 可选（缺失时 analyzer.sh 走 PARTIAL 分支，不阻断） |
| `cnpapi` (二进制) | 无直接调用 | — | 不依赖（Skill 仅在文档提及，见下表） |
| `cnas` / `cnlink` / `objdump` / `cngdb` / `cnsanitizer` | 无任何引用 | — | 不依赖 |

## 2. 头文件（#include）

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `bang.h` | bangc_vector_add.mlu:1; GenerateCode.md:84,112,453; Extractor.md:39,126; 全部 code-gen examples; platform-rules.md:82; test_env_code.py:72,151,156 | BANG C 设备侧 intrinsic/地址空间定义 | 必须 |
| `cnrt.h` | bangc_vector_add.mlu:2; GenerateCode.md:84,113,454; Extractor.md:39,127; 全部 examples; platform-rules.md:83,150; test_env_code.py:73,151,157 | CNRT 运行时 API/类型/返回码 | 必须 |
| `cndrv.h` / `cndev.h` | platform-rules.md:102（仅文字：「需要 cndrv/cndev 等库时只按源码实际依赖增加」） | 可选驱动/设备库 | 仅文档（源码无 `#include`） |
| `cnpapi.h` / `cnperf_api.h` / `cnnl.h` / `mlu_runtime_api.h` | 无 `#include` | — | 不依赖 |

## 3. 库（链接）

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `libcnrt` (`-lcnrt` / `libcnrt.so*`) | test_env_code.py:84,161; GenTestCode.md:433; platform-rules.md:84,97 | CNRT 运行时库 | 必须 |
| `libstdc++` (`-lstdc++`) | **无任何引用**（test_env_code.py:116、GenTestCode.md:430-436、platform-rules.md:93-100 链接行均只有 `-lcnrt`） | C++ 标准库（smoke test 用 `std::vector`） | **隐式必须但 Skill 未声明**（见不匹配项） |
| `libcndrv` / `libcndev` / `libcnpapi` / `libcnas` | 无 `-l` 链接 | — | 不依赖（仅文档提及可选） |

## 4. BANG C 设备侧语法 / intrinsic

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `__mlu_global__` | bangc_vector_add.mlu:28; Extractor.md:33,132; GenerateCode.md:131,139,460; GenerateSpec.md:75,296; 全部 examples/策略; primitives.md:30 | kernel 入口限定符 | 必须 |
| `__mlu_entry__` | Extractor.md:33; GenerateCode.md:139; StaticReviewer.md:89 | 备选入口限定符 | 可选 |
| `__nram__` / `__sram__` / `__wram__` | bangc_vector_add.mlu:22-24; Extractor.md:40,136-138; GenerateCode.md:204; DynamicFixer.md:298; primitives.md:31,50; platform-rules.md:229,254 | 片上地址空间 | 必须 |
| `__memcpy` (GDRAM2NRAM/NRAM2GDRAM 等) | bangc_vector_add.mlu:40,44,56; Extractor.md:42,144-147; GenerateCode.md:88,232,239; 全部 examples; StaticReviewer.md:145; kernel-info/strategy.md:263; retiling/* | 片上↔片外搬运 | 必须 |
| `__memcpy_async` | primitives.md:56; platform-rules.md:254 | 异步搬运 | 可选 |
| `__bang_*` (add/sub/add_scalar/floor/sumpool/write_value) | Extractor.md:43,146; GenerateCode.md:89,277; StaticReviewer.md:257; common_error.md:245; primitives.md:37,62-66; bangc_vector_add.mlu:50(注释) | 向量/reduction/math intrinsic | 可选（按算子需要） |
| `taskId` / `taskDim` (及 X/Y/Z、clusterId/coreId/coreDim) | bangc_vector_add.mlu:35,100; platform-rules.md:154-166; primitives.md:32-33,53-54; GenerateCode.md:86,169-171; modify-grid/*; kernel-info/*; retiling/* 等 | 任务索引内建变量 | 必须 |

## 5. CNRT 运行时 API

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `cnrtSetDevice` | bangc_vector_add.mlu:76; GenerateSpec.md:250; platform-rules.md:141; examples | 设备选择 | 必须 |
| `cnrtQueueCreate` / `cnrtQueueSync` / `cnrtQueueDestroy` | bangc_vector_add.mlu:79,108,130; Extractor.md:44,152,161,166; examples; platform-rules.md:142,145,147 | 队列管理 | 必须 |
| `cnrtMalloc` / `cnrtFree` | bangc_vector_add.mlu:86-88,127-129; Extractor.md:44,154-156,163-165; GenerateCode.md:365; GenTestCode.md:327; platform-rules.md:143,146 | 设备显存 | 必须 |
| `cnrtMemcpy` (+ `cnrtMemcpyHostToDev`/`DevToHost`) | bangc_vector_add.mlu:90,94,109; Extractor.md:44; examples; common_error.md:72; platform-rules.md:144 | H2D/D2H | 必须 |
| `cnrtNotifierCreate` / `cnrtPlaceNotifier` / `cnrtNotifierDuration` | examples(generate_code_matrix_transpose.md:107-120; generate_code_transpose_elementwise.md:123-138); perf-analyzer/strategy.md:102 | kernel 计时 | 可选（性能阶段必须） |
| `cnrtDim3_t` / `cnrtFunctionType_t` / `cnrtFuncTypeBlock` | bangc_vector_add.mlu:102-103; Extractor.md:158-159; GenerateSpec.md:255-256; GenerateCode.md:377-378; modify-grid/* | launch 维度/类型 | 必须 |
| `cnrtFuncTypeUnion` | 无引用 | — | 不依赖 |
| `CNRT_CHECK` (宏) | bangc_vector_add.mlu:13(自定义); GenerateCode.md:118; StaticReviewer.md:286; 全部 host 代码 | 错误检查 | 必须 |
| `CNRT_RET_SUCCESS` (成功常量) | bangc_vector_add.mlu:16(仅此处) | 返回码比较 | 必须（但 Skill 用旧名，见不匹配项） |

## 6. MLISA / 汇编 / 中间产物

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
| `.s` / `.S` / `.asm` / `.mlisa` | analyzer_cncc_artifacts.py:15,17,18,76-77; perf-analyzer/strategy.md:12,95,144-145,219; platform-rules.md:424 | 汇编/MLISA 产物识别 | 可选（缺失写 UNAVAILABLE） |
| `.o` / `.cnbin` / `.fatbin` | analyzer_cncc_artifacts.py:19,22,23,78 | 二进制/目标产物识别 | 可选 |
| `.ll` / `.bc` | analyzer_cncc_artifacts.py:24,25,80-81 | LLVM IR 产物识别 | 可选 |
| cncc 汇编/MLISA 生成参数 (`-S`/`-save-temps`/`-emit-llvm`) | **无硬编码**；perf-analyzer/strategy.md:95 明确「如当前 CNCC --help 明确支持…可附加，若不支持不猜」 | 生成中间产物 | 可选（Skill 不预设，按 --help 探测） |

## 7. 环境变量

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `NEUWARE_HOME` | test_env_code.py:53; GenTestCode.md:431-432; GenerateSpec.md:363; platform-rules.md:74; primitives.md:89,93; libdevice.md:54-55; troubleshooting.md:28 | NeuWare 安装根 | 必须（cncc 自身也依赖它定位 include/libdevice） |
| `CNTOOLKIT_HOME` | test_env_code.py:53; platform-rules.md:75 | 备选 NeuWare 根 | 可选 |
| `PATH` | get_device_info.py:102(shutil.which); analyzer.sh:49,61(command -v) | 查找 cnmon/cncc/cnperf-cli | 必须（需含 NeuWare bin） |
| `LD_LIBRARY_PATH` | test_env_code.py:205-206 | 运行 binary 时定位 libcnrt | 必须（需含 NeuWare lib64） |
| `BANGC_ARCH` / `BANGC_ARCH_FLAG` | test_env_code.py:5,90-96,171-172,194; GenTestCode.md:434,439; platform-rules.md:114 | 已确认的 arch flag | 可选（未设则用 CNCC 默认；Skill 禁止凭设备名猜） |
| `BANGC_CNCC_EXTRA_FLAGS` | test_env_code.py:120 | 附加 cncc 参数 | 可选 |
| `CNCC` | test_env_code.py:131 | 覆盖编译器可执行名 | 可选 |
| `MLU_DEVICE_TYPE` | submit_task_to_worker.py:60-61 | Worker 设备类型（默认 mlu590） | 仅 Worker 模式 |
| `JOB_ID` | submit_task_to_worker.py:64,178-180; EnvConfig.md:23,36; SKILL.md:65,228; DynamicFixer.md:182 | Worker 任务归属 | 仅 Worker 模式必须 |
| `JOB_ROOT` | submit_task_to_worker.py:155-156 | Worker 日志定位 | 仅 Worker 模式可选 |
## 8. 子进程 / Shell 调用

| 依赖名称 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `subprocess.run` (Python) | get_device_info.py:14,29-33; test_env_code.py:15,28-34 | 调用 cnmon/cncc/binary | 必须 |
| `shutil.which` | get_device_info.py:13,100; test_env_code.py:14,132 | PATH 查找可执行 | 必须 |
| `urllib.request` (HTTP) | submit_task_to_worker.py:全文件 | 调 Agent-Service (127.0.0.1:8086) | 仅 Worker 模式 |
| `bash` (analyzer.sh) | analyzer.sh:1,12; perf-analyzer/strategy.md:110-111 | perf 采集脚本 | 可选（性能阶段） |
| `spawn_agent(` | code-gen/SKILL.md:141,179,216,254,302,345; main/SKILL.md:103,123; optimize/SKILL.md:222; code-review/SKILL.md:235,278 | 分发 subagent | 必须（编排层） |
| `Skill(` | code-gen/SKILL.md:392; main/SKILL.md:170 | 加载子 Skill | 必须（编排层） |
| `os.system` | 无 | — | 不依赖 |

## 9. cncc 编译参数（Skill 实际使用/构造的）

| 参数 | Skill 中使用位置 | 用途 | 是否必须 |
|---|---|---|---|
| `--bang-mlu-arch=<value>` | test_env_code.py:96; code-review/SKILL.md:152 | 指定 MLU 架构 | 可选（未设用默认；禁猜） |
| `-o` / `-I` / `-L` / `-lcnrt` / `-std=c++11` | GenTestCode.md:430-435; test_env_code.py:106-118; platform-rules.md:94-100 | 编译/链接 | 必须 |
| `--bang-arch=compute_50` (=mtp_592) | 无（仅官方 build.sh 用） | — | Skill 未用 |
| `-O` / `-c` / `-S` / `--emit-*` / `--fmlisa` / `--fbang-code` | 无硬编码 | 中间产物 | 可选（按 --help 探测） |
## 10. MLU 架构枚举

| 架构名 | Skill 中使用位置 | 是否必须 |
|---|---|---|
| `mlu590` | submit_task_to_worker.py:60-61（`MLU_DEVICE_TYPE` 默认值） | 仅 Worker 默认 |
| `mtp_592` / `compute_50` | 无（Skill 不硬编码，由 EnvConfig 探测） | — |
| mlu370/580/580h/2xx/220/270/290/5xx | 无任何引用 | — |

> 注：Skill 明确禁止「根据设备营销名直接猜 `--bang-mlu-arch`」（code-review/SKILL.md:152；test_env_code.py:5-8,194），arch 必须来自 `BANGC_ARCH_FLAG`/`BANGC_ARCH` 或 CNCC 默认。