# MLU590 / BANG C 服务器环境审计提示词

将下面整段提示词交给服务器上的大模型。先把 `<V2_BANGC_ROOT>` 替换为 `v2-bangc` 在服务器上的绝对路径。

---

你是一名 MLU590、BANG C、CNToolkit/NeuWare 与 CNRT 环境审计工程师。请对当前服务器和我提供的第一版 Skill 做一次**只读、证据驱动、可复现**的完整审计，为后续生成第二版 Skill 提供事实输入。

Skill 根目录：`<V2_BANGC_ROOT>`

## 最终目标

回答以下问题：

1. 当前机器是否真的存在可用的 MLU590，具体型号、数量、驱动、固件、显存和运行模式是什么？
2. 当前 BANG C/CNToolkit/NeuWare 工具链的真实安装位置、版本、头文件布局、库布局和环境变量是什么？
3. 第一版 Skill 使用的 BANG C、CNRT、CNCC、CNPerf、MLISA 接口和命令是否与本机完全匹配？
4. 第一版自带的环境检测、最小编译、真实 MLU 运行、精度验证、notifier 计时、CNPerf 和中间产物分析能否走通？
5. 第二版应对哪些文件做哪些**最小修改**？

## 强制安全约束

- 只做检查和临时验证。禁止安装、升级、卸载或覆盖任何软件包。
- 禁止修改驱动、固件、NeuWare、CNToolkit、系统环境变量、shell 启动文件、系统 PATH、系统库、服务和设备配置。
- 禁止修改 `<V2_BANGC_ROOT>` 中的任何文件。
- 如需试验性修正，只能先把相关文件复制到 `/tmp/bangc_skill_audit_<时间戳>/`，在临时副本上修改、编译和运行；报告必须明确原版结果和临时修正版结果。
- 不得根据“MLU590”这个营销型号猜测编译架构、核心数、Cluster 数、NRAM/WRAM/SRAM 容量、对齐、任务维度或 intrinsic 支持。每项事实都必须给出命令、头文件、官方 sample/build 脚本或真实运行证据。
- 不得把静态推断、CPU 结果或其它型号设备结果写成 MLU590 实测。
- 命令失败也要保留真实 exit code、stdout 和 stderr，不得用推测补齐。
- 不要泄露密钥、token、账号、内部 URL 或无关环境变量；发现敏感值时只报告“已设置/未设置”，不要打印值。

## 审计步骤

### 1. 盘点第一版 Skill

1. 确认 `<V2_BANGC_ROOT>` 的绝对路径、目录结构和文件列表。
2. 计算所有 `.md`、`.py`、`.sh`、`.mlu` 文件的 SHA-256，保证审计对象可追溯。
3. 全文扫描第一版中的外部依赖、命令、头文件、库、环境变量、CNRT API、BANG C qualifier/intrinsic、CNCC 参数、CNPerf 参数和固定硬件假设。
4. 特别扫描以下残留或风险词：`triton`、`triton.language`、`tl.`、`torch`、`CUDA`、`num_warps`、`num_stages`、`MLUIR`、硬编码 NRAM 容量、硬编码 arch flag、旧 CNRT API 名称。
5. 列出每个 Skill 的 `SKILL.md` frontmatter 和所有 `.claude/skills/...` 内部路径，检查目标文件是否真实存在。

### 2. 操作系统与基础编译环境

采集并报告：

- `uname -a`、`uname -m`、`/etc/os-release`。
- CPU 架构、逻辑核数、内存。
- `python3 --version`、默认 shell。
- 可用的 `gcc/g++/clang/ld/cmake/make/ninja` 绝对路径和版本；不存在则写明。
- 当前进程的 `PATH`、`LD_LIBRARY_PATH`、`NEUWARE_HOME`、`CNTOOLKIT_HOME` 是否设置。路径类变量可以报告，敏感变量不得输出值。

### 3. MLU590 硬件与驱动

1. 定位并执行 `cnmon`；保留完整原始输出。
2. 解析并交叉验证：
   - 卡数量与 card id；
   - 完整设备名；
   - Driver、CNMON、Firmware 版本；
   - PCI Bus ID；
   - 总显存/已用显存；
   - FULL/虚拟化等运行模式；
   - 当前利用率和是否有其它任务占用。
3. 先查看 `cnmon --help`，再使用本版本真实支持的只读子命令采集更多设备属性；禁止编造不存在的选项。
4. 尽可能通过本机已安装的官方头文件、sample、只读工具或最小临时 probe 获取并记录以下信息及证据来源：
   - Cluster 数；
   - 每 Cluster 的 MLU Core 数；
   - 总可调度 Core 数；
   - 每 Core 的 NRAM/WRAM 容量；
   - 每 Cluster 的 SRAM 容量；
   - 支持的 `cnrtFunctionType_t`/Union 类型；
   - `cnrtDim3_t`/taskDim 的限制；
   - 向量/NFU 对齐要求。
5. 无法可靠获取的字段必须写 `null`/`UNKNOWN`，并说明已经尝试的证据路径。

### 4. NeuWare/CNToolkit 与工具链

逐项定位绝对路径、解析符号链接并报告版本：

- `cncc`
- `cnmon`
- `cnperf-cli`（以及存在时的 `cnperf-gui`）
- `cnas`
- `cngdb`
- `cnsanitizer` 或本机等价安全检查工具
- `cnpapi` 相关二进制、头文件和库

查找候选 NeuWare/CNToolkit 根目录时，至少检查：

- `NEUWARE_HOME`
- `CNTOOLKIT_HOME`
- `cncc` 的真实路径及其父目录
- `/usr/local/neuware`
- 本机其它真实安装路径

在候选根目录下定位并报告：

- `bang.h` 的全部位置，包括 `include/` 和 `lib/clang/*/include/`；
- `cnrt.h`；
- `libcnrt.so*` 的位置、真实链接目标和版本；
- libdevice 头文件/库/文档；
- BANG C 官方 samples；
- CMake/config/build 脚本。

### 5. API 与编译参数兼容性

直接检查本机 `cnrt.h`、`bang.h` 和官方 sample，确认第一版所用符号是否存在以及准确拼写/签名。至少核对：

- `CNRT_CHECK` 是否由 `cnrt.h` 定义；
- 成功返回值是 `cnrtSuccess`、`CNRT_RET_SUCCESS` 还是其它名称；
- Queue API 使用 `cnrtQueueCreate`/`cnrtQueueSync`/`cnrtQueueDestroy`，还是 `cnrtCreateQueue`/`cnrtSyncQueue`/`cnrtDestroyQueue`；
- `cnrtSetDevice`、`cnrtMalloc`、`cnrtMemcpy`、`cnrtFree`；
- `cnrtMemcpyHostToDev`、`cnrtMemcpyDevToHost`；
- `cnrtNotifierCreate`、`cnrtPlaceNotifier`、`cnrtNotifierDuration`；
- `cnrtDim3_t`、`cnrtFunctionType_t`、Block/Union 枚举；
- `__mlu_global__`、`__mlu_entry__`、`__mlu_func__`；
- `__nram__`、`__wram__`、`__sram__`；
- `taskId/taskDim` 及 X/Y/Z 变体；
- `__memcpy`、`__memcpy_async`、GDRAM/NRAM/SRAM 搬运方向；
- 第一版实际引用到的全部 `__bang_*` intrinsic。

执行 `cncc --version` 和 `cncc --help`，从真实输出或本机官方 MLU590 sample/build 脚本确定：

- MLU590 对应的 arch flag；
- arch flag 的两种可能写法及等价关系（若本机证据存在）；
- C++ 标准、include、library 与链接参数；
- `-S`、`-save-temps`、`-emit-llvm`、cnbin/fatbin 等中间产物参数是否支持。

arch 结论必须注明来源，例如“本机官方 build.sh 的 MLU590 分支”或“cncc help”；不得仅按设备名推断。

### 6. 第一版原样验证

先保持第一版完全不变，按其文档依次执行：

1. 对所有 Python 文件做语法编译检查。
2. 对所有 Shell 文件执行 `bash -n`。
3. 运行四个 Skill 目录的元数据/frontmatter 校验（若第一版附带验证命令则使用它）。
4. 运行：
   - `share/mlu/runtime/get_device_info.py`
   - `share/mlu/runtime/test_env_code.py`
5. 记录每条命令、exit code、stdout、stderr、实际使用的 cncc 命令和二进制路径。
6. 确认 smoke test 是否真的执行了 MLU kernel，并报告实际误差。

如果原样失败：

- 先定位唯一根因；
- 不修改第一版；
- 把相关文件复制到 `/tmp/bangc_skill_audit_<时间戳>/`；
- 只做能验证根因的最小临时改动；
- 重新编译和运行；
- 在报告中并列“原版失败”与“临时修正版结果”，给出精确 diff。

### 7. 真实 BANG C 编译、运行与精度链路

在原版通过或临时副本修正后，验证：

1. `.mlu -> cncc -> executable`。
2. CNRT 设备选择、Queue 创建、GDRAM 申请、H2D、kernel launch、Queue sync、D2H 和释放。
3. CPU reference 对比；报告 `atol`、`rtol`、`max_abs_error`、`max_rel_error` 和 pass/fail。
4. 非整 tile 尾部、零长度（若接口允许）、小尺寸和大尺寸至少各一个 case。
5. 所有 CNRT 调用和 kernel 异步错误都能导致非零退出。
6. 使用 notifier 做 kernel-only 计时，报告 warmup、repeat、统计量和单位。

不得为了通过测试而放宽容差、跳过 kernel、改用 CPU 输出或吞掉错误。

### 8. CNPerf、汇编与 MLISA

1. 查看 `cnperf-cli --help`，只使用本版本支持的真实命令。
2. 按第一版 `share/mlu/perf-analyzer/analyzer.sh` 的接口，对 smoke binary 运行一次最小 profiling。
3. 保存 CNPerf 原始文本，报告 record/kernel/monitor 等实际可用子命令、版本和输出文件。
4. 把真实 CNPerf 输出交给第一版解析器，报告解析到的 duration、counter、table 数量，并指出所有格式不匹配。
5. 若 `cncc --help` 确认支持，在临时目录用 `-S`、`-save-temps` 或等价参数生成中间产物；列出 `.s/.mlisa/.bc/.ll/.o/.cnbin/.cnfatbin` 等实际文件。
6. 运行第一版中间产物分析器，核对分类结果。
7. 不存在某工具或选项时记录 `UNAVAILABLE`，不得用其它平台工具伪装。

### 9. Worker 兜底链路

只做非敏感检查：

- `JOB_ID`、`JOB_ROOT`、`MLU_DEVICE_TYPE` 是否设置（不输出敏感值）；
- 第一版配置的 Agent-Service 地址是否可达；
- Worker 脚本能否显示 `--help` 并通过 Python 语法检查。

除非当前环境本来就属于受控 Job 且调用不会创建额外资源，否则不要实际提交 Worker Task。

## 必须返回的格式

### A. 审计结论

只允许：

- `READY`：第一版原样完成设备检测、编译、真实运行、精度与基本 profiling。
- `PARTIALLY_READY`：底层开发链路可用，但第一版存在可定位的不匹配，或优化工具部分缺失。
- `NOT_READY`：没有可用 MLU590，或基础 BANG C 编译/运行链路无法建立。

给出一句话判定依据。

### B. 事实表

至少覆盖硬件、驱动、NeuWare/CNToolkit、cncc、头文件、CNRT、arch、编译命令、运行、精度、notifier、CNPerf、MLISA、Python/Shell、Worker。

每行包含：`项目 | 实际值 | 状态 | 证据命令/文件`。

### C. 不匹配项

按 P0/P1/P2/P3 排序。每项必须包含：

- 第一版文件与行号；
- 第一版假设；
- 本机事实；
- 真实错误；
- 分类：环境配置 / Skill 假设 / SDK 版本差异 / 工具缺失 / 硬件限制；
- 第二版最小修改建议；
- 是否已在临时副本验证。

### D. 可直接用于第二版的 JSON

最后输出一个完整 JSON 代码块。未知值必须为 `null`，禁止猜测。至少使用以下结构：

```json
{
  "schema_version": 1,
  "audit_time": "ISO-8601",
  "skill_root": "absolute path",
  "system": {
    "os": null,
    "kernel": null,
    "arch": null,
    "python": null,
    "host_compiler": null
  },
  "device": {
    "count": null,
    "model": null,
    "driver": null,
    "firmware": null,
    "memory_bytes": null,
    "cluster_count": null,
    "cores_per_cluster": null,
    "total_cores": null,
    "nram_bytes_per_core": null,
    "wram_bytes_per_core": null,
    "sram_bytes_per_cluster": null,
    "facts_source": []
  },
  "toolchain": {
    "neuware_root": null,
    "neuware_version": null,
    "cncc_path": null,
    "cncc_version": null,
    "cnperf_path": null,
    "cnperf_version": null,
    "cnas_path": null,
    "bang_h_paths": [],
    "cnrt_h_path": null,
    "libcnrt_path": null,
    "environment": {
      "NEUWARE_HOME_set": false,
      "CNTOOLKIT_HOME_set": false,
      "neuware_bin_on_PATH": false,
      "neuware_lib_on_LD_LIBRARY_PATH": false
    }
  },
  "compiler": {
    "arch_flag": null,
    "arch_flag_source": null,
    "working_compile_command": null,
    "required_link_libraries": [],
    "supported_artifact_flags": []
  },
  "cnrt_api": {
    "success_symbol": null,
    "check_macro_in_header": null,
    "queue_create_symbol": null,
    "queue_sync_symbol": null,
    "queue_destroy_symbol": null,
    "notifier_symbols": [],
    "function_types": [],
    "memcpy_kinds": []
  },
  "validation": {
    "get_device_info": {"pass": false, "exit_code": null},
    "test_env_code": {"pass": false, "exit_code": null},
    "compile": {"pass": false, "exit_code": null},
    "run": {"pass": false, "exit_code": null},
    "accuracy": {
      "pass": false,
      "atol": null,
      "rtol": null,
      "max_abs_error": null,
      "max_rel_error": null
    },
    "notifier": {"pass": false, "median_us": null},
    "cnperf": {"status": null, "parser_status": null},
    "mlisa": {"status": null, "artifacts": []}
  },
  "mismatches": [
    {
      "priority": "P0",
      "file": null,
      "line": null,
      "assumption": null,
      "server_fact": null,
      "minimal_change": null,
      "temp_fix_verified": false
    }
  ]
}
```

### E. 原始证据附录

列出实际执行过的命令、exit code 和必要的 stdout/stderr 摘要。长输出可以给保存路径和 SHA-256，但关键版本、错误和成功标志必须直接展示。

审计完成后停止，不要修改第一版，不要直接生成第二版。让我把完整报告和 JSON 带回开发端。
