# EnvConfig

## 职责

验证 MLU590 BANG C/CNRT 的真实开发链路，而不是只检查 Python 包。必须采集设备与工具链事实，并完成 `.mlu → cncc → binary → MLU execution → accuracy` 向量加法 smoke test。

本地环境可用时直接执行；本地不可用且当前存在受控 Job 时，通过 `${BANGC_SKILL_ROOT}/mlu-bangc-main/subagents/scripts/submit_task_to_worker.py` 在当前 Job 的 MLU590 Worker 上执行同一组检查。

先接收并验证 `BANGC_SKILL_ROOT`。显式值优先；未提供时，取当前 `mlu-bangc-main/SKILL.md` 所在目录的父目录。该根下必须同时存在 `share/mlu`、`mlu-bangc-main/SKILL.md`、`mlu-bangc-code-gen/SKILL.md`、`mlu-bangc-code-review/SKILL.md` 和 `mlu-bangc-optimize/SKILL.md`；否则报告定位错误。显式根无效时直接 blocked，不再猜测其他根目录。

## 输入

| 来源 | 内容 |
|---|---|
| 调用方 | `output_dir` |
| Skill 根 | 已验证的 `BANGC_SKILL_ROOT` |
| 设备探测 | `${BANGC_SKILL_ROOT}/share/mlu/runtime/get_device_info.py` |
| BANG C smoke test | `${BANGC_SKILL_ROOT}/share/mlu/runtime/test_env_code.py` 与同目录 `bangc_vector_add.mlu` |
| Worker 兜底 | `${BANGC_SKILL_ROOT}/mlu-bangc-main/subagents/scripts/submit_task_to_worker.py` |
| 可选环境变量 | `CNCC`、`NEUWARE_HOME`、`CNTOOLKIT_HOME`、`BANGC_ARCH_FLAG`、`BANGC_ARCH`、`BANGC_CNCC_EXTRA_FLAGS` |

不要设置或持久修改系统环境变量。探测脚本可以为其子进程构造局部环境。

## 输出

| 文件 | 契约 |
|---|---|
| `{output_dir}/EnvConfig/config.md` | 结构化的人类可读环境决策、版本、路径、编译命令和验证结论 |
| `{output_dir}/EnvConfig/runtime_info.txt` | 实际执行的两段脚本 stdout/stderr 原文及退出码 |

返回摘要：

```json
{
  "status": "ready | blocked",
  "execution_backend": "local | worker | unavailable",
  "target_verified": true,
  "env_check_task_id": "local | <worker-task-id>",
  "config_path": "<output_dir>/EnvConfig/config.md",
  "runtime_info_path": "<output_dir>/EnvConfig/runtime_info.txt"
}
```

`target_verified` 只有在真实 MLU590 上 smoke test 编译、运行与精度全部通过时才可为 `true`。

## 步骤 1：本地设备与工具链探测

在仓库根目录顺序执行并同时捕获 stdout、stderr 和退出码：

```bash
python3 "${BANGC_SKILL_ROOT}/share/mlu/runtime/get_device_info.py"
python3 "${BANGC_SKILL_ROOT}/share/mlu/runtime/test_env_code.py"
```

检查项：

- `cnmon` 可执行且能识别至少一张目标 MLU590 设备。
- `cncc` 可执行，输出真实路径与版本。
- NeuWare/CNToolkit 根、`cnrt.h`、`libcnrt.so` 可定位。
- `bang.h` 可以由 CNCC 解析。不得只假设它位于 `$NEUWARE_HOME/include`；不同版本也可能放在 `lib/clang/<version>/include`。
- smoke source 使用 `<bang.h>`、`<cnrt.h>`、BANG C kernel、CNRT queue/memory/copy/launch API。
- 编译产物实际启动，执行 H2D、kernel、D2H，并与 host reference 比较。
- 最大误差满足 smoke test 固定阈值 `1e-5`。

2026-08-15 已审计的 MLU590-M9DG 环境中，两条命令在未设置 `NEUWARE_HOME`/`CNTOOLKIT_HOME`、`cncc` 不在原始 `PATH` 时仍通过：脚本自动定位 `/usr/local/neuware`、clang resource 目录中的 `bang.h` 和 `lib64/libcnrt.so`。这是兼容性基线；本次输出必须仍来自当前运行结果。

两条命令均为退出码 `0` 时，选择 `execution_backend=local`，进入步骤 3。

## 步骤 2：Worker 兜底

本地任一命令失败且 `JOB_ID` 非空时，在当前 Job 下前台同步执行：

```bash
python3 "${BANGC_SKILL_ROOT}/mlu-bangc-main/subagents/scripts/submit_task_to_worker.py" \
  --task-type custom \
  --workdir <仓库根目录绝对路径> \
  --timeout-sec 600 \
  --command "python3 '${BANGC_SKILL_ROOT}/share/mlu/runtime/get_device_info.py' && python3 '${BANGC_SKILL_ROOT}/share/mlu/runtime/test_env_code.py'"
```

规则：

- 禁止 `&` 后台执行或并发提交多条环境检查。
- 等待脚本退出；以退出码和 `stdout.log`、`stderr.log`、`result.json` 为准。
- Worker 成功时选择 `execution_backend=worker`，并记录真实 task id。
- `JOB_ID` 缺失时不调用 Worker 脚本；记录 `worker_context=not_controlled_job`。这不会否定已通过的本地链路，但在本地失败时必须选择 `execution_backend=unavailable`。
- Worker 失败或 Agent-Service 不可达时，选择 `execution_backend=unavailable`，写入真实错误后停止依赖运行结果的工作流。
- 禁止通过 Worker 安装或修改 NeuWare/CNToolkit、驱动、系统库、PATH 或共享依赖。

## 步骤 3：写入环境记录

将实际采用后端的完整日志写入 `{output_dir}/EnvConfig/runtime_info.txt`。若本地失败后改用 Worker，也保留本地失败摘要与 Worker 原始证据，不能覆盖失败来源。

使用以下结构生成 `{output_dir}/EnvConfig/config.md`：

```markdown
# MLU590 BANG C Environment

## Decision
- status: ready | blocked
- execution_backend: local | worker | unavailable
- target_verified: true | false
- env_check_task_id: local | <worker-task-id> | N/A
- timestamp: <ISO-8601>

## Device
- requested_device: MLU590
- detected_device: <cnmon 原始型号>
- card_id: <id>
- bus_id: <value-or-N/A>
- firmware: <value-or-N/A>
- driver_or_cnmon_version: <value-or-N/A>
- memory_total_mib: <value-or-N/A>
- memory_used_mib: <value-or-N/A>
- cnrt_attribute_source: current_probe | N/A
- cluster_count: <current-probe-value-or-N/A>
- mcore_per_cluster: <current-probe-value-or-N/A>
- nram_bytes_per_mcore: <current-probe-value-or-N/A>
- wram_bytes_per_mcore: <current-probe-value-or-N/A>
- sram_attr_name: <current-probe-raw-name-or-N/A>
- sram_attr_value_bytes: <current-probe-raw-value-or-N/A>
- sram_attr_scope: <confirmed-scope-or-unresolved>
- max_dim_xyz: <current-probe-value-or-N/A>
- max_cluster_count_per_union_task_raw: <current-probe-value-or-N/A>
- max_cluster_per_union_limit_task_raw: <current-probe-value-or-N/A>
- isa_version: <current-probe-value-or-N/A>

## Toolchain
- cncc_path: <path-or-N/A>
- cncc_version: <value-or-N/A>
- neuware_root: <path-or-N/A>
- neuware_env_mapping: export NEUWARE_HOME="<neuware_root>" | N/A
- neuware_version: <value-or-N/A>
- cnrt_header: <path-or-N/A>
- bang_header: <path-or-CNCC-resolved>
- libcnrt: <path-or-N/A>
- arch_flag: <verified-value | CNCC default>
- arch_flag_source: BANGC_ARCH_FLAG | BANGC_ARCH | CNCC default | N/A
- extra_flags: <value-or-empty>
- cnperf_path: <path-or-N/A>
- cnperf_version: <value-or-N/A>
- cnas_version: <value-or-N/A>

## Smoke test
- source: {BANGC_SKILL_ROOT}/share/mlu/runtime/bangc_vector_add.mlu
- compile_command: <真实命令>
- compile_pass: true | false
- run_pass: true | false
- accuracy_pass: true | false
- atol: 1e-5
- max_diff: <value-or-N/A>

## Provenance
- bangc_skill_root: <validated-absolute-path>
- runtime_info: EnvConfig/runtime_info.txt
- local_check_exit_codes: device=<code>, smoke=<code>
- worker_context: available | not_controlled_job | unreachable | N/A
- worker_result: <task-id/status | N/A>
- unavailable_reason: <N/A-or-reason>

## Audited baseline comparison
- baseline_date: 2026-08-15
- baseline_device: MLU590-M9DG
- baseline_match: true | false | unknown
- baseline_match_basis: <current identity fields used>
- audited_profile_source: {BANGC_SKILL_ROOT}/share/mlu/references/platform-rules.md#2026-08-15-真机审计基线
- observed_differences: <N/A-or-current-vs-baseline-differences>
```

任何未由本次运行实际检测的动态字段写 `N/A`，`cnrt_attribute_source=N/A`；不得把审计表中的 profile 值抄成当前 probe 值。`neuware_root` 是下游唯一的 NeuWare 根字段；其为已验证路径时，执行编译前将该值原样导出为 `NEUWARE_HOME`，并在 `neuware_env_mapping` 记录映射。

`baseline_match=true` 只表示当前原始证据同时确认 exact device model `MLU590-M9DG`、Firmware `v1.1.1`、Driver/CNMON `v6.5.26`、NeuWare `4.6.2` 和 CNCC `5.6.2`；缺任一身份字段时写 `unknown`，任一不同时写 `false`。它不表示 ISA、拓扑或容量已在本次重测。下游确需引用匹配 profile 的非冲突值时，必须标记 `source=audited_profile_2026-08-15` 并仍通过当前编译、精度与性能验证。审计附件对 `cnrtAttrSramSizePerMcore=2097152` 的属性名与“每 Cluster”文字口径不一致；在当前 `cnrt.h`/probe 确认 scope 前只保留 profile 原始属性名和值，不用它计算 tile 或 SRAM 预算。`MaxClusterCountPerUnionTask` 等调度属性同样仅保留 profile 原始值，不直接导出 launch policy。

## 兼容性规则

- 未设置架构参数时允许 CNCC 使用其默认值；不得从 `MLU590` 直接生成 `--bang-mlu-arch=...`。
- 如用户或环境提供 `BANGC_ARCH_FLAG`，只把它作为完整、已确认的参数传给 CNCC。
- `BANGC_ARCH` 仅接受已确认值，由脚本拼成相应参数；其确切形式必须以当前 `cncc --help` 或服务器审计为准。
- 2026-08-15 基线已验证 `--bang-arch=compute_50` 与 `--bang-mlu-arch=mtp_592` 等价，且 MLU590-M9DG 上显式两种参数和 CNCC 默认三条路径均通过。只有当前 `cncc --help`、当前 SDK sample 或环境配置再次确认时才可复用；不匹配时回退到当前动态事实或 CNCC 默认。
- 不定义名为 `CNRT_CHECK` 的宏，避免与新版 `cnrt.h` 冲突。smoke source 使用自有名称 `BANGC_CNRT_CHECK` 并以非零返回码判失败，从而避免依赖 `CNRT_RET_SUCCESS`/`cnrtSuccess` 的版本差异。
- 使用 C++ 标准库的 `.mlu` 编译链接至少保留 `-lcnrt -lstdc++ -lm -lpthread`；若当前工具链不接受，记录真实错误，留待当前环境验证，不静默删减。

## 下游执行

需要 CNCC 编译、MLU 精度验证或 benchmark 时，先读取 `config.md`：

- `local`：在当前环境运行原命令。
- `worker`：通过 Worker 脚本前台同步运行同一原命令。
- `unavailable` 或 `target_verified=false`：只允许静态分析，不得报告真机编译、精度或性能成功。

## 回退

| 场景 | 处理 |
|---|---|
| `cnmon` 不可用或无 MLU590 | 保留日志；仅在受控 Job 中尝试 Worker |
| `cncc`/头文件/库不可定位 | 保留日志；仅在受控 Job 中尝试 Worker |
| smoke 编译失败 | 保存完整编译命令和 stderr；仅在受控 Job 中尝试 Worker |
| smoke 运行或精度失败 | 保存 stdout/stderr/max_diff；仅在受控 Job 中尝试 Worker |
| Worker 也失败 | 生成 blocked 配置并停止动态步骤 |
| 仅性能工具缺失 | 基础环境仍可 ready；性能工具标记 `UNAVAILABLE` |

完成前检查 `config.md` 与 `runtime_info.txt` 均存在且非空，且决策可以从原始日志复核。
