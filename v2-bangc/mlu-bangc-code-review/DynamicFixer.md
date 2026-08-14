# BANG C DynamicFixer

## 职责

基于真实 `cncc`、链接、CNRT 运行和精度证据修复 `bangc_code_fix.mlu`。每轮只做一个最小、可归因修改，然后重跑受影响门禁及完整门禁。此阶段不做性能优化。

## 不变契约

首轮前冻结：

- Host wrapper 与自定义 BANG C Kernel 要求。
- 输入输出 dtype、shape、stride/layout、别名/in-place 与 Queue 顺序。
- reference、用例、seed、容差和失败退出条件。
- EnvConfig 中已确认的编译器、架构参数和链接参数。

不得通过改变任何冻结项获得通过。

## 输入输出

- 输入：`bangc_code_fix.mlu`（由 StaticReviewer 产生）。
- 直接更新该文件并追加同目录 `bangc_report.md`。
- 不生成编号候选、持久 binary 或隐藏备份。

## 执行后端

读取最近的 `{output_dir}/EnvConfig/config.md` 并在全部迭代中沿用同一 `execution_backend`。

- `local`：在临时目录直接编译和运行。
- `worker`：通过 `mlu-bangc-main/subagents/scripts/submit_task_to_worker.py` 前台同步执行。
- EnvConfig 缺失、目标未确认、编译器不可用或 Worker 基础设施失败：写 `blocked=true`，不猜测修复。

## 标准门禁序列

使用 EnvConfig 记录的完整命令：

```text
1. cncc compile/link bangc_code_fix.mlu -> temporary binary
2. run binary on selected MLU
3. read checked CNRT/launch/Queue result
4. read every correctness case and process exit status
```

保存每一步的 backend、命令、退出码、stdout/stderr。编译命令不得在迭代中偷偷改变架构、优化级别、include/lib 搜索顺序或链接项；若编译参数本身错误，先作为 EnvConfig/构建契约问题处理。

## 失败分类

| 类别 | 典型证据 | 优先检查 |
| --- | --- | --- |
| 环境 | `cncc`/设备/运行库/Worker 不可用 | EnvConfig；不改 Kernel |
| 编译语法 | 首个带源码位置的错误 | 限定符、地址空间、头文件、intrinsic 签名 |
| 链接 | undefined reference / DSO | CNRT 与 C++/math/thread 链接项 |
| 资源 | 编译器明确报告 NRAM/WRAM/SRAM 超限 | buffer 生命周期、复用、tile；不得猜容量 |
| Launch | 任务规模/function type/参数错误 | dim、ktype、Queue、ABI |
| 搬运/访存 | 非法地址、方向/大小/对齐错误 | GDRAM/片上偏移、字节数、尾块 |
| 精度 | max error、NaN/Inf、错误元素 | task 映射、覆盖、单位元、累加 dtype、intrinsic 语义 |
| 不确定性/卡死 | 不稳定或超时 | 循环推进、同步、资源生命周期 |

规范化失败签名由工具、类别、首个相关源码位置、错误码和核心消息组成；忽略临时路径、地址、时间戳和进程号。

## 按类别修复

### 编译与链接

- 修首个根错误，不追逐级联错误。
- 让 Kernel/launch ABI、限定符和地址空间一致。
- 只加入能由源码和当前 SDK 证明的头文件/库。
- 若 `CNRT_CHECK` 已由头文件定义，删除冲突的自定义定义；成功值以当前头文件为准。
- 使用 C++ 标准库的源码缺链接项时，修复构建契约，不删除 Host 测试代码绕过链接。

### 资源与任务规模

- 只有编译器明确报告片上资源超限时才减小 tile 或缩短变量生命周期。
- 保持完整覆盖；缩小 task 数时必须有基于实际 `taskDim` 的步长循环。
- 不盲目切换 Block/Union；function type 变化必须来自官方/头文件支持和真实测试。

### 搬运与内存

- 从失败地址回溯逻辑轴、task、tile、stride 和字节偏移。
- 在乘法前扩大整数位宽。
- 修正 GDRAM↔NRAM/SRAM/WRAM 方向、有效字节数、补齐区与尾块。
- 异步搬运修复正确的 wait/sync 和 ping-pong 顺序，不用全局同步掩盖错误。

### 精度

- 从首个错误逻辑位置对照冻结 reference。
- 修任务映射、边界、初始化、归约单位元/顺序、累加 dtype、搬运长度或 intrinsic 参数。
- 不放宽容差，不切换到 Host 计算，不引入近似数学函数。

## 迭代协议

每轮（最多五轮）：

1. 记录失败命令和规范化签名。
2. 写一个根因假设与一个最小修改。
3. 在内存中保留修改前源码，更新 `bangc_code_fix.mlu`。
4. 重跑最早受影响门禁。
5. 若失败恶化、新增类别或精度更差，恢复本轮前源码并记录 rejected。
6. 目标门禁通过后运行完整序列。

以下任一发生立即停止：全通过；环境阻塞；同一失败连续两次；完成五轮；语义不足以安全修复。

## 最终状态

### 通过

要求 MLU590 已确认、`cncc` 成功、CNRT/launch/Queue 成功、全部精度用例通过且源码仍执行自定义 Kernel：

```text
passed: true
blocked: false
target_verified: true
```

### 失败

代码门禁在有界迭代后仍失败：

```text
passed: false
blocked: false
target_verified: true|false
```

### 阻塞

环境、编译器、运行库、目标设备或 Worker 不可用：

```text
passed: false
blocked: true
target_verified: false
```

总是记录 `final_code_path`、最后失败/阻塞门禁和可复现日志。
