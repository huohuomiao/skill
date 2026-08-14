# BANG C StaticReviewer

## 职责

只读分析一个完整 `.mlu` 文件，不编译、不运行。修复仅限能从源码确定的缺陷；保留算子语义、Host API、Kernel 名称、reference、测试 Shape/dtype/stride 和容差。

先读取：

- `.claude/skills/share/mlu/references/platform-rules.md`
- `.claude/skills/share/mlu/references/primitives.md`
- 使用数学 intrinsic 时读取 `.claude/skills/share/mlu/references/libdevice.md`
- `ref/common_error.md`

共享清单不是当前 NeuWare 头文件的完整镜像。清单未列出的 API 标记为“需编译确认”，不得无证据删除。

## 输入输出

- 输入：一个现有 `.mlu` 文件。
- 输出：同目录固定文件 `bangc_code_fix.mlu` 与 `bangc_report.md`。
- 无确定缺陷时逐字复制输入，并写明 `no definite static repair`。

## 检查流程

### 1. 提取接口契约

记录：

- Kernel 名、入口限定符、参数类型/const/address space。
- Host wrapper、任务规模 `cnrtDim3_t`、function type、Queue 与 launch 参数。
- 输入输出的 dtype、shape、stride/layout、对齐、别名/in-place 关系。
- GDRAM/NRAM/WRAM/SRAM buffer 及其字节数、生命周期和复用关系。
- reference、用例、容差、精度失败退出条件。
- 源码要求的编译/链接参数。

契约含糊且修改可能改变语义时只报告，不改源码。

### 2. 检查后端残留和绕过

要求真实 BANG C Kernel 与 CNRT 启动路径。以下作为执行主体出现时属于确定问题：

- 其他设备后端的 Kernel/launch 语法。
- 用 Host 循环、框架算子或高层库替代用户要求的自定义 Kernel。
- 测试绕过 Kernel、复制 reference 到输出或预计算答案。

独立 CPU reference 允许且必须保留。

### 3. 检查 Kernel 与 launch ABI

逐项核对：

- launch 实参与 Kernel 形参的数量、顺序、标量宽度、指针 constness。
- `cnrtDim3_t` 三维均为合法正值；空输入在 Host 侧有明确分支。
- function type 与 Kernel 使用的硬件资源相容；无法从源码证明时标记动态确认。
- Queue 在 launch 前有效，结果被消费或资源释放前有检查过的完成点。
- Host/Device 指针没有混用，异步资源生命周期覆盖全部排队操作。

### 4. 检查任务划分与覆盖

追踪 `taskId/taskDim`、`taskIdX/Y/Z`、`taskDimX/Y/Z` 到每个逻辑输出：

- 每个有效元素恰好处理一次，除非算法明确需要原子或归并。
- `taskId` 与多维 ID 的线性化/反解一致。
- `data_per_task`、余数、最后 Task、tile 循环和尾块覆盖正确。
- Task 数被限制时，Kernel 内存在完整的步长循环；不得直接丢弃剩余工作。
- 多 Task 写同一输出时有经当前工具链确认的原子、workspace+二阶段归并或其他正确同步方案。

### 5. 检查存储空间与搬运

对每个 buffer 和搬运检查：

- `__nram__`、`__wram__`、`__sram__` 与实际用途一致；容量只按 EnvConfig/编译器证据判断。
- `__memcpy`/异步搬运方向匹配地址空间；第三个参数是字节数而非元素数。
- 片上 buffer 不重叠，偏移与对齐可证明，尾块不会读写越界。
- 向量 intrinsic 的处理长度、dtype、输入输出别名与对齐符合共享原语清单。
- 异步 load/compute/store 的 wait/sync 和 ping-pong buffer 生命周期正确；不得盲加同步。
- 只把有效字节写回 GDRAM，不能因对齐补齐而覆盖逻辑尾部。

### 6. 检查 Host CNRT 生命周期

确认：

- `cnrtSetDevice`、Queue 创建、分配、H2D、launch、同步、D2H、释放/销毁均检查返回值。
- H2D/D2H 方向和字节数正确，Host 与 Device buffer 大小匹配。
- 输出在需要累加/归并时按正确单位元初始化；不能一律清零。
- 每个失败路径都释放已获得资源。
- 不与安装头文件中的 `CNRT_CHECK` 宏冲突；旧返回码符号必须以当前 `cnrt.h` 为准。

### 7. 检查数值语义

- 低精度长归约使用契约要求的累加类型，通常需要更高精度中间值。
- 整数有符号/无符号、溢出、除法/取模与转换保持语义。
- 快速数学 intrinsic 只在原契约允许时使用；静态修复不得擅自引入近似。
- NaN/Inf、空归约、极值和单位元处理与 reference 一致。

### 8. 保护测试

确认 reference 独立，比较所有输出，固定 seed，覆盖小 Shape、非整 tile、尾块、非连续 stride（接口支持时）和边界值。测试失败必须返回非零。缺少明确检查时可补最小测试门禁，但不得改变期望结果。

## 修复原则

允许的确定性最小修复示例：

- 修正 Kernel/launch 参数不一致。
- 在乘法前将全局索引或字节数提升到足够位宽。
- 修正可证明错误的搬运方向、字节计数或尾块长度。
- 补齐 CNRT/launch/Queue 返回值检查。
- 修正任务步长使逻辑覆盖完整。
- 删除与当前 `cnrt.h` 冲突的自定义检查宏，改用头文件提供的接口。

禁止仅凭经验更换 intrinsic、function type、片上 tile 大小或架构参数。

## 报告

按 `ref/report_template.md` 追加：接口契约、确定缺陷、需动态确认项、精确改动和未执行门禁。静态检查不得设置最终 `passed=true`。
