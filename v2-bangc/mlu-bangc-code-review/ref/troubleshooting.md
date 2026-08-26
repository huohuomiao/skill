# BANG C/CNRT 故障排查

始终从最早失败门禁和第一条根诊断开始。修改前保存完整命令与原始输出。

## 环境与编译器

| 证据 | 分类 | 动作 |
| --- | --- | --- |
| `cncc` 不可用 | 环境 | 回到 EnvConfig，不改源码 |
| MLU/驱动/运行库不可用 | 环境 | 标记 blocked |
| 找不到 `bang.h` | SDK 布局或环境 | 让 EnvConfig 同时检查标准 include 与 Clang resource include；不盲加路径 |
| 不支持的架构参数 | 构建契约 | 从 `cncc --help`/官方构建/EnvConfig 取值，不猜 |
| undefined reference | 链接 | 核对 CNRT、C++、math、thread 等实际依赖 |
| 检查宏重定义/旧返回码未声明 | SDK 版本 | 读取当前 `cnrt.h`，使用其宏与符号 |

环境失败不能通过改 Kernel 消除。

## `cncc` 编译诊断

先修第一条带源码位置的错误：入口/函数限定符、地址空间、Host/Device 调用边界、intrinsic 签名、dtype、头文件、模板/常量表达式。资源溢出只有在编译器明确给出 NRAM/WRAM/SRAM 所需值和上限时才归为资源问题。

不要为通过编译删除测试、Queue 检查或自定义 Kernel。

## CNRT 与 launch

- 核对设备选择、Queue 创建、allocation、copy、launch、sync、D2H、free/destroy 的首个失败返回值。
- 参数无效：检查空指针、字节数、方向、dim、function type 与 ABI。
- 异步失败可能在 Queue 同步处暴露；向前定位该 Queue 上第一个 launch/copy。
- 资源释放前必须有建立完成关系的 Queue 操作。
- 发生设备上下文破坏类错误后不要继续 benchmark。

## 搬运与片上存储

1. 写出源/目的地址空间与有效范围。
2. 将元素范围转换为字节范围，乘法前使用足够位宽。
3. 分离 GDRAM 有效字节和片上对齐长度。
4. 检查 NRAM/WRAM/SRAM buffer 布局、复用、ping-pong 索引和生命周期。
5. 对异步搬运确认等待发生在消费前、复用发生在上一轮结束后。
6. 片上容量未知时不得填写经验常数；以编译器和 EnvConfig 为准。

## Task 映射与精度

先用最小失败 Shape 找第一个错误输出，再核对：

1. `taskId/taskDim` 与多维 task ID 映射。
2. 每 Task 起点、步长、余数、尾 tile。
3. 输入/output stride 和 layout。
4. 输出初始化与完整覆盖。
5. 多 Task 写同一地址的归并方案。
6. 归约单位元、累加 dtype、顺序与特殊值。
7. intrinsic 的 dtype、长度、对齐和 in-place 约束。

禁止缩小 Shape、删用例、修改 seed、放宽容差或让 reference 复用被测索引。

## 性能与正确性隔离

代码审查阶段只要求正确。notifier、CNPerf 或 MLISA 可以帮助定位，但不能取代编译/运行/精度门禁，也不能用 profiler 计时声称修复成功。性能修改交给 `mlu-bangc-optimize`，并重新建立公平 baseline。
