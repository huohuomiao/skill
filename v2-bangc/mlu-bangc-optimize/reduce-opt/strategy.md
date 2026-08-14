# BANG C Reduce 类算子综合优化

## 职责

按原 v1 顺序执行七个归约相关策略。每个策略先做保守准入，只生成一个可解释候选；所有候选在最后统一按相同 `cncc`、correctness 和 MLU590 notifier 契约验证。片上容量、原子、归约 intrinsic、对齐和同步支持读取当前共享原语/平台文档，不得猜测。

| 顺序 | 策略 | BANG C 目标 |
| ---: | --- | --- |
| 1 | while 转 for | 规范化固定步长控制流，为编译器/流水分析提供清晰循环 |
| 2 | 归约轴循环优化 | 依据 NRAM 字节和 intrinsic 能力选择单批或分块局部归约 |
| 3 | 三维片上 tile 转二维 | 改善归约维布局和向量 intrinsic 处理形状 |
| 4 | 单次循环消除 | 消除由固定配置证明只执行一次的循环 |
| 5 | 冗余访存消除 | 复用等价 GDRAM2NRAM 搬运结果 |
| 6 | 全维度 Reduce 重构 | 单 Kernel 局部归约 + 合法跨 Task 合并，或 workspace 多阶段 |
| 7 | layout 变换消除 | 把仅为归约服务的 Host/额外设备转置融合进 Kernel 访问 |

任一策略不满足准入即跳过；通过后继续下一策略。代码改写期间不测试，统一在末尾验证以保持 v1 流程。

## 策略 1：固定步长 while 转 for

### 准入

只有能证明以下全部条件时改写：循环变量初值、终点、比较方向、固定净步长可确定；循环内无改变迭代次数的 break/return；循环后不依赖其最终值；有符号溢出/边界语义不变。

### 改写

```cpp
// before
int64_t begin = 0;
while (begin < reduce_size) {
  process(begin);
  begin += REDUCE_TILE;
}

// candidate
for (int64_t begin = 0; begin < reduce_size; begin += REDUCE_TILE) {
  process(begin);
}
```

仅凭 `for` 形式不能声称形成软件流水；是否改善由 CNCC/MLISA/CNPerf 和 notifier 证明。

## 策略 2：归约轴分块优化

### 准入

识别遍历归约轴的循环、输入 dtype、片上 buffer、局部归约路径和累加器。不得使用固定 `MAX_REDUCE_DIM`。是否能一次处理整轴由以下共同决定：

- 当前 Shape 的归约长度。
- 所有同时存活 NRAM/WRAM/SRAM buffer 字节。
- intrinsic 的 dtype、长度、对齐与形状约束。
- 编译器实际资源诊断。

### 候选

1. 整轴确实放得下且 intrinsic 支持：去掉分块循环，一次搬入、局部归约。
2. 否则保留分块：调整安全 `REDUCE_TILE`，复用 buffer，减少循环/搬运开销。
3. 低精度输入的跨 tile 累加使用契约要求的累加类型。

尾块必须用有效字节搬入；片上补齐区填对应归约单位元。参考 `references/reduce_dim_loop_opt_example.md`。

## 策略 3：三维片上 tile 转二维

### 准入

输入局部表示包含 `[A,R,B]`，在 R 上归约，且证据表明当前三维布局/中间轴归约效率低。必须能把 A 的单元工作映射为独立 Task/循环，并将局部 buffer 改为 `[R,B]`，同时保持 GDRAM stride、tail 和输出映射。

### 改写

- 让一个候选的 A tile 为 1，但不通过语言装饰器强制；用集中配置/任务映射表达。
- A 坐标变标量，局部 buffer 移除 A 维。
- R/B 的搬运、边界和局部归约适配二维 layout。
- 原归约“轴编号”不直接机械减一；应按所用 BANG intrinsic 的实际参数/布局重新构造。

参考 `references/reduce_3d_to_2d_opt_example.md`。片上字节降低不等于性能必然提升。

## 策略 4：单次循环消除

### 准入

集中配置或编译期常量能证明：循环起点为 0，正步长不小于上界，循环只执行一次；循环变量只用于当前 tile 地址/有效长度；无必须跨迭代累积的副作用。

### 改写

提升循环体，将循环变量替换为起点；只有能证明中间 accumulator 仅承载单次结果时才删除。runtime Shape 与配置关系不确定时跳过。

## 策略 5：冗余 GDRAM 搬运消除

### 准入

两次搬运只有同时满足以下条件才等价：

- GDRAM base、元素/字节 offset、有效字节、方向完全等价。
- 片上目标 buffer 的 dtype、布局和消费语义相容。
- 两次之间 GDRAM 未被写，片上首次结果未被覆盖或复用。
- 位于同一确定控制流和同一逻辑迭代。
- 异步搬运已完成，别名风险可排除。

### 改写

保留最早搬运并复用片上值；必要时延长 live range。重新核算片上峰值，若延长生命周期导致资源压力或性能回退则回退。softmax 多 pass 在整行无法驻留时通常不能直接消除后续读取。

## 策略 6：全维度 Reduce 重构

### 准入

baseline/reference 明确是输出标量的全维度可结合归约，目标代码块没有其他副作用，输入输出无危险别名。提取 sum/max/min/逻辑归约、dtype、单位元和 finalize（mean/norm 等）。

### 合并方案优先级

1. **每输出单 Task**：一个 Task 遍历全部 tile，适合工作量和并行度允许的场景。
2. **局部归约 + 已确认原子**：只有共享原语/当前头文件明确支持目标操作、dtype 和地址空间时使用；不要创造 atomic API 名。
3. **workspace + 第二阶段**：每 Task 写独立 partial，后续 Kernel 归并。完整 benchmark 必须包含 workspace 初始化/分配策略和所有阶段。

不强制“单 Kernel 一定更快”。原子 contention、额外初始化、Task 数和数值顺序均需实测。输出单位元按语义选择，不能一律为零。

## 策略 7：layout 变换消除

### 准入

Host 或前置设备阶段只为改变归约轴而执行二维 transpose/permute/contiguous 等价搬运，且中间结果无其他消费者。去除后能通过原始 shape/stride 构造正确的 GDRAM2NRAM 访问。

### 改写

- 删除专用于转置的中间 allocation/copy/Kernel。
- 调整归约逻辑轴、GDRAM stride 和片上布局。
- 若连续 NRAM 计算需要 gather/stride copy，使用当前 SDK 确认的接口；否则保守回退。
- benchmark 范围包含原方案的转置与新方案全部逻辑，避免只比较子 Kernel。

当前仅对可证明的二维交换自动改写；更高维复杂排列仅报告候选。

## 统一结果测试

七个策略遍历后：

1. 用同一 `cncc` 契约编译最终候选。
2. 运行全部 correctness；重点覆盖非整 tile、小/空归约、极值、NaN/Inf、低精度和多 Task 合并。
3. 可用时运行 CNPerf/MLISA 诊断。
4. 在真实 MLU590 上交错测量 baseline/candidate notifier 样本。
5. 任一精度/运行/资源失败，或性能未测/回退，输出原始 `input.mlu`。

最多三次仅修复本套策略引入的明确错误；不得改变 reference、容差或输入。报告逐项列出 applied/skipped/reverted 及证据。
