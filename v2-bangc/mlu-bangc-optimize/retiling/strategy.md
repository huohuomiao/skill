# retiling：BANG C 片上分块优化

## 职责

分析 GDRAM↔NRAM/WRAM/SRAM 搬运与向量 intrinsic 的分块方案，生成一个只改变 tile/buffer 布局的候选。目标是提高连续搬运和片上复用、减少冗余 GDRAM 访问，并在有证据时建立 load/compute/store 流水。

平台容量、对齐和异步 API 读取共享 MLU 文档与当前 SDK；不得写死硬件值。

## 步骤 1：提取 Kernel 信息

按 `kernel-info/strategy.md` 取得：

- 每个 tensor 的 shape/stride/axis/role。
- task ID 到逻辑 tile 的映射。
- 每个 `__memcpy` 的方向、有效字节和片上 buffer。
- 每个 buffer 的 storage space、dtype、字节表达式和 live range。
- 当前 tile、循环、tail、intrinsic 长度和 pipeline 结构。

无法证明 shape、stride、dtype 或地址空间时不做改写。

## 步骤 2：核算片上布局

为每个候选建立字节布局表：

| buffer | space | bytes expression | live range | reusable with | alignment evidence |
| --- | --- | --- | --- | --- | --- |

按生命周期峰值计算同时存活字节；只有能证明互斥时才复用。编译器未提供容量事实时不以经验常数扩张 tile。

## 步骤 3：选择一个分块变化

一次只能选一个：

1. 调整连续轴每批元素/字节。
2. 合并相邻连续逻辑轴为一段线性搬运。
3. 把跨 stride 访问改为合法的 stride 搬运/片上重排。
4. 复用生命周期不重叠的片上 buffer。
5. 从单 buffer 改为经当前 SDK 证明正确的 ping-pong 流水。
6. 将重复 GDRAM 读取保留在片上（容量和生命周期可证明时）。

不得同轮改变 task dimension、function type、归约算法和数学近似。

## 步骤 4：改写搬运/计算/写回

参考 `references/template_parallel_retiling.md`：

- GDRAM 有效字节与片上对齐长度分开。
- 对齐补齐区在参与 intrinsic 前按数学单位元初始化。
- `__memcpy` 方向与地址空间一致。
- intrinsic 的 dtype、长度、对齐和别名约束来自当前原语表。
- 写回只覆盖有效输出字节。
- 异步搬运必须有与 buffer 复用相匹配的 wait/sync。

## 步骤 5：验证

1. 用相同 `cncc` 契约编译，记录明确的片上资源诊断。
2. 运行全部 correctness，重点覆盖小 Shape、非整 tile、对齐前后、非连续 stride 和 in-place（接口支持时）。
3. 在真实 MLU590 上公平测量 notifier 样本。
4. 可用时用 CNPerf/MLISA 说明 GDRAM 访问、流水或指令变化；格式未知时引用原始证据。
5. 精度、资源、运行或 no-regression 任一门禁失败即回退。

不允许仅凭“更大 tile”或“用了双缓冲”声明优化成功。
