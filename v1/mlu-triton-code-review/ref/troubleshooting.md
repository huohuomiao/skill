# Triton 通用故障排查与修复指南

本文只保存可跨后端复用的 Triton 排错方法。RTX 3090 的设备、launch/grid、shared memory、寄存器、occupancy 和 CUDA 后端规则读取 `.claude/skills/share/gpu/references/platform-rules.md`。

## 精度问题

### 逻辑运算符

Tile 条件使用逐元素逻辑运算，避免 Python 标量语义：

```python
mask = (offsets < size) & (values > 0)
```

### `tl.load` 的 `other`

为越界位置选择不会参与有效计算的填充值。最大值归约常用 `-inf`，最小值归约常用 `inf`，普通求和可用 0。

### Load 与 Store mask

读取 mask 可包含输入过滤条件；写回 mask 只描述合法输出位置。不要因为某个输入无效而漏写本应产生的输出。

### 累加 dtype

低精度输入进行长归约时，把累加器提升到 fp32，并在输出边界处按需求转换。

### 精度验证

- 使用原始 PyTorch reference，不重写 reference 来适配错误 Kernel。
- 保留用户提供的 `atol`、`rtol` 和测试 Shape。
- 同时记录 `max_diff`、NaN/Inf 位置和失败输入。

## 编译与原语错误

1. 确认原语名称、参数和 Triton 版本一致。
2. 确认目标后端支持该原语与 dtype。
3. 避免在 Kernel 内使用普通 Python 动态对象或运行时不可解析的控制流。
4. NVIDIA GPU 目标额外读取 `.claude/skills/share/gpu/references/primitives.md`。

## Kernel 接口不一致

- Kernel 定义、wrapper 调用和 autotune/heuristics 中的参数名必须一致。
- 不要在 Kernel 内直接引用未传入的模块级变量。
- 编译期常量通过 `tl.constexpr` 显式传入。
- 不要同时在 autotune config 和 Kernel 内重新定义同一个 Block 参数。

```python
@triton.jit
def kernel(x_ptr, y_ptr, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    ...

kernel[grid](x, y, alpha=0.1, BLOCK_SIZE=128)
```

## 数据类型错误

- 检查输入、常量、中间值、累加器和输出的 dtype 链路。
- 混合类型计算时显式转换，避免依赖不同后端可能不同的隐式提升。
- 索引计算优先使用足够位宽的整数类型，但必须符合目标后端原语支持表。

## 内存越界与非法访问

### 基址与程序编号

```python
offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
value = tl.load(x_ptr + offsets, mask=mask)
tl.store(y_ptr + offsets, value, mask=mask)
```

检查以下问题：

- 是否遗漏 `pid * BLOCK_SIZE`。
- Shape 不是 Block 整数倍时是否正确使用 mask。
- 广播后的地址表达式是否与 Tile 形状一致。
- Stride、转置和 Flatten 后的地址是否仍指向预期元素。
- Persistent Kernel 是否覆盖每个逻辑块且没有重复写冲突。

## 调试与定位流程

1. 先运行最小合法 Shape，确认能够编译和启动。
2. 分离环境错误与业务错误；环境不可用时不要修改 Kernel。
3. 固定随机种子，保存首个失败输入。
4. 检查 NaN/Inf，再检查最大误差位置。
5. 缩小到单个 Kernel、单组输入和单个失败分支。
6. 每轮只做一个可解释的修改，重新执行相同测试。
7. 精度通过后再进行性能比较。

## 禁止的替代式修复

- 禁止用 PyTorch/CPU 实现替代 Triton Kernel。
- 禁止跳过 Kernel 调用或伪造成功输出。
- 禁止缩小用户测试 Shape、放宽误差阈值或删除失败用例。
- 禁止用逐元素标量 Kernel 规避本应保留的块并行逻辑。
