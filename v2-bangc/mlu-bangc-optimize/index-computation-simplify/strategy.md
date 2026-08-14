# BANG C 地址与索引计算简化

## 职责

减少设备 Kernel 热循环中的重复地址计算、整数除法/取模和循环不变量，同时保持 task 覆盖、stride/layout、别名、位宽和尾块语义。本策略不改变 task dimension、function type、片上 tile、归约树、intrinsic 或编译 flags。

## Step 1：规范化地址

对每个 GDRAM/NRAM/WRAM/SRAM 访问写成：

```text
base + axis0*stride0 + axis1*stride1 + ... + constant
```

记录变量来源：task ID、Kernel 标量参数、循环 induction variable、编译期常量、runtime shape/stride、片上 buffer offset。明确表达式单位是元素还是字节。

先检查 CNCC/MLISA 是否已经完成同类消除。生成代码无变化且源码更复杂时回退。

## Step 2：候选模式

### 2.1 循环不变量外提

```cpp
int64_t row_base = row * stride;
for (int64_t k = 0; k < K; ++k) {
  value = input[row_base + k];
}
```

乘法前使用足够位宽；stride 语义必须可证明。

### 2.2 指针递增

连续 stride 且无别名/越界风险时，可用局部指针递增替代每轮乘加。不得跨越合法对象边界，也不得把非连续 layout 当连续。

### 2.3 复用商与余数

```cpp
int64_t q = flat / N;
int64_t r = flat - q * N;
```

只有 divisor 为正、取整语义一致且生成代码证明确实减少指令时保留。多层坐标反解可复用中间商，但不能改变维度顺序。

### 2.4 编译期常量强度削减

真正固定的 tile/stride 可通过宏或模板参数暴露给编译器。runtime Shape 不得冻结为测试常量。

### 2.5 元素/字节换算合并

把重复的 `count*sizeof(T)` 或 buffer base 计算移出热循环，但需检查 `size_t/int64_t` 溢出和 `__memcpy` 第三个参数的字节语义。

## Step 3：拒绝条件

- 依赖未证明的 shape、stride、整除、对齐或连续性。
- signed overflow、除零、负数除法/取模语义变化。
- Host/Device 索引宽度或维度顺序不一致。
- in-place/alias 导致 load/store 顺序有语义。
- 改变 GDRAM 访问顺序并引入多 Task 写冲突。
- 同时改变 task dimension、tile 或数学计算。

## Step 4：验证

1. 只改地址/索引表达式生成 `candidate.mlu`。
2. 用相同 `cncc` 契约编译；可用时对比 MLISA 中相关整数指令。
3. 运行所有 Shape，重点覆盖非整 tile、大索引、非连续 stride、0/1 边界和 alias/in-place。
4. 真实 MLU590 notifier 比较 median/p20/p80。
5. correctness、运行安全或 no-regression 任一失败即回退。

报告必须记录 expression before/after、等价证明、整数位宽、生成代码证据和 keep/revert 原因。
