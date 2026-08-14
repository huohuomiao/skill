# Reduction 任务规模

## 首选：每个输出由一个 Task 完成

对于行归约，优先让一个 Task 循环处理该行的全部归约 tile，再写一次输出。不同 Task 负责不同输出，避免跨 Task 同址写。

```cpp
for (int64_t out_id = taskId; out_id < output_count; out_id += taskDim) {
  Acc acc = identity;
  for (int64_t begin = 0; begin < reduce_size; begin += REDUCE_TILE) {
    // GDRAM -> local, local reduce, merge into acc
  }
  output[out_id] = finalize(acc);
}
```

## 跨 Task 归并

当单输出需要多个 Task 才有性能收益时，只允许：

1. 当前 SDK/原语表明确支持目标 dtype/操作/address space 的原子；或
2. 每 Task 写独立 workspace，使用第二阶段 Kernel/Host 调度完成归并。

不能假定任意 atomic 名称存在，也不能让多个 Task 普通写同一标量。workspace 必须记录大小、初始化、Queue 顺序与生命周期。

## 性能口径

若候选新增输出初始化或第二阶段 Kernel，notifier 范围必须包含完整逻辑调用。只测局部归约 Kernel 会产生虚假收益。

## 验证

- 单输出、高冲突、非整 tile、极值、NaN/Inf（契约适用）。
- 多次重复结果符合确定性契约。
- 单位元随 sum/max/min/逻辑归约和 dtype 正确变化，不能一律清零。
- MLU590 上完整调用无回退才保留。
