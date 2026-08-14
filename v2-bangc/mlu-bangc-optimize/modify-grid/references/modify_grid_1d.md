# 情况 C：一维任务规模

## 正确映射

```cpp
for (int64_t tile = taskId; tile < total_tiles; tile += taskDim) {
  // tile 唯一确定逻辑输出范围
}
```

Host 的 `dim.x` 可以等于逻辑 Task 数，也可以由经验证的 policy 限制；后者必须依赖上面的完整步长循环。`dim.y`/`dim.z` 的具体单位值按当前 ABI 初始化。

## 调优候选

一次只调整一个：

- task 数与每 Task 循环次数。
- 每 Task 绑定一个输出行或一段连续 tile。
- 已确认合法的 function type。

不得同时改变片上 TILE。比较不同候选的负载尾部、CNPerf 原始设备并行证据和 notifier median。

## 边界

- `total_tiles=0` 时 Host 不启动。
- 全局线性索引使用足够位宽。
- 输出独占；若多个 Task 写同址，转入 reduction 方案。
- dim 和 function type 的合法范围来自 runtime/共享规则，不在示例中写常数。
