# 情况 D：多维任务规模

## 直接多维映射

当逻辑轴自然对应任务 x/y/z，且当前 function type 支持时，设备侧使用对应 task ID；Host `cnrtDim3_t` 的维度顺序必须与源码一致。

```cpp
int64_t i = taskIdX;
int64_t j = taskIdY;
int64_t k = taskIdZ;
```

具体内建变量可用性以当前 BANG C 头文件/编译器为准。

## 安全线性化

如果候选改为一维 task：

```cpp
for (int64_t flat = taskId; flat < total_tiles; flat += taskDim) {
  int64_t i = flat / (tiles_y * tiles_z);
  int64_t rem = flat - i * tiles_y * tiles_z;
  int64_t j = rem / tiles_z;
  int64_t k = rem - j * tiles_z;
  // 原 tile 处理逻辑
}
```

`total_tiles=tiles_x*tiles_y*tiles_z` 的乘法前使用足够位宽。反解顺序与 Host 维度顺序必须相同。

## 保留门禁

- 所有组合 tile 恰好覆盖一次。
- 不改变 tensor stride/layout。
- 线性化减少 launch/调度开销的结论来自 notifier；不能仅凭代码更短保留。
- function type 不在本策略中无证据更换。
