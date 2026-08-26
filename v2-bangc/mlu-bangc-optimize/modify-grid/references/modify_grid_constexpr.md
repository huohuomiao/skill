# 情况 B：单 Task 改为可覆盖的多 Task

## 准入

仅当每个逻辑 tile 互相独立，或归并方案已证明正确时处理。首先计算 `total_tiles`，再由 Host policy 选择合法的 task dimension；选择不能凭硬件名称猜测。

## Device 映射

```cpp
__mlu_global__ void kernel(/* pointers, shapes */) {
  for (int64_t tile = taskId; tile < total_tiles; tile += taskDim) {
    int64_t begin = tile * TILE_ELEMS;
    int64_t valid = min((int64_t)TILE_ELEMS, total_elems - begin);
    // 搬运、计算、写回 valid 个元素。
  }
}
```

## Host 映射

```cpp
int64_t total_tiles = ceil_div(total_elems, (int64_t)TILE_ELEMS);
cnrtDim3_t dim = policy_for_independent_tiles(total_tiles, device_properties);
cnrtFunctionType_t ktype = confirmed_function_type;
kernel<<<dim, ktype, queue>>>(/* args */);
```

`policy_for_independent_tiles` 表示项目内经验证的 policy，不是虚构 API。

## 校验

- taskDim 大于 total_tiles 时空闲 Task 不访问内存。
- taskDim 小于 total_tiles 时步长循环覆盖全部工作。
- 尾 tile 只写有效字节。
- 单 Task 与多 Task 输出逐元素一致。
