# BANG C 分块与流水改写模板

以下是分析模板，不是可直接编译的固定实现。intrinsic 名称/签名、对齐、片上容量和同步 API 必须由当前 SDK 证实。

## 1. 选择逻辑 tile

对每个轴记录：

```text
axis: M, N
role: PARALLEL, PARALLEL
tile: TILE_M, TILE_N
stride: stride_m, stride_n
```

相邻且物理连续的轴才可合并为线性 tile。layout、broadcast、stride=0、切片或 gather/scatter 场景不能按 shape 猜连续性。

## 2. 在 Kernel 内集中声明片上 buffer

```cpp
#define TILE_ELEMS <compile_time_candidate>

__mlu_global__ void kernel(/* GDRAM pointers and shapes */) {
  __nram__ float input_nram[TILE_ELEMS];
  __nram__ float output_nram[TILE_ELEMS];
  // task loop follows
}
```

若使用 WRAM/SRAM，必须解释其用途和可见范围。记录每个 buffer 的字节表达式和 live range；不得超过由编译器/环境确认的容量。

## 3. Task 步长与 tile 起点

```cpp
for (int64_t tile = taskId; tile < total_tiles; tile += taskDim) {
  int64_t begin = tile * TILE_ELEMS;
  int64_t valid = min((int64_t)TILE_ELEMS, total_elems - begin);
  size_t valid_bytes = (size_t)valid * sizeof(float);
  // load / compute / store
}
```

所有乘法在足够位宽中进行。Task 数被限制时仍覆盖全部 tile。

## 4. 搬入与尾部

```cpp
__memcpy(input_nram, input_gdram + begin, valid_bytes, GDRAM2NRAM);
```

若向量 intrinsic 要求 `compute_elems > valid`：

1. 证明 `compute_elems` 不超过 buffer。
2. 用对应运算单位元初始化 `[valid, compute_elems)`。
3. intrinsic 使用 `compute_elems`。
4. 写回仍使用 `valid_bytes`。

不能从 GDRAM 多读补齐区来省略片上初始化。

## 5. 计算

```cpp
// 使用共享 primitives.md 中确认的 intrinsic；
// dst/src dtype、长度、对齐、in-place 约束必须匹配。
```

当没有合适向量 intrinsic 时，保留正确实现并报告，不编造 API。标量只可用于控制、尾部或当前 SDK 明确无向量路径的部分，不能把主体搬回 Host。

## 6. 写回

```cpp
__memcpy(output_gdram + begin, output_nram, valid_bytes, NRAM2GDRAM);
```

输入与输出地址/stride 不同时分别构造映射。不得复用另一个 tensor 的 offset。

## 7. Ping-pong 流水

候选结构：

```text
prologue: load tile 0 into ping
loop i:
  start load tile i+1 into pong
  compute ping
  store previous output
  wait at the dependency boundary
  swap ping/pong
epilogue: compute/store final tile
```

只有当前 SDK 已确认异步搬运和同步原语时实现。检查：

- 生产 buffer 在消费前完成。
- buffer 在上一消费者结束前不复用。
- 尾 tile 的有效长度随对应 buffer 传播。
- prologue/steady/epilogue 每个 tile 恰好一次。

## 8. 验证记录

报告 tile 前后值、片上峰值字节、搬运次数、pipeline 状态、编译器资源、全部精度用例和 MLU590 notifier 分布。未取得 CNPerf/MLISA 时写 `N/A`，不推导硬件利用率。
