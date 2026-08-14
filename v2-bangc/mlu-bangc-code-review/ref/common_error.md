# BANG C/CNRT 常见错误目录

只有满足前置条件且可从源码证明时才自动修复；否则标记“需编译/运行确认”。

| 类别 | 典型错误 | 后果 | 修正原则 |
| --- | --- | --- | --- |
| Kernel/launch ABI | 参数个数、顺序、标量宽度或 const 不一致 | 编译失败或错误地址 | 逐项匹配入口签名与 launch |
| 平台残留 | 执行路径仍调用其他后端 | 无法在目标栈运行或绕过 Kernel | 保留 BANG C+CNRT 主体 |
| 搬运单位 | 将元素数当作 `__memcpy`/CNRT copy 字节数 | 少拷贝、越界或精度错误 | 使用有效元素数 × `sizeof(T)`，防溢出 |
| 搬运方向 | Host/Device 或 GDRAM/片上方向错误 | 非法访问或错误数据 | 根据源/目的地址空间选方向 |
| Task 覆盖 | 余数、最后 Task 或步长循环错误 | 漏算、重复写或越界 | 证明每个逻辑元素覆盖一次 |
| 片上布局 | buffer 重叠、越界、生命周期估计错误 | 编译资源错误或数据破坏 | 用字节布局和生命周期核算 |
| 尾块 | 对齐后长度直接读写有效区外 | OOB 或覆盖邻接数据 | 片上补齐与 GDRAM 有效字节分离 |
| intrinsic/dtype | 签名、dtype、长度、对齐不受支持 | 编译失败或结果错误 | 读取共享原语表和当前头文件 |
| Queue/生命周期 | launch 后未同步即 D2H/free/destroy | 异步失败或悬空资源 | 使用检查过的相关 Queue 完成点 |
| function type/dim | 非法任务规模或资源类型不匹配 | launch 失败或不可预测行为 | 依据 runtime 属性和官方约束选择 |
| 归约合并 | 多 Task 普通写同一输出 | 数据竞争 | 每输出独占、支持的原子或 workspace 归并 |
| 检查宏/返回码 | 重定义 `CNRT_CHECK`、使用当前头文件无此符号的旧成功常量 | 编译警告/错误 | 以安装的 `cnrt.h` 为准 |

## 关键检查示例

### 字节数必须显式

```cpp
size_t bytes = static_cast<size_t>(count) * sizeof(float);
CNRT_CHECK(cnrtMemcpy(dst, src, bytes, direction));
```

乘法前提升位宽。设备侧搬运同样区分 `valid_bytes` 与对齐后的片上计算长度。

### Task 步长必须覆盖全部 tile

```cpp
for (int64_t tile = taskId; tile < total_tiles; tile += taskDim) {
  // tile -> logical coordinates; handle tail explicitly
}
```

若 Host 只启动有限 Task 而 Kernel 每个 Task 仅处理一个 tile，剩余 tile 会永久丢失。

### launch 与完成检查

```cpp
kernel<<<dim, ktype, queue>>>(...);
// 使用当前 CNRT/编译器提供的 launch error 机制；
// 在读取结果或释放资源前检查对应 Queue 的完成状态。
```

API 拼写与返回类型必须来自当前头文件，不在 Skill 中猜测版本差异。

### 尾块搬运

```cpp
int64_t begin = tile * tile_elems;
int64_t valid = min(tile_elems, total - begin);
size_t valid_bytes = static_cast<size_t>(valid) * sizeof(T);
__memcpy(local, global + begin, valid_bytes, GDRAM2NRAM);
// 若 intrinsic 要求更长的对齐长度，先初始化片上补齐区；写回仍用 valid_bytes。
```

### reference 不得成为替代实现

Host reference 可以计算期望值，但测试输出必须来自真实 CNRT launch+D2H。打印 `FAIL` 后返回 0、只比较前缀或把 reference 拷贝到输出均视为无效测试。
