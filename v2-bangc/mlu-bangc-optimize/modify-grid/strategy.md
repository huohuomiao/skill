# BANG C 任务规模优化（保留目录名 modify-grid）

## 职责

分析 Host 侧 `cnrtDim3_t`、function type、Queue launch 与设备侧 task ID 映射，生成一个只改变任务划分的候选。目标是完整覆盖、核间负载均衡和较少尾批开销；不修改片上 tile、归约算法或数学表达式。

BANG C 的任务规模不是“必须等于物理 Core 数”。运行时可调度多批 Task；只有构造受控 persistent/task-stride 方案时才会限制任务数，并且 Kernel 必须用实际 `taskDim` 覆盖全部逻辑 tile。

## Step 1：提取启动结构

定位：

- Kernel 入口和 Host `<<<dim, function_type, queue>>>` launch。
- `cnrtDim3_t` 的 x/y/z 表达式。
- `cnrtFunctionType_t` 的来源。
- 设备侧 `taskId/taskDim`、三维 task ID/dimension、cluster/core 内建变量。
- 每个 Task 负责的逻辑轴、tile 和输出范围。

若只有 Kernel 定义、没有 Host launch，参照 `references/modify_grid_without_grid.md`：报告缺失契约，不凭猜测生成可宣称已验证的 launch。

## Step 2：分类

| 情况 | 特征 | 动作 |
| --- | --- | --- |
| A | 无法解析 Host launch | 仅给出待补信息，候选 `not_applicable` |
| B | 任务规模为单 Task | 仅在工作可独立划分时引入 task-stride 覆盖 |
| C | 一维任务规模 | 校验覆盖、尾部和负载；必要时调 Task 数 |
| D | 多维任务规模 | 校验 x/y/z 与逻辑轴对应，或安全线性化 |
| Reduce | 多 Task 可能写同一输出 | 改为输出独占或受支持的合并方案 |

参考文件名保持 v1 拓扑：`modify_grid_constexpr.md` 现在描述情况 B，而非某种语言编译期限定符。

## Step 3：合法性与覆盖证明

候选前必须证明：

1. 非空工作量的 dim 各维为正；空输入在 Host 侧不启动。
2. task ID 到逻辑 tile 的映射无漏算、无重复写。
3. 限制 Task 数时使用 `for (tile=taskId; tile<total_tiles; tile+=taskDim)` 或等价完整步长。
4. 多维线性化与反解使用同一维度顺序，乘法前提升位宽。
5. function type 由当前 SDK/设备属性和共享规则确认，不能盲目在 Block/Union 间切换。
6. Union 类约束、Cluster/Core 数和最大 dim 全部来自真实查询/官方规则；未知则不改。
7. 归约写冲突使用每输出独占、经确认支持的原子或 workspace+后续归并。

## Step 4：生成一个候选

只改变 task dimension/function type/mapping 中的一个因素。Host 与 Device 同步更新：

- `cnrtDim3_t` 与 policy 函数。
- Kernel 内 task 坐标、步长、线性化/反解。
- 必需的总 tile 数参数。
- 空 Shape、尾块和错误检查。

不得硬编码 MLU590 Core/Cluster 数或架构代号；使用 EnvConfig 已验证的 runtime query/policy helper。

## Step 5：验证

1. 相同 `cncc` 命令编译。
2. correctness 覆盖 taskDim 前后、非整分、单元素、空输入（接口允许）和大索引。
3. 归约候选验证高冲突/重复运行一致性。
4. 真实 MLU590 notifier 比较；CNPerf 可用于观察设备并行度，但原始输出格式不作假设。
5. 精度、资源、launch 或 no-regression 失败即回退。

性能提升必须来自真实测量，不能以“使用了全部 Core”代替证据。
