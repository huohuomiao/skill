# BANG C 离线配置生成（保留目录名 gen-autotune-config）

## 职责

BANG C 源码没有本 Skill 可依赖的运行时装饰器调优机制。本策略保留 v1 目录名，但将行为改为：生成有限的 Host/Kernel 编译期配置候选，逐个使用相同 `cncc` 契约编译并在真实 MLU590 上运行，最后只把一个经过门禁的配置冻结进源码。

## 工作流程

1. 提取轴、片上 buffer、任务映射、测试 Shape/dtype/stride 与当前配置。
2. 构造安全候选表。
3. 每个候选独立编译、精度验证和 notifier 测量。
4. 选择全用例无回退的 best-so-far；无可靠测量时保留输入。
5. 输出单一固定配置，不留下运行时搜索循环。

## Step 1：提取信息

读取 [get-tensor-axis-info.md](references/get-tensor-axis-info.md) 并生成：

```json
{
  "axes": [
    {"name": "M", "role": "PARALLEL", "stride": 1, "tile_symbol": "TILE_M", "has_loop": true}
  ],
  "buffers": [
    {"name": "input_nram", "space": "NRAM", "dtype": "float", "bytes_expr": "TILE_M * sizeof(float)", "live_range": "load_to_compute"}
  ],
  "launch": {"dim_expr": "...", "function_type": "..."}
}
```

字段是分析结果 schema，示例值不是目标设备事实。

## Step 2：形成候选参数

可调项仅包括源码已经暴露或能用最小改动引入的：

- `TILE_*`/每批元素数。
- 单 Task 循环处理的 tile 数。
- `cnrtDim3_t` 的合法任务规模。
- 已由当前 SDK 证明支持的 function type。
- intrinsic 向量长度/对齐后的处理长度。
- 单/双 buffer 或有限流水级数。

约束：

1. 片上字节总量按 buffer 生命周期和复用计算，不简单相加所有声明。
2. 容量上限必须来自 EnvConfig、编译器或设备查询；未知时不据此扩张候选。
3. 连续访问、搬运粒度和 intrinsic 对齐要求来自当前原语/平台文档。
4. 归约轴必须保持语义和单位元；不能仅为减少循环把整轴塞入未知容量。
5. 候选总数保持有限，优先当前配置附近和证据指向的方向。

## Step 3：实例化候选

使用一个集中配置区，避免在算法中散落常量：

```cpp
#ifndef TILE_ELEMS
#define TILE_ELEMS <candidate_value>
#endif

#ifndef PIPELINE_STAGES
#define PIPELINE_STAGES <candidate_value>
#endif
```

或使用 Host policy 函数返回任务规模/function type。不得把 runtime Shape、设备容量或 Core 数错误写成测试机常量；设备相关值应由已验证查询/策略取得。

## Step 4：编译、验证和冻结

每个候选：

1. 生成独立 `candidate_<id>.mlu`。
2. 用相同 compiler/flags/link 语义编译。
3. 运行所有 correctness 用例。
4. 在真实 MLU590 上按相同 notifier 口径采样。
5. 记录 CNCC 资源诊断；可用时引用 CNPerf/MLISA，不要求固定输出格式。

只有精度通过、目标确认、性能已测且无回退的候选可获胜。最后把单一最佳常量/policy 写入 `bangc_optimized.mlu`，删除搜索脚手架；完整搜索表保存在报告。

## 边界

- 不硬编码候选范围、架构 flag、片上容量或核心数量。
- 不用 Host reference 时间选择设备配置。
- 不把一次最小耗时当赢家；使用 median 和预设噪声门限。
- 不修改测试、容差或 reference。
- 没有目标设备实测时输出原输入，性能字段为 `N/A`。
