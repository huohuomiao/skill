# 情况 A：缺少可解析的 Host launch

## 场景

源码只有 BANG C Kernel 入口，或 launch 被宏/外部翻译单元隐藏，无法确定 `cnrtDim3_t`、function type、Queue 和参数绑定。

## 处理

1. 提取 Kernel 签名和使用的 task 内建变量。
2. 列出生成 Host launch 仍缺少的事实：逻辑 Shape、任务映射、function type、Queue、设备查询与参数顺序。
3. 标记策略 `not_applicable`，逐字返回输入。
4. 可以给出非执行的接口骨架，但不得填硬件常数或宣称候选已验证。

```cpp
// 骨架：实际 dim/ktype 必须由调用契约和当前设备规则决定。
cnrtDim3_t dim = {/* x */, /* y */, /* z */};
cnrtFunctionType_t ktype = /* confirmed policy */;
kernel<<<dim, ktype, queue>>>(/* exact arguments */);
```

缺少 launch 属于输入完整性问题，必要时返回 code generation/review 补全，而不是在性能策略里猜测。
