# index-computation-simplify

## 职责概述

该策略文档用于消除 Triton kernel 中 `tl.load` / `tl.store` 地址计算里的冗余 index 运算。典型模式是：kernel 先构造一个 flat index（如 `pid * BLOCK_SIZE + tl.arange(...)`），再把它拆成多维索引（如 `// N`、`% N`），最后按 stride 加权组合成地址偏移。若目标 tensor 的 stride 与该拆分方式严格匹配，使得"拆分 + 重组"等价于原始 flat index（还原为 `f` 或 `base + f`），则中间计算属于冗余，应当消除。

## 修改部分

处理时以 `tl.load` / `tl.store` 为粒度独立判断地址计算是否可化简，对能够严格证明与原表达式等价的部分进行改写，否则保持输入不变，仅修改 kernel 函数中与 `tl.load` / `tl.store` 地址计算直接相关的表达式，不修改函数签名、wrapper、grid 配置及其他无关代码，同时删除无用的中间变量。

## 前置知识

判断 `tl.load` / `tl.store` 的地址计算能否化简，核心不是只看 shape，或某一维 stride 是否为 1，而是要同时看 **逻辑形状** 与 **物理存储** 是否仍保持同一套线性关系。更具体地说，需要判断：把 flat index 拆成多维索引后，再按当前 tensor 的 stride 规则重建地址，是否还能严格还原为原始线性下标。可按下面原则统一判断：

- **视图操作** 通常不改底层存储，只改变逻辑解释方式；因此是否可把“拆分后的多维索引”还原为原始 flat index，必须重新检查新的 size/stride 映射，不能默认成立。
- **拷贝操作** 可能重新生成物理存储；若结果 tensor 变为新的连续布局，则其地址映射应按新的 stride 重新判断。
- 若 `//`、`%` 承担的是**必要的逻辑索引变换**，而不是“拆分后又按同一线性规则重组”的中间步骤，则不能化简。

以下情况通常不能直接还原为原始的 flat index，除非能进一步证明在当前 size/stride 映射下两者严格等价：

- `transpose` / `permute` / 非标准 stride 对应的重排访问
- `expand`、broadcast 或 stride 为 0 的地址映射
- 带步长的切片、子采样、带洞访问、非连续子视图
- gather / scatter / 间接寻址等非直接仿射访问
- 构造 flat index 时采用的逻辑形状，与实际目标 tensor 的逻辑形状或 stride 语义不一致

## 优化步骤

### 步骤 1：预过滤

只有当代码中同时满足以下两个条件时才继续处理，任一条件不满足，就直接返回原始代码：

- 出现 `tl.load` 或 `tl.store`
- 出现 flatten index 的构造迹象，如 load/store 中的多维 offset 是由同一张量通过 `// N`、`% N` 方式获得

示例如下：

```python
f = pid * BLOCK + tl.arange(0, BLOCK)  # 构造连续的 flat index
row = f // N  # 由 f 拆出行索引，属于地址重组的中间步骤
col = f % N   # 由 f 拆出列索引，属于地址重组的中间步骤
x_ptrs = x_ptr + row * stride_x0 + col * stride_x1  # 用多维索引和 stride 重新拼出访存地址
x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # 按重组后的地址执行加载
```

### 步骤 2：Stride 分析

从测试代码中分析每个相关 tensor 的 shape 和 stride：

1. 定位 tensor 的创建方式和操作链
2. 根据张量操作链推断 stride 变化
3. 提取每个 `tl.load` / `tl.store` 的地址表达式、相关stride 参数和 index 变量
4. 记录候选 flat index 与拆分变量

### 步骤 3：判定与改写
对每个 `tl.load` / `tl.store` 逐项检查：

1. 确认存在已知 flat index 变量 `f`
2. 确认多维索引由 `f` 拆分得到（`// N`、`% N` 等）
3. 将 stride 代入地址表达式进行数学化简，确认是否可部分或完全还原为 `f`

满足 1-3 即视为化简候选，随后完成以下检查与改写：

4. **逐一确认**待删除的中间变量不被 kernel 中任何其他语句引用
5. 检查 mask 是否可直接保留；若不可直接保留，则仅在能够严格证明等价的前提下进行改写。若 mask 依赖于待删除变量，应同步重写其表达式。
6. 记录改写方案：原始表达式 → 代入 stride 后的化简过程 → 替换后的 flat index → 变量处理 → mask 处理
7. 说明 `//`、`%` 是"地址重组中间步骤"还是"必要的逻辑索引变换"

**注意**不能直接化简成原始 `f` 不等于完全不能做任何代数化简，若某些场景能部分地化简成更简单的仿射表达式仍需改写。

例如原始代码片段为：

```python
f = pid * BLOCK + tl.arange(0, BLOCK)  # 构造连续的 flat index

row = f // N  # 由 f 拆出行索引，属于地址重组的中间步骤
col = f % N   # 由 f 拆出列索引，属于地址重组的中间步骤

x_ptrs = x_ptr + row * stride_x0 + col * stride_x1  # 用多维索引和 stride 重新拼出访存地址
x_mask = (row < M) & (col < N)  # 基于拆出的 row/col 构造边界 mask

x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # 按重组后的地址执行加载
```

其中：

```
stride_x0 = N 且 stride_x1 = 1，所以：
row * stride_x0 + col * stride_x1 = f
```

改写后的代码片段为：

```python
f = pid * BLOCK + tl.arange(0, BLOCK)  # 保留 flat index，作为改写后的直接地址基准

x = tl.load(x_ptr + f, mask=(f < M * N), other=0.0)  # 将地址化简为 flat 形式，并同步将 mask 改写为等价的一维边界条件
```

### 步骤 4：验证生成代码，输出最终结果

运行生成的代码，保证其能够正确执行，且精度测试通过。如果有错误，根据错误信息进行调试修改。若性能有回退，则输入代码保持原样，作为最终输出结果。

