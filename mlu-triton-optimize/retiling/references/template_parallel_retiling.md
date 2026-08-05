**步骤 1**：确定最终 `block_size` 列表

接收与轴等长的 `block_size` 列表。同名 `block_size` 且在轴上相邻，表示这些轴合并为同一个 block：

```python
# 独立分块 — 相邻不同名
block_size = ["BLOCK_M", "BLOCK_N"]  # M 和 N 独立分块

# 合并分块 — 相邻同名
block_size = ["BLOCK_MN", "BLOCK_MN"]  # M 和 N 合并为一个 block

# 归约+并行 — K 归约独立，M/N 并行合并
block_size = ["BLOCK_MN", "BLOCK_MN", "BLOCK_K"]  # M、N 合并，K 独立
```

检测规则：遍历 `block_size` 列表，相邻同名的项归为一组（合并轴），不同名的项各自独立。

**步骤 2**：修改 Kernel 签名，保留输入、输出、shape、stride 等非分块参数，添加最终会用到的 `BLOCK_*: tl.constexpr`，删除旧的无用 `BLOCK_*`。

**步骤 3**：重写 Grid

Grid 必须是严格的一维 tuple。从 `block_size` 列表推导 grid 时，归约轴严禁参与 grid 划分：

```python
# block_size = ["BLOCK_MN", "BLOCK_MN", "BLOCK_K"]  → M、N 合并分块，K 独立分块
num_blocks_mn = triton.cdiv(M * N, BLOCK_MN)   # 合并轴组：组内维度相乘后再 cdiv
num_blocks_k = triton.cdiv(K, BLOCK_K)          # 独立轴组：各自 cdiv
grid = (num_blocks_mn * num_blocks_k,)          # 各组相乘 → 压成一维
```

**步骤 4**：重写 PID 解码

Kernel 内只使用 `pid = tl.program_id(0)`，弃用 `program_id(1)` 等多维索引。从 `block_size` 的独立 block group 逐级整除/取余还原各并行轴块索引：

```python
# block_size = ["BLOCK_M", "BLOCK_N"] → 独立分块
pid = tl.program_id(0)
num_blocks_n = (N + BLOCK_N - 1) // BLOCK_N
pid_m = pid // num_blocks_n
pid_n = pid % num_blocks_n

# block_size = ["BLOCK_MN", "BLOCK_MN"] → 合并分块，直接展开为一维
flat_mn = pid * BLOCK_MN + tl.arange(0, BLOCK_MN)
offs_m = flat_mn // N
offs_n = flat_mn % N
```

**步骤 5**：生成 Tile Offset 和 Mask，根据 pid 解码结果生成当前 tile 的逻辑 offset，并为越界位置生成 mask。

```python
# block_size = ["BLOCK_M", "BLOCK_N"] → 独立分块
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

# block_size = ["BLOCK_MN", "BLOCK_MN"] → 合并分块
mask = (offs_m < M) & (offs_n < N)
```
**重点强调**：mask 直接影响精度，要确保每个 load/store 正确添加 mask。

**步骤 6**：按 Tensor 独立构造地址，每个 tensor 必须按自己的真实轴顺序和 stride 构造地址，不能复用其他 tensor 的 mapping。

```python
# block_size = ["BLOCK_M", "BLOCK_N"] → 独立分块，2D 广播地址
x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

# block_size = ["BLOCK_MN", "BLOCK_MN"] → 合并分块，1D 线性地址
x_ptrs = x_ptr + offs_m * stride_xm + offs_n * stride_xn
y_ptrs = y_ptr + offs_m * stride_ym + offs_n * stride_yn
```

**重点强调**：如果 load/store 增加了维度，意味着 offsets 需要做维度扩展，不能随意确定新增维度的位置，要保证插入的顺序与在原始 tensor 中的位置一致。

**步骤 7**：重写 Load / Compute / Store，按目标算子的真实语义重写计算逻辑。

```python
# 两种分块方式的 load/compute/store 逻辑相同，仅 tile 形状不同
# 独立分块 → load 出 (BLOCK_M, BLOCK_N) 的 2D tile
# 合并分块 → load 出 (BLOCK_MN,) 的 1D tile
x_val = tl.load(x_ptrs, mask=mask, other=0.0)
out = x_val + 1.0
tl.store(y_ptrs, out, mask=mask)
```

**步骤 8**：同步 Wrapper Launch，若 Kernel 签名变了，wrapper 的 launch 实参和 meta 参数必须同步更新。

```python
# block_size = ["BLOCK_M", "BLOCK_N"] → 独立分块
BLOCK_M, BLOCK_N = 32, 64
num_blocks_m = triton.cdiv(M, BLOCK_M)
num_blocks_n = triton.cdiv(N, BLOCK_N)
grid = (num_blocks_m * num_blocks_n,)
kernel[grid](x, y, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

# block_size = ["BLOCK_MN", "BLOCK_MN"] → 合并分块
BLOCK_MN = 256
grid = (triton.cdiv(M * N, BLOCK_MN),)
kernel[grid](x, y, M, N, stride_xm, stride_xn, stride_ym, stride_yn,
             BLOCK_MN=BLOCK_MN)
```