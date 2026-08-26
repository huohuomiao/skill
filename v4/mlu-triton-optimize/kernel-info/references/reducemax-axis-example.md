# Reducemax 轴分析例子

本例展示 2D 输入 `[M, N]` 沿 `dim=1`（N 轴）分块求 max。M 轴无分块参数，N 轴使用 `BLOCK_N` 分块 + for loop 做 tiled reduction，输出为 `[M]`。

## 示例代码

```python
@triton.jit
def max_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offsets = tl.arange(0, BLOCK_N)

    acc_val = float("-inf")
    for n in range(0, N, BLOCK_N):
        n_idx = n + n_offsets
        mask = (pid < M) & (n_idx < N)
        x = tl.load(x_ptr + pid * N + n_idx, mask=mask, other=-float("inf"))
        acc_val = tl.maximum(acc_val, tl.max(x, axis=0))

    tl.store(y_ptr + pid, acc_val)


def max_wrapper(input):
    M, N = input.shape
    output = torch.empty((M,), dtype=torch.float32, device=input.device)
    BLOCK_N = 32
    grid = (M,)
    max_kernel[grid](
        input,
        output,
        M,
        N,
        BLOCK_N,
    )
    return output


def test_max_wrapper():
    inp = torch.randn((32, 128), dtype=torch.float32, device="mlu")
    out = max_wrapper(inp)
    ref = torch.max(inp, dim=1).values
    torch.testing.assert_close(out, ref)
    return out
```

代表性测例：

```python
inp = torch.randn((32, 128), dtype=torch.float32, device="mlu")
```

根据 wrapper 中的 kernel launch，`x_ptr` 绑定到 wrapper 的 `input` 参数，在代表性测例中对应 `inp = torch.randn((32, 128), ...)`，contiguous 布局，因此真实 `shape` 是 `[32, 128]`，真实 `stride` 是 `[128, 1]`。

`y_ptr` 绑定到 wrapper 内部的 `output`，而 `output = torch.empty((M,), ...)`，真实 `shape` 是 `[32]`，真实 `stride` 是 `[1]`。
## Step 3 地址解析

`x_ptr` 的完整地址计算为：

```python
x_ptr + pid * N + n_idx

= x_ptr
  + pid * N
  + (n + tl.arange(0, BLOCK_N)) * 1
```

根据最后一层展开式，可以得到 `x_ptr` 的地址解析表：

| 逻辑轴 | offset | stride | loop |
|--------|--------|--------|------|
| `M` | `pid` | `N` | 无 |
| `N` | `n + tl.arange(0, BLOCK_N)` | `1` | `for n in range(0, N, BLOCK_N)` |

M 轴的 offset 为标量 `pid`，无 `tl.arange`、无 for loop 依赖。N 轴的 offset 依赖 for loop 迭代变量 `n`，分块从 `tl.arange(0, BLOCK_N)` 读取。

`y_ptr` 的完整地址计算为：

```python
y_ptr + pid

= y_ptr + pid * 1
```

| 逻辑轴 | offset | stride | loop |
|--------|--------|--------|------|
| `M` | `pid` | `1` | 无 |

## Step 4 block_size 与 has_loop

对 `x_ptr`：

| 逻辑轴 | block_size | has_loop | 判断依据 |
|--------|------------|----------|----------|
| `M` | `null` | `false` | offset 中无 `tl.arange(0, BLOCK_*)`，无 loop 依赖 |
| `N` | `BLOCK_N` | `true` | offset 中有 `tl.arange(0, BLOCK_N)`，且依赖 `for n in range(0, N, BLOCK_N)` |

对 `y_ptr`：

| 逻辑轴 | block_size | has_loop | 判断依据 |
|--------|------------|----------|----------|
| `M` | `null` | `false` | offset 中无 `tl.arange(0, BLOCK_*)`，无 loop 依赖 |

## Step 5 reduced_axis_set

kernel 沿 N 轴分块求 max：`tl.max(x, axis=0)` 对 tile 内 N 维度做 reduce 得到标量，`tl.maximum(acc_val, ...)` 逐块合并到标量 accumulator：

```python
reduced_axis_set = {"N"}
```

## 输出

```json
{
  "x_ptr": {
    "type": "input",
    "shape": [32, 128],
    "stride": [128, 1],
    "axis": ["M", "N"],
    "has_loop": [false, true],
    "axis_type": ["PARALLEL", "REDUCE"],
    "block_size": [null, "BLOCK_N"]
  },
  "y_ptr": {
    "type": "output",
    "shape": [32],
    "stride": [1],
    "axis": ["M"],
    "has_loop": [false],
    "axis_type": ["PARALLEL"],
    "block_size": [null]
  }
}
```
