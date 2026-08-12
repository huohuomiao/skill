# Softmax 轴分析例子

本例展示 `N/H/W` 布局下沿 `W` 轴做 softmax 的情形。kernel 内将 `N/H` 两个逻辑轴合并为一维 `NH` 行索引，并使用同一个 `BLOCK_NH` 分块参数；`W` 轴使用 `BLOCK_W`，并在 softmax 计算中发生 reduce。

## 示例代码

```python
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    H,
    W,
    BLOCK_NH: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    nh_offsets = pid * BLOCK_NH + tl.arange(0, BLOCK_NH)[:, None]
    w_offsets = tl.arange(0, BLOCK_W)[None, :]

    mask = (nh_offsets < N * H) & (w_offsets < W)

    input_index = nh_offsets * W + w_offsets
    x = tl.load(input_ptr + input_index, mask=mask, other=-float("inf"))

    max_val = tl.max(x, axis=1)[:, None]
    exp_val = tl.exp(x - max_val)
    sum_val = tl.sum(exp_val, axis=1)[:, None]
    y = exp_val / sum_val

    output_index = nh_offsets * W + w_offsets
    tl.store(output_ptr + output_index, y, mask=mask)


def softmax_wrapper(input):
    N, H, W = input.shape
    output = torch.empty_like(input)
    BLOCK_NH = 16
    BLOCK_W = triton.next_power_of_2(W)
    grid = (triton.cdiv(N * H, BLOCK_NH),)
    softmax_kernel[grid](
        input,
        output,
        N,
        H,
        W,
        BLOCK_NH,
        BLOCK_W,
    )
    return output


def test_softmax_wrapper():
    inp = torch.randn((4, 8, 64), dtype=torch.float32, device="cuda")
    out = softmax_wrapper(inp)
    ref = torch.softmax(inp, dim=2)
    torch.testing.assert_close(out, ref)
    return out
```

代表性测例：
```python
inp = torch.randn((4, 8, 64), dtype=torch.float32, device="cuda")
```

根据 wrapper 中的 kernel launch，`input_ptr` 绑定到 wrapper 的 `input` 参数，在代表性测例中对应 `inp = torch.randn((4, 8, 64), dtype=torch.float32, device="cuda")`，contiguous 布局，因此真实 `shape` 是 `[4, 8, 64]`，真实 `stride` 是 `[512, 64, 1]`。

`output_ptr` 绑定到 wrapper 内部的 `output`，而 `output = torch.empty_like(input)`，因此 `output_ptr` 的真实 `shape` 也是 `[4, 8, 64]`，真实 `stride` 也是 `[512, 64, 1]`。

## Step 3 地址解析

`input_ptr` 的完整地址计算为：

```python
input_ptr + input_index

input_index
= nh_offsets * W + w_offsets

= (pid * BLOCK_NH + tl.arange(0, BLOCK_NH)) * W
  + tl.arange(0, BLOCK_W) * 1
```

根据最后一层展开式，可以得到 `input_ptr` 的地址解析表：

| 逻辑轴 | offset | stride | loop |
|--------|--------|--------|------|
| `N` | `pid * BLOCK_NH + tl.arange(0, BLOCK_NH)` | `W` | 无 |
| `H` | `pid * BLOCK_NH + tl.arange(0, BLOCK_NH)` | `W` | 无 |
| `W` | `tl.arange(0, BLOCK_W)` | `1` | 无 |

`N/H` 共享同一个融合后的 `NH` offset 和 stride `W`，对应同一个 `BLOCK_NH` 分块参数。contiguous 布局下 `stride_h = W`、`stride_w = 1`，地址公式中 `W` 即 H 轴步长，`1` 为 W 轴缺省步长。`output_ptr` 的地址解析同 `input_ptr`。

## Step 4 block_size 与 has_loop

对 `input_ptr`：

| 逻辑轴 | block_size | has_loop | 判断依据 |
|--------|------------|----------|----------|
| `N` | `BLOCK_NH` | `false` | offset 中有 `tl.arange(0, BLOCK_NH)`，没有依赖 for loop |
| `H` | `BLOCK_NH` | `false` | offset 中有 `tl.arange(0, BLOCK_NH)`，没有依赖 for loop |
| `W` | `BLOCK_W` | `false` | offset 中有 `tl.arange(0, BLOCK_W)`，没有依赖 for loop |

`output_ptr` 同理。

## Step 5 reduced_axis_set

softmax 沿 `W` 轴计算，包含两次 reduce：

```python
max_val = tl.max(x, axis=1)
sum_val = tl.sum(exp_val, axis=1)
```

`axis=1` 对应 tile 中的 `W` 维度，因此：

```python
reduced_axis_set = {"W"}
```

## 输出

```json
{
  "input_ptr": {
    "type": "input",
    "shape": [4, 8, 64],
    "stride": [512, 64, 1],
    "axis": ["N", "H", "W"],
    "has_loop": [false, false, false],
    "axis_type": ["PARALLEL", "PARALLEL", "REDUCE"],
    "block_size": ["BLOCK_NH", "BLOCK_NH", "BLOCK_W"]
  },
  "output_ptr": {
    "type": "output",
    "shape": [4, 8, 64],
    "stride": [512, 64, 1],
    "axis": ["N", "H", "W"],
    "has_loop": [false, false, false],
    "axis_type": ["PARALLEL", "PARALLEL", "REDUCE"],
    "block_size": ["BLOCK_NH", "BLOCK_NH", "BLOCK_W"]
  }
}
```
