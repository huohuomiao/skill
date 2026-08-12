# 多维轴 Flatten 分析例子

本例展示将 `N/H/W` 多个逻辑轴的 block 摊平成一维 `flat_idx`，再通过整除、取余恢复各轴块索引的情形。

## 示例代码

```python
@triton.jit
def flatten_kernel(
    input_ptr,
    output_ptr,
    stride_in0,
    stride_in1,
    stride_in2,
    stride_out0,
    stride_out1,
    stride_out2,
    N,
    H,
    W,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    num_blocks_h = tl.cdiv(H, BLOCK_H)
    num_blocks_w = tl.cdiv(W, BLOCK_W)
    total_blocks = num_blocks_n * num_blocks_h * num_blocks_w
    for flat_idx in range(pid, total_blocks, num_programs):
        w_idx = flat_idx % num_blocks_w
        h_idx = (flat_idx // num_blocks_w) % num_blocks_h
        n_idx = flat_idx // (num_blocks_h * num_blocks_w)

        n_offset = n_idx * BLOCK_N
        h_offset = h_idx * BLOCK_H
        w_offset = w_idx * BLOCK_W

        n_i = tl.arange(0, BLOCK_N)[:, None, None]
        h_i = tl.arange(0, BLOCK_H)[None, :, None]
        w_i = tl.arange(0, BLOCK_W)[None, None, :]
        mask = (
            (n_offset + n_i < N)
            & (h_offset + h_i < H)
            & (w_offset + w_i < W)
        )

        input_index = (
            (n_offset + n_i) * stride_in0
            + (h_offset + h_i) * stride_in1
            + (w_offset + w_i) * stride_in2
        )
        x = tl.load(input_ptr + input_index, mask=mask, other=0.0)
        out_val = tl.where(x > 0, x, 0.0)

        output_index = (
            (n_offset + n_i) * stride_out0
            + (h_offset + h_i) * stride_out1
            + (w_offset + w_i) * stride_out2
        )
        tl.store(output_ptr + output_index, out_val, mask=mask)

def flatten_wrapper(input):
    N, H, W = input.shape
    output = torch.empty_like(input)
    BLOCK_N = 4
    BLOCK_H = 8
    BLOCK_W = 32
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    num_blocks_h = triton.cdiv(H, BLOCK_H)
    num_blocks_w = triton.cdiv(W, BLOCK_W)
    total_blocks = num_blocks_n * num_blocks_h * num_blocks_w
    grid = (total_blocks,)
    flatten_kernel[grid](
        input,
        output,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        N,
        H,
        W,
        BLOCK_N,
        BLOCK_H,
        BLOCK_W,
    )
    return output


def test_flatten_wrapper():
    inp = torch.randn((32, 64, 256), dtype=torch.float32, device="cuda")
    out = flatten_wrapper(inp)
    ref = torch.relu(inp)
    torch.testing.assert_close(out, ref)
    return out
```

代表性测例：

```python
inp = torch.randn((32, 64, 256), dtype=torch.float32, device="cuda")
```

根据 wrapper 中的 kernel launch，`input_ptr` 绑定到 `inp`，所以 `input_ptr` 的真实 `shape` 是 `[32, 64, 256]`，真实 `stride` 是 `[16384, 256, 1]`。

`output_ptr` 绑定到 wrapper 内部的 `output`，而 `output = torch.empty_like(input)`，因此 `output_ptr` 的真实 `shape` 也是 `[32, 64, 256]`，真实 `stride` 也是 `[16384, 256, 1]`。

## Step 3 地址解析

`input_ptr` 的完整地址计算为：

```python
input_ptr + input_index

input_index
= (n_offset + n_i) * stride_in0
  + (h_offset + h_i) * stride_in1
  + (w_offset + w_i) * stride_in2

= (n_idx * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_in0
  + (h_idx * BLOCK_H + tl.arange(0, BLOCK_H)) * stride_in1
  + (w_idx * BLOCK_W + tl.arange(0, BLOCK_W)) * stride_in2

= ((flat_idx // (num_blocks_h * num_blocks_w)) * BLOCK_N + tl.arange(0, BLOCK_N)) * stride_in0
  + (((flat_idx // num_blocks_w) % num_blocks_h) * BLOCK_H + tl.arange(0, BLOCK_H)) * stride_in1
  + ((flat_idx % num_blocks_w) * BLOCK_W + tl.arange(0, BLOCK_W)) * stride_in2
```

根据最后一层展开式，可以得到 `input_ptr` 的地址解析表：

| 逻辑轴 | offset | stride | loop |
|--------|--------|--------|------|
| `N` | `(flat_idx // (num_blocks_h * num_blocks_w)) * BLOCK_N + tl.arange(0, BLOCK_N)` | `stride_in0` | `for flat_idx in range(pid, total_blocks, num_programs)` |
| `H` | `((flat_idx // num_blocks_w) % num_blocks_h) * BLOCK_H + tl.arange(0, BLOCK_H)` | `stride_in1` | `for flat_idx in range(pid, total_blocks, num_programs)` |
| `W` | `(flat_idx % num_blocks_w) * BLOCK_W + tl.arange(0, BLOCK_W)` | `stride_in2` | `for flat_idx in range(pid, total_blocks, num_programs)` |

这里 `num_programs = tl.num_programs(0)`；`flat_idx` 是一维块索引，`N/H/W` 三个逻辑轴的块索引都由 `flat_idx` 通过整除、取余恢复得到。

`output_ptr` 的地址解析同 `input_ptr`，只是 stride 表达式替换为 `stride_out0/stride_out1/stride_out2`。

## Step 4 block_size 与 has_loop

对 `input_ptr`：
| 逻辑轴 | block_size | has_loop | 判断依据 |
|--------|------------|----------|----------|
| `N` | `BLOCK_N` | `true` | offset 中有 `tl.arange(0, BLOCK_N)`，且依赖一维 `flat_idx` 循环 |
| `H` | `BLOCK_H` | `true` | offset 中有 `tl.arange(0, BLOCK_H)`，且依赖一维 `flat_idx` 循环 |
| `W` | `BLOCK_W` | `true` | offset 中有 `tl.arange(0, BLOCK_W)`，且依赖一维 `flat_idx` 循环 |

`output_ptr` 同理。

## Step 5 reduced_axis_set

该 kernel 没有 reduce 操作：

```python
reduced_axis_set = set()
```

## 输出

```json
{
  "input_ptr": {
    "type": "input",
    "shape": [32, 64, 256],
    "stride": [16384, 256, 1],
    "axis": ["N", "H", "W"],
    "has_loop": [true, true, true],
    "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
    "block_size": ["BLOCK_N", "BLOCK_H", "BLOCK_W"]
  },
  "output_ptr": {
    "type": "output",
    "shape": [32, 64, 256],
    "stride": [16384, 256, 1],
    "axis": ["N", "H", "W"],
    "has_loop": [true, true, true],
    "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
    "block_size": ["BLOCK_N", "BLOCK_H", "BLOCK_W"]
  }
}
```
