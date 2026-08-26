# MLU Triton kernel 获取 tensor 轴信息详解

本文档提供了获取 Triton kernel 中所有 tensor 轴信息的详细方法和示例。

## 返回格式

请以字典格式返回结果：

```python

{
    "a":
    {
        "axis": ["M", "N"],
        "axis_type": ["PARALLEL", "REDUCE"],
        "stride": [256, 1],
        "block_size": ["BLOCK_M", "BLOCK_N"],
        "has_loop"：[True, False],
    },
    "b":
    {
        "axis": ["M", "N"],
        "axis_type": ["PARALLEL", "REDUCE"],
        "stride": [256, 1],
        "block_size": ["BLOCK_M", "BLOCK_N"],
        "has_loop"：[True, False],
    }
}
```

**字段说明：**
- `a`/`b`: tl.load 或 tl.store 的 base ptr 名称
- `axis`: tensor 轴的名称
- `axis_type`: 轴类型，有 "PARALLEL" 和 "REDUCE" 两种情况，"PARALLEL" 表示并行轴，"REDUCE" 表示reduce轴
- `stride`: 轴的 stride 大小，**真值通过输入测例获取或计算得到**
- `block_size`: 在轴上的切分块名称
- `has_loop`: 在轴上是否有 for loop

**重要提示：**
- 返回的结果中，轴的顺序需要按照 stride 从大到小排序
- `axis_type`，`stride`，`block_size`，`has_loop` 的元素数量一致，且一一对应

## 分析步骤

按下列步骤依次处理 kernel 中的 `tl.load` 或 `tl.store` 指令，并将结果填入值

### Step 1: 展开 offsets
load/store 的第一个输入可表示为 base_ptr + offsets，base_ptr 是基地址；offsets 是总的地址偏移，可表示成多维 offset 和 stride 的线性组合，即 offsets = ∑ (offset_i * stride_i)，offset_i 是第 i 维的偏移，stride_i 是第 i 维的步长。将 offset_i 展开为 base_ptr 与基础变量（如 `tl.program_id`，`tl.arange`，shape 参数，stride 参数）的组合。

**注意**：如果展开后里面没有 tl.arange，说明访存的标量数据，后续不做处理。
**注意**：在解析加法表达式时，不要将每个加法项独立作为一项，根据上下文判断加法项是否指向同一个轴，这些项可能共同组成一个offset
**注意**: 没有显示 stride_i 的时候，stride_i 等于 1，stride_i 必须是 kernel 参数或者参数乘积
**注意**: offset_i 中一般是起始地址加切分块，必须包含 tl.arange 表达式

### Step 2: 依次分析 offset_i

offset_i 能是以下元素的组合，各元素代表含义：

- tl.program_id(n)：表示该维度在第 n 个 pid 上有拆分，且该轴一定是并行轴
- tl.arange(0, BLOCK_SIZE)：表示该轴的切分块大小为 BLOCK_SIZE

**其它需要仔细判断的情况**：

1. **offset_i 跟 for 循环上的迭代索引相关时**
此时应该展开 for 循环上 range(start, end, step) 的参数，如果 start 或 end 参数展开之后与 tl.program_id(n) 相关，表示这个轴在线程 n 上有拆分，且该轴一定是并行轴，另外如果 start 或 end 或 step 参数展开之后，与 BLOCK SIZE 有关，说明该 BLOCK_SIZE 对应的 has_loop 为 true；如果 start 和 end 参数与 tl.program_id(n) 无关，只跟某些轴的形状有关，例如 range(0, N, BLOCK_N)，大概率是reduce轴。step 可能与某个 BLOCK_SIZE 参数有关，表示该轴的拆分块大小很可能是 BLOCK_SIZE，对应的 has_loop 选项为 true。

2. **axis type判断**
   **判断规则**：
   - 若 axis 的 offset 与 pid 相关，则 axis 的 type 是 "PARALLEL"
   - 若 axis 的 offset 与 for 循环迭代索引相关，且 range 参数跟 pid 无关，且在 for 循环迭代完成之后，且对应的轴被 reduce 掉了，则该 axis 的 type 就是 "REDUCE"

3. **has_loop判断**
has_loop 与 BLOCK SIZE是一一对应的关系。查看kernel内所有for循环，将for循环上 range函数的start, end, step参数完全展开为基础变量（如 `tl.program_id`，`tl.arange`，shape 参数，stride 参数）组合，如果里面跟某个BLOCK SIZE有关，那么其对应的has_loop为true，如果BLOCK SIZE没有跟任意for循环上的参数相关，那么has_loop为false。
4. **flatten轴的分析**

当某个轴拆分到某个grid上，其offset一般与对应pid相关，但常见地，为保证负载均衡，通常会有多个并行轴共同拆分到同一个grid。

```python
pid = tl.program_id(0)
total_blocks = num_blocks_n * num_blocks_w * num_blocks_h
for flat_idx in range(pid, total_blocks, num_programs):
    h_idx = flat_idx % num_blocks_h
    w_idx = (flat_idx // num_blocks_h) % num_blocks_w
    n_idx = flat_idx // (num_blocks_w * num_blocks_h)
```

如上所示，pid0上同时拆了n，w，h，flat_idx是3个块摊平后的总索引，然后通过取整取余方式来反算出各维度索引。这种情况，每一维的 axis_type，has_loop 都继承摊平后的维度。

## 示例

### 示例1：一维简单拆分

```python
@triton.jit
def test_kernel_1(
    inp,
    mid,
    n_elements,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < n_elements
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
    all_val = tl.reduce(inp_val != 0, axis=0, combine_fn=reduce_all)
    mid_ptr = mid + pid
    tl.store(mid_ptr, all_val)

def test1(inp):
    n_elements = inp.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    mid_size = triton.cdiv(n_elements, block_size)
    mid = torch.empty((mid_size,), dtype=torch.bool, device=inp.device)
    with torch_device_fn.device(inp.device):
        all_kernel_1[(mid_size, 1)](inp, mid, n_elements, mid_size, block_size)
    return mid

def run_test1():
    inp = torch.randn((4096, ), dtype=torch.float32, device='mlu')
    return test1(inp)
```

**返回：**
```python
{
    "inp":
    {
        "axis": ["n_elements"],
        "axis_type": ["PARALLEL"],
        "stride": [1],
        "block_size": ["BLOCK_SIZE"],
        "has_loop": [False],
    }
}
```

**分析过程：**

分别分析kernel的访存指令：

1.inp_val = tl.load(inp_ptrs, mask=mask, other=1.0)
展开 inp_ptrs 为 inp + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)，inp 是 base_ptr，tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 组成第0维的offsets，这里隐式的 stride=1。这里的offset是规整的线性访问，tl.program_id(0) * BLOCK_SIZE 为offset的起始地址，tl.arange(0, BLOCK_SIZE)为访问的切分块。起始地址与pid相关，说明这个轴的axis_type是"PARALLEL"，另外起始地址跟访问块都指向第0维的的block size是BLOCK_SIZE，没有任何for循环参数使用到BLOCK_SIZE，所以这里has_loop是False。

2.tl.store(mid_ptr, all_val)

展开 mid_ptr 为 mid + pid，没有任何 tl.arange 参与，说明访问的不是一块数据，而是标量，所以不分析。

### 示例2：简单一维拆分
```python
@triton.jit
def test_kernel_2(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    block_start = block_start.to(tl.int64)
    for block_start_offset in range(block_start, n_elements, step):
        offsets = block_start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

def test2(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_mlu and y.is_mlu and output.is_mlu
    n_elements = output.numel()
    core_num = torch.mlu.get_device_properties().multi_processor_count
    grid = lambda meta: (min(triton.cdiv(n_elements, meta['BLOCK_SIZE']), core_num), )
    add_kernel[grid](x, y, output, n_elements)
    return output

def run_test2():
    x = torch.randn((4096, ), dtype=torch.float32, device='mlu')
    y = torch.randn((4096, ), dtype=torch.float32, device='mlu')
    return test2(x, y)
    ```

**返回：**
```python
{
    "x_ptr":
    {
        "axis": ["n_elements"],
        "axis_type": ["PARALLEL"],
        "stride": [1],
        "block_size": ["BLOCK_SIZE"],
        "has_loop": [True],
    },
    "y_ptr":
    {
        "axis": ["n_elements"],
        "axis_type": ["PARALLEL"],
        "stride": [1],
        "block_size": ["BLOCK_SIZE"],
        "has_loop": [True],
    },
    "output_ptr":
    {
        "axis": ["n_elements"],
        "axis_type": ["PARALLEL"],
        "stride": [1],
        "block_size": ["BLOCK_SIZE"],
        "has_loop": [True],
    }
}
```

**分析过程：**

分析上面kernel的访存指令：

1.x = tl.load(x_ptr + offsets, mask=mask)

展开 load输入为 x_ptr + block_start_offset + tl.arange(0, BLOCK_SIZE)，x_ptr 是 base_ptr，block_start_offset + tl.arange(0, BLOCK_SIZE) 组成第0维的offsets，这里隐式的 stride=1。block_start_offset是for循环上的迭代量，for循环的参数为range(block_start, n_elements, step)，展开 block_start=tl.program_id(axis=0) * BLOCK_SIZE，step=tl.num_programs(axis=0)*BLOCK_SIZE，说明该轴为并行轴，与pid相关，block_size为BLOCK_SIZE，且has_loop是True。

2.y = tl.load(y_ptr + offsets, mask=mask)

分析过程同x。

3.tl.store(output_ptr + offsets, output, mask=mask)

分析过程同x。

### 示例3：二维简单拆分

```python
@triton.jit
def test_kernel_3(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M
    _all = tl.full([BLOCK_M, BLOCK_N], value=1, dtype=tl.int1)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=1.0)
        _all = _all and (a != 0)
    all = tl.reduce(_all, axis=1, combine_fn=reduce_all)
    tl.store(out, all[:, None], row_mask)

def test3(inp, dim=None):
    shape = list(inp.shape)
    dim = dim % inp.ndim
    inp = dim_compress(inp, dim)
    N = shape[dim]
    shape[dim] = 1
    M = inp.numel() // N
    out = torch.empty(shape, dtype=torch.bool, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        all_kernel_dim[grid](inp, out, M, N)
    return out

def run_test3():
    inp = torch.randn((4096, 2048), dtype=torch.float32, device='mlu')
    return test3(inp, 1)
```

**返回：**
```python
{
    "inp":
    {
        "axis": ["M", "N"],
        "axis_type": ["PARALLEL", "REDUCE"],
        "stride": [2048, 1],
        "block_size": ["BLOCK_M", "BLOCK_N"],
        "has_loop": [False, True],
    },
    "out":
    {
        "axis": ["M"],
        "axis_type": ["PARALLEL"],
        "stride": [1],
        "block_size": ["BLOCK_M"],
        "has_loop": [False],
    }
}
```

**分析过程：**

分析上面kernel的访存指令：
1.a = tl.load(inp + cols, mask, other=1.0)

展开 load 输入为 inp + (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)) * N + off + tl.arange(0, BLOCK_N)，这是一个规整的2维线性访问，(tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)) * N 为第0维的offset和stride，与pid0有关，并行轴且block_size是BLOCK_M，stride为N的大小，has_loop为False；off + tl.arange(0, BLOCK_N) 为第1维的offset和stride，off为for循环迭代变量，for循环参数为range(0, N, BLOCK_N)，这是典型的reduction轴拆分，block_size是BLOCK_N，stride为1，has_loop为True。

2.tl.store(out, all[:, None], row_mask)

展开 store 输入为 out + tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M) 为典型的1维线性访问，与pid0有关，并行轴且block_size是BLOCK_M，stride为1，has_loop为False。


### 示例4：多维flatten之后拆分

```python
@triton.jit
def test_kernel4(
    input_ptr, output_ptr,
    stride_in0, stride_in1, stride_in2,
    stride_out0, stride_out1, stride_out2,
    N, H, W,
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    num_programs: tl.constexpr,
    num_blocks_n: tl.constexpr,
    num_blocks_w: tl.constexpr,
    num_blocks_h: tl.constexpr,
):
    pid = tl.program_id(0)
    total_blocks = num_blocks_n * num_blocks_w * num_blocks_h
    for flat_idx in range(pid, total_blocks, num_programs):
        w_idx = flat_idx % num_blocks_w
        h_idx = (flat_idx // num_blocks_w) % num_blocks_h
        n_idx = flat_idx // (num_blocks_w * num_blocks_h)
        n_offset = n_idx * BLOCK_N
        h_offset = h_idx * BLOCK_H
        w_offset = w_idx * BLOCK_W
        n_i = tl.arange(0, BLOCK_N)[:, None, None]
        h_i = tl.arange(0, BLOCK_W)[None, :, None]
        w_i = tl.arange(0, BLOCK_H)[None, None, :]
        mask = ...
        input_index = (
            n_offset * stride_in0 + h_offset * stride_in1 + w_offset * stride_in2 +
            n_i * stride_in0 + h_i * stride_in1 + w_i * stride_in2
        )
        x = tl.load(input_ptr + input_index,ask=mask)
        out_val = tl.where(x > 0, x, 0.0)
        output_index = (
            n_offset * stride_out0 + h_offset * stride_out1 + w_offset * stride_out2 +
            n_i * stride_out0 + h_i * stride_out1 + w_i * stride_out2
        )
        tl.store(output_ptr + output_index, out_val, mask=mask)

def test4(input: torch.Tensor) -> torch.Tensor:
    N, H, W = input.shape
    output = torch.empty((N, H, W), dtype=input.dtype, device=input.device)
    core_num = torch.mlu.get_device_properties(0).multi_processor_count
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    num_blocks_h = triton.cdiv(H, BLOCK_H)
    num_blocks_w = triton.cdiv(W, BLOCK_W)
    total_blocks = num_blocks_n * num_blocks_w * num_blocks_h
    grid = (min(total_blocks, core_num),)
    trans_relu_3d_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        stride_in0=input.stride(0),
        stride_in1=input.stride(1),
        stride_in2=input.stride(2),
        stride_out0=output.stride(0),
        stride_out1=output.stride(1),
        stride_out2=output.stride(2),
        N=N, H=H, W=W,
        BLOCK_N=BLOCK_N, BLOCK_W=BLOCK_W, BLOCK_H=BLOCK_H,
        num_programs=grid[0],
        num_blocks_n=num_blocks_n,
        num_blocks_w=num_blocks_w,
        num_blocks_h=num_blocks_h,
    )
    return output

def run_test4():
    inp = torch.randn((32, 64，256), dtype=torch.float32, device='mlu')
    return test4(inp)

```

**返回：**
```python
{
    "input_ptr":
    {
        "axis": ["N", "H", "W"],
        "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
        "stride": [16384, 256, 1],
        "block_size": ["BLOCK_N", "BLOCK_H", "BLOCK_W"],
        "has_loop": [True, True, True],
    },
    "output_ptr":
    {
        "axis": ["N", "H", "W"],
        "axis_type": ["PARALLEL", "PARALLEL", "PARALLEL"],
        "stride": [16384, 256, 1],
        "block_size": ["BLOCK_N", "BLOCK_H", "BLOCK_W"],
        "has_loop": [True, True, True],
    }
}
```
**分析过程：**

分析上面kernel的访存指令：

1.x = tl.load(input_ptr + input_index, mask=mask)

展开 load输入为 input_ptr + (n_offset * stride_in0 + h_offset * stride_in1 + w_offset * stride_in2 + n_i * stride_in0 + h_i * stride_in1 + w_i * stride_in2)，可以明显看到有3组，(n_offset + n_i) * stride_in0 为 n 维，(h_offset + h_i) * stride_in1 为 h 维，(w_offset + w_i) * stride_in2 为 w 维。
展开n维offset为 flat_idx // (num_blocks_w * num_blocks_h) * BLOCK_N + tl.arange(0, BLOCK_N)，这是典型的线性访问，与单线程跨单block不同的是，这里单线程跨了多个轴，这里 h_idx = flat_idx % num_blocks_h，w_idx = (flat_idx // num_blocks_h) % num_blocks_w，n_idx = flat_idx // (num_blocks_w * num_blocks_h)，为典型的多轴flatten表现，flat_idx为flatten之后的总索引，这里是for的迭代变量，for上的参数为range(pid, total_blocks, num_programs)，与tl.program_id(0)有关，所以这里n，h，w轴均与pid0有关，为并行轴，block_size分别是BLOCK_N，BLOCK_H，BLOCK_W，共同组成pid0上的拆分，并参与循环，所有has_loop均为True。stride大小可以通过wrapper函数确定，例如输入n维的stride为stride_in0，从wrapper函数中可以得到为input.stride(0)，也就是H*W，即64*256=16384。

2.tl.store(out, all[:, None], row_mask)

同上。
