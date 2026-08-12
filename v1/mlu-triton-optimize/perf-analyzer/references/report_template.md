# Triton Kernel 性能分析报告

## 1. Triton kernel 信息

### 1.1 输入规模
- **Shape**: (M, N)
- **Dtype**: torch.float32

### 1.2 config 信息

- **BLOCK_M**: XX
- **BLOCK_N**: XX
- **num_stages**：XX
- **num_warps**：XX

## 2. CUDA / NCU 信息

- **Registers / thread**: XX
- **Shared memory / block**: XX B
- **Theoretical / achieved occupancy**: XX% / XX%
- **Memory / compute bottleneck**: XX

## 3. 优化策略建议

XXXXXX
