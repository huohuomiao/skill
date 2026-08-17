# Triton Kernel 性能分析报告

## 1. Triton kernel 信息

### 1.1 输入规模
- **Shape**: (M, N)
- **Dtype**: torch.float32

### 1.2 config 信息

- **BLOCK_M**: XX
- **BLOCK_N**: XX
- **BLOCK_K / GROUP_M / SPLIT_K**: XX / XX / XX
- **num_stages**：XX
- **num_warps**：XX
- **Launch family**: ordinary / persistent / split-K

## 2. CUDA / NCU 信息

- **Registers / thread**: XX
- **Local spill requests / percent**: XX / XX%
- **Shared memory / block**: XX B
- **Theoretical / achieved occupancy**: XX% / XX%
- **Occupancy limiter / active blocks per SM**: registers|shared_memory|blocks|warps / XX
- **Tensor Core activity**: XX% / 未采集
- **Matmul throughput / dense peak ratio**: XX TFLOPS / XX%（注明峰值来源）
- **Memory / compute bottleneck**: XX

## 3. 优化策略建议

- **strategy**: config-tuner / gen-autotune-config / libdevice-opt / div-to-mul / none
- **architecture_reselect_required**: true / false
- **force_non_persistent / consider_split_k**: true / false, true / false
- **evidence and next experiment**: XXXXXX
