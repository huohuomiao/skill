#!/bin/bash

# ================================
# 参数检查
# ================================
if [ $# -lt 2 ]; then
    echo "Usage: bash analyzer.sh <input_path> <output_dir>"
    exit 1
fi
output_dir="$1"
input_path="$2"


# ================================
# 进入工作目录
# ================================
cd "$output_dir" || { echo "Failed to cd to $output_dir"; exit 1; }

echo "Working directory: $(pwd)"

output_rep_path="$output_dir/kernel.cnperf-rep"
output_kernel_path="$output_dir/kernel.txt"

# ================================
# 清理缓存
# ================================
rm -rf ~/.triton/* 

export TRITON_KERNEL_DUMP=1

# ================================
# 性能采集
# ================================
cnperf-cli record --pmu --replay_mode=kernel -o "$output_rep_path" python "$input_path"

# ================================
# kernel 分析
# ================================
cnperf-cli kernel "$output_rep_path" > "$output_kernel_path"

# ================================
# mluiropt 分析
# ================================
output_mluiropt_path=$(find ~/.triton/dump/ -name "*.mluiropt")

python scripts/analyzer_mluiropt.py "$output_mluiropt_path"

# ================================
# 清理
# ================================
rm -rf ~/.triton/* 

echo "=== DONE ==="