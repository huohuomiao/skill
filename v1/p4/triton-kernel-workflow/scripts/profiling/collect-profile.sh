#!/bin/bash

# ================================
# 参数检查
# ================================
if [ $# -lt 2 ]; then
    echo "Usage: bash collect-profile.sh <output_dir> <input_path>"
    exit 1
fi
output_dir="$1"
input_path="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ================================
# 进入工作目录
# ================================
cd "$output_dir" || { echo "Failed to cd to $output_dir"; exit 1; }

echo "Working directory: $(pwd)"

output_rep_path="$output_dir/kernel.cnperf-rep"
output_kernel_path="$output_dir/kernel.txt"

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
# 编译器 IR 分析
# ================================
compiler_ir_path=$(find "${TRITON_DUMP_DIR:-$HOME/.triton/dump}" -name "*.mluiropt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)

if [ -z "$compiler_ir_path" ]; then
    echo "No compiler IR dump found" >&2
    exit 1
fi

python "$script_dir/analyze-compiler-ir.py" "$compiler_ir_path"

echo "=== DONE ==="
