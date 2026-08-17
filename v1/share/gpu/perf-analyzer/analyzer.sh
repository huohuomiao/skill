#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: NCU_KERNEL_NAME='regex:.*kernel_name.*' bash analyzer.sh <output_dir> <input.py> [program arguments...]" >&2
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

output_dir=$1
input_path=$2
shift 2

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: NVIDIA Nsight Compute CLI (ncu) was not found on PATH." >&2
    exit 3
fi

python_executable=${PYTHON_EXECUTABLE:-python3}
if ! command -v "$python_executable" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        python_executable=python
    else
        echo "ERROR: neither '$python_executable' nor 'python' was found on PATH." >&2
        exit 3
    fi
fi

if [[ ! -f "$input_path" ]]; then
    echo "ERROR: input Python file does not exist: $input_path" >&2
    exit 2
fi

if [[ -z "${NCU_KERNEL_NAME:-}" ]]; then
    echo "ERROR: NCU_KERNEL_NAME is required so NCU does not profile a reference/framework kernel by mistake." >&2
    usage
    exit 2
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd -P)
input_dir=$(cd "$(dirname "$input_path")" && pwd -P)
input_path="$input_dir/$(basename "$input_path")"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

report_base="$output_dir/kernel"
report_path="$report_base.ncu-rep"
csv_path="$output_dir/kernel.ncu.csv"
summary_path="$output_dir/kernel.ncu.summary.txt"
json_path="$output_dir/kernel.ncu.summary.json"

echo "Profiling: $input_path"
echo "Output:    $output_dir"

# Standard Nsight Compute section identifiers. Kernel replay may run the selected
# kernel more than once; the profiled program must be deterministic and replay-safe.
ncu \
    --force-overwrite \
    --target-processes all \
    --replay-mode kernel \
    --kernel-name-base function \
    --kernel-name "$NCU_KERNEL_NAME" \
    --launch-count 1 \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    -o "$report_base" \
    "$python_executable" "$input_path" "$@"

if [[ ! -f "$report_path" ]]; then
    echo "ERROR: ncu completed but report was not created: $report_path" >&2
    exit 4
fi

# Raw page preserves stable metric identifiers used by analyzer_ncu.py.
ncu --import "$report_path" --page raw --csv >"$csv_path"

"$python_executable" "$script_dir/analyzer_ncu.py" \
    "$csv_path" \
    --json-out "$json_path" | tee "$summary_path"

echo "NCU report:   $report_path"
echo "NCU raw CSV:  $csv_path"
echo "Text summary: $summary_path"
echo "JSON summary: $json_path"
