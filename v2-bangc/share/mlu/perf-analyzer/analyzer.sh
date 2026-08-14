#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash analyzer.sh <output_dir> <executable> [artifact_dir] [program args ...]" >&2
  exit 2
fi

output_dir="$1"
input_binary="$2"
shift 2

artifact_dir="$output_dir"
if [[ $# -gt 0 && -d "$1" ]]; then
  artifact_dir="$1"
  shift
fi
program_args=("$@")

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$output_dir" || exit 2
output_dir="$(cd "$output_dir" && pwd)"

if [[ ! -f "$input_binary" ]]; then
  echo "ERROR: executable not found: $input_binary" >&2
  exit 2
fi
if [[ ! -x "$input_binary" ]]; then
  echo "ERROR: input is not executable: $input_binary" >&2
  exit 2
fi
input_binary="$(cd "$(dirname "$input_binary")" && pwd)/$(basename "$input_binary")"

find_tool() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return 0
  fi
  local candidate
  for candidate in \
    "${NEUWARE_HOME:-}/bin/$name" \
    "${CNTOOLKIT_HOME:-}/bin/$name" \
    "/usr/local/neuware/bin/$name" \
    "/usr/local/cntoolkit/bin/$name"; do
    if [[ "$candidate" != "/bin/$name" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

status_file="$output_dir/analysis_status.txt"
run_stdout="$output_dir/run_stdout.log"
run_stderr="$output_dir/run_stderr.log"
cnmon_output="$output_dir/cnmon.txt"
cnperf_rep="$output_dir/kernel.cnperf-rep"
cnperf_kernel="$output_dir/kernel_cnperf.txt"
artifact_json="$output_dir/cncc_artifacts.json"

: >"$status_file"
printf 'INPUT_BINARY=%s\n' "$input_binary" | tee -a "$status_file"
printf 'ARTIFACT_DIR=%s\n' "$artifact_dir" | tee -a "$status_file"

if cnmon_path="$(find_tool cnmon)"; then
  "$cnmon_path" >"$cnmon_output" 2>&1 || true
  printf 'CNMON=AVAILABLE:%s\n' "$cnmon_path" | tee -a "$status_file"
else
  printf 'CNMON=UNAVAILABLE\n' | tee -a "$status_file"
fi

"$input_binary" "${program_args[@]}" >"$run_stdout" 2>"$run_stderr"
run_status=$?
printf 'BASELINE_RUN_EXIT_CODE=%s\n' "$run_status" | tee -a "$status_file"
if [[ $run_status -ne 0 ]]; then
  echo "ERROR: baseline executable failed; see $run_stdout and $run_stderr" >&2
  printf 'STATUS=FAILED_BASELINE_RUN\n' | tee -a "$status_file"
  exit 1
fi

cnperf_status="PARTIAL_NO_CNPERF"
if cnperf_path="$(find_tool cnperf-cli)"; then
  printf 'CNPERF=AVAILABLE:%s\n' "$cnperf_path" | tee -a "$status_file"
  "$cnperf_path" record --pmu --replay_mode=kernel -o "$cnperf_rep" \
    "$input_binary" "${program_args[@]}" \
    >"$output_dir/cnperf_record_stdout.log" \
    2>"$output_dir/cnperf_record_stderr.log"
  record_status=$?
  printf 'CNPERF_RECORD_EXIT_CODE=%s\n' "$record_status" | tee -a "$status_file"
  if [[ $record_status -eq 0 ]]; then
    "$cnperf_path" kernel "$cnperf_rep" >"$cnperf_kernel" \
      2>"$output_dir/cnperf_kernel_stderr.log"
    kernel_status=$?
    printf 'CNPERF_KERNEL_EXIT_CODE=%s\n' "$kernel_status" | tee -a "$status_file"
    if [[ $kernel_status -eq 0 ]]; then
      cnperf_status="COMPLETE_WITH_CNPERF"
    else
      cnperf_status="PARTIAL_CNPERF_KERNEL_FAILED"
    fi
  else
    cnperf_status="PARTIAL_CNPERF_RECORD_FAILED"
  fi
else
  printf 'CNPERF=UNAVAILABLE\n' | tee -a "$status_file"
fi

python_path=""
if command -v python3 >/dev/null 2>&1; then
  python_path="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  python_path="$(command -v python)"
fi

if [[ -n "$python_path" ]]; then
  "$python_path" "$script_dir/analyzer_cncc_artifacts.py" \
    "$artifact_dir" --json-out "$artifact_json" \
    >"$output_dir/cncc_artifacts_stdout.log" \
    2>"$output_dir/cncc_artifacts_stderr.log"
  artifact_status=$?
  printf 'CNCC_ARTIFACT_SCAN_EXIT_CODE=%s\n' "$artifact_status" | tee -a "$status_file"
else
  printf 'CNCC_ARTIFACT_SCAN=UNAVAILABLE_NO_PYTHON\n' | tee -a "$status_file"
fi

printf 'STATUS=%s\n' "$cnperf_status" | tee -a "$status_file"
printf 'RAW_CNPERF_REPORT=%s\n' "$cnperf_kernel" | tee -a "$status_file"
printf 'CNCC_ARTIFACTS=%s\n' "$artifact_json" | tee -a "$status_file"

echo "=== DONE: $cnperf_status ==="
exit 0
