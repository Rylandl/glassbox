#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 LOG_DIRECTORY OUTPUT_DIRECTORY [SAMPLE_RATE_HZ]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
log_dir=$1
output_dir=$2
sample_rate_hz=${3:-50}

mkdir -p "$output_dir"

log_count=0
trajectory_count=0
failure_count=0

while IFS= read -r -d '' log_path; do
  log_count=$((log_count + 1))
  log_name=$(basename "$log_path" .ulg)

  for state_source in ground_truth estimated; do
    output_path="$output_dir/${log_name}_${state_source}.npz"
    if uv --directory "$project_dir" run glassbox-ulog extract \
      "$log_path" "$output_path" \
      --rate "$sample_rate_hz" \
      --state-source "$state_source" \
      --actuator-topic actuator_motors \
      --actuator-field control; then
      trajectory_count=$((trajectory_count + 1))
    else
      failure_count=$((failure_count + 1))
      echo "skipped unusable ${state_source} trajectory: $log_path" >&2
    fi
  done
done < <(find "$log_dir" -type f -name '*.ulg' -print0 | sort -z)

if [[ $log_count -eq 0 ]]; then
  echo "no ULogs found under $log_dir" >&2
  exit 1
fi

echo "processed $log_count logs: $trajectory_count trajectories, $failure_count failures"
if [[ $trajectory_count -eq 0 ]]; then
  exit 1
fi
