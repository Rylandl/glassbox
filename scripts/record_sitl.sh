#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
data_dir=${1:-"$project_dir/artifacts/sitl"}
px4_image=${GLASSBOX_PX4_IMAGE:-px4io/px4-sitl:latest}
startup_wait_s=${GLASSBOX_STARTUP_WAIT_S:-8}
flight_wait_s=${GLASSBOX_FLIGHT_WAIT_S:-15}
landing_wait_s=${GLASSBOX_LANDING_WAIT_S:-12}

mkdir -p "$data_dir/px4/rootfs/fs/microsd/etc/logging"
cp \
  "$project_dir/config/logging/logger_topics.txt" \
  "$data_dir/px4/rootfs/fs/microsd/etc/logging/logger_topics.txt"

absolute_data_dir=$(cd "$data_dir" && pwd)
console_log="$absolute_data_dir/px4_console.log"

(
  sleep "$startup_wait_s"
  echo "commander takeoff"
  sleep "$flight_wait_s"
  echo "commander land"
  sleep "$landing_wait_s"
  echo "shutdown"
) | docker run \
  --pull=never \
  --rm \
  -i \
  -v "$absolute_data_dir:/data" \
  -v "$project_dir/config/logging:/opt/px4/etc/logging:ro" \
  -e XDG_DATA_HOME=/data \
  -e PX4_SIM_MODEL=sihsim_quadx \
  "$px4_image" | tee "$console_log"

latest_log=$(find "$absolute_data_dir" -type f -name '*.ulg' | sort | tail -n 1)
if [[ -z "$latest_log" ]]; then
  echo "PX4 exited without producing a ULog" >&2
  exit 1
fi

log_stem=${latest_log%.ulg}
uv --directory "$project_dir" run glassbox-ulog extract \
  "$latest_log" "${log_stem}_estimated.npz" --rate 250 \
  --actuator-topic actuator_outputs_sim --actuator-field output
uv --directory "$project_dir" run glassbox-ulog extract \
  "$latest_log" "${log_stem}_ground_truth.npz" --rate 250 \
  --state-source ground_truth \
  --actuator-topic actuator_outputs_sim --actuator-field output
uv --directory "$project_dir" run glassbox-fit \
  "${log_stem}_ground_truth.npz" \
  --training-horizons 2.0 \
  --steps 2000 \
  --learning-rate 0.01 \
  --model "${log_stem}_model.json" \
  --report "${log_stem}_fit.json"

echo "ULog: $latest_log"
echo "Estimated trajectory: ${log_stem}_estimated.npz"
echo "Ground-truth trajectory: ${log_stem}_ground_truth.npz"
echo "Fitted model: ${log_stem}_model.json"
echo "No-lag model: ${log_stem}_model_no_motor_lag.json"
echo "Fit report: ${log_stem}_fit.json"
