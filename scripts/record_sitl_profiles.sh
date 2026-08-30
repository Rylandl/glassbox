#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
data_dir=${1:-"$project_dir/artifacts/sitl/multirotor_v2"}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  profiles=("$@")
else
  profiles=(vertical_steps lateral_steps yaw_steps combined)
fi

px4_image=${GLASSBOX_PX4_IMAGE:-px4io/px4-sitl@sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44}
replicates=${GLASSBOX_PROFILE_REPLICATES:-2}
sample_rate_hz=${GLASSBOX_PROFILE_SAMPLE_RATE_HZ:-50}
condition_list=${GLASSBOX_PROFILE_CONDITIONS:-low,medium,high}
initial_yaw_list=${GLASSBOX_PROFILE_INITIAL_YAWS:-0,45}
IFS=',' read -r -a conditions <<<"$condition_list"
IFS=',' read -r -a initial_yaws <<<"$initial_yaw_list"
container_name=""
docker_log_pid=""

if [[ ${#conditions[@]} -lt 1 || ${#initial_yaws[@]} -lt 1 ]]; then
  echo "conditions and initial yaw lists cannot be empty" >&2
  exit 2
fi

stop_container() {
  if [[ -n "$container_name" ]]; then
    docker stop --timeout 5 "$container_name" >/dev/null 2>&1 || true
  fi
  if [[ -n "$docker_log_pid" ]]; then
    wait "$docker_log_pid" 2>/dev/null || true
    docker_log_pid=""
  fi
}
trap stop_container EXIT

for profile in "${profiles[@]}"; do
  for condition in "${conditions[@]}"; do
    for ((replicate = 1; replicate <= replicates; replicate++)); do
    yaw_index=$(((replicate - 1) % ${#initial_yaws[@]}))
    initial_yaw=${initial_yaws[$yaw_index]}
    run_dir="$data_dir/$profile/$condition/run_$replicate"
    if [[ -e "$run_dir" ]]; then
      echo "refusing to overwrite existing run directory: $run_dir" >&2
      exit 1
    fi
    mkdir -p "$run_dir/px4/rootfs/fs/microsd/etc/logging"
    cp \
      "$project_dir/config/logging/logger_topics.txt" \
      "$run_dir/px4/rootfs/fs/microsd/etc/logging/logger_topics.txt"
    absolute_run_dir=$(cd "$run_dir" && pwd)
    container_name="glassbox-${profile//_/-}-${condition}-${replicate}-$$"

    echo "recording $profile condition $condition replicate $replicate yaw $initial_yaw"
    docker run --pull=never --rm -dit \
      --name "$container_name" \
      -v "$absolute_run_dir:/data" \
      -v "$project_dir/config/logging:/opt/px4/etc/logging:ro" \
      -e XDG_DATA_HOME=/data \
      -e PX4_SIM_MODEL=sihsim_quadx \
      "$px4_image" >/dev/null
    docker logs -f "$container_name" >"$absolute_run_dir/px4_console.log" 2>&1 &
    docker_log_pid=$!

    started=false
    for _ in {1..60}; do
      if docker logs "$container_name" 2>&1 | \
        grep -q 'Startup script returned successfully'; then
        started=true
        break
      fi
      sleep 0.5
    done
    if [[ "$started" != true ]]; then
      docker logs "$container_name" >&2
      echo "PX4 did not finish startup" >&2
      exit 1
    fi

    ready=false
    for _ in {1..30}; do
      if docker logs "$container_name" 2>&1 | grep -q 'Ready for takeoff'; then
        ready=true
        break
      fi
      sleep 0.5
    done
    if [[ "$ready" != true ]]; then
      docker logs "$container_name" >&2
      echo "PX4 did not become ready for takeoff" >&2
      exit 1
    fi

    # Use PX4's ordinary takeoff mode so arming and health checks follow the
    # same path as the baseline recorder. The profile driver takes over only
    # after the vehicle is airborne and its setpoint stream is established.
    docker exec -w /root "$container_name" \
      /opt/px4/bin/px4-commander takeoff
    sleep 6
    uv --directory "$project_dir" run glassbox-sitl-profile "$profile" \
      --condition "$condition" \
      --initial-yaw "$initial_yaw"
    docker exec -w /root "$container_name" \
      /opt/px4/bin/px4-shutdown >/dev/null 2>&1 &
    sleep 1
    stop_container
    container_name=""
    if [[ -n "$docker_log_pid" ]]; then
      wait "$docker_log_pid" 2>/dev/null || true
      docker_log_pid=""
    fi

    latest_log=$(find "$absolute_run_dir" -type f -name '*.ulg' | sort | tail -n 1)
    if [[ -z "$latest_log" ]]; then
      echo "PX4 exited without producing a ULog" >&2
      exit 1
    fi
    echo "profile=$profile condition=$condition replicate=$replicate raw_log=$latest_log"

    output_stem="$absolute_run_dir/${profile}_${condition}_${replicate}"
    uv --directory "$project_dir" run glassbox-ulog extract \
      "$latest_log" "${output_stem}_estimated.npz" \
      --rate "$sample_rate_hz" \
      --profile "$profile" \
      --condition "$condition" \
      --replicate "$replicate" \
      --initial-yaw "$initial_yaw" \
      --actuator-topic actuator_motors \
      --actuator-field control
    uv --directory "$project_dir" run glassbox-ulog extract \
      "$latest_log" "${output_stem}_ground_truth.npz" \
      --rate "$sample_rate_hz" \
      --profile "$profile" \
      --condition "$condition" \
      --replicate "$replicate" \
      --initial-yaw "$initial_yaw" \
      --state-source ground_truth \
      --actuator-topic actuator_motors \
      --actuator-field control
    done
  done
done

echo "profile dataset: $data_dir"
