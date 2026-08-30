#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
data_dir=${1:-"$project_dir/artifacts/sitl/fixedwing_v1"}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  profiles=("$@")
else
  profiles=(throttle_steps roll_steps pitch_steps combined)
fi

px4_image=${GLASSBOX_PX4_IMAGE:-px4io/px4-sitl@sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44}
replicates=${GLASSBOX_PROFILE_REPLICATES:-2}
replicate_start=${GLASSBOX_PROFILE_REPLICATE_START:-1}
sample_rate_hz=${GLASSBOX_PROFILE_SAMPLE_RATE_HZ:-50}
condition_list=${GLASSBOX_PROFILE_CONDITIONS:-low,medium,high}
takeoff_wait_s=${GLASSBOX_FIXEDWING_TAKEOFF_WAIT_S:-40}
IFS=',' read -r -a conditions <<<"$condition_list"
container_name=""
docker_log_pid=""

stop_container() {
  if [[ -n "$container_name" ]]; then
    docker exec -w /root "$container_name" \
      /opt/px4/bin/px4-logger stop >/dev/null 2>&1 || true
    sleep 0.5
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
    for ((replicate = replicate_start; replicate <= replicates; replicate++)); do
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
      container_name="glassbox-fw-${profile//_/-}-${condition}-${replicate}-$$"

      echo "recording fixedwing $profile condition $condition replicate $replicate"
      docker run --pull=never --rm -dit \
        --name "$container_name" \
        -v "$absolute_run_dir:/data" \
        -v "$project_dir/config/logging:/opt/px4/etc/logging:ro" \
        -e XDG_DATA_HOME=/data \
        -e PX4_SIM_MODEL=sihsim_airplane \
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
        echo "PX4 fixed-wing SITL did not finish startup" >&2
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
        echo "PX4 fixed-wing SITL did not become ready for takeoff" >&2
        exit 1
      fi

      # The upstream SIH airplane peaks near 6 m/s while generic plane defaults
      # require 10--15 m/s. These explicit settings make runway rotation and
      # the controller's airspeed envelope consistent with the simulated plant.
      parameter_settings=(
        'FW_AIRSPD_STALL 3'
        'FW_AIRSPD_MIN 4'
        'FW_AIRSPD_TRIM 5'
        'FW_AIRSPD_MAX 8'
        'FW_TKO_AIRSPD 5'
        'RWTO_ROT_AIRSPD 4.5'
        'RWTO_PSP 12'
      )
      for setting in "${parameter_settings[@]}"; do
        docker exec -w /root "$container_name" \
          /opt/px4/bin/px4-param set $setting >/dev/null
      done

      docker exec -w /root "$container_name" \
        /opt/px4/bin/px4-commander takeoff
      sleep "$takeoff_wait_s"

      # Rotate away the shared runway/takeoff transient. The new ULog starts
      # immediately before the warmup and profile setpoints, so throttle and
      # surface coverage are not dominated by full-power takeoff samples.
      docker exec -w /root "$container_name" \
        /opt/px4/bin/px4-logger stop >/dev/null
      sleep 0.5
      docker exec -w /root "$container_name" \
        /opt/px4/bin/px4-logger start -m all >/dev/null
      sleep 0.5
      uv --directory "$project_dir" run glassbox-fixedwing-sitl-profile "$profile" \
        --condition "$condition"

      stop_container
      container_name=""

      latest_log=$(find "$absolute_run_dir" -type f -name '*.ulg' | sort | tail -n 1)
      if [[ -z "$latest_log" ]]; then
        echo "PX4 exited without producing a ULog" >&2
        exit 1
      fi
      echo "profile=$profile condition=$condition replicate=$replicate raw_log=$latest_log"

      output_stem="$absolute_run_dir/${profile}_${condition}_${replicate}"
      uv --directory "$project_dir" run glassbox-ulog extract-fixedwing \
        "$latest_log" "${output_stem}_estimated.npz" \
        --rate "$sample_rate_hz" \
        --profile "$profile" \
        --condition "$condition" \
        --replicate "$replicate" \
        --min-height 0.5
      uv --directory "$project_dir" run glassbox-ulog extract-fixedwing \
        "$latest_log" "${output_stem}_ground_truth.npz" \
        --rate "$sample_rate_hz" \
        --profile "$profile" \
        --condition "$condition" \
        --replicate "$replicate" \
        --min-height 0.5 \
        --state-source ground_truth
    done
  done
done

echo "fixed-wing profile dataset: $data_dir"
