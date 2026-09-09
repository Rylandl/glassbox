#!/usr/bin/env bash
# Record scripted PX4 SITL maneuver profiles for one vehicle family.
#
# usage: record_sitl_profiles.sh [--family multirotor|fixedwing] [DATA_DIR] [PROFILE...]
#
# Each run starts a disposable PX4 SIH container, takes off, flies one bounded
# profile through `glassbox sitl-profile`, and extracts the resulting ULog into
# canonical trajectories with `glassbox extract`. The `baseline` profile flies
# PX4's own takeoff and land instead of an offboard profile and additionally
# fits a model from the ground-truth extraction; it is the smallest end-to-end
# check that the whole pipeline still runs.
set -euo pipefail

family=multirotor
if [[ ${1:-} == --family ]]; then
  family=${2:?--family needs a value}
  shift 2
fi
case "$family" in
  multirotor|fixedwing) ;;
  *) echo "unknown family: $family" >&2; exit 2 ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd "$script_dir/.." && pwd)
if [[ "$family" == fixedwing ]]; then
  default_data_dir="$project_dir/artifacts/sitl/fixedwing_v1"
  default_profiles=(throttle_steps roll_steps pitch_steps combined)
  sim_model=sihsim_airplane
else
  default_data_dir="$project_dir/artifacts/sitl/multirotor_v2"
  default_profiles=(vertical_steps lateral_steps yaw_steps combined)
  sim_model=sihsim_quadx
fi

data_dir=${1:-$default_data_dir}
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  profiles=("$@")
else
  profiles=("${default_profiles[@]}")
fi

px4_image=${GLASSBOX_PX4_IMAGE:-px4io/px4-sitl@sha256:01866d912ac22ca6119a996b830cf628a6d47dfb60fdccc41cd9f44b62935a44}
replicates=${GLASSBOX_PROFILE_REPLICATES:-2}
replicate_start=${GLASSBOX_PROFILE_REPLICATE_START:-1}
sample_rate_hz=${GLASSBOX_PROFILE_SAMPLE_RATE_HZ:-50}
condition_list=${GLASSBOX_PROFILE_CONDITIONS:-low,medium,high}
initial_yaw_list=${GLASSBOX_PROFILE_INITIAL_YAWS:-0,45}
takeoff_wait_s=${GLASSBOX_FIXEDWING_TAKEOFF_WAIT_S:-40}
baseline_flight_wait_s=${GLASSBOX_FLIGHT_WAIT_S:-15}
baseline_landing_wait_s=${GLASSBOX_LANDING_WAIT_S:-12}
baseline_sample_rate_hz=${GLASSBOX_BASELINE_SAMPLE_RATE_HZ:-250}
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

wait_for_console() {
  local needle=$1 attempts=$2 message=$3
  for _ in $(seq 1 "$attempts"); do
    if docker logs "$container_name" 2>&1 | grep -q "$needle"; then
      return 0
    fi
    sleep 0.5
  done
  docker logs "$container_name" >&2
  echo "$message" >&2
  exit 1
}

for profile in "${profiles[@]}"; do
  for condition in "${conditions[@]}"; do
    for ((replicate = replicate_start; replicate <= replicates; replicate++)); do
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
      container_name="glassbox-${family:0:2}-${profile//_/-}-${condition}-${replicate}-$$"

      echo "recording $family $profile condition $condition replicate $replicate"
      docker run --pull=never --rm -dit \
        --name "$container_name" \
        -v "$absolute_run_dir:/data" \
        -v "$project_dir/config/logging:/opt/px4/etc/logging:ro" \
        -e XDG_DATA_HOME=/data \
        -e PX4_SIM_MODEL="$sim_model" \
        "$px4_image" >/dev/null
      docker logs -f "$container_name" >"$absolute_run_dir/px4_console.log" 2>&1 &
      docker_log_pid=$!

      wait_for_console 'Startup script returned successfully' 60 \
        "PX4 $family SITL did not finish startup"
      wait_for_console 'Ready for takeoff' 30 \
        "PX4 $family SITL did not become ready for takeoff"

      if [[ "$family" == fixedwing ]]; then
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
          # shellcheck disable=SC2086
          docker exec -w /root "$container_name" \
            /opt/px4/bin/px4-param set $setting >/dev/null
        done
      fi

      # Use PX4's ordinary takeoff mode so arming and health checks follow the
      # same path for every profile. The profile driver takes over only after
      # the vehicle is airborne and its setpoint stream is established.
      docker exec -w /root "$container_name" \
        /opt/px4/bin/px4-commander takeoff

      if [[ "$profile" == baseline ]]; then
        sleep "$baseline_flight_wait_s"
        docker exec -w /root "$container_name" \
          /opt/px4/bin/px4-commander land
        sleep "$baseline_landing_wait_s"
      elif [[ "$family" == fixedwing ]]; then
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
        uv --directory "$project_dir" run glassbox sitl-profile "$profile" \
          --family fixedwing \
          --condition "$condition"
      else
        sleep 6
        uv --directory "$project_dir" run glassbox sitl-profile "$profile" \
          --family multirotor \
          --condition "$condition" \
          --initial-yaw "$initial_yaw"
      fi

      docker exec -w /root "$container_name" \
        /opt/px4/bin/px4-shutdown >/dev/null 2>&1 &
      sleep 1
      stop_container
      container_name=""

      latest_log=$(find "$absolute_run_dir" -type f -name '*.ulg' | sort | tail -n 1)
      if [[ -z "$latest_log" ]]; then
        echo "PX4 exited without producing a ULog" >&2
        exit 1
      fi
      echo "profile=$profile condition=$condition replicate=$replicate raw_log=$latest_log"

      output_stem="$absolute_run_dir/${profile}_${condition}_${replicate}"
      if [[ "$profile" == baseline ]]; then
        for state_source in estimated ground_truth; do
          uv --directory "$project_dir" run glassbox extract \
            "$latest_log" "${output_stem}_${state_source}.npz" \
            --family multirotor \
            --rate "$baseline_sample_rate_hz" \
            --state-source "$state_source" \
            --actuator-topic actuator_outputs_sim \
            --actuator-field output
        done
        uv --directory "$project_dir" run glassbox fit \
          "${output_stem}_ground_truth.npz" \
          --training-horizons 2.0 \
          --steps 2000 \
          --learning-rate 0.01 \
          --model "${output_stem}_model.json" \
          --report "${output_stem}_fit.json"
      elif [[ "$family" == fixedwing ]]; then
        for state_source in estimated ground_truth; do
          uv --directory "$project_dir" run glassbox extract \
            "$latest_log" "${output_stem}_${state_source}.npz" \
            --family fixedwing \
            --rate "$sample_rate_hz" \
            --state-source "$state_source" \
            --profile "$profile" \
            --condition "$condition" \
            --replicate "$replicate" \
            --min-height 0.5
        done
      else
        for state_source in estimated ground_truth; do
          uv --directory "$project_dir" run glassbox extract \
            "$latest_log" "${output_stem}_${state_source}.npz" \
            --family multirotor \
            --rate "$sample_rate_hz" \
            --state-source "$state_source" \
            --profile "$profile" \
            --condition "$condition" \
            --replicate "$replicate" \
            --initial-yaw "$initial_yaw" \
            --actuator-topic actuator_motors \
            --actuator-field control
        done
      fi
    done
  done
done

echo "$family profile dataset: $data_dir"
