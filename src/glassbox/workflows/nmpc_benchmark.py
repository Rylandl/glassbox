"""Maintained closed-loop acceptance and timing benchmark for Glassbox NMPC."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.nmpc import NMPCController, ReferenceTrajectory
from glassbox.core.data import TrajectorySpec, make_trajectory_spec
from glassbox.core.dynamics import (
    FIXED_WING_CONTROL_NAMES,
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    control_state_after_history,
    fixed_wing_trim_control,
    hover_control,
    step_with_latent,
)
from glassbox.core.fixedwing_synthetic import (
    TRIM_AIRSPEED_M_S,
    fixed_wing_trim_state,
    generate_fixed_wing_trajectory,
    true_fixed_wing_parameters,
)
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.runtime import (
    DirectActuationMap,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import (
    generate_trajectory,
    resting_state,
    true_parameters,
)

ACCEPTANCE_AGGREGATE_RATIO = 0.80
ACCEPTANCE_NOMINAL_SCENARIO_RATIO = 1.05
ACCEPTANCE_MISMATCH_SCENARIO_RATIO = 1.10
MODEL_DT_S = 0.05

ReferenceFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _Scenario:
    name: str
    model_key: str
    duration_s: float
    initial_state: np.ndarray
    reference: ReferenceFunction


@dataclass(frozen=True)
class ScenarioMetrics:
    name: str
    condition: str
    platform: str
    normalized_tracking_rms: float
    baseline_normalized_tracking_rms: float
    tracking_ratio: float
    maximum_command_bound_violation: float
    maximum_validity_utilization: float
    maximum_auxiliary_command_excursion: float
    fallback_count: int
    finite: bool
    warmup_solve_time_s: float
    solve_time_median_s: float
    solve_time_p90_s: float
    solve_time_maximum_s: float


def _quaternion_from_euler(
    roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
) -> np.ndarray:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = np.cos(half_roll), np.sin(half_roll)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    return np.column_stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _multirotor_reference(kind: str) -> ReferenceFunction:
    def reference(time_s: np.ndarray) -> np.ndarray:
        states = np.repeat(resting_state()[None, :], len(time_s), axis=0)
        if kind == "translation":
            time_constant_s = 0.55
            states[:, 0] = 0.55 * (1.0 - np.exp(-time_s / time_constant_s))
            states[:, 3] = 0.55 / time_constant_s * np.exp(-time_s / time_constant_s)
        return states

    return reference


def _fixed_wing_reference(kind: str) -> ReferenceFunction:
    def reference(time_s: np.ndarray) -> np.ndarray:
        states = np.repeat(fixed_wing_trim_state()[None, :], len(time_s), axis=0)
        states[:, 0] = TRIM_AIRSPEED_M_S * time_s
        roll = np.zeros(len(time_s))
        pitch = np.zeros(len(time_s))
        yaw = np.zeros(len(time_s))
        if kind in {"altitude", "flap"}:
            amplitude_m = 1.2 if kind == "altitude" else 0.8
            time_constant_s = 1.2
            exponential = np.exp(-time_s / time_constant_s)
            states[:, 2] = amplitude_m * (1.0 - exponential)
            states[:, 5] = amplitude_m / time_constant_s * exponential
            pitch = -np.arctan2(states[:, 5], states[:, 3])
        elif kind == "path":
            amplitude_m = 1.8
            frequency_rad_s = 0.32
            states[:, 1] = amplitude_m * np.sin(frequency_rad_s * time_s)
            states[:, 4] = (
                amplitude_m * frequency_rad_s * np.cos(frequency_rad_s * time_s)
            )
            yaw = np.arctan2(states[:, 4], states[:, 3])
            states[:, 12] = (
                -(amplitude_m * frequency_rad_s**2)
                * np.sin(frequency_rad_s * time_s)
                / TRIM_AIRSPEED_M_S
            )
        elif kind == "turn":
            # This 188 m coordinated turn remains inside the lateral-velocity
            # envelope established by the synthetic identification profile.
            yaw_rate_rad_s = 0.08
            radius_m = TRIM_AIRSPEED_M_S / yaw_rate_rad_s
            yaw = yaw_rate_rad_s * time_s
            states[:, 0] = radius_m * np.sin(yaw)
            states[:, 1] = radius_m * (1.0 - np.cos(yaw))
            states[:, 3] = TRIM_AIRSPEED_M_S * np.cos(yaw)
            states[:, 4] = TRIM_AIRSPEED_M_S * np.sin(yaw)
            roll[:] = -math.atan(TRIM_AIRSPEED_M_S * yaw_rate_rad_s / 9.80665)
            states[:, 12] = yaw_rate_rad_s
        states[:, 6:10] = _quaternion_from_euler(roll, pitch, yaw)
        return states

    return reference


def _scenario_definitions() -> tuple[_Scenario, ...]:
    hover_initial = resting_state()
    hover_initial[0:3] = (0.25, -0.20, -0.15)
    attitude_initial = resting_state()
    attitude_initial[6:10] = _quaternion_from_euler(
        np.asarray([0.22]), np.asarray([-0.12]), np.asarray([0.16])
    )[0]
    trim_initial = fixed_wing_trim_state()
    trim_initial[2] = 0.35
    trim_initial[3] = 14.8
    return (
        _Scenario(
            "multirotor_hover",
            "multirotor",
            1.5,
            hover_initial,
            _multirotor_reference("hover"),
        ),
        _Scenario(
            "multirotor_translation",
            "multirotor",
            1.5,
            resting_state(),
            _multirotor_reference("translation"),
        ),
        _Scenario(
            "multirotor_attitude",
            "multirotor",
            1.5,
            attitude_initial,
            _multirotor_reference("attitude"),
        ),
        _Scenario(
            "fixedwing_trim",
            "fixedwing",
            2.0,
            trim_initial,
            _fixed_wing_reference("trim"),
        ),
        _Scenario(
            "fixedwing_altitude",
            "fixedwing",
            2.0,
            fixed_wing_trim_state(),
            _fixed_wing_reference("altitude"),
        ),
        _Scenario(
            "fixedwing_path",
            "fixedwing",
            2.0,
            fixed_wing_trim_state(),
            _fixed_wing_reference("path"),
        ),
        _Scenario(
            "fixedwing_turn",
            "fixedwing",
            2.0,
            fixed_wing_trim_state(),
            _fixed_wing_reference("turn"),
        ),
        _Scenario(
            "fixedwing_flap",
            "fixedwing_flap",
            2.0,
            fixed_wing_trim_state(),
            _fixed_wing_reference("flap"),
        ),
    )


def _fixed_wing_params_with_flap() -> FixedWingDynamicsParams:
    return true_fixed_wing_parameters()._replace(
        log_flap_lift_accel_per_speed_sq=jnp.log(jnp.asarray(0.008)),
        log_flap_drag_accel_per_speed_sq=jnp.log(jnp.asarray(0.0015)),
        flap_pitch_angular_accel_per_speed_sq=jnp.asarray(-0.002),
    )


def _perturbed_multirotor_params() -> DynamicsParams:
    return DynamicsParams.from_physical(
        thrust_accel=5.62,
        angular_accel=(17.0, 17.2, 7.1),
        linear_drag=0.20,
        angular_drag=(0.26, 0.19, 0.15),
        motor_time_constant=0.095,
    )


def _perturbed_fixed_wing_params(
    base: FixedWingDynamicsParams,
) -> FixedWingDynamicsParams:
    physical = base.physical()

    def vector(name: str) -> tuple[float, ...]:
        return tuple(float(value) for value in np.asarray(physical[name]))

    return FixedWingDynamicsParams.from_physical(
        thrust_accel=float(physical["thrust_accel"]) * 1.06,
        lift_accel_per_speed_sq=(float(physical["lift_accel_per_speed_sq"]) * 0.96),
        lift_alpha_accel_per_speed_sq=(
            float(physical["lift_alpha_accel_per_speed_sq"]) * 1.04
        ),
        drag_accel_per_speed_sq=(float(physical["drag_accel_per_speed_sq"]) * 1.08),
        side_force_accel_per_speed=(
            float(physical["side_force_accel_per_speed"]) * 0.94
        ),
        surface_angular_accel_per_speed_sq=tuple(
            value * factor
            for value, factor in zip(
                vector("surface_angular_accel_per_speed_sq"),
                (0.94, 1.06, 0.97),
            )
        ),
        lateral_surface_cross_angular_accel_per_speed_sq=vector(
            "lateral_surface_cross_angular_accel_per_speed_sq"
        ),
        pitch_stability_accel_per_speed_sq=(
            float(physical["pitch_stability_accel_per_speed_sq"]) * 1.05
        ),
        lateral_stability_angular_accel_per_speed_sq=vector(
            "lateral_stability_angular_accel_per_speed_sq"
        ),
        angular_drag_per_speed=tuple(
            value * factor
            for value, factor in zip(
                vector("angular_drag_per_speed"), (1.05, 0.95, 1.04)
            )
        ),
        actuator_time_constant=float(physical["actuator_time_constant"]) * 1.15,
        surface_trim=vector("surface_trim"),
        flap_lift_accel_per_speed_sq=(
            float(physical["flap_lift_accel_per_speed_sq"]) * 0.95
        ),
        flap_drag_accel_per_speed_sq=(
            float(physical["flap_drag_accel_per_speed_sq"]) * 1.08
        ),
        flap_pitch_angular_accel_per_speed_sq=(
            float(physical["flap_pitch_angular_accel_per_speed_sq"]) * 1.05
        ),
        flap_trim=float(physical["flap_trim"]),
    )


def _model_contracts() -> dict[
    str, tuple[ModelParams, TrajectorySpec, RuntimeModelSpec]
]:
    multirotor_training = generate_trajectory(seed=7, duration_s=6.0, dt_s=MODEL_DT_S)
    fixed_wing_training = generate_fixed_wing_trajectory(
        seed=7, duration_s=6.0, dt_s=MODEL_DT_S
    )
    fixed_wing_spec = fixed_wing_training.spec
    flap_spec = make_trajectory_spec(
        FIXED_WING_CONTROL_NAMES + ("flap",),
        family="fixedwing",
        observation_source="simulator_truth",
        configuration_id="synthetic_fixedwing_flap",
    )
    return {
        "multirotor": (
            true_parameters(),
            multirotor_training.spec,
            runtime_spec_from_trajectory(multirotor_training),
        ),
        "fixedwing": (
            true_fixed_wing_parameters(),
            fixed_wing_spec,
            runtime_spec_from_trajectory(fixed_wing_training),
        ),
        "fixedwing_flap": (
            _fixed_wing_params_with_flap(),
            flap_spec,
            runtime_spec_from_trajectory(fixed_wing_training),
        ),
    }


def _trim_command(model: RuntimeDynamicsModel) -> jax.Array:
    if model.input_spec.vehicle.family == "multirotor":
        return hover_control(model.params)
    assert isinstance(model.params, FixedWingDynamicsParams)
    return fixed_wing_trim_control(
        model.params,
        TRIM_AIRSPEED_M_S,
        model.input_spec.control_roles,
    )


def _simulate(
    scenario: _Scenario,
    controller: NMPCController,
    plant_params: ModelParams,
    *,
    optimize: bool,
) -> tuple[np.ndarray, np.ndarray, list[float], int, float]:
    model = controller.model
    interval_count = round(scenario.duration_s / MODEL_DT_S)
    states = np.empty((interval_count + 1, 13), dtype=np.float64)
    commands = np.empty((interval_count, model.command_size), dtype=np.float64)
    states[0] = scenario.initial_state
    previous_command = _trim_command(model)
    plant_latent = control_state_after_history(
        plant_params,
        jnp.asarray(previous_command)[None, :],
        MODEL_DT_S,
        model.input_spec.control_roles,
    )
    warm_start = None
    solve_times: list[float] = []
    fallback_count = 0
    maximum_predicted_validity = 0.0
    for index in range(interval_count):
        if optimize:
            future_time = (
                index + np.arange(controller.prediction_steps + 1)
            ) * MODEL_DT_S
            reference = ReferenceTrajectory(scenario.reference(future_time))
            result = controller.solve(
                jnp.asarray(states[index]),
                reference,
                previous_command,
                latent_state=plant_latent,
                warm_start=warm_start,
            )
            command = result.command
            warm_start = result.warm_start
            solve_times.append(result.diagnostics.solve_time_s)
            fallback_count += int(result.used_fallback)
            maximum_predicted_validity = max(
                maximum_predicted_validity,
                result.diagnostics.maximum_validity_utilization,
            )
        else:
            command = previous_command
        commands[index] = np.asarray(command)
        next_state, plant_latent = step_with_latent(
            plant_params,
            jnp.asarray(states[index]),
            plant_latent,
            jnp.asarray(command),
            MODEL_DT_S,
            model.input_spec.control_roles,
        )
        states[index + 1] = np.asarray(next_state)
        previous_command = command
    return (
        states,
        commands,
        solve_times,
        fallback_count,
        maximum_predicted_validity,
    )


def _normalized_rms(
    controller: NMPCController,
    states: np.ndarray,
    reference: np.ndarray,
) -> float:
    errors = np.asarray(
        jax.vmap(rigid_body_local_error)(jnp.asarray(reference), jnp.asarray(states))
    )
    normalized = errors / np.asarray(controller.tolerances.local_state_scale)
    return float(np.sqrt(np.mean(np.square(normalized))))


def _maximum_actual_validity(model: RuntimeDynamicsModel, states: np.ndarray) -> float:
    utilization = jax.vmap(model.validity_utilization)(jnp.asarray(states))
    return float(np.max(np.asarray(utilization)))


def _scenario_metrics(
    scenario: _Scenario,
    condition: str,
    controller: NMPCController,
    plant_params: ModelParams,
) -> ScenarioMetrics:
    baseline_states, _, _, _, _ = _simulate(
        scenario, controller, plant_params, optimize=False
    )
    states, commands, solve_times, fallback_count, predicted_validity = _simulate(
        scenario, controller, plant_params, optimize=True
    )
    time_s = np.arange(len(states)) * MODEL_DT_S
    reference = scenario.reference(time_s)
    baseline_rms = _normalized_rms(controller, baseline_states, reference)
    candidate_rms = _normalized_rms(controller, states, reference)
    minimum = np.asarray(controller.model.command_minimum)
    maximum = np.asarray(controller.model.command_maximum)
    maximum_bound_violation = float(
        max(
            np.max(minimum - commands),
            np.max(commands - maximum),
            0.0,
        )
    )
    # The cold and warm-start control-flow paths each compile on first use.
    # Exclude both explicitly from the post-JIT distribution.
    post_jit_times = np.asarray(solve_times[2:] or solve_times)
    trim_command = np.asarray(_trim_command(controller.model))
    auxiliary_excursion = 0.0
    auxiliary_roles = controller.model.input_spec.vehicle.auxiliary_controls
    if auxiliary_roles:
        auxiliary_indices = [
            controller.model.input_spec.control_roles.index(role)
            for role in auxiliary_roles
        ]
        auxiliary_excursion = float(
            np.max(
                np.abs(commands[:, auxiliary_indices] - trim_command[auxiliary_indices])
            )
        )
    return ScenarioMetrics(
        name=scenario.name,
        condition=condition,
        platform=controller.model.input_spec.vehicle.family,
        normalized_tracking_rms=candidate_rms,
        baseline_normalized_tracking_rms=baseline_rms,
        tracking_ratio=candidate_rms / baseline_rms,
        maximum_command_bound_violation=maximum_bound_violation,
        maximum_validity_utilization=max(
            predicted_validity,
            _maximum_actual_validity(controller.model, states),
        ),
        maximum_auxiliary_command_excursion=auxiliary_excursion,
        fallback_count=fallback_count,
        finite=bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(commands))
            and np.isfinite(candidate_rms)
        ),
        warmup_solve_time_s=float(np.max(solve_times[:2])),
        solve_time_median_s=float(np.median(post_jit_times)),
        solve_time_p90_s=float(np.quantile(post_jit_times, 0.90)),
        solve_time_maximum_s=float(np.max(post_jit_times)),
    )


def _geometric_mean(values: list[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values)))))


def _processor_name() -> str:
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine() or "unknown"


def run_nmpc_benchmark() -> dict[str, object]:
    """Run the maintained nominal/mismatch suite and return its evidence."""

    contracts = _model_contracts()
    controllers: dict[str, NMPCController] = {}
    for key, (params, spec, runtime_spec) in contracts.items():
        model = RuntimeDynamicsModel(
            params,
            spec,
            runtime_spec,
            DirectActuationMap(spec.controls),
        )
        controllers[key] = NMPCController(model)

    results: list[ScenarioMetrics] = []
    for scenario in _scenario_definitions():
        controller = controllers[scenario.model_key]
        nominal_params = contracts[scenario.model_key][0]
        mismatch_params: ModelParams
        if scenario.model_key == "multirotor":
            mismatch_params = _perturbed_multirotor_params()
        else:
            assert isinstance(nominal_params, FixedWingDynamicsParams)
            mismatch_params = _perturbed_fixed_wing_params(nominal_params)
        results.append(
            _scenario_metrics(scenario, "nominal", controller, nominal_params)
        )
        results.append(
            _scenario_metrics(scenario, "model_mismatch", controller, mismatch_params)
        )

    nominal = [result for result in results if result.condition == "nominal"]
    mismatch = [result for result in results if result.condition == "model_mismatch"]
    nominal_ratio = _geometric_mean([result.tracking_ratio for result in nominal])
    mismatch_ratio = _geometric_mean([result.tracking_ratio for result in mismatch])
    scenario_medians = [result.solve_time_median_s for result in results]
    checks = {
        "finite": all(result.finite for result in results),
        "hard_command_bounds": all(
            result.maximum_command_bound_violation <= 1e-6 for result in results
        ),
        "no_fallbacks": all(result.fallback_count == 0 for result in results),
        "nominal_aggregate": nominal_ratio <= ACCEPTANCE_AGGREGATE_RATIO,
        "nominal_individual": all(
            result.tracking_ratio <= ACCEPTANCE_NOMINAL_SCENARIO_RATIO
            for result in nominal
        ),
        "mismatch_aggregate": mismatch_ratio < 1.0,
        "mismatch_individual": all(
            result.tracking_ratio <= ACCEPTANCE_MISMATCH_SCENARIO_RATIO
            for result in mismatch
        ),
        "mismatch_validity": all(
            result.maximum_validity_utilization <= 1.0 for result in mismatch
        ),
        "flap_authority_exercised": next(
            result.maximum_auxiliary_command_excursion
            for result in nominal
            if result.name == "fixedwing_flap"
        )
        > 1e-4,
    }
    return {
        "format_version": 1,
        "baseline": "constant model-derived hover or level-flight trim command",
        "normalized_error": (
            "RMS of 12 local rigid-body errors divided by maintained physical "
            "tracking tolerances"
        ),
        "thresholds": {
            "nominal_aggregate_ratio_maximum": ACCEPTANCE_AGGREGATE_RATIO,
            "nominal_individual_ratio_maximum": (ACCEPTANCE_NOMINAL_SCENARIO_RATIO),
            "mismatch_individual_ratio_maximum": (ACCEPTANCE_MISMATCH_SCENARIO_RATIO),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": _processor_name(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "model_step_s": MODEL_DT_S,
        },
        "summary": {
            "passed": all(checks.values()),
            "checks": checks,
            "nominal_geometric_mean_tracking_ratio": nominal_ratio,
            "mismatch_geometric_mean_tracking_ratio": mismatch_ratio,
            "post_jit_solve_time_s": {
                "median_of_scenario_medians": float(np.median(scenario_medians)),
                "maximum_scenario_p90": max(
                    result.solve_time_p90_s for result in results
                ),
                "maximum_observed": max(
                    result.solve_time_maximum_s for result in results
                ),
            },
        },
        "scenarios": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the maintained Glassbox NMPC acceptance benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON evidence path; stdout is always populated.",
    )
    args = parser.parse_args()
    report = run_nmpc_benchmark()
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not report["summary"]["passed"]:  # type: ignore[index]
        raise SystemExit(1)


if __name__ == "__main__":
    main()
