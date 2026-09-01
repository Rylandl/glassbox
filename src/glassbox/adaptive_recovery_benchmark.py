"""Prewarmed synthetic recovery after an explicit multirotor configuration change."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.data import Trajectory
from glassbox.dynamics import (
    DynamicsParams,
    control_state_after_history,
    hover_control,
    quaternion_to_rotation,
    step_with_latent,
)
from glassbox.evaluation import windowed_rollout_evaluation
from glassbox.geometry import rigid_body_local_error
from glassbox.nmpc import NMPCController, TrackingTolerances
from glassbox.parameter_prior import StructuredParameterPrior
from glassbox.runtime import (
    DirectActuationMap,
    RuntimeDynamicsModel,
    runtime_spec_from_trajectory,
)
from glassbox.synthetic import generate_trajectory, resting_state, true_parameters

SAMPLE_DT_S = 0.02
ADAPTATION_HORIZON_STEPS = 5
CONTROL_HORIZON_STEPS = 30
SHORT_HORIZON_FLEET_DURATION_S = 1.2
CONTROL_HORIZON_FLEET_DURATION_S = 2.0
FLEET_PROFILE_BASE_SEEDS = (100, 200, 300)
ADAPTATION_DURATION_S = 0.8
EVALUATION_DURATION_S = 1.6
RECOVERY_DURATION_S = 1.2
RECOVERY_TAIL_DURATION_S = 0.4
FLEET_LOG_ARM_LENGTH_RATIOS = (-0.25, -0.125, 0.0, 0.125, 0.25)
TARGET_LOG_ARM_LENGTH_RATIO = 0.20
BENCHMARK_METHOD_VERSION = 4
BENCHMARK_SOURCE_FILES = (
    "adaptation.py",
    "adaptive_recovery_benchmark.py",
    "belief.py",
    "covariance.py",
    "data.py",
    "dynamics.py",
    "evaluation.py",
    "geometry.py",
    "linearization.py",
    "nmpc/__init__.py",
    "nmpc/solver.py",
    "nmpc/types.py",
    "parameter_prior.py",
    "runtime.py",
    "synthetic.py",
)
_NONDETERMINISTIC_RECOVERY_FIELDS = frozenset(
    {
        "prewarm_wall_time_s",
        "solve_time_median_s",
        "solve_time_p90_s",
        "solve_time_maximum_s",
    }
)


def _json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def adaptive_recovery_source_fingerprint() -> str:
    """Bind evidence to the maintained source modules that produce it."""

    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative_path in BENCHMARK_SOURCE_FILES:
        path = source_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def normalized_adaptive_recovery_report(report: dict[str, Any]) -> dict[str, Any]:
    """Remove machine- and timing-dependent fields for artifact freshness checks."""

    normalized = json.loads(json.dumps(report, allow_nan=False))
    normalized.pop("environment", None)
    for recovery in normalized.get("recovery", []):
        for field_name in _NONDETERMINISTIC_RECOVERY_FIELDS:
            recovery.pop(field_name, None)
    return normalized


@dataclass(frozen=True)
class RecoveryMetrics:
    """One prewarmed closed-loop recovery trace summarized without a gate."""

    condition: str
    model_role: str
    predictive_error_current: bool | None
    parameter_uncertainty_available: bool
    prediction_horizon_s: float
    normalized_tracking_rms: float
    tail_normalized_tracking_rms: float
    normalized_attitude_rate_rms: float
    tail_normalized_attitude_rate_rms: float
    terminal_attitude_error_rad: float
    terminal_angular_velocity_error_rad_s: float
    terminal_attitude_rate_tolerance_entry_time_s: float | None
    maximum_actual_validity_utilization: float
    maximum_predicted_validity_utilization: float
    maximum_command_bound_violation: float
    minimum_command_authority_fraction: float
    median_command_authority_fraction: float
    maximum_normalized_model_uncertainty_standard_deviation: float
    maximum_next_step_robust_validity_utilization: float
    support_horizon_s: float
    maximum_support_horizon_robust_validity_utilization: float
    support_filter_mode_counts: dict[str, int]
    support_filter_applied_count: int
    inside_support_step_count: int
    inside_support_best_effort_count: int
    outside_support_step_count: int
    outside_support_best_effort_count: int
    fallback_count: int
    finite: bool
    prewarm_wall_time_s: float
    solve_time_median_s: float
    solve_time_p90_s: float
    solve_time_maximum_s: float


def _arm_configuration_parameters(
    base: DynamicsParams,
    log_arm_length_ratio: float,
) -> DynamicsParams:
    """Map arm length into effective roll/pitch authority for this diagnostic.

    With motor mass dominating roll/pitch inertia, torque scales with arm length
    while inertia scales approximately with its square. The effective angular
    authority therefore scales inversely with arm length. Glassbox still learns
    the effective coefficient; it does not require this geometric decomposition.
    """

    names = structured_parameter_names(base)
    indices = {name: index for index, name in enumerate(names)}
    vector = np.asarray(structured_parameter_vector(base), dtype=np.float64).copy()
    for name in ("log_angular_accel[0]", "log_angular_accel[1]"):
        vector[indices[name]] -= log_arm_length_ratio
    return with_structured_parameter_vector(base, jnp.asarray(vector))


def _configuration_trajectory(
    params: DynamicsParams,
    log_arm_length_ratio: float,
    *,
    seed: int,
    duration_s: float,
    source_group: str,
) -> Trajectory:
    trajectory = generate_trajectory(
        seed=seed,
        duration_s=duration_s,
        dt_s=SAMPLE_DT_S,
        params=params,
    )
    arm_ratio = math.exp(log_arm_length_ratio)
    vehicle = replace(
        trajectory.spec.vehicle,
        configuration_id=f"synthetic_adjustable_arm_{arm_ratio:.6f}",
        fixed_states={"arm_length_ratio": arm_ratio},
    )
    return replace(
        trajectory,
        spec=replace(trajectory.spec, vehicle=vehicle),
        labels={
            **trajectory.labels,
            "source_group": source_group,
            "arm_length_ratio": arm_ratio,
        },
    )


def _build_beliefs() -> tuple[
    DynamicsBelief,
    DynamicsBelief,
    DynamicsParams,
    StructuredParameterPrior,
    dict[str, Any],
]:
    base = true_parameters()
    support_trajectory = generate_trajectory(
        seed=700,
        duration_s=6.0,
        dt_s=SAMPLE_DT_S,
        params=base,
    )
    runtime_spec = runtime_spec_from_trajectory(support_trajectory)

    member_params = tuple(
        _arm_configuration_parameters(base, value)
        for value in FLEET_LOG_ARM_LENGTH_RATIOS
    )
    member_labels = tuple(
        f"arm-ratio-{math.exp(value):.6f}" for value in FLEET_LOG_ARM_LENGTH_RATIOS
    )
    members = tuple(
        DynamicsBelief(
            params=params,
            input_spec=_configuration_trajectory(
                params,
                log_ratio,
                seed=800 + index,
                duration_s=0.2,
                source_group=member_labels[index],
            ).spec,
            runtime_spec=runtime_spec,
        )
        for index, (params, log_ratio) in enumerate(
            zip(member_params, FLEET_LOG_ARM_LENGTH_RATIOS)
        )
    )
    fleet_prior = StructuredParameterPrior.from_beliefs(
        members,
        source="synthetic_adjustable_arm_configuration_fleet",
        member_labels=member_labels,
    )

    samples_by_horizon: dict[float, list[EmpiricalErrorSample]] = {
        ADAPTATION_HORIZON_STEPS * SAMPLE_DT_S: [],
        CONTROL_HORIZON_STEPS * SAMPLE_DT_S: [],
    }
    for profile_index, base_seed in enumerate(FLEET_PROFILE_BASE_SEEDS):
        for index, (params, log_ratio, label) in enumerate(
            zip(member_params, FLEET_LOG_ARM_LENGTH_RATIOS, member_labels)
        ):
            short_trajectory = _configuration_trajectory(
                params,
                log_ratio,
                seed=base_seed + index,
                duration_s=SHORT_HORIZON_FLEET_DURATION_S,
                source_group=label,
            )
            _, errors = windowed_rollout_evaluation(
                base,
                short_trajectory,
                horizon_steps=ADAPTATION_HORIZON_STEPS,
                stride_steps=ADAPTATION_HORIZON_STEPS,
            )
            samples_by_horizon[ADAPTATION_HORIZON_STEPS * SAMPLE_DT_S].append(
                EmpiricalErrorSample(
                    errors=errors,
                    source_group=label,
                    trajectory_id=f"{label}-short-profile-{profile_index}",
                )
            )
            control_trajectory = _configuration_trajectory(
                params,
                log_ratio,
                seed=base_seed + index,
                duration_s=CONTROL_HORIZON_FLEET_DURATION_S,
                source_group=label,
            )
            _, errors = windowed_rollout_evaluation(
                base,
                control_trajectory,
                horizon_steps=CONTROL_HORIZON_STEPS,
                stride_steps=CONTROL_HORIZON_STEPS,
            )
            samples_by_horizon[CONTROL_HORIZON_STEPS * SAMPLE_DT_S].append(
                EmpiricalErrorSample(
                    errors=errors,
                    source_group=label,
                    trajectory_id=f"{label}-control-profile-{profile_index}",
                )
            )
    predictive_error = replace(
        EmpiricalHorizonPredictiveError.from_samples(
            {horizon: tuple(samples) for horizon, samples in samples_by_horizon.items()}
        ),
        source="synthetic_arm_fleet_rollout_endpoints",
    )

    target_params = _arm_configuration_parameters(
        base,
        TARGET_LOG_ARM_LENGTH_RATIO,
    )
    adaptation_telemetry = _configuration_trajectory(
        target_params,
        TARGET_LOG_ARM_LENGTH_RATIO,
        seed=21,
        duration_s=ADAPTATION_DURATION_S,
        source_group="target-configuration-adaptation",
    )

    # The vehicle was identified before its geometry changed. Unchanged
    # coefficients are therefore anchored by that vehicle-local evidence, while
    # the fleet-derived between-configuration covariance describes the supported
    # arm-change direction. The broad fleet completion is reported but is not
    # substituted for evidence that unchanged coefficients moved.
    parameter_belief = LocalGaussianParameterBelief(
        parameter_names=structured_parameter_names(base),
        covariance=fleet_prior.between_member_covariance,
        source="prewarmed_vehicle_plus_fleet_configuration_delta",
        evidence_count=fleet_prior.member_count,
        effective_sample_count=float(fleet_prior.member_count),
    )
    belief = DynamicsBelief(
        params=base,
        input_spec=adaptation_telemetry.spec,
        runtime_spec=runtime_spec,
        predictive_error=predictive_error,
        parameter_belief=parameter_belief,
        provenance={
            "benchmark_role": "prechange_vehicle_belief",
            "configuration_change": {
                "kind": "arm_length_ratio",
                "from": 1.0,
                "to": math.exp(TARGET_LOG_ARM_LENGTH_RATIO),
            },
        },
    )
    updated, update_report = belief.update(adaptation_telemetry)

    evaluation_telemetry = _configuration_trajectory(
        target_params,
        TARGET_LOG_ARM_LENGTH_RATIO,
        seed=22,
        duration_s=EVALUATION_DURATION_S,
        source_group="target-configuration-independent-evaluation",
    )
    _, before_errors = windowed_rollout_evaluation(
        belief.params,
        evaluation_telemetry,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    _, after_errors = windowed_rollout_evaluation(
        updated.params,
        evaluation_telemetry,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    tolerances = np.asarray(
        TrackingTolerances.for_platform("multirotor").local_state_scale
    )
    before_rms = float(np.sqrt(np.mean(np.square(before_errors / tolerances))))
    after_rms = float(np.sqrt(np.mean(np.square(after_errors / tolerances))))
    evidence = {
        "fleet": {
            "member_count": fleet_prior.member_count,
            "parameter_count": len(fleet_prior.parameter_names),
            "empirical_rank": fleet_prior.empirical_rank,
            "completion_fraction_in_natural_coordinates": (
                fleet_prior.completion_fraction_in_natural_coordinates
            ),
            "configuration_delta_covariance_rank": int(
                np.linalg.matrix_rank(fleet_prior.between_member_covariance)
            ),
            "predictive_error_horizons_s": list(predictive_error.horizons_s),
            "predictive_error_group_count": list(
                predictive_error.independent_group_count
            ),
            "predictive_error_raw_endpoint_count": list(
                predictive_error.raw_sample_count
            ),
        },
        "adaptation": update_report.to_dict(),
        "independent_prediction": {
            "horizon_s": CONTROL_HORIZON_STEPS * SAMPLE_DT_S,
            "normalized_rms_before": before_rms,
            "normalized_rms_after": after_rms,
            "normalized_rms_ratio": after_rms / before_rms,
        },
    }
    return belief, updated, target_params, fleet_prior, evidence


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _recovery_initial_state(belief: DynamicsBelief) -> np.ndarray:
    state = resting_state()
    state[0:3] = (0.25, -0.20, -0.15)
    quaternion = _quaternion_from_euler(0.24, -0.17, 0.12)
    state[6:10] = quaternion
    envelope = belief.runtime_spec.validity_envelope
    body_velocity = np.asarray(envelope.body_velocity_center_m_s) + 0.25 * np.asarray(
        envelope.body_velocity_half_width_m_s
    ) * np.asarray((1.0, -1.0, -1.0))
    state[3:6] = np.asarray(quaternion_to_rotation(jnp.asarray(quaternion))) @ (
        body_velocity
    )
    state[10:13] = np.asarray(envelope.angular_velocity_center_rad_s) + 0.35 * (
        np.asarray(envelope.angular_velocity_half_width_rad_s)
        * np.asarray((1.0, -1.0, 1.0))
    )
    return state


def _prewarm_controller(
    controller: NMPCController,
    state: np.ndarray,
    target_params: DynamicsParams,
) -> float:
    previous = hover_control(target_params)
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    started_at = time.perf_counter()
    cold = controller.solve(
        jnp.asarray(state),
        reference,
        previous,
        applied_command=previous,
    )
    jax.block_until_ready(cold.command)
    warm = controller.solve(
        jnp.asarray(state),
        reference,
        previous,
        applied_command=previous,
        warm_start=cold.warm_start,
    )
    jax.block_until_ready(warm.command)
    return time.perf_counter() - started_at


def _terminal_tolerance_entry_time(
    errors: np.ndarray,
    tolerances: TrackingTolerances,
) -> float | None:
    scale = np.asarray(tolerances.local_state_scale)
    within = np.all(np.abs(errors[:, 6:9]) <= scale[6:9], axis=1) & np.all(
        np.abs(errors[:, 9:12]) <= scale[9:12],
        axis=1,
    )
    sustained = np.logical_and.accumulate(within[::-1])[::-1]
    indices = np.flatnonzero(sustained)
    return None if not len(indices) else float(indices[0] * SAMPLE_DT_S)


def _simulate_recovery(
    condition: str,
    model_role: str,
    controller: NMPCController,
    target_params: DynamicsParams,
    initial_state: np.ndarray,
) -> RecoveryMetrics:
    prewarm_time = _prewarm_controller(controller, initial_state, target_params)
    interval_count = round(RECOVERY_DURATION_S / SAMPLE_DT_S)
    states = np.empty((interval_count + 1, 13), dtype=np.float64)
    commands = np.empty((interval_count, controller.model.command_size))
    states[0] = initial_state
    previous = hover_control(target_params)
    plant_latent = control_state_after_history(
        target_params,
        previous[None, :],
        SAMPLE_DT_S,
        controller.model.input_spec.control_roles,
    )
    reference = controller.hold_reference(jnp.asarray(resting_state()))
    warm_start = None
    solve_times: list[float] = []
    authority: list[float] = []
    uncertainty: list[float] = []
    predicted_validity: list[float] = []
    next_step_robust_validity: list[float] = []
    support_horizon_robust_validity: list[float] = []
    current_validity: list[float] = []
    support_modes: list[str] = []
    support_filter_applied_count = 0
    fallback_count = 0
    for index in range(interval_count):
        result = controller.solve(
            jnp.asarray(states[index]),
            reference,
            previous,
            applied_command=plant_latent[: controller.model.command_size],
            warm_start=warm_start,
        )
        command = result.command
        warm_start = result.warm_start
        commands[index] = np.asarray(command)
        solve_times.append(result.diagnostics.solve_time_s)
        authority.append(result.diagnostics.command_authority_fraction)
        uncertainty.append(
            result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
        )
        predicted_validity.append(result.diagnostics.maximum_validity_utilization)
        next_step_robust_validity.append(
            result.diagnostics.next_step_robust_validity_utilization
        )
        support_horizon_robust_validity.append(
            result.diagnostics.support_horizon_maximum_robust_validity_utilization
        )
        current_validity.append(result.diagnostics.current_validity_utilization)
        support_modes.append(result.diagnostics.support_filter_mode.value)
        support_filter_applied_count += int(result.diagnostics.support_filter_applied)
        fallback_count += int(result.used_fallback)
        next_state, plant_latent = step_with_latent(
            target_params,
            jnp.asarray(states[index]),
            plant_latent,
            command,
            SAMPLE_DT_S,
            controller.model.input_spec.control_roles,
        )
        states[index + 1] = np.asarray(next_state)
        previous = command

    reference_states = np.repeat(resting_state()[None, :], len(states), axis=0)
    errors = np.asarray(
        jax.vmap(rigid_body_local_error)(
            jnp.asarray(reference_states),
            jnp.asarray(states),
        )
    )
    scale = np.asarray(controller.tolerances.local_state_scale)
    normalized = errors / scale
    tail_steps = round(RECOVERY_TAIL_DURATION_S / SAMPLE_DT_S)
    actual_validity = np.asarray(
        jax.vmap(controller.model.validity_utilization)(jnp.asarray(states))
    )
    minimum = np.asarray(controller.model.command_minimum)
    maximum = np.asarray(controller.model.command_maximum)
    command_violation = float(
        max(
            np.max(minimum - commands),
            np.max(commands - maximum),
            0.0,
        )
    )
    runtime_belief = controller.belief
    return RecoveryMetrics(
        condition=condition,
        model_role=model_role,
        predictive_error_current=(
            runtime_belief.predictive_error_current
            if runtime_belief.predictive_error_available
            else None
        ),
        parameter_uncertainty_available=(
            runtime_belief.parameter_uncertainty_available
        ),
        prediction_horizon_s=controller.prediction_horizon_s,
        normalized_tracking_rms=float(np.sqrt(np.mean(np.square(normalized)))),
        tail_normalized_tracking_rms=float(
            np.sqrt(np.mean(np.square(normalized[-tail_steps:])))
        ),
        normalized_attitude_rate_rms=float(
            np.sqrt(np.mean(np.square(normalized[:, 6:12])))
        ),
        tail_normalized_attitude_rate_rms=float(
            np.sqrt(np.mean(np.square(normalized[-tail_steps:, 6:12])))
        ),
        terminal_attitude_error_rad=float(np.linalg.norm(errors[-1, 6:9])),
        terminal_angular_velocity_error_rad_s=float(np.linalg.norm(errors[-1, 9:12])),
        terminal_attitude_rate_tolerance_entry_time_s=_terminal_tolerance_entry_time(
            errors,
            controller.tolerances,
        ),
        maximum_actual_validity_utilization=float(np.max(actual_validity)),
        maximum_predicted_validity_utilization=max(predicted_validity),
        maximum_command_bound_violation=command_violation,
        minimum_command_authority_fraction=min(authority),
        median_command_authority_fraction=float(np.median(authority)),
        maximum_normalized_model_uncertainty_standard_deviation=max(uncertainty),
        maximum_next_step_robust_validity_utilization=max(next_step_robust_validity),
        support_horizon_s=result.diagnostics.support_horizon_s,
        maximum_support_horizon_robust_validity_utilization=max(
            support_horizon_robust_validity
        ),
        support_filter_mode_counts=dict(sorted(Counter(support_modes).items())),
        support_filter_applied_count=support_filter_applied_count,
        inside_support_step_count=sum(
            value <= 1.0 + 1e-6 for value in current_validity
        ),
        inside_support_best_effort_count=sum(
            current <= 1.0 + 1e-6 and mode == "boundary_best_effort"
            for current, mode in zip(current_validity, support_modes)
        ),
        outside_support_step_count=sum(
            value > 1.0 + 1e-6 for value in current_validity
        ),
        outside_support_best_effort_count=sum(
            current > 1.0 + 1e-6 and mode == "recovery_best_effort"
            for current, mode in zip(current_validity, support_modes)
        ),
        fallback_count=fallback_count,
        finite=bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(commands))
            and np.all(np.isfinite(normalized))
        ),
        prewarm_wall_time_s=prewarm_time,
        solve_time_median_s=float(np.median(solve_times)),
        solve_time_p90_s=float(np.quantile(solve_times, 0.90)),
        solve_time_maximum_s=max(solve_times),
    )


def run_adaptive_recovery_benchmark() -> dict[str, Any]:
    """Run one fixed diagnostic with no acceptance thresholds or tuning surface."""

    belief, updated, target_params, fleet_prior, evidence = _build_beliefs()
    initial_state = _recovery_initial_state(belief)
    equal_horizon_runtime = replace(
        belief.runtime_spec,
        certified_prediction_horizon_s=CONTROL_HORIZON_STEPS * SAMPLE_DT_S,
        certification_source="synthetic equal-horizon recovery comparison",
    )
    actuation = DirectActuationMap(belief.input_spec.controls)
    controllers = (
        (
            "stale_belief",
            "prechange mean plus current fleet forecast error",
            NMPCController(belief),
        ),
        (
            "adapted_belief",
            "adapted mean plus parameter uncertainty; forecast error stale",
            NMPCController(updated),
        ),
        (
            "adapted_mean_point",
            "adapted mean without uncertainty",
            NMPCController(
                RuntimeDynamicsModel(
                    updated.params,
                    belief.input_spec,
                    equal_horizon_runtime,
                    actuation,
                )
            ),
        ),
        (
            "oracle_mean_point",
            "hidden target mean without uncertainty",
            NMPCController(
                RuntimeDynamicsModel(
                    target_params,
                    belief.input_spec,
                    equal_horizon_runtime,
                    actuation,
                )
            ),
        ),
    )
    recovery = tuple(
        _simulate_recovery(
            condition,
            role,
            controller,
            target_params,
            initial_state,
        )
        for condition, role, controller in controllers
    )
    by_condition = {item.condition: item for item in recovery}
    stale = by_condition["stale_belief"]
    adapted = by_condition["adapted_belief"]
    oracle = by_condition["oracle_mean_point"]
    semantics = {
        "diagnostic_only": True,
        "acceptance_gate": False,
        "synthetic": True,
        "prewarmed_controller": True,
        "compile_latency_excluded_from_recovery_timing": True,
        "independent_fallback_controller_included": False,
        "support_candidates_derived_only_from_nmpc_and_previous_command": True,
        "solver_failure_returns_explicit_bounded_hold": True,
        "independent_flight_watchdog_included": False,
        "actuator_reaction_horizon_belief_support_filter_included": True,
        "hard_prediction_horizon_validity_constraint_included": False,
        "support_filter_best_effort_when_no_candidate_is_feasible": True,
        "flight_safety_claim": False,
        "throw_to_recover_claim": False,
        "posterior_calibration_claim": False,
        "prechange_vehicle_anchors_unchanged_parameters": True,
        "configuration_delta_direction_derived_from_fleet": True,
        "adaptation_and_evaluation_telemetry_disjoint": True,
        "validation_actuator_context_excluded_from_evidence": True,
        "stale_predictive_error_is_not_applied_at_runtime": True,
    }
    configuration = {
        "sample_period_s": SAMPLE_DT_S,
        "adaptation_horizon_s": ADAPTATION_HORIZON_STEPS * SAMPLE_DT_S,
        "control_horizon_s": CONTROL_HORIZON_STEPS * SAMPLE_DT_S,
        "short_horizon_fleet_duration_s": SHORT_HORIZON_FLEET_DURATION_S,
        "control_horizon_fleet_duration_s": CONTROL_HORIZON_FLEET_DURATION_S,
        "adaptation_duration_s": ADAPTATION_DURATION_S,
        "evaluation_duration_s": EVALUATION_DURATION_S,
        "recovery_duration_s": RECOVERY_DURATION_S,
        "reported_tail_duration_s": RECOVERY_TAIL_DURATION_S,
        "fleet_profile_base_seeds": list(FLEET_PROFILE_BASE_SEEDS),
        "support_trajectory_seed": 700,
        "fleet_member_spec_seed_start": 800,
        "adaptation_trajectory_seed": 21,
        "evaluation_trajectory_seed": 22,
        "arm_length_ratio_before": 1.0,
        "arm_length_ratio_after": math.exp(TARGET_LOG_ARM_LENGTH_RATIO),
        "fleet_arm_length_ratios": [
            math.exp(value) for value in FLEET_LOG_ARM_LENGTH_RATIOS
        ],
        "effective_authority_mapping": (
            "roll_pitch_angular_authority_inverse_arm_length_ratio"
        ),
        "recovery_conditions": [
            {"condition": condition, "model_role": role}
            for condition, role, _ in controllers
        ],
        "recovery_initial_state": initial_state.tolist(),
        "tracking_tolerances": {
            "local_state_scale": np.asarray(
                TrackingTolerances.for_platform("multirotor").local_state_scale
            ).tolist()
        },
    }
    return {
        "format_version": 4,
        "artifact_type": "glassbox_synthetic_adaptive_recovery_diagnostic",
        "implementation": {
            "method_version": BENCHMARK_METHOD_VERSION,
            "source_files": list(BENCHMARK_SOURCE_FILES),
            "source_sha256": adaptive_recovery_source_fingerprint(),
            "scenario_sha256": _json_fingerprint(
                {
                    "method_version": BENCHMARK_METHOD_VERSION,
                    "semantics": semantics,
                    "configuration": configuration,
                }
            ),
        },
        "semantics": semantics,
        "configuration": configuration,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "evidence": evidence,
        "recovery": [asdict(item) for item in recovery],
        "comparisons": {
            "adapted_vs_stale_tail_tracking_rms_ratio": (
                adapted.tail_normalized_tracking_rms
                / stale.tail_normalized_tracking_rms
            ),
            "adapted_vs_stale_tail_attitude_rate_rms_ratio": (
                adapted.tail_normalized_attitude_rate_rms
                / stale.tail_normalized_attitude_rate_rms
            ),
            "adapted_vs_oracle_tail_tracking_rms_ratio": (
                adapted.tail_normalized_tracking_rms
                / oracle.tail_normalized_tracking_rms
            ),
            "adapted_vs_oracle_tail_attitude_rate_rms_ratio": (
                adapted.tail_normalized_attitude_rate_rms
                / oracle.tail_normalized_attitude_rate_rms
            ),
        },
        "observations": {
            "update_applied": evidence["adaptation"]["applied"],
            "independent_prediction_improved": (
                evidence["independent_prediction"]["normalized_rms_after"]
                < evidence["independent_prediction"]["normalized_rms_before"]
            ),
            "all_recovery_traces_finite": all(item.finite for item in recovery),
            "all_commands_within_bounds": all(
                item.maximum_command_bound_violation <= 1e-6 for item in recovery
            ),
            "all_recovery_traces_without_fallback": all(
                item.fallback_count == 0 for item in recovery
            ),
            "all_actual_recovery_within_validity_support": all(
                item.maximum_actual_validity_utilization <= 1.0 for item in recovery
            ),
            "all_next_step_robust_predictions_within_validity_support": all(
                item.maximum_next_step_robust_validity_utilization <= 1.0
                for item in recovery
            ),
            "all_support_horizon_projections_within_validity_support": all(
                item.maximum_support_horizon_robust_validity_utilization <= 1.0
                for item in recovery
            ),
            "all_full_nmpc_predictions_within_validity_support": all(
                item.maximum_predicted_validity_utilization <= 1.0 for item in recovery
            ),
            "support_filter_intervened": any(
                item.support_filter_applied_count > 0 for item in recovery
            ),
            "all_inside_support_steps_found_supported_commands": all(
                item.inside_support_best_effort_count == 0 for item in recovery
            ),
            "outside_support_progress_condition_exercised": any(
                item.outside_support_step_count > 0 for item in recovery
            ),
            "all_outside_support_steps_found_progress_commands": (
                all(item.outside_support_best_effort_count == 0 for item in recovery)
                if any(item.outside_support_step_count > 0 for item in recovery)
                else None
            ),
            "adapted_tail_tracking_better_than_stale": (
                adapted.tail_normalized_tracking_rms
                < stale.tail_normalized_tracking_rms
            ),
            "adapted_tail_attitude_rate_better_than_stale": (
                adapted.tail_normalized_attitude_rate_rms
                < stale.tail_normalized_attitude_rate_rms
            ),
        },
        "limitations": [
            "The plant, fleet, telemetry, and configuration change are synthetic.",
            "Controller compilation is prewarmed and excluded from recovery timing.",
            "No independent fallback or airframe-specific recovery controller is included; support candidates are projections between the optimized NMPC command and the previous bounded command.",
            "The prechange vehicle model anchors coefficients not affected by the known arm change.",
            "The benchmark starts from a bounded in-envelope disturbance, not an unknown physical throw.",
            "The support filter evaluates held projected commands over a bounded actuator-reaction horizon with componentwise one-standard-deviation margins; it is not a hard full-prediction-horizon or flight-safety guarantee.",
            "If no enumerated candidate satisfies the inside-support or recovery-progress condition, the least-bad bounded command is returned and labeled best effort.",
            "Boundary filtering requires additional belief rollouts; this diagnostic does not establish a hard real-time deadline on other hardware or uncertainty representations.",
            "Fleet completion uncertainty is reported but not treated as evidence that every coefficient changed.",
        ],
        "fleet_prior_completion_fraction": (
            fleet_prior.completion_fraction_in_natural_coordinates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_adaptive_recovery_benchmark()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"wrote {args.output}")
    else:
        print(payload, end="")
    prediction = report["evidence"]["independent_prediction"]
    comparisons = report["comparisons"]
    print(
        "independent 0.6 s prediction RMS: "
        f"{prediction['normalized_rms_before']:.6f} -> "
        f"{prediction['normalized_rms_after']:.6f}"
    )
    print(
        "adapted/stale recovery-tail ratios: "
        f"tracking={comparisons['adapted_vs_stale_tail_tracking_rms_ratio']:.3f}, "
        "attitude+rate="
        f"{comparisons['adapted_vs_stale_tail_attitude_rate_rms_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
