"""Prewarmed synthetic recovery after an explicit multirotor configuration change."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import glassbox
from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import (
    EmpiricalErrorSample,
    ForecastErrorEnvelope,
)
from glassbox.belief.information import (
    ParameterInformation,
    estimable_structured_parameters,
)
from glassbox.belief.parameter_evidence import innovation_noise
from glassbox.control.fitted import NMPCController, default_solver_policy
from glassbox.control.plan import TrackingTolerances
from glassbox.core.data import Trajectory
from glassbox.core.dynamics import (
    DynamicsParams,
    control_state_after_history,
    hover_control,
    quaternion_to_rotation,
    step_with_latent,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import (
    quaternion_from_euler,
    rigid_body_local_error,
)
from glassbox.core.metrics import one_step_innovations, predict_windows
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import generate_trajectory, resting_state, true_parameters

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
BENCHMARK_METHOD_VERSION = 9
BENCHMARK_SOURCE_FILES = (
    "belief/belief.py",
    "belief/forecast_error.py",
    "belief/information.py",
    "belief/linearization.py",
    "belief/parameter_evidence.py",
    "belief/update.py",
    "control/fitted.py",
    "control/plan.py",
    "control/solver.py",
    "core/data.py",
    "core/dynamics.py",
    "core/families.py",
    "core/geometry.py",
    "core/metrics.py",
    "core/diagnostics.py",
    "core/model.py",
    "core/synthetic.py",
    "workflows/benchmarks/recovery.py",
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
    """Bind evidence to the maintained source modules that produce it.

    Every entry of :data:`BENCHMARK_SOURCE_FILES` is written relative to the
    ``glassbox`` package root, so the root is resolved from the package rather
    than from this module's own depth inside it.
    """

    source_root = Path(glassbox.__file__).resolve().parent
    digest = hashlib.sha256()
    for relative_path in BENCHMARK_SOURCE_FILES:
        path = source_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RecoveryMetrics:
    """One prewarmed closed-loop recovery trace summarized without a gate."""

    condition: str
    model_role: str
    forecast_error_available: bool
    parameter_information_rank: int
    prediction_horizon_s: float
    normalized_tracking_rms: float
    tail_normalized_tracking_rms: float
    normalized_attitude_rate_rms: float
    tail_normalized_attitude_rate_rms: float
    terminal_attitude_error_rad: float
    terminal_angular_velocity_error_rad_s: float
    terminal_attitude_rate_tolerance_entry_time_s: float | None
    maximum_actual_validity_utilization: float
    maximum_predicted_validity_utilization: float | None
    maximum_command_bound_violation: float
    maximum_normalized_model_uncertainty_standard_deviation: float | None
    parameter_uncertainty_complete: bool
    unresolved_parameters_allowed: bool
    inside_support_step_count: int
    outside_support_step_count: int
    fallback_count: int
    solve_status_counts: dict[str, int]
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

    samples_by_horizon: dict[float, list[EmpiricalErrorSample]] = {
        ADAPTATION_HORIZON_STEPS * SAMPLE_DT_S: [],
        CONTROL_HORIZON_STEPS * SAMPLE_DT_S: [],
    }
    fleet_trajectories: list[Trajectory] = []
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
            fleet_trajectories.append(short_trajectory)
            errors = predict_windows(
                base,
                short_trajectory,
                horizon_steps=ADAPTATION_HORIZON_STEPS,
                stride=ADAPTATION_HORIZON_STEPS,
            ).endpoint_tangent_errors()
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
            errors = predict_windows(
                base,
                control_trajectory,
                horizon_steps=CONTROL_HORIZON_STEPS,
                stride=CONTROL_HORIZON_STEPS,
            ).endpoint_tangent_errors()
            samples_by_horizon[CONTROL_HORIZON_STEPS * SAMPLE_DT_S].append(
                EmpiricalErrorSample(
                    errors=errors,
                    source_group=label,
                    trajectory_id=f"{label}-control-profile-{profile_index}",
                )
            )
    forecast_error = replace(
        ForecastErrorEnvelope.from_samples(
            {horizon: tuple(samples) for horizon, samples in samples_by_horizon.items()}
        ),
        source="synthetic_arm_fleet_rollout_endpoints",
    )
    noise = innovation_noise(
        one_step_innovations(base, item) for item in fleet_trajectories
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

    # What the sibling configurations hand a new belief is one direction and
    # its spread: that subspace inverts to precision and everything else stays
    # at zero precision, which is to say unknown. The prechange mean is where
    # the belief starts, not something it claims to know; the telemetry is what
    # resolves the rest, and a direction the telemetry does not excite takes no
    # step at all.
    information = ParameterInformation.seeded_from_members(
        base,
        member_params,
        innovation_noise=noise,
        estimable=estimable_structured_parameters(base),
        source="prewarmed_vehicle_configuration_spread",
    )
    belief = DynamicsBelief(
        model=ExecutableModel(base, adaptation_telemetry.spec, runtime_spec),
        information=information,
        forecast_error=forecast_error,
        provenance={
            "benchmark_role": "prechange_vehicle_belief",
            "configuration_change": {
                "kind": "arm_length_ratio",
                "from": 1.0,
                "to": math.exp(TARGET_LOG_ARM_LENGTH_RATIO),
            },
        },
    )
    updated, update_result = belief.absorb(adaptation_telemetry)
    if not update_result.absorbed:
        raise RuntimeError(
            f"the recovery telemetry was refused: {update_result.reason}"
        )

    evaluation_telemetry = _configuration_trajectory(
        target_params,
        TARGET_LOG_ARM_LENGTH_RATIO,
        seed=22,
        duration_s=EVALUATION_DURATION_S,
        source_group="target-configuration-independent-evaluation",
    )
    before_errors = predict_windows(
        belief.params,
        evaluation_telemetry,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride=CONTROL_HORIZON_STEPS,
    ).endpoint_tangent_errors()
    after_errors = predict_windows(
        updated.params,
        evaluation_telemetry,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride=CONTROL_HORIZON_STEPS,
    ).endpoint_tangent_errors()
    tolerances = np.asarray(
        TrackingTolerances.for_platform("multirotor").local_state_scale
    )
    before_rms = float(np.sqrt(np.mean(np.square(before_errors / tolerances))))
    after_rms = float(np.sqrt(np.mean(np.square(after_errors / tolerances))))
    evidence = {
        "fleet": {
            "member_count": len(member_params),
            "parameter_count": len(information.names),
            "estimable_parameter_count": information.estimable_count,
            "seed_resolved_rank": information.resolved_rank(),
            "posterior_resolved_rank": updated.information.resolved_rank(),
            "prior_covariance_trace": float(np.trace(information.covariance())),
            "posterior_covariance_trace": float(
                np.trace(updated.information.covariance())
            ),
            "one_step_innovation_noise": noise.tolist(),
            "forecast_error_horizons_s": list(forecast_error.horizons_s),
            "forecast_error_group_count": list(forecast_error.independent_group_count),
            "forecast_error_raw_endpoint_count": list(forecast_error.raw_sample_count),
        },
        "adaptation": update_result.to_dict(),
        "independent_prediction": {
            "horizon_s": CONTROL_HORIZON_STEPS * SAMPLE_DT_S,
            "normalized_rms_before": before_rms,
            "normalized_rms_after": after_rms,
            "normalized_rms_ratio": after_rms / before_rms,
        },
    }
    return belief, updated, target_params, evidence


def _recovery_initial_state(belief: DynamicsBelief) -> np.ndarray:
    state = resting_state()
    state[0:3] = (0.25, -0.20, -0.15)
    quaternion = quaternion_from_euler(0.24, -0.17, 0.12)
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
    uncertainty: list[float] = []
    predicted_validity: list[float] = []
    fallback_count = 0
    solve_status_counts: dict[str, int] = {}
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
        uncertainty.append(
            result.diagnostics.maximum_normalized_model_uncertainty_standard_deviation
        )
        predicted_validity.append(result.diagnostics.maximum_validity_utilization)
        fallback_count += int(result.used_fallback)
        status = result.status.value
        solve_status_counts[status] = solve_status_counts.get(status, 0) + 1
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
    # The state each solve saw, which is every state but the last.
    solved_validity = np.max(actual_validity[:-1], axis=1)
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
        forecast_error_available=runtime_belief.forecast_error_available,
        parameter_information_rank=runtime_belief.information.resolved_rank(),
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
        maximum_predicted_validity_utilization=(
            max(predicted_validity) if np.isfinite(max(predicted_validity)) else None
        ),
        maximum_command_bound_violation=command_violation,
        maximum_normalized_model_uncertainty_standard_deviation=(
            max(uncertainty) if np.isfinite(max(uncertainty)) else None
        ),
        parameter_uncertainty_complete=controller.plan.uncertainty_complete,
        unresolved_parameters_allowed=controller.plan.policy.allow_unresolved_parameters,
        inside_support_step_count=int(np.count_nonzero(solved_validity <= 1.0 + 1e-6)),
        outside_support_step_count=int(np.count_nonzero(solved_validity > 1.0 + 1e-6)),
        fallback_count=fallback_count,
        solve_status_counts=solve_status_counts,
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


def _diagnostic_controller(model: DynamicsBelief | ExecutableModel) -> NMPCController:
    """The simulation comparison explicitly permits partial parameter evidence."""

    return NMPCController(
        model,
        policy=replace(default_solver_policy(model), allow_unresolved_parameters=True),
    )


def run_adaptive_recovery_benchmark() -> dict[str, Any]:
    """Run one fixed diagnostic with no acceptance thresholds or tuning surface."""

    belief, updated, target_params, evidence = _build_beliefs()
    initial_state = _recovery_initial_state(belief)
    # The point arms carry the belief's own runtime contract, so every arm
    # plans the same horizon: the maintained multirotor default is
    # CONTROL_HORIZON_STEPS at this sample period, and the two belief arms
    # reach the same number through their forecast envelope's own cap.
    equal_horizon_runtime = belief.runtime_spec
    actuation = DirectActuationMap(belief.input_spec.controls)
    controllers = (
        (
            "stale_belief",
            "prechange mean plus the fleet seed and its forecast envelope",
            _diagnostic_controller(belief),
        ),
        (
            "adapted_belief",
            "adapted mean plus the information the absorb accumulated",
            _diagnostic_controller(updated),
        ),
        (
            "adapted_mean_point",
            "adapted mean without uncertainty",
            _diagnostic_controller(
                ExecutableModel(
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
            _diagnostic_controller(
                ExecutableModel(
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
        "solver_failure_returns_explicit_bounded_hold": True,
        "independent_flight_watchdog_included": False,
        "predicted_spread_charged_in_the_tracking_cost": True,
        "robust_validity_charged_in_the_objective": True,
        "hard_prediction_horizon_validity_constraint_included": False,
        "flight_safety_claim": False,
        "throw_to_recover_claim": False,
        "posterior_calibration_claim": False,
        "configuration_delta_direction_derived_from_fleet": True,
        "unexcited_parameter_directions_completed_by_assumption": False,
        "adaptation_and_evaluation_telemetry_disjoint": True,
        # The update is one recursive absorb: every usable transition adds its
        # information, nothing proposes, nothing validates on a second split,
        # and numerical backtracking uses this block's actual residual.
        "update_is_a_recursive_information_absorb": True,
        "update_proposal_and_validation_split": False,
        "update_improvement_margin": False,
        "nonlinear_update_backtracking": True,
        "unresolved_parameter_planning_explicitly_allowed": True,
        "parameter_covariance_updated_by_the_update": True,
        "information_discounted_or_forgotten": False,
        "held_out_forecast_bias_applied_at_runtime": False,
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
        "format_version": 6,
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
            "update_absorbed": evidence["adaptation"]["absorbed"],
            "posterior_resolves_more_than_the_seed": (
                evidence["fleet"]["posterior_resolved_rank"]
                > evidence["fleet"]["seed_resolved_rank"]
            ),
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
            "all_full_nmpc_predictions_within_validity_support": all(
                item.maximum_predicted_validity_utilization is not None
                and item.maximum_predicted_validity_utilization <= 1.0
                for item in recovery
            ),
            "any_solve_started_outside_validity_support": any(
                item.outside_support_step_count > 0 for item in recovery
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
            "No independent fallback or airframe-specific recovery controller is included; a failed solve returns a bounded hold of the previous command.",
            "The belief starts from the prechange mean but claims to know only the one direction the sibling configurations moved; every other coefficient is resolved, or not, by the telemetry itself.",
            "The benchmark starts from a bounded in-envelope disturbance, not an unknown physical throw.",
            "The objective charges predicted spread in the tracking cost and a componentwise one-standard-deviation margin on the validity envelope; neither is a hard prediction-horizon constraint or a flight-safety guarantee.",
            "Charging spread requires one extra forward rollout per resolved parameter direction; this diagnostic does not establish a hard real-time deadline on other hardware or uncertainty representations.",
            "The seed's spread is a rank-one direction; what the belief learns about every other coefficient comes from 0.8 seconds of one telemetry block.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
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
