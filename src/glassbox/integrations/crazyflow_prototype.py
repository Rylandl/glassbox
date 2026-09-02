"""Fixed Crazyflow hidden-plant prototype for adjustable-arm recovery work.

This module is the command-line entry point and the orchestration that ties the
pieces together. The pieces themselves live next to it: telemetry generation
and closed-loop recording in ``crazyflow_telemetry``, fleet identification and
the configuration-change prior in ``crazyflow_fleet``, online adaptation and
its worker process in ``crazyflow_online``, and the supervisor fault campaign
in ``crazyflow_supervisor_campaign``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.nmpc import NMPCController
from glassbox.core.data import Trajectory, save_trajectory_npz
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.runtime import DirectActuationMap, RuntimeDynamicsModel
from glassbox.core.synthetic import resting_state
from glassbox.integrations.crazyflow import CrazyflowPlant
from glassbox.integrations.crazyflow_fleet import (
    ADAPTATION_HORIZON_STEPS,
    FLEET_LOG_ARM_LENGTH_RATIOS,
    FLEET_PROFILE_COUNT,
    _build_crazyflow_beliefs,
)
from glassbox.integrations.crazyflow_online import (
    ONLINE_EVIDENCE_DURATION_S,
    ONLINE_INTEGRATED_PREWARM_STEPS,
    ONLINE_MAXIMUM_DURATION_S,
    ONLINE_POST_INSTALL_DURATION_S,
    _recovery_initial_state,
    _simulate_online_adaptation_recovery,
)
from glassbox.integrations.crazyflow_supervisor_campaign import (
    _simulate_supervisor_fault_campaign,
)
from glassbox.integrations.crazyflow_telemetry import (
    CONTROL_HORIZON_STEPS,
    DEFAULT_ARM_LENGTH_RATIO,
    DEFAULT_DURATION_S,
    PROTOTYPE_SCHEMA_VERSION,
    _crazyflow_solver_policy,
    _trajectory_statistics,
    generate_crazyflow_trajectory,
    prewarm_controller,
)

DEFAULT_BASELINE_SEED = 101
DEFAULT_MODIFIED_SEED = 102
RECOVERY_DURATION_S = 0.8
RECOVERY_TAIL_DURATION_S = 0.2

__all__ = [
    "ADAPTATION_HORIZON_STEPS",
    "CONTROL_HORIZON_STEPS",
    "DEFAULT_ARM_LENGTH_RATIO",
    "DEFAULT_BASELINE_SEED",
    "DEFAULT_DURATION_S",
    "DEFAULT_MODIFIED_SEED",
    "FLEET_LOG_ARM_LENGTH_RATIOS",
    "FLEET_PROFILE_COUNT",
    "ONLINE_EVIDENCE_DURATION_S",
    "ONLINE_INTEGRATED_PREWARM_STEPS",
    "ONLINE_MAXIMUM_DURATION_S",
    "ONLINE_POST_INSTALL_DURATION_S",
    "PROTOTYPE_SCHEMA_VERSION",
    "RECOVERY_DURATION_S",
    "RECOVERY_TAIL_DURATION_S",
    "generate_crazyflow_trajectory",
    "main",
    "run_crazyflow_adaptive_recovery_prototype",
    "run_crazyflow_prototype",
]


def _simulate_recovery(
    condition: str,
    model_role: str,
    controller: NMPCController,
    initial_state: np.ndarray,
) -> dict[str, Any]:
    plant = CrazyflowPlant()
    plant.set_arm_length_ratio(DEFAULT_ARM_LENGTH_RATIO)
    hover = np.full(4, plant.hover_motor_thrust_fraction, dtype=np.float64)
    reference_state = resting_state()
    reference_state[2] = 1.0
    reference = controller.hold_reference(jnp.asarray(reference_state))
    started_at = time.perf_counter()
    _, warm = prewarm_controller(controller, reference, initial_state, hover)
    prewarm_s = time.perf_counter() - started_at

    sample = plant.reset(
        initial_state,
        applied_motor_thrust_fraction=hover,
    )
    interval_count = round(RECOVERY_DURATION_S / plant.sample_period_s)
    states = np.empty((interval_count + 1, 13), dtype=np.float64)
    commands = np.empty((interval_count, 4), dtype=np.float64)
    states[0] = sample.state
    previous = hover
    warm_start = warm.warm_start
    solve_times: list[float] = []
    support_modes: list[str] = []
    solve_statuses: list[str] = []
    solve_messages: list[str] = []
    support_applied = 0
    fallback_count = 0
    predicted_validity: list[float] = []
    support_validity: list[float] = []
    nonfinite_diagnostic_count = 0
    try:
        for index in range(interval_count):
            result = controller.solve(
                jnp.asarray(states[index]),
                reference,
                jnp.asarray(previous),
                applied_command=jnp.asarray(sample.applied_motor_thrust_fraction),
                warm_start=warm_start,
            )
            command = np.asarray(result.command)
            commands[index] = command
            sample = plant.step(command)
            states[index + 1] = sample.state
            previous = command
            warm_start = result.warm_start
            solve_times.append(result.diagnostics.solve_time_s)
            solve_statuses.append(result.status.value)
            solve_messages.append(result.message)
            support_modes.append(result.diagnostics.support_filter_mode.value)
            support_applied += int(result.diagnostics.support_filter_applied)
            fallback_count += int(result.used_fallback)
            diagnostic_values = (
                result.diagnostics.maximum_validity_utilization,
                result.diagnostics.support_horizon_maximum_robust_validity_utilization,
            )
            nonfinite_diagnostic_count += sum(
                not np.isfinite(value) for value in diagnostic_values
            )
            if np.isfinite(diagnostic_values[0]):
                predicted_validity.append(float(diagnostic_values[0]))
            if np.isfinite(diagnostic_values[1]):
                support_validity.append(float(diagnostic_values[1]))
    finally:
        plant.close()

    references = np.repeat(reference_state[None, :], len(states), axis=0)
    errors = np.asarray(
        jax.vmap(rigid_body_local_error)(
            jnp.asarray(references),
            jnp.asarray(states),
        )
    )
    scale = np.asarray(controller.tolerances.local_state_scale)
    normalized = errors / scale
    tail_steps = round(RECOVERY_TAIL_DURATION_S / plant.sample_period_s)
    actual_validity = np.asarray(
        jax.vmap(controller.model.validity_utilization)(jnp.asarray(states))
    )
    minimum = np.asarray(controller.model.command_minimum)
    maximum = np.asarray(controller.model.command_maximum)
    command_violation = max(
        float(np.max(minimum - commands)),
        float(np.max(commands - maximum)),
        0.0,
    )
    return {
        "condition": condition,
        "model_role": model_role,
        "normalized_tracking_rms": float(np.sqrt(np.mean(np.square(normalized)))),
        "tail_normalized_tracking_rms": float(
            np.sqrt(np.mean(np.square(normalized[-tail_steps:])))
        ),
        "tail_normalized_attitude_rate_rms": float(
            np.sqrt(np.mean(np.square(normalized[-tail_steps:, 6:12])))
        ),
        "terminal_attitude_error_rad": float(np.linalg.norm(errors[-1, 6:9])),
        "terminal_angular_velocity_error_rad_s": float(
            np.linalg.norm(errors[-1, 9:12])
        ),
        "maximum_actual_validity_utilization": float(np.max(actual_validity)),
        "maximum_predicted_validity_utilization": (
            max(predicted_validity) if predicted_validity else None
        ),
        "maximum_support_horizon_robust_validity_utilization": (
            max(support_validity) if support_validity else None
        ),
        "maximum_command_bound_violation": command_violation,
        "support_filter_mode_counts": dict(sorted(Counter(support_modes).items())),
        "solve_status_counts": dict(sorted(Counter(solve_statuses).items())),
        "solve_message_counts": dict(sorted(Counter(solve_messages).items())),
        "support_filter_applied_count": support_applied,
        "fallback_count": fallback_count,
        "nonfinite_diagnostic_count": nonfinite_diagnostic_count,
        "finite": bool(
            np.all(np.isfinite(states))
            and np.all(np.isfinite(commands))
            and np.all(np.isfinite(normalized))
        ),
        "prewarm_wall_time_s": prewarm_s,
        "solve_time_median_s": float(np.median(solve_times)),
        "solve_time_p90_s": float(np.quantile(solve_times, 0.9)),
        "solve_time_maximum_s": max(solve_times),
    }


def run_crazyflow_adaptive_recovery_prototype() -> dict[str, Any]:
    """Fit, adapt, and recover against an independently implemented hidden plant."""

    belief, updated, oracle_runtime, evidence = _build_crazyflow_beliefs()
    initial_state = _recovery_initial_state(belief)
    point_runtime_spec = replace(
        belief.runtime_spec,
        certified_prediction_horizon_s=(
            CONTROL_HORIZON_STEPS * belief.runtime_spec.sample_period_s
        ),
        certification_source="Crazyflow prototype equal-horizon comparison",
    )
    actuation = DirectActuationMap(belief.input_spec.controls)
    controllers = (
        (
            "stale_belief",
            "prechange telemetry fit plus current fleet forecast error",
            NMPCController(belief, _policy=_crazyflow_solver_policy()),
        ),
        (
            "adapted_belief",
            "transactionally adapted mean plus parameter uncertainty",
            NMPCController(updated, _policy=_crazyflow_solver_policy()),
        ),
        (
            "adapted_mean_point",
            "adapted telemetry mean without uncertainty",
            NMPCController(
                RuntimeDynamicsModel(
                    updated.params,
                    belief.input_spec,
                    point_runtime_spec,
                    actuation,
                ),
                _policy=_crazyflow_solver_policy(),
            ),
        ),
        (
            "oracle_telemetry_fit",
            "independent full target-configuration telemetry fit",
            NMPCController(
                oracle_runtime,
                _policy=_crazyflow_solver_policy(),
            ),
        ),
    )
    recovery = [
        _simulate_recovery(condition, role, controller, initial_state)
        for condition, role, controller in controllers
    ]
    online_recovery = _simulate_online_adaptation_recovery(belief, initial_state)
    supervisor_fault_campaign = _simulate_supervisor_fault_campaign(controllers[1][2])
    by_condition = {item["condition"]: item for item in recovery}
    stale = by_condition["stale_belief"]
    adapted = by_condition["adapted_belief"]
    oracle = by_condition["oracle_telemetry_fit"]
    comparisons = {
        "adapted_vs_stale_tail_tracking_rms_ratio": (
            adapted["tail_normalized_tracking_rms"]
            / stale["tail_normalized_tracking_rms"]
        ),
        "adapted_vs_stale_tail_attitude_rate_rms_ratio": (
            adapted["tail_normalized_attitude_rate_rms"]
            / stale["tail_normalized_attitude_rate_rms"]
        ),
        "adapted_vs_oracle_tail_tracking_rms_ratio": (
            adapted["tail_normalized_tracking_rms"]
            / oracle["tail_normalized_tracking_rms"]
        ),
    }
    return {
        "schema_version": PROTOTYPE_SCHEMA_VERSION,
        "artifact_type": "glassbox_crazyflow_adaptive_recovery_prototype",
        "semantics": {
            "diagnostic_only": True,
            "acceptance_gate": False,
            "flight_safety_claim": False,
            "throw_to_recover_claim": False,
            "hidden_physical_parameters_supplied_to_glassbox": False,
            "prechange_model_fitted_from_telemetry": True,
            "fleet_members_fitted_independently_from_telemetry": True,
            "adaptation_and_evaluation_telemetry_disjoint": True,
            "online_update_uses_recovery_prefix": True,
            "online_candidate_prepared_off_control_path": True,
            "background_compute_isolated": True,
            "background_compute_uses_spawned_process": True,
            "background_process_payload_is_numerical": True,
            "online_controller_install_transactional": True,
            "online_post_install_tail_is_independent_validation": False,
            "transactional_belief_update": True,
            "rotor_level_closed_loop_glassbox_control": True,
            "support_filter_not_independent_fallback_controller": True,
            "independent_flight_watchdog_included": True,
            "supervisor_uses_fitted_dynamics_model": False,
            "supervisor_tracks_position_or_flight_trajectory": False,
            "integrated_supervisor_fault_campaign_included": True,
            "fault_campaign_telemetry_used_for_model_fitting_or_belief_update": False,
        },
        "configuration": {
            "crazyflow_version": "0.3.2",
            "drone": "cf21B_500",
            "simulation_frequency_hz": 500,
            "control_frequency_hz": 50,
            "arm_length_ratio_before": 1.0,
            "arm_length_ratio_after": DEFAULT_ARM_LENGTH_RATIO,
            "fleet_arm_length_ratios": [
                math.exp(value) for value in FLEET_LOG_ARM_LENGTH_RATIOS
            ],
            "adaptation_duration_s": 0.8,
            "independent_prediction_horizon_s": (
                CONTROL_HORIZON_STEPS * belief.runtime_spec.sample_period_s
            ),
            "recovery_duration_s": RECOVERY_DURATION_S,
            "online_evidence_duration_s": ONLINE_EVIDENCE_DURATION_S,
            "online_post_install_duration_s": ONLINE_POST_INSTALL_DURATION_S,
            "online_maximum_duration_s": ONLINE_MAXIMUM_DURATION_S,
            "recovery_initial_state": initial_state.tolist(),
            "solver_policy": {
                "horizon_steps": CONTROL_HORIZON_STEPS,
                "block_count": 10,
                "maximum_iterations": 6,
                "line_search_steps": 16,
                "reason": (
                    "fixed deeper backtracking for Crazyflie-scale angular "
                    "command sensitivity with a six-iteration ceiling for "
                    "50 Hz supervised-command timing headroom"
                ),
            },
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
        },
        "evidence": evidence,
        "recovery": recovery,
        "online_recovery": online_recovery,
        "supervisor_fault_campaign": supervisor_fault_campaign,
        "comparisons": comparisons,
        "observations": {
            "update_applied": evidence["adaptation"]["applied"],
            "independent_prediction_improved": (
                evidence["independent_prediction"]["normalized_rms_after"]
                < evidence["independent_prediction"]["normalized_rms_before"]
            ),
            "all_recovery_traces_finite": all(item["finite"] for item in recovery),
            "all_commands_within_bounds": all(
                item["maximum_command_bound_violation"] <= 1e-6 for item in recovery
            ),
            "all_recovery_traces_without_fallback": all(
                item["fallback_count"] == 0 for item in recovery
            ),
            "adapted_tail_tracking_better_than_stale": (
                comparisons["adapted_vs_stale_tail_tracking_rms_ratio"] < 1.0
            ),
            "adapted_tail_attitude_rate_better_than_stale": (
                comparisons["adapted_vs_stale_tail_attitude_rate_rms_ratio"] < 1.0
            ),
            "online_update_applied": online_recovery["update"]["applied"],
            "online_controller_installed": online_recovery["handoff"][
                "atomic_install_at_control_boundary"
            ],
            "online_control_continued_during_candidate_preparation": (
                online_recovery["handoff"][
                    "control_continued_during_candidate_preparation"
                ]
            ),
            "online_control_deadlines_all_met": (
                online_recovery["timing"]["outer_deadline_miss_count"] == 0
            ),
            "online_absolute_schedule_deadlines_all_met": (
                online_recovery["timing"]["absolute_deadline_miss_count"] == 0
            ),
            "online_supervisor_fault_handled": online_recovery["supervisor"][
                "fault_injection"
            ]["handled"],
            "supervisor_fault_campaign_passed": supervisor_fault_campaign[
                "all_cases_passed"
            ],
            "online_gate_passed": bool(
                online_recovery["handoff"]["atomic_install_at_control_boundary"]
                and online_recovery["timing"]["outer_deadline_miss_count"] == 0
                and online_recovery["fallback_count"] == 0
                and online_recovery["supervisor"]["fault_injection"]["handled"]
                and online_recovery["finite"]
                and online_recovery["maximum_command_bound_violation"] <= 1e-6
            ),
            "supervised_prototype_gate_passed": bool(
                online_recovery["handoff"]["atomic_install_at_control_boundary"]
                and online_recovery["timing"]["outer_deadline_miss_count"] == 0
                and online_recovery["fallback_count"] == 0
                and online_recovery["supervisor"]["fault_injection"]["handled"]
                and online_recovery["finite"]
                and online_recovery["maximum_command_bound_violation"] <= 1e-6
                and supervisor_fault_campaign["all_cases_passed"]
                and supervisor_fault_campaign["all_typed_reasons_covered"]
                and supervisor_fault_campaign["nominal_case_transparent"]
                and supervisor_fault_campaign["all_supervised_commands_finite"]
                and supervisor_fault_campaign["all_supervised_commands_bounded"]
                and supervisor_fault_campaign["all_true_plant_steps_finite"]
            ),
        },
        "limitations": [
            "The prechange, fleet, target, and oracle models are all simulator telemetry fits; Crazyflow physical parameters are never passed to Glassbox.",
            "The recovery begins from a bounded disturbance rather than a ballistic throw.",
            "The online update reuses the recovery prefix, so its post-install tail is a closed-loop demonstration rather than independent validation evidence.",
            "The isolated adaptation process is prewarmed and thread-limited, but desktop scheduling is not a hard real-time guarantee.",
            "The independent supervisor is a freshness, bounds, attitude, and rate-arrest boundary; it is not a flight-safety guarantee or position controller.",
            "The oracle is a longer target-configuration telemetry fit, not access to hidden simulator parameters.",
            "The arm change uses a point-mass inertia scaling approximation.",
            "No sensor noise, delay, packet loss, ground contact, or firmware scheduling is included yet.",
        ],
    }


def run_crazyflow_prototype(
    *,
    duration_s: float = DEFAULT_DURATION_S,
    arm_length_ratio: float = DEFAULT_ARM_LENGTH_RATIO,
) -> tuple[dict[str, Any], Trajectory, Trajectory]:
    """Run the fixed first vertical slice without making a recovery claim."""

    baseline = generate_crazyflow_trajectory(
        seed=DEFAULT_BASELINE_SEED,
        duration_s=duration_s,
        arm_length_ratio=1.0,
        source_group="crazyflow-prechange-vehicle",
    )
    modified = generate_crazyflow_trajectory(
        seed=DEFAULT_MODIFIED_SEED,
        duration_s=duration_s,
        arm_length_ratio=arm_length_ratio,
        source_group="crazyflow-postchange-vehicle",
    )
    report = {
        "schema_version": PROTOTYPE_SCHEMA_VERSION,
        "artifact_type": "glassbox_crazyflow_hidden_plant_prototype",
        "semantics": {
            "diagnostic_only": True,
            "acceptance_gate": False,
            "flight_safety_claim": False,
            "throw_to_recover_claim": False,
            "closed_loop_glassbox_control_included": False,
            "online_belief_update_included": False,
            "hidden_physical_parameters_supplied_to_glassbox": False,
            "rotor_level_plant_control": True,
            "applied_rotor_state_recorded_as_typed_observation": True,
        },
        "configuration": {
            "baseline_seed": DEFAULT_BASELINE_SEED,
            "modified_seed": DEFAULT_MODIFIED_SEED,
            "duration_s": duration_s,
            "arm_length_ratio_before": 1.0,
            "arm_length_ratio_after": arm_length_ratio,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "crazyflow": baseline.provenance["adapter"]["crazyflow_version"],
        },
        "trajectories": {
            "baseline": _trajectory_statistics(baseline),
            "modified": _trajectory_statistics(modified),
        },
        "observations": {
            "all_telemetry_finite": bool(
                _trajectory_statistics(baseline)["finite"]
                and _trajectory_statistics(modified)["finite"]
            ),
            "all_commands_bounded": bool(
                np.min(baseline.controls) >= 0.0
                and np.max(baseline.controls) <= 1.0
                and np.min(modified.controls) >= 0.0
                and np.max(modified.controls) <= 1.0
            ),
        },
        "next_gate": (
            "fit the prechange effective model from baseline telemetry, derive "
            "configuration-change covariance from separately fitted fleet members, "
            "and run stale/adapted/oracle recovery against this hidden plant"
        ),
        "limitations": [
            "This slice validates simulator, frame, motor-order, actuation, and telemetry contracts only.",
            "The stabilizing data-generation controller is not Glassbox NMPC and is not part of identification telemetry.",
            "The arm change scales the complete inertia tensor with arm length squared as a point-mass approximation.",
            "Crazyflow remains a simulator and cannot validate firmware, radio, estimator, or hardware timing.",
        ],
    }
    return report, baseline, modified


def main(argv: Sequence[str] | None = None) -> None:
    # Set before any lazy `import crazyflow` reaches its own SciPy-array-API guard.
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--plant-contract-only",
        action="store_true",
        help="write only the fast frame/actuation/telemetry qualification slice",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plant_contract_only:
        report, baseline, modified = run_crazyflow_prototype()
        save_trajectory_npz(baseline, args.output_dir / "baseline.npz")
        save_trajectory_npz(modified, args.output_dir / "modified.npz")
    else:
        report = run_crazyflow_adaptive_recovery_prototype()
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
