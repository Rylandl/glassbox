"""Supervisor fault campaign against the hidden Crazyflow plant.

Every supervisor reason is provoked deliberately, one at a time, so the fixed
expectation table below can assert that the fault is detected, that the plant
stays bounded, and that the mode the supervisor selects is the intended one.
"""

from __future__ import annotations

import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.control.nmpc import NMPCController
from glassbox.core.synthetic import resting_state
from glassbox.integrations.crazyflow import CrazyflowPlant
from glassbox.integrations.crazyflow_telemetry import (
    DEFAULT_ARM_LENGTH_RATIO,
    prewarm_controller,
)

_SUPERVISOR_FAULT_EXPECTATIONS: tuple[
    tuple[str, SupervisorMode, SupervisorReason | None], ...
] = (
    ("nominal", SupervisorMode.NOMINAL, None),
    (
        "time_regression",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.TIME_REGRESSION,
    ),
    (
        "state_timestamp_invalid",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.STATE_TIMESTAMP_INVALID,
    ),
    (
        "state_stale",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.STATE_STALE,
    ),
    (
        "state_from_future",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.STATE_FROM_FUTURE,
    ),
    (
        "state_invalid",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.STATE_INVALID,
    ),
    (
        "quaternion_invalid",
        SupervisorMode.COLLECTIVE_HOLD,
        SupervisorReason.QUATERNION_INVALID,
    ),
    (
        "command_timestamp_invalid",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.COMMAND_TIMESTAMP_INVALID,
    ),
    (
        "command_stale",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.COMMAND_STALE,
    ),
    (
        "command_from_future",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.COMMAND_FROM_FUTURE,
    ),
    (
        "command_invalid",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.COMMAND_INVALID,
    ),
    (
        "command_out_of_bounds",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.COMMAND_OUT_OF_BOUNDS,
    ),
    (
        "controller_deadline_exceeded",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.CONTROLLER_UNUSABLE,
    ),
    (
        "tilt_limit",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.TILT_LIMIT,
    ),
    (
        "angular_rate_limit",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.ANGULAR_RATE_LIMIT,
    ),
    (
        "arrest_latched",
        SupervisorMode.RATE_ARREST,
        SupervisorReason.ARREST_LATCHED,
    ),
)


def _simulate_supervisor_fault_campaign(
    controller: NMPCController,
) -> dict[str, Any]:
    """Exercise every supervisor reason through controller and hidden plant."""

    plant = CrazyflowPlant()
    plant.set_arm_length_ratio(DEFAULT_ARM_LENGTH_RATIO)
    hover = np.full(4, plant.hover_motor_thrust_fraction, dtype=np.float64)
    reference_state = resting_state()
    reference_state[2] = 1.0
    reference = controller.hold_reference(jnp.asarray(reference_state))
    supervisor_config = MultirotorSupervisorConfig(
        command_minimum=tuple(
            float(value) for value in controller.model.command_minimum
        ),
        command_maximum=tuple(
            float(value) for value in controller.model.command_maximum
        ),
        collective_hold_command=tuple(float(value) for value in hover),
        maximum_state_age_s=2.0 * plant.sample_period_s,
        maximum_command_age_s=plant.sample_period_s,
    )
    prewarm_started_at = time.perf_counter()
    _, warm = prewarm_controller(controller, reference, reference_state, hover)
    prewarm_wall_time_s = time.perf_counter() - prewarm_started_at
    if not warm.command_usable:
        plant.close()
        raise RuntimeError("supervisor fault campaign controller prewarm failed")

    minimum = np.asarray(supervisor_config.command_minimum)
    maximum = np.asarray(supervisor_config.command_maximum)
    cases: list[dict[str, Any]] = []
    decision_times: list[float] = []
    try:
        for name, expected_mode, expected_reason in _SUPERVISOR_FAULT_EXPECTATIONS:
            actual_state = reference_state.copy()
            if name == "tilt_limit":
                angle_rad = supervisor_config.maximum_tilt_rad + 0.10
                actual_state[6:10] = (
                    math.cos(0.5 * angle_rad),
                    math.sin(0.5 * angle_rad),
                    0.0,
                    0.0,
                )
            elif name == "angular_rate_limit":
                actual_state[10] = supervisor_config.maximum_angular_rate_rad_s[0] + 0.5

            sample = plant.reset(
                actual_state,
                applied_motor_thrust_fraction=hover,
            )
            observed_state = np.asarray(sample.state).copy()
            if name == "state_invalid":
                observed_state[3] = math.nan
            elif name == "quaternion_invalid":
                observed_state[6:10] = 0.0

            solve_deadline_s = 1e-9 if name == "controller_deadline_exceeded" else None
            result = controller.solve(
                jnp.asarray(observed_state),
                reference,
                jnp.asarray(hover),
                applied_command=jnp.asarray(sample.applied_motor_thrust_fraction),
                warm_start=warm.warm_start,
                deadline_s=solve_deadline_s,
            )
            jax.block_until_ready(result.command)

            candidate = np.asarray(result.command).copy()
            if name == "command_invalid":
                candidate[0] = math.nan
            elif name == "command_out_of_bounds":
                candidate[0] = maximum[0] + 0.1

            now_s = 100.0
            state_received_at_s = now_s
            command_generated_at_s = now_s
            if name == "state_timestamp_invalid":
                state_received_at_s = math.nan
            elif name == "state_stale":
                state_received_at_s -= 2.0 * supervisor_config.maximum_state_age_s
            elif name == "state_from_future":
                state_received_at_s += plant.sample_period_s
            elif name == "command_timestamp_invalid":
                command_generated_at_s = math.nan
            elif name == "command_stale":
                command_generated_at_s -= 2.0 * supervisor_config.maximum_command_age_s
            elif name == "command_from_future":
                command_generated_at_s += plant.sample_period_s

            supervisor = MultirotorFlightSupervisor(supervisor_config)
            prime_passed = True
            if name == "time_regression":
                primed = supervisor.supervise(
                    state=observed_state,
                    state_received_at_s=now_s + 1.0,
                    candidate_command=result.command,
                    command_generated_at_s=now_s + 1.0,
                    now_s=now_s + 1.0,
                    controller_command_usable=result.command_usable,
                    previous_applied_command=sample.applied_motor_thrust_fraction,
                )
                prime_passed = primed.mode == SupervisorMode.NOMINAL
            elif name == "arrest_latched":
                primed = supervisor.supervise(
                    state=observed_state,
                    state_received_at_s=now_s - 0.01,
                    candidate_command=result.command,
                    command_generated_at_s=now_s - 0.01,
                    now_s=now_s - 0.01,
                    controller_command_usable=False,
                    previous_applied_command=sample.applied_motor_thrust_fraction,
                )
                prime_passed = primed.mode == SupervisorMode.RATE_ARREST

            decision_started_at = time.perf_counter()
            decision = supervisor.supervise(
                state=observed_state,
                state_received_at_s=state_received_at_s,
                candidate_command=candidate,
                command_generated_at_s=command_generated_at_s,
                now_s=now_s,
                controller_command_usable=result.command_usable,
                previous_applied_command=sample.applied_motor_thrust_fraction,
            )
            decision_wall_time_s = time.perf_counter() - decision_started_at
            decision_times.append(decision_wall_time_s)
            post_step = plant.step(decision.command)
            command = np.asarray(decision.command)
            command_bound_violation = max(
                float(np.max(minimum - command)),
                float(np.max(command - maximum)),
                0.0,
            )
            reason_matched = (
                decision.reasons == ()
                if expected_reason is None
                else expected_reason in decision.reasons
            )
            nominal_transparent = (
                bool(np.array_equal(command, candidate))
                if expected_mode == SupervisorMode.NOMINAL
                else None
            )
            nominal_contract_passed = (
                expected_mode != SupervisorMode.NOMINAL or nominal_transparent is True
            )
            output_finite = bool(np.all(np.isfinite(command)))
            plant_state_finite = bool(np.all(np.isfinite(post_step.state)))
            passed = bool(
                prime_passed
                and decision.mode == expected_mode
                and reason_matched
                and nominal_contract_passed
                and output_finite
                and command_bound_violation <= 1e-12
                and plant_state_finite
            )
            cases.append(
                {
                    "name": name,
                    "expected_mode": expected_mode.value,
                    "expected_reason": (
                        None if expected_reason is None else expected_reason.value
                    ),
                    "actual_mode": decision.mode.value,
                    "actual_reasons": [reason.value for reason in decision.reasons],
                    "controller_status": result.status.value,
                    "controller_command_usable": result.command_usable,
                    "controller_used_fallback": result.used_fallback,
                    "observed_state_finite": bool(np.all(np.isfinite(observed_state))),
                    "true_plant_state_finite": bool(np.all(np.isfinite(sample.state))),
                    "supervised_command": command.tolist(),
                    "supervised_command_finite": output_finite,
                    "maximum_command_bound_violation": command_bound_violation,
                    "plant_state_finite_after_step": plant_state_finite,
                    "nominal_command_transparent": nominal_transparent,
                    "prime_passed": prime_passed,
                    "decision_wall_time_s": decision_wall_time_s,
                    "passed": passed,
                }
            )
    finally:
        plant.close()

    expected_reasons = {
        reason.value
        for _, _, reason in _SUPERVISOR_FAULT_EXPECTATIONS
        if reason is not None
    }
    return {
        "fixed": True,
        "controller_model_role": (
            "transactionally adapted mean plus parameter uncertainty"
        ),
        "campaign_telemetry_used_for_model_fitting_or_belief_update": False,
        "controller_to_supervisor_to_hidden_plant": True,
        "case_count": len(cases),
        "fault_count": len(cases) - 1,
        "configuration": supervisor_config.to_dict(),
        "prewarm_wall_time_s": prewarm_wall_time_s,
        "maximum_supervisor_decision_wall_time_s": max(decision_times, default=0.0),
        "all_typed_reasons_covered": expected_reasons
        == {reason.value for reason in SupervisorReason},
        "nominal_case_transparent": bool(
            cases and cases[0]["nominal_command_transparent"] is True
        ),
        "all_cases_passed": all(case["passed"] for case in cases),
        "all_supervised_commands_finite": all(
            case["supervised_command_finite"] for case in cases
        ),
        "all_supervised_commands_bounded": all(
            case["maximum_command_bound_violation"] <= 1e-12 for case in cases
        ),
        "all_true_plant_steps_finite": all(
            case["plant_state_finite_after_step"] for case in cases
        ),
        "cases": cases,
    }
