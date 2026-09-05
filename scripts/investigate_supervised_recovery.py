"""Trace model support and qualify supervised synthetic recovery.

This offline diagnostic drives the maintained control loop and supervisor.
Its discrete clock measures telemetry age and latch duration; solver CPU time
is reported separately. Deadline faults deliberately pass an expired solver
budget, rather than relying on host load. No command reaches physical hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from investigate_recovery import ADDITIONAL_DURATION_S, ADDITIONAL_SEEDS, initial_state

from glassbox.belief.belief import DynamicsBelief
from glassbox.control.fitted import NMPCController
from glassbox.control.supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
)
from glassbox.core.data import Trajectory
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    hover_control,
    quaternion_to_rotation,
    rollout_with_latent,
    step_with_latent,
)
from glassbox.core.geometry import quaternion_from_euler, rigid_body_local_error
from glassbox.core.metrics import predict_windows
from glassbox.core.model import ModelValidityEnvelope
from glassbox.core.synthetic import resting_state
from glassbox.integrations.loop import Observation, run_control_loop
from glassbox.workflows.benchmarks import recovery

DT = recovery.SAMPLE_DT_S
FEATURES = (
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
)
CALIBRATION_PHASES = (0.0, np.pi / 2)
EXPLORATORY_PHASES = (np.pi / 4, 3 * np.pi / 4)
VALIDATION_PHASES = (np.pi / 6, 5 * np.pi / 6)
OPERATING_ROLL_AMPLITUDE_RAD = 0.25
MAXIMUM_VALIDATION_COMPONENT_RMS = (
    0.10  # Fraction of the maintained tracking tolerance.
)
INTERVALS = 120  # 2.4 seconds permits a recovery tail after the injected faults.


def model_features(states):
    rotations = np.asarray(
        jax.vmap(quaternion_to_rotation)(jnp.asarray(states[:, 6:10]))
    )
    velocity = np.einsum("nji,nj->ni", rotations, states[:, 3:6])
    return np.concatenate((velocity, states[:, 10:13]), axis=1)


def coverage_flight(belief, target, phase, amplitude=0.35):
    """A bounded roll excitation; distinct phases reserve validation flights."""
    states = [resting_state()]
    controls = []
    latent = hover_control(target)
    hover = float(latent[0])
    omega = 2 * np.pi * 0.8

    @jax.jit
    def advance(state, applied, command):
        return step_with_latent(
            target, state, applied, command, DT, belief.input_spec.control_roles
        )

    for tick in range(300):
        time_s = tick * DT
        state = states[-1]
        ramp = min(time_s / 0.5, 1.0)
        desired = resting_state()
        desired[6:10] = quaternion_from_euler(
            ramp * amplitude * np.sin(omega * time_s + phase), 0.0, 0.0
        )
        attitude_error = np.asarray(
            rigid_body_local_error(jnp.asarray(state), jnp.asarray(desired))
        )[6:9]
        desired_rates = np.array(
            (ramp * amplitude * omega * np.cos(omega * time_s + phase), 0.0, 0.0)
        )
        differential = np.clip(
            0.8 * attitude_error + 0.22 * (desired_rates - state[10:13]), -0.8, 0.8
        )
        rotation = np.asarray(quaternion_to_rotation(jnp.asarray(state[6:10])))
        collective = (
            hover / max(rotation[2, 2], 0.5) - 0.035 * state[2] - 0.025 * state[5]
        )
        command = np.clip(
            collective + 0.25 * np.asarray(MOTOR_MIXER).T @ differential, 0.0, 1.0
        )
        next_state, latent = advance(jnp.asarray(state), latent, jnp.asarray(command))
        states.append(np.asarray(next_state))
        controls.append(command)
    return Trajectory(
        time_s=np.arange(301) * DT,
        states=np.asarray(states),
        controls=np.asarray(controls),
        spec=belief.input_spec,
        labels={"source_group": f"roll-coverage-phase-{phase:.12g}"},
    )


def support_from_calibration(belief, flights):
    """Extend only to observed feature extrema; retain the prior declaration.

    This forms a marginal feature box, not a certificate for every joint
    combination in that box. Held-out flights separately test prediction error.
    The recovery trajectory never contributes to these bounds.
    """
    old = belief.support
    center = np.asarray(
        (*old.body_velocity_center_m_s, *old.angular_velocity_center_rad_s)
    )
    width = np.asarray(
        (*old.body_velocity_half_width_m_s, *old.angular_velocity_half_width_rad_s)
    )
    values = np.concatenate([model_features(flight.states) for flight in flights])
    lower = np.minimum(center - width, np.min(values, axis=0))
    upper = np.maximum(center + width, np.max(values, axis=0))
    center, width = (lower + upper) / 2, (upper - lower) / 2
    envelope = ModelValidityEnvelope(
        tuple(center[:3]), tuple(width[:3]), tuple(center[3:]), tuple(width[3:])
    )
    return replace(
        belief,
        model=replace(
            belief.model,
            runtime_spec=replace(belief.runtime_spec, validity_envelope=envelope),
        ),
    )


def validate_coverage(belief, original, target, *, phases, amplitude):
    scale = np.asarray(NMPCController(belief).tolerances.local_state_scale)
    reports = []
    for phase in phases:
        trajectory = coverage_flight(belief, target, phase, amplitude)
        prediction = predict_windows(
            belief.params, trajectory, horizon_steps=30, stride=30
        )
        errors = prediction.endpoint_tangent_errors() / scale
        component_rms = np.sqrt(np.mean(errors**2, axis=0))
        utilization = np.asarray(
            jax.vmap(belief.model.validity_utilization)(jnp.asarray(trajectory.states))
        )
        old_utilization = np.asarray(
            jax.vmap(original.model.validity_utilization)(
                jnp.asarray(trajectory.states)
            )
        )
        reports.append(
            {
                "phase_rad": phase,
                "roll_amplitude_rad": amplitude,
                "window_count": len(errors),
                "normalized_endpoint_component_rms": component_rms.tolist(),
                "maximum_validity_utilization": float(np.max(utilization)),
                "maximum_original_validity_utilization": float(np.max(old_utilization)),
                "prediction_error_passed": bool(
                    np.all(np.isfinite(errors))
                    and np.all(component_rms <= MAXIMUM_VALIDATION_COMPONENT_RMS)
                ),
                "all_observations_inside_support": bool(
                    np.all(utilization <= 1.0 + 1e-6)
                ),
            }
        )
    return reports


class SimulatedLink:
    writable = True
    command_size = 4
    command_bounds = (np.zeros(4), np.ones(4))

    def __init__(self, belief, target, state, fault):
        self.belief = belief
        self.target = target
        self.states = [np.asarray(state)]
        self.latents = [np.asarray(hover_control(target))]
        self.commands = []
        self.fault = fault
        self.tick = 0

    def clock(self):
        return 100.0 + self.tick * DT

    def active_fault(self):
        if self.fault == "combined":
            return {
                **dict.fromkeys(range(10, 13), "stale_state"),
                **dict.fromkeys(range(30, 33), "deadline"),
                **dict.fromkeys(range(50, 53), "unresolved_parameters"),
            }.get(self.tick)
        return self.fault if 10 <= self.tick < 13 else None

    def read(self, *, timeout_s):
        source_tick = (
            self.tick - 3 if self.active_fault() == "stale_state" else self.tick
        )
        return Observation(
            state=self.states[source_tick],
            applied_command=self.latents[source_tick],
            received_at_s=100.0 + source_tick * DT,
            source_time_s=source_tick * DT,
            receive_age_s=(self.tick - source_tick) * DT,
        )

    def write(self, command):
        state, latent = step_with_latent(
            self.target,
            jnp.asarray(self.states[-1]),
            jnp.asarray(self.latents[-1]),
            jnp.asarray(command),
            DT,
            self.belief.input_spec.control_roles,
        )
        self.states.append(np.asarray(state))
        self.latents.append(np.asarray(latent))
        self.commands.append(np.asarray(command))
        self.tick += 1


class OfflineController:
    sample_period_s = DT

    def __init__(self, belief, link):
        self.nominal = NMPCController(belief)
        self.unresolved = NMPCController(DynamicsBelief(belief.model))
        self.link = link

    def solve(self, *args, **kwargs):
        fault = self.link.active_fault()
        kwargs["deadline_s"] = 1e-12 if fault == "deadline" else None
        controller = (
            self.unresolved if fault == "unresolved_parameters" else self.nominal
        )
        return controller.solve(*args, **kwargs)


def feature_summary(utilization):
    rows = {}
    for index, feature in enumerate(FEATURES):
        outside = np.flatnonzero(utilization[:, index] > 1.0 + 1e-6)
        rows[feature] = {
            "maximum_utilization": float(np.max(utilization[:, index])),
            "first_exit_s": None if not len(outside) else float(outside[0] * DT),
            "outside_sample_count": len(outside),
        }
    return rows


def simulate(belief, target, state, *, supervised, fault=None):
    link = SimulatedLink(belief, target, state, fault)
    controller = OfflineController(belief, link)
    supervisor = (
        MultirotorFlightSupervisor(
            MultirotorSupervisorConfig(
                collective_hold_command=np.asarray(hover_control(target))
            ),
            allocate=lambda differential: (
                0.25 * np.asarray(MOTOR_MIXER).T @ differential
            ),
        )
        if supervised
        else None
    )
    samples = []
    summary = run_control_loop(
        link,
        controller,
        supervisor,
        steps=INTERVALS,
        reference=controller.nominal.hold_reference(jnp.asarray(resting_state())),
        on_sample=samples.append,
        clock=link.clock,
    )
    states = np.asarray(link.states)
    utilization = np.asarray(
        jax.vmap(belief.model.validity_utilization)(jnp.asarray(states))
    )
    scales = np.asarray(controller.nominal.tolerances.local_state_scale)
    errors = (
        np.asarray(
            jax.vmap(
                lambda state: rigid_body_local_error(
                    jnp.asarray(resting_state()), state
                )
            )(jnp.asarray(states))
        )
        / scales
    )
    prediction_errors = {"inside": [], "outside": []}
    forecast_crossings = []

    @jax.jit
    def planned_truth(state, applied, commands):
        return rollout_with_latent(
            target, state, commands, DT, applied, belief.input_spec.control_roles
        )[0]

    for sample in samples:
        result = sample.result
        if not result.command_usable:
            continue
        predicted = result.predicted_states
        actual = planned_truth(
            predicted[0], result.predicted_latent_states[0], result.predicted_commands
        )
        predicted_utilization = np.asarray(
            jax.vmap(belief.model.validity_utilization)(predicted)
        )
        error = (
            np.asarray(jax.vmap(rigid_body_local_error)(predicted[1:], actual[1:]))
            / scales
        )
        outside = np.max(predicted_utilization[1:], axis=1) > 1.0 + 1e-6
        for key, mask in (("inside", ~outside), ("outside", outside)):
            prediction_errors[key].extend(np.mean(error[mask] ** 2, axis=1).tolist())
        if np.any(predicted_utilization > 1.0 + 1e-6):
            step, feature = np.argwhere(predicted_utilization > 1.0 + 1e-6)[0]
            forecast_crossings.append(
                {
                    "solve_time_s": sample.step * DT,
                    "first_exit_horizon_s": int(step) * DT,
                    "feature": FEATURES[int(feature)],
                }
            )
    events = []
    previous = None
    for sample in samples:
        if sample.decision is None:
            continue
        decision = sample.decision
        key = (decision.mode.value, tuple(r.value for r in decision.reasons))
        if key != previous:
            events.append({"time_s": sample.step * DT, **decision.to_dict()})
            previous = key
    commands = np.asarray(link.commands)
    accepted = [
        s
        for s in samples
        if s.decision is not None and s.decision.nominal_command_accepted
    ]
    reasons = Counter(
        r.value for s in samples if s.decision is not None for r in s.decision.reasons
    )
    modes = Counter(s.decision.mode.value for s in samples if s.decision is not None)
    return {
        "supervised": supervised,
        "fault": fault,
        "loop": summary.to_dict(),
        "initial_normalized_tracking_rms": float(np.sqrt(np.mean(errors[0] ** 2))),
        "tail_normalized_tracking_rms": float(np.sqrt(np.mean(errors[-20:] ** 2))),
        "terminal_normalized_tracking_rms": float(np.sqrt(np.mean(errors[-1] ** 2))),
        "terminal_attitude_rate_within_tolerances": bool(
            np.all(np.abs(errors[-1, 6:]) <= 1.0)
        ),
        "finite": bool(np.all(np.isfinite(states)) and np.all(np.isfinite(commands))),
        "maximum_command_bound_violation": float(
            max(np.max(-commands), np.max(commands - 1.0), 0.0)
        ),
        "features": feature_summary(utilization),
        "first_unsupported_forecast": forecast_crossings[0]
        if forecast_crossings
        else None,
        "counterfactual_planned_prediction_error": {
            key: {
                "stage_count": len(values),
                "normalized_rms": float(np.sqrt(np.mean(values))) if values else None,
            }
            for key, values in prediction_errors.items()
        },
        "supervisor_mode_counts": dict(modes),
        "supervisor_reason_counts": dict(reasons),
        "accepted_nominal_command_count": len(accepted),
        "all_observed_states_within_support": bool(np.all(utilization <= 1.0 + 1e-6)),
        "all_accepted_plans_within_support": all(
            s.result.diagnostics.maximum_validity_utilization <= 1.0 + 1e-6
            for s in accepted
        ),
        "all_accepted_commands_usable": all(s.result.command_usable for s in accepted),
        "all_accepted_states_fresh": all(
            s.decision.state_age_s <= supervisor.config.maximum_state_age_s
            for s in accepted
        ),
        "supervisor_events": events,
        "final_supervisor_mode": samples[-1].decision.mode.value
        if supervised
        else None,
    }


def run(output):
    stale, informed, target, _ = recovery._build_beliefs()
    for seed in ADDITIONAL_SEEDS:
        trajectory = recovery._configuration_trajectory(
            target,
            recovery.TARGET_LOG_ARM_LENGTH_RATIO,
            seed=seed,
            duration_s=ADDITIONAL_DURATION_S,
            source_group=f"additional-adaptation-{seed}",
        )
        informed, _ = informed.absorb(trajectory)
    calibration = [
        coverage_flight(informed, target, phase) for phase in CALIBRATION_PHASES
    ]
    qualified = support_from_calibration(informed, calibration)
    exploratory_validation = validate_coverage(
        qualified, informed, target, phases=EXPLORATORY_PHASES, amplitude=0.35
    )
    validation = validate_coverage(
        qualified,
        informed,
        target,
        phases=VALIDATION_PHASES,
        amplitude=OPERATING_ROLL_AMPLITUDE_RAD,
    )
    supported = all(
        row["prediction_error_passed"] and row["all_observations_inside_support"]
        for row in validation
    )
    report = {
        "format_version": 1,
        "diagnostic_only": True,
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "semantics": {
            "synthetic": True,
            "discrete_supervision_clock": True,
            "cpu_deadline_gate_disabled_for_nominal_offline_solves": True,
            "faults_injected": True,
            "support_box_is_not_a_physical_safety_set": True,
            "model_mean_and_covariance_unchanged_by_support_calibration": True,
            "reported_plan_support_uses_mean_trajectory": True,
            "feature_traces_assume_zero_wind": True,
        },
        "configuration": {
            "duration_s": INTERVALS * DT,
            "sample_period_s": DT,
            "additional_identification_seeds": list(ADDITIONAL_SEEDS),
            "calibration_phases_rad": list(CALIBRATION_PHASES),
            "validation_phases_rad": list(VALIDATION_PHASES),
            "operating_roll_amplitude_rad": OPERATING_ROLL_AMPLITUDE_RAD,
            "coverage_duration_s": 6.0,
            "coverage_roll_amplitude_rad": 0.35,
            "coverage_frequency_hz": 0.8,
            "maximum_validation_component_rms_fraction": MAXIMUM_VALIDATION_COMPONENT_RMS,
            "fault_interval_indices": [10, 11, 12],
            "combined_fault_interval_indices": {
                "stale_state": [10, 11, 12],
                "deadline": [30, 31, 32],
                "unresolved_parameters": [50, 51, 52],
            },
        },
        "original_support": informed.support.to_dict(),
        "calibrated_support": qualified.support.to_dict(),
        "coverage_validation": validation,
        "unqualified_higher_amplitude_validation": exploratory_validation,
        "coverage_validation_passed": supported,
        "scenarios": [],
    }
    cases = [
        ("original_without_supervisor", informed, "original", False, None),
        ("original_with_supervisor", informed, "original", True, None),
    ]
    if supported:
        cases += [
            ("calibrated_original_recovery", qualified, "original", True, None),
            ("calibrated_small_recovery", qualified, "small", True, None),
        ]
        cases += [
            (f"calibrated_{disturbance}_{fault}", qualified, disturbance, True, fault)
            for disturbance in ("original", "small")
            for fault in (
                "stale_state",
                "deadline",
                "unresolved_parameters",
                "combined",
            )
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, belief, disturbance, supervised, fault in cases:
        print(name, flush=True)
        row = simulate(
            belief,
            target,
            initial_state(stale, disturbance),
            supervised=supervised,
            fault=fault,
        )
        report["scenarios"].append({"name": name, "disturbance": disturbance, **row})
        print(
            "tail",
            row["tail_normalized_tracking_rms"],
            "support",
            max(f["maximum_utilization"] for f in row["features"].values()),
            "modes",
            row["supervisor_mode_counts"],
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    sources = [
        Path(__file__),
        Path(__file__).with_name("investigate_recovery.py"),
        *[
            Path(recovery.glassbox.__file__).parent / name
            for name in (
                *recovery.BENCHMARK_SOURCE_FILES,
                "control/supervisor.py",
                "integrations/loop.py",
            )
        ],
    ]
    report["source_sha256"] = {
        str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
