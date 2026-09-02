"""Fixed Crazyflow hidden-plant prototype for adjustable-arm recovery work."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import signal
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    RuntimeDynamicsBelief,
    parameter_belief_from_dict,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.belief.parameter_prior import StructuredParameterPrior
from glassbox.control.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.control.nmpc import NMPCController, TrackingTolerances
from glassbox.control.nmpc.solver import _SolverPolicy
from glassbox.core.data import (
    ObservationChannel,
    Trajectory,
    make_trajectory_spec,
    save_trajectory_npz,
)
from glassbox.core.dynamics import (
    MOTOR_MIXER,
    QUADROTOR_CONTROL_NAMES,
    ModelParams,
    physics_parameters,
)
from glassbox.core.evaluation import windowed_rollout_evaluation
from glassbox.core.geometry import rigid_body_local_error
from glassbox.core.identification import fit_dynamics
from glassbox.core.runtime import (
    DirectActuationMap,
    RuntimeDynamicsModel,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import initial_parameter_guess, resting_state
from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig

PROTOTYPE_SCHEMA_VERSION = 2
DEFAULT_BASELINE_SEED = 101
DEFAULT_MODIFIED_SEED = 102
DEFAULT_DURATION_S = 6.0
DEFAULT_ARM_LENGTH_RATIO = math.exp(0.20)
ADAPTATION_HORIZON_STEPS = 5
CONTROL_HORIZON_STEPS = 30
FLEET_LOG_ARM_LENGTH_RATIOS = (-0.25, -0.125, 0.0, 0.125, 0.25)
FLEET_PROFILE_COUNT = 2
RECOVERY_DURATION_S = 0.8
RECOVERY_TAIL_DURATION_S = 0.2
ONLINE_EVIDENCE_DURATION_S = 0.8
ONLINE_POST_INSTALL_DURATION_S = 1.0
ONLINE_MAXIMUM_DURATION_S = 7.5
ONLINE_INTEGRATED_PREWARM_STEPS = 10

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


@dataclass(frozen=True)
class _PreparedOnlineController:
    """Candidate belief/controller pair prepared away from the control loop."""

    update_report: dict[str, Any]
    update_wall_time_s: float
    controller_validation_wall_time_s: float | None
    total_wall_time_s: float
    validation_statuses: tuple[str, ...]
    ready_for_install: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class _WorkerBeliefUpdate:
    """Pickle-safe numerical result returned by the isolated update process."""

    parameter_leaves: tuple[np.ndarray, ...]
    parameter_belief_payload: dict[str, Any]
    provenance: dict[str, Any]
    predictive_error_parameter_update_count: int
    update_report: dict[str, Any]
    update_wall_time_s: float
    update_cpu_time_s: float
    process_id: int
    process_niceness: int | None


def _initialize_adaptation_worker() -> None:
    """Lower worker scheduling priority without touching the control process."""

    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except OSError:
            pass


def _process_niceness() -> int | None:
    if not hasattr(os, "nice"):
        return None
    try:
        return os.nice(0)
    except OSError:
        return None


def _set_process_suspended(process_id: int, *, suspended: bool) -> bool:
    """Best-effort POSIX temporal partition for the adaptation worker."""

    process_signal = getattr(signal, "SIGSTOP" if suspended else "SIGCONT", None)
    if process_signal is None:
        return False
    try:
        os.kill(process_id, process_signal)
    except (OSError, ProcessLookupError):
        return False
    if suspended and all(
        hasattr(os, name)
        for name in ("waitid", "P_PID", "WSTOPPED", "WEXITED", "WNOWAIT")
    ):
        try:
            status = os.waitid(
                os.P_PID,
                process_id,
                os.WSTOPPED | os.WEXITED | os.WNOWAIT,
            )
        except (ChildProcessError, OSError):
            return False
        return status is not None and status.si_code == getattr(
            os,
            "CLD_STOPPED",
            status.si_code,
        )
    return True


def _run_isolated_belief_update(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> _WorkerBeliefUpdate:
    started_at = time.perf_counter()
    cpu_started_at = time.process_time()
    updated, update_report = belief.update(telemetry)
    update_cpu_time_s = time.process_time() - cpu_started_at
    return _WorkerBeliefUpdate(
        parameter_leaves=tuple(
            np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(updated.params)
        ),
        parameter_belief_payload=updated.parameter_belief.to_dict(),
        provenance=dict(updated.provenance),
        predictive_error_parameter_update_count=(
            updated.predictive_error_parameter_update_count
        ),
        update_report=update_report.to_dict(),
        update_wall_time_s=time.perf_counter() - started_at,
        update_cpu_time_s=update_cpu_time_s,
        process_id=os.getpid(),
        process_niceness=_process_niceness(),
    )


def _belief_from_worker_update(
    belief: DynamicsBelief,
    update: _WorkerBeliefUpdate,
) -> DynamicsBelief:
    return replace(
        belief,
        params=_parameters_from_worker_update(belief.params, update),
        parameter_belief=parameter_belief_from_dict(update.parameter_belief_payload),
        predictive_error_parameter_update_count=(
            update.predictive_error_parameter_update_count
        ),
        provenance=update.provenance,
    )


def _parameters_from_worker_update(
    template: ModelParams,
    update: _WorkerBeliefUpdate,
) -> ModelParams:
    template_leaves, structure = jax.tree_util.tree_flatten(template)
    if len(template_leaves) != len(update.parameter_leaves):
        raise ValueError("isolated update parameter structure changed")
    leaves = []
    for template_leaf, candidate_leaf in zip(
        template_leaves,
        update.parameter_leaves,
        strict=True,
    ):
        template_array = np.asarray(template_leaf)
        candidate_array = np.asarray(candidate_leaf)
        if candidate_array.shape != template_array.shape:
            raise ValueError("isolated update parameter shape changed")
        if candidate_array.dtype != template_array.dtype:
            raise ValueError("isolated update parameter dtype changed")
        if not np.all(np.isfinite(candidate_array)):
            raise ValueError("isolated update parameters must be finite")
        leaves.append(candidate_array)
    return jax.tree_util.tree_unflatten(structure, leaves)


def _runtime_belief_from_worker_update(
    template: NMPCController,
    update: _WorkerBeliefUpdate,
) -> RuntimeDynamicsBelief:
    """Bind numerical worker output to the prevalidated runtime template."""

    parameter_belief = template.belief.parameter_belief
    if update.parameter_belief_payload != parameter_belief.to_dict():
        raise ValueError("isolated update parameter uncertainty changed")
    if (
        update.predictive_error_parameter_update_count
        != template.belief.predictive_error_parameter_update_count
    ):
        raise ValueError("isolated update predictive-error semantics changed")
    nominal = template.model.rebind_parameters(
        _parameters_from_worker_update(template.model.params, update)
    )
    return replace(template.belief, nominal=nominal)


def _post_update_controller_template(belief: DynamicsBelief) -> DynamicsBelief:
    """Create the exact non-mean runtime semantics expected after one update."""

    parameter_belief = belief.parameter_belief
    if not isinstance(parameter_belief, LocalGaussianParameterBelief):
        raise ValueError("online controller template requires parameter uncertainty")
    return replace(
        belief,
        parameter_belief=replace(
            parameter_belief,
            evidence_count=parameter_belief.evidence_count + 1,
            effective_sample_count=parameter_belief.effective_sample_count + 1.0,
            update_count=parameter_belief.update_count + 1,
        ),
    )


def _applied_motor_observation_channels() -> tuple[ObservationChannel, ...]:
    return tuple(
        ObservationChannel(
            name=f"applied_{motor}_motor_thrust_fraction",
            role=f"applied_{motor}_motor_thrust_fraction",
            semantic="normalized_per_motor_thrust_fraction",
            unit="1",
            frame="FLU",
            source="crazyflow_first_principles_rotor_state",
        )
        for motor in (
            "front_left",
            "front_right",
            "rear_right",
            "rear_left",
        )
    )


def _crazyflow_solver_policy() -> _SolverPolicy:
    """Return the fixed 50 Hz prototype policy with explicit timing headroom."""

    return _SolverPolicy(
        horizon_steps=CONTROL_HORIZON_STEPS,
        block_count=10,
        maximum_iterations=6,
        line_search_steps=16,
    )


def _trajectory_statistics(trajectory: Trajectory) -> dict[str, Any]:
    return {
        "duration_s": float(trajectory.time_s[-1]),
        "sample_count": len(trajectory.time_s),
        "finite": bool(
            np.all(np.isfinite(trajectory.states))
            and np.all(np.isfinite(trajectory.controls))
        ),
        "position_minimum_m": np.min(trajectory.states[:, 0:3], axis=0).tolist(),
        "position_maximum_m": np.max(trajectory.states[:, 0:3], axis=0).tolist(),
        "velocity_maximum_absolute_m_s": np.max(
            np.abs(trajectory.states[:, 3:6]), axis=0
        ).tolist(),
        "angular_velocity_maximum_absolute_rad_s": np.max(
            np.abs(trajectory.states[:, 10:13]), axis=0
        ).tolist(),
        "command_minimum": np.min(trajectory.controls, axis=0).tolist(),
        "command_maximum": np.max(trajectory.controls, axis=0).tolist(),
    }


def generate_crazyflow_trajectory(
    *,
    seed: int,
    duration_s: float = DEFAULT_DURATION_S,
    arm_length_ratio: float = 1.0,
    source_group: str | None = None,
    configuration_id: str = "crazyflow_adjustable_arm_unknown",
    plant_config: CrazyflowPlantConfig | None = None,
) -> Trajectory:
    """Generate canonical command/state telemetry from a hidden Crazyflow plant."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(arm_length_ratio) or arm_length_ratio <= 0.0:
        raise ValueError("arm_length_ratio must be finite and positive")
    config = CrazyflowPlantConfig() if plant_config is None else plant_config
    interval_count = round(duration_s / config.sample_period_s)
    if interval_count < 1:
        raise ValueError("duration is shorter than one control interval")

    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    frequency_scale = rng.uniform(0.9, 1.1)
    plant = CrazyflowPlant(config)
    try:
        plant.set_arm_length_ratio(arm_length_ratio)
        initial_state = resting_state()
        initial_state[2] = 1.0
        hover = plant.hover_motor_thrust_fraction
        hover_command = np.full(4, hover, dtype=np.float64)
        initial = plant.reset(
            initial_state,
            applied_motor_thrust_fraction=hover_command,
        )

        states = np.empty((interval_count + 1, 13), dtype=np.float64)
        controls = np.empty((interval_count, 4), dtype=np.float64)
        applied = np.empty((interval_count + 1, 4), dtype=np.float64)
        states[0] = initial.state
        applied[0] = initial.applied_motor_thrust_fraction
        mixer_transpose = np.asarray(MOTOR_MIXER.T)
        for index in range(interval_count):
            time_s = index * config.sample_period_s
            state = states[index]
            excitation_ramp = min(time_s / 0.5, 1.0)
            attitude_vector = np.sign(state[6] or 1.0) * state[7:10]
            desired_angles = np.asarray(
                (
                    0.05 * state[1] + 0.08 * state[4],
                    -0.05 * state[0] - 0.08 * state[3],
                    0.0,
                )
            )
            desired_attitude_vector = 0.5 * desired_angles
            differential_excitation = excitation_ramp * np.asarray(
                (
                    0.006
                    * np.sin(frequency_scale * 2.0 * np.pi * 0.53 * time_s + phases[2]),
                    0.006 * np.sin(2.0 * np.pi * 0.67 * time_s + phases[3]),
                    0.004 * np.sin(2.0 * np.pi * 0.41 * time_s + phases[4]),
                )
            )
            desired_differential = (
                -0.28 * (attitude_vector - desired_attitude_vector)
                - 0.035 * state[10:13]
                + differential_excitation
            )
            collective = excitation_ramp * (
                0.004
                * np.sin(frequency_scale * 2.0 * np.pi * 0.37 * time_s + phases[0])
                + 0.002 * np.sin(2.0 * np.pi * 0.83 * time_s + phases[1])
            )
            collective += -0.12 * (state[2] - 1.0) - 0.08 * state[5]
            command = hover + collective + 0.25 * mixer_transpose @ desired_differential
            command = np.clip(command, 0.05, 0.95)
            controls[index] = command
            sample = plant.step(command)
            states[index + 1] = sample.state
            applied[index + 1] = sample.applied_motor_thrust_fraction

        group = f"crazyflow-trajectory-{seed}" if source_group is None else source_group
        spec = make_trajectory_spec(
            QUADROTOR_CONTROL_NAMES,
            family="multirotor",
            observation_source="simulator_truth",
            configuration_id=configuration_id,
            observations=_applied_motor_observation_channels(),
        )
        return Trajectory(
            time_s=np.arange(interval_count + 1, dtype=np.float64)
            * config.sample_period_s,
            states=states,
            controls=controls,
            observations=applied,
            spec=spec,
            labels={
                "seed": seed,
                "source_group": group,
            },
            provenance={
                "adapter": {
                    "name": "crazyflow_hidden_plant",
                    "schema_version": PROTOTYPE_SCHEMA_VERSION,
                    "crazyflow_version": plant.crazyflow_version,
                },
                "simulator_contract": {
                    "drone": config.drone,
                    "dynamics": "first_principles",
                    "control": "rotor_vel",
                    "simulation_frequency_hz": config.simulation_frequency_hz,
                    "control_frequency_hz": config.control_frequency_hz,
                    "canonical_motor_order": (
                        "front_left,front_right,rear_right,rear_left"
                    ),
                    "quaternion_storage": "wxyz",
                    "world_frame": "NWU",
                    "body_frame": "FLU",
                    "command_semantic": "normalized_per_motor_thrust_fraction",
                    "maximum_motor_thrust_n": config.maximum_motor_thrust_n,
                },
                "generator": {"seed": seed},
                "telemetry_only": {
                    "hidden_physical_parameters_supplied_to_glassbox": False,
                    "applied_actuator_state_retained_as_typed_observation": True,
                },
            },
        )
    finally:
        plant.close()


def _fit_effective_model(
    trajectories: list[Trajectory],
    initial_params: ModelParams,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[ModelParams, dict[str, Any]]:
    from glassbox.core.data import trajectory_windows

    windows = trajectory_windows(
        trajectories,
        horizon=ADAPTATION_HORIZON_STEPS,
        stride=2,
        maximum_windows=1_000,
    )
    started_at = time.perf_counter()
    result = fit_dynamics(
        windows,
        initial_params,
        steps=steps,
        learning_rate=learning_rate,
        instantaneous_rotational_response=True,
        diagonal_angular_control=True,
    )
    elapsed_s = time.perf_counter() - started_at
    physical = physics_parameters(result.params).physical()
    return result.params, {
        "trajectory_count": len(trajectories),
        "window_count": len(windows.initial_states),
        "horizon_s": ADAPTATION_HORIZON_STEPS * windows.dt_s,
        "optimization_steps": steps,
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "wall_time_s": elapsed_s,
        "effective_parameters": {
            "thrust_accel": float(physical["thrust_accel"]),
            "angular_accel": np.asarray(physical["angular_accel"]).tolist(),
            "linear_drag": float(physical["linear_drag"]),
            "angular_drag": np.asarray(physical["angular_drag"]).tolist(),
            "motor_time_constant": float(physical["motor_time_constant"]),
        },
    }


def _configuration_direction_prior(
    members: list[DynamicsBelief],
    configuration_coordinates: tuple[float, ...],
) -> tuple[StructuredParameterPrior, int]:
    """Keep only fleet variation explained by one known configuration axis."""

    if len(members) != len(configuration_coordinates) or len(members) < 2:
        raise ValueError("configuration coordinates must identify every fleet member")
    raw = StructuredParameterPrior.from_beliefs(
        tuple(members),
        source="independent_crazyflow_configuration_fits",
        member_labels=tuple(
            f"fleet-configuration-{index}" for index in range(len(members))
        ),
    )
    coordinates = np.asarray(configuration_coordinates, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)) or np.std(coordinates) <= 0.0:
        raise ValueError("configuration coordinates must contain finite variation")
    centered_coordinate = coordinates - np.mean(coordinates)
    values = np.stack(
        [
            np.asarray(structured_parameter_vector(member.params), dtype=np.float64)
            for member in members
        ]
    )
    centered_values = values - np.mean(values, axis=0)
    slope = (
        centered_coordinate @ centered_values / np.sum(np.square(centered_coordinate))
    )
    coordinate_variance = float(np.var(coordinates, ddof=1))
    between = np.outer(slope, slope) * coordinate_variance

    inverse_scale = 1.0 / raw.natural_scale
    normalized = between * inverse_scale[:, None] * inverse_scale[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (normalized + normalized.T))
    tolerance = (
        np.finfo(np.float64).eps
        * len(eigenvalues)
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank != 1:
        raise ValueError("one-dimensional configuration regression must have rank one")
    unresolved = eigenvectors[:, : len(eigenvalues) - rank]
    completion_normalized = unresolved @ unresolved.T
    completion = (
        completion_normalized * raw.natural_scale[:, None] * raw.natural_scale[None, :]
    )
    prior = replace(
        raw,
        between_member_covariance=between,
        completion_covariance=completion,
        source="one_dimensional_arm_coordinate_regression_from_crazyflow_fits",
    )
    return prior, raw.empirical_rank


def _build_crazyflow_beliefs() -> tuple[
    DynamicsBelief,
    DynamicsBelief,
    RuntimeDynamicsModel,
    dict[str, Any],
]:
    baseline_trajectories = [
        generate_crazyflow_trajectory(
            seed=101 + index,
            duration_s=DEFAULT_DURATION_S,
            arm_length_ratio=1.0,
            source_group=f"prechange-flight-{index}",
            configuration_id="prechange_vehicle",
        )
        for index in range(2)
    ]
    baseline_params, baseline_fit = _fit_effective_model(
        baseline_trajectories,
        initial_parameter_guess(),
        steps=250,
        learning_rate=0.02,
    )
    runtime_spec = runtime_spec_from_trajectory(baseline_trajectories[0])

    members: list[DynamicsBelief] = []
    member_reports: list[dict[str, Any]] = []
    error_samples: dict[int, list[EmpiricalErrorSample]] = {
        ADAPTATION_HORIZON_STEPS: [],
        CONTROL_HORIZON_STEPS: [],
    }
    for member_index, log_ratio in enumerate(FLEET_LOG_ARM_LENGTH_RATIOS):
        ratio = math.exp(log_ratio)
        label = f"fleet-configuration-{member_index}"
        trajectories = [
            generate_crazyflow_trajectory(
                seed=200 + 10 * member_index + profile_index,
                duration_s=4.0,
                arm_length_ratio=ratio,
                source_group=label,
                configuration_id=label,
            )
            for profile_index in range(FLEET_PROFILE_COUNT)
        ]
        member_params, fit_report = _fit_effective_model(
            trajectories,
            baseline_params,
            steps=100,
            learning_rate=0.01,
        )
        members.append(
            DynamicsBelief(
                member_params,
                trajectories[0].spec,
                runtime_spec,
                provenance={"role": "independently_fitted_fleet_member"},
            )
        )
        member_reports.append(
            {
                "member": label,
                "experiment_arm_length_ratio": ratio,
                "fit": fit_report,
            }
        )
        for profile_index, trajectory in enumerate(trajectories):
            for horizon_steps in error_samples:
                _, errors = windowed_rollout_evaluation(
                    baseline_params,
                    trajectory,
                    horizon_steps=horizon_steps,
                    stride_steps=horizon_steps,
                )
                error_samples[horizon_steps].append(
                    EmpiricalErrorSample(
                        errors=errors,
                        source_group=label,
                        trajectory_id=f"{label}-profile-{profile_index}",
                    )
                )

    fleet_prior, unprojected_fit_scatter_rank = _configuration_direction_prior(
        members,
        FLEET_LOG_ARM_LENGTH_RATIOS,
    )
    sample_period_s = baseline_trajectories[0].nominal_dt_s
    predictive_error = EmpiricalHorizonPredictiveError.from_samples(
        {
            horizon_steps * sample_period_s: tuple(samples)
            for horizon_steps, samples in error_samples.items()
        }
    )
    parameter_belief = LocalGaussianParameterBelief(
        parameter_names=structured_parameter_names(baseline_params),
        covariance=fleet_prior.between_member_covariance,
        source="independent_crazyflow_configuration_fits",
        evidence_count=fleet_prior.member_count,
        effective_sample_count=float(fleet_prior.member_count),
    )

    adaptation = generate_crazyflow_trajectory(
        seed=21,
        duration_s=0.8,
        arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
        source_group="unknown-target-adaptation",
        configuration_id="unknown_target_configuration",
    )
    belief = DynamicsBelief(
        baseline_params,
        adaptation.spec,
        runtime_spec,
        predictive_error=predictive_error,
        parameter_belief=parameter_belief,
        provenance={
            "role": "prechange_vehicle_plus_configuration_fleet",
            "hidden_target_configuration_supplied": False,
        },
    )
    updated, update_report = belief.update(adaptation)

    oracle_trajectories = [
        generate_crazyflow_trajectory(
            seed=31 + index,
            duration_s=5.0,
            arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
            source_group=f"target-oracle-flight-{index}",
            configuration_id="target_oracle_only",
        )
        for index in range(2)
    ]
    oracle_params, oracle_fit = _fit_effective_model(
        oracle_trajectories,
        baseline_params,
        steps=150,
        learning_rate=0.01,
    )
    oracle_runtime = RuntimeDynamicsModel(
        oracle_params,
        belief.input_spec,
        replace(
            runtime_spec,
            certified_prediction_horizon_s=(CONTROL_HORIZON_STEPS * sample_period_s),
            certification_source="Crazyflow prototype equal-horizon comparison",
        ),
        DirectActuationMap(belief.input_spec.controls),
    )

    evaluation = generate_crazyflow_trajectory(
        seed=22,
        duration_s=1.6,
        arm_length_ratio=DEFAULT_ARM_LENGTH_RATIO,
        source_group="unknown-target-independent-evaluation",
        configuration_id="unknown_target_configuration",
    )
    _, before_errors = windowed_rollout_evaluation(
        belief.params,
        evaluation,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    _, after_errors = windowed_rollout_evaluation(
        updated.params,
        evaluation,
        horizon_steps=CONTROL_HORIZON_STEPS,
        stride_steps=CONTROL_HORIZON_STEPS,
    )
    tolerances = np.asarray(
        TrackingTolerances.for_platform("multirotor").local_state_scale
    )
    before_rms = float(np.sqrt(np.mean(np.square(before_errors / tolerances))))
    after_rms = float(np.sqrt(np.mean(np.square(after_errors / tolerances))))
    evidence = {
        "baseline_fit": baseline_fit,
        "fleet": {
            "member_count": fleet_prior.member_count,
            "empirical_rank": fleet_prior.empirical_rank,
            "unprojected_fit_scatter_rank": unprojected_fit_scatter_rank,
            "configuration_delta_covariance_rank": int(
                np.linalg.matrix_rank(fleet_prior.between_member_covariance)
            ),
            "known_configuration_coordinate_used_for_fleet_only": True,
            "completion_fraction_in_natural_coordinates": (
                fleet_prior.completion_fraction_in_natural_coordinates
            ),
            "members": member_reports,
            "predictive_error_horizons_s": list(predictive_error.horizons_s),
            "predictive_error_group_count": list(
                predictive_error.independent_group_count
            ),
        },
        "adaptation": update_report.to_dict(),
        "oracle_fit": oracle_fit,
        "independent_prediction": {
            "horizon_s": CONTROL_HORIZON_STEPS * sample_period_s,
            "normalized_rms_before": before_rms,
            "normalized_rms_after": after_rms,
            "normalized_rms_ratio": after_rms / before_rms,
        },
    }
    return belief, updated, oracle_runtime, evidence


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
    state[0:3] = (0.10, -0.08, 1.12)
    state[6:10] = _quaternion_from_euler(0.16, -0.12, 0.08)
    envelope = belief.runtime_spec.validity_envelope
    state[3:6] = np.asarray(envelope.body_velocity_center_m_s) + 0.15 * np.asarray(
        envelope.body_velocity_half_width_m_s
    ) * np.asarray((1.0, -1.0, -1.0))
    state[10:13] = np.asarray(
        envelope.angular_velocity_center_rad_s
    ) + 0.25 * np.asarray(envelope.angular_velocity_half_width_rad_s) * np.asarray(
        (1.0, -1.0, 1.0)
    )
    return state


def _online_recovery_trajectory(
    belief: DynamicsBelief,
    states: list[np.ndarray],
    controls: list[np.ndarray],
    applied_motor_thrust: list[np.ndarray],
) -> Trajectory:
    """Package a state-aligned prefix without exposing the hidden configuration."""

    if len(states) != len(controls) + 1:
        raise ValueError("online recovery states must bracket every control")
    if len(applied_motor_thrust) != len(states):
        raise ValueError("applied motor observations must align with recovery states")
    if len(controls) < 1:
        raise ValueError("online recovery evidence needs at least one transition")
    sample_period_s = belief.runtime_spec.sample_period_s
    return Trajectory(
        time_s=np.arange(len(states), dtype=np.float64) * sample_period_s,
        states=np.asarray(states, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64),
        observations=np.asarray(applied_motor_thrust, dtype=np.float64),
        spec=replace(
            belief.input_spec,
            observations=_applied_motor_observation_channels(),
        ),
        labels={
            "source_group": "unknown-target-online-recovery",
            "evidence_role": "stale-controller-recovery-prefix",
        },
        provenance={
            "collection": {
                "controller_model": "prechange_stale_belief",
                "state_aligned_applied_actuator_observations": True,
            },
            "telemetry_only": {
                "hidden_physical_parameters_supplied_to_glassbox": False,
                "target_configuration_label_supplied_to_glassbox": False,
            },
        },
    )


def _normalized_rms(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _eligible_recovery_evidence_start(
    state_validity: list[float],
    evidence_steps: int,
) -> int | None:
    """Return the first usable streaming-window start under the fixed gate."""

    if evidence_steps < 1:
        raise ValueError("evidence_steps must be positive")
    start = len(state_validity) - (evidence_steps + 1)
    if start < 0:
        return None
    recent = state_validity[start:]
    if not all(np.isfinite(value) for value in recent):
        return None
    return start if max(recent) <= 1.0 else None


def _simulate_online_adaptation_recovery(
    belief: DynamicsBelief,
    initial_state: np.ndarray,
) -> dict[str, Any]:
    """Continue stale control while preparing and atomically installing an update."""

    plant = CrazyflowPlant()
    plant.set_arm_length_ratio(DEFAULT_ARM_LENGTH_RATIO)
    hover = np.full(4, plant.hover_motor_thrust_fraction, dtype=np.float64)
    reference_state = resting_state()
    reference_state[2] = 1.0
    stale_controller = NMPCController(belief, _policy=_crazyflow_solver_policy())
    reference = stale_controller.hold_reference(jnp.asarray(reference_state))

    prewarm_started_at = time.perf_counter()
    cold = stale_controller.solve(
        jnp.asarray(initial_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(cold.command)
    warm = stale_controller.solve(
        jnp.asarray(initial_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=cold.warm_start,
    )
    jax.block_until_ready(warm.command)
    stale_prewarm_wall_time_s = time.perf_counter() - prewarm_started_at
    if not cold.command_usable or not warm.command_usable:
        plant.close()
        raise RuntimeError(
            "stale controller could not be prewarmed for online recovery"
        )

    controller_template = NMPCController(
        _post_update_controller_template(belief),
        _policy=_crazyflow_solver_policy(),
    )
    template_reference = controller_template.hold_reference(
        jnp.asarray(reference_state)
    )
    template_prewarm_started_at = time.perf_counter()
    template_cold = controller_template.solve(
        jnp.asarray(initial_state),
        template_reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(template_cold.command)
    template_warm = controller_template.solve(
        jnp.asarray(initial_state),
        template_reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=template_cold.warm_start,
    )
    jax.block_until_ready(template_warm.command)
    template_hover_cold = controller_template.solve(
        jnp.asarray(reference_state),
        template_reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(template_hover_cold.command)
    template_hover_warm = controller_template.solve(
        jnp.asarray(reference_state),
        template_reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=template_hover_cold.warm_start,
    )
    jax.block_until_ready(template_hover_warm.command)
    candidate_template_prewarm_wall_time_s = (
        time.perf_counter() - template_prewarm_started_at
    )
    if not all(
        result.command_usable
        for result in (
            template_cold,
            template_warm,
            template_hover_cold,
            template_hover_warm,
        )
    ):
        plant.close()
        raise RuntimeError("post-update controller template could not be prewarmed")

    worker_prewarm_telemetry = generate_crazyflow_trajectory(
        seed=299,
        duration_s=ONLINE_EVIDENCE_DURATION_S,
        arm_length_ratio=math.exp(FLEET_LOG_ARM_LENGTH_RATIOS[-1]),
        source_group="known-fleet-adaptation-worker-prewarm",
        configuration_id="known_fleet_worker_prewarm",
    )
    worker_environment = {
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1 "
            "--xla_force_host_platform_device_count=1"
        ),
        "NPROC": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUM_INTRA_THREADS": "1",
        "NUM_INTER_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    }
    previous_environment = {name: os.environ.get(name) for name in worker_environment}
    worker_prewarm_started_at = time.perf_counter()
    executor: ProcessPoolExecutor | None = None
    try:
        os.environ.update(worker_environment)
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_adaptation_worker,
        )
        worker_prewarm = executor.submit(
            _run_isolated_belief_update,
            belief,
            worker_prewarm_telemetry,
        ).result()
    except Exception:
        plant.close()
        if executor is not None:
            executor.shutdown(wait=True)
        raise
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    worker_prewarm_wall_time_s = time.perf_counter() - worker_prewarm_started_at
    assert executor is not None

    # Prewarm the integrated controller/plant path before auditing deadlines.
    integrated_prewarm_started_at = time.perf_counter()
    prewarm_sample = plant.reset(
        initial_state,
        applied_motor_thrust_fraction=hover,
    )
    prewarm_previous = hover
    prewarm_warm_start = warm.warm_start
    for _ in range(ONLINE_INTEGRATED_PREWARM_STEPS):
        prewarm_result = stale_controller.solve(
            jnp.asarray(prewarm_sample.state),
            reference,
            jnp.asarray(prewarm_previous),
            applied_command=jnp.asarray(prewarm_sample.applied_motor_thrust_fraction),
            warm_start=prewarm_warm_start,
        )
        jax.block_until_ready(prewarm_result.command)
        if not prewarm_result.command_usable:
            plant.close()
            executor.shutdown(wait=True)
            raise RuntimeError("integrated stale control prewarm was not usable")
        prewarm_previous = np.asarray(prewarm_result.command)
        prewarm_warm_start = prewarm_result.warm_start
        prewarm_sample = plant.step(prewarm_previous)
    integrated_prewarm_wall_time_s = time.perf_counter() - integrated_prewarm_started_at
    sample = plant.reset(initial_state, applied_motor_thrust_fraction=hover)

    sample_period_s = plant.sample_period_s
    supervisor_config = MultirotorSupervisorConfig(
        command_minimum=tuple(
            float(value) for value in stale_controller.model.command_minimum
        ),
        command_maximum=tuple(
            float(value) for value in stale_controller.model.command_maximum
        ),
        collective_hold_command=tuple(float(value) for value in hover),
        maximum_state_age_s=2.0 * sample_period_s,
        maximum_command_age_s=sample_period_s,
    )
    supervisor = MultirotorFlightSupervisor(supervisor_config)
    evidence_steps = round(ONLINE_EVIDENCE_DURATION_S / sample_period_s)
    maximum_steps = round(ONLINE_MAXIMUM_DURATION_S / sample_period_s)
    post_install_steps = round(ONLINE_POST_INSTALL_DURATION_S / sample_period_s)
    states = [np.asarray(sample.state)]
    controls: list[np.ndarray] = []
    applied = [np.asarray(sample.applied_motor_thrust_fraction)]
    state_validity = [
        float(
            np.max(
                np.asarray(
                    stale_controller.model.validity_utilization(
                        jnp.asarray(sample.state)
                    )
                )
            )
        )
    ]
    control_modes: list[str] = []
    solve_times: list[float] = []
    outer_step_times: list[float] = []
    solve_statuses: list[str] = []
    solve_messages: list[str] = []
    support_modes: list[str] = []
    predicted_validity: list[float] = []
    support_validity: list[float] = []
    loop_lag: list[float] = []
    fallback_count = 0
    support_applied = 0
    nonfinite_diagnostic_count = 0
    outer_deadline_miss_count = 0
    outer_deadline_miss_steps: list[int] = []
    outer_deadline_miss_details: list[dict[str, Any]] = []
    absolute_deadline_miss_steps: list[int] = []
    absolute_completion_lateness: list[float] = []
    supervisor_modes: list[str] = []
    supervisor_reasons: list[str] = []
    supervisor_decision_times: list[float] = []
    supervisor_intervention_count = 0
    supervisor_fault_injected = False
    supervisor_fault_step: int | None = None
    supervisor_fault_response: dict[str, Any] | None = None
    previous = hover
    active_controller = stale_controller
    active_reference = reference
    active_warm_start = warm.warm_start
    active_mode = "stale"
    submission_wall_time: float | None = None
    submission_sim_time_s: float | None = None
    evidence_start_sim_time_s: float | None = None
    evidence_maximum_validity: float | None = None
    candidate_ready_wall_time_s: float | None = None
    candidate_resolved_sim_time_s: float | None = None
    install_wall_time_s: float | None = None
    install_sim_time_s: float | None = None
    install_step: int | None = None
    prepared: _PreparedOnlineController | None = None
    worker_update: _WorkerBeliefUpdate | None = None
    candidate_future: Future[_WorkerBeliefUpdate] | None = None
    future_consumed = False
    worker_error: str | None = None
    worker_result_wall_time_s: float | None = None
    pending_controller: NMPCController | None = None
    pending_reference: Any | None = None
    worker_suspend_count = 0
    worker_resume_count = 0
    temporal_partition_available = hasattr(signal, "SIGSTOP") and hasattr(
        signal, "SIGCONT"
    )
    control_loop_started_at = time.perf_counter()
    control_loop_ended_at = control_loop_started_at
    try:
        for index in range(maximum_steps):
            step_started_at = time.perf_counter()
            simulation_time_s = index * sample_period_s
            worker_suspended = False
            candidate_deferred_for_tick = False
            worker_resolution_wall_time_s = 0.0
            if (
                candidate_future is not None
                and not future_consumed
                and not candidate_future.done()
            ):
                worker_suspended = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=True,
                )
                worker_suspend_count += int(worker_suspended)
            if (
                candidate_future is not None
                and not future_consumed
                and candidate_future.done()
            ):
                worker_resolution_started_at = time.perf_counter()
                future_consumed = True
                worker_result_wall_time_s = (
                    time.perf_counter() - control_loop_started_at
                )
                candidate_resolved_sim_time_s = simulation_time_s
                try:
                    worker_update = candidate_future.result()
                    if not bool(worker_update.update_report.get("applied", False)):
                        reason = str(
                            worker_update.update_report.get(
                                "reason",
                                "isolated belief update was not applied",
                            )
                        )
                        prepared = _PreparedOnlineController(
                            update_report=worker_update.update_report,
                            update_wall_time_s=worker_update.update_wall_time_s,
                            controller_validation_wall_time_s=None,
                            total_wall_time_s=worker_update.update_wall_time_s,
                            validation_statuses=(),
                            ready_for_install=False,
                            rejection_reason=reason,
                        )
                        candidate_ready_wall_time_s = worker_result_wall_time_s
                    else:
                        updated_belief = _runtime_belief_from_worker_update(
                            controller_template,
                            worker_update,
                        )
                        pending_controller = controller_template.rebind_belief(
                            updated_belief
                        )
                        pending_reference = pending_controller.hold_reference(
                            jnp.asarray(reference_state)
                        )
                        candidate_deferred_for_tick = True
                except Exception as error:
                    worker_error = f"{type(error).__name__}: {error}"
                    reason = f"isolated update failed: {worker_error}"
                    update_wall_time_s = (
                        0.0
                        if worker_update is None
                        else worker_update.update_wall_time_s
                    )
                    prepared = _PreparedOnlineController(
                        update_report={"applied": False, "reason": reason},
                        update_wall_time_s=update_wall_time_s,
                        controller_validation_wall_time_s=None,
                        total_wall_time_s=update_wall_time_s,
                        validation_statuses=(),
                        ready_for_install=False,
                        rejection_reason=reason,
                    )
                    candidate_ready_wall_time_s = worker_result_wall_time_s
                worker_resolution_wall_time_s = (
                    time.perf_counter() - worker_resolution_started_at
                )

            if install_step is not None and index - install_step >= post_install_steps:
                break
            if (
                future_consumed
                and pending_controller is None
                and install_step is None
                and submission_sim_time_s is not None
                and simulation_time_s - submission_sim_time_s
                >= ONLINE_POST_INSTALL_DURATION_S
            ):
                break

            validating_candidate = (
                pending_controller is not None and not candidate_deferred_for_tick
            )
            controller_for_step = (
                pending_controller if validating_candidate else active_controller
            )
            reference_for_step = (
                pending_reference if validating_candidate else active_reference
            )
            warm_start_for_step = (
                template_hover_warm.warm_start
                if validating_candidate
                else active_warm_start
            )
            saved_active_warm_start = active_warm_start
            validation_started_at = time.perf_counter()
            result = controller_for_step.solve(
                jnp.asarray(states[-1]),
                reference_for_step,
                jnp.asarray(previous),
                applied_command=jnp.asarray(sample.applied_motor_thrust_fraction),
                warm_start=warm_start_for_step,
                deadline_s=sample_period_s,
            )
            command_mode = active_mode
            if validating_candidate:
                validation_wall_time_s = time.perf_counter() - validation_started_at
                assert worker_update is not None
                if result.command_usable:
                    active_controller = controller_for_step
                    active_reference = reference_for_step
                    active_warm_start = result.warm_start
                    active_mode = "adapted"
                    command_mode = "adapted"
                    install_step = index
                    install_sim_time_s = simulation_time_s
                    install_wall_time_s = time.perf_counter() - control_loop_started_at
                    candidate_ready_wall_time_s = install_wall_time_s
                    prepared = _PreparedOnlineController(
                        update_report=worker_update.update_report,
                        update_wall_time_s=worker_update.update_wall_time_s,
                        controller_validation_wall_time_s=validation_wall_time_s,
                        total_wall_time_s=(
                            worker_update.update_wall_time_s + validation_wall_time_s
                        ),
                        validation_statuses=(result.status.value,),
                        ready_for_install=True,
                        rejection_reason=None,
                    )
                else:
                    active_warm_start = saved_active_warm_start
                    command_mode = "candidate_rejected"
                    reason = (
                        "first adapted solve was not command-usable: "
                        f"{result.status.value}: {result.message}"
                    )
                    prepared = _PreparedOnlineController(
                        update_report=worker_update.update_report,
                        update_wall_time_s=worker_update.update_wall_time_s,
                        controller_validation_wall_time_s=validation_wall_time_s,
                        total_wall_time_s=(
                            worker_update.update_wall_time_s + validation_wall_time_s
                        ),
                        validation_statuses=(result.status.value,),
                        ready_for_install=False,
                        rejection_reason=reason,
                    )
                    candidate_ready_wall_time_s = (
                        time.perf_counter() - control_loop_started_at
                    )
                pending_controller = None
                pending_reference = None
            else:
                active_warm_start = result.warm_start
            command_generated_at_s = time.perf_counter()
            fault_this_step = (
                install_step is not None
                and not supervisor_fault_injected
                and index - install_step == 10
            )
            supervisor_command_timestamp_s = command_generated_at_s
            if fault_this_step:
                supervisor_command_timestamp_s -= (
                    2.0 * supervisor_config.maximum_command_age_s
                )
            supervisor_started_at = time.perf_counter()
            supervised = supervisor.supervise(
                state=states[-1],
                state_received_at_s=step_started_at,
                candidate_command=result.command,
                command_generated_at_s=supervisor_command_timestamp_s,
                now_s=time.perf_counter(),
                controller_command_usable=result.command_usable,
                previous_applied_command=sample.applied_motor_thrust_fraction,
            )
            supervisor_decision_times.append(
                time.perf_counter() - supervisor_started_at
            )
            supervisor_modes.append(supervised.mode.value)
            supervisor_reasons.extend(reason.value for reason in supervised.reasons)
            supervisor_intervention_count += int(supervised.intervened)
            if fault_this_step:
                supervisor_fault_injected = True
                supervisor_fault_step = index
                supervisor_fault_response = supervised.to_dict()
            command = np.asarray(supervised.command)
            sample = plant.step(command)
            controls.append(command)
            states.append(np.asarray(sample.state))
            applied.append(np.asarray(sample.applied_motor_thrust_fraction))
            state_validity.append(
                float(
                    np.max(
                        np.asarray(
                            stale_controller.model.validity_utilization(
                                jnp.asarray(sample.state)
                            )
                        )
                    )
                )
            )
            control_modes.append(command_mode)
            previous = command
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

            eligible_start = _eligible_recovery_evidence_start(
                state_validity,
                evidence_steps,
            )
            if candidate_future is None and eligible_start is not None:
                recent_validity = state_validity[eligible_start:]
                evidence = _online_recovery_trajectory(
                    belief,
                    states[eligible_start:],
                    controls[eligible_start:],
                    applied[eligible_start:],
                )
                submission_wall_time = time.perf_counter()
                submission_sim_time_s = len(controls) * sample_period_s
                evidence_start_sim_time_s = eligible_start * sample_period_s
                evidence_maximum_validity = max(recent_validity)
                candidate_future = executor.submit(
                    _run_isolated_belief_update,
                    belief,
                    evidence,
                )
                worker_suspended = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=True,
                )
                worker_suspend_count += int(worker_suspended)

            step_elapsed_s = time.perf_counter() - step_started_at
            outer_step_times.append(step_elapsed_s)
            if step_elapsed_s > sample_period_s:
                outer_deadline_miss_count += 1
                outer_deadline_miss_steps.append(index)
                outer_deadline_miss_details.append(
                    {
                        "step": index,
                        "outer_step_time_s": step_elapsed_s,
                        "solver_reported_time_s": result.diagnostics.solve_time_s,
                        "solver_status": result.status.value,
                        "control_mode": command_mode,
                        "worker_suspended": worker_suspended,
                        "worker_resolution_wall_time_s": (
                            worker_resolution_wall_time_s
                        ),
                        "validating_candidate": validating_candidate,
                    }
                )

            command_deadline = control_loop_started_at + (
                len(controls) * sample_period_s
            )
            absolute_lateness_s = max(
                0.0,
                time.perf_counter() - command_deadline,
            )
            absolute_completion_lateness.append(absolute_lateness_s)
            if absolute_lateness_s > 0.0:
                absolute_deadline_miss_steps.append(index)

            if worker_suspended:
                resumed = _set_process_suspended(
                    worker_prewarm.process_id,
                    suspended=False,
                )
                worker_resume_count += int(resumed)

            target_wall_time = control_loop_started_at + (
                len(controls) * sample_period_s
            )
            remaining_s = target_wall_time - time.perf_counter()
            if remaining_s > 0.0:
                time.sleep(remaining_s)
                loop_lag.append(0.0)
            else:
                loop_lag.append(-remaining_s)
    finally:
        control_loop_ended_at = time.perf_counter()
        plant.close()
        _set_process_suspended(worker_prewarm.process_id, suspended=False)
        executor.shutdown(wait=True)

    control_loop_wall_time_s = control_loop_ended_at - control_loop_started_at
    completed_after_control_loop = False
    if candidate_future is not None and not future_consumed:
        completed_after_control_loop = True
        future_consumed = True
        try:
            worker_update = candidate_future.result()
            reason = "isolated update completed after the fixed control window"
            prepared = _PreparedOnlineController(
                update_report=worker_update.update_report,
                update_wall_time_s=worker_update.update_wall_time_s,
                controller_validation_wall_time_s=None,
                total_wall_time_s=worker_update.update_wall_time_s,
                validation_statuses=(),
                ready_for_install=False,
                rejection_reason=reason,
            )
        except Exception as error:
            worker_error = f"{type(error).__name__}: {error}"
            reason = f"isolated update failed: {worker_error}"
            prepared = _PreparedOnlineController(
                update_report={"applied": False, "reason": reason},
                update_wall_time_s=0.0,
                controller_validation_wall_time_s=None,
                total_wall_time_s=0.0,
                validation_statuses=(),
                ready_for_install=False,
                rejection_reason=reason,
            )
    if prepared is None:
        reason = (
            "no contiguous evidence window remained inside the learned validity "
            "envelope"
        )
        prepared = _PreparedOnlineController(
            update_report={"applied": False, "reason": reason},
            update_wall_time_s=0.0,
            controller_validation_wall_time_s=None,
            total_wall_time_s=0.0,
            validation_statuses=(),
            ready_for_install=False,
            rejection_reason=reason,
        )

    states_array = np.asarray(states)
    commands_array = np.asarray(controls)
    references = np.repeat(reference_state[None, :], len(states_array), axis=0)
    errors = np.asarray(
        jax.vmap(rigid_body_local_error)(
            jnp.asarray(references),
            jnp.asarray(states_array),
        )
    )
    scale = np.asarray(stale_controller.tolerances.local_state_scale)
    normalized = errors / scale
    pre_install_stop = len(states_array) if install_step is None else install_step + 1
    pre_install_normalized = normalized[:pre_install_stop]
    post_install_normalized = (
        np.empty((0, normalized.shape[1]))
        if install_step is None
        else normalized[install_step:]
    )
    actual_validity = np.asarray(state_validity)
    minimum = np.asarray(stale_controller.model.command_minimum)
    maximum = np.asarray(stale_controller.model.command_maximum)
    command_violation = max(
        float(np.max(minimum - commands_array)),
        float(np.max(commands_array - maximum)),
        0.0,
    )
    submission_to_ready_s = None
    if submission_wall_time is not None:
        if candidate_ready_wall_time_s is not None:
            submission_to_ready_s = (
                control_loop_started_at
                + candidate_ready_wall_time_s
                - submission_wall_time
            )
    submission_to_install_s = (
        None
        if submission_wall_time is None or install_wall_time_s is None
        else (control_loop_started_at + install_wall_time_s - submission_wall_time)
    )
    stale_after_submission_count = 0
    if submission_sim_time_s is not None:
        submission_step = round(submission_sim_time_s / sample_period_s)
        stale_after_submission_count = sum(
            mode == "stale" for mode in control_modes[submission_step:]
        )
    return {
        "condition": "online_stale_to_adapted",
        "evidence": {
            "duration_s": ONLINE_EVIDENCE_DURATION_S,
            "sample_count": evidence_steps + 1,
            "submitted": submission_wall_time is not None,
            "eligibility_rule": (
                "first contiguous window with validity utilization <= 1.0"
            ),
            "start_sim_time_s": evidence_start_sim_time_s,
            "maximum_validity_utilization": evidence_maximum_validity,
            "excluded_prefix_state_count": (
                None
                if evidence_start_sim_time_s is None
                else round(evidence_start_sim_time_s / sample_period_s)
            ),
            "excluded_prefix_maximum_validity_utilization": (
                None
                if evidence_start_sim_time_s in (None, 0.0)
                else max(
                    state_validity[: round(evidence_start_sim_time_s / sample_period_s)]
                )
            ),
            "source_group": "unknown-target-online-recovery",
            "collected_under_controller": "stale",
            "hidden_target_configuration_supplied": False,
            "applied_actuator_state_retained": True,
        },
        "update": prepared.update_report,
        "candidate": {
            "ready_for_install": prepared.ready_for_install,
            "shared_precompiled_parameterized_kernels": True,
            "isolated_adaptation_process": True,
            "rejection_reason": prepared.rejection_reason,
            "validation_statuses": list(prepared.validation_statuses),
            "completed_after_control_loop": completed_after_control_loop,
            "worker_process_id": (
                None if worker_update is None else worker_update.process_id
            ),
            "worker_process_niceness": (
                None if worker_update is None else worker_update.process_niceness
            ),
            "worker_error": worker_error,
        },
        "supervisor": {
            "included": True,
            "independent_of_fitted_dynamics": True,
            "configuration": supervisor_config.to_dict(),
            "mode_counts": dict(sorted(Counter(supervisor_modes).items())),
            "reason_counts": dict(sorted(Counter(supervisor_reasons).items())),
            "intervention_count": supervisor_intervention_count,
            "maximum_decision_wall_time_s": max(
                supervisor_decision_times,
                default=0.0,
            ),
            "fault_injection": {
                "injected": supervisor_fault_injected,
                "type": "stale_adapted_command_timestamp",
                "step": supervisor_fault_step,
                "expected_mode": SupervisorMode.RATE_ARREST.value,
                "expected_reason": SupervisorReason.COMMAND_STALE.value,
                "response": supervisor_fault_response,
                "handled": bool(
                    supervisor_fault_response is not None
                    and supervisor_fault_response["mode"]
                    == SupervisorMode.RATE_ARREST.value
                    and SupervisorReason.COMMAND_STALE.value
                    in supervisor_fault_response["reasons"]
                ),
            },
        },
        "isolation": {
            "start_method": "spawn",
            "numerical_payload_only": True,
            "control_process_id": os.getpid(),
            "worker_process_id": worker_prewarm.process_id,
            "separate_process": worker_prewarm.process_id != os.getpid(),
            "worker_process_niceness": worker_prewarm.process_niceness,
            "lower_scheduling_priority_requested": True,
            "temporal_partition_available": temporal_partition_available,
            "worker_suspend_count": worker_suspend_count,
            "worker_resume_count": worker_resume_count,
            "worker_runs_only_in_control_slack": temporal_partition_available,
            "worker_environment": worker_environment,
            "prewarm_known_fleet_only": True,
            "prewarm_hidden_target_configuration_supplied": False,
            "prewarm_update": worker_prewarm.update_report,
        },
        "handoff": {
            "atomic_install_at_control_boundary": install_step is not None,
            "submission_sim_time_s": submission_sim_time_s,
            "candidate_resolved_sim_time_s": candidate_resolved_sim_time_s,
            "install_sim_time_s": install_sim_time_s,
            "install_step": install_step,
            "stale_steps_after_submission": stale_after_submission_count,
            "control_continued_during_candidate_preparation": (
                stale_after_submission_count > 0
            ),
        },
        "timing": {
            "control_period_s": sample_period_s,
            "stale_controller_prewarm_wall_time_s": stale_prewarm_wall_time_s,
            "candidate_template_prewarm_wall_time_s": (
                candidate_template_prewarm_wall_time_s
            ),
            "worker_process_prewarm_wall_time_s": worker_prewarm_wall_time_s,
            "worker_prewarm_update_wall_time_s": (worker_prewarm.update_wall_time_s),
            "worker_prewarm_update_cpu_time_s": (worker_prewarm.update_cpu_time_s),
            "integrated_control_plant_prewarm_wall_time_s": (
                integrated_prewarm_wall_time_s
            ),
            "belief_update_wall_time_s": prepared.update_wall_time_s,
            "belief_update_cpu_time_s": (
                None if worker_update is None else worker_update.update_cpu_time_s
            ),
            "candidate_first_solve_validation_wall_time_s": (
                prepared.controller_validation_wall_time_s
            ),
            "submission_to_worker_result_wall_time_s": (
                None
                if submission_wall_time is None or worker_result_wall_time_s is None
                else (
                    control_loop_started_at
                    + worker_result_wall_time_s
                    - submission_wall_time
                )
            ),
            "submission_to_candidate_ready_wall_time_s": submission_to_ready_s,
            "submission_to_install_wall_time_s": submission_to_install_s,
            "control_loop_wall_time_s": control_loop_wall_time_s,
            "outer_control_step_median_s": float(np.median(outer_step_times)),
            "outer_control_step_p90_s": float(np.quantile(outer_step_times, 0.9)),
            "outer_control_step_maximum_s": max(outer_step_times),
            "solver_reported_median_s": float(np.median(solve_times)),
            "solver_reported_p90_s": float(np.quantile(solve_times, 0.9)),
            "solver_reported_maximum_s": max(solve_times),
            "outer_deadline_miss_count": outer_deadline_miss_count,
            "outer_deadline_miss_steps": outer_deadline_miss_steps,
            "outer_deadline_miss_details": outer_deadline_miss_details,
            "absolute_deadline_miss_count": len(absolute_deadline_miss_steps),
            "absolute_deadline_miss_steps": absolute_deadline_miss_steps,
            "maximum_absolute_completion_lateness_s": max(
                absolute_completion_lateness,
                default=0.0,
            ),
            "maximum_schedule_lag_s": max(loop_lag, default=0.0),
        },
        "control_mode_counts": dict(sorted(Counter(control_modes).items())),
        "solve_status_counts": dict(sorted(Counter(solve_statuses).items())),
        "solve_message_counts": dict(sorted(Counter(solve_messages).items())),
        "support_filter_mode_counts": dict(sorted(Counter(support_modes).items())),
        "support_filter_applied_count": support_applied,
        "fallback_count": fallback_count,
        "nonfinite_diagnostic_count": nonfinite_diagnostic_count,
        "normalized_tracking_rms": _normalized_rms(normalized),
        "pre_install_normalized_tracking_rms": _normalized_rms(pre_install_normalized),
        "post_install_normalized_tracking_rms": _normalized_rms(
            post_install_normalized
        ),
        "pre_install_normalized_attitude_rate_rms": _normalized_rms(
            pre_install_normalized[:, 6:12]
        ),
        "post_install_normalized_attitude_rate_rms": _normalized_rms(
            post_install_normalized[:, 6:12]
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
        "finite": bool(
            np.all(np.isfinite(states_array))
            and np.all(np.isfinite(commands_array))
            and np.all(np.isfinite(normalized))
        ),
    }


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
    cold = controller.solve(
        jnp.asarray(reference_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(cold.command)
    warm = controller.solve(
        jnp.asarray(reference_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=cold.warm_start,
    )
    jax.block_until_ready(warm.command)
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
    cold = controller.solve(
        jnp.asarray(initial_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
    )
    jax.block_until_ready(cold.command)
    warm = controller.solve(
        jnp.asarray(initial_state),
        reference,
        jnp.asarray(hover),
        applied_command=jnp.asarray(hover),
        warm_start=cold.warm_start,
    )
    jax.block_until_ready(warm.command)
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


def main() -> None:
    # Set before any lazy `import crazyflow` reaches its own SciPy-array-API guard.
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--plant-contract-only",
        action="store_true",
        help="write only the fast frame/actuation/telemetry qualification slice",
    )
    args = parser.parse_args()
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
