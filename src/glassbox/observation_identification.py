"""Observation-first initialization for structured multirotor dynamics.

This module intentionally has no configuration surface.  It implements one
auditable policy for extracting thrust, drag, actuator lag, and rotational
response from typed state-aligned sensor outputs.  Multi-step rollout fitting
remains the final objective; this stage only supplies a data-derived starting
point and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from glassbox.data import (
    NORMALIZED_MOTOR_COMMAND_SEMANTICS,
    PHYSICAL_MOTOR_THRUST_SEMANTICS,
    Trajectory,
)
from glassbox.dynamics import (
    MAX_ANGULAR_CONTROL_CROSS_COUPLING,
    MOTOR_MIXER,
    DynamicsParams,
)
from glassbox.evaluation import parameter_dict
from glassbox.synthetic import initial_parameter_guess


MAX_ALIGNMENT_LAG_S = 0.20
MINIMUM_ALIGNMENT_CORRELATION = 0.30
MINIMUM_ALIGNMENT_GAIN = 0.05
MEMORYLESS_TIME_CONSTANT_S = 1e-4
TRAIN_FRACTION = 0.70
MAX_NUISANCE_ACCELEROMETER_BIAS_M_S2 = 0.5


@dataclass(frozen=True)
class AlignmentDiagnostic:
    """Bounded actuator-to-accelerometer timing diagnostic."""

    lag_steps: int
    lag_s: float
    correlation: float
    zero_lag_correlation: float
    correlation_gain: float
    alignment_applied: bool
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_steps": self.lag_steps,
            "lag_s": self.lag_s,
            "correlation": self.correlation,
            "zero_lag_correlation": self.zero_lag_correlation,
            "correlation_gain": self.correlation_gain,
            "alignment_applied": self.alignment_applied,
            "policy": self.policy,
            "maximum_lag_s": MAX_ALIGNMENT_LAG_S,
            "minimum_correlation": MINIMUM_ALIGNMENT_CORRELATION,
            "minimum_correlation_gain": MINIMUM_ALIGNMENT_GAIN,
        }


@dataclass(frozen=True)
class ObservationFitResult:
    """Structured initializer and the evidence used to produce it."""

    params: DynamicsParams
    report: dict[str, Any]


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3:
        return 0.0
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(left @ right / denominator)


def _specific_force(trajectory: Trajectory) -> np.ndarray:
    roles = trajectory.spec.observation_roles
    required = tuple(f"specific_force_{axis}" for axis in "xyz")
    missing = [role for role in required if role not in roles]
    if missing:
        raise ValueError(
            "observation-first identification requires typed specific force "
            f"channels; missing {', '.join(missing)}"
        )
    return trajectory.observations[
        :, [roles.index(role) for role in required]
    ]


def actuator_observation_alignment(
    trajectory: Trajectory,
) -> AlignmentDiagnostic:
    """Estimate a bounded positive lag between collective input and force.

    A positive lag means the accelerometer response follows the logged control.
    The estimate is applied only to measured physical actuator-state inputs.
    Normalized commands retain the lag because it can be real actuator physics.
    """

    specific_force = _specific_force(trajectory)[:-1, 2]
    collective = np.sum(trajectory.controls, axis=1)
    maximum_steps = min(
        max(0, int(np.floor(MAX_ALIGNMENT_LAG_S / trajectory.nominal_dt_s))),
        max(0, len(collective) // 4),
    )
    correlations = np.asarray(
        [
            _correlation(
                collective[: len(collective) - lag] if lag else collective,
                specific_force[lag:],
            )
            for lag in range(maximum_steps + 1)
        ]
    )
    lag_steps = int(np.argmax(correlations))
    correlation = float(correlations[lag_steps])
    zero_lag = float(correlations[0])
    gain = correlation - zero_lag
    semantics = set(trajectory.spec.control_semantics)
    physical_actuator_state = semantics.issubset(
        PHYSICAL_MOTOR_THRUST_SEMANTICS
    )
    apply_alignment = bool(
        physical_actuator_state
        and lag_steps > 0
        and correlation >= MINIMUM_ALIGNMENT_CORRELATION
        and gain >= MINIMUM_ALIGNMENT_GAIN
    )
    policy = (
        "applied_to_measured_actuator_state"
        if apply_alignment
        else "retained_as_physical_response_for_command_input"
        if not physical_actuator_state
        else "no_confident_transport_alignment"
    )
    return AlignmentDiagnostic(
        lag_steps=lag_steps,
        lag_s=lag_steps * trajectory.nominal_dt_s,
        correlation=correlation,
        zero_lag_correlation=zero_lag,
        correlation_gain=gain,
        alignment_applied=apply_alignment,
        policy=policy,
    )


def _rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = quaternion_wxyz / np.linalg.norm(
        quaternion_wxyz, axis=1, keepdims=True
    )
    w, x, y, z = quaternion.T
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((-1, 3, 3))


def _body_velocity(trajectory: Trajectory) -> np.ndarray:
    rotations = _rotation_matrices(trajectory.states[:-1, 6:10])
    wind_world = np.zeros((len(trajectory.controls), 3), dtype=np.float64)
    for axis, suffix in enumerate(("wind_north", "wind_west", "wind_up")):
        matching_roles = [
            role
            for role in trajectory.spec.exogenous_roles
            if role == suffix or role.endswith(f"_{suffix}")
        ]
        if matching_roles:
            role = matching_roles[0]
            wind_world[:, axis] = trajectory.exogenous[
                :-1, trajectory.spec.exogenous_roles.index(role)
            ]
    return np.einsum(
        "nij,nj->ni",
        rotations.transpose(0, 2, 1),
        trajectory.states[:-1, 3:6] - wind_world,
    )


def _first_order_response(
    commands: np.ndarray, dt_s: float, time_constant_s: float
) -> np.ndarray:
    commands = np.asarray(commands, dtype=np.float64)
    if time_constant_s <= MEMORYLESS_TIME_CONSTANT_S * 1.0001:
        return commands.copy()
    response = np.empty_like(commands)
    response[0] = commands[0]
    decay = float(np.exp(-dt_s / time_constant_s))
    for index in range(1, len(commands)):
        response[index] = commands[index - 1] + (
            response[index - 1] - commands[index - 1]
        ) * decay
    return response


def _split_mask(length: int) -> np.ndarray:
    split = min(max(int(round(TRAIN_FRACTION * length)), 1), length - 1)
    mask = np.zeros(length, dtype=bool)
    mask[:split] = True
    return mask


def _ridge_solution(
    design: np.ndarray, target: np.ndarray, ridge: float = 1e-8
) -> np.ndarray:
    gram = design.T @ design
    scale = max(float(np.trace(gram) / max(1, len(gram))), 1.0)
    return np.linalg.solve(
        gram + ridge * scale * np.eye(design.shape[1]),
        design.T @ target,
    )


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def _force_design(
    collective: np.ndarray, body_velocity: np.ndarray
) -> np.ndarray:
    count = len(collective)
    design = np.zeros((count * 3, 5), dtype=np.float64)
    design[2::3, 0] = collective
    design[:, 1] = -body_velocity.reshape(-1)
    for axis in range(3):
        design[axis::3, 2 + axis] = 1.0
    return design


def _force_solution(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit force coefficients with a bounded sensor-bias nuisance term."""

    coefficients = _ridge_solution(design, target)
    bias = np.clip(
        coefficients[2:5],
        -MAX_NUISANCE_ACCELEROMETER_BIAS_M_S2,
        MAX_NUISANCE_ACCELEROMETER_BIAS_M_S2,
    )
    bias_prediction = design[:, 2:5] @ bias
    physical = _ridge_solution(design[:, 0:2], target - bias_prediction)
    physical = np.maximum(physical, 1e-6)
    return np.concatenate((physical, bias))


def _candidate_time_constants() -> npt.NDArray[np.float64]:
    return np.concatenate(
        (
            np.asarray([MEMORYLESS_TIME_CONSTANT_S]),
            np.geomspace(0.003, 0.25, 24),
        )
    )


def _candidate_offsets(command_semantics: bool) -> npt.NDArray[np.float64]:
    # A shared offset is strongly confounded with thrust scale and vertical
    # accelerometer bias.  Keep the opinionated reference at zero here; rollout
    # model selection remains the place for an explicit offset experiment.
    del command_semantics
    return np.asarray([0.0])


def _aligned_indices(
    trajectory: Trajectory, alignment: AlignmentDiagnostic
) -> tuple[np.ndarray, np.ndarray]:
    count = len(trajectory.controls)
    lag = alignment.lag_steps if alignment.alignment_applied else 0
    return np.arange(count - lag), np.arange(lag, count)


def _angular_acceleration(trajectory: Trajectory) -> tuple[np.ndarray, str]:
    roles = trajectory.spec.observation_roles
    required = tuple(f"angular_acceleration_{axis}" for axis in "xyz")
    if all(role in roles for role in required):
        return (
            trajectory.observations[
                :-1, [roles.index(role) for role in required]
            ],
            "typed_sensor_output",
        )
    angular_velocity = trajectory.states[:, 10:13]
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    padded = np.pad(angular_velocity, ((2, 2), (0, 0)), mode="edge")
    smoothed = np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)]
    )
    return (
        np.gradient(smoothed, trajectory.nominal_dt_s, axis=0)[:-1],
        "smoothed_state_derivative",
    )


def _validate_trajectories(trajectories: Sequence[Trajectory]) -> bool:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    semantics = set(trajectories[0].spec.control_semantics)
    command_semantics = semantics.issubset(NORMALIZED_MOTOR_COMMAND_SEMANTICS)
    physical_semantics = semantics.issubset(PHYSICAL_MOTOR_THRUST_SEMANTICS)
    if not (command_semantics or physical_semantics):
        raise ValueError(
            "observation-first multirotor identification requires normalized "
            "motor commands or measured squared rotor-speed ratios"
        )
    for trajectory in trajectories:
        if trajectory.spec.vehicle.family != "multirotor":
            raise ValueError("observation-first fitting is currently multirotor-only")
        if trajectory.control_size != 4:
            raise ValueError("observation-first fitting requires four motors")
        if set(trajectory.spec.control_semantics) != semantics:
            raise ValueError("all trajectories must share control semantics")
        if len(trajectory.controls) < 10:
            raise ValueError("observation-first fitting needs at least ten intervals")
        _specific_force(trajectory)
    return command_semantics


def fit_multirotor_observations(
    trajectories: Sequence[Trajectory],
) -> ObservationFitResult:
    """Fit a structured initializer from accelerometer and rate outputs.

    Candidate actuator and rotational time constants are selected against the
    chronological final 30% of every supplied trajectory.  Coefficients are
    estimated only on the leading 70%, making the reported score useful as a
    guard against promoting an attractive in-sample explanation.
    """

    command_semantics = _validate_trajectories(trajectories)
    alignments = tuple(
        actuator_observation_alignment(trajectory)
        for trajectory in trajectories
    )
    force_records = []
    for trajectory, alignment in zip(trajectories, alignments):
        control_indices, observation_indices = _aligned_indices(
            trajectory, alignment
        )
        force_records.append(
            {
                "trajectory": trajectory,
                "control_indices": control_indices,
                "observation_indices": observation_indices,
                "specific_force": _specific_force(trajectory)[
                    observation_indices
                ],
                "body_velocity": _body_velocity(trajectory)[
                    observation_indices
                ],
                "train": _split_mask(len(control_indices)),
            }
        )

    time_constants = (
        _candidate_time_constants()
        if command_semantics
        else np.asarray([MEMORYLESS_TIME_CONSTANT_S])
    )
    candidates: list[dict[str, Any]] = []
    applied_cache: dict[tuple[int, float], np.ndarray] = {}
    for time_constant in time_constants:
        for record_index, record in enumerate(force_records):
            trajectory = record["trajectory"]
            applied_cache[(record_index, float(time_constant))] = (
                _first_order_response(
                    trajectory.controls,
                    trajectory.nominal_dt_s,
                    float(time_constant),
                )
            )
        for offset in _candidate_offsets(command_semantics):
            designs = []
            targets = []
            masks = []
            for record_index, record in enumerate(force_records):
                indices = record["control_indices"]
                applied = applied_cache[(record_index, float(time_constant))][
                    indices
                ]
                collective = np.sum(np.maximum(applied - offset, 0.0), axis=1)
                designs.append(
                    _force_design(collective, record["body_velocity"])
                )
                targets.append(record["specific_force"].reshape(-1))
                masks.append(np.repeat(record["train"], 3))
            design = np.concatenate(designs)
            target = np.concatenate(targets)
            train = np.concatenate(masks)
            coefficients = _force_solution(design[train], target[train])
            validation_rmse = _rmse(
                target[~train], design[~train] @ coefficients
            )
            candidates.append(
                {
                    "time_constant_s": float(time_constant),
                    "offset": float(offset),
                    "coefficients": coefficients,
                    "validation_rmse": validation_rmse,
                    "design": design,
                    "target": target,
                    "train": train,
                }
            )
    raw_force_fit = min(
        candidates,
        key=lambda item: (
            item["validation_rmse"],
            item["time_constant_s"],
            abs(item["offset"]),
        ),
    )
    force_fit = raw_force_fit
    time_constant_boundary_fallback = False
    if command_semantics and np.isclose(
        raw_force_fit["time_constant_s"], time_constants[-1]
    ):
        correlated_lags = [
            item.lag_s
            for item in alignments
            if item.lag_steps > 0
            and item.correlation >= MINIMUM_ALIGNMENT_CORRELATION
        ]
        fallback_time_constant = (
            float(np.median(correlated_lags))
            if correlated_lags
            else float(initial_parameter_guess().physical()["motor_time_constant"])
        )
        force_fit = min(
            candidates,
            key=lambda item: (
                abs(item["time_constant_s"] - fallback_time_constant),
                item["validation_rmse"],
            ),
        )
        time_constant_boundary_fallback = True
    selected_motor_time_constant = float(force_fit["time_constant_s"])
    selected_offset = float(force_fit["offset"])

    angular_records = []
    angular_sources = set()
    mixer = np.asarray(MOTOR_MIXER, dtype=np.float64)
    for record_index, record in enumerate(force_records):
        trajectory = record["trajectory"]
        control_indices = record["control_indices"]
        observation_indices = record["observation_indices"]
        angular_acceleration, source = _angular_acceleration(trajectory)
        angular_sources.add(source)
        applied = applied_cache[
            (record_index, selected_motor_time_constant)
        ]
        applied = np.maximum(applied - selected_offset, 0.0)
        angular_records.append(
            {
                "mixer": applied @ mixer.T,
                "control_indices": control_indices,
                "target": angular_acceleration[observation_indices],
                "omega": trajectory.states[observation_indices, 10:13],
                "train": record["train"],
                "dt_s": trajectory.nominal_dt_s,
            }
        )

    angular_matrix = np.zeros((3, 3), dtype=np.float64)
    angular_drag = np.empty(3, dtype=np.float64)
    angular_bias = np.empty(3, dtype=np.float64)
    angular_response = np.empty(3, dtype=np.float64)
    angular_reports = []
    initial = parameter_dict(initial_parameter_guess())
    for axis in range(3):
        axis_candidates = []
        for time_constant in _candidate_time_constants():
            designs = []
            targets = []
            masks = []
            for record in angular_records:
                response = _first_order_response(
                    record["mixer"], record["dt_s"], float(time_constant)
                )[record["control_indices"]]
                design = np.column_stack(
                    (
                        response,
                        -record["omega"][:, axis],
                        np.ones(len(response)),
                    )
                )
                designs.append(design)
                targets.append(record["target"][:, axis])
                masks.append(record["train"])
            design = np.concatenate(designs)
            target = np.concatenate(targets)
            train = np.concatenate(masks)
            coefficients = _ridge_solution(design[train], target[train], 1e-6)
            coefficients[3] = max(coefficients[3], 1e-6)
            axis_candidates.append(
                {
                    "time_constant_s": float(time_constant),
                    "coefficients": coefficients,
                    "validation_rmse": _rmse(
                        target[~train], design[~train] @ coefficients
                    ),
                    "target": target,
                    "design": design,
                    "train": train,
                }
            )
        selected = min(
            axis_candidates,
            key=lambda item: (item["validation_rmse"], item["time_constant_s"]),
        )
        coefficients = selected["coefficients"]
        candidate_coefficients = coefficients.copy()
        candidate_response_time_constant = float(
            selected["time_constant_s"]
        )
        validation_constant_baseline_rmse = _rmse(
            selected["target"][~selected["train"]],
            np.full(
                np.sum(~selected["train"]),
                np.mean(selected["target"][selected["train"]]),
            ),
        )
        accepted = bool(
            selected["validation_rmse"]
            <= 0.95 * validation_constant_baseline_rmse
            and not np.isclose(
                selected["time_constant_s"],
                _candidate_time_constants()[-1],
            )
        )
        diagonal = float(coefficients[axis])
        if diagonal <= 1e-6 or not accepted:
            diagonal = float(initial["angular_accel"][axis])
            coefficients[:3] = 0.0
            coefficients[axis] = diagonal
            coefficients[3] = float(initial["angular_drag"][axis])
            coefficients[4] = 0.0
            selected["time_constant_s"] = float(
                initial["angular_response_time_constant"][axis]
            )
        angular_matrix[axis] = coefficients[:3]
        angular_drag[axis] = float(coefficients[3])
        angular_bias[axis] = float(coefficients[4])
        angular_response[axis] = float(selected["time_constant_s"])
        target = selected["target"]
        design = selected["design"]
        train = selected["train"]
        angular_reports.append(
            {
                "axis": "xyz"[axis],
                "response_time_constant_s": float(selected["time_constant_s"]),
                "candidate_response_time_constant_s": (
                    candidate_response_time_constant
                ),
                "passed_sensor_residual_gate": accepted,
                "candidate_training_rmse_rad_s2": _rmse(
                    target[train], design[train] @ candidate_coefficients
                ),
                "candidate_validation_rmse_rad_s2": float(
                    selected["validation_rmse"]
                ),
                "validation_constant_baseline_rmse_rad_s2": (
                    validation_constant_baseline_rmse
                ),
                "candidate_nuisance_bias_rad_s2": float(
                    candidate_coefficients[4]
                ),
            }
        )

    angular_accel = np.maximum(np.diag(angular_matrix), 1e-6)
    cross_coupling = angular_matrix / angular_accel[:, None]
    np.fill_diagonal(cross_coupling, 0.0)
    cross_limit = MAX_ANGULAR_CONTROL_CROSS_COUPLING * 0.98
    cross_coupling = np.clip(cross_coupling, -cross_limit, cross_limit)
    params = DynamicsParams.from_physical(
        thrust_accel=float(force_fit["coefficients"][0]),
        thrust_command_offset=selected_offset,
        angular_accel=tuple(float(value) for value in angular_accel),
        linear_drag=float(force_fit["coefficients"][1]),
        angular_drag=tuple(float(value) for value in angular_drag),
        motor_time_constant=selected_motor_time_constant,
        angular_response_time_constant=tuple(
            float(value) for value in angular_response
        ),
        angular_control_cross_coupling=tuple(
            tuple(float(value) for value in row) for row in cross_coupling
        ),
    )

    design = force_fit["design"]
    target = force_fit["target"]
    train = force_fit["train"]
    coefficients = force_fit["coefficients"]
    report = {
        "policy": "typed_observation_identification_v1",
        "trajectory_count": len(trajectories),
        "control_interpretation": (
            "normalized_command" if command_semantics else "measured_thrust_proxy"
        ),
        "alignment": [item.to_dict() for item in alignments],
        "force_fit": {
            "motor_time_constant_s": selected_motor_time_constant,
            "unconstrained_selected_motor_time_constant_s": float(
                raw_force_fit["time_constant_s"]
            ),
            "time_constant_boundary_fallback": time_constant_boundary_fallback,
            "thrust_command_offset": selected_offset,
            "thrust_accel": float(coefficients[0]),
            "linear_drag": float(coefficients[1]),
            "nuisance_accelerometer_bias_m_s2": coefficients[2:5].tolist(),
            "training_rmse_m_s2": _rmse(
                target[train], design[train] @ coefficients
            ),
            "validation_rmse_m_s2": float(force_fit["validation_rmse"]),
            "validation_constant_baseline_rmse_m_s2": _rmse(
                target[~train],
                np.full(np.sum(~train), np.mean(target[train])),
            ),
            "candidate_time_constant_count": len(time_constants),
            "candidate_offset_count": len(_candidate_offsets(command_semantics)),
            "passed_sensor_residual_gate": bool(
                force_fit["validation_rmse"]
                <= 0.8
                * _rmse(
                    target[~train],
                    np.full(np.sum(~train), np.mean(target[train])),
                )
            ),
        },
        "angular_fit": {
            "target_source": sorted(angular_sources),
            "axes": angular_reports,
            "unclipped_control_matrix": angular_matrix.tolist(),
            "nuisance_bias_rad_s2": angular_bias.tolist(),
        },
        "parameters": parameter_dict(params),
        "selection_data": {
            "chronological_training_fraction": TRAIN_FRACTION,
            "chronological_validation_fraction": 1.0 - TRAIN_FRACTION,
            "test_split_used": any(
                trajectory.labels.get("benchmark_split") == "test"
                for trajectory in trajectories
            ),
        },
    }
    return ObservationFitResult(params=params, report=report)
