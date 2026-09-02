"""Fly the certified snapshot or the working belief, and compare the two.

The online throw diagnostic keeps a working belief that is updated after every
actuated interval, but flies a frozen snapshot that a prequential transaction
only replaces when a candidate out-predicts the incumbent.  This study runs the
same closed loop twice per case, once in each :attr:`RecursiveBootstrapConfig
.control_model`, and records flight quality, readiness, allocation stability,
and what the transaction accepted or rejected on each run.

Two variants are added to the canonical release and the fixed development
campaign.  A state-noise variant gives the estimator and the controller a
fixed-seed noisy view of the state while the plant and every recorded metric
keep the true one.  A configuration-change variant mutates the hidden arm
length in place at four seconds without resetting the simulator, which is the
case the transaction is supposed to protect against and also the case where a
frozen snapshot is the stale model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.control._common import quaternion_to_rotation
from glassbox.control.online_bootstrap import (
    ProgressiveBootstrapController,
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
)
from glassbox.core.dynamics import GRAVITY_M_S2
from glassbox.experimental.dual_control import (
    DualControlConfig,
    DualControlNMPC,
    DualControlResult,
    command_information_log_determinant,
)
from glassbox.integrations.crazyflow import CrazyflowPlant, CrazyflowPlantConfig
from glassbox.integrations.crazyflow_telemetry import (
    PlantTelemetryRecorder,
    initial_plant_state,
    tilt_rad,
)
from glassbox.integrations.crazyflow_throw import (
    CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS,
    CrazyflowThrowScenario,
)

#: The two transactional arms the report differences against each other.
CONTROL_MODELS = ("certified", "working")
#: The experimental third arm: one dual-control optimization, no cascade, no
#: certification, flying the working belief from zero information.
DUAL_CONTROL_MODEL = "dual_control_nmpc"
#: Every arm the command line runs by default.
STUDY_CONTROL_MODELS = (*CONTROL_MODELS, DUAL_CONTROL_MODEL)
DEFAULT_OUTPUT_PATH = Path("artifacts/crazyflow_throw_study/report.json")
#: Commands kept verbatim from model enable, for the early-action analysis.
EARLY_COMMAND_COUNT = 30
MODEL_ENABLE_DELAY_S = 1.0
TARGET_TRIAL_DURATION_S = 10.0
#: The closing stretch every case should be quietly hovering through.  Chatter
#: measured here cannot be confused with the arrest transient, which the two
#: modes enter at different times.
SETTLED_WINDOW_S = 3.0
COMMAND_BOUND_TOLERANCE = 1e-8
HOVER_ENVELOPE = {
    "speed_m_s": 0.10,
    "angular_rate_rad_s": 0.10,
    "absolute_vertical_speed_m_s": 0.05,
    "tilt_rad": 0.05,
}


@dataclass(frozen=True)
class ThrowStudyStateNoise:
    """Fixed-seed measurement noise on the state the online stack sees."""

    position_m: float = 0.005
    velocity_m_s: float = 0.02
    attitude_rad: float = 0.005
    angular_velocity_rad_s: float = 0.01
    seed: int = 20260901

    def __post_init__(self) -> None:
        scales = (
            self.position_m,
            self.velocity_m_s,
            self.attitude_rad,
            self.angular_velocity_rad_s,
        )
        if not np.all(np.isfinite(scales)) or np.any(np.asarray(scales) < 0.0):
            raise ValueError("state noise scales must be finite and nonnegative")
        if self.seed < 0:
            raise ValueError("state noise seed cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": self.position_m,
            "velocity_m_s": self.velocity_m_s,
            "attitude_rad": self.attitude_rad,
            "angular_velocity_rad_s": self.angular_velocity_rad_s,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ThrowStudyConfigurationChange:
    """One hidden airframe change applied in place during a run."""

    time_s: float = 4.0
    arm_length_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s <= MODEL_ENABLE_DELAY_S:
            raise ValueError("configuration change must fall after model enable")
        if not math.isfinite(self.arm_length_ratio) or self.arm_length_ratio <= 0.0:
            raise ValueError("changed arm length ratio must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "arm_length_ratio": self.arm_length_ratio,
        }


@dataclass(frozen=True)
class ThrowStudyCase:
    """One release, plus whatever the study perturbs on top of it."""

    name: str
    scenario: CrazyflowThrowScenario
    state_noise: ThrowStudyStateNoise | None = None
    configuration_change: ThrowStudyConfigurationChange | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("study case name cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario_name": self.scenario.name,
            "hidden_arm_length_ratio": self.scenario.arm_length_ratio,
            "release_height_m": self.scenario.release_height_m,
            "release_world_velocity_m_s": list(self.scenario.world_velocity_m_s),
            "release_angular_velocity_rad_s": list(
                self.scenario.angular_velocity_rad_s
            ),
            "release_roll_rad": self.scenario.roll_rad,
            "release_pitch_rad": self.scenario.pitch_rad,
            "state_noise": (
                None if self.state_noise is None else self.state_noise.to_dict()
            ),
            "configuration_change": (
                None
                if self.configuration_change is None
                else self.configuration_change.to_dict()
            ),
        }


def _study_cases() -> tuple[ThrowStudyCase, ...]:
    """Return the campaign releases plus the noise and change variants.

    The campaign's first scenario is the canonical release, so it supplies the
    canonical case rather than being run a second time under another name.
    """

    canonical = CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS[0]
    cases = [
        ThrowStudyCase(name=scenario.name, scenario=scenario)
        for scenario in CRAZYFLOW_THROW_CAMPAIGN_SCENARIOS
    ]
    cases.append(
        ThrowStudyCase(
            name="canonical_state_noise",
            scenario=canonical,
            state_noise=ThrowStudyStateNoise(),
        )
    )
    cases.append(
        ThrowStudyCase(
            name="canonical_mid_flight_arm_change",
            scenario=canonical,
            configuration_change=ThrowStudyConfigurationChange(),
        )
    )
    return tuple(cases)


CRAZYFLOW_THROW_STUDY_CASES = _study_cases()


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two WXYZ quaternions."""

    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return np.asarray(
        (
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        )
    )


def _perturbed_quaternion(
    quaternion: np.ndarray,
    rotation_vector: np.ndarray,
) -> np.ndarray:
    """Rotate one attitude by a small body-frame rotation vector."""

    half = 0.5 * rotation_vector
    delta = np.concatenate((np.ones(1), half))
    product = _quaternion_product(quaternion, delta / np.linalg.norm(delta))
    return product / np.linalg.norm(product)


class _StateObserver:
    """Turn true plant states into the states the online stack is given."""

    def __init__(self, noise: ThrowStudyStateNoise | None) -> None:
        self._noise = noise
        self._generator = None if noise is None else np.random.default_rng(noise.seed)

    def observe(self, state: np.ndarray) -> np.ndarray:
        observed = np.asarray(state, dtype=np.float64).copy()
        if self._noise is None or self._generator is None:
            return observed
        draw = self._generator.standard_normal
        observed[0:3] += self._noise.position_m * draw(3)
        observed[3:6] += self._noise.velocity_m_s * draw(3)
        observed[6:10] = _perturbed_quaternion(
            observed[6:10] / np.linalg.norm(observed[6:10]),
            self._noise.attitude_rad * draw(3),
        )
        observed[10:13] += self._noise.angular_velocity_rad_s * draw(3)
        return observed


def _effective_allocation(belief: RecursiveBootstrapBelief) -> np.ndarray:
    """Return the motor command the belief asks for per unit of desired action.

    Column zero is the collective column the cascade's thrust reference moves
    along, and the remaining three are the angular columns the cascade's tilt
    error is allocated through.  Both carry the authority scaling, so this is
    the map that actually reaches the motors rather than the raw fit.
    """

    allocation = np.zeros((4, 4), dtype=np.float64)
    collective_sum = float(np.sum(belief.collective_acceleration_per_command))
    if collective_sum > 1e-8:
        allocation[:, 0] = belief.collective_authority / collective_sum
    supported_effect = (
        belief.angular_acceleration_per_command
        @ belief.normalized_command_support_projector
    )
    if belief.angular_effect_rank:
        allocation[:, 1:] = (
            np.linalg.pinv(supported_effect, rcond=1e-5)
            @ belief.angular_output_support_projector
            @ np.diag(belief.angular_axis_authority)
        )
    return allocation


def _hover_relative_error(
    belief: RecursiveBootstrapBelief,
    hidden_hover: float,
) -> float:
    if belief.hover_command is None:
        return float("nan")
    return abs(float(np.mean(belief.hover_command)) - hidden_hover) / hidden_hover


def _first_finite(values: np.ndarray) -> float:
    finite = np.flatnonzero(np.isfinite(values))
    return float("nan") if len(finite) == 0 else float(values[finite[0]])


def _optional(value: float) -> float | None:
    """JSON cannot carry a NaN, so an unavailable metric becomes null."""

    return None if not math.isfinite(value) else float(value)


def _flown_prediction_error(
    belief: RecursiveBootstrapBelief,
    previous_state: np.ndarray,
    current_state: np.ndarray,
    command: np.ndarray,
    sample_period_s: float,
) -> tuple[float, float]:
    """Return how far the flown belief missed the interval it just commanded.

    This is the same one-step regression target the identifier fits, evaluated
    against the model the controller actually used, so a frozen snapshot that
    the airframe has moved away from shows up here as a rising error rather
    than having to be inferred from the belief's coefficients.
    """

    rotation = quaternion_to_rotation(previous_state[6:10])
    world_acceleration = (current_state[3:6] - previous_state[3:6]) / sample_period_s
    body_specific_force = rotation.T @ (
        world_acceleration + np.asarray((0.0, 0.0, GRAVITY_M_S2))
    )
    body_velocity = rotation.T @ previous_state[3:6]
    angular_acceleration = (
        current_state[10:13] - previous_state[10:13]
    ) / sample_period_s
    collective_error = body_specific_force[2] - (
        belief.predict_collective_specific_force(command, body_velocity)
    )
    angular_error = angular_acceleration - belief.predict_angular_acceleration(
        command,
        previous_state[10:13],
    )
    return float(collective_error), float(np.linalg.norm(angular_error))


def _window_rmse(
    values: list[float],
    start: int,
    stop: int | None = None,
) -> float:
    window = np.asarray(values[start:stop])
    if len(window) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window))))


def _flip_counts(flags: np.ndarray) -> tuple[int, int]:
    """Return how often a readiness series was lost and then regained."""

    ready = np.flatnonzero(flags)
    if len(ready) == 0:
        return 0, 0
    tail = flags[ready[0] :]
    changes = np.diff(tail.astype(np.int8))
    return int(np.sum(changes == -1)), int(np.sum(changes == 1))


@dataclass
class _TrialRecord:
    """Everything one closed-loop run accumulates, before it is reduced.

    The ``dual_`` lists stay empty in the two cascade modes, which have no
    optimizer to report on, so the reduction below emits the dual-control
    section only for the mode that filled them.
    """

    flown_allocations: list[np.ndarray]
    flown_angular_row_norms: list[np.ndarray]
    working_angular_row_norms: list[np.ndarray]
    working_hover_errors: list[float]
    working_supported: list[bool]
    control_model_ready: list[bool]
    information_action_fractions: list[float]
    flown_collective_errors: list[float]
    flown_angular_errors: list[float]
    command_evidence_ranks: list[int]
    dual_results: list[DualControlResult]
    dual_log_determinants: list[float]
    first_supported_control_step: int | None = None
    control_model_step: int | None = None
    configuration_change_step: int | None = None
    command_rank_four_step: int | None = None


def _new_record() -> _TrialRecord:
    return _TrialRecord(
        flown_allocations=[],
        flown_angular_row_norms=[],
        working_angular_row_norms=[],
        working_hover_errors=[],
        working_supported=[],
        control_model_ready=[],
        information_action_fractions=[],
        flown_collective_errors=[],
        flown_angular_errors=[],
        command_evidence_ranks=[],
        dual_results=[],
        dual_log_determinants=[],
    )


def _fly_trial(
    case: ThrowStudyCase,
    control_model: str,
    plant: CrazyflowPlant,
) -> tuple[_TrialRecord, PlantTelemetryRecorder, np.ndarray, dict[str, Any]]:
    """Run one release-to-hover trial and return its raw telemetry.

    The loop is the canonical throw loop: motors and model are off for the
    first second, then every interval issues one bounded command and folds the
    resulting transition into the identifier.  The only additions are the
    observer that may add measurement noise, the in-place configuration change,
    and the per-interval bookkeeping the study reduces afterwards.
    """

    scenario = case.scenario
    # The dual-control arm has no certification transaction of its own: it flies
    # the working belief from zero information, which is what ``working`` mode
    # means to the identifier.  The shadow transaction is still scored.
    dual = control_model == DUAL_CONTROL_MODEL
    identifier = RecursiveBootstrapIdentifier(
        RecursiveBootstrapConfig(control_model="working" if dual else control_model)
    )
    dual_config = DualControlConfig(sample_period_s=plant.sample_period_s)
    dual_controller = DualControlNMPC(dual_config) if dual else None
    controller = None if dual else ProgressiveBootstrapController(identifier.config)
    observer = _StateObserver(case.state_noise)
    release_state = initial_plant_state(
        world_velocity_m_s=scenario.world_velocity_m_s,
        angular_velocity_rad_s=scenario.angular_velocity_rad_s,
        roll_rad=scenario.roll_rad,
        pitch_rad=scenario.pitch_rad,
    )
    release_state[2] = scenario.release_height_m
    plant.set_arm_length_ratio(scenario.arm_length_ratio)
    sample = plant.reset(release_state, applied_motor_thrust_fraction=np.zeros(4))
    telemetry = PlantTelemetryRecorder(sample)

    enable_step_count = round(MODEL_ENABLE_DELAY_S / plant.sample_period_s)
    zero_command = np.zeros(4, dtype=np.float64)
    requested_commands: list[np.ndarray] = []
    for _ in range(enable_step_count):
        requested_commands.append(zero_command)
        sample = plant.step(zero_command)
        telemetry.record(sample)

    online_step_count = round(
        (TARGET_TRIAL_DURATION_S - telemetry.timestamps_s[-1]) / plant.sample_period_s
    )
    record = _new_record()
    hidden_hover = plant.hover_motor_thrust_fraction
    previous_command = zero_command
    dual_plan: DualControlResult | None = None
    observed_state = observer.observe(sample.state)
    for step in range(online_step_count):
        change = case.configuration_change
        if (
            change is not None
            and record.configuration_change_step is None
            and sample.time_s >= change.time_s
        ):
            plant.set_arm_length_ratio(change.arm_length_ratio)
            record.configuration_change_step = step
        flown = identifier.predictive_belief
        if (
            record.first_supported_control_step is None
            and flown.has_any_control_authority
        ):
            record.first_supported_control_step = step
        record.flown_allocations.append(_effective_allocation(flown))
        record.flown_angular_row_norms.append(
            np.linalg.norm(flown.angular_acceleration_per_command, axis=1)
        )
        if dual_controller is not None:
            dual_decision = dual_controller.solve(
                observed_state,
                flown,
                previous_command,
                warm_start=dual_plan,
            )
            dual_plan = dual_decision if dual_decision.command_usable else None
            record.dual_results.append(dual_decision)
            record.dual_log_determinants.append(
                command_information_log_determinant(flown, dual_config)
            )
            command = dual_decision.command
        else:
            assert controller is not None
            decision = controller.command(
                observed_state,
                flown,
                previous_command=previous_command,
                # The committed excitation cap exists to keep a large probe out
                # of a candidate's validation window.  Working mode opens no
                # such window, so it never applies the cap and its excitation is
                # decided by exploration completion alone, which is the point of
                # the mode.
                online_belief=(
                    None if identifier.certified_belief is None else identifier.belief
                ),
            )
            record.information_action_fractions.append(
                decision.information_action_fraction
            )
            command = decision.command
        previous_observed = observed_state
        previous_applied = sample.applied_motor_thrust_fraction
        previous_command = command
        requested_commands.append(command)
        sample = plant.step(command)
        observed_state = observer.observe(sample.state)
        average_applied = 0.5 * (
            previous_applied + sample.applied_motor_thrust_fraction
        )
        collective_error, angular_error = _flown_prediction_error(
            flown,
            previous_observed,
            observed_state,
            average_applied,
            plant.sample_period_s,
        )
        record.flown_collective_errors.append(collective_error)
        record.flown_angular_errors.append(angular_error)
        working = identifier.update(
            previous_observed,
            observed_state,
            average_applied,
            plant.sample_period_s,
        )
        telemetry.record(sample)
        record.working_supported.append(identifier.working_belief_supported)
        record.control_model_ready.append(identifier.control_model_ready)
        if record.control_model_step is None and identifier.control_model_ready:
            record.control_model_step = step
        record.working_angular_row_norms.append(
            np.linalg.norm(working.angular_acceleration_per_command, axis=1)
        )
        record.working_hover_errors.append(_hover_relative_error(working, hidden_hover))
        record.command_evidence_ranks.append(working.command_evidence_rank)
        if record.command_rank_four_step is None and working.command_evidence_rank == 4:
            record.command_rank_four_step = step

    identification: dict[str, Any] = {
        "working_interval_count": identifier.belief.interval_count,
        "terminal_working_belief_supported": identifier.working_belief_supported,
        "terminal_working_hover_relative_error": _optional(
            _hover_relative_error(identifier.belief, hidden_hover)
        ),
        "hidden_hover_motor_command": hidden_hover,
    }
    if control_model == "certified":
        history = identifier.validation_history
        identification.update(
            {
                "accepted_update_count": identifier.accepted_update_count,
                "rejected_update_count": identifier.rejected_update_count,
                "accepted_replacement_count": sum(
                    report.accepted and not report.initial_admission
                    for report in history
                ),
                "rejected_replacement_count": sum(
                    not report.accepted for report in history
                ),
                "rejection_reasons": _reason_counts(history),
                "certified_interval_count": (
                    None
                    if identifier.certified_belief is None
                    else identifier.certified_belief.interval_count
                ),
            }
        )
    else:
        history = identifier.shadow_validation_history
        identification.update(
            {
                "shadow_accepted_update_count": (
                    identifier.shadow_accepted_update_count
                ),
                "shadow_rejected_update_count": (
                    identifier.shadow_rejected_update_count
                ),
                "shadow_accepted_replacement_count": sum(
                    report.accepted and not report.initial_admission
                    for report in history
                ),
                "shadow_rejected_replacement_count": sum(
                    not report.accepted for report in history
                ),
                "shadow_rejection_reasons": _reason_counts(history),
                "shadow_certified_interval_count": (
                    None
                    if identifier.shadow_certified_belief is None
                    else identifier.shadow_certified_belief.interval_count
                ),
            }
        )
    identification["validation_history"] = [report.to_dict() for report in history]
    return record, telemetry, np.asarray(requested_commands), identification


def _reason_counts(history: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in history:
        if report.accepted:
            continue
        counts[report.reason] = counts.get(report.reason, 0) + 1
    return counts


def _sustained_hover_duration_s(
    timestamps: np.ndarray,
    speed: np.ndarray,
    rate: np.ndarray,
    vertical_speed: np.ndarray,
    tilt: np.ndarray,
    enable_index: int,
) -> tuple[float | None, float]:
    """Return when the strict hover envelope was entered for good, and for how long.

    Both modes are measured from the same model-enable index rather than from
    whenever each mode's own control model arrived, so an earlier handover
    cannot lengthen the window by definition.
    """

    inside = (
        (speed < HOVER_ENVELOPE["speed_m_s"])
        & (rate < HOVER_ENVELOPE["angular_rate_rad_s"])
        & (np.abs(vertical_speed) < HOVER_ENVELOPE["absolute_vertical_speed_m_s"])
        & (tilt < HOVER_ENVELOPE["tilt_rad"])
    )
    sustained = np.logical_and.accumulate(inside[::-1])[::-1]
    candidates = np.flatnonzero(sustained & (np.arange(len(inside)) >= enable_index))
    if len(candidates) == 0:
        return None, 0.0
    start = int(candidates[0])
    return float(timestamps[start]), float(timestamps[-1] - timestamps[start])


def _allocation_changes(
    allocations: list[np.ndarray],
    first_step: int | None,
) -> tuple[float, float]:
    """Return the largest absolute and relative per-interval allocation move.

    The relative figure is scaled by the window's median allocation size rather
    than by each interval's own, because an allocation that is still growing
    out of zero would otherwise report an unbounded relative move that has
    nothing to do with chattering.
    """

    if first_step is None or first_step + 1 >= len(allocations):
        return 0.0, 0.0
    stack = np.asarray(allocations[first_step:])
    differences = np.linalg.norm(np.diff(stack, axis=0), axis=(1, 2))
    scale = max(float(np.median(np.linalg.norm(stack, axis=(1, 2)))), 1e-9)
    return float(np.max(differences)), float(np.max(differences) / scale)


def _maximum_step(values: np.ndarray) -> float:
    return 0.0 if len(values) < 2 else float(np.max(np.abs(np.diff(values, axis=0))))


def _row_norm_samples(
    row_norms: list[np.ndarray],
    reference_step: int,
) -> dict[str, Any]:
    stack = np.asarray(row_norms)
    reference = stack[min(reference_step, len(stack) - 1)]
    terminal = stack[-1]
    return {
        "reference": reference.tolist(),
        "terminal": terminal.tolist(),
        "terminal_to_reference_ratio": (
            terminal / np.maximum(reference, 1e-12)
        ).tolist(),
    }


def _dual_control_metrics(
    record: _TrialRecord,
    timestamps: np.ndarray,
    enable_index: int,
    requested: np.ndarray,
) -> dict[str, Any]:
    """Reduce one dual-control run to what the design asks to be measured.

    The per-step information gain and log-determinant trajectories are kept in
    full, because the question this arm answers is what the optimizer did early
    rather than where it ended up.
    """

    results = record.dual_results
    gains = np.asarray([result.information_gain for result in results])
    iterations = np.asarray([result.iterations for result in results])
    statuses: dict[str, int] = {}
    for result in results:
        key = str(result.status)
        statuses[key] = statuses.get(key, 0) + 1
    early = requested[enable_index : enable_index + EARLY_COMMAND_COUNT]
    ranks = np.asarray(record.command_evidence_ranks)
    return {
        "command_information_rank_four_step": record.command_rank_four_step,
        "command_information_rank_four_time_s": (
            None
            if record.command_rank_four_step is None
            else float(timestamps[enable_index + record.command_rank_four_step + 1])
        ),
        "terminal_command_evidence_rank": (None if len(ranks) == 0 else int(ranks[-1])),
        "maximum_command_evidence_rank": (
            None if len(ranks) == 0 else int(np.max(ranks))
        ),
        "information_gain_per_step": gains.tolist(),
        "command_information_log_determinant": (list(record.dual_log_determinants)),
        "total_information_gain": float(np.sum(gains)),
        "maximum_information_gain": float(np.max(gains)) if len(gains) else 0.0,
        "solve_iterations": {
            "total": int(np.sum(iterations)),
            "mean": float(np.mean(iterations)) if len(iterations) else 0.0,
            "maximum": int(np.max(iterations)) if len(iterations) else 0,
        },
        "status_counts": statuses,
        "unusable_command_count": int(
            sum(not result.command_usable for result in results)
        ),
        "objective_terms": {
            name: {
                "mean": float(np.mean(values)) if len(values) else 0.0,
                "maximum": float(np.max(values)) if len(values) else 0.0,
                "terminal": float(values[-1]) if len(values) else 0.0,
            }
            for name, values in (
                (
                    "objective_value",
                    np.asarray([result.objective_value for result in results]),
                ),
                (
                    "tracking_cost",
                    np.asarray([result.tracking_cost for result in results]),
                ),
                (
                    "command_rate_cost",
                    np.asarray([result.command_rate_cost for result in results]),
                ),
                (
                    "information_gain",
                    gains,
                ),
                (
                    "altitude_penalty",
                    np.asarray([result.altitude_penalty for result in results]),
                ),
                (
                    "tilt_penalty",
                    np.asarray([result.tilt_penalty for result in results]),
                ),
            )
        },
        "chance_constraints": {
            "altitude_active_step_total": int(
                sum(result.altitude_constraint_active_steps for result in results)
            ),
            "tilt_active_step_total": int(
                sum(result.tilt_constraint_active_steps for result in results)
            ),
            "altitude_saturated_step_total": int(
                sum(result.altitude_constraint_saturated_steps for result in results)
            ),
            "tilt_saturated_step_total": int(
                sum(result.tilt_constraint_saturated_steps for result in results)
            ),
            "intervals_with_active_altitude_penalty": int(
                sum(result.altitude_constraint_active_steps > 0 for result in results)
            ),
            "intervals_with_active_tilt_penalty": int(
                sum(result.tilt_constraint_active_steps > 0 for result in results)
            ),
            "maximum_predicted_altitude_spread_m": float(
                max(
                    (result.maximum_altitude_spread_m for result in results),
                    default=0.0,
                )
            ),
            "maximum_predicted_tilt_spread_rad": float(
                max((result.maximum_tilt_spread_rad for result in results), default=0.0)
            ),
            "maximum_predicted_rate_spread_rad_s": float(
                max(
                    (result.maximum_rate_spread_rad_s for result in results),
                    default=0.0,
                )
            ),
        },
        "warm_start_use_count": int(sum(result.used_warm_start for result in results)),
        "early_commands": early.tolist(),
        "early_information_gain": gains[:EARLY_COMMAND_COUNT].tolist(),
        "early_command_evidence_rank": ranks[:EARLY_COMMAND_COUNT].tolist(),
    }


def run_throw_study_trial(
    case: ThrowStudyCase,
    control_model: str,
    plant: CrazyflowPlant | None = None,
) -> dict[str, Any]:
    """Run one case in one control model and reduce it to study metrics."""

    if control_model not in STUDY_CONTROL_MODELS:
        raise ValueError(f"unknown control model {control_model!r}")
    owned = plant is None
    plant = (
        CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
        if plant is None
        else plant
    )
    try:
        record, telemetry, requested, identification = _fly_trial(
            case,
            control_model,
            plant,
        )
        timestamps = telemetry.timestamp_array()
        states = telemetry.state_array()
        applied = telemetry.applied_array()
        enable_index = round(MODEL_ENABLE_DELAY_S / plant.sample_period_s)
        online_step_count = len(record.working_supported)
        speed = np.linalg.norm(states[:, 3:6], axis=1)
        rate = np.linalg.norm(states[:, 10:13], axis=1)
        tilt = np.asarray([tilt_rad(state) for state in states])
        minimum = np.asarray(RecursiveBootstrapConfig().command_minimum)
        maximum = np.asarray(RecursiveBootstrapConfig().command_maximum)
        non_finite_count = int(
            np.count_nonzero(~np.isfinite(states))
            + np.count_nonzero(~np.isfinite(applied))
            + np.count_nonzero(~np.isfinite(requested))
        )
        bound_violations = int(
            np.count_nonzero(requested < minimum - COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(requested > maximum + COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(applied < minimum - COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(applied > maximum + COMMAND_BOUND_TOLERANCE)
        )
        hover_start_s, hover_duration_s = _sustained_hover_duration_s(
            timestamps,
            speed,
            rate,
            states[:, 5],
            tilt,
            enable_index,
        )
        supported = np.asarray(record.working_supported, dtype=bool)
        readiness_step = (
            None if not supported.any() else int(np.flatnonzero(supported)[0])
        )
        loss_count, regain_count = _flip_counts(supported)
        # A mode that never reached its control model is measured from model
        # enable, so the window is defined for every run rather than empty.
        handover_step = (
            0 if record.control_model_step is None else record.control_model_step
        )
        absolute_move, relative_move = _allocation_changes(
            record.flown_allocations,
            handover_step,
        )
        # ``requested`` still carries the pre-enable zeros, so the window that
        # measures chattering starts past them as well as past the handover.
        handover_index = enable_index + handover_step
        maximum_command_step = _maximum_step(requested[handover_index:])
        # The handover window still contains the violent arrest, and the two
        # modes hand over at different times, so a settled window is what
        # separates chattering from an expected transient.
        settled_step = online_step_count - round(
            SETTLED_WINDOW_S / plant.sample_period_s
        )
        settled_absolute, settled_relative = _allocation_changes(
            record.flown_allocations,
            max(settled_step, 0),
        )
        hover_errors = np.asarray(record.working_hover_errors)
        reference_step = (
            record.configuration_change_step
            if record.configuration_change_step is not None
            else online_step_count // 2
        )
        return {
            "control_model": control_model,
            "flight": {
                "terminal_speed_m_s": float(speed[-1]),
                "terminal_angular_rate_rad_s": float(rate[-1]),
                "terminal_tilt_rad": float(tilt[-1]),
                "terminal_vertical_speed_m_s": float(states[-1, 5]),
                "sustained_hover_start_time_s": hover_start_s,
                "sustained_hover_duration_s": hover_duration_s,
                "minimum_altitude_m": float(np.min(states[:, 2])),
                "maximum_tilt_after_enable_rad": float(np.max(tilt[enable_index:])),
                "maximum_speed_after_enable_m_s": float(np.max(speed[enable_index:])),
                "non_finite_value_count": non_finite_count,
                "command_bound_violation_count": bound_violations,
            },
            "readiness": {
                "first_supported_control_time_s": _optional(
                    float("nan")
                    if record.first_supported_control_step is None
                    else timestamps[enable_index + record.first_supported_control_step]
                ),
                "readiness_time_s": _optional(
                    float("nan")
                    if readiness_step is None
                    else timestamps[enable_index + readiness_step + 1]
                ),
                "control_model_time_s": _optional(
                    float("nan")
                    if record.control_model_step is None
                    else timestamps[enable_index + record.control_model_step + 1]
                ),
                "readiness_loss_count": loss_count,
                "readiness_regain_count": regain_count,
                "settled_readiness_loss_count": _flip_counts(
                    supported[max(settled_step, 0) :]
                )[0],
                "supported_interval_fraction": float(np.mean(supported)),
            },
            "stability": {
                "maximum_allocation_change": absolute_move,
                "maximum_relative_allocation_change": relative_move,
                "maximum_command_step": maximum_command_step,
                "settled_window_s": SETTLED_WINDOW_S,
                "settled_maximum_allocation_change": settled_absolute,
                "settled_maximum_relative_allocation_change": settled_relative,
                "settled_maximum_command_step": _maximum_step(
                    requested[enable_index + max(settled_step, 0) :]
                ),
                "nonzero_information_action_count": int(
                    np.count_nonzero(
                        np.asarray(record.information_action_fractions) > 0.0
                    )
                ),
                "total_information_action_fraction": float(
                    np.sum(record.information_action_fractions)
                ),
            },
            "hover_estimate": {
                "first_available_relative_error": _optional(
                    _first_finite(hover_errors)
                ),
                "reference_relative_error": _optional(
                    hover_errors[min(reference_step, len(hover_errors) - 1)]
                ),
                "terminal_relative_error": _optional(hover_errors[-1]),
            },
            "flown_model_error": {
                "collective_rmse_after_handover_m_s2": _window_rmse(
                    record.flown_collective_errors, handover_step
                ),
                "angular_rmse_after_handover_rad_s2": _window_rmse(
                    record.flown_angular_errors, handover_step
                ),
                "collective_rmse_before_reference_m_s2": _window_rmse(
                    record.flown_collective_errors, handover_step, reference_step
                ),
                "angular_rmse_before_reference_rad_s2": _window_rmse(
                    record.flown_angular_errors, handover_step, reference_step
                ),
                "collective_rmse_after_reference_m_s2": _window_rmse(
                    record.flown_collective_errors, reference_step
                ),
                "angular_rmse_after_reference_rad_s2": _window_rmse(
                    record.flown_angular_errors, reference_step
                ),
                "collective_rmse_settled_m_s2": _window_rmse(
                    record.flown_collective_errors, max(settled_step, 0)
                ),
                "angular_rmse_settled_rad_s2": _window_rmse(
                    record.flown_angular_errors, max(settled_step, 0)
                ),
            },
            "angular_effect": {
                "reference_time_s": float(
                    timestamps[enable_index + reference_step + 1]
                ),
                "working_belief_row_norm": _row_norm_samples(
                    record.working_angular_row_norms,
                    reference_step,
                ),
                "flown_belief_row_norm": _row_norm_samples(
                    record.flown_angular_row_norms,
                    reference_step,
                ),
            },
            "identification": identification,
            **(
                {
                    "dual_control": _dual_control_metrics(
                        record,
                        timestamps,
                        enable_index,
                        requested,
                    )
                }
                if control_model == DUAL_CONTROL_MODEL
                else {}
            ),
        }
    finally:
        if owned:
            plant.close()


_DIFFERENCE_SECTIONS = (
    "flight",
    "readiness",
    "stability",
    "hover_estimate",
    "flown_model_error",
)


def _mode_difference(
    certified: dict[str, Any],
    working: dict[str, Any],
) -> dict[str, Any]:
    """Return working minus certified on every numeric metric they share."""

    difference: dict[str, Any] = {}
    for section in _DIFFERENCE_SECTIONS:
        entries: dict[str, Any] = {}
        for name, value in certified[section].items():
            other = working[section].get(name)
            if isinstance(value, bool) or isinstance(other, bool):
                entries[name] = None if other is None else bool(other) != value
            elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
                entries[name] = float(other) - float(value)
            else:
                entries[name] = None
        difference[section] = entries
    return difference


def run_crazyflow_throw_study(
    cases: tuple[ThrowStudyCase, ...] = CRAZYFLOW_THROW_STUDY_CASES,
    control_models: Sequence[str] = CONTROL_MODELS,
) -> dict[str, Any]:
    """Run every case in the named control models and return the comparison.

    The default is the two transactional arms the report differences against
    each other.  The command line adds :data:`DUAL_CONTROL_MODEL`, which is a
    third independent arm rather than a second half of that difference.
    """

    if not cases:
        raise ValueError("the throw study needs at least one case")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("throw study case names must be unique")
    models = tuple(control_models)
    if not models:
        raise ValueError("the throw study needs at least one control model")
    unknown = sorted(set(models) - set(STUDY_CONTROL_MODELS))
    if unknown:
        raise ValueError(f"unknown control model(s): {', '.join(unknown)}")
    results: list[dict[str, Any]] = []
    for case in cases:
        modes = {model: run_throw_study_trial(case, model) for model in models}
        entry: dict[str, Any] = {"case": case.to_dict(), "modes": modes}
        if set(CONTROL_MODELS) <= set(modes):
            entry["difference_working_minus_certified"] = _mode_difference(
                modes["certified"],
                modes["working"],
            )
        results.append(entry)
    report = {
        "artifact_type": "glassbox_crazyflow_throw_control_model_study",
        "schema_version": 1,
        "semantics": {
            "diagnostic_only": True,
            "flight_safety_claim": False,
            "deterministic": True,
            "same_closed_loop_in_both_modes": True,
            "state_noise_applied_to_estimator_and_controller_only": True,
            "hidden_configuration_changed_in_place_without_reset": True,
            "sustained_hover_measured_from_model_enable_in_both_modes": True,
            "working_mode_records_a_shadow_transaction": True,
        },
        "control_models": list(models),
        "case_count": len(results),
        "cases": results,
        "aggregate": _aggregate(results, models),
        "limitations": [
            "This is a two-arm comparison on eight deterministic releases, not held-out validation.",
            "Only one noise realisation and one configuration change are exercised.",
            "The plant, release states, and controller gains are unchanged from the recorded throw diagnostic.",
            "Working mode caps excitation once its own readiness holds, which is earlier than certification, so the excitation schedules are not identical.",
        ],
    }
    json.dumps(report, allow_nan=False)
    return report


def _aggregate(
    results: list[dict[str, Any]],
    models: tuple[str, ...] = CONTROL_MODELS,
) -> dict[str, Any]:
    def worst(model: str, section: str, name: str) -> float:
        return max(float(result["modes"][model][section][name]) for result in results)

    def best(model: str, section: str, name: str) -> float:
        return min(float(result["modes"][model][section][name]) for result in results)

    aggregate: dict[str, Any] = {
        "all_values_finite_and_bounded": all(
            result["modes"][model]["flight"]["non_finite_value_count"] == 0
            and result["modes"][model]["flight"]["command_bound_violation_count"] == 0
            for result in results
            for model in models
        ),
    }
    if "working" in models:
        aggregate["working_mode_never_lost_readiness"] = all(
            result["modes"]["working"]["readiness"]["readiness_loss_count"] == 0
            for result in results
        )
        aggregate["working_mode_reached_readiness_in_every_case"] = all(
            result["modes"]["working"]["readiness"]["control_model_time_s"] is not None
            for result in results
        )
    if DUAL_CONTROL_MODEL in models:
        aggregate["dual_control_reached_command_rank_four_in_every_case"] = all(
            result["modes"][DUAL_CONTROL_MODEL]["dual_control"][
                "command_information_rank_four_time_s"
            ]
            is not None
            for result in results
        )
        aggregate["dual_control_unusable_command_total"] = sum(
            result["modes"][DUAL_CONTROL_MODEL]["dual_control"][
                "unusable_command_count"
            ]
            for result in results
        )
    for model in models:
        aggregate[model] = {
            "worst_terminal_speed_m_s": worst(model, "flight", "terminal_speed_m_s"),
            "worst_terminal_angular_rate_rad_s": worst(
                model, "flight", "terminal_angular_rate_rad_s"
            ),
            "worst_terminal_tilt_rad": worst(model, "flight", "terminal_tilt_rad"),
            "minimum_sustained_hover_duration_s": best(
                model, "flight", "sustained_hover_duration_s"
            ),
            "minimum_altitude_m": best(model, "flight", "minimum_altitude_m"),
            "worst_maximum_command_step": worst(
                model, "stability", "maximum_command_step"
            ),
            "worst_maximum_relative_allocation_change": worst(
                model, "stability", "maximum_relative_allocation_change"
            ),
            "worst_settled_maximum_command_step": worst(
                model, "stability", "settled_maximum_command_step"
            ),
            "worst_settled_maximum_relative_allocation_change": worst(
                model, "stability", "settled_maximum_relative_allocation_change"
            ),
        }
    if set(CONTROL_MODELS) <= set(models):
        aggregate["cases_where_working_terminal_speed_is_no_worse"] = [
            result["case"]["name"]
            for result in results
            if result["modes"]["working"]["flight"]["terminal_speed_m_s"]
            <= result["modes"]["certified"]["flight"]["terminal_speed_m_s"]
        ]
    aggregate["case_count"] = len(results)
    return aggregate


_TABLE_COLUMNS: tuple[tuple[str, str, str, str], ...] = (
    ("speed", "flight", "terminal_speed_m_s", "{:.4f}"),
    ("rate", "flight", "terminal_angular_rate_rad_s", "{:.4f}"),
    ("tilt", "flight", "terminal_tilt_rad", "{:.4f}"),
    ("hover s", "flight", "sustained_hover_duration_s", "{:.2f}"),
    ("min alt", "flight", "minimum_altitude_m", "{:.3f}"),
    ("max tilt", "flight", "maximum_tilt_after_enable_rad", "{:.3f}"),
    ("max spd", "flight", "maximum_speed_after_enable_m_s", "{:.2f}"),
    ("ready s", "readiness", "readiness_time_s", "{:.2f}"),
    ("model s", "readiness", "control_model_time_s", "{:.2f}"),
    ("flips", "readiness", "readiness_loss_count", "{:.0f}"),
    ("set flips", "readiness", "settled_readiness_loss_count", "{:.0f}"),
    ("alloc d", "stability", "maximum_relative_allocation_change", "{:.3f}"),
    (
        "set alloc d",
        "stability",
        "settled_maximum_relative_allocation_change",
        "{:.4f}",
    ),
    ("set cmd step", "stability", "settled_maximum_command_step", "{:.4f}"),
    ("hover err", "hover_estimate", "terminal_relative_error", "{:.5f}"),
    ("ang rmse", "flown_model_error", "angular_rmse_settled_rad_s2", "{:.3f}"),
)


def format_study_table(report: dict[str, Any]) -> str:
    """Render the per-case, per-mode comparison as a markdown table."""

    header = ["case", "mode", *(column[0] for column in _TABLE_COLUMNS)]
    rows = [header, ["---"] * len(header)]
    for result in report["cases"]:
        for model in report["control_models"]:
            metrics = result["modes"][model]
            row = [result["case"]["name"], model]
            for _, section, name, template in _TABLE_COLUMNS:
                value = metrics[section][name]
                row.append("n/a" if value is None else template.format(value))
            rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines = [
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rows
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    # Set before any lazy `import crazyflow` reaches its own SciPy-array-API guard.
    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    parser = argparse.ArgumentParser(
        description=(
            "Compare flying the certified snapshot, flying the working belief, "
            "and flying one dual-control optimization on the Crazyflow throw "
            "diagnostic."
        )
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=DEFAULT_OUTPUT_PATH,
        help="where to write the study report (default: %(default)s)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="run only the named case; may be repeated",
    )
    parser.add_argument(
        "--control-model",
        action="append",
        default=None,
        choices=STUDY_CONTROL_MODELS,
        help=(
            "run only the named control model; may be repeated "
            f"(default: {' '.join(STUDY_CONTROL_MODELS)})"
        ),
    )
    args = parser.parse_args(argv)
    cases = CRAZYFLOW_THROW_STUDY_CASES
    if args.case:
        by_name = {case.name: case for case in cases}
        unknown = sorted(set(args.case) - set(by_name))
        if unknown:
            parser.error(f"unknown case name(s): {', '.join(unknown)}")
        cases = tuple(by_name[name] for name in args.case)
    models = tuple(args.control_model) if args.control_model else STUDY_CONTROL_MODELS
    report = run_crazyflow_throw_study(cases, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(format_study_table(report))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
