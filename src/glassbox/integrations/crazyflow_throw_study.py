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
import functools
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
    dual_control_config,
)
from glassbox.integrations.crazyflow import (
    CrazyflowDivergenceError,
    CrazyflowPlant,
    CrazyflowPlantConfig,
)
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
#: The experimental arms: one dual-control optimization, no cascade, no
#: certification, flying the working belief from zero information.  Each arm
#: names one :data:`glassbox.experimental.dual_control.DUAL_CONTROL_VARIANTS`
#: entry, so the passes of the design are separate rows of the same study
#: rather than separate studies.
DUAL_CONTROL_MODEL = "dual_control_nmpc"
DUAL_CONTROL_PASS2A_MODEL = "dual_control_nmpc_pass2a"
DUAL_CONTROL_PASS2B_MODEL = "dual_control_nmpc_pass2b"
DUAL_CONTROL_PASS3_MODEL = "dual_control_nmpc_pass3"
DUAL_CONTROL_PASS4_MODEL = "dual_control_nmpc_pass4"
DUAL_CONTROL_PASS5_MODEL = "dual_control_nmpc_pass5"
DUAL_CONTROL_PASS6_MODEL = "dual_control_nmpc_pass6"
#: Which objective each arm plans with.  Pass three plans with the pass-2b
#: objective unchanged: its two changes are in the identifier, not in the
#: optimization, so re-running the objective would confound them.  Pass four
#: changes only where the plan starts and what the rate cost charges for.  Pass
#: five replaces the plan parameterization, the spread model, and the seeds at
#: once: it is one goal over a one-second horizon of slew-bounded moves.
DUAL_CONTROL_MODEL_VARIANTS = {
    DUAL_CONTROL_MODEL: "pass1",
    DUAL_CONTROL_PASS2A_MODEL: "pass2a",
    DUAL_CONTROL_PASS2B_MODEL: "pass2b",
    DUAL_CONTROL_PASS3_MODEL: "pass2b",
    DUAL_CONTROL_PASS4_MODEL: "pass4",
    DUAL_CONTROL_PASS5_MODEL: "pass5",
    DUAL_CONTROL_PASS6_MODEL: "pass6",
}
#: Identifier switches each arm turns on.  Both are opt-in, so every other arm
#: in this study, the two cascade modes included, runs the identifier it always
#: has.
DUAL_CONTROL_IDENTIFIER_OPTIONS: dict[str, dict[str, Any]] = {
    DUAL_CONTROL_PASS3_MODEL: {
        "staged_regressors": True,
        "enforce_collective_sign": True,
    },
    DUAL_CONTROL_PASS6_MODEL: {
        "transition_aggregation_steps": 2,
        "integrated_collective": True,
    },
}
#: Every arm the command line will accept.
STUDY_CONTROL_MODELS = (*CONTROL_MODELS, *DUAL_CONTROL_MODEL_VARIANTS)
#: Every arm the command line runs by default.  The first dual-control pass is
#: superseded and reachable only by asking for it: it is kept runnable so its
#: recorded failure stays reproducible, not because it is worth re-measuring.
DEFAULT_CONTROL_MODELS = (
    *CONTROL_MODELS,
    DUAL_CONTROL_PASS2A_MODEL,
    DUAL_CONTROL_PASS2B_MODEL,
    DUAL_CONTROL_PASS3_MODEL,
    DUAL_CONTROL_PASS4_MODEL,
    DUAL_CONTROL_PASS5_MODEL,
    DUAL_CONTROL_PASS6_MODEL,
)
#: What to call each arm on screen.  Used by the renderer so an overlay names
#: the control model actually flying rather than assuming the cascade.
ARM_DISPLAY_NAMES: dict[str, str] = {
    "certified": "FROZEN SNAPSHOT CASCADE",
    "working": "WORKING-BELIEF CASCADE",
    DUAL_CONTROL_MODEL: "DUAL-CONTROL NMPC pass 1",
    DUAL_CONTROL_PASS2A_MODEL: "DUAL-CONTROL NMPC pass 2a",
    DUAL_CONTROL_PASS2B_MODEL: "DUAL-CONTROL NMPC pass 2b",
    DUAL_CONTROL_PASS3_MODEL: "DUAL-CONTROL NMPC pass 3",
    DUAL_CONTROL_PASS4_MODEL: "DUAL-CONTROL NMPC pass 4",
    DUAL_CONTROL_PASS5_MODEL: "DUAL-CONTROL NMPC pass 5",
    DUAL_CONTROL_PASS6_MODEL: "DUAL-CONTROL NMPC pass 6",
}
#: The arms the ensemble protocol compares.  Two cascade references, the design
#: the third pass left standing, the fourth pass's base action, and the fifth
#: pass's one-goal formulation.  The superseded passes are excluded: an ensemble
#: is expensive and re-measuring a configuration whose failure is already
#: explained buys nothing.  The four earlier arms stay in so the comparison is
#: paired on the same releases rather than merged across runs.
ENSEMBLE_CONTROL_MODELS = (
    *CONTROL_MODELS,
    DUAL_CONTROL_PASS2B_MODEL,
    DUAL_CONTROL_PASS4_MODEL,
    DUAL_CONTROL_PASS5_MODEL,
    DUAL_CONTROL_PASS6_MODEL,
)
DEFAULT_OUTPUT_PATH = Path("artifacts/crazyflow_throw_study/report.json")
#: Altitude at or below which the vehicle is on the floor.  A trial stops at
#: its first floor contact and is a failure from then on: everything the
#: simulator would produce afterwards is ground contact, not flight, and none
#: of it should reach the identifier, the metrics, or the ensemble's clock.
FLOOR_CONTACT_ALTITUDE_M = 0.0
#: Commands kept verbatim from model enable, for the early-action analysis.
EARLY_COMMAND_COUNT = 30
MODEL_ENABLE_DELAY_S = 1.0
TARGET_TRIAL_DURATION_S = 10.0
#: The closing stretch every case should be quietly hovering through.  Chatter
#: measured here cannot be confused with the arrest transient, which the two
#: modes enter at different times.
SETTLED_WINDOW_S = 3.0
#: Intervals of the settled window, at the study's fixed hundred-hertz loop.
SETTLED_INTERVAL_COUNT = 300
#: Intervals between samples of the spread-against-tracking charge series.
CHARGE_SERIES_STRIDE = 10
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
class ThrowReleasePerturbation:
    """One draw from the declared distribution of releases around a scenario.

    The release state is the only thing perturbed: the hidden airframe, the
    release height, the loop rate, and every controller setting are exactly the
    scenario's.  Scales multiply the released world velocity and body rate
    componentwise; the rotation vector is applied in the body frame on top of
    the scenario's own roll and pitch, so a draw is a rotated, rescaled version
    of the same throw rather than a different throw.
    """

    velocity_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    angular_velocity_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tilt_rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: The seed this draw came from — the ensemble seed, the case index, and the
    #: replicate index — carried so a single row of an ensemble reproduces on
    #: its own without re-deriving the whole draw order.
    seed: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                *self.velocity_scale,
                *self.angular_velocity_scale,
                *self.tilt_rotation_rad,
            ),
            dtype=np.float64,
        )
        if values.shape != (9,) or not np.all(np.isfinite(values)):
            raise ValueError("release perturbation values must be finite")
        if np.any(values[:6] <= 0.0):
            raise ValueError("release perturbation scales must be positive")

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Return the perturbed release state; the altitude is left alone."""

        perturbed = np.asarray(state, dtype=np.float64).copy()
        perturbed[3:6] *= np.asarray(self.velocity_scale)
        perturbed[10:13] *= np.asarray(self.angular_velocity_scale)
        perturbed[6:10] = _perturbed_quaternion(
            perturbed[6:10] / np.linalg.norm(perturbed[6:10]),
            np.asarray(self.tilt_rotation_rad),
        )
        return perturbed

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity_scale": list(self.velocity_scale),
            "angular_velocity_scale": list(self.angular_velocity_scale),
            "tilt_rotation_rad": list(self.tilt_rotation_rad),
            "seed": list(self.seed),
        }


@dataclass(frozen=True)
class ThrowStudyCase:
    """One release, plus whatever the study perturbs on top of it."""

    name: str
    scenario: CrazyflowThrowScenario
    state_noise: ThrowStudyStateNoise | None = None
    configuration_change: ThrowStudyConfigurationChange | None = None
    release_perturbation: ThrowReleasePerturbation | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("study case name cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        # The perturbation key appears only on a drawn case, so a deterministic
        # study's case block is exactly what it always was.
        entry = {
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
        if self.release_perturbation is not None:
            entry["release_perturbation"] = self.release_perturbation.to_dict()
        return entry


@dataclass(frozen=True)
class CrazyflowStudyTrace:
    """State-aligned telemetry for one study arm, playable by the throw renderer.

    Shares its state-aligned field names with
    :class:`glassbox.integrations.crazyflow_throw.CrazyflowThrowTrace` so the
    same renderer plays either one, but an arm here need not have earned or
    validated a control belief at all: ``certified_belief_sample_index`` is
    ``None`` and ``validated`` is ``False`` for every arm except ``certified``,
    honestly, rather than repurposing a working-belief or dual-control
    readiness moment as if it were a certification.
    """

    arm: str
    case_name: str
    sample_period_s: float
    model_enable_sample_index: int
    first_supported_control_sample_index: int | None
    command_rank_four_sample_index: int | None
    certified_belief_sample_index: int | None
    validated: bool
    timestamps_s: np.ndarray
    states: np.ndarray
    applied_motor_commands: np.ndarray
    requested_motor_commands: np.ndarray
    working_interval_counts: np.ndarray
    command_evidence_ranks: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        states = np.asarray(self.states, dtype=np.float64)
        applied = np.asarray(self.applied_motor_commands, dtype=np.float64)
        requested = np.asarray(self.requested_motor_commands, dtype=np.float64)
        interval_counts = np.asarray(self.working_interval_counts, dtype=np.int64)
        ranks = np.asarray(self.command_evidence_ranks, dtype=np.int64)
        if not self.arm:
            raise ValueError("a study trace must name its arm")
        if not self.case_name:
            raise ValueError("a study trace must name its case")
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("timestamps_s must contain at least two samples")
        sample_count = len(timestamps)
        aligned_shapes = {
            "states": (sample_count, 13),
            "applied_motor_commands": (sample_count, 4),
            "working_interval_counts": (sample_count,),
            "command_evidence_ranks": (sample_count,),
        }
        values = {
            "states": states,
            "applied_motor_commands": applied,
            "working_interval_counts": interval_counts,
            "command_evidence_ranks": ranks,
        }
        for name, shape in aligned_shapes.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have state-aligned shape {shape}")
        if requested.shape != (sample_count - 1, 4):
            raise ValueError("requested_motor_commands must be interval-aligned")
        if (
            not np.all(np.isfinite(timestamps))
            or not np.all(np.diff(timestamps) > 0.0)
            or not np.all(np.isfinite(states))
            or not np.all(np.isfinite(applied))
            or not np.all(np.isfinite(requested))
        ):
            raise ValueError("study trace values must be finite and ordered")
        if not (0 < self.model_enable_sample_index < sample_count):
            raise ValueError("model_enable_sample_index must be interior to the trace")
        for name, value in (
            (
                "first_supported_control_sample_index",
                self.first_supported_control_sample_index,
            ),
            ("command_rank_four_sample_index", self.command_rank_four_sample_index),
            ("certified_belief_sample_index", self.certified_belief_sample_index),
        ):
            if value is not None and not (
                self.model_enable_sample_index <= value < sample_count
            ):
                raise ValueError(f"{name} must fall within the trace when present")
        if self.validated and self.certified_belief_sample_index is None:
            raise ValueError("a validated trace must carry its certification index")
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "applied_motor_commands", applied)
        object.__setattr__(self, "requested_motor_commands", requested)
        object.__setattr__(self, "working_interval_counts", interval_counts)
        object.__setattr__(self, "command_evidence_ranks", ranks)


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
    working_interval_counts: list[int]
    dual_results: list[DualControlResult]
    dual_log_determinants: list[float]
    dual_information_ranks: list[int]
    staged_collective: list[bool]
    staged_angular: list[bool]
    sign_projection_counts: list[int]
    sign_projection_magnitudes: list[float]
    dual_config: dict[str, Any] | None = None
    identifier_config: dict[str, Any] | None = None
    first_supported_control_step: int | None = None
    control_model_step: int | None = None
    configuration_change_step: int | None = None
    #: Online step at which the vehicle first touched the floor, at which the
    #: trial was stopped; ``None`` for a trial that flew to the end.
    floor_contact_step: int | None = None
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
        working_interval_counts=[],
        dual_results=[],
        dual_log_determinants=[],
        dual_information_ranks=[],
        staged_collective=[],
        staged_angular=[],
        sign_projection_counts=[],
        sign_projection_magnitudes=[],
    )


def _fly_trial(
    case: ThrowStudyCase,
    control_model: str,
    plant: CrazyflowPlant,
    dual_controller: DualControlNMPC | None = None,
    dual_config: DualControlConfig | None = None,
    identifier_options: dict[str, Any] | None = None,
) -> tuple[
    _TrialRecord,
    PlantTelemetryRecorder,
    np.ndarray,
    dict[str, Any],
    CrazyflowStudyTrace,
]:
    """Run one release-to-hover trial and return its raw telemetry.

    The loop is the canonical throw loop: motors and model are off for the
    first second, then every interval issues one bounded command and folds the
    resulting transition into the identifier.  The only additions are the
    observer that may add measurement noise, the in-place configuration change,
    the declared release perturbation an ensemble draws, and the per-interval
    bookkeeping the study reduces afterwards.

    ``dual_controller`` lets a caller hand in a controller already compiled for
    this arm.  The controller carries no state between solves — the plan it
    warm-starts from is passed back in by this loop — so reusing one across
    trials is exactly the same computation, and it saves recompiling the solve
    program for every release of an ensemble.

    Alongside the record, this returns a :class:`CrazyflowStudyTrace`: the same
    state-aligned telemetry the report is reduced from, kept in a shape the
    throw renderer can play back directly, whether or not this arm ever
    certified a belief.
    """

    scenario = case.scenario
    # The dual-control arms have no certification transaction of their own: they
    # fly the working belief from zero information, which is what ``working``
    # mode means to the identifier.  The shadow transaction is still scored.
    variant = DUAL_CONTROL_MODEL_VARIANTS.get(control_model)
    dual = variant is not None
    identifier_config = RecursiveBootstrapConfig(
        control_model="working" if dual else control_model,
        **(
            DUAL_CONTROL_IDENTIFIER_OPTIONS.get(control_model, {})
            | (identifier_options or {})
        ),
    )
    identifier = RecursiveBootstrapIdentifier(identifier_config)
    if dual_config is None:
        dual_config = dual_control_config(
            variant if dual else "pass2b",
            sample_period_s=plant.sample_period_s,
        )
    elif not dual:
        raise ValueError("a dual-control config needs a dual-control arm")
    if not dual:
        dual_controller = None
    elif dual_controller is None:
        dual_controller = DualControlNMPC(dual_config)
    elif dual_controller.config != dual_config:
        raise ValueError("the supplied dual controller is configured for another arm")
    controller = None if dual else ProgressiveBootstrapController(identifier.config)
    observer = _StateObserver(case.state_noise)
    release_state = initial_plant_state(
        world_velocity_m_s=scenario.world_velocity_m_s,
        angular_velocity_rad_s=scenario.angular_velocity_rad_s,
        roll_rad=scenario.roll_rad,
        pitch_rad=scenario.pitch_rad,
    )
    release_state[2] = scenario.release_height_m
    if case.release_perturbation is not None:
        release_state = case.release_perturbation.apply(release_state)
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
    if dual:
        record.dual_config = dual_config.to_dict()
        record.identifier_config = {
            "staged_regressors": identifier_config.staged_regressors,
            "staging_sample_multiple": identifier_config.staging_sample_multiple,
            "enforce_collective_sign": identifier_config.enforce_collective_sign,
            "transition_aggregation_steps": (
                identifier_config.transition_aggregation_steps
            ),
        }
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
                # Before this loop's first command the vehicle is falling with
                # the motors off, so the command it carries is the diagnostic's
                # by fiat rather than the controller's own action.
                previous_command_owned=step > 0,
            )
            dual_plan = dual_decision if dual_decision.command_usable else None
            record.dual_results.append(dual_decision)
            record.dual_log_determinants.append(
                command_information_log_determinant(flown, dual_config)
            )
            # The rank of the information the controller is actually planning
            # against, which is the incumbent's rank rather than the rank the
            # identifier's own support rule admits for control.
            record.dual_information_ranks.append(
                _information_rank(flown, dual_config.epsilon)
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
        record.working_interval_counts.append(working.interval_count)
        record.staged_collective.append(working.collective_nuisance_staged)
        record.staged_angular.append(working.angular_nuisance_staged)
        record.sign_projection_counts.append(working.collective_sign_projection_count)
        record.sign_projection_magnitudes.append(
            working.collective_sign_projection_magnitude
        )
        if record.command_rank_four_step is None and working.command_evidence_rank == 4:
            record.command_rank_four_step = step
        if sample.state[2] <= FLOOR_CONTACT_ALTITUDE_M:
            # Floor contact ends the trial.  The contact sample is kept so the
            # trace shows where it happened; nothing after it is simulated.
            record.floor_contact_step = step
            break

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
    requested_array = np.asarray(requested_commands)
    trace = _build_study_trace(
        case,
        control_model,
        plant.sample_period_s,
        enable_step_count,
        record,
        telemetry,
        requested_array,
    )
    return record, telemetry, requested_array, identification, trace


def _build_study_trace(
    case: ThrowStudyCase,
    control_model: str,
    sample_period_s: float,
    enable_index: int,
    record: _TrialRecord,
    telemetry: PlantTelemetryRecorder,
    requested: np.ndarray,
) -> CrazyflowStudyTrace:
    """Assemble the render trace from one already-flown trial's telemetry.

    ``enable_index`` is the sample index of model enable: the identifier and
    controller are disabled for that many intervals, so the arrays the study
    accumulates per online step are prefixed with that many zeros to become
    state-aligned, exactly as :class:`.CrazyflowThrowTrace` does for the single
    default trial.
    """

    pre_enable = np.zeros(enable_index + 1, dtype=np.int64)
    ranks = np.concatenate(
        (pre_enable, np.asarray(record.command_evidence_ranks, dtype=np.int64))
    )
    interval_counts = np.concatenate(
        (pre_enable, np.asarray(record.working_interval_counts, dtype=np.int64))
    )
    first_supported_control_sample_index = (
        None
        if record.first_supported_control_step is None
        else enable_index + record.first_supported_control_step
    )
    command_rank_four_sample_index = (
        None
        if record.command_rank_four_step is None
        else enable_index + record.command_rank_four_step + 1
    )
    # Only certified mode's own readiness is a certification: working mode and
    # every dual-control arm read this from the working belief's own support
    # rule instead, which is a different claim and stays out of this field.
    certified_belief_sample_index = (
        enable_index + record.control_model_step + 1
        if control_model == "certified" and record.control_model_step is not None
        else None
    )
    return CrazyflowStudyTrace(
        arm=control_model,
        case_name=case.name,
        sample_period_s=sample_period_s,
        model_enable_sample_index=enable_index,
        first_supported_control_sample_index=first_supported_control_sample_index,
        command_rank_four_sample_index=command_rank_four_sample_index,
        certified_belief_sample_index=certified_belief_sample_index,
        validated=certified_belief_sample_index is not None,
        timestamps_s=telemetry.timestamp_array(),
        states=telemetry.state_array(),
        applied_motor_commands=telemetry.applied_array(),
        requested_motor_commands=requested,
        working_interval_counts=interval_counts,
        command_evidence_ranks=ranks,
    )


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


def _information_rank(belief: RecursiveBootstrapBelief, epsilon: float) -> int:
    """Numerical rank of the command information the dual controller plans on.

    The threshold is the same regularizing ``epsilon`` the objective floors the
    information with, expressed as a precision, so a direction counts as known
    exactly when the evidence about it outweighs the prior that keeps the
    log-determinant finite.
    """

    variance = max(float(np.mean(np.square(belief.angular_residual_std_rad_s2))), 1e-12)
    information = (
        np.asarray(belief.normalized_command_information, dtype=np.float64) / variance
    )
    return int(
        np.sum(np.linalg.eigvalsh(0.5 * (information + information.T)) > epsilon)
    )


def _charge_series(
    results: Sequence[DualControlResult],
    timestamps: np.ndarray,
    enable_index: int,
    stride: int = CHARGE_SERIES_STRIDE,
) -> list[dict[str, float]]:
    """The spread charge against the tracking charge, sampled along the run.

    Sampled rather than kept per interval: the two charges move on the scale of
    the flight, not of the control interval, and the full series is nine
    hundred entries per case per arm.
    """

    return [
        {
            "time_s": float(timestamps[enable_index + index + 1]),
            "tracking_cost": float(results[index].tracking_cost),
            "spread_charge": float(results[index].spread_charge),
            "command_rate_cost": float(results[index].command_rate_cost),
            "information_gain": float(results[index].information_gain),
        }
        for index in range(0, len(results), stride)
    ]


def _early_mean_collective(requested: np.ndarray, enable_index: int) -> float:
    """Mean commanded collective over the first 0.3 s after model enable.

    The third pass found this to be the quantity that tracks the outcome across
    arms, so it is measured the same way for every arm: the plain mean of every
    motor command issued in the window, as a fraction of the command range.
    """

    window = requested[enable_index : enable_index + EARLY_COMMAND_COUNT]
    return 0.0 if len(window) == 0 else float(np.mean(window))


def _first_true_step(flags: Sequence[bool]) -> int | None:
    """Index of the first true entry of an online-step series, if any."""

    found = np.flatnonzero(np.asarray(flags, dtype=bool))
    return None if len(found) == 0 else int(found[0])


def _moment(
    timestamps: np.ndarray,
    states: np.ndarray,
    enable_index: int,
    step: int | None,
) -> dict[str, Any] | None:
    """When one online step happened and what the vehicle was doing then.

    The step index is the online-loop index, whose transition ends on telemetry
    sample ``enable_index + step + 1``: the same convention every other time in
    this report uses, so a moment and a time series line up.
    """

    if step is None:
        return None
    index = enable_index + step + 1
    return {
        "step": step,
        "time_s": float(timestamps[index]),
        "time_from_enable_s": float(timestamps[index] - timestamps[enable_index]),
        "altitude_m": float(states[index, 2]),
        "descent_rate_m_s": float(-states[index, 5]),
    }


def _staging_metrics(
    record: _TrialRecord,
    timestamps: np.ndarray,
    states: np.ndarray,
    enable_index: int,
) -> dict[str, Any]:
    """When each nuisance block was admitted, and what the projection did."""

    options = record.identifier_config or {}
    counts = np.asarray(record.sign_projection_counts, dtype=int)
    magnitudes = np.asarray(record.sign_projection_magnitudes, dtype=float)
    fired = counts > 0
    fired_steps = np.flatnonzero(fired)
    return {
        "staged_regressors": bool(options.get("staged_regressors", False)),
        "staging_sample_multiple": float(options.get("staging_sample_multiple", 0.0)),
        "collective_transition": _moment(
            timestamps,
            states,
            enable_index,
            _first_true_step(record.staged_collective),
        ),
        "angular_transition": _moment(
            timestamps,
            states,
            enable_index,
            _first_true_step(record.staged_angular),
        ),
        "collective_staged_interval_fraction": (
            float(np.mean(np.asarray(record.staged_collective, dtype=bool)))
            if record.staged_collective
            else 0.0
        ),
        "angular_staged_interval_fraction": (
            float(np.mean(np.asarray(record.staged_angular, dtype=bool)))
            if record.staged_angular
            else 0.0
        ),
        "collective_sign_projection": {
            "enforced": bool(options.get("enforce_collective_sign", False)),
            "fired_interval_count": int(np.count_nonzero(fired)),
            "fired_interval_fraction": (float(np.mean(fired)) if len(counts) else 0.0),
            "maximum_projected_command_count": (
                int(np.max(counts)) if len(counts) else 0
            ),
            "maximum_magnitude_m_s2_per_command": (
                float(np.max(magnitudes)) if len(magnitudes) else 0.0
            ),
            "mean_magnitude_when_fired_m_s2_per_command": (
                float(np.mean(magnitudes[fired])) if fired.any() else 0.0
            ),
            "first_fired": _moment(
                timestamps,
                states,
                enable_index,
                None if not fired.any() else int(fired_steps[0]),
            ),
            "last_fired": _moment(
                timestamps,
                states,
                enable_index,
                None if not fired.any() else int(fired_steps[-1]),
            ),
        },
    }


def _dual_control_metrics(
    record: _TrialRecord,
    timestamps: np.ndarray,
    states: np.ndarray,
    enable_index: int,
    requested: np.ndarray,
) -> dict[str, Any]:
    """Reduce one dual-control run to what the design asks to be measured.

    The per-step information gain and log-determinant trajectories are kept in
    full, because the question this arm answers is what the optimizer did early
    rather than where it ended up.  So is the multi-start selection, which is
    the record of which declared design the objective preferred at each
    interval.
    """

    results = record.dual_results
    gains = np.asarray([result.information_gain for result in results])
    iterations = np.asarray([result.iterations for result in results])
    statuses: dict[str, int] = {}
    for result in results:
        key = str(result.status)
        statuses[key] = statuses.get(key, 0) + 1
    selections: dict[str, int] = {}
    for result in results:
        selections[result.selected_candidate] = (
            selections.get(result.selected_candidate, 0) + 1
        )
    selected_amplitudes = np.asarray([result.selected_amplitude for result in results])
    plan_amplitudes = np.asarray([result.plan_amplitude for result in results])
    planned_ranks = np.asarray(
        [result.planned_information_rank for result in results], dtype=int
    )
    information_ranks = np.asarray(record.dual_information_ranks, dtype=int)
    rank_four = np.flatnonzero(information_ranks == 4)
    early = requested[enable_index : enable_index + EARLY_COMMAND_COUNT]
    ranks = np.asarray(record.command_evidence_ranks)
    sources = [result.design_center_source for result in results]
    source_counts: dict[str, int] = {}
    for source in sources:
        source_counts[source] = source_counts.get(source, 0) + 1
    hover_centered = [source == "hover_estimate" for source in sources]
    return {
        "config": record.dual_config,
        "identifier": record.identifier_config,
        "staging": _staging_metrics(record, timestamps, states, enable_index),
        "base_action": {
            "center_source_counts": source_counts,
            "early_center_source": sources[:EARLY_COMMAND_COUNT],
            "hover_centered_handover": _moment(
                timestamps,
                states,
                enable_index,
                _first_true_step(hover_centered),
            ),
            "hover_centered_interval_fraction": (
                float(np.mean(hover_centered)) if hover_centered else 0.0
            ),
            "first_center": (results[0].design_center.tolist() if results else []),
            "uncharged_transition_count": int(
                sum(not result.charged_initial_transition for result in results)
            ),
        },
        # The quantity the third pass identified as decisive: what is commanded
        # before any model exists.  Thirds of the same window are kept because
        # the third pass read the arms apart on their shape, not only on their
        # mean.
        "early_mean_collective": _early_mean_collective(requested, enable_index),
        "early_mean_collective_thirds": [
            float(np.mean(early[start : start + 10]))
            for start in range(0, EARLY_COMMAND_COUNT, 10)
        ],
        "first_supported_model": _moment(
            timestamps,
            states,
            enable_index,
            _first_true_step(record.working_supported),
        ),
        "command_rank_four": _moment(
            timestamps,
            states,
            enable_index,
            record.command_rank_four_step,
        ),
        "multi_start": {
            "selection_counts": selections,
            "selected_candidate": [result.selected_candidate for result in results],
            "selected_amplitude_mean": (
                float(np.mean(selected_amplitudes)) if len(results) else 0.0
            ),
            "excited_candidate_count": int(np.count_nonzero(selected_amplitudes > 0.0)),
            "early_selected_candidate": [
                result.selected_candidate for result in results[:EARLY_COMMAND_COUNT]
            ],
            "early_selected_amplitude": (
                selected_amplitudes[:EARLY_COMMAND_COUNT].tolist()
            ),
            "plan_amplitude_mean": (
                float(np.mean(plan_amplitudes)) if len(results) else 0.0
            ),
            "plan_amplitude_maximum": (
                float(np.max(plan_amplitudes)) if len(results) else 0.0
            ),
            "early_plan_amplitude": plan_amplitudes[:EARLY_COMMAND_COUNT].tolist(),
            "settled_plan_amplitude_mean": (
                float(np.mean(plan_amplitudes[-SETTLED_INTERVAL_COUNT:]))
                if len(results)
                else 0.0
            ),
        },
        "planned_information_rank": planned_ranks.tolist(),
        "information_rank": information_ranks.tolist(),
        "information_rank_four_step": (
            None if len(rank_four) == 0 else int(rank_four[0])
        ),
        "information_rank_four_time_s": (
            None
            if len(rank_four) == 0
            else float(timestamps[enable_index + int(rank_four[0]) + 1])
        ),
        "charge_series": _charge_series(results, timestamps, enable_index),
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
                    "spread_charge",
                    np.asarray([result.spread_charge for result in results]),
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
                    "body_rate_penalty",
                    np.asarray([result.body_rate_penalty for result in results]),
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
    dual_config: DualControlConfig | None = None,
) -> dict[str, Any]:
    """Run one case in one control model and reduce it to study metrics.

    ``dual_config`` flies a dual-control arm on a configuration other than the
    one its name declares.  It exists for quick single-release iteration on a
    candidate design before that design earns an arm of its own; the recorded
    study never sets it.
    """

    if control_model not in STUDY_CONTROL_MODELS:
        raise ValueError(f"unknown control model {control_model!r}")
    owned = plant is None
    plant = (
        CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
        if plant is None
        else plant
    )
    try:
        record, telemetry, requested, identification, _trace = _fly_trial(
            case,
            control_model,
            plant,
            dual_config=dual_config,
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
        truncated = record.floor_contact_step is not None
        if truncated:
            # A trial stopped at floor contact has no end of flight to
            # measure: it failed, and its terminal and settled quantities are
            # absent rather than read off the contact sample.
            hover_start_s, hover_duration_s = None, 0.0

        def flown(value: float) -> float | None:
            return None if truncated else float(value)

        return {
            "control_model": control_model,
            "flight": {
                "floor_contact_time_s": (
                    None
                    if record.floor_contact_step is None
                    else float(timestamps[enable_index + record.floor_contact_step + 1])
                ),
                "terminal_speed_m_s": flown(speed[-1]),
                "terminal_angular_rate_rad_s": flown(rate[-1]),
                "terminal_tilt_rad": flown(tilt[-1]),
                "terminal_vertical_speed_m_s": flown(states[-1, 5]),
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
                "settled_readiness_loss_count": (
                    None
                    if truncated
                    else _flip_counts(supported[max(settled_step, 0) :])[0]
                ),
                "supported_interval_fraction": float(np.mean(supported)),
            },
            "stability": {
                "maximum_allocation_change": absolute_move,
                "maximum_relative_allocation_change": relative_move,
                "maximum_command_step": maximum_command_step,
                "settled_window_s": SETTLED_WINDOW_S,
                "settled_maximum_allocation_change": flown(settled_absolute),
                "settled_maximum_relative_allocation_change": flown(settled_relative),
                "settled_maximum_command_step": flown(
                    _maximum_step(requested[enable_index + max(settled_step, 0) :])
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
                        states,
                        enable_index,
                        requested,
                    )
                }
                if control_model in DUAL_CONTROL_MODEL_VARIANTS
                else {}
            ),
        }
    finally:
        if owned:
            plant.close()


def run_throw_study_render_trial(
    case: ThrowStudyCase,
    control_model: str,
    plant: CrazyflowPlant | None = None,
) -> CrazyflowStudyTrace:
    """Fly one arm on one case and return only what the throw renderer needs.

    This runs exactly the same closed loop as :func:`run_throw_study_trial`
    but skips reducing it to a report, so any of :data:`STUDY_CONTROL_MODELS`
    can be handed to the renderer without paying for metrics nothing will
    read.
    """

    if control_model not in STUDY_CONTROL_MODELS:
        raise ValueError(f"unknown control model {control_model!r}")
    owned = plant is None
    plant = (
        CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
        if plant is None
        else plant
    )
    try:
        _record, _telemetry, _requested, _identification, trace = _fly_trial(
            case,
            control_model,
            plant,
        )
        return trace
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
    def present(model: str, section: str, name: str) -> list[float]:
        values = [result["modes"][model][section][name] for result in results]
        return [float(value) for value in values if value is not None]

    def worst(model: str, section: str, name: str) -> float | None:
        values = present(model, section, name)
        return max(values) if values else None

    def best(model: str, section: str, name: str) -> float | None:
        values = present(model, section, name)
        return min(values) if values else None

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
    for dual_model in DUAL_CONTROL_MODEL_VARIANTS:
        if dual_model not in models:
            continue

        def dual(name: str, model: str = dual_model) -> list[Any]:
            return [result["modes"][model]["dual_control"][name] for result in results]

        aggregate[dual_model + "_dual_control"] = {
            "reached_command_rank_four_in_every_case": all(
                value is not None
                for value in dual("command_information_rank_four_time_s")
            ),
            "reached_information_rank_four_in_every_case": all(
                value is not None for value in dual("information_rank_four_time_s")
            ),
            "worst_command_rank_four_time_s": (
                None
                if any(
                    value is None
                    for value in dual("command_information_rank_four_time_s")
                )
                else max(dual("command_information_rank_four_time_s"))
            ),
            "unusable_command_total": sum(dual("unusable_command_count")),
            "excited_candidate_total": sum(
                entry["excited_candidate_count"] for entry in dual("multi_start")
            ),
            "worst_first_supported_time_from_enable_s": (
                None
                if any(entry is None for entry in dual("first_supported_model"))
                else max(
                    entry["time_from_enable_s"]
                    for entry in dual("first_supported_model")
                )
            ),
            "sign_projection_fired_interval_total": sum(
                entry["collective_sign_projection"]["fired_interval_count"]
                for entry in dual("staging")
            ),
            "cases_without_floor_contact": [
                result["case"]["name"]
                for result in results
                if result["modes"][dual_model]["flight"]["minimum_altitude_m"] > 0.0
            ],
        }
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


# ----------------------------------------------------------------------
# ensemble protocol
# ----------------------------------------------------------------------
#
# Seven deterministic releases cannot distinguish two designs whose early
# trajectory is chaotically sensitive to a posterior fitted from three or four
# samples: the third pass showed one clipped coefficient at interval three
# turning a recovery into a floor contact.  So a design is compared on a
# declared distribution of releases instead, with a fixed seed per draw, and
# every arm flies exactly the same draws.

#: Perturbed releases per case.  Sixteen would halve the width of the per-case
#: interval, and the pooled interval is the one that carries the comparison.
ENSEMBLE_REPLICATE_COUNT = 16
ENSEMBLE_SEED = 20260902
#: Two-sided normal quantile for a 95 percent Wilson score interval.
_WILSON_Z = 1.959963984540054


@dataclass(frozen=True)
class ThrowEnsembleConfig:
    """The declared distribution of releases, and how many are drawn.

    The release height is deliberately not perturbed: it is the altitude budget
    every arm is spending, so holding it fixed keeps the cases comparable and
    keeps the perturbation to the part of the release the arms actually differ
    on.
    """

    replicate_count: int = ENSEMBLE_REPLICATE_COUNT
    seed: int = ENSEMBLE_SEED
    #: The throw is never weaker than the case declares and may be up to a
    #: fifth stronger: a throw is the operator's act, and a release that is
    #: thrown too low to be caught by any controller measures the throw, not
    #: the controller.  The variation the ensemble is meant to resolve is in
    #: the angular impulse the release carries, so that is where the width is.
    velocity_scale_minimum: float = 1.0
    velocity_scale_maximum: float = 1.2
    angular_velocity_scale_minimum: float = 0.5
    angular_velocity_scale_maximum: float = 1.5
    maximum_tilt_perturbation_rad: float = 0.1

    def __post_init__(self) -> None:
        if self.replicate_count < 1:
            raise ValueError("an ensemble needs at least one replicate")
        if self.seed < 0:
            raise ValueError("the ensemble seed cannot be negative")
        bounds = (
            self.velocity_scale_minimum,
            self.velocity_scale_maximum,
            self.angular_velocity_scale_minimum,
            self.angular_velocity_scale_maximum,
            self.maximum_tilt_perturbation_rad,
        )
        if not np.all(np.isfinite(bounds)) or np.any(np.asarray(bounds) < 0.0):
            raise ValueError("ensemble bounds must be finite and nonnegative")
        if self.velocity_scale_minimum > self.velocity_scale_maximum:
            raise ValueError("velocity scale bounds are inverted")
        if self.angular_velocity_scale_minimum > self.angular_velocity_scale_maximum:
            raise ValueError("angular velocity scale bounds are inverted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicate_count": self.replicate_count,
            "seed": self.seed,
            "velocity_scale_range": [
                self.velocity_scale_minimum,
                self.velocity_scale_maximum,
            ],
            "angular_velocity_scale_range": [
                self.angular_velocity_scale_minimum,
                self.angular_velocity_scale_maximum,
            ],
            "maximum_tilt_perturbation_rad": self.maximum_tilt_perturbation_rad,
            "release_height_perturbed": False,
        }

    def draw(self, case_index: int, replicate_index: int) -> ThrowReleasePerturbation:
        """One release draw, determined entirely by the seed and the indices.

        Seeding from the triple rather than from a running counter means a
        single replicate reproduces on its own, and adding a case or a
        replicate never moves the draws of the others.
        """

        seed = (self.seed, case_index, replicate_index)
        generator = np.random.default_rng(seed)
        velocity = generator.uniform(
            self.velocity_scale_minimum,
            self.velocity_scale_maximum,
            3,
        )
        angular = generator.uniform(
            self.angular_velocity_scale_minimum,
            self.angular_velocity_scale_maximum,
            3,
        )
        azimuth = generator.uniform(0.0, 2.0 * math.pi)
        magnitude = generator.uniform(0.0, self.maximum_tilt_perturbation_rad)
        rotation = magnitude * np.asarray((math.cos(azimuth), math.sin(azimuth), 0.0))
        return ThrowReleasePerturbation(
            velocity_scale=tuple(velocity.tolist()),
            angular_velocity_scale=tuple(angular.tolist()),
            tilt_rotation_rad=tuple(rotation.tolist()),
            seed=seed,
        )


def wilson_interval(
    success_count: int,
    trial_count: int,
    z: float = _WILSON_Z,
) -> tuple[float, float]:
    """Wilson score interval for a binomial rate.

    The normal approximation is useless at the rates this study reports — a
    zero-of-eight recovery rate has a symmetric interval of exactly zero width —
    and the Wilson interval stays inside ``[0, 1]`` and stays non-degenerate at
    both ends, which is what makes "recovers none of these releases" and
    "recovers all of them" readable as evidence rather than as certainty.
    """

    if trial_count <= 0:
        return 0.0, 1.0
    rate = success_count / trial_count
    denominator = 1.0 + z * z / trial_count
    center = (rate + z * z / (2.0 * trial_count)) / denominator
    spread = (
        z
        * math.sqrt(rate * (1.0 - rate) / trial_count + z * z / (4.0 * trial_count**2))
        / denominator
    )
    return max(center - spread, 0.0), min(center + spread, 1.0)


def build_ensemble_cases(
    case: ThrowStudyCase,
    case_index: int,
    config: ThrowEnsembleConfig,
) -> tuple[ThrowStudyCase, ...]:
    """The perturbed releases one study case contributes to the ensemble."""

    return tuple(
        ThrowStudyCase(
            name=f"{case.name}#{replicate:02d}",
            scenario=case.scenario,
            state_noise=case.state_noise,
            configuration_change=case.configuration_change,
            release_perturbation=config.draw(case_index, replicate),
        )
        for replicate in range(config.replicate_count)
    )


def _ensemble_trial(
    case: ThrowStudyCase,
    control_model: str,
    plant: CrazyflowPlant,
    dual_controller: DualControlNMPC | None = None,
    identifier_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reduce one ensemble release to the handful of numbers it contributes.

    Deliberately far smaller than :func:`run_throw_study_trial`'s record: an
    ensemble is hundreds of flights, and the per-interval series that make the
    deterministic study readable would make this report unreadable.
    """

    record, telemetry, requested, _identification, _trace = _fly_trial(
        case,
        control_model,
        plant,
        dual_controller,
        identifier_options=identifier_options,
    )
    timestamps = telemetry.timestamp_array()
    states = telemetry.state_array()
    applied = telemetry.applied_array()
    enable_index = round(MODEL_ENABLE_DELAY_S / plant.sample_period_s)
    speed = np.linalg.norm(states[:, 3:6], axis=1)
    rate = np.linalg.norm(states[:, 10:13], axis=1)
    tilt = np.asarray([tilt_rad(state) for state in states])
    hover_start_s, hover_duration_s = _sustained_hover_duration_s(
        timestamps,
        speed,
        rate,
        states[:, 5],
        tilt,
        enable_index,
    )
    minimum_altitude_m = float(np.min(states[:, 2]))
    truncated = record.floor_contact_step is not None
    if truncated:
        hover_start_s, hover_duration_s = None, 0.0

    def flown(value: float) -> float | None:
        return None if truncated else float(value)

    online_step_count = len(record.working_supported)
    settled_step = max(
        online_step_count - round(SETTLED_WINDOW_S / plant.sample_period_s),
        0,
    )
    settled_absolute, settled_relative = _allocation_changes(
        record.flown_allocations,
        settled_step,
    )
    minimum = np.asarray(RecursiveBootstrapConfig().command_minimum)
    maximum = np.asarray(RecursiveBootstrapConfig().command_maximum)
    return {
        "case": case.name,
        "control_model": control_model,
        "simulator_diverged": False,
        # The success criterion the design page states: the hover envelope
        # reached, and the vehicle never on the floor.  A vehicle resting on the
        # ground satisfies the envelope, so the altitude test is what makes the
        # rate mean anything.
        "recovered": bool(hover_start_s is not None and not truncated),
        "reached_hover_envelope": hover_start_s is not None,
        "touched_floor": truncated,
        "floor_contact_time_s": (
            None
            if record.floor_contact_step is None
            else float(timestamps[enable_index + record.floor_contact_step + 1])
        ),
        "terminal_speed_m_s": flown(speed[-1]),
        "terminal_angular_rate_rad_s": flown(rate[-1]),
        "terminal_tilt_rad": flown(tilt[-1]),
        "minimum_altitude_m": minimum_altitude_m,
        "sustained_hover_duration_s": hover_duration_s,
        "time_to_rank_four_s": (
            None
            if record.command_rank_four_step is None
            else float(
                timestamps[enable_index + record.command_rank_four_step + 1]
                - timestamps[enable_index]
            )
        ),
        "early_mean_collective": _early_mean_collective(requested, enable_index),
        "settled_maximum_command_step": flown(
            _maximum_step(requested[enable_index + settled_step :])
        ),
        "settled_maximum_relative_allocation_change": flown(settled_relative),
        "settled_maximum_allocation_change": flown(settled_absolute),
        "non_finite_value_count": int(
            np.count_nonzero(~np.isfinite(states))
            + np.count_nonzero(~np.isfinite(applied))
            + np.count_nonzero(~np.isfinite(requested))
        ),
        "command_bound_violation_count": int(
            np.count_nonzero(requested < minimum - COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(requested > maximum + COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(applied < minimum - COMMAND_BOUND_TOLERANCE)
            + np.count_nonzero(applied > maximum + COMMAND_BOUND_TOLERANCE)
        ),
    }


def _diverged_ensemble_trial(
    case: ThrowStudyCase,
    control_model: str,
    reason: str,
) -> dict[str, Any]:
    """One release whose simulation could not be integrated to the end.

    A trial that ends in a non-finite simulator state is a result, not an
    accident of the harness, and losing the other five hundred trials to it
    would be the accident.  It is recorded with the same keys every other trial
    carries, with every measured quantity absent rather than invented: it did
    not recover, and nothing else about it is known.  ``simulator_diverged``
    and the arm's own ``all_values_finite_and_bounded`` are what carry it into
    the report, so a diverged arm cannot pass the finiteness criterion.
    """

    return {
        "case": case.name,
        "control_model": control_model,
        "simulator_diverged": True,
        "divergence_reason": reason,
        "recovered": False,
        "reached_hover_envelope": False,
        "touched_floor": False,
        "floor_contact_time_s": None,
        "terminal_speed_m_s": None,
        "terminal_angular_rate_rad_s": None,
        "terminal_tilt_rad": None,
        "minimum_altitude_m": None,
        "sustained_hover_duration_s": None,
        "time_to_rank_four_s": None,
        "early_mean_collective": None,
        "settled_maximum_command_step": None,
        "settled_maximum_relative_allocation_change": None,
        "settled_maximum_allocation_change": None,
        "non_finite_value_count": None,
        "command_bound_violation_count": None,
    }


def _spread(values: Sequence[float | None], worst: str) -> dict[str, Any]:
    """Median and worst of one metric over an ensemble, ignoring absences.

    ``worst`` names the direction: ``"maximum"`` for a quantity a design wants
    small, ``"minimum"`` for one it wants large.
    """

    present = [float(value) for value in values if value is not None]
    if not present:
        return {"median": None, "worst": None, "available_count": 0}
    array = np.asarray(present)
    return {
        "median": float(np.median(array)),
        "worst": float(np.max(array) if worst == "maximum" else np.min(array)),
        "available_count": len(present),
    }


def _ensemble_summary(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the trials of one arm, on one case or pooled, to a summary."""

    count = len(trials)
    recovered = sum(trial["recovered"] for trial in trials)
    low, high = wilson_interval(recovered, count)
    diverged = sum(bool(trial.get("simulator_diverged")) for trial in trials)
    collectives = np.asarray(
        [
            trial["early_mean_collective"]
            for trial in trials
            if trial["early_mean_collective"] is not None
        ]
    )
    return {
        "trial_count": count,
        "recovery_count": recovered,
        "recovery_rate": (recovered / count if count else 0.0),
        "recovery_rate_wilson_95": [low, high],
        "simulator_diverged_count": diverged,
        "floor_contact_count": sum(trial["touched_floor"] for trial in trials),
        "hover_envelope_count": sum(
            trial["reached_hover_envelope"] for trial in trials
        ),
        "terminal_speed_m_s": _spread(
            [trial["terminal_speed_m_s"] for trial in trials], "maximum"
        ),
        "terminal_angular_rate_rad_s": _spread(
            [trial["terminal_angular_rate_rad_s"] for trial in trials], "maximum"
        ),
        "terminal_tilt_rad": _spread(
            [trial["terminal_tilt_rad"] for trial in trials], "maximum"
        ),
        "minimum_altitude_m": _spread(
            [trial["minimum_altitude_m"] for trial in trials], "minimum"
        ),
        "time_to_rank_four_s": _spread(
            [trial["time_to_rank_four_s"] for trial in trials], "maximum"
        ),
        "early_mean_collective": {
            "mean": (float(np.mean(collectives)) if collectives.size else 0.0),
            "median": (float(np.median(collectives)) if collectives.size else 0.0),
            "minimum": (float(np.min(collectives)) if collectives.size else 0.0),
            "maximum": (float(np.max(collectives)) if collectives.size else 0.0),
            "available_count": int(collectives.size),
        },
        "settled_maximum_command_step": _spread(
            [trial["settled_maximum_command_step"] for trial in trials], "maximum"
        ),
        "settled_maximum_relative_allocation_change": _spread(
            [trial["settled_maximum_relative_allocation_change"] for trial in trials],
            "maximum",
        ),
        "all_values_finite_and_bounded": all(
            trial["non_finite_value_count"] == 0
            and trial["command_bound_violation_count"] == 0
            for trial in trials
        ),
    }


def _ensemble_jobs(
    cases: Sequence[ThrowStudyCase],
    control_models: Sequence[str],
    config: ThrowEnsembleConfig,
) -> list[tuple[str, ThrowStudyCase, str]]:
    """Every (base case, release, arm) trial the ensemble runs, in a fixed order.

    The base case name is carried alongside rather than recovered from the draw's
    own name, so the per-case grouping of the results is the same structure that
    generated them.  Jobs are grouped by arm rather than by release so that a
    worker which has compiled one arm's solve program runs a long stretch of
    that arm before paying for the next one.
    """

    jobs: list[tuple[str, ThrowStudyCase, str]] = []
    for model in control_models:
        for case_index, case in enumerate(cases):
            jobs.extend(
                (case.name, draw, model)
                for draw in build_ensemble_cases(case, case_index, config)
            )
    return jobs


def _run_ensemble_jobs(
    jobs: Sequence[tuple[ThrowStudyCase, str]],
    identifier_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run a chunk of ensemble trials on one plant, reusing compiled arms.

    Module level and picklable so a process pool can call it.  The plant is
    reset for every trial and carries nothing across them — measured, not
    assumed — so a chunk's results do not depend on how the chunk was cut.
    """

    os.environ.setdefault("SCIPY_ARRAY_API", "1")
    plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
    controllers: dict[str, DualControlNMPC] = {}
    try:
        results = []
        for case, model in jobs:
            controller = None
            variant = DUAL_CONTROL_MODEL_VARIANTS.get(model)
            if variant is not None:
                controller = controllers.get(model)
                if controller is None:
                    controller = DualControlNMPC(
                        dual_control_config(
                            variant,
                            sample_period_s=plant.sample_period_s,
                        )
                    )
                    controllers[model] = controller
            try:
                results.append(
                    _ensemble_trial(case, model, plant, controller, identifier_options)
                )
            except CrazyflowDivergenceError as error:
                # The plant refuses a non-finite simulator state rather than
                # returning one, which is right.  Here it means this release
                # could not be integrated to the end; the trial is recorded as
                # diverged and the plant is rebuilt so the rest of the chunk
                # starts from a clean simulator.  Only that refusal is caught:
                # any other error in a trial is a defect and still ends the run.
                results.append(_diverged_ensemble_trial(case, model, str(error)))
                plant.close()
                plant = CrazyflowPlant(CrazyflowPlantConfig(control_frequency_hz=100))
        return results
    finally:
        plant.close()


def run_crazyflow_throw_ensemble(
    cases: Sequence[ThrowStudyCase] = CRAZYFLOW_THROW_STUDY_CASES,
    control_models: Sequence[str] = ENSEMBLE_CONTROL_MODELS,
    config: ThrowEnsembleConfig | None = None,
    worker_count: int = 1,
    identifier_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fly every arm on the same declared distribution of releases.

    The report is per case and arm, plus one pooled rate per arm over every
    release: a recovery rate with a Wilson interval, the median and worst of
    each flight-quality metric, the time to command rank four, the first-0.3 s
    mean collective, and the settled chatter.  Every arm sees the same draws, so
    the comparison is paired.

    ``worker_count`` shards the trials across processes.  Each trial is decided
    entirely by its own seeded release and its arm, and the results are
    reassembled in the job order rather than the completion order, so the report
    is identical whatever the worker count.
    """

    settings = ThrowEnsembleConfig() if config is None else config
    if not cases:
        raise ValueError("the throw ensemble needs at least one case")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("throw study case names must be unique")
    models = tuple(control_models)
    if not models:
        raise ValueError("the throw ensemble needs at least one control model")
    unknown = sorted(set(models) - set(STUDY_CONTROL_MODELS))
    if unknown:
        raise ValueError(f"unknown control model(s): {', '.join(unknown)}")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")

    jobs = _ensemble_jobs(cases, models, settings)
    work = [(draw, model) for _base, draw, model in jobs]
    if worker_count == 1:
        trials = _run_ensemble_jobs(work, identifier_options)
    else:
        from concurrent.futures import ProcessPoolExecutor

        chunks = [work[index::worker_count] for index in range(worker_count)]
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            outputs = list(
                pool.map(
                    functools.partial(
                        _run_ensemble_jobs, identifier_options=identifier_options
                    ),
                    chunks,
                )
            )
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk in outputs:
            for trial in chunk:
                merged[(trial["case"], trial["control_model"])] = trial
        trials = [merged[(draw.name, model)] for draw, model in work]

    by_arm: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    by_case_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (base, _draw, model), trial in zip(jobs, trials, strict=True):
        by_arm[model].append(trial)
        by_case_arm.setdefault((base, model), []).append(trial)
    case_entries = [
        {
            "case": case.to_dict(),
            "releases": [
                draw.release_perturbation.to_dict()
                for draw in build_ensemble_cases(case, case_index, settings)
                if draw.release_perturbation is not None
            ],
            "arms": {
                model: _ensemble_summary(by_case_arm[(case.name, model)])
                for model in models
            },
        }
        for case_index, case in enumerate(cases)
    ]
    report = {
        "artifact_type": "glassbox_crazyflow_throw_release_ensemble",
        "schema_version": 1,
        "semantics": {
            "diagnostic_only": True,
            "flight_safety_claim": False,
            "deterministic_given_the_seed": True,
            "every_arm_flies_the_same_releases": True,
            "release_height_unperturbed": True,
            "recovered_means_hover_envelope_without_floor_contact": True,
            "trials_stop_at_first_floor_contact": True,
            "diverged_trials_recorded_not_dropped": True,
        },
        "ensemble": settings.to_dict(),
        "control_models": list(models),
        "case_count": len(cases),
        "releases_per_arm": len(cases) * settings.replicate_count,
        "trial_count": len(trials),
        "cases": case_entries,
        "pooled": {model: _ensemble_summary(by_arm[model]) for model in models},
        "trials": trials,
        "limitations": [
            "The perturbation is a declared distribution over the release state only; the hidden airframe, the loop rate, and every controller setting are unchanged.",
            "Recovery is a binary read of one envelope on one ten-second window, so a marginal arrest and a comfortable one score the same.",
            "The state-noise case draws one noise realisation per release, not a distribution over realisations.",
            "A release the simulator could not integrate to the end is counted as not recovered and contributes nothing to any other statistic; `simulator_diverged_count` per arm is what makes those releases visible.",
        ],
    }
    json.dumps(report, allow_nan=False)
    return report


_ENSEMBLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("terminal speed", "terminal_speed_m_s", "{:.3f}"),
    ("terminal rate", "terminal_angular_rate_rad_s", "{:.3f}"),
    ("terminal tilt", "terminal_tilt_rad", "{:.4f}"),
    ("min alt", "minimum_altitude_m", "{:.3f}"),
    ("rank four s", "time_to_rank_four_s", "{:.2f}"),
    ("set cmd step", "settled_maximum_command_step", "{:.4f}"),
)


def format_ensemble_table(report: dict[str, Any]) -> str:
    """Render the per-case, per-arm ensemble summary as a markdown table."""

    header = [
        "case",
        "arm",
        "recovered",
        "rate",
        "wilson 95",
        "early collective",
        *(f"{name} med/worst" for name, _, _ in _ENSEMBLE_COLUMNS),
    ]
    rows = [header, ["---"] * len(header)]
    entries = [(entry["case"]["name"], entry["arms"]) for entry in report["cases"]] + [
        ("pooled", report["pooled"])
    ]
    for name, arms in entries:
        for model in report["control_models"]:
            summary = arms[model]
            low, high = summary["recovery_rate_wilson_95"]
            row = [
                name,
                model,
                f"{summary['recovery_count']}/{summary['trial_count']}",
                f"{summary['recovery_rate']:.2f}",
                f"{low:.2f}-{high:.2f}",
                f"{summary['early_mean_collective']['mean']:.3f}",
            ]
            for _, key, template in _ENSEMBLE_COLUMNS:
                spread = summary[key]
                row.append(
                    "n/a"
                    if spread["median"] is None
                    else (
                        template.format(spread["median"])
                        + "/"
                        + template.format(spread["worst"])
                    )
                )
            rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    return "\n".join(
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rows
    )


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
            f"(default: {' '.join(DEFAULT_CONTROL_MODELS)})"
        ),
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "fly every arm on a seeded distribution of perturbed releases "
            "instead of the single deterministic release per case"
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=ENSEMBLE_REPLICATE_COUNT,
        help="perturbed releases per case in ensemble mode (default: %(default)s)",
    )
    parser.add_argument(
        "--ensemble-seed",
        type=int,
        default=ENSEMBLE_SEED,
        help="seed the ensemble draws come from (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "processes to shard ensemble trials across; the report is identical "
            "at any worker count (default: %(default)s)"
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
    if args.ensemble:
        models = (
            tuple(args.control_model) if args.control_model else ENSEMBLE_CONTROL_MODELS
        )
        report = run_crazyflow_throw_ensemble(
            cases,
            models,
            ThrowEnsembleConfig(
                replicate_count=args.replicates,
                seed=args.ensemble_seed,
            ),
            worker_count=args.workers,
        )
        table = format_ensemble_table(report)
    else:
        models = (
            tuple(args.control_model) if args.control_model else DEFAULT_CONTROL_MODELS
        )
        report = run_crazyflow_throw_study(cases, models)
        table = format_study_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(table)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
