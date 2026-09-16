"""Experimental, single-writer coordination for scoring telemetry before learning.

This is a session-scoped replay workflow, not a vehicle link or scheduler.
Recordings must have stable identities and contiguous, non-overlapping control
intervals. Each block is forecast with pinned revisions before it is absorbed.
The application chooses whether and when to adopt a scored revision.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from glassbox.belief.belief import DynamicsBelief, PredictiveTrajectory
from glassbox.belief.update import UpdateResult
from glassbox.core.data import Trajectory, trajectory_content_digest
from glassbox.core.metrics import state_rmse_metrics
from glassbox.core.model import DirectActuationMap


@dataclass(frozen=True)
class ModelRevision:
    """A belief snapshot identified within one refinement session."""

    revision_id: str
    belief: DynamicsBelief

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "update_count": self.belief.update_count,
            "parameter_distance_since_measurement": (
                self.belief.parameter_distance_since_measurement
            ),
            "parameter_information_rank": self.belief.information.resolved_rank(),
            "parameter_information_complete": self.belief.parameter_information_complete,
            "forecast_error_available": self.belief.forecast_error_available,
            "maximum_error_horizon_s": self.belief.maximum_error_horizon_s,
        }


@dataclass(frozen=True)
class ForecastScore:
    """A conditional command replay, initialized only at the block's first state.

    The supplied commands and exogenous inputs were observed during the block.
    This measures dynamics prediction error, not a prospective closed-loop forecast.
    The shared initial state is excluded from the score.
    """

    revision: ModelRevision
    prediction: PredictiveTrajectory
    rmse: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        utilization = np.asarray(self.prediction.validity_utilization)
        return {
            "revision": self.revision.to_dict(),
            "finite_state_forecast": bool(np.isfinite(self.prediction.states).all()),
            "rmse": None if self.rmse is None else dict(self.rmse),
            "parameter_covariance_computed": self.prediction.parameter_covariance
            is not None,
            "forecast_error_horizon_supported": (
                self.prediction.forecast_error_horizon_supported
            ),
            "maximum_validity_utilization": (
                float(utilization.max()) if np.isfinite(utilization).all() else None
            ),
        }


@dataclass(frozen=True)
class RefinementResult:
    """One scored block and the distinct revision produced by learning from it."""

    recording_id: str
    start_interval: int
    stop_interval: int
    content_sha256: str
    horizon_s: float
    active_score: ForecastScore
    candidate_score: ForecastScore
    candidate_after: ModelRevision
    update: UpdateResult
    command_history_steps: int = 0
    command_history_truncated: bool = False

    def actuator_memory_fraction(self, revision: ModelRevision) -> float:
        """Fraction of an initial actuator offset retained under this model.

        This exponential decay factor describes the model's initialization
        assumption, not a calibrated probability or a bound on plant mismatch.
        """
        constants = np.asarray(revision.belief.model.latent_response_time_constants_s)
        if not len(constants):
            return 0.0
        memory = np.zeros_like(constants)
        positive = constants > 0
        memory[positive] = np.exp(
            -self.command_history_steps
            * revision.belief.sample_period_s
            / constants[positive]
        )
        return float(np.max(memory))

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "start_interval": self.start_interval,
            "stop_interval": self.stop_interval,
            "content_sha256": self.content_sha256,
            "horizon_s": self.horizon_s,
            "active_score": self.active_score.to_dict(),
            "candidate_score": self.candidate_score.to_dict(),
            "candidate_after": self.candidate_after.to_dict(),
            "update": self.update.to_dict(),
            "command_history_steps": self.command_history_steps,
            "command_history_truncated": self.command_history_truncated,
            "actuator_memory_fraction": {
                "active": self.actuator_memory_fraction(self.active_score.revision),
                "candidate": self.actuator_memory_fraction(
                    self.candidate_score.revision
                ),
            },
        }


@dataclass(frozen=True)
class Adoption:
    previous_revision_id: str
    adopted_revision_id: str
    after_block_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class _RecordingCursor:
    stop_interval: int
    preceding_commands: np.ndarray | None
    last_state: np.ndarray | None
    last_exogenous: np.ndarray | None


class ModelRefiner:
    """Score fresh telemetry, refine a candidate, and explicitly adopt snapshots.

    Call methods from one owning worker. ``active`` and ``candidate`` return
    snapshots; learning never replaces ``active``. A controller may pin an
    active belief separately, but this class does not rebind a controller.

    Defaults retain full command history and all results. ``history_steps``,
    ``retained_blocks`` and ``max_recordings`` bound these independently; supply
    all three to bound session retention. A capped history is an explicitly
    reported actuator initialization approximation. Retired model revisions
    cannot be adopted unless held for an outstanding decision.

    There is no durable restart protocol, forgetting, automatic admission
    policy, or forecast-error recalibration. Callers must not mutate nested
    metadata on beliefs they have handed to the session.
    """

    def __init__(
        self,
        belief: DynamicsBelief,
        *,
        session_id: str | None = None,
        history_steps: int | None = None,
        retained_blocks: int | None = None,
        max_recordings: int | None = None,
        propagate_parameter_covariance: bool = True,
    ) -> None:
        if not isinstance(belief, DynamicsBelief):
            raise TypeError("belief must be a DynamicsBelief")
        if not isinstance(belief.model.actuation, DirectActuationMap) or (
            belief.model.actuation.command_channels != belief.input_spec.controls
        ):
            raise ValueError(
                "refinement requires a direct command model with matching telemetry "
                "coordinates; measured-actuation and custom maps are not supported"
            )
        self.session_id = uuid4().hex if session_id is None else session_id
        _require_name(self.session_id, "session_id")
        for label, value in (
            ("history_steps", history_steps),
            ("retained_blocks", retained_blocks),
            ("max_recordings", max_recordings),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{label} must be a positive integer or None")
        self.history_steps = history_steps
        if not isinstance(propagate_parameter_covariance, bool):
            raise TypeError("propagate_parameter_covariance must be a bool")
        self.propagate_parameter_covariance = propagate_parameter_covariance
        self.max_recordings = max_recordings
        self.block_count = 0
        self.skipped_interval_count = 0
        self._next_revision = 1
        self._adoption_hold: str | None = None
        self._active = self._candidate = ModelRevision(f"{self.session_id}:0", belief)
        self._revisions = {self._active.revision_id: self._active}
        self._scored: set[str] = set()
        self._recordings: dict[str, _RecordingCursor] = {}
        self._content: set[str] = set()
        self._results: deque[RefinementResult] = deque(maxlen=retained_blocks)
        self._adoptions: deque[Adoption] = deque(maxlen=retained_blocks)
        # This catches exact reuse of a fitted source. Different segmentation
        # is not a proof of independence; source identity remains caller-owned.
        identity = belief.provenance.get("data_identity", {})
        if identity.get("algorithm") == "trajectory_sha256_v1":
            for role in ("training", "forecast_error_calibration"):
                self._content.update(identity.get(role, ()))
        self._reserved_content = frozenset(self._content)

    @property
    def retained_revision_count(self) -> int:
        return len(self._revisions)

    def hold_for_adoption(self, revision_id: str) -> None:
        """Retain one evaluated snapshot while a consumer prepares its decision."""
        if self._adoption_hold is not None:
            raise ValueError("an adoption decision is already outstanding")
        if revision_id not in self._scored:
            raise ValueError("only an evaluated revision can be held for adoption")
        self._adoption_hold = revision_id

    def release_adoption_hold(self) -> None:
        self._adoption_hold = None
        self._prune()

    def skip(
        self, *, recording_id: str, start_interval: int, stop_interval: int, reason: str
    ) -> dict[str, Any]:
        """Account for missing intervals explicitly, without inventing transitions.

        A gap breaks endpoint/history continuity. The next block supplies its
        own known preceding commands. Persist the returned event in the caller's
        journal; only the cumulative skipped count is retained here.
        """
        _require_name(recording_id, "recording_id")
        _require_name(reason, "reason")
        self._check_recording_limit(recording_id)
        previous = self._recordings.get(recording_id)
        expected = 0 if previous is None else previous.stop_interval
        if (
            any(
                isinstance(v, bool) or not isinstance(v, int)
                for v in (start_interval, stop_interval)
            )
            or start_interval != expected
            or stop_interval <= start_interval
        ):
            raise ValueError(
                "gap must advance the recording cursor by a positive integer interval count"
            )
        self._recordings[recording_id] = _RecordingCursor(
            stop_interval, None, None, None
        )
        self.skipped_interval_count += stop_interval - start_interval
        return {
            "recording_id": recording_id,
            "start_interval": start_interval,
            "stop_interval": stop_interval,
            "reason": reason,
        }

    def _check_recording_limit(self, recording_id: str) -> None:
        if (
            recording_id not in self._recordings
            and self.max_recordings is not None
            and len(self._recordings) >= self.max_recordings
        ):
            raise ValueError(
                "recording retention limit reached; use a separate session"
            )

    def _prune(self) -> None:
        if self._results.maxlen is None:
            return
        keep = {self._active.revision_id, self._candidate.revision_id}
        if self._adoption_hold is not None:
            keep.add(self._adoption_hold)
        for result in self._results:
            keep.update(
                (
                    result.active_score.revision.revision_id,
                    result.candidate_score.revision.revision_id,
                    result.candidate_after.revision_id,
                )
            )
        self._revisions = {
            key: value for key, value in self._revisions.items() if key in keep
        }
        self._scored.intersection_update(keep)
        self._content = set(self._reserved_content) | {
            result.content_sha256 for result in self._results
        }

    @property
    def active(self) -> ModelRevision:
        return self._active

    @property
    def candidate(self) -> ModelRevision:
        return self._candidate

    @property
    def results(self) -> tuple[RefinementResult, ...]:
        return tuple(self._results)

    @property
    def adoptions(self) -> tuple[Adoption, ...]:
        return tuple(self._adoptions)

    def observe(
        self, telemetry: Trajectory, *, recording_id: str, start_interval: int
    ) -> RefinementResult:
        """Score both incoming revisions, then absorb once into the candidate.

        Intervals are indexed from zero within each recording, independent of
        a segment's rebased timestamps. Later blocks must preserve the shared
        endpoint and command prefix, capped to ``history_steps`` when specified.
        ``skip`` explicitly breaks continuity across unavailable intervals.
        An update refusal consumes the scored block; an exception commits no
        session state and permits a corrected retry.
        """

        digest, cursor = self._validate(telemetry, recording_id, start_interval)
        active, candidate = self._active, self._candidate
        active_score = _score(
            active,
            telemetry,
            propagate_parameter_covariance=self.propagate_parameter_covariance,
        )
        candidate_score = (
            active_score
            if active is candidate
            else _score(
                candidate,
                telemetry,
                propagate_parameter_covariance=self.propagate_parameter_covariance,
            )
        )
        updated, update = candidate.belief.absorb(telemetry)
        successor = (
            ModelRevision(f"{self.session_id}:{self._next_revision}", updated)
            if update.absorbed
            else candidate
        )
        result = RefinementResult(
            recording_id=recording_id,
            start_interval=int(start_interval),
            stop_interval=cursor.stop_interval,
            content_sha256=digest,
            horizon_s=len(telemetry.controls) * candidate.belief.sample_period_s,
            active_score=active_score,
            candidate_score=candidate_score,
            candidate_after=successor,
            update=update,
            command_history_steps=0
            if telemetry.control_prefix is None
            else len(telemetry.control_prefix),
            command_history_truncated=self.history_steps is not None
            and start_interval
            > (
                0 if telemetry.control_prefix is None else len(telemetry.control_prefix)
            ),
        )
        # All numerical work succeeds before advancing any session state.
        self._scored.update((active.revision_id, candidate.revision_id))
        self._revisions[successor.revision_id] = successor
        self._candidate = successor
        self._recordings[recording_id] = cursor
        self._content.add(digest)
        self._results.append(result)
        self.block_count += 1
        self._next_revision += int(update.absorbed)
        self._prune()
        return result

    def adopt(
        self, revision_id: str, *, expected_active_revision: str, reason: str
    ) -> Adoption:
        """Select a previously scored revision using an application decision.

        ``result.candidate_score.revision`` is the evaluated snapshot;
        ``result.candidate_after`` has just learned and is not yet scored.
        The expected active ID rejects a decision based on a superseded active
        model. This is not a thread synchronization or model-readiness check.
        Older scored revisions remain selectable for explicit rollback.
        """

        _require_name(reason, "reason")
        if expected_active_revision != self._active.revision_id:
            raise ValueError(
                "active revision changed; reconsider the adoption decision"
            )
        if revision_id not in self._scored:
            raise ValueError(
                "adoption requires a revision scored on a subsequent block"
            )
        adoption = Adoption(
            self._active.revision_id, revision_id, self.block_count, reason
        )
        self._active = self._revisions[revision_id]
        self._adoptions.append(adoption)
        return adoption

    def _validate(
        self, telemetry: Trajectory, recording_id: str, start_interval: int
    ) -> tuple[str, _RecordingCursor]:
        if not isinstance(telemetry, Trajectory):
            raise TypeError("telemetry must be a Trajectory")
        _require_name(recording_id, "recording_id")
        self._check_recording_limit(recording_id)
        if isinstance(start_interval, bool) or not isinstance(
            start_interval, (int, np.integer)
        ):
            raise TypeError("start_interval must be an integer")
        previous = self._recordings.get(recording_id)
        expected_start = 0 if previous is None else previous.stop_interval
        if start_interval != expected_start:
            raise ValueError(
                f"recording {recording_id!r}: expected start_interval {expected_start}; "
                "overlaps, gaps, and out-of-order blocks are not supported"
            )
        belief = self._candidate.belief
        if telemetry.spec.prediction_spec() != belief.input_spec.prediction_spec():
            raise ValueError(
                "telemetry prediction contract or vehicle configuration changed"
            )
        if not np.allclose(
            np.diff(telemetry.time_s), belief.sample_period_s, rtol=1e-6, atol=1e-9
        ):
            raise ValueError("telemetry must match the model's fixed sample period")
        prefix = telemetry.control_prefix
        if prefix is None:
            prefix = np.empty((0, telemetry.controls.shape[1]))
        if self.history_steps is not None and len(prefix) > self.history_steps:
            raise ValueError("telemetry command prefix exceeds history_steps")
        if start_interval > 0 and len(prefix) == 0:
            raise ValueError(
                "telemetry after recording start requires preceding commands"
            )
        if previous is not None and previous.preceding_commands is not None:
            if not np.array_equal(prefix, previous.preceding_commands):
                raise ValueError(
                    "telemetry must preserve the recording's full command prefix"
                )
            if not np.array_equal(telemetry.states[0], previous.last_state):
                raise ValueError("telemetry shared endpoint state changed")
            if not np.array_equal(telemetry.exogenous[0], previous.last_exogenous):
                raise ValueError("telemetry shared endpoint exogenous inputs changed")
        digest = trajectory_content_digest(telemetry)
        if digest in self._content:
            raise ValueError(
                "telemetry content was already seen or reserved during fitting"
            )
        return digest, _RecordingCursor(
            stop_interval=int(start_interval) + len(telemetry.controls),
            preceding_commands=(
                np.concatenate((prefix, telemetry.controls))
                if self.history_steps is None
                else np.concatenate((prefix, telemetry.controls))[-self.history_steps :]
            ),
            last_state=telemetry.states[-1],
            last_exogenous=telemetry.exogenous[-1],
        )


def _require_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _score(
    revision: ModelRevision,
    telemetry: Trajectory,
    *,
    propagate_parameter_covariance: bool = True,
) -> ForecastScore:
    prediction = revision.belief.rollout(
        telemetry.states[0],
        telemetry.controls,
        command_history=telemetry.control_prefix,
        exogenous=telemetry.exogenous[:-1],
        propagate_parameter_covariance=propagate_parameter_covariance,
    )
    states = np.asarray(prediction.states)
    rmse = None
    if np.isfinite(states).all():
        measured = state_rmse_metrics(states[1:], telemetry.states[1:])
        if all(np.isfinite(value) for value in measured.values()):
            rmse = measured
    return ForecastScore(revision, prediction, rmse)
