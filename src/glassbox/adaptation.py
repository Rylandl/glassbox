"""Fail-closed, evidence-preserving updates for structured dynamics beliefs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    LocalGaussianParameterBelief,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.covariance import SupportedCovariance, supported_covariance
from glassbox.data import Trajectory, duration_to_steps
from glassbox.dynamics import control_state_after_history, step_with_latent
from glassbox.linearization import (
    compiled_endpoint_tangent_error,
    compiled_endpoint_tangent_linearization,
)
from glassbox.runtime import model_validity_utilization

MAXIMUM_ONLINE_UPDATE_WINDOWS = 64
ACTUATOR_HISTORY_DURATION_S = 1.0
MAXIMUM_LOCAL_PARAMETER_STEP_RMS = 1.0
VALIDITY_BOUNDARY_TOLERANCE = 1e-6
LINE_SEARCH_FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class BeliefUpdateProposal:
    """A parameter move that is not authoritative until separately validated."""

    base_parameter_vector: np.ndarray
    base_parameter_covariance: np.ndarray
    candidate_parameter_vector: np.ndarray
    normalized_information_matrix: np.ndarray
    covariance_scope: ErrorCovarianceScope
    base_update_count: int
    update_horizon_s: float
    update_horizon_steps: int
    proposal_window_count: int
    normalized_innovation_rms_before: float
    normalized_innovation_rms_after: float
    prior_standardized_step_rms: float
    proposal_step_fraction: float
    maximum_validity_utilization: float
    source_group: str
    evidence_transition_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        base = np.asarray(self.base_parameter_vector, dtype=np.float64)
        candidate = np.asarray(self.candidate_parameter_vector, dtype=np.float64)
        covariance = np.asarray(self.base_parameter_covariance, dtype=np.float64)
        information = np.asarray(
            self.normalized_information_matrix,
            dtype=np.float64,
        )
        size = len(base)
        if candidate.shape != (size,) or covariance.shape != (size, size):
            raise ValueError("update proposal parameter geometry is incompatible")
        if information.ndim != 2 or information.shape[0] != information.shape[1]:
            raise ValueError("update proposal information must be square")
        if not all(
            np.all(np.isfinite(value))
            for value in (base, candidate, covariance, information)
        ):
            raise ValueError("update proposal must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-9):
            raise ValueError("update proposal covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
            raise ValueError("update proposal covariance must be PSD")
        if not np.allclose(information, information.T, atol=1e-9):
            raise ValueError("update proposal information must be symmetric")
        if np.min(np.linalg.eigvalsh(information)) < -1e-8:
            raise ValueError("update proposal information must be PSD")
        if information.shape != (
            supported_covariance(covariance).rank,
            supported_covariance(covariance).rank,
        ):
            raise ValueError("update proposal information has incompatible rank")
        if self.base_update_count < 0:
            raise ValueError("update proposal count cannot be negative")
        if self.update_horizon_s <= 0.0 or self.update_horizon_steps < 1:
            raise ValueError("update proposal horizon must be positive")
        if self.proposal_window_count < 1:
            raise ValueError("update proposal requires evidence windows")
        if not (
            np.isfinite(self.normalized_innovation_rms_before)
            and np.isfinite(self.normalized_innovation_rms_after)
            and self.normalized_innovation_rms_after
            < self.normalized_innovation_rms_before
        ):
            raise ValueError("update proposal must improve its proposal evidence")
        if not (
            np.isfinite(self.prior_standardized_step_rms)
            and 0.0 < self.prior_standardized_step_rms
            <= MAXIMUM_LOCAL_PARAMETER_STEP_RMS * (1.0 + 1e-9)
        ):
            raise ValueError("update proposal exceeds the local trust region")
        if not (
            np.isfinite(self.proposal_step_fraction)
            and 0.0 < self.proposal_step_fraction <= 1.0
        ):
            raise ValueError("update proposal step fraction is invalid")
        if not (
            np.isfinite(self.maximum_validity_utilization)
            and self.maximum_validity_utilization
            <= 1.0 + VALIDITY_BOUNDARY_TOLERANCE
        ):
            raise ValueError("update proposal lies outside model support")
        if not self.source_group.strip():
            raise ValueError("update proposal source group is required")
        transition_hashes = tuple(str(value) for value in self.evidence_transition_hashes)
        if not transition_hashes or len(set(transition_hashes)) != len(
            transition_hashes
        ) or any(len(value) != 64 for value in transition_hashes):
            raise ValueError("update proposal transition evidence is invalid")
        object.__setattr__(
            self,
            "covariance_scope",
            ErrorCovarianceScope(self.covariance_scope),
        )
        object.__setattr__(self, "base_parameter_vector", base)
        object.__setattr__(self, "base_parameter_covariance", covariance)
        object.__setattr__(self, "candidate_parameter_vector", candidate)
        object.__setattr__(self, "normalized_information_matrix", information)
        object.__setattr__(self, "evidence_transition_hashes", transition_hashes)


@dataclass(frozen=True)
class BeliefUpdateReport:
    """Audit record for one attempted immutable belief update."""

    applied: bool
    proposal_available: bool
    validation_performed: bool
    reason: str | None
    source_group: str
    update_horizon_s: float | None
    update_horizon_steps: int | None
    candidate_window_count: int
    used_window_count: int
    proposal_window_count: int
    validation_window_count: int
    normalized_innovation_rms_before: float | None
    normalized_innovation_rms_after: float | None
    normalized_validation_rms_before: float | None
    normalized_validation_rms_after: float | None
    realized_local_information_gain_nats: float | None
    structured_parameter_delta_norm: float | None
    prior_standardized_step_rms: float | None
    accepted_step_fraction: float | None
    prior_covariance_trace: float | None
    posterior_covariance_trace: float | None
    covariance_scope: ErrorCovarianceScope | None
    covariance_updated: bool
    maximum_validity_utilization: float
    prior_update_count: int
    posterior_update_count: int
    predictive_error_marked_stale: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "proposal_available": self.proposal_available,
            "validation_performed": self.validation_performed,
            "reason": self.reason,
            "source_group": self.source_group,
            "update_horizon_s": self.update_horizon_s,
            "update_horizon_steps": self.update_horizon_steps,
            "candidate_window_count": self.candidate_window_count,
            "used_window_count": self.used_window_count,
            "proposal_window_count": self.proposal_window_count,
            "validation_window_count": self.validation_window_count,
            "normalized_innovation_rms_before": (
                self.normalized_innovation_rms_before
            ),
            "normalized_innovation_rms_after": (
                self.normalized_innovation_rms_after
            ),
            "normalized_validation_rms_before": (
                self.normalized_validation_rms_before
            ),
            "normalized_validation_rms_after": (
                self.normalized_validation_rms_after
            ),
            "realized_local_information_gain_nats": (
                self.realized_local_information_gain_nats
            ),
            "structured_parameter_delta_norm": self.structured_parameter_delta_norm,
            "prior_standardized_step_rms": self.prior_standardized_step_rms,
            "accepted_step_fraction": self.accepted_step_fraction,
            "prior_covariance_trace": self.prior_covariance_trace,
            "posterior_covariance_trace": self.posterior_covariance_trace,
            "covariance_scope": (
                None if self.covariance_scope is None else self.covariance_scope.value
            ),
            "covariance_updated": self.covariance_updated,
            "maximum_validity_utilization": self.maximum_validity_utilization,
            "prior_update_count": self.prior_update_count,
            "posterior_update_count": self.posterior_update_count,
            "predictive_error_marked_stale": self.predictive_error_marked_stale,
        }


@dataclass(frozen=True)
class _UpdateWindow:
    initial_state: jax.Array
    control_history: jax.Array
    controls: jax.Array
    target_state: jax.Array
    exogenous: jax.Array
    bias: jax.Array
    error_support: SupportedCovariance


@dataclass(frozen=True)
class _UpdateContext:
    dt_s: float
    horizon_s: float
    horizon_steps: int
    candidate_window_count: int
    windows: tuple[_UpdateWindow, ...]
    maximum_validity_utilization: float


def _source_group(trajectory: Trajectory) -> str:
    for key in ("source_group", "flight_id", "trajectory_id", "vehicle_id"):
        value = trajectory.labels.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unlabeled_telemetry_segment"


def _transition_hashes(trajectory: Trajectory) -> tuple[str, ...]:
    """Fingerprint transitions so proposal evidence cannot be reused to validate."""

    hashes: list[str] = []
    for index, control in enumerate(trajectory.controls):
        digest = hashlib.sha256()
        for values in (
            trajectory.time_s[index : index + 2],
            trajectory.states[index],
            control,
            trajectory.states[index + 1],
            trajectory.exogenous[index],
            trajectory.exogenous[index + 1],
        ):
            array = np.ascontiguousarray(values, dtype=np.float64)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        hashes.append(digest.hexdigest())
    return tuple(hashes)


def _compatible_telemetry(belief: DynamicsBelief, telemetry: Trajectory) -> None:
    expected = belief.input_spec
    actual = telemetry.spec.prediction_spec()
    if actual.state_schema != expected.state_schema:
        raise ValueError("telemetry state schema does not match belief")
    if actual.vehicle.family != expected.vehicle.family:
        raise ValueError("telemetry vehicle family does not match belief")
    for attribute in ("control_roles", "control_semantics", "exogenous_roles"):
        if getattr(actual, attribute) != getattr(expected, attribute):
            raise ValueError(f"telemetry {attribute} do not match belief")
    for attribute in ("unit", "frame"):
        if tuple(getattr(channel, attribute) for channel in actual.controls) != tuple(
            getattr(channel, attribute) for channel in expected.controls
        ):
            raise ValueError(f"telemetry control {attribute}s do not match belief")
    for attribute in ("unit", "semantic", "frame"):
        if tuple(getattr(channel, attribute) for channel in actual.exogenous) != tuple(
            getattr(channel, attribute) for channel in expected.exogenous
        ):
            raise ValueError(f"telemetry exogenous {attribute}s do not match belief")


def _window_starts(candidate_count: int) -> np.ndarray:
    if candidate_count <= MAXIMUM_ONLINE_UPDATE_WINDOWS:
        return np.arange(candidate_count, dtype=np.int64)
    count = MAXIMUM_ONLINE_UPDATE_WINDOWS
    return ((2 * np.arange(count, dtype=np.int64) + 1) * candidate_count) // (
        2 * count
    )


def _maximum_validity(belief: DynamicsBelief, telemetry: Trajectory) -> float:
    utilization = np.asarray(
        jax.vmap(
            lambda state, context: model_validity_utilization(
                state,
                context,
                belief.input_spec,
                belief.runtime_spec.validity_envelope,
            )
        )(
            jnp.asarray(telemetry.states),
            jnp.asarray(telemetry.exogenous),
        )
    )
    return float(np.max(utilization))


def _base_report(
    belief: DynamicsBelief,
    telemetry: Trajectory,
    *,
    reason: str | None,
    maximum_validity: float,
    proposal_available: bool = False,
    validation_performed: bool = False,
    horizon_s: float | None = None,
    horizon_steps: int | None = None,
    candidate_count: int = 0,
    used_count: int = 0,
    proposal_count: int = 0,
    validation_count: int = 0,
) -> BeliefUpdateReport:
    parameter_belief = belief.parameter_belief
    covariance_trace = (
        float(np.trace(parameter_belief.covariance))
        if isinstance(parameter_belief, LocalGaussianParameterBelief)
        else None
    )
    covariance_scope = (
        belief.predictive_error.covariance_scope
        if isinstance(belief.predictive_error, EmpiricalHorizonPredictiveError)
        else None
    )
    return BeliefUpdateReport(
        applied=False,
        proposal_available=proposal_available,
        validation_performed=validation_performed,
        reason=reason,
        source_group=_source_group(telemetry),
        update_horizon_s=horizon_s,
        update_horizon_steps=horizon_steps,
        candidate_window_count=candidate_count,
        used_window_count=used_count,
        proposal_window_count=proposal_count,
        validation_window_count=validation_count,
        normalized_innovation_rms_before=None,
        normalized_innovation_rms_after=None,
        normalized_validation_rms_before=None,
        normalized_validation_rms_after=None,
        realized_local_information_gain_nats=None,
        structured_parameter_delta_norm=None,
        prior_standardized_step_rms=None,
        accepted_step_fraction=None,
        prior_covariance_trace=covariance_trace,
        posterior_covariance_trace=covariance_trace,
        covariance_scope=covariance_scope,
        covariance_updated=False,
        maximum_validity_utilization=maximum_validity,
        prior_update_count=parameter_belief.update_count,
        posterior_update_count=parameter_belief.update_count,
        predictive_error_marked_stale=False,
    )


def _preflight(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> tuple[_UpdateContext | None, BeliefUpdateReport | None]:
    if not isinstance(telemetry, Trajectory):
        raise TypeError("belief updates require one canonical Trajectory")
    _compatible_telemetry(belief, telemetry)
    maximum_validity = _maximum_validity(belief, telemetry)
    if not np.isfinite(maximum_validity):
        return None, _base_report(
            belief,
            telemetry,
            reason="telemetry validity utilization is non-finite",
            maximum_validity=maximum_validity,
        )
    if maximum_validity > 1.0 + VALIDITY_BOUNDARY_TOLERANCE:
        return None, _base_report(
            belief,
            telemetry,
            reason="telemetry lies outside the learned validity envelope",
            maximum_validity=maximum_validity,
        )
    if not isinstance(belief.parameter_belief, LocalGaussianParameterBelief):
        return None, _base_report(
            belief,
            telemetry,
            reason=(
                "live adaptation requires an evidence-derived parameter covariance"
            ),
            maximum_validity=maximum_validity,
        )
    if not isinstance(
        belief.predictive_error,
        EmpiricalHorizonPredictiveError,
    ):
        return None, _base_report(
            belief,
            telemetry,
            reason="live adaptation requires empirical predictive-error evidence",
            maximum_validity=maximum_validity,
        )
    if not belief.predictive_error_current:
        return None, _base_report(
            belief,
            telemetry,
            reason="predictive-error evidence is stale after a parameter update",
            maximum_validity=maximum_validity,
        )
    intervals = np.diff(telemetry.time_s)
    dt_s = float(np.median(intervals))
    if not np.allclose(intervals, dt_s, atol=1e-7, rtol=0.0):
        return None, _base_report(
            belief,
            telemetry,
            reason="live adaptation requires fixed-rate canonical telemetry",
            maximum_validity=maximum_validity,
        )
    if not np.isclose(
        dt_s,
        belief.runtime_spec.sample_period_s,
        atol=1e-7,
        rtol=0.0,
    ):
        return None, _base_report(
            belief,
            telemetry,
            reason="telemetry sample period does not match the runtime model",
            maximum_validity=maximum_validity,
        )
    shortest_horizon = float(belief.predictive_error.horizons_s[0])
    horizon_steps = duration_to_steps(shortest_horizon, dt_s)
    horizon_s = horizon_steps * dt_s
    if horizon_s > shortest_horizon * (1.0 + 1e-9):
        return None, _base_report(
            belief,
            telemetry,
            reason="telemetry period is longer than the shortest supported horizon",
            maximum_validity=maximum_validity,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
        )
    candidate_count = len(telemetry.controls) // horizon_steps
    if candidate_count < 1:
        return None, _base_report(
            belief,
            telemetry,
            reason="telemetry is shorter than the maintained update horizon",
            maximum_validity=maximum_validity,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
        )

    history_steps = max(1, int(np.ceil(ACTUATOR_HISTORY_DURATION_S / dt_s)))
    windows: list[_UpdateWindow] = []
    for ordinal in _window_starts(candidate_count):
        start = int(ordinal) * horizon_steps
        stop = start + horizon_steps
        history_start = max(0, start - history_steps)
        history = telemetry.controls[history_start:start]
        padding = np.repeat(
            telemetry.controls[0:1],
            history_steps - len(history),
            axis=0,
        )
        control_history = np.concatenate((padding, history), axis=0)
        controls = jnp.asarray(telemetry.controls[start:stop])
        context = jnp.asarray(telemetry.exogenous[start:stop])
        bias, covariance = belief.predictive_error.moments(
            horizon_s,
            state=jnp.asarray(telemetry.states[start]),
            command=controls[-1],
            exogenous=context[-1],
        )
        support = supported_covariance(np.asarray(covariance, dtype=np.float64))
        if support.rank == 0:
            continue
        windows.append(
            _UpdateWindow(
                initial_state=jnp.asarray(telemetry.states[start]),
                control_history=jnp.asarray(control_history),
                controls=controls,
                target_state=jnp.asarray(telemetry.states[stop]),
                exogenous=context,
                bias=bias,
                error_support=support,
            )
        )
    if not windows:
        return None, _base_report(
            belief,
            telemetry,
            reason="predictive-error covariance has no supported direction",
            maximum_validity=maximum_validity,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
            candidate_count=candidate_count,
        )
    return (
        _UpdateContext(
            dt_s=dt_s,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
            candidate_window_count=candidate_count,
            windows=tuple(windows),
            maximum_validity_utilization=maximum_validity,
        ),
        None,
    )


def _window_error(
    belief: DynamicsBelief,
    context: _UpdateContext,
    window: _UpdateWindow,
    vector: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        compiled_endpoint_tangent_error(
            jnp.asarray(vector),
            belief.params,
            window.initial_state,
            window.control_history,
            window.controls,
            window.target_state,
            window.exogenous,
            window.bias,
            dt_s=context.dt_s,
            control_roles=belief.input_spec.control_roles,
            exogenous_roles=belief.input_spec.exogenous_roles,
        ),
        dtype=np.float64,
    )


def _normalized_rms(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
) -> float:
    squared_error = 0.0
    supported_dimension = 0
    for window in context.windows:
        whitened = window.error_support.whiten_vector(
            _window_error(belief, context, window, vector)
        )
        squared_error += float(whitened @ whitened)
        supported_dimension += len(whitened)
    return float(np.sqrt(squared_error / supported_dimension))


def _candidate_rollouts_supported(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
) -> bool:
    """Require every proposed rollout path to remain finite and in support."""

    params = with_structured_parameter_vector(belief.params, jnp.asarray(vector))
    for window in context.windows:
        latent = control_state_after_history(
            params,
            window.control_history,
            context.dt_s,
            belief.input_spec.control_roles,
        )

        def transition(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            state, latent_state = carry
            control, exogenous = inputs
            next_state, next_latent = step_with_latent(
                params,
                state,
                latent_state,
                control,
                context.dt_s,
                belief.input_spec.control_roles,
                exogenous,
                belief.input_spec.exogenous_roles,
            )
            return (next_state, next_latent), (next_state, next_latent)

        _, (states, latent_states) = jax.lax.scan(
            transition,
            (window.initial_state, latent),
            (window.controls, window.exogenous),
        )
        states_np = np.asarray(states)
        latent_np = np.asarray(latent_states)
        if not (
            np.all(np.isfinite(states_np)) and np.all(np.isfinite(latent_np))
        ):
            return False
        utilization = np.asarray(
            jax.vmap(
                lambda state, exogenous: model_validity_utilization(
                    state,
                    exogenous,
                    belief.input_spec,
                    belief.runtime_spec.validity_envelope,
                )
            )(states, window.exogenous)
        )
        if not np.all(np.isfinite(utilization)) or float(np.max(utilization)) > (
            1.0 + VALIDITY_BOUNDARY_TOLERANCE
        ):
            return False
    return True


def _proposal_geometry(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
    prior_support: SupportedCovariance,
) -> tuple[np.ndarray, np.ndarray]:
    prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
    rows: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for window in context.windows:
        error, jacobian = compiled_endpoint_tangent_linearization(
            jnp.asarray(vector),
            belief.params,
            window.initial_state,
            window.control_history,
            window.controls,
            window.target_state,
            window.exogenous,
            window.bias,
            dt_s=context.dt_s,
            control_roles=belief.input_spec.control_roles,
            exogenous_roles=belief.input_spec.exogenous_roles,
        )
        error_np = np.asarray(error, dtype=np.float64)
        jacobian_np = np.asarray(jacobian, dtype=np.float64)
        residuals.append(window.error_support.whiten_vector(error_np))
        rows.append(window.error_support.whiten_rows(jacobian_np @ prior_factor))
    evidence_scale = np.sqrt(len(context.windows))
    return (
        np.concatenate(rows, axis=0) / evidence_scale,
        np.concatenate(residuals, axis=0) / evidence_scale,
    )


def propose_dynamics_belief_update(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> tuple[BeliefUpdateProposal | None, BeliefUpdateReport]:
    """Propose a bounded local move; do not mutate the authoritative belief."""

    context, rejected = _preflight(belief, telemetry)
    if rejected is not None:
        return None, rejected
    assert context is not None
    parameter_belief = belief.parameter_belief
    assert isinstance(parameter_belief, LocalGaussianParameterBelief)
    prior_vector = np.asarray(
        structured_parameter_vector(belief.params),
        dtype=np.float64,
    )
    prior_covariance = np.asarray(parameter_belief.covariance, dtype=np.float64)
    try:
        prior_support = supported_covariance(prior_covariance)
        if prior_support.rank == 0:
            return None, _base_report(
                belief,
                telemetry,
                reason="parameter covariance has no supported update direction",
                maximum_validity=context.maximum_validity_utilization,
                horizon_s=context.horizon_s,
                horizon_steps=context.horizon_steps,
                candidate_count=context.candidate_window_count,
            )
        design, residual = _proposal_geometry(
            belief,
            context,
            prior_vector,
            prior_support,
        )
        information = design.T @ design
        local_step = -np.linalg.solve(
            np.eye(prior_support.rank) + information,
            design.T @ residual,
        )
        local_rms = float(np.sqrt(np.mean(np.square(local_step))))
        if not np.isfinite(local_rms) or local_rms == 0.0:
            return None, _base_report(
                belief,
                telemetry,
                reason="local update produced no finite parameter movement",
                maximum_validity=context.maximum_validity_utilization,
                horizon_s=context.horizon_s,
                horizon_steps=context.horizon_steps,
                candidate_count=context.candidate_window_count,
                used_count=len(context.windows),
                proposal_count=len(context.windows),
            )
        trust_fraction = min(
            1.0,
            MAXIMUM_LOCAL_PARAMETER_STEP_RMS / local_rms,
        )
        prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
        raw_delta = prior_factor @ local_step
        score_before = _normalized_rms(belief, context, prior_vector)
        candidate: np.ndarray | None = None
        score_after: float | None = None
        selected_fraction: float | None = None
        selected_local_rms: float | None = None
        for fraction in LINE_SEARCH_FRACTIONS:
            combined_fraction = trust_fraction * fraction
            selected = prior_vector + combined_fraction * raw_delta
            score = _normalized_rms(belief, context, selected)
            if (
                np.isfinite(score)
                and score < score_before
                and _candidate_rollouts_supported(belief, context, selected)
            ):
                candidate = selected
                score_after = score
                selected_fraction = combined_fraction
                selected_local_rms = combined_fraction * local_rms
                break
        if candidate is None:
            return None, _base_report(
                belief,
                telemetry,
                reason="no bounded local step improved proposal telemetry",
                maximum_validity=context.maximum_validity_utilization,
                horizon_s=context.horizon_s,
                horizon_steps=context.horizon_steps,
                candidate_count=context.candidate_window_count,
                used_count=len(context.windows),
                proposal_count=len(context.windows),
            )
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None, _base_report(
            belief,
            telemetry,
            reason="local proposal geometry was non-finite or singular",
            maximum_validity=context.maximum_validity_utilization,
            horizon_s=context.horizon_s,
            horizon_steps=context.horizon_steps,
            candidate_count=context.candidate_window_count,
            used_count=len(context.windows),
            proposal_count=len(context.windows),
        )
    assert score_after is not None
    assert selected_fraction is not None
    assert selected_local_rms is not None
    predictive_error = belief.predictive_error
    assert isinstance(predictive_error, EmpiricalHorizonPredictiveError)
    proposal = BeliefUpdateProposal(
        base_parameter_vector=prior_vector,
        base_parameter_covariance=prior_covariance,
        candidate_parameter_vector=candidate,
        normalized_information_matrix=information,
        covariance_scope=predictive_error.covariance_scope,
        base_update_count=parameter_belief.update_count,
        update_horizon_s=context.horizon_s,
        update_horizon_steps=context.horizon_steps,
        proposal_window_count=len(context.windows),
        normalized_innovation_rms_before=score_before,
        normalized_innovation_rms_after=score_after,
        prior_standardized_step_rms=selected_local_rms,
        proposal_step_fraction=selected_fraction,
        maximum_validity_utilization=context.maximum_validity_utilization,
        source_group=_source_group(telemetry),
        evidence_transition_hashes=_transition_hashes(telemetry),
    )
    report = _base_report(
        belief,
        telemetry,
        reason="proposal requires disjoint validation telemetry before commit",
        maximum_validity=context.maximum_validity_utilization,
        proposal_available=True,
        horizon_s=context.horizon_s,
        horizon_steps=context.horizon_steps,
        candidate_count=context.candidate_window_count,
        used_count=len(context.windows),
        proposal_count=len(context.windows),
    )
    report = replace(
        report,
        normalized_innovation_rms_before=score_before,
        normalized_innovation_rms_after=score_after,
        structured_parameter_delta_norm=float(
            np.linalg.norm(candidate - prior_vector)
        ),
        prior_standardized_step_rms=selected_local_rms,
        accepted_step_fraction=selected_fraction,
    )
    return proposal, report


def _proposal_matches_belief(
    belief: DynamicsBelief,
    proposal: BeliefUpdateProposal,
) -> bool:
    if not isinstance(belief.parameter_belief, LocalGaussianParameterBelief):
        return False
    return (
        belief.parameter_belief.update_count == proposal.base_update_count
        and np.array_equal(
            np.asarray(structured_parameter_vector(belief.params), dtype=np.float64),
            proposal.base_parameter_vector,
        )
        and np.array_equal(
            np.asarray(belief.parameter_belief.covariance, dtype=np.float64),
            proposal.base_parameter_covariance,
        )
    )


def validate_and_commit_dynamics_belief_update(
    belief: DynamicsBelief,
    proposal: BeliefUpdateProposal,
    validation_telemetry: Trajectory,
) -> tuple[DynamicsBelief, BeliefUpdateReport]:
    """Commit only the portion of a proposal that improves disjoint evidence."""

    if not isinstance(proposal, BeliefUpdateProposal):
        raise TypeError("validation requires a BeliefUpdateProposal")
    _compatible_telemetry(belief, validation_telemetry)
    maximum_validity = max(
        proposal.maximum_validity_utilization,
        _maximum_validity(belief, validation_telemetry),
    )
    if not _proposal_matches_belief(belief, proposal):
        return belief, _base_report(
            belief,
            validation_telemetry,
            reason="proposal no longer matches the authoritative belief",
            maximum_validity=maximum_validity,
            proposal_available=True,
            validation_performed=False,
            horizon_s=proposal.update_horizon_s,
            horizon_steps=proposal.update_horizon_steps,
            proposal_count=proposal.proposal_window_count,
        )
    if set(proposal.evidence_transition_hashes) & set(
        _transition_hashes(validation_telemetry)
    ):
        return belief, _base_report(
            belief,
            validation_telemetry,
            reason="validation telemetry overlaps proposal transitions",
            maximum_validity=maximum_validity,
            proposal_available=True,
            validation_performed=False,
            horizon_s=proposal.update_horizon_s,
            horizon_steps=proposal.update_horizon_steps,
            proposal_count=proposal.proposal_window_count,
        )
    context, rejected = _preflight(belief, validation_telemetry)
    if rejected is not None:
        report = replace(
            rejected,
            proposal_available=True,
            proposal_window_count=proposal.proposal_window_count,
            candidate_window_count=(
                rejected.candidate_window_count + proposal.proposal_window_count
            ),
            maximum_validity_utilization=maximum_validity,
        )
        return belief, report
    assert context is not None
    if (
        context.horizon_steps != proposal.update_horizon_steps
        or not np.isclose(context.horizon_s, proposal.update_horizon_s)
    ):
        return belief, _base_report(
            belief,
            validation_telemetry,
            reason="validation telemetry resolves to a different evidence horizon",
            maximum_validity=maximum_validity,
            proposal_available=True,
            validation_performed=False,
            horizon_s=proposal.update_horizon_s,
            horizon_steps=proposal.update_horizon_steps,
            candidate_count=(
                proposal.proposal_window_count + context.candidate_window_count
            ),
            proposal_count=proposal.proposal_window_count,
            validation_count=len(context.windows),
        )
    base = proposal.base_parameter_vector
    proposed_delta = proposal.candidate_parameter_vector - base
    try:
        validation_before: float | None = _normalized_rms(belief, context, base)
        selected: np.ndarray | None = None
        validation_after: float | None = None
        validation_fraction: float | None = None
        for fraction in LINE_SEARCH_FRACTIONS:
            candidate = base + fraction * proposed_delta
            score = _normalized_rms(belief, context, candidate)
            if np.isfinite(score) and score < validation_before:
                selected = candidate
                validation_after = score
                validation_fraction = fraction
                break
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        selected = None
        validation_after = None
        validation_fraction = None
        validation_before = None
    if selected is None:
        report = _base_report(
            belief,
            validation_telemetry,
            reason="proposal did not improve disjoint validation telemetry",
            maximum_validity=maximum_validity,
            proposal_available=True,
            validation_performed=True,
            horizon_s=proposal.update_horizon_s,
            horizon_steps=proposal.update_horizon_steps,
            candidate_count=(
                proposal.proposal_window_count + context.candidate_window_count
            ),
            used_count=proposal.proposal_window_count + len(context.windows),
            proposal_count=proposal.proposal_window_count,
            validation_count=len(context.windows),
        )
        report = replace(
            report,
            normalized_innovation_rms_before=(
                proposal.normalized_innovation_rms_before
            ),
            normalized_innovation_rms_after=(
                proposal.normalized_innovation_rms_after
            ),
            normalized_validation_rms_before=validation_before,
        )
        return belief, report

    assert validation_after is not None
    assert validation_fraction is not None
    parameter_belief = belief.parameter_belief
    assert isinstance(parameter_belief, LocalGaussianParameterBelief)
    prior_covariance = proposal.base_parameter_covariance
    accepted_fraction = proposal.proposal_step_fraction * validation_fraction
    covariance = prior_covariance
    information_gain: float | None = None
    covariance_updated = False
    if proposal.covariance_scope == ErrorCovarianceScope.CONDITIONAL_INNOVATION:
        prior_support = supported_covariance(prior_covariance)
        prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
        normalized_posterior_precision = (
            np.eye(prior_support.rank)
            + accepted_fraction * proposal.normalized_information_matrix
        )
        covariance = prior_factor @ np.linalg.solve(
            normalized_posterior_precision,
            prior_factor.T,
        )
        covariance = 0.5 * (covariance + covariance.T)
        sign, logdet = np.linalg.slogdet(normalized_posterior_precision)
        if sign <= 0.0 or not np.isfinite(logdet):
            return belief, _base_report(
                belief,
                validation_telemetry,
                reason="validated covariance update was non-finite",
                maximum_validity=maximum_validity,
                proposal_available=True,
                validation_performed=True,
                horizon_s=proposal.update_horizon_s,
                horizon_steps=proposal.update_horizon_steps,
                proposal_count=proposal.proposal_window_count,
                validation_count=len(context.windows),
            )
        information_gain = float(0.5 * logdet)
        covariance_updated = not np.array_equal(covariance, prior_covariance)

    update_count = parameter_belief.update_count + 1
    updated_parameter_belief = LocalGaussianParameterBelief(
        parameter_names=parameter_belief.parameter_names,
        covariance=covariance,
        source=parameter_belief.source,
        evidence_count=parameter_belief.evidence_count + 1,
        effective_sample_count=parameter_belief.effective_sample_count + 1.0,
        update_count=update_count,
    )
    report = BeliefUpdateReport(
        applied=True,
        proposal_available=True,
        validation_performed=True,
        reason=None,
        source_group=(
            f"proposal:{proposal.source_group};validation:"
            f"{_source_group(validation_telemetry)}"
        ),
        update_horizon_s=proposal.update_horizon_s,
        update_horizon_steps=proposal.update_horizon_steps,
        candidate_window_count=(
            proposal.proposal_window_count + context.candidate_window_count
        ),
        used_window_count=proposal.proposal_window_count + len(context.windows),
        proposal_window_count=proposal.proposal_window_count,
        validation_window_count=len(context.windows),
        normalized_innovation_rms_before=(
            proposal.normalized_innovation_rms_before
        ),
        normalized_innovation_rms_after=proposal.normalized_innovation_rms_after,
        normalized_validation_rms_before=validation_before,
        normalized_validation_rms_after=validation_after,
        realized_local_information_gain_nats=information_gain,
        structured_parameter_delta_norm=float(np.linalg.norm(selected - base)),
        prior_standardized_step_rms=(
            proposal.prior_standardized_step_rms * validation_fraction
        ),
        accepted_step_fraction=accepted_fraction,
        prior_covariance_trace=float(np.trace(prior_covariance)),
        posterior_covariance_trace=float(np.trace(covariance)),
        covariance_scope=proposal.covariance_scope,
        covariance_updated=covariance_updated,
        maximum_validity_utilization=maximum_validity,
        prior_update_count=parameter_belief.update_count,
        posterior_update_count=update_count,
        predictive_error_marked_stale=True,
    )
    provenance = dict(belief.provenance)
    provenance["online_adaptation"] = {
        "update_count": update_count,
        "last_update": report.to_dict(),
    }
    updated = DynamicsBelief(
        params=with_structured_parameter_vector(belief.params, jnp.asarray(selected)),
        input_spec=belief.input_spec,
        runtime_spec=belief.runtime_spec,
        predictive_error=belief.predictive_error,
        parameter_belief=updated_parameter_belief,
        parameter_evidence=belief.parameter_evidence,
        predictive_error_parameter_update_count=(
            belief.predictive_error_parameter_update_count
        ),
        provenance=provenance,
    )
    return updated, report


def _trajectory_slice(
    telemetry: Trajectory,
    start: int,
    stop: int,
    *,
    phase: str,
) -> Trajectory:
    labels = dict(telemetry.labels)
    labels["adaptation_phase"] = phase
    return Trajectory(
        time_s=telemetry.time_s[start : stop + 1],
        states=telemetry.states[start : stop + 1],
        controls=telemetry.controls[start:stop],
        spec=telemetry.spec,
        exogenous=telemetry.exogenous[start : stop + 1],
        observations=telemetry.observations[start : stop + 1],
        labels=labels,
        provenance=telemetry.provenance,
    )


def update_dynamics_belief(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> tuple[DynamicsBelief, BeliefUpdateReport]:
    """Propose on early telemetry and commit only after later validation."""

    if not isinstance(telemetry, Trajectory):
        raise TypeError("belief updates require one canonical Trajectory")
    _compatible_telemetry(belief, telemetry)
    if not isinstance(
        belief.predictive_error,
        EmpiricalHorizonPredictiveError,
    ):
        maximum_validity = _maximum_validity(belief, telemetry)
        return belief, _base_report(
            belief,
            telemetry,
            reason="live adaptation requires empirical predictive-error evidence",
            maximum_validity=maximum_validity,
        )
    intervals = np.diff(telemetry.time_s)
    dt_s = float(np.median(intervals))
    horizon_steps = duration_to_steps(
        float(belief.predictive_error.horizons_s[0]),
        dt_s,
    )
    complete_windows = len(telemetry.controls) // horizon_steps
    if complete_windows < 2:
        maximum_validity = _maximum_validity(belief, telemetry)
        return belief, _base_report(
            belief,
            telemetry,
            reason=(
                "transactional adaptation requires proposal and validation windows"
            ),
            maximum_validity=maximum_validity,
            horizon_s=horizon_steps * dt_s,
            horizon_steps=horizon_steps,
            candidate_count=complete_windows,
        )
    proposal_windows = complete_windows // 2
    split = proposal_windows * horizon_steps
    proposal_telemetry = _trajectory_slice(
        telemetry,
        0,
        split,
        phase="proposal",
    )
    validation_telemetry = _trajectory_slice(
        telemetry,
        split,
        complete_windows * horizon_steps,
        phase="validation",
    )
    proposal, report = propose_dynamics_belief_update(belief, proposal_telemetry)
    if proposal is None:
        return belief, report
    return validate_and_commit_dynamics_belief_update(
        belief,
        proposal,
        validation_telemetry,
    )
