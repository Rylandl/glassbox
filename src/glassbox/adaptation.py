"""Fail-closed, evidence-preserving updates for structured dynamics beliefs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    EmpiricalErrorSample,
    EmpiricalHorizonPredictiveError,
    ErrorCovarianceScope,
    LocalGaussianParameterBelief,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.covariance import SupportedCovariance, supported_covariance
from glassbox.data import Trajectory, duration_to_steps
from glassbox.dynamics import (
    ModelParams,
    control_state_after_history,
    step_with_latent,
)
from glassbox.evaluation import windowed_rollout_evaluation
from glassbox.linearization import (
    compiled_batched_endpoint_tangent_error,
    compiled_batched_endpoint_tangent_linearization,
)
from glassbox.runtime import model_validity_utilization_from_components

MAXIMUM_ONLINE_UPDATE_WINDOWS = 64
ACTUATOR_HISTORY_DURATION_S = 1.0
MAXIMUM_LOCAL_PARAMETER_STEP_RMS = 1.0
VALIDITY_BOUNDARY_TOLERANCE = 1e-6
LINE_SEARCH_FRACTIONS = (1.0, 0.5, 0.25, 0.125, 0.0625)

# How candidates are scored, recorded in every report. A commit marks the
# held-out bias correction stale, so the runtime stops applying it: candidates
# are therefore scored *without* that correction, while the incumbent keeps the
# correction the runtime is applying today. Accepting a candidate then means the
# uncorrected candidate forecast beats the forecast the vehicle currently flies.
VALIDATION_SCORING = "candidate_uncorrected_vs_nominal_bias_corrected"

# A commit must beat the incumbent by more than the noise of the evidence that
# scores it. The paired per-window reduction in whitened squared endpoint error
# must exceed this many standard errors of its own total. Two standard errors is
# a one-sided level of about 2 percent under a normal approximation, and the
# same hurdle is cleared twice on disjoint telemetry -- once to propose and once
# to commit -- so the transaction as a whole is far more conservative than the
# per-stage level suggests. It is a library invariant, not a tuning knob.
IMPROVEMENT_MARGIN_STANDARD_ERRORS = 2.0

# Source recorded on predictive-error evidence rebuilt around current parameters.
RECALIBRATION_SOURCE = "recalibrated_from_telemetry"


def _readonly_array(value: np.ndarray) -> np.ndarray:
    """Own one immutable copy rather than aliasing authoritative belief state."""

    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _valid_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _update_revision_fingerprint(belief: DynamicsBelief) -> str:
    """Fingerprint every belief component that changes update semantics."""

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(belief.params):
        array = np.ascontiguousarray(np.asarray(leaf))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    payload = {
        "input_spec": belief.input_spec.to_dict(),
        "runtime_spec": belief.runtime_spec.to_dict(),
        "predictive_error": belief.predictive_error.to_dict(),
        "parameter_belief": belief.parameter_belief.to_dict(),
        "parameter_evidence": belief.parameter_evidence.to_dict(),
        "predictive_error_parameter_update_count": (
            belief.predictive_error_parameter_update_count
        ),
    }
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _trajectory_spec_fingerprint(trajectory: Trajectory) -> str:
    payload = trajectory.spec.prediction_spec().to_dict()
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _control_history_fingerprint(control_history: np.ndarray) -> str:
    """Fingerprint context commands without treating them as evidence."""

    array = np.ascontiguousarray(control_history, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prior_standardized_step_rms(
    base: np.ndarray,
    candidate: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """Measure a candidate in supported prior coordinates, rejecting leakage."""

    support = supported_covariance(covariance)
    if support.rank == 0:
        raise ValueError("update proposal covariance has no supported direction")
    delta = candidate - base
    projected = support.projector @ delta
    residual = delta - projected
    tolerance = np.sqrt(np.finfo(np.float64).eps) * max(
        1.0,
        float(np.linalg.norm(delta)),
    )
    if float(np.linalg.norm(residual)) > tolerance:
        raise ValueError("update proposal moves in an unsupported parameter direction")
    local_step = (support.basis.T @ delta) / np.sqrt(support.variances)
    return float(np.sqrt(np.mean(np.square(local_step))))


@dataclass(frozen=True)
class _ImprovementEvidence:
    """One paired, noise-aware comparison of incumbent and candidate forecasts."""

    before_rms: float
    after_rms: float
    total_reduction: float
    margin: float

    @property
    def improves(self) -> bool:
        return (
            np.isfinite(self.total_reduction)
            and np.isfinite(self.margin)
            and np.isfinite(self.after_rms)
            and self.total_reduction > self.margin
        )


def _improvement_evidence(
    before_squared: np.ndarray,
    after_squared: np.ndarray,
    dimensions: np.ndarray,
) -> _ImprovementEvidence:
    """Score a candidate against the incumbent with an effect-size margin.

    Windows are the independent evidence units. The paired per-window reduction
    in whitened squared endpoint error must exceed a one-sided margin scaled by
    the standard error of its own total. The per-window variance is floored by
    the chi-square scale of the incumbent's own error, ``2 * s**2 / k`` for
    ``k`` supported error dimensions and mean incumbent whitened squared error
    ``s``, so a short evidence block whose window-to-window spread happens to be
    small cannot manufacture significance and a single window still carries a
    usable scale. Held-out error covariance is empirical rather than calibrated,
    so the floor is anchored to the error level actually observed instead of
    assuming ``s == k``; the two agree exactly when the whitening is calibrated.
    """

    differences = np.asarray(before_squared, dtype=np.float64) - np.asarray(
        after_squared, dtype=np.float64
    )
    count = len(differences)
    if count < 1:
        raise ValueError("improvement evidence requires at least one window")
    supported_dimension = float(np.sum(dimensions))
    finite = bool(np.all(np.isfinite(differences)))
    sample_variance = (
        float(np.var(differences, ddof=1)) if count > 1 and finite else 0.0
    )
    mean_dimension = float(np.mean(dimensions))
    mean_incumbent_squared = float(np.mean(before_squared))
    chi_square_variance = 2.0 * mean_incumbent_squared**2 / mean_dimension
    per_window_variance = max(sample_variance, chi_square_variance)
    margin = IMPROVEMENT_MARGIN_STANDARD_ERRORS * float(
        np.sqrt(count * per_window_variance)
    )
    return _ImprovementEvidence(
        before_rms=float(np.sqrt(np.sum(before_squared) / supported_dimension)),
        after_rms=float(np.sqrt(np.sum(after_squared) / supported_dimension)),
        total_reduction=float(np.sum(differences)),
        margin=margin,
    )


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
    normalized_innovation_improvement: float
    normalized_innovation_improvement_margin: float
    prior_standardized_step_rms: float
    proposal_step_fraction: float
    maximum_validity_utilization: float
    source_group: str
    evidence_transition_hashes: tuple[str, ...]
    base_belief_fingerprint: str
    target_spec_fingerprint: str

    def __post_init__(self) -> None:
        base = np.array(self.base_parameter_vector, dtype=np.float64, copy=True)
        candidate = np.array(
            self.candidate_parameter_vector,
            dtype=np.float64,
            copy=True,
        )
        covariance = np.array(
            self.base_parameter_covariance,
            dtype=np.float64,
            copy=True,
        )
        information = np.array(
            self.normalized_information_matrix,
            dtype=np.float64,
            copy=True,
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
        prior_support = supported_covariance(covariance)
        if information.shape != (prior_support.rank, prior_support.rank):
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
            np.isfinite(self.normalized_innovation_improvement)
            and np.isfinite(self.normalized_innovation_improvement_margin)
            and self.normalized_innovation_improvement_margin >= 0.0
            and self.normalized_innovation_improvement
            > self.normalized_innovation_improvement_margin
        ):
            raise ValueError(
                "update proposal must clear its proposal effect-size margin"
            )
        if not (
            np.isfinite(self.prior_standardized_step_rms)
            and 0.0
            < self.prior_standardized_step_rms
            <= MAXIMUM_LOCAL_PARAMETER_STEP_RMS * (1.0 + 1e-9)
        ):
            raise ValueError("update proposal exceeds the local trust region")
        measured_step_rms = _prior_standardized_step_rms(
            base,
            candidate,
            covariance,
        )
        if not np.isclose(
            measured_step_rms,
            self.prior_standardized_step_rms,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError("update proposal trust-region evidence is inconsistent")
        if not (
            np.isfinite(self.proposal_step_fraction)
            and 0.0 < self.proposal_step_fraction <= 1.0
        ):
            raise ValueError("update proposal step fraction is invalid")
        if not (
            np.isfinite(self.maximum_validity_utilization)
            and self.maximum_validity_utilization <= 1.0 + VALIDITY_BOUNDARY_TOLERANCE
        ):
            raise ValueError("update proposal lies outside model support")
        if not self.source_group.strip():
            raise ValueError("update proposal source group is required")
        transition_hashes = tuple(
            str(value) for value in self.evidence_transition_hashes
        )
        if not transition_hashes or any(
            not _valid_sha256(value) for value in transition_hashes
        ):
            raise ValueError("update proposal transition evidence is invalid")
        if not _valid_sha256(self.base_belief_fingerprint):
            raise ValueError("update proposal belief revision is invalid")
        if not _valid_sha256(self.target_spec_fingerprint):
            raise ValueError("update proposal target specification is invalid")
        object.__setattr__(
            self,
            "covariance_scope",
            ErrorCovarianceScope(self.covariance_scope),
        )
        object.__setattr__(self, "base_parameter_vector", _readonly_array(base))
        object.__setattr__(
            self,
            "base_parameter_covariance",
            _readonly_array(covariance),
        )
        object.__setattr__(
            self,
            "candidate_parameter_vector",
            _readonly_array(candidate),
        )
        object.__setattr__(
            self,
            "normalized_information_matrix",
            _readonly_array(information),
        )
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
    actuator_context_sample_count: int
    actuator_context_fingerprint: str | None
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
    validation_scoring: str = VALIDATION_SCORING
    normalized_innovation_improvement: float | None = None
    normalized_innovation_improvement_margin: float | None = None
    normalized_validation_improvement: float | None = None
    normalized_validation_improvement_margin: float | None = None

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
            "actuator_context_sample_count": self.actuator_context_sample_count,
            "actuator_context_fingerprint": self.actuator_context_fingerprint,
            "normalized_innovation_rms_before": (self.normalized_innovation_rms_before),
            "normalized_innovation_rms_after": (self.normalized_innovation_rms_after),
            "normalized_validation_rms_before": (self.normalized_validation_rms_before),
            "normalized_validation_rms_after": (self.normalized_validation_rms_after),
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
            "validation_scoring": self.validation_scoring,
            "normalized_innovation_improvement": (
                self.normalized_innovation_improvement
            ),
            "normalized_innovation_improvement_margin": (
                self.normalized_innovation_improvement_margin
            ),
            "normalized_validation_improvement": (
                self.normalized_validation_improvement
            ),
            "normalized_validation_improvement_margin": (
                self.normalized_validation_improvement_margin
            ),
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
    actuator_context_sample_count: int
    actuator_context_fingerprint: str | None


def _stacked_window_values(
    context: _UpdateContext,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return tuple(
        jnp.stack(tuple(getattr(window, field) for window in context.windows))
        for field in (
            "initial_state",
            "control_history",
            "controls",
            "target_state",
            "exogenous",
            "bias",
        )
    )  # type: ignore[return-value]


def _batched_candidate_rollouts(
    vector: jax.Array,
    template_params: ModelParams,
    initial_states: jax.Array,
    control_histories: jax.Array,
    controls: jax.Array,
    exogenous: jax.Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> tuple[jax.Array, jax.Array]:
    params = with_structured_parameter_vector(template_params, vector)

    def rollout_one(
        initial_state: jax.Array,
        control_history: jax.Array,
        window_controls: jax.Array,
        window_exogenous: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        latent = control_state_after_history(
            params,
            control_history,
            dt_s,
            control_roles,
        )

        def transition(
            carry: tuple[jax.Array, jax.Array],
            inputs: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            state, latent_state = carry
            control, context = inputs
            next_state, next_latent = step_with_latent(
                params,
                state,
                latent_state,
                control,
                dt_s,
                control_roles,
                context,
                exogenous_roles,
            )
            return (next_state, next_latent), (next_state, next_latent)

        _, (states, latent_states) = jax.lax.scan(
            transition,
            (initial_state, latent),
            (window_controls, window_exogenous),
        )
        return states, latent_states

    return jax.vmap(rollout_one)(
        initial_states,
        control_histories,
        controls,
        exogenous,
    )


def _batched_model_validity_utilization(
    states: jax.Array,
    exogenous: jax.Array,
    body_velocity_center: jax.Array,
    body_velocity_half_width: jax.Array,
    angular_velocity_center: jax.Array,
    angular_velocity_half_width: jax.Array,
    *,
    exogenous_roles: tuple[str, ...],
) -> jax.Array:
    """Evaluate validity over arbitrary leading batch dimensions."""

    flat_states = states.reshape((-1, states.shape[-1]))
    flat_exogenous = exogenous.reshape((flat_states.shape[0], exogenous.shape[-1]))

    def utilization_one(state: jax.Array, context: jax.Array) -> jax.Array:
        return model_validity_utilization_from_components(
            state,
            context,
            body_velocity_center,
            body_velocity_half_width,
            angular_velocity_center,
            angular_velocity_half_width,
            exogenous_roles=exogenous_roles,
        )

    flat_utilization = jax.vmap(utilization_one)(flat_states, flat_exogenous)
    return flat_utilization.reshape((*states.shape[:-1], 6))


_compiled_batched_model_validity_utilization = jax.jit(
    _batched_model_validity_utilization,
    static_argnames=("exogenous_roles",),
)


def _batched_candidate_rollouts_supported(
    vector: jax.Array,
    template_params: ModelParams,
    initial_states: jax.Array,
    control_histories: jax.Array,
    controls: jax.Array,
    exogenous: jax.Array,
    body_velocity_center: jax.Array,
    body_velocity_half_width: jax.Array,
    angular_velocity_center: jax.Array,
    angular_velocity_half_width: jax.Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> jax.Array:
    states, latent_states = _batched_candidate_rollouts(
        vector,
        template_params,
        initial_states,
        control_histories,
        controls,
        exogenous,
        dt_s=dt_s,
        control_roles=control_roles,
        exogenous_roles=exogenous_roles,
    )
    utilization = _batched_model_validity_utilization(
        states,
        exogenous,
        body_velocity_center,
        body_velocity_half_width,
        angular_velocity_center,
        angular_velocity_half_width,
        exogenous_roles=exogenous_roles,
    )
    return (
        jnp.all(jnp.isfinite(states))
        & jnp.all(jnp.isfinite(latent_states))
        & jnp.all(jnp.isfinite(utilization))
        & (jnp.max(utilization) <= 1.0 + VALIDITY_BOUNDARY_TOLERANCE)
    )


_compiled_batched_candidate_rollouts_supported = jax.jit(
    _batched_candidate_rollouts_supported,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)


def _source_group(trajectory: Trajectory) -> str:
    for key in ("source_group", "flight_id", "trajectory_id", "vehicle_id"):
        value = trajectory.labels.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unlabeled_telemetry_segment"


def _transition_hashes(trajectory: Trajectory) -> tuple[str, ...]:
    """Fingerprint transition content independent of an arbitrary time origin."""

    hashes: list[str] = []
    for index, control in enumerate(trajectory.controls):
        digest = hashlib.sha256()
        for values in (
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
    return ((2 * np.arange(count, dtype=np.int64) + 1) * candidate_count) // (2 * count)


def _maximum_validity(belief: DynamicsBelief, telemetry: Trajectory) -> float:
    envelope = belief.runtime_spec.validity_envelope
    utilization = np.asarray(
        _compiled_batched_model_validity_utilization(
            jnp.asarray(telemetry.states),
            jnp.asarray(telemetry.exogenous),
            jnp.asarray(envelope.body_velocity_center_m_s),
            jnp.asarray(envelope.body_velocity_half_width_m_s),
            jnp.asarray(envelope.angular_velocity_center_rad_s),
            jnp.asarray(envelope.angular_velocity_half_width_rad_s),
            exogenous_roles=belief.input_spec.exogenous_roles,
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
    actuator_context_sample_count: int = 0,
    actuator_context_fingerprint: str | None = None,
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
        actuator_context_sample_count=actuator_context_sample_count,
        actuator_context_fingerprint=actuator_context_fingerprint,
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
    *,
    preceding_control_history: np.ndarray | None = None,
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
    observed_dt_s = float(np.median(intervals))
    if not np.allclose(intervals, observed_dt_s, atol=1e-7, rtol=0.0):
        return None, _base_report(
            belief,
            telemetry,
            reason="live adaptation requires fixed-rate canonical telemetry",
            maximum_validity=maximum_validity,
        )
    if not np.isclose(
        observed_dt_s,
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
    # The runtime contract is authoritative after the telemetry period passes
    # validation. Using a median reconstructed from timestamps as a JAX static
    # argument lets insignificant floating-point differences defeat the
    # precompiled update cache.
    dt_s = float(belief.runtime_spec.sample_period_s)
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
    context_history: np.ndarray | None = None
    context_sample_count = 0
    context_fingerprint: str | None = None
    if preceding_control_history is not None:
        try:
            provided_history = np.asarray(
                preceding_control_history,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            provided_history = np.empty((0, 0), dtype=np.float64)
        if (
            provided_history.ndim != 2
            or provided_history.shape[1:] != telemetry.controls.shape[1:]
            or len(provided_history) < 1
            or not np.all(np.isfinite(provided_history))
        ):
            return None, _base_report(
                belief,
                telemetry,
                reason="actuator context must contain finite preceding commands",
                maximum_validity=maximum_validity,
                horizon_s=horizon_s,
                horizon_steps=horizon_steps,
                candidate_count=candidate_count,
            )
        available = provided_history[-history_steps:]
        padding = np.repeat(
            available[0:1],
            history_steps - len(available),
            axis=0,
        )
        context_history = np.concatenate((padding, available), axis=0)
        context_sample_count = len(available)
        context_fingerprint = _control_history_fingerprint(context_history)
    windows: list[_UpdateWindow] = []
    for ordinal in _window_starts(candidate_count):
        start = int(ordinal) * horizon_steps
        stop = start + horizon_steps
        local_history = telemetry.controls[:start]
        history = (
            local_history
            if context_history is None
            else np.concatenate((context_history, local_history), axis=0)
        )
        history = history[-history_steps:]
        initial_command = telemetry.controls[0:1] if not len(history) else history[0:1]
        padding = np.repeat(
            initial_command,
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
            actuator_context_sample_count=context_sample_count,
            actuator_context_fingerprint=context_fingerprint,
        )
    return (
        _UpdateContext(
            dt_s=dt_s,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
            candidate_window_count=candidate_count,
            windows=tuple(windows),
            maximum_validity_utilization=maximum_validity,
            actuator_context_sample_count=context_sample_count,
            actuator_context_fingerprint=context_fingerprint,
        ),
        None,
    )


def _window_squared_errors(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
    *,
    apply_bias: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window whitened squared endpoint error and supported rank.

    ``apply_bias`` selects the incumbent convention (the held-out bias the
    runtime applies today) or the candidate convention (no bias, because a
    commit marks that evidence stale).
    """

    initial, history, controls, target, exogenous, bias = _stacked_window_values(
        context
    )
    if not apply_bias:
        bias = jnp.zeros_like(bias)
    errors = np.asarray(
        compiled_batched_endpoint_tangent_error(
            jnp.asarray(vector),
            belief.params,
            initial,
            history,
            controls,
            target,
            exogenous,
            bias,
            dt_s=context.dt_s,
            control_roles=belief.input_spec.control_roles,
            exogenous_roles=belief.input_spec.exogenous_roles,
        ),
        dtype=np.float64,
    )
    squared = np.empty(len(context.windows), dtype=np.float64)
    dimensions = np.empty(len(context.windows), dtype=np.int64)
    for index, (window, error) in enumerate(zip(context.windows, errors)):
        whitened = window.error_support.whiten_vector(error)
        squared[index] = float(whitened @ whitened)
        dimensions[index] = len(whitened)
    return squared, dimensions


def _normalized_rms(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
    *,
    apply_bias: bool,
) -> float:
    squared, dimensions = _window_squared_errors(
        belief,
        context,
        vector,
        apply_bias=apply_bias,
    )
    return float(np.sqrt(np.sum(squared) / float(np.sum(dimensions))))


def _candidate_rollouts_supported(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
) -> bool:
    """Require every proposed rollout path to remain finite and in support."""

    initial, history, controls, _, exogenous, _ = _stacked_window_values(context)
    envelope = belief.runtime_spec.validity_envelope
    supported = _compiled_batched_candidate_rollouts_supported(
        jnp.asarray(vector),
        belief.params,
        initial,
        history,
        controls,
        exogenous,
        jnp.asarray(envelope.body_velocity_center_m_s),
        jnp.asarray(envelope.body_velocity_half_width_m_s),
        jnp.asarray(envelope.angular_velocity_center_rad_s),
        jnp.asarray(envelope.angular_velocity_half_width_rad_s),
        dt_s=context.dt_s,
        control_roles=belief.input_spec.control_roles,
        exogenous_roles=belief.input_spec.exogenous_roles,
    )
    return bool(np.asarray(supported))


def _proposal_geometry(
    belief: DynamicsBelief,
    context: _UpdateContext,
    vector: np.ndarray,
    prior_support: SupportedCovariance,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize the *uncorrected* endpoint error around one parameter vector.

    The held-out bias is excluded because a committed update stales it: the
    objective a candidate is proposed for is the same uncorrected forecast the
    acceptance test scores and the runtime then flies.
    """

    prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
    rows: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    initial, history, controls, target, exogenous, bias = _stacked_window_values(
        context
    )
    errors, jacobians = compiled_batched_endpoint_tangent_linearization(
        jnp.asarray(vector),
        belief.params,
        initial,
        history,
        controls,
        target,
        exogenous,
        jnp.zeros_like(bias),
        dt_s=context.dt_s,
        control_roles=belief.input_spec.control_roles,
        exogenous_roles=belief.input_spec.exogenous_roles,
    )
    for window, error, jacobian in zip(context.windows, errors, jacobians):
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
        before_squared, dimensions = _window_squared_errors(
            belief,
            context,
            prior_vector,
            apply_bias=True,
        )
        candidate: np.ndarray | None = None
        evidence: _ImprovementEvidence | None = None
        selected_fraction: float | None = None
        selected_local_rms: float | None = None
        for fraction in LINE_SEARCH_FRACTIONS:
            combined_fraction = trust_fraction * fraction
            selected = prior_vector + combined_fraction * raw_delta
            after_squared, _ = _window_squared_errors(
                belief,
                context,
                selected,
                apply_bias=False,
            )
            scored = _improvement_evidence(before_squared, after_squared, dimensions)
            if scored.improves and _candidate_rollouts_supported(
                belief, context, selected
            ):
                candidate = selected
                evidence = scored
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
    assert evidence is not None
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
        normalized_innovation_rms_before=evidence.before_rms,
        normalized_innovation_rms_after=evidence.after_rms,
        normalized_innovation_improvement=evidence.total_reduction,
        normalized_innovation_improvement_margin=evidence.margin,
        prior_standardized_step_rms=selected_local_rms,
        proposal_step_fraction=selected_fraction,
        maximum_validity_utilization=context.maximum_validity_utilization,
        source_group=_source_group(telemetry),
        evidence_transition_hashes=_transition_hashes(telemetry),
        base_belief_fingerprint=_update_revision_fingerprint(belief),
        target_spec_fingerprint=_trajectory_spec_fingerprint(telemetry),
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
        normalized_innovation_rms_before=evidence.before_rms,
        normalized_innovation_rms_after=evidence.after_rms,
        normalized_innovation_improvement=evidence.total_reduction,
        normalized_innovation_improvement_margin=evidence.margin,
        structured_parameter_delta_norm=float(np.linalg.norm(candidate - prior_vector)),
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
        and _update_revision_fingerprint(belief) == proposal.base_belief_fingerprint
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
    *,
    validation_control_history: np.ndarray | None = None,
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
    if (
        _trajectory_spec_fingerprint(validation_telemetry)
        != proposal.target_spec_fingerprint
    ):
        return belief, _base_report(
            belief,
            validation_telemetry,
            reason="validation telemetry target specification changed",
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
    if validation_control_history is None:
        return belief, _base_report(
            belief,
            validation_telemetry,
            reason="validation requires preceding actuator command context",
            maximum_validity=maximum_validity,
            proposal_available=True,
            validation_performed=False,
            horizon_s=proposal.update_horizon_s,
            horizon_steps=proposal.update_horizon_steps,
            proposal_count=proposal.proposal_window_count,
        )
    context, rejected = _preflight(
        belief,
        validation_telemetry,
        preceding_control_history=validation_control_history,
    )
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
    if context.horizon_steps != proposal.update_horizon_steps or not np.isclose(
        context.horizon_s, proposal.update_horizon_s
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
            actuator_context_sample_count=context.actuator_context_sample_count,
            actuator_context_fingerprint=context.actuator_context_fingerprint,
        )
    base = proposal.base_parameter_vector
    proposed_delta = proposal.candidate_parameter_vector - base
    try:
        before_squared, dimensions = _window_squared_errors(
            belief,
            context,
            base,
            apply_bias=True,
        )
        validation_before: float | None = float(
            np.sqrt(np.sum(before_squared) / float(np.sum(dimensions)))
        )
        selected: np.ndarray | None = None
        validation_evidence: _ImprovementEvidence | None = None
        validation_fraction: float | None = None
        improved_but_unsupported = False
        best_evidence: _ImprovementEvidence | None = None
        for fraction in LINE_SEARCH_FRACTIONS:
            candidate = base + fraction * proposed_delta
            after_squared, _ = _window_squared_errors(
                belief,
                context,
                candidate,
                apply_bias=False,
            )
            scored = _improvement_evidence(before_squared, after_squared, dimensions)
            if (
                best_evidence is None
                or scored.total_reduction > best_evidence.total_reduction
            ):
                best_evidence = scored
            if not scored.improves:
                continue
            if not _candidate_rollouts_supported(belief, context, candidate):
                improved_but_unsupported = True
                continue
            selected = candidate
            validation_evidence = scored
            validation_fraction = fraction
            break
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        selected = None
        validation_evidence = None
        validation_fraction = None
        validation_before = None
        improved_but_unsupported = False
        best_evidence = None
    if selected is None:
        reason = (
            "proposal validation rollouts left the learned validity envelope"
            if improved_but_unsupported
            else "proposal did not improve disjoint validation telemetry"
        )
        report = _base_report(
            belief,
            validation_telemetry,
            reason=reason,
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
            actuator_context_sample_count=context.actuator_context_sample_count,
            actuator_context_fingerprint=context.actuator_context_fingerprint,
        )
        report = replace(
            report,
            normalized_innovation_rms_before=(
                proposal.normalized_innovation_rms_before
            ),
            normalized_innovation_rms_after=(proposal.normalized_innovation_rms_after),
            normalized_innovation_improvement=(
                proposal.normalized_innovation_improvement
            ),
            normalized_innovation_improvement_margin=(
                proposal.normalized_innovation_improvement_margin
            ),
            normalized_validation_rms_before=validation_before,
            normalized_validation_improvement=(
                None if best_evidence is None else best_evidence.total_reduction
            ),
            normalized_validation_improvement_margin=(
                None if best_evidence is None else best_evidence.margin
            ),
        )
        return belief, report

    assert validation_evidence is not None
    assert validation_fraction is not None
    parameter_belief = belief.parameter_belief
    assert isinstance(parameter_belief, LocalGaussianParameterBelief)
    prior_covariance = proposal.base_parameter_covariance
    accepted_fraction = proposal.proposal_step_fraction * validation_fraction
    try:
        covariance = prior_covariance
        information_gain: float | None = None
        covariance_updated = False
        if proposal.covariance_scope == ErrorCovarianceScope.CONDITIONAL_INNOVATION:
            prior_support = supported_covariance(prior_covariance)
            prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
            validation_design, _ = _proposal_geometry(
                belief,
                context,
                base,
                prior_support,
            )
            validation_information = validation_design.T @ validation_design
            normalized_posterior_precision = (
                np.eye(prior_support.rank)
                + validation_fraction * validation_information
            )
            if not np.all(np.isfinite(normalized_posterior_precision)):
                raise FloatingPointError
            covariance = prior_factor @ np.linalg.solve(
                normalized_posterior_precision,
                prior_factor.T,
            )
            covariance = 0.5 * (covariance + covariance.T)
            sign, logdet = np.linalg.slogdet(normalized_posterior_precision)
            if (
                sign <= 0.0
                or not np.isfinite(logdet)
                or not np.all(np.isfinite(covariance))
            ):
                raise FloatingPointError
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
            actuator_context_sample_count=context.actuator_context_sample_count,
            actuator_context_fingerprint=context.actuator_context_fingerprint,
            normalized_innovation_rms_before=(
                proposal.normalized_innovation_rms_before
            ),
            normalized_innovation_rms_after=(proposal.normalized_innovation_rms_after),
            normalized_innovation_improvement=(
                proposal.normalized_innovation_improvement
            ),
            normalized_innovation_improvement_margin=(
                proposal.normalized_innovation_improvement_margin
            ),
            normalized_validation_rms_before=validation_before,
            normalized_validation_rms_after=validation_evidence.after_rms,
            normalized_validation_improvement=(
                validation_evidence.total_reduction
            ),
            normalized_validation_improvement_margin=validation_evidence.margin,
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
            params=with_structured_parameter_vector(
                belief.params,
                jnp.asarray(selected),
            ),
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
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError, ValueError):
        report = _base_report(
            belief,
            validation_telemetry,
            reason="validated belief construction was non-finite or inconsistent",
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
            actuator_context_sample_count=context.actuator_context_sample_count,
            actuator_context_fingerprint=context.actuator_context_fingerprint,
        )
        return belief, replace(
            report,
            normalized_innovation_rms_before=(
                proposal.normalized_innovation_rms_before
            ),
            normalized_innovation_rms_after=(proposal.normalized_innovation_rms_after),
            normalized_innovation_improvement=(
                proposal.normalized_innovation_improvement
            ),
            normalized_innovation_improvement_margin=(
                proposal.normalized_innovation_improvement_margin
            ),
            normalized_validation_rms_before=validation_before,
            normalized_validation_rms_after=validation_evidence.after_rms,
            normalized_validation_improvement=(
                validation_evidence.total_reduction
            ),
            normalized_validation_improvement_margin=validation_evidence.margin,
        )
    return updated, report


@dataclass(frozen=True)
class HorizonEndpointErrorEvidence:
    """One trajectory's windowed endpoint evidence at one evaluation horizon."""

    horizon_s: float
    horizon_steps: int
    sample: EmpiricalErrorSample
    window_metrics: dict[str, Any]


def endpoint_error_evidence_by_horizon(
    params: ModelParams,
    trajectory: Trajectory,
    *,
    horizons_s: Sequence[float],
    source_group: str,
    trajectory_id: str,
) -> tuple[HorizonEndpointErrorEvidence, ...]:
    """Score nonoverlapping windows and label their endpoint tangent errors.

    This is the single implementation behind both offline evaluation reports and
    online predictive-error recalibration. Horizons longer than the trajectory
    are skipped rather than raising, so one horizon schedule can be applied to
    flights of different lengths.
    """

    records: list[HorizonEndpointErrorEvidence] = []
    for requested in horizons_s:
        steps = duration_to_steps(requested, trajectory.nominal_dt_s)
        if steps > len(trajectory.controls):
            continue
        metrics, endpoint_errors = windowed_rollout_evaluation(
            params,
            trajectory,
            horizon_steps=steps,
            stride_steps=steps,
        )
        metrics["requested_horizon_s"] = requested
        metrics["horizon_steps"] = steps
        records.append(
            HorizonEndpointErrorEvidence(
                horizon_s=float(requested),
                horizon_steps=steps,
                sample=EmpiricalErrorSample(
                    endpoint_errors,
                    source_group=source_group,
                    trajectory_id=trajectory_id,
                ),
                window_metrics=metrics,
            )
        )
    return tuple(records)


def _trajectory_content_hash(trajectory: Trajectory) -> str:
    """Fingerprint the transition content of one telemetry block.

    The digest covers measured transitions rather than timestamps, so the same
    content hashes identically after a time shift. A caller that records this
    hash beside an update's evidence can later show whether recalibration reused
    the block that validated the update.
    """

    digest = hashlib.sha256()
    for value in _transition_hashes(trajectory):
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _recalibration_labels(trajectory: Trajectory) -> tuple[str, str]:
    group = _source_group(trajectory)
    for key in ("trajectory_id", "flight_id", "source_group"):
        value = trajectory.labels.get(key)
        if value is not None and str(value).strip():
            return group, str(value)
    return group, f"recalibration:{_trajectory_content_hash(trajectory)}"


def recalibrate_predictive_error(
    belief: DynamicsBelief,
    trajectory: Trajectory,
    *,
    horizons_s: Sequence[float] | None = None,
    quantile_levels: tuple[float, ...] | None = None,
    covariance_scope: ErrorCovarianceScope | None = None,
    source_group: str | None = None,
    trajectory_id: str | None = None,
) -> DynamicsBelief:
    """Rebuild predictive-error evidence around the belief's current parameters.

    A commit and ``condition_parameter_prior`` both move the parameters and mark
    the attached error evidence stale, because that evidence describes a model
    the belief no longer holds. This is the documented way back: measure the
    forecast errors of the *current* parameters on fresh telemetry and attach
    them, which makes the refreshed belief current again.

    The telemetry must be disjoint from any evidence the caller intends to
    validate a later update against; the returned provenance records a content
    hash so that can be shown after the fact.
    """

    if not isinstance(trajectory, Trajectory):
        raise TypeError("recalibration requires one canonical Trajectory")
    _compatible_telemetry(belief, trajectory)
    existing = belief.predictive_error
    if horizons_s is None:
        if not isinstance(existing, EmpiricalHorizonPredictiveError):
            raise ValueError(
                "recalibration needs explicit horizons when the belief carries "
                "no empirical predictive-error model"
            )
        horizons_s = existing.horizons_s
    requested_horizons = tuple(float(value) for value in horizons_s)
    if not requested_horizons or len(set(requested_horizons)) != len(
        requested_horizons
    ):
        raise ValueError("recalibration horizons must be unique and nonempty")
    if quantile_levels is None:
        quantile_levels = (
            existing.quantile_levels
            if isinstance(existing, EmpiricalHorizonPredictiveError)
            else (0.5, 0.8, 0.9)
        )
    if covariance_scope is None:
        covariance_scope = (
            existing.covariance_scope
            if isinstance(existing, EmpiricalHorizonPredictiveError)
            else ErrorCovarianceScope.TOTAL_FORECAST
        )
    observed_dt_s = trajectory.nominal_dt_s
    if not np.isclose(
        observed_dt_s,
        belief.runtime_spec.sample_period_s,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ValueError(
            "recalibration telemetry sample period does not match the runtime model"
        )
    maximum_validity = _maximum_validity(belief, trajectory)
    if (
        not np.isfinite(maximum_validity)
        or maximum_validity > 1.0 + VALIDITY_BOUNDARY_TOLERANCE
    ):
        raise ValueError(
            "recalibration telemetry lies outside the learned validity envelope"
        )
    default_group, default_id = _recalibration_labels(trajectory)
    records = endpoint_error_evidence_by_horizon(
        belief.params,
        trajectory,
        horizons_s=requested_horizons,
        source_group=default_group if source_group is None else source_group,
        trajectory_id=default_id if trajectory_id is None else trajectory_id,
    )
    if not records:
        raise ValueError(
            "recalibration telemetry is shorter than every requested horizon"
        )
    samples: Mapping[float, tuple[EmpiricalErrorSample, ...]] = {
        record.horizon_s: (record.sample,) for record in records
    }
    predictive_error = replace(
        EmpiricalHorizonPredictiveError.from_samples(
            samples,
            quantile_levels=tuple(quantile_levels),
            covariance_scope=ErrorCovarianceScope(covariance_scope),
        ),
        source=RECALIBRATION_SOURCE,
    )
    update_count = belief.parameter_belief.update_count
    provenance = dict(belief.provenance)
    provenance["predictive_error_recalibration"] = {
        "source": RECALIBRATION_SOURCE,
        "parameter_update_count": update_count,
        "horizons_s": [record.horizon_s for record in records],
        "horizon_steps": [record.horizon_steps for record in records],
        "window_count_by_horizon": [len(record.sample.errors) for record in records],
        "requested_horizons_s": list(requested_horizons),
        "covariance_scope": predictive_error.covariance_scope.value,
        "telemetry_transition_count": len(trajectory.controls),
        "telemetry_content_hash": _trajectory_content_hash(trajectory),
        "telemetry_spec_fingerprint": _trajectory_spec_fingerprint(trajectory),
        "telemetry_source_group": _source_group(trajectory),
        "maximum_validity_utilization": maximum_validity,
    }
    return DynamicsBelief(
        params=belief.params,
        input_spec=belief.input_spec,
        runtime_spec=belief.runtime_spec,
        predictive_error=predictive_error,
        parameter_belief=belief.parameter_belief,
        parameter_evidence=belief.parameter_evidence,
        predictive_error_parameter_update_count=update_count,
        provenance=provenance,
    )


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
        validation_control_history=telemetry.controls[:split],
    )
