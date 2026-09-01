"""First-class predictive beliefs around differentiable Glassbox dynamics.

The nominal dynamics remain a deterministic differentiable model. A belief adds
the forecast errors supported by held-out telemetry, explicit parameter-
uncertainty semantics, and a controller-ready rollout contract. It does not
describe empirical residuals as a Bayesian posterior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree

from glassbox.covariance import supported_covariance
from glassbox.data import TrajectorySpec
from glassbox.dynamics import (
    ModelParams,
    ResidualDynamicsParams,
    control_state_after_history,
    model_family,
    quaternion_multiply,
    step_with_latent,
    structured_parameters,
)
from glassbox.geometry import rigid_body_local_error
from glassbox.runtime import (
    ActuationMap,
    DirectActuationMap,
    RuntimeDynamicsModel,
    RuntimeModelSpec,
)

if TYPE_CHECKING:
    from glassbox.adaptation import BeliefUpdateProposal, BeliefUpdateReport
    from glassbox.data import Trajectory
    from glassbox.parameter_prior import StructuredParameterPrior

TANGENT_STATE_SIZE = 12
TANGENT_STATE_ORDER = (
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "attitude_x",
    "attitude_y",
    "attitude_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
)
TANGENT_GROUP_ORDER = (
    "position",
    "velocity",
    "attitude",
    "angular_velocity",
)
TANGENT_GROUP_SLICES = (
    slice(0, 3),
    slice(3, 6),
    slice(6, 9),
    slice(9, 12),
)
PREDICTIVE_ERROR_FORMAT_VERSION = 2
PARAMETER_BELIEF_FORMAT_VERSION = 1
PARAMETER_EVIDENCE_FORMAT_VERSION = 2


class ErrorCovarianceScope(StrEnum):
    """What variation an empirical tangent covariance already contains."""

    TOTAL_FORECAST = "total_forecast_error"
    CONDITIONAL_INNOVATION = "conditional_innovation_error"


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, level, side="left")), len(values) - 1)
    return float(sorted_values[index])


@dataclass(frozen=True)
class EmpiricalErrorSample:
    """Endpoint tangent errors from one independently labeled trajectory."""

    errors: np.ndarray
    source_group: str
    trajectory_id: str

    def __post_init__(self) -> None:
        errors = np.asarray(self.errors, dtype=np.float64)
        if errors.ndim != 2 or errors.shape[1] != TANGENT_STATE_SIZE:
            raise ValueError("empirical tangent errors must have shape (sample, 12)")
        if len(errors) < 1 or not np.all(np.isfinite(errors)):
            raise ValueError("empirical tangent errors must be finite and nonempty")
        if not self.source_group or not self.trajectory_id:
            raise ValueError("error samples require source-group and trajectory labels")
        object.__setattr__(self, "errors", errors)


@dataclass(frozen=True)
class UnavailablePredictiveError:
    """Explicit absence of predictive-error evidence.

    Runtime moments are zero so the nominal model remains executable, while the
    separate ``available`` flag prevents zero from being mistaken for certainty.
    """

    reason: str = "no held-out predictive-error evidence"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("unavailable predictive error requires a reason")

    @property
    def available(self) -> bool:
        return False

    @property
    def maximum_horizon_s(self) -> float | None:
        return None

    def moments(
        self,
        horizon_s: Array | float,
        state: Array | None = None,
        command: Array | None = None,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array]:
        del horizon_s, state, command, exogenous
        return jnp.zeros(TANGENT_STATE_SIZE), jnp.zeros(
            (TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PREDICTIVE_ERROR_FORMAT_VERSION,
            "kind": "unavailable",
            "available": False,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EmpiricalHorizonPredictiveError:
    """Held-out forecast bias and covariance in rigid-body tangent space."""

    horizons_s: tuple[float, ...]
    tangent_bias: np.ndarray
    tangent_covariance: np.ndarray
    quantile_levels: tuple[float, ...]
    group_radius_quantiles: np.ndarray
    raw_sample_count: tuple[int, ...]
    effective_sample_count: tuple[float, ...]
    independent_group_count: tuple[int, ...]
    covariance_scope: ErrorCovarianceScope = ErrorCovarianceScope.TOTAL_FORECAST
    source: str = "held_out_rollout_endpoints"
    weighting: str = "equal_source_group_then_trajectory_then_endpoint"

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_s)
        if (
            not horizons
            or any(not np.isfinite(value) or value <= 0.0 for value in horizons)
            or any(right <= left for left, right in pairwise(horizons))
        ):
            raise ValueError("predictive-error horizons must be finite and increasing")
        levels = tuple(float(value) for value in self.quantile_levels)
        if (
            not levels
            or any(not 0.0 < value < 1.0 for value in levels)
            or any(right <= left for left, right in pairwise(levels))
        ):
            raise ValueError("quantile levels must be increasing values within (0, 1)")
        bias = np.asarray(self.tangent_bias, dtype=np.float64)
        covariance = np.asarray(self.tangent_covariance, dtype=np.float64)
        radii = np.asarray(self.group_radius_quantiles, dtype=np.float64)
        count = len(horizons)
        if bias.shape != (count, TANGENT_STATE_SIZE):
            raise ValueError("tangent bias must have shape (horizon, 12)")
        if covariance.shape != (count, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE):
            raise ValueError("tangent covariance must have shape (horizon, 12, 12)")
        if radii.shape != (count, len(levels), len(TANGENT_GROUP_ORDER)):
            raise ValueError(
                "group radii must have shape (horizon, quantile, state_group)"
            )
        if not (
            np.all(np.isfinite(bias))
            and np.all(np.isfinite(covariance))
            and np.all(np.isfinite(radii))
        ):
            raise ValueError("predictive-error statistics must be finite")
        if np.any(radii < 0.0):
            raise ValueError("predictive-error radii cannot be negative")
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-10):
            raise ValueError("tangent covariance must be symmetric")
        if any(np.min(np.linalg.eigvalsh(item)) < -1e-9 for item in covariance):
            raise ValueError("tangent covariance must be positive semidefinite")
        counts = tuple(int(value) for value in self.raw_sample_count)
        effective = tuple(float(value) for value in self.effective_sample_count)
        groups = tuple(int(value) for value in self.independent_group_count)
        try:
            covariance_scope = ErrorCovarianceScope(self.covariance_scope)
        except ValueError as error:
            raise ValueError("unsupported predictive-error covariance scope") from error
        if not (
            len(counts) == len(effective) == len(groups) == count
            and all(value > 0 for value in counts)
            and all(np.isfinite(value) and value > 0.0 for value in effective)
            and all(value > 0 for value in groups)
        ):
            raise ValueError("predictive-error evidence counts must be positive")
        if not self.source.strip() or not self.weighting.strip():
            raise ValueError("predictive-error source and weighting are required")
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "quantile_levels", levels)
        object.__setattr__(self, "tangent_bias", bias)
        object.__setattr__(self, "tangent_covariance", covariance)
        object.__setattr__(self, "group_radius_quantiles", radii)
        object.__setattr__(self, "raw_sample_count", counts)
        object.__setattr__(self, "effective_sample_count", effective)
        object.__setattr__(self, "independent_group_count", groups)
        object.__setattr__(self, "covariance_scope", covariance_scope)

    @property
    def available(self) -> bool:
        return True

    @property
    def maximum_horizon_s(self) -> float:
        return self.horizons_s[-1]

    @classmethod
    def from_samples(
        cls,
        samples_by_horizon: Mapping[float, Sequence[EmpiricalErrorSample]],
        *,
        quantile_levels: tuple[float, ...] = (0.5, 0.8, 0.9),
        covariance_scope: ErrorCovarianceScope = (ErrorCovarianceScope.TOTAL_FORECAST),
    ) -> EmpiricalHorizonPredictiveError:
        """Fit group-balanced empirical moments without a Gaussian claim."""

        if not samples_by_horizon:
            raise ValueError("predictive-error fitting requires at least one horizon")
        horizons = tuple(sorted(float(value) for value in samples_by_horizon))
        biases = []
        covariances = []
        radius_quantiles = []
        raw_counts = []
        effective_counts = []
        group_counts = []
        for horizon in horizons:
            samples = tuple(samples_by_horizon[horizon])
            if not samples:
                raise ValueError(
                    f"predictive-error horizon {horizon:g}s has no samples"
                )
            groups = tuple(dict.fromkeys(sample.source_group for sample in samples))
            errors_parts: list[np.ndarray] = []
            weight_parts: list[np.ndarray] = []
            for group in groups:
                group_samples = tuple(
                    sample for sample in samples if sample.source_group == group
                )
                trajectories = tuple(
                    dict.fromkeys(sample.trajectory_id for sample in group_samples)
                )
                for trajectory in trajectories:
                    trajectory_samples = tuple(
                        sample
                        for sample in group_samples
                        if sample.trajectory_id == trajectory
                    )
                    values = np.concatenate(
                        [sample.errors for sample in trajectory_samples], axis=0
                    )
                    weight = 1.0 / (len(groups) * len(trajectories) * len(values))
                    errors_parts.append(values)
                    weight_parts.append(np.full(len(values), weight))
            errors = np.concatenate(errors_parts, axis=0)
            weights = np.concatenate(weight_parts, axis=0)
            weights /= np.sum(weights)
            bias = np.sum(weights[:, None] * errors, axis=0)
            centered = errors - bias
            covariance = np.einsum(
                "n,ni,nj->ij", weights, centered, centered, optimize=True
            )
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            group_radii = np.column_stack(
                [
                    np.linalg.norm(centered[:, group_slice], axis=1)
                    for group_slice in TANGENT_GROUP_SLICES
                ]
            )
            radius_quantiles.append(
                np.asarray(
                    [
                        [
                            _weighted_quantile(group_radii[:, index], weights, level)
                            for index in range(len(TANGENT_GROUP_ORDER))
                        ]
                        for level in quantile_levels
                    ]
                )
            )
            biases.append(bias)
            covariances.append(covariance)
            raw_counts.append(len(errors))
            effective_counts.append(1.0 / float(np.sum(np.square(weights))))
            group_counts.append(len(groups))
        return cls(
            horizons_s=horizons,
            tangent_bias=np.asarray(biases),
            tangent_covariance=np.asarray(covariances),
            quantile_levels=quantile_levels,
            group_radius_quantiles=np.asarray(radius_quantiles),
            raw_sample_count=tuple(raw_counts),
            effective_sample_count=tuple(effective_counts),
            independent_group_count=tuple(group_counts),
            covariance_scope=covariance_scope,
        )

    def moments(
        self,
        horizon_s: Array | float,
        state: Array | None = None,
        command: Array | None = None,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array]:
        """Interpolate predictive moments at one horizon.

        State, command, and exogenous arguments are part of the stable contract;
        this first implementation is deliberately horizon-conditioned only.
        """

        del state, command, exogenous
        horizon = jnp.maximum(jnp.asarray(horizon_s), 0.0)
        knots = jnp.asarray((0.0, *self.horizons_s))
        bias_values = jnp.concatenate(
            (jnp.zeros((1, TANGENT_STATE_SIZE)), jnp.asarray(self.tangent_bias)),
            axis=0,
        )
        covariance_values = jnp.concatenate(
            (
                jnp.zeros((1, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)),
                jnp.asarray(self.tangent_covariance),
            ),
            axis=0,
        )
        bias = jax.vmap(lambda values: jnp.interp(horizon, knots, values))(
            bias_values.T
        )
        covariance = jax.vmap(lambda values: jnp.interp(horizon, knots, values))(
            covariance_values.reshape((len(knots), -1)).T
        ).reshape((TANGENT_STATE_SIZE, TANGENT_STATE_SIZE))
        return bias, 0.5 * (covariance + covariance.T)

    def radius_quantiles(self, horizon_s: Array | float) -> Array:
        """Interpolate empirical state-group error radii at one horizon."""

        horizon = jnp.maximum(jnp.asarray(horizon_s), 0.0)
        knots = jnp.asarray((0.0, *self.horizons_s))
        values = jnp.concatenate(
            (
                jnp.zeros((1, len(self.quantile_levels), len(TANGENT_GROUP_ORDER))),
                jnp.asarray(self.group_radius_quantiles),
            ),
            axis=0,
        )
        return jax.vmap(lambda series: jnp.interp(horizon, knots, series))(
            values.reshape((len(knots), -1)).T
        ).reshape((len(self.quantile_levels), len(TANGENT_GROUP_ORDER)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PREDICTIVE_ERROR_FORMAT_VERSION,
            "kind": "empirical_horizon_tangent_moments",
            "available": True,
            "posterior": False,
            "calibrated_distribution": False,
            "covariance_scope": self.covariance_scope.value,
            "source": self.source,
            "weighting": self.weighting,
            "tangent_state_order": list(TANGENT_STATE_ORDER),
            "state_group_order": list(TANGENT_GROUP_ORDER),
            "horizons_s": list(self.horizons_s),
            "tangent_bias": self.tangent_bias.tolist(),
            "tangent_covariance": self.tangent_covariance.tolist(),
            "quantile_levels": list(self.quantile_levels),
            "group_radius_quantiles": self.group_radius_quantiles.tolist(),
            "raw_sample_count": list(self.raw_sample_count),
            "effective_sample_count": list(self.effective_sample_count),
            "independent_group_count": list(self.independent_group_count),
        }


PredictiveErrorModel = UnavailablePredictiveError | EmpiricalHorizonPredictiveError


def predictive_error_from_dict(payload: Mapping[str, Any]) -> PredictiveErrorModel:
    """Restore one versioned predictive-error implementation."""

    if payload.get("format_version") != PREDICTIVE_ERROR_FORMAT_VERSION:
        raise ValueError("unsupported predictive-error format")
    kind = payload.get("kind")
    if kind == "unavailable":
        return UnavailablePredictiveError(reason=str(payload["reason"]))
    if kind == "empirical_horizon_tangent_moments":
        if payload.get("tangent_state_order") != list(TANGENT_STATE_ORDER):
            raise ValueError("predictive-error tangent state order is incompatible")
        if payload.get("state_group_order") != list(TANGENT_GROUP_ORDER):
            raise ValueError("predictive-error state-group order is incompatible")
        return EmpiricalHorizonPredictiveError(
            horizons_s=tuple(payload["horizons_s"]),
            tangent_bias=np.asarray(payload["tangent_bias"]),
            tangent_covariance=np.asarray(payload["tangent_covariance"]),
            quantile_levels=tuple(payload["quantile_levels"]),
            group_radius_quantiles=np.asarray(payload["group_radius_quantiles"]),
            raw_sample_count=tuple(payload["raw_sample_count"]),
            effective_sample_count=tuple(payload["effective_sample_count"]),
            independent_group_count=tuple(payload["independent_group_count"]),
            covariance_scope=ErrorCovarianceScope(str(payload["covariance_scope"])),
            source=str(payload["source"]),
            weighting=str(payload["weighting"]),
        )
    raise ValueError(f"unsupported predictive-error kind: {kind!r}")


@dataclass(frozen=True)
class PointParameterBelief:
    """Explicit statement that only one fitted parameter member is available."""

    update_count: int = 0

    def __post_init__(self) -> None:
        if self.update_count < 0:
            raise ValueError("parameter-belief update count cannot be negative")

    @property
    def uncertainty_available(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_BELIEF_FORMAT_VERSION,
            "kind": "point_estimate",
            "uncertainty_available": False,
            "scenario_count": 1,
            "update_count": self.update_count,
        }


def structured_parameter_names(params: ModelParams) -> tuple[str, ...]:
    """Return stable scalar names in JAX's structured-parameter leaf order."""

    base = structured_parameters(params)
    names: list[str] = []
    for field_name, value in base._asdict().items():
        array = np.asarray(value)
        if array.ndim == 0:
            names.append(field_name)
            continue
        names.extend(
            f"{field_name}[{','.join(str(index) for index in location)}]"
            for location in np.ndindex(array.shape)
        )
    return tuple(names)


def structured_parameter_vector(params: ModelParams) -> Array:
    """Flatten only the interpretable structured coefficient block."""

    vector, _ = ravel_pytree(structured_parameters(params))
    return vector


def with_structured_parameter_vector(params: ModelParams, vector: Array) -> ModelParams:
    """Replace the structured block while leaving any residual network fixed."""

    expected, unravel = ravel_pytree(structured_parameters(params))
    vector = jnp.asarray(vector)
    if vector.shape != expected.shape:
        raise ValueError(
            f"structured parameter vector has shape {vector.shape}, "
            f"expected {expected.shape}"
        )
    updated_base = unravel(vector)
    return (
        params._replace(base=updated_base)
        if isinstance(params, ResidualDynamicsParams)
        else updated_base
    )


@dataclass(frozen=True)
class LocalGaussianParameterBelief:
    """Local covariance over unconstrained structured effective coefficients.

    The fitted parameters stored by :class:`DynamicsBelief` are the center.
    Residual-network weights are deliberately excluded from this fast belief.
    This is a local approximation and is not labeled a calibrated posterior.
    """

    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    source: str
    evidence_count: int
    effective_sample_count: float
    update_count: int = 0

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.parameter_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("parameter names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (len(names), len(names)):
            raise ValueError("parameter covariance does not match parameter names")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("parameter covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("parameter covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-9:
            raise ValueError("parameter covariance must be positive semidefinite")
        if not np.any(np.diag(covariance) > 0.0):
            raise ValueError("parameter covariance must contain some uncertainty")
        if not self.source.strip():
            raise ValueError("parameter-belief source is required")
        if self.evidence_count < 1:
            raise ValueError("parameter-belief evidence count must be positive")
        if (
            not np.isfinite(self.effective_sample_count)
            or self.effective_sample_count <= 0.0
        ):
            raise ValueError("effective parameter sample count must be positive")
        if self.update_count < 0:
            raise ValueError("parameter-belief update count cannot be negative")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "covariance", 0.5 * (covariance + covariance.T))

    @property
    def uncertainty_available(self) -> bool:
        return True

    @property
    def effective_rank(self) -> int:
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        tolerance = (
            np.finfo(np.float64).eps
            * max(self.covariance.shape)
            * max(float(np.max(eigenvalues)), 1.0)
        )
        return int(np.count_nonzero(eigenvalues > tolerance))

    @classmethod
    def from_members(
        cls,
        nominal: ModelParams,
        members: Sequence[ModelParams],
        *,
        source: str,
        weights: Sequence[float] | None = None,
        update_count: int = 0,
    ) -> LocalGaussianParameterBelief:
        """Summarize evidence members around a separately chosen nominal model."""

        if not members:
            raise ValueError("parameter belief requires at least one evidence member")
        names = structured_parameter_names(nominal)
        center = np.asarray(structured_parameter_vector(nominal), dtype=np.float64)
        vectors = []
        for member in members:
            if structured_parameter_names(member) != names:
                raise ValueError("parameter-belief members have incompatible structure")
            vectors.append(
                np.asarray(structured_parameter_vector(member), dtype=np.float64)
            )
        values = np.asarray(vectors)
        member_weights = (
            np.ones(len(values), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        if member_weights.shape != (len(values),) or not np.all(
            np.isfinite(member_weights)
        ):
            raise ValueError("parameter-belief weights must match finite members")
        if np.any(member_weights < 0.0) or not np.any(member_weights > 0.0):
            raise ValueError("parameter-belief weights must be nonnegative and nonzero")
        member_weights /= np.sum(member_weights)
        deviations = values - center
        covariance = np.einsum(
            "n,ni,nj->ij",
            member_weights,
            deviations,
            deviations,
            optimize=True,
        )
        return cls(
            parameter_names=names,
            covariance=covariance,
            source=source,
            evidence_count=len(values),
            effective_sample_count=1.0 / float(np.sum(np.square(member_weights))),
            update_count=update_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_BELIEF_FORMAT_VERSION,
            "kind": "local_gaussian_structured_parameters",
            "uncertainty_available": True,
            "posterior": False,
            "calibrated_distribution": False,
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_names": list(self.parameter_names),
            "covariance": self.covariance.tolist(),
            "source": self.source,
            "evidence_count": self.evidence_count,
            "effective_sample_count": self.effective_sample_count,
            "effective_rank": self.effective_rank,
            "update_count": self.update_count,
        }


ParameterBelief = PointParameterBelief | LocalGaussianParameterBelief


def parameter_belief_from_dict(payload: Mapping[str, Any]) -> ParameterBelief:
    version = payload.get("format_version", PARAMETER_BELIEF_FORMAT_VERSION)
    if version != PARAMETER_BELIEF_FORMAT_VERSION:
        raise ValueError("unsupported parameter-belief format")
    kind = payload.get("kind")
    if kind == "point_estimate":
        return PointParameterBelief(update_count=int(payload.get("update_count", 0)))
    if kind == "local_gaussian_structured_parameters":
        if payload.get("coordinate_system") != (
            "unconstrained_structured_parameter_vector"
        ):
            raise ValueError("unsupported parameter-belief coordinate system")
        return LocalGaussianParameterBelief(
            parameter_names=tuple(payload["parameter_names"]),
            covariance=np.asarray(payload["covariance"]),
            source=str(payload["source"]),
            evidence_count=int(payload["evidence_count"]),
            effective_sample_count=float(payload["effective_sample_count"]),
            update_count=int(payload.get("update_count", 0)),
        )
    raise ValueError(f"unsupported parameter-belief kind: {kind!r}")


@dataclass(frozen=True)
class UnavailableParameterEvidence:
    """Explicit absence of local parameter-identification evidence."""

    reason: str = "local parameter information was not evaluated"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("unavailable parameter evidence requires a reason")

    @property
    def available(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_EVIDENCE_FORMAT_VERSION,
            "kind": "unavailable",
            "available": False,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LocalParameterInformation:
    """Rank-aware local loss geometry for structured coefficients.

    This object records how held-out-error-whitened rollout predictions change
    around one fitted parameter vector. Unresolved directions remain explicit
    and are never converted into zero variance. The covariance scope records
    whether this geometry can support probabilistic contraction or only a
    regularized mean update.
    """

    parameter_names: tuple[str, ...]
    center: np.ndarray
    information_matrix: np.ndarray
    parameter_scale: np.ndarray
    fitted_parameter_mask: np.ndarray
    horizons_s: tuple[float, ...]
    window_count_by_horizon: tuple[int, ...]
    residual_precision_rank_by_horizon: tuple[int, ...]
    group_labels: tuple[str, ...]
    group_score_vectors: np.ndarray
    independent_group_count: int
    trajectory_count: int
    rank_relative_tolerance: float
    covariance_scope: ErrorCovarianceScope = ErrorCovarianceScope.TOTAL_FORECAST
    source: str = "grouped_rollout_jacobians"
    weighting: str = "sum_independent_groups_mean_horizons_mean_windows"
    residual_precision: str = "held_out_tangent_covariance_pseudoinverse"

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.parameter_names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("parameter-evidence names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("parameter-evidence names must be unique")
        size = len(names)
        center = np.asarray(self.center, dtype=np.float64)
        information = np.asarray(self.information_matrix, dtype=np.float64)
        scale = np.asarray(self.parameter_scale, dtype=np.float64)
        fitted = np.asarray(self.fitted_parameter_mask, dtype=bool)
        if center.shape != (size,) or not np.all(np.isfinite(center)):
            raise ValueError("parameter-evidence center must match finite names")
        if information.shape != (size, size) or not np.all(np.isfinite(information)):
            raise ValueError("parameter information must be a finite square matrix")
        if not np.allclose(information, information.T, atol=1e-9):
            raise ValueError("parameter information must be symmetric")
        information_eigenvalues = np.linalg.eigvalsh(information)
        information_scale = max(float(np.max(np.abs(information_eigenvalues))), 1.0)
        if float(np.min(information_eigenvalues)) < -1e-9 * information_scale:
            raise ValueError("parameter information must be positive semidefinite")
        if (
            scale.shape != (size,)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("parameter-evidence scale must be finite and positive")
        if fitted.shape != (size,):
            raise ValueError("fitted-parameter mask must match parameter names")
        if not np.any(fitted):
            raise ValueError("parameter evidence requires a fitted parameter")
        inactive = ~fitted
        if np.any(np.abs(information[inactive]) > 1e-10 * information_scale):
            raise ValueError(
                "parameters excluded from fitting cannot contain information"
            )
        horizons = tuple(float(value) for value in self.horizons_s)
        windows = tuple(int(value) for value in self.window_count_by_horizon)
        precision_ranks = tuple(
            int(value) for value in self.residual_precision_rank_by_horizon
        )
        group_labels = tuple(str(value) for value in self.group_labels)
        group_scores = np.asarray(self.group_score_vectors, dtype=np.float64)
        if (
            not horizons
            or any(not np.isfinite(value) or value <= 0.0 for value in horizons)
            or any(right <= left for left, right in pairwise(horizons))
        ):
            raise ValueError(
                "parameter-evidence horizons must be finite and increasing"
            )
        if not (
            len(horizons) == len(windows) == len(precision_ranks)
            and all(value > 0 for value in windows)
            and all(0 < value <= TANGENT_STATE_SIZE for value in precision_ranks)
        ):
            raise ValueError("parameter-evidence horizon counts are invalid")
        if self.independent_group_count < 1 or self.trajectory_count < 1:
            raise ValueError("parameter evidence requires groups and trajectories")
        if (
            len(group_labels) != self.independent_group_count
            or len(set(group_labels)) != len(group_labels)
            or any(not value.strip() for value in group_labels)
            or group_scores.shape != (self.independent_group_count, size)
            or not np.all(np.isfinite(group_scores))
        ):
            raise ValueError("parameter-evidence group scores are invalid")
        if np.any(np.abs(group_scores[:, inactive]) > 1e-10 * information_scale):
            raise ValueError("parameters excluded from fitting cannot have scores")
        if (
            not np.isfinite(self.rank_relative_tolerance)
            or not 0.0 < self.rank_relative_tolerance < 1.0
        ):
            raise ValueError("parameter-evidence rank tolerance must lie within (0, 1)")
        if not (
            self.source.strip()
            and self.weighting.strip()
            and self.residual_precision.strip()
        ):
            raise ValueError("parameter-evidence semantics are required")
        try:
            covariance_scope = ErrorCovarianceScope(self.covariance_scope)
        except ValueError as error:
            raise ValueError(
                "unsupported parameter-evidence covariance scope"
            ) from error
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "center", center)
        object.__setattr__(
            self,
            "information_matrix",
            0.5 * (information + information.T),
        )
        object.__setattr__(self, "parameter_scale", scale)
        object.__setattr__(self, "fitted_parameter_mask", fitted)
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "window_count_by_horizon", windows)
        object.__setattr__(
            self,
            "residual_precision_rank_by_horizon",
            precision_ranks,
        )
        object.__setattr__(self, "group_labels", group_labels)
        object.__setattr__(self, "group_score_vectors", group_scores)
        object.__setattr__(self, "covariance_scope", covariance_scope)

    @property
    def available(self) -> bool:
        return True

    @property
    def fitted_parameter_count(self) -> int:
        return int(np.count_nonzero(self.fitted_parameter_mask))

    @property
    def normalized_information_matrix(self) -> np.ndarray:
        return (
            self.parameter_scale[:, None]
            * self.information_matrix
            * self.parameter_scale[None, :]
        )

    @property
    def normalized_information_eigenvalues(self) -> np.ndarray:
        active = self.fitted_parameter_mask
        values = np.linalg.eigvalsh(
            self.normalized_information_matrix[np.ix_(active, active)]
        )
        return np.maximum(values, 0.0)

    @property
    def numerical_rank(self) -> int:
        eigenvalues = self.normalized_information_eigenvalues
        if not np.any(eigenvalues > 0.0):
            return 0
        return int(
            np.count_nonzero(
                eigenvalues > self.rank_relative_tolerance * float(np.max(eigenvalues))
            )
        )

    @property
    def unresolved_fitted_direction_count(self) -> int:
        return self.fitted_parameter_count - self.numerical_rank

    @property
    def score_vector(self) -> np.ndarray:
        """Return the grouped local loss gradient at the evidence center."""

        return np.sum(self.group_score_vectors, axis=0)

    @property
    def group_score_second_moment(self) -> np.ndarray:
        """Return the cluster score outer-product used by sandwich estimators."""

        return self.group_score_vectors.T @ self.group_score_vectors

    @property
    def unresolved_direction_basis(self) -> np.ndarray:
        """Return normalized-coordinate directions unsupported by the evidence."""

        active_indices = np.flatnonzero(self.fitted_parameter_mask)
        inactive_indices = np.flatnonzero(~self.fitted_parameter_mask)
        active_information = self.normalized_information_matrix[
            np.ix_(active_indices, active_indices)
        ]
        eigenvalues, eigenvectors = np.linalg.eigh(active_information)
        threshold = (
            self.rank_relative_tolerance * float(np.max(eigenvalues))
            if len(eigenvalues) and np.max(eigenvalues) > 0.0
            else np.inf
        )
        columns: list[np.ndarray] = []
        for index in inactive_indices:
            direction = np.zeros(len(self.parameter_names))
            direction[index] = 1.0
            columns.append(direction)
        for direction in eigenvectors[:, eigenvalues <= threshold].T:
            embedded = np.zeros(len(self.parameter_names))
            embedded[active_indices] = direction
            columns.append(embedded)
        return (
            np.column_stack(columns)
            if columns
            else np.empty((len(self.parameter_names), 0))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_EVIDENCE_FORMAT_VERSION,
            "kind": "local_structured_parameter_information",
            "available": True,
            "posterior": False,
            "complete_parameter_uncertainty": False,
            "covariance_scope": self.covariance_scope.value,
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_names": list(self.parameter_names),
            "center": self.center.tolist(),
            "information_matrix": self.information_matrix.tolist(),
            "parameter_scale": self.parameter_scale.tolist(),
            "parameter_scale_semantics": (
                "one_transformed_unit_or_same_axis_effective_authority"
            ),
            "fitted_parameter_mask": self.fitted_parameter_mask.tolist(),
            "horizons_s": list(self.horizons_s),
            "window_count_by_horizon": list(self.window_count_by_horizon),
            "residual_precision_rank_by_horizon": list(
                self.residual_precision_rank_by_horizon
            ),
            "group_labels": list(self.group_labels),
            "group_score_vectors": self.group_score_vectors.tolist(),
            "score_vector": self.score_vector.tolist(),
            "group_score_second_moment": (self.group_score_second_moment.tolist()),
            "independent_group_count": self.independent_group_count,
            "trajectory_count": self.trajectory_count,
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "numerical_rank": self.numerical_rank,
            "fitted_parameter_count": self.fitted_parameter_count,
            "unresolved_fitted_direction_count": (
                self.unresolved_fitted_direction_count
            ),
            "normalized_information_eigenvalues": (
                self.normalized_information_eigenvalues.tolist()
            ),
            "source": self.source,
            "weighting": self.weighting,
            "residual_precision": self.residual_precision,
        }


ParameterEvidence = UnavailableParameterEvidence | LocalParameterInformation


def parameter_evidence_from_dict(payload: Mapping[str, Any]) -> ParameterEvidence:
    """Restore one versioned parameter-evidence implementation."""

    if payload.get("format_version") != PARAMETER_EVIDENCE_FORMAT_VERSION:
        raise ValueError("unsupported parameter-evidence format")
    kind = payload.get("kind")
    if kind == "unavailable":
        return UnavailableParameterEvidence(reason=str(payload["reason"]))
    if kind == "local_structured_parameter_information":
        if payload.get("coordinate_system") != (
            "unconstrained_structured_parameter_vector"
        ):
            raise ValueError("unsupported parameter-evidence coordinate system")
        if payload.get("parameter_scale_semantics") != (
            "one_transformed_unit_or_same_axis_effective_authority"
        ):
            raise ValueError("unsupported parameter-evidence scale semantics")
        return LocalParameterInformation(
            parameter_names=tuple(payload["parameter_names"]),
            center=np.asarray(payload["center"]),
            information_matrix=np.asarray(payload["information_matrix"]),
            parameter_scale=np.asarray(payload["parameter_scale"]),
            fitted_parameter_mask=np.asarray(payload["fitted_parameter_mask"]),
            horizons_s=tuple(payload["horizons_s"]),
            window_count_by_horizon=tuple(payload["window_count_by_horizon"]),
            residual_precision_rank_by_horizon=tuple(
                payload["residual_precision_rank_by_horizon"]
            ),
            group_labels=tuple(payload["group_labels"]),
            group_score_vectors=np.asarray(payload["group_score_vectors"]),
            independent_group_count=int(payload["independent_group_count"]),
            trajectory_count=int(payload["trajectory_count"]),
            rank_relative_tolerance=float(payload["rank_relative_tolerance"]),
            covariance_scope=ErrorCovarianceScope(str(payload["covariance_scope"])),
            source=str(payload["source"]),
            weighting=str(payload["weighting"]),
            residual_precision=str(payload["residual_precision"]),
        )
    raise ValueError(f"unsupported parameter-evidence kind: {kind!r}")


def apply_tangent_correction(state: Array, correction: Array) -> Array:
    """Apply one local 12-vector correction to a rigid-body state."""

    angle = jnp.linalg.norm(correction[6:9])
    quaternion_scale = 0.5 * jnp.sinc(angle / (2.0 * jnp.pi))
    delta_quaternion = jnp.concatenate(
        (
            jnp.cos(0.5 * angle)[None],
            quaternion_scale * correction[6:9],
        )
    )
    quaternion = quaternion_multiply(state[6:10], delta_quaternion)
    quaternion /= jnp.maximum(jnp.linalg.norm(quaternion), 1e-12)
    return jnp.concatenate(
        (
            state[0:3] + correction[0:3],
            state[3:6] + correction[3:6],
            quaternion,
            state[10:13] + correction[9:12],
        )
    )


@dataclass(frozen=True)
class PredictiveTrajectory:
    """Nominal and evidence-corrected rollout with tangent uncertainty."""

    nominal_states: Array
    mean_states: Array
    latent_states: Array
    commands: Array
    tangent_bias: Array
    empirical_error_tangent_covariance: Array
    parameter_tangent_covariance: Array
    parameter_tangent_jacobian: Array | None
    quantile_levels: tuple[float, ...]
    group_radius_quantiles: Array | None
    validity_utilization: Array
    predictive_error_available: bool
    predictive_error_current: bool
    predictive_error_horizon_supported: bool
    parameter_uncertainty_available: bool
    empirical_error_covariance_scope: ErrorCovarianceScope | None

    @property
    def tangent_covariance(self) -> Array:
        """Return total local model uncertainty from distinct components."""

        if self.empirical_error_covariance_scope == ErrorCovarianceScope.TOTAL_FORECAST:
            return self.empirical_error_tangent_covariance
        return (
            self.empirical_error_tangent_covariance + self.parameter_tangent_covariance
        )

    @property
    def parameter_covariance_combined_with_empirical_error(self) -> bool:
        return (
            self.parameter_uncertainty_available
            and self.predictive_error_current
            and self.empirical_error_covariance_scope
            == ErrorCovarianceScope.CONDITIONAL_INNOVATION
        )

    @property
    def uncertainty_available(self) -> bool:
        return self.predictive_error_current or self.parameter_uncertainty_available

    @property
    def uncertainty_horizon_supported(self) -> bool:
        """Compatibility summary for the empirical predictive-error horizon."""

        return (
            not self.predictive_error_current or self.predictive_error_horizon_supported
        )

    @property
    def tangent_standard_deviation(self) -> Array:
        return jnp.sqrt(
            jnp.maximum(jnp.diagonal(self.tangent_covariance, axis1=-2, axis2=-1), 0.0)
        )


@dataclass(frozen=True)
class PlanAssessment:
    """Forecast and local information geometry for one candidate maneuver."""

    prediction: PredictiveTrajectory
    maximum_validity_utilization: float
    expected_parameter_information_gain_nats: float | None
    expected_parameter_covariance: np.ndarray | None
    information_available: bool
    information_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_validity_utilization):
            raise ValueError("plan validity utilization must be finite")
        if self.information_available:
            if (
                self.expected_parameter_information_gain_nats is None
                or not np.isfinite(self.expected_parameter_information_gain_nats)
                or self.expected_parameter_information_gain_nats < 0.0
                or self.expected_parameter_covariance is None
            ):
                raise ValueError("available plan information must be finite")
            if self.information_unavailable_reason is not None:
                raise ValueError("available plan information cannot have a reason")
            covariance = np.asarray(
                self.expected_parameter_covariance, dtype=np.float64
            )
            if (
                covariance.ndim != 2
                or covariance.shape[0] != covariance.shape[1]
                or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, atol=1e-9)
                or np.min(np.linalg.eigvalsh(covariance)) < -1e-8
            ):
                raise ValueError("expected parameter covariance must be finite PSD")
            object.__setattr__(self, "expected_parameter_covariance", covariance)
        elif not self.information_unavailable_reason:
            raise ValueError("unavailable plan information requires a reason")


@dataclass(frozen=True)
class DynamicsBelief:
    """Serializable fitted dynamics plus the errors supported by evidence."""

    params: ModelParams
    input_spec: TrajectorySpec
    runtime_spec: RuntimeModelSpec
    predictive_error: PredictiveErrorModel = field(
        default_factory=UnavailablePredictiveError
    )
    parameter_belief: ParameterBelief = field(default_factory=PointParameterBelief)
    parameter_evidence: ParameterEvidence = field(
        default_factory=UnavailableParameterEvidence
    )
    predictive_error_parameter_update_count: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prediction_spec = self.input_spec.prediction_spec()
        family = model_family(self.params)
        if prediction_spec.vehicle.family != family.platform:
            raise ValueError("belief input spec does not match model family")
        family.validate_control_schema(
            prediction_spec.control_names,
            prediction_spec.control_roles,
        )
        names = structured_parameter_names(self.params)
        if (
            isinstance(self.parameter_belief, LocalGaussianParameterBelief)
            and self.parameter_belief.parameter_names != names
        ):
            raise ValueError(
                "parameter belief does not match the nominal structured parameters"
            )
        if (
            isinstance(self.parameter_evidence, LocalParameterInformation)
            and self.parameter_evidence.parameter_names != names
        ):
            raise ValueError(
                "parameter evidence does not match the structured parameters"
            )
        error_update_count = self.predictive_error_parameter_update_count
        if error_update_count is None:
            error_update_count = self.parameter_belief.update_count
        if not 0 <= error_update_count <= self.parameter_belief.update_count:
            raise ValueError(
                "predictive-error update count cannot exceed the parameter belief"
            )
        object.__setattr__(self, "input_spec", prediction_spec)
        object.__setattr__(
            self,
            "predictive_error_parameter_update_count",
            error_update_count,
        )
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def predictive_error_current(self) -> bool:
        return (
            self.predictive_error.available
            and self.predictive_error_parameter_update_count
            == self.parameter_belief.update_count
        )

    def compile_for_nmpc(
        self,
        *,
        actuation: ActuationMap | None = None,
    ) -> RuntimeDynamicsBelief:
        """Bind actionable commands and return the compact runtime belief."""

        selected_actuation = (
            DirectActuationMap(self.input_spec.controls)
            if actuation is None
            else actuation
        )
        nominal = RuntimeDynamicsModel(
            self.params,
            self.input_spec,
            self.runtime_spec,
            selected_actuation,
        )
        return RuntimeDynamicsBelief(
            nominal=nominal,
            predictive_error=self.predictive_error,
            parameter_belief=self.parameter_belief,
            predictive_error_parameter_update_count=(
                self.predictive_error_parameter_update_count
            ),
        )

    def with_parameter_members(
        self,
        members: Sequence[DynamicsBelief | ModelParams],
        *,
        source: str,
        weights: Sequence[float] | None = None,
    ) -> DynamicsBelief:
        """Attach a fleet, configuration, or resampling-derived local prior."""

        member_params = tuple(
            member.params if isinstance(member, DynamicsBelief) else member
            for member in members
        )
        parameter_belief = LocalGaussianParameterBelief.from_members(
            self.params,
            member_params,
            source=source,
            weights=weights,
            update_count=self.parameter_belief.update_count,
        )
        return replace(self, parameter_belief=parameter_belief)

    def condition_parameter_prior(
        self,
        prior: StructuredParameterPrior,
    ) -> DynamicsBelief:
        """Combine local loss geometry with a fleet/configuration prior.

        Conditional innovation covariance supports a local Gaussian contraction.
        Total forecast covariance supports a regularized mean update only, so
        the prior covariance is preserved. Directions absent from the local
        geometry retain their prior mean and covariance.
        """

        if not isinstance(self.parameter_evidence, LocalParameterInformation):
            raise ValueError("conditioning requires local parameter information")
        prior.validate_input_spec(self.input_spec)
        names = self.parameter_evidence.parameter_names
        if prior.parameter_names != names:
            raise ValueError("parameter prior and local evidence are incompatible")
        prior_center = np.asarray(prior.mean, dtype=np.float64)
        prior_covariance = np.asarray(prior.covariance, dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(prior_covariance)
        tolerance = (
            np.finfo(np.float64).eps
            * max(prior_covariance.shape)
            * float(np.max(eigenvalues))
        )
        if np.min(eigenvalues) <= tolerance:
            raise ValueError(
                "conditioning requires a full-rank prior covariance; "
                "empirical subspace spread is incomplete"
            )
        prior_precision = np.linalg.solve(
            prior_covariance,
            np.eye(len(prior_covariance)),
        )
        local_information = self.parameter_evidence.information_matrix
        conditioned_precision = prior_precision + local_information
        conditional_covariance = np.linalg.solve(
            conditioned_precision,
            np.eye(len(conditioned_precision)),
        )
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )
        conditioned_center = conditional_covariance @ (
            prior_precision @ prior_center
            + local_information @ self.parameter_evidence.center
            - self.parameter_evidence.score_vector
        )
        contracts_covariance = (
            self.parameter_evidence.covariance_scope
            == ErrorCovarianceScope.CONDITIONAL_INNOVATION
        )
        conditioned_covariance = (
            conditional_covariance if contracts_covariance else prior_covariance
        )
        update_count = self.parameter_belief.update_count + 1
        parameter_belief = LocalGaussianParameterBelief(
            parameter_names=names,
            covariance=conditioned_covariance,
            source=(
                f"conditional_parameter_prior:{prior.source}"
                if contracts_covariance
                else f"parameter_prior_mean_update:{prior.source}"
            ),
            evidence_count=(
                prior.member_count
                + (
                    self.parameter_evidence.independent_group_count
                    if contracts_covariance
                    else 0
                )
            ),
            effective_sample_count=(
                prior.member_count
                + (
                    self.parameter_evidence.independent_group_count
                    if contracts_covariance
                    else 0
                )
            ),
            update_count=update_count,
        )
        provenance = dict(self.provenance)
        provenance["parameter_prior_conditioning"] = {
            "prior_source": prior.source,
            "prior_method": prior.method,
            "prior_member_count": prior.member_count,
            "prior_empirical_rank": prior.empirical_rank,
            "prior_completion_fraction_in_natural_coordinates": (
                prior.completion_fraction_in_natural_coordinates
            ),
            "prior_artifact": prior.to_dict(),
            "local_evidence_source": self.parameter_evidence.source,
            "local_information_rank": self.parameter_evidence.numerical_rank,
            "local_covariance_scope": (self.parameter_evidence.covariance_scope.value),
            "parameter_covariance_updated": contracts_covariance,
            "local_independent_group_count": (
                self.parameter_evidence.independent_group_count
            ),
        }
        return DynamicsBelief(
            params=with_structured_parameter_vector(
                self.params,
                jnp.asarray(conditioned_center),
            ),
            input_spec=self.input_spec,
            runtime_spec=self.runtime_spec,
            predictive_error=self.predictive_error,
            parameter_belief=parameter_belief,
            parameter_evidence=self.parameter_evidence,
            predictive_error_parameter_update_count=(
                self.predictive_error_parameter_update_count
            ),
            provenance=provenance,
        )

    def update(
        self, telemetry: Trajectory
    ) -> tuple[DynamicsBelief, BeliefUpdateReport]:
        """Propose on early telemetry and commit after disjoint validation."""

        from glassbox.adaptation import update_dynamics_belief

        return update_dynamics_belief(self, telemetry)

    def propose_update(
        self,
        telemetry: Trajectory,
    ) -> tuple[BeliefUpdateProposal | None, BeliefUpdateReport]:
        """Create a bounded update proposal without changing this belief."""

        from glassbox.adaptation import propose_dynamics_belief_update

        return propose_dynamics_belief_update(self, telemetry)

    def commit_update(
        self,
        proposal: BeliefUpdateProposal,
        validation_telemetry: Trajectory,
        *,
        validation_control_history: np.ndarray | None = None,
    ) -> tuple[DynamicsBelief, BeliefUpdateReport]:
        """Validate and commit using explicit pre-segment actuator context."""

        from glassbox.adaptation import (
            validate_and_commit_dynamics_belief_update,
        )

        return validate_and_commit_dynamics_belief_update(
            self,
            proposal,
            validation_telemetry,
            validation_control_history=validation_control_history,
        )

    def save(self, path: str | Path) -> None:
        from glassbox.belief_io import save_dynamics_belief

        save_dynamics_belief(self, path)

    @classmethod
    def load(cls, path: str | Path) -> DynamicsBelief:
        from glassbox.belief_io import load_dynamics_belief

        return load_dynamics_belief(path)


@dataclass(frozen=True)
class RuntimeDynamicsBelief:
    """Executable nominal dynamics and compact predictive-error model."""

    nominal: RuntimeDynamicsModel
    predictive_error: PredictiveErrorModel = field(
        default_factory=UnavailablePredictiveError
    )
    parameter_belief: ParameterBelief = field(default_factory=PointParameterBelief)
    predictive_error_parameter_update_count: int | None = None

    def __post_init__(self) -> None:
        if isinstance(
            self.parameter_belief, LocalGaussianParameterBelief
        ) and self.parameter_belief.parameter_names != structured_parameter_names(
            self.nominal.params
        ):
            raise ValueError("runtime parameter belief does not match nominal model")
        error_update_count = self.predictive_error_parameter_update_count
        if error_update_count is None:
            error_update_count = self.parameter_belief.update_count
        if not 0 <= error_update_count <= self.parameter_belief.update_count:
            raise ValueError("invalid runtime predictive-error update count")
        object.__setattr__(
            self,
            "predictive_error_parameter_update_count",
            error_update_count,
        )

    @classmethod
    def from_nominal(cls, nominal: RuntimeDynamicsModel) -> RuntimeDynamicsBelief:
        return cls(nominal=nominal)

    @property
    def predictive_error_available(self) -> bool:
        return self.predictive_error.available

    @property
    def parameter_uncertainty_available(self) -> bool:
        return self.parameter_belief.uncertainty_available

    @property
    def uncertainty_available(self) -> bool:
        return self.predictive_error_current or self.parameter_uncertainty_available

    @property
    def predictive_error_current(self) -> bool:
        return (
            self.predictive_error.available
            and self.predictive_error_parameter_update_count
            == self.parameter_belief.update_count
        )

    @property
    def maximum_error_horizon_s(self) -> float | None:
        return (
            self.predictive_error.maximum_horizon_s
            if self.predictive_error_current
            else None
        )

    def error_moments(
        self,
        horizon_s: Array | float,
        state: Array | None = None,
        command: Array | None = None,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array]:
        if not self.predictive_error_current:
            return (
                jnp.zeros(TANGENT_STATE_SIZE),
                jnp.zeros((TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)),
            )
        return self.predictive_error.moments(
            horizon_s,
            state=state,
            command=command,
            exogenous=exogenous,
        )

    def corrected_state(
        self,
        nominal_state: Array,
        horizon_s: Array | float,
        command: Array | None = None,
        exogenous: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        bias, covariance = self.error_moments(
            horizon_s,
            state=nominal_state,
            command=command,
            exogenous=exogenous,
        )
        return apply_tangent_correction(nominal_state, bias), bias, covariance

    def _rollout_with_params(
        self,
        params: ModelParams,
        initial_state: Array,
        commands: Array,
        command_history: Array,
        initial_latent_state: Array | None,
        exogenous: Array,
    ) -> tuple[Array, Array, Array]:
        model_controls = jax.vmap(self.nominal.actuation.model_control)(commands)
        if initial_latent_state is None:
            history_controls = jax.vmap(self.nominal.actuation.model_control)(
                command_history
            )
            initial_latent = control_state_after_history(
                params,
                history_controls,
                self.nominal.runtime_spec.sample_period_s,
                self.nominal.input_spec.control_roles,
            )
        else:
            initial_latent = initial_latent_state

        def transition(
            carry: tuple[Array, Array],
            inputs: tuple[Array, Array],
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            state, latent = carry
            control, context = inputs
            next_state, next_latent = step_with_latent(
                params,
                state,
                latent,
                control,
                self.nominal.runtime_spec.sample_period_s,
                self.nominal.input_spec.control_roles,
                context,
                self.nominal.input_spec.exogenous_roles,
            )
            return (next_state, next_latent), (next_state, next_latent)

        _, (future_states, future_latent) = jax.lax.scan(
            transition,
            (initial_state, initial_latent),
            (model_controls, exogenous),
        )
        return future_states, future_latent, initial_latent

    def _parameter_tangent_jacobian(
        self,
        params: ModelParams,
        initial_state: Array,
        commands: Array,
        command_history: Array,
        initial_latent_state: Array | None,
        exogenous: Array,
        mean_future: Array,
        tangent_bias: Array,
    ) -> Array | None:
        if not isinstance(self.parameter_belief, LocalGaussianParameterBelief):
            return None
        center = structured_parameter_vector(params)

        def varied_tangent(vector: Array) -> Array:
            varied_params = with_structured_parameter_vector(params, vector)
            varied_states, _, _ = self._rollout_with_params(
                varied_params,
                initial_state,
                commands,
                command_history,
                initial_latent_state,
                exogenous,
            )
            varied_mean = jax.vmap(apply_tangent_correction)(
                varied_states, tangent_bias
            )
            return jax.vmap(rigid_body_local_error)(mean_future, varied_mean)

        return jax.jacrev(varied_tangent)(center)

    def rollout(
        self,
        initial_state: Array,
        commands: Array,
        *,
        model_parameters: ModelParams | None = None,
        command_history: Array | None = None,
        initial_latent_state: Array | None = None,
        exogenous: Array | None = None,
    ) -> PredictiveTrajectory:
        """Roll out nominal dynamics and attach predictive tangent moments."""

        commands = jnp.asarray(commands)
        if commands.ndim != 2 or commands.shape[1] != self.nominal.command_size:
            raise ValueError("commands must have shape (time, command_size)")
        if len(commands) < 1:
            raise ValueError("belief rollout requires at least one command")
        if exogenous is None:
            exogenous = jnp.zeros((len(commands), self.nominal.exogenous_size))
        else:
            exogenous = jnp.asarray(exogenous)
        if exogenous.shape != (len(commands), self.nominal.exogenous_size):
            raise ValueError("exogenous forecast does not match command timeline")
        initial_state = jnp.asarray(initial_state)
        history = (
            commands[0:1] if command_history is None else jnp.asarray(command_history)
        )
        if history.ndim == 1:
            history = history[None, :]
        if history.ndim != 2 or history.shape[1] != self.nominal.command_size:
            raise ValueError("command history must have shape (time, command_size)")
        provided_latent = (
            None if initial_latent_state is None else jnp.asarray(initial_latent_state)
        )
        selected_parameters = (
            self.nominal.params if model_parameters is None else model_parameters
        )
        future_states, future_latent, resolved_initial_latent = (
            self._rollout_with_params(
                selected_parameters,
                initial_state,
                commands,
                history,
                provided_latent,
                exogenous,
            )
        )
        nominal_states = jnp.concatenate((initial_state[None, :], future_states))
        latent_states = jnp.concatenate(
            (resolved_initial_latent[None, :], future_latent)
        )
        horizons = self.nominal.runtime_spec.sample_period_s * jnp.arange(
            1, len(commands) + 1
        )
        mean_future, bias, residual_covariance = jax.vmap(self.corrected_state)(
            future_states,
            horizons,
            commands,
            exogenous,
        )
        parameter_jacobian = self._parameter_tangent_jacobian(
            selected_parameters,
            initial_state,
            commands,
            history,
            provided_latent,
            exogenous,
            mean_future,
            bias,
        )
        if parameter_jacobian is None:
            parameter_covariance = jnp.zeros_like(residual_covariance)
        else:
            parameter_covariance = jnp.einsum(
                "tip,pq,tjq->tij",
                parameter_jacobian,
                jnp.asarray(self.parameter_belief.covariance),
                parameter_jacobian,
            )
        mean_states = jnp.concatenate((initial_state[None, :], mean_future))
        tangent_bias = jnp.concatenate((jnp.zeros((1, TANGENT_STATE_SIZE)), bias))
        empirical_error_tangent_covariance = jnp.concatenate(
            (
                jnp.zeros((1, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)),
                residual_covariance,
            )
        )
        parameter_tangent_covariance = jnp.concatenate(
            (
                jnp.zeros((1, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)),
                parameter_covariance,
            )
        )
        if (
            isinstance(
                self.predictive_error,
                EmpiricalHorizonPredictiveError,
            )
            and self.predictive_error_current
        ):
            future_radii = jax.vmap(self.predictive_error.radius_quantiles)(horizons)
            group_radius_quantiles = jnp.concatenate(
                (
                    jnp.zeros(
                        (
                            1,
                            len(self.predictive_error.quantile_levels),
                            len(TANGENT_GROUP_ORDER),
                        )
                    ),
                    future_radii,
                )
            )
            quantile_levels = self.predictive_error.quantile_levels
        else:
            group_radius_quantiles = None
            quantile_levels = ()
        initial_context = exogenous[0]
        validity = jnp.concatenate(
            (
                self.nominal.validity_utilization(initial_state, initial_context)[
                    None, :
                ],
                jax.vmap(self.nominal.validity_utilization)(mean_future, exogenous),
            )
        )
        maximum_horizon = self.maximum_error_horizon_s
        return PredictiveTrajectory(
            nominal_states=nominal_states,
            mean_states=mean_states,
            latent_states=latent_states,
            commands=commands,
            tangent_bias=tangent_bias,
            empirical_error_tangent_covariance=(empirical_error_tangent_covariance),
            parameter_tangent_covariance=parameter_tangent_covariance,
            parameter_tangent_jacobian=parameter_jacobian,
            quantile_levels=quantile_levels,
            group_radius_quantiles=group_radius_quantiles,
            validity_utilization=validity,
            predictive_error_available=self.predictive_error_available,
            predictive_error_current=self.predictive_error_current,
            predictive_error_horizon_supported=(
                maximum_horizon is not None
                and len(commands) * self.nominal.runtime_spec.sample_period_s
                <= maximum_horizon + 1e-12
            ),
            parameter_uncertainty_available=self.parameter_uncertainty_available,
            empirical_error_covariance_scope=(
                self.predictive_error.covariance_scope
                if isinstance(
                    self.predictive_error,
                    EmpiricalHorizonPredictiveError,
                )
                and self.predictive_error_current
                else None
            ),
        )

    def assess_plan(
        self,
        initial_state: Array,
        commands: Array,
        *,
        command_history: Array | None = None,
        initial_latent_state: Array | None = None,
        exogenous: Array | None = None,
    ) -> PlanAssessment:
        """Evaluate model support and expected local information for a plan."""

        prediction = self.rollout(
            initial_state,
            commands,
            command_history=command_history,
            initial_latent_state=initial_latent_state,
            exogenous=exogenous,
        )
        maximum_validity = float(np.max(np.asarray(prediction.validity_utilization)))
        unavailable_reason = None
        if not isinstance(self.parameter_belief, LocalGaussianParameterBelief):
            unavailable_reason = "parameter uncertainty is unavailable"
        elif not self.predictive_error_available:
            unavailable_reason = "predictive-error covariance is unavailable"
        elif not self.predictive_error_current:
            unavailable_reason = "predictive-error evidence is stale"
        elif not prediction.predictive_error_horizon_supported:
            unavailable_reason = "candidate plan exceeds predictive-error evidence"
        elif prediction.parameter_tangent_jacobian is None:
            unavailable_reason = "parameter sensitivity is unavailable"
        elif not isinstance(
            self.predictive_error,
            EmpiricalHorizonPredictiveError,
        ) or (
            self.predictive_error.covariance_scope
            != ErrorCovarianceScope.CONDITIONAL_INNOVATION
        ):
            unavailable_reason = (
                "parameter information requires conditional innovation covariance"
            )
        if unavailable_reason is not None:
            return PlanAssessment(
                prediction=prediction,
                maximum_validity_utilization=maximum_validity,
                expected_parameter_information_gain_nats=None,
                expected_parameter_covariance=None,
                information_available=False,
                information_unavailable_reason=unavailable_reason,
            )

        prior = np.asarray(self.parameter_belief.covariance, dtype=np.float64)
        prior_support = supported_covariance(prior)
        if prior_support.rank == 0:
            return PlanAssessment(
                prediction=prediction,
                maximum_validity_utilization=maximum_validity,
                expected_parameter_information_gain_nats=None,
                expected_parameter_covariance=None,
                information_available=False,
                information_unavailable_reason=(
                    "parameter covariance has no supported direction"
                ),
            )
        jacobian = np.asarray(
            prediction.parameter_tangent_jacobian[-1], dtype=np.float64
        )
        residual_support = supported_covariance(
            np.asarray(prediction.empirical_error_tangent_covariance[-1])
        )
        if residual_support.rank == 0:
            return PlanAssessment(
                prediction=prediction,
                maximum_validity_utilization=maximum_validity,
                expected_parameter_information_gain_nats=None,
                expected_parameter_covariance=None,
                information_available=False,
                information_unavailable_reason=(
                    "conditional innovation covariance has rank zero"
                ),
            )
        prior_factor = prior_support.basis * np.sqrt(prior_support.variances)
        whitened_jacobian = residual_support.whiten_rows(jacobian @ prior_factor)
        normalized_information = (
            np.eye(prior_support.rank) + whitened_jacobian.T @ whitened_jacobian
        )
        sign, logdet = np.linalg.slogdet(normalized_information)
        if sign <= 0.0 or not np.isfinite(logdet):
            return PlanAssessment(
                prediction=prediction,
                maximum_validity_utilization=maximum_validity,
                expected_parameter_information_gain_nats=None,
                expected_parameter_covariance=None,
                information_available=False,
                information_unavailable_reason=(
                    "conditional information geometry is non-finite"
                ),
            )
        posterior = prior_factor @ np.linalg.solve(
            normalized_information,
            prior_factor.T,
        )
        posterior = 0.5 * (posterior + posterior.T)
        return PlanAssessment(
            prediction=prediction,
            maximum_validity_utilization=maximum_validity,
            expected_parameter_information_gain_nats=float(0.5 * logdet),
            expected_parameter_covariance=posterior,
            information_available=True,
        )
