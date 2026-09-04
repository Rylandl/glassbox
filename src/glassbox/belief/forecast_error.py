"""How wrong a model's forecasts have been on evidence it did not see.

:class:`ForecastErrorEnvelope` is the held-out forecast-error covariance by
horizon, in the twelve rigid-body tangent coordinates. It answers one question,
how much spread a forecast of a given length has shown, and it answers it for
the horizon cap and for the spread the control objective charges. It is not a
correction: the envelope's own mean error stays a number in the fit report and
nothing subtracts it from a runtime forecast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.core.geometry import (
    TANGENT_GROUP_ORDER,
    TANGENT_STATE_ORDER,
    TANGENT_STATE_SIZE,
)

if TYPE_CHECKING:
    from glassbox.core.data import Trajectory
    from glassbox.core.dynamics import ModelParams

FORECAST_ERROR_FORMAT_VERSION = 4


def _owned_array(value: np.ndarray) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


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
        object.__setattr__(self, "errors", _owned_array(errors))


def _grouped_weights(
    samples: Sequence[EmpiricalErrorSample],
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
]:
    """Give each source group, then trajectory, then endpoint equal mass."""

    groups = tuple(dict.fromkeys(sample.source_group for sample in samples))
    error_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for group in groups:
        group_samples = tuple(
            sample for sample in samples if sample.source_group == group
        )
        trajectories = tuple(
            dict.fromkeys(sample.trajectory_id for sample in group_samples)
        )
        for trajectory in trajectories:
            values = np.concatenate(
                [
                    sample.errors
                    for sample in group_samples
                    if sample.trajectory_id == trajectory
                ],
                axis=0,
            )
            weight = 1.0 / (len(groups) * len(trajectories) * len(values))
            error_parts.append(values)
            weight_parts.append(np.full(len(values), weight))
    errors = np.concatenate(error_parts, axis=0)
    weights = np.concatenate(weight_parts, axis=0)
    return errors, weights / np.sum(weights), len(groups)


def mean_error_by_horizon(
    samples_by_horizon: Mapping[float, Sequence[EmpiricalErrorSample]],
) -> dict[str, list[float]]:
    """Return the held-out mean tangent error at each horizon.

    This is the systematic part of the forecast error. It is recorded as a
    number a reader can act on and is deliberately not applied to any runtime
    forecast: correcting a mean measured on other flights moves the model
    without moving the account of what is known.
    """

    means: dict[str, list[float]] = {}
    for horizon in sorted(float(value) for value in samples_by_horizon):
        samples = tuple(samples_by_horizon[horizon])
        if not samples:
            continue
        errors, weights, _ = _grouped_weights(samples)
        means[f"{horizon:g}s"] = np.sum(weights[:, None] * errors, axis=0).tolist()
    return means


@dataclass(frozen=True)
class ForecastErrorEnvelope:
    """Held-out forecast-error second moments by horizon.

    ``tangent_covariance`` is the group-balanced mean of the endpoint error
    outer product, uncentered. Uncentered is the point: nothing removes the
    mean error at runtime, so the envelope has to carry the whole of how wrong
    the forecast has been rather than only its spread about a correction that
    is never applied.
    """

    horizons_s: tuple[float, ...]
    tangent_covariance: np.ndarray
    raw_sample_count: tuple[int, ...]
    effective_sample_count: tuple[float, ...]
    independent_group_count: tuple[int, ...]
    source: str = "held_out_rollout_endpoints"
    weighting: str = "equal_source_group_then_trajectory_then_endpoint"

    def __post_init__(self) -> None:
        horizons = tuple(float(value) for value in self.horizons_s)
        if (
            not horizons
            or any(not np.isfinite(value) or value <= 0.0 for value in horizons)
            or any(right <= left for left, right in pairwise(horizons))
        ):
            raise ValueError("forecast-error horizons must be finite and increasing")
        covariance = np.asarray(self.tangent_covariance, dtype=np.float64)
        count = len(horizons)
        if covariance.shape != (count, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE):
            raise ValueError("tangent covariance must have shape (horizon, 12, 12)")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("forecast-error statistics must be finite")
        if not np.allclose(covariance, np.swapaxes(covariance, 1, 2), atol=1e-10):
            raise ValueError("tangent covariance must be symmetric")
        if any(np.min(np.linalg.eigvalsh(item)) < -1e-9 for item in covariance):
            raise ValueError("tangent covariance must be positive semidefinite")
        counts = tuple(int(value) for value in self.raw_sample_count)
        effective = tuple(float(value) for value in self.effective_sample_count)
        groups = tuple(int(value) for value in self.independent_group_count)
        if not (
            len(counts) == len(effective) == len(groups) == count
            and all(value > 0 for value in counts)
            and all(np.isfinite(value) and value > 0.0 for value in effective)
            and all(value > 0 for value in groups)
        ):
            raise ValueError("forecast-error evidence counts must be positive")
        if not self.source.strip() or not self.weighting.strip():
            raise ValueError("forecast-error source and weighting are required")
        object.__setattr__(self, "horizons_s", horizons)
        object.__setattr__(self, "tangent_covariance", _owned_array(covariance))
        object.__setattr__(self, "raw_sample_count", counts)
        object.__setattr__(self, "effective_sample_count", effective)
        object.__setattr__(self, "independent_group_count", groups)

    @property
    def maximum_horizon_s(self) -> float:
        return self.horizons_s[-1]

    @classmethod
    def from_samples(
        cls,
        samples_by_horizon: Mapping[float, Sequence[EmpiricalErrorSample]],
    ) -> ForecastErrorEnvelope:
        """Fit group-balanced empirical second moments without a Gaussian claim."""

        if not samples_by_horizon:
            raise ValueError("a forecast envelope requires at least one horizon")
        horizons = tuple(sorted(float(value) for value in samples_by_horizon))
        covariances = []
        raw_counts = []
        effective_counts = []
        group_counts = []
        for horizon in horizons:
            samples = tuple(samples_by_horizon[horizon])
            if not samples:
                raise ValueError(f"forecast horizon {horizon:g}s has no samples")
            errors, weights, group_count = _grouped_weights(samples)
            covariance = np.einsum(
                "n,ni,nj->ij", weights, errors, errors, optimize=True
            )
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            covariances.append(covariance)
            raw_counts.append(len(errors))
            effective_counts.append(1.0 / float(np.sum(np.square(weights))))
            group_counts.append(group_count)
        return cls(
            horizons_s=horizons,
            tangent_covariance=np.asarray(covariances),
            raw_sample_count=tuple(raw_counts),
            effective_sample_count=tuple(effective_counts),
            independent_group_count=tuple(group_counts),
        )

    def covariance_at(self, horizon_s: Array | float) -> Array:
        """Interpolate the forecast-error covariance at one horizon.

        Zero horizon carries zero error, and the interpolation is linear in
        each entry between the measured horizons, held flat beyond the longest.
        """

        horizon = jnp.maximum(jnp.asarray(horizon_s), 0.0)
        knots = jnp.asarray((0.0, *self.horizons_s))
        values = jnp.concatenate(
            (
                jnp.zeros((1, TANGENT_STATE_SIZE, TANGENT_STATE_SIZE)),
                jnp.asarray(self.tangent_covariance),
            ),
            axis=0,
        )
        covariance = jax.vmap(lambda entry: jnp.interp(horizon, knots, entry))(
            values.reshape((len(knots), -1)).T
        ).reshape((TANGENT_STATE_SIZE, TANGENT_STATE_SIZE))
        return 0.5 * (covariance + covariance.T)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORECAST_ERROR_FORMAT_VERSION,
            "kind": "held_out_horizon_tangent_second_moments",
            "posterior": False,
            "calibrated_distribution": False,
            "centered": False,
            "source": self.source,
            "weighting": self.weighting,
            "tangent_state_order": list(TANGENT_STATE_ORDER),
            "state_group_order": list(TANGENT_GROUP_ORDER),
            "horizons_s": list(self.horizons_s),
            "tangent_covariance": self.tangent_covariance.tolist(),
            "raw_sample_count": list(self.raw_sample_count),
            "effective_sample_count": list(self.effective_sample_count),
            "independent_group_count": list(self.independent_group_count),
        }


def forecast_error_from_dict(payload: Mapping[str, Any]) -> ForecastErrorEnvelope:
    """Restore one serialized :class:`ForecastErrorEnvelope`."""

    if payload.get("format_version") != FORECAST_ERROR_FORMAT_VERSION:
        raise ValueError("unsupported forecast-error format")
    if payload.get("kind") != "held_out_horizon_tangent_second_moments":
        raise ValueError(f"unsupported forecast-error kind: {payload.get('kind')!r}")
    if payload.get("tangent_state_order") != list(TANGENT_STATE_ORDER):
        raise ValueError("forecast-error tangent state order is incompatible")
    if payload.get("state_group_order") != list(TANGENT_GROUP_ORDER):
        raise ValueError("forecast-error state-group order is incompatible")
    return ForecastErrorEnvelope(
        horizons_s=tuple(payload["horizons_s"]),
        tangent_covariance=np.asarray(payload["tangent_covariance"]),
        raw_sample_count=tuple(payload["raw_sample_count"]),
        effective_sample_count=tuple(payload["effective_sample_count"]),
        independent_group_count=tuple(payload["independent_group_count"]),
        source=str(payload["source"]),
        weighting=str(payload["weighting"]),
    )


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

    Horizons longer than the trajectory are skipped rather than raising, so one
    horizon schedule can be applied to flights of different lengths.
    """

    from glassbox.core.data import duration_to_steps
    from glassbox.core.metrics import predict_windows, rollout_metrics

    records: list[HorizonEndpointErrorEvidence] = []
    for requested in horizons_s:
        steps = duration_to_steps(requested, trajectory.nominal_dt_s)
        if steps > len(trajectory.controls):
            continue
        prediction = predict_windows(
            params,
            trajectory,
            horizon_steps=steps,
            stride=steps,
        )
        metrics = rollout_metrics(prediction)
        metrics["requested_horizon_s"] = requested
        metrics["horizon_steps"] = steps
        records.append(
            HorizonEndpointErrorEvidence(
                horizon_s=float(requested),
                horizon_steps=steps,
                sample=EmpiricalErrorSample(
                    prediction.endpoint_tangent_errors(),
                    source_group=source_group,
                    trajectory_id=trajectory_id,
                ),
                window_metrics=metrics,
            )
        )
    return tuple(records)
