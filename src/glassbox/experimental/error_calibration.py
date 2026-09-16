"""Frozen-model residual calibration for generic vector predictions.

This host-side experiment uses normalized split conformal scores. It predicts
observation error, including bias and hidden variation, without claiming to
separate their causes. Coverage is joint across outputs for one future row,
marginal over exchangeable calibration/test observations. It is NOT conditional
coverage at every input, a trajectory tube, or a distribution-shift guarantee.

Fit the mean, fit the error scale, calibrate the score, then evaluate on four
separate data roles. IDs catch declared overlap; independence and honest IDs
remain the caller's responsibility. No platform equations or GP variances enter.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


def _ids(values, name):
    result = tuple(values)
    if not result or any(not isinstance(v, str) or not v for v in result):
        raise ValueError(f"{name} must contain nonempty string identifiers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _array(value, name, *, ndim):
    result = np.array(value, dtype=float, copy=True)
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PredictionContract:
    """Exact model revision, ordered channels (including units), and interval.

    Use a content fingerprint for model_id. Include causal history in input_names
    when used by the mean. A change of model, units, channels, or timing invalidates
    existing calibration. The caller supplies the current contract at prediction.
    """

    model_id: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dt_s: float

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must identify a frozen model")
        for name in ("input_names", "output_names"):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be finite and positive")


@dataclass(frozen=True)
class ResidualSamples:
    """Observed-minus-predicted errors from a frozen model on reserved rows."""

    contract: PredictionContract
    features: np.ndarray
    residuals: np.ndarray
    sample_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("features", "residuals"):
            object.__setattr__(self, name, _array(getattr(self, name), name, ndim=2))
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids, "sample_ids"))
        n = len(self.sample_ids)
        if self.features.shape != (n, len(self.contract.input_names)):
            raise ValueError("features do not match sample IDs and input contract")
        if self.residuals.shape != (n, len(self.contract.output_names)):
            raise ValueError("residuals do not match sample IDs and output contract")


def conformal_quantile(scores, miscoverage):
    """Order statistic ceil((n+1)*(1-alpha)); infinity if n is insufficient."""
    scores = _array(scores, "scores", ndim=1)
    if len(scores) == 0 or np.any(scores < 0):
        raise ValueError("scores must be nonempty and nonnegative")
    if not np.isfinite(miscoverage) or not 0 < miscoverage < 1:
        raise ValueError("miscoverage must be between zero and one")
    rank = math.ceil((len(scores) + 1) * (1 - miscoverage))
    return (
        float(np.partition(scores, rank - 1)[rank - 1])
        if rank <= len(scores)
        else np.inf
    )


class ErrorInterval(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class ErrorCalibration:
    """A frozen, auditable scale model and independently calibrated score.

    Local scales are weighted neighbor RMS residuals, including bias. The global
    baseline uses per-output RMS on the same scale-fit observations. The positive
    floor is 5% of that RMS (one unit for an exactly zero RMS), purely to define
    normalized scores. It is not a sensor-noise estimate. No GP variance is added.
    """

    contract: PredictionContract
    mode: str
    neighbors: int
    miscoverage: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    reference_features: np.ndarray
    reference_residuals: np.ndarray
    calibration_scores: np.ndarray
    training_sample_ids: tuple[str, ...]
    scale_sample_ids: tuple[str, ...]
    calibration_sample_ids: tuple[str, ...]

    def __post_init__(self):
        if self.mode not in ("global", "local"):
            raise ValueError("mode must be 'global' or 'local'")
        if (
            not isinstance(self.neighbors, int)
            or isinstance(self.neighbors, bool)
            or self.neighbors < 1
        ):
            raise ValueError("neighbors must be a positive integer")
        for name in ("feature_mean", "feature_scale", "calibration_scores"):
            object.__setattr__(self, name, _array(getattr(self, name), name, ndim=1))
        for name in ("reference_features", "reference_residuals"):
            object.__setattr__(self, name, _array(getattr(self, name), name, ndim=2))
        roles = ("training_sample_ids", "scale_sample_ids", "calibration_sample_ids")
        used = set()
        for name in roles:
            ids = _ids(getattr(self, name), name)
            if used.intersection(ids):
                raise ValueError("training, scale and calibration IDs must be disjoint")
            used.update(ids)
            object.__setattr__(self, name, ids)
        n, d, s = (
            len(self.scale_sample_ids),
            len(self.contract.input_names),
            len(self.contract.output_names),
        )
        if self.reference_features.shape != (
            n,
            d,
        ) or self.reference_residuals.shape != (n, s):
            raise ValueError("reference arrays do not match scale IDs and contract")
        if (
            self.feature_mean.shape != (d,)
            or self.feature_scale.shape != (d,)
            or np.any(self.feature_scale <= 0)
        ):
            raise ValueError("feature normalization must match inputs and be positive")
        if len(self.calibration_scores) != len(self.calibration_sample_ids):
            raise ValueError("calibration scores do not match calibration IDs")
        conformal_quantile(self.calibration_scores, self.miscoverage)

    @property
    def quantile(self):
        return conformal_quantile(self.calibration_scores, self.miscoverage)

    def error_scale(self, features, *, contract):
        if contract != self.contract:
            raise ValueError("calibration is stale: prediction contract changed")
        query = np.asarray(features, dtype=float)
        if (
            query.ndim < 1
            or query.shape[-1] != len(contract.input_names)
            or not np.all(np.isfinite(query))
        ):
            raise ValueError("features must be finite and match the input contract")
        shape = (*query.shape[:-1], len(contract.output_names))
        rms = np.sqrt(np.mean(self.reference_residuals**2, axis=0))
        floor = np.where(rms > 0, 0.05 * rms, 1.0)
        if self.mode == "global":
            return np.broadcast_to(np.maximum(rms, floor), shape).copy()
        normalized = ((query - self.feature_mean) / self.feature_scale).reshape(
            -1, len(contract.input_names)
        )
        k = min(self.neighbors, len(self.reference_features))
        scales = []
        # Bounded memory in the number of queries; stable ties for reproducibility.
        for point in normalized:
            distance_sq = np.sum((self.reference_features - point) ** 2, axis=1)
            ids = np.argsort(distance_sq, kind="stable")[:k]
            radius_sq = distance_sq[ids[-1]]
            weights = (
                np.exp(-2 * distance_sq[ids] / radius_sq)
                if radius_sq > 0
                else np.ones(k)
            )
            local_rms = np.sqrt(
                weights @ (self.reference_residuals[ids] ** 2) / weights.sum()
            )
            scales.append(np.maximum(local_rms, floor))
        return np.asarray(scales).reshape(shape)

    def interval(self, features, mean, *, contract):
        scale = self.error_scale(features, contract=contract)
        mean = np.asarray(mean, dtype=float)
        if mean.shape != scale.shape or not np.all(np.isfinite(mean)):
            raise ValueError("mean must be finite and match batch and output contract")
        half_width = self.quantile * scale
        return ErrorInterval(mean - half_width, mean + half_width)

    def save(self, path: str | Path):
        metadata = {
            "format": "glassbox.experimental.error_calibration.v1",
            "contract": asdict(self.contract),
            **{
                name: getattr(self, name)
                for name in (
                    "mode",
                    "neighbors",
                    "miscoverage",
                    "training_sample_ids",
                    "scale_sample_ids",
                    "calibration_sample_ids",
                )
            },
        }
        arrays = {
            name: getattr(self, name)
            for name in (
                "feature_mean",
                "feature_scale",
                "reference_features",
                "reference_residuals",
                "calibration_scores",
            )
        }
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream, metadata=json.dumps(metadata, allow_nan=False), **arrays
            )

    @classmethod
    def load(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.pop("format") != "glassbox.experimental.error_calibration.v1":
                raise ValueError("unsupported error calibration format")
            metadata["contract"] = PredictionContract(**metadata["contract"])
            arrays = {
                name: archive[name] for name in archive.files if name != "metadata"
            }
        return cls(**metadata, **arrays)


def fit_error_calibration(
    scale_samples: ResidualSamples,
    calibration_samples: ResidualSamples,
    *,
    training_sample_ids,
    mode="local",
    neighbors=24,
    miscoverage=0.05,
) -> ErrorCalibration:
    """Keep both residual datasets out of mean fitting and model selection.

    Scale-fitting and score-calibration rows must also be separate. A single score
    per calibration row is max_j(abs(error_j)/scale_j), giving joint vector coverage
    under exchangeability, rather than counting correlated channels as extra rows.
    """
    if scale_samples.contract != calibration_samples.contract:
        raise ValueError("residual sample contracts must match")
    center = scale_samples.features.mean(axis=0)
    spread = scale_samples.features.std(axis=0)
    spread = np.where(spread > 1e-8, spread, 1.0)
    arguments = dict(
        contract=scale_samples.contract,
        mode=mode,
        neighbors=neighbors,
        miscoverage=miscoverage,
        feature_mean=center,
        feature_scale=spread,
        reference_features=(scale_samples.features - center) / spread,
        reference_residuals=scale_samples.residuals,
        training_sample_ids=training_sample_ids,
        scale_sample_ids=scale_samples.sample_ids,
        calibration_sample_ids=calibration_samples.sample_ids,
    )
    provisional = ErrorCalibration(
        **arguments, calibration_scores=np.zeros(len(calibration_samples.sample_ids))
    )
    scale = provisional.error_scale(
        calibration_samples.features, contract=provisional.contract
    )
    scores = np.max(np.abs(calibration_samples.residuals) / scale, axis=1)
    return ErrorCalibration(**arguments, calibration_scores=scores)
