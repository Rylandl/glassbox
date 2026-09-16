"""Learned affine trends with generic nonlinear kernel corrections.

This is a mean estimator, not an uncertainty model. Input and output widths are
independent. No interpretation of a channel as a vehicle quantity is required.
The affine and nonlinear terms are fitted in two stages, not jointly. Model
selection must use observations separate from those passed to the fitter.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

KINDS = ("linear", "rbf", "additive", "pairwise")


def covariance(left, right, kind, length_scale, *, xp=np):
    """Unit-diagonal kernels; pairwise mixes first/second order equally.

    Products of one-dimensional RBF kernels represent pairwise interactions.
    Averaging keeps the diagonal and regularization comparable across widths.
    Looping over features avoids an N x M x D temporary at prediction time.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kernel: {kind}")
    shape = (left.shape[0], right.shape[0])
    total, squares, distance = (xp.zeros(shape) for _ in range(3))
    for column in range(left.shape[1]):
        d = ((left[:, None, column] - right[None, :, column]) / length_scale) ** 2
        if kind == "rbf":
            distance = distance + d
        else:
            k = xp.exp(-0.5 * d)
            total = total + k
            squares = squares + k * k
    if kind == "rbf":
        return xp.exp(-0.5 * distance / left.shape[1])
    first = total / left.shape[1]
    if kind == "pairwise" and left.shape[1] > 1:
        second = (total * total - squares) / (left.shape[1] * (left.shape[1] - 1))
        return (first + second) / 2
    return first


def array_fingerprint(metadata, arrays):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def save_arrays(path, metadata, arrays):
    meta = {**metadata, "fingerprint": array_fingerprint(metadata, arrays)}
    np.savez_compressed(path, metadata=json.dumps(meta), **arrays)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
        arrays = {k: archive[k] for k in archive.files if k != "metadata"}
    identity = meta.pop("fingerprint")
    if identity != array_fingerprint(meta, arrays):
        raise ValueError("model fingerprint mismatch")
    return meta, arrays


@dataclass(frozen=True)
class StructuredRegressor:
    kind: str
    length_scale: float
    regularization: float
    arrays: dict

    def predict(self, features):
        x = jnp.asarray(features)
        a = self.arrays
        if x.ndim < 1 or x.shape[-1] != len(a["feature_mean"]):
            raise ValueError("feature width does not match fitted model")
        shape = x.shape[:-1]
        z = ((x - a["feature_mean"]) / a["feature_scale"]).reshape(-1, x.shape[-1])
        y = z @ a["linear"]
        if self.kind != "linear":
            y = (
                y
                + covariance(z, a["features"], self.kind, self.length_scale, xp=jnp)
                @ a["alpha"]
            )
        return (y * a["target_scale"] + a["target_mean"]).reshape(
            *shape, len(a["target_mean"])
        )

    def metadata(self):
        return {
            "format": "glassbox-structured-regression-v1",
            "kind": self.kind,
            "length_scale": self.length_scale,
            "regularization": self.regularization,
        }

    def fingerprint(self):
        return array_fingerprint(self.metadata(), self.arrays)

    def save(self, path: str | Path):
        save_arrays(path, self.metadata(), self.arrays)

    @classmethod
    def load(cls, path):
        meta, arrays = load_arrays(path)
        if meta.pop("format") != "glassbox-structured-regression-v1":
            raise ValueError("unsupported structured regression format")
        return cls(**meta, arrays=arrays)


def fit_structured_regressor(
    features, targets, *, kind="pairwise", length_scale=1.0, regularization=0.1
):
    """Fit an affine ridge trend (unit penalty), then a kernel ridge residual.

    RBF uses mean squared standardized distance. Additive/pairwise use a shared
    length across individual standardized features. These are deliberately
    small hyperparameter families, not learned feature selection or metrics.
    """
    x, y = np.array(features, dtype=float), np.array(targets, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(x) != len(y)
        or len(x) < 3
        or min(x.shape[1], y.shape[1]) == 0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError(
            "features/targets must be finite matrices with >=3 paired rows"
        )
    if kind not in KINDS:
        raise ValueError(f"unknown kernel: {kind}")
    if (
        not np.isfinite([length_scale, regularization]).all()
        or min(length_scale, regularization) <= 0
    ):
        raise ValueError("length_scale and regularization must be positive and finite")
    xm, xs, ym, ys = x.mean(0), x.std(0), y.mean(0), y.std(0)
    xs, ys = np.where(xs > 1e-8, xs, 1.0), np.where(ys > 1e-8, ys, 1.0)
    z, t = (x - xm) / xs, (y - ym) / ys
    linear = np.linalg.solve(z.T @ z + np.eye(z.shape[1]), z.T @ t)
    alpha = np.zeros_like(t)
    if kind != "linear":
        kernel = covariance(z, z, kind, length_scale)
        alpha = np.linalg.solve(
            kernel + regularization * np.eye(len(x)), t - z @ linear
        )
    arrays = dict(
        feature_mean=xm,
        feature_scale=xs,
        target_mean=ym,
        target_scale=ys,
        features=z,
        linear=linear,
        alpha=alpha,
    )
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("structured fit produced nonfinite parameters")
    return StructuredRegressor(kind, float(length_scale), float(regularization), arrays)
