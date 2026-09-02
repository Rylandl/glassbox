"""Rank-supported covariance operations for empirical evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_FLOAT32_EPSILON = np.finfo(np.float32).eps


@dataclass(frozen=True)
class SupportedCovariance:
    """Positive covariance subspace without invented precision.

    Empirical zero-eigenvalue directions are absent from ``basis``. They are
    never floored into small variances and therefore never become high-precision
    observations.
    """

    basis: np.ndarray
    variances: np.ndarray
    ambient_size: int
    relative_tolerance: float

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if self.ambient_size < 1:
            raise ValueError("supported covariance requires a positive size")
        if basis.shape != (self.ambient_size, len(variances)):
            raise ValueError("supported covariance basis has incompatible shape")
        if not (
            np.all(np.isfinite(basis))
            and np.all(np.isfinite(variances))
            and np.all(variances > 0.0)
        ):
            raise ValueError("supported covariance must be finite and positive")
        if len(variances) and not np.allclose(
            basis.T @ basis,
            np.eye(len(variances)),
            atol=1e-9,
        ):
            raise ValueError("supported covariance basis must be orthonormal")
        if not (np.isfinite(self.relative_tolerance) and self.relative_tolerance > 0.0):
            raise ValueError("supported covariance tolerance must be positive")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "variances", variances)

    @property
    def rank(self) -> int:
        return len(self.variances)

    @property
    def covariance(self) -> np.ndarray:
        return (self.basis * self.variances) @ self.basis.T

    @property
    def precision(self) -> np.ndarray:
        return (self.basis * (1.0 / self.variances)) @ self.basis.T

    @property
    def projector(self) -> np.ndarray:
        return self.basis @ self.basis.T

    @property
    def log_pseudodeterminant(self) -> float:
        return float(np.sum(np.log(self.variances)))

    def whiten_vector(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.ambient_size,):
            raise ValueError("vector does not match covariance dimension")
        return (self.basis.T @ values) / np.sqrt(self.variances)

    def whiten_rows(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.ambient_size:
            raise ValueError("matrix rows do not match covariance dimension")
        return (self.basis.T @ values) / np.sqrt(self.variances)[:, None]

    def normalized_rms(self, vector: np.ndarray) -> float | None:
        if self.rank == 0:
            return None
        whitened = self.whiten_vector(vector)
        return float(np.sqrt(np.mean(np.square(whitened))))


def supported_covariance(
    covariance: np.ndarray,
) -> SupportedCovariance:
    """Return the float32-resolvable positive subspace of one covariance."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 1:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    maximum = max(float(np.max(eigenvalues)), 0.0)
    relative_tolerance = max(matrix.shape) * _FLOAT32_EPSILON
    tolerance = relative_tolerance * maximum
    if maximum <= 0.0:
        retained = np.zeros(len(eigenvalues), dtype=bool)
    else:
        retained = eigenvalues > tolerance
    return SupportedCovariance(
        basis=eigenvectors[:, retained],
        variances=eigenvalues[retained],
        ambient_size=len(matrix),
        relative_tolerance=relative_tolerance,
    )
