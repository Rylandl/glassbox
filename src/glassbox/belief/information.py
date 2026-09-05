"""What the evidence has resolved about one model's structured parameters.

:class:`ParameterInformation` is the belief's account of its own ignorance. It
carries an accumulated precision over the structured coefficient block, the
per-coordinate scale that makes directions of different physical units
comparable, the mask of coordinates a fitter is allowed to move, and the
one-step innovation noise that weights every new observation. Rank zero is a
point estimate: nothing is known and no direction is resolved. New evidence
can resolve directions before an update takes its step. Nothing here is
floored into small variances, so an unexcited direction
stays unresolved instead of becoming a high-precision observation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from glassbox.core.dynamics import (
    FixedWingDynamicsParams,
    ModelParams,
    model_family,
    structured_parameter_names,
    structured_parameter_vector,
    structured_parameters,
)
from glassbox.core.geometry import TANGENT_GROUP_INDICES, TANGENT_STATE_SIZE

PARAMETER_INFORMATION_FORMAT_VERSION = 2
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)

# The declared floor on one-step innovation variance, one entry per rigid-body
# tangent group. It is a numerical floor rather than a sensor model: it keeps
# the whitening finite when a fitted model reproduces held-out telemetry to
# integration accuracy, and it is far below the innovation any real telemetry
# produces. Raising it would weaken every observation uniformly; the step is
# invariant to a uniform rescaling of the noise, so only the ratios between
# groups carry into the update.
INNOVATION_NOISE_FLOOR_BY_GROUP = {
    "position": 1e-8,
    "velocity": 1e-6,
    "attitude": 1e-8,
    "angular_velocity": 1e-6,
}


def innovation_noise_floor() -> np.ndarray:
    """Return the declared per-coordinate one-step innovation variance floor."""

    floor = np.empty(TANGENT_STATE_SIZE, dtype=np.float64)
    for group, indices in TANGENT_GROUP_INDICES.items():
        floor[list(indices)] = INNOVATION_NOISE_FLOOR_BY_GROUP[group]
    floor.setflags(write=False)
    return floor


def _owned_array(value: np.ndarray, *, dtype: Any = np.float64) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class SupportedCovariance:
    """Positive covariance subspace without invented precision.

    Empirical zero-eigenvalue directions are absent from ``basis``. They are
    never floored into small variances and therefore never become
    high-precision observations.
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


def supported_covariance(covariance: np.ndarray) -> SupportedCovariance:
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
    retained = (
        np.zeros(len(eigenvalues), dtype=bool)
        if maximum <= 0.0
        else eigenvalues > tolerance
    )
    return SupportedCovariance(
        basis=eigenvectors[:, retained],
        variances=eigenvalues[retained],
        ambient_size=len(matrix),
        relative_tolerance=relative_tolerance,
    )


def estimable_structured_parameters(
    params: ModelParams,
    *,
    fixed_response_time: bool = False,
    learn_thrust_command_offset: bool = False,
    diagonal_angular_control: bool = False,
) -> np.ndarray:
    """Return the structured coordinates a fitter is allowed to move.

    A coordinate outside this mask is not merely unexcited: it is held fixed by
    construction, such as the diagonal of the angular control coupling, which
    the effective angular authority already carries.
    """

    names = structured_parameter_names(params)
    estimable = np.ones(len(names), dtype=bool)
    platform = model_family(params).platform
    for index, name in enumerate(names):
        if fixed_response_time and name in {
            "log_motor_time_constant",
            "log_actuator_time_constant",
        }:
            estimable[index] = False
        if platform != "multirotor":
            continue
        if name == "thrust_command_offset_unconstrained":
            estimable[index] = learn_thrust_command_offset
        if name.startswith("angular_control_cross_coupling_unconstrained["):
            location = name.removeprefix(
                "angular_control_cross_coupling_unconstrained["
            ).removesuffix("]")
            row, column = (int(value) for value in location.split(","))
            if row == column or diagonal_angular_control:
                estimable[index] = False
    return estimable


def structured_parameter_scale(params: ModelParams) -> np.ndarray:
    """Return natural perturbation scales for numerical-rank diagnostics."""

    names = structured_parameter_names(params)
    center = np.asarray(structured_parameter_vector(params), dtype=np.float64)
    scale = np.ones(len(names), dtype=np.float64)
    base = structured_parameters(params)
    if not isinstance(base, FixedWingDynamicsParams):
        return scale
    surface_authority = np.asarray(
        base.physical()["surface_angular_accel_per_speed_sq"],
        dtype=np.float64,
    )
    direct_scales = {
        "lateral_surface_cross_angular_accel_per_speed_sq[0]": (surface_authority[0]),
        "lateral_surface_cross_angular_accel_per_speed_sq[1]": (surface_authority[2]),
        "flap_pitch_angular_accel_per_speed_sq": surface_authority[1],
    }
    for index, name in enumerate(names):
        if name in direct_scales:
            scale[index] = max(
                abs(center[index]),
                float(direct_scales[name]),
                1e-6,
            )
    return scale


def default_rank_relative_tolerance(observation_rows: int, size: int) -> float:
    """Return the relative eigenvalue threshold separating resolved from not."""

    return min(0.01, max(size, observation_rows) * _FLOAT32_EPSILON)


@dataclass(frozen=True)
class ParameterInformation:
    """Accumulated precision over one model's structured coefficient block.

    ``precision`` is the information matrix in parameter coordinates. It only
    ever grows: every absorbed observation adds ``J' R^-1 J`` and nothing
    discounts what came before. ``scale`` defines the normalized coordinate
    ``u`` by ``parameters = center + diag(scale) @ u``, so the normalized
    precision ``diag(scale) @ precision @ diag(scale)`` has eigenvalues that
    are comparable across coordinates of different physical units, and the rank
    test is stated on those. The first nonzero matrix sets ``rank_threshold``
    from the relative tolerance. Subsequent ``with_precision`` calls preserve
    this absolute threshold, so adding information cannot erase an unrelated
    resolved direction. ``estimable`` is the mask the fitter declares;
    rows and columns outside it are exactly zero. ``innovation_noise`` is the
    per-coordinate one-step innovation variance ``R`` in the twelve rigid-body
    tangent coordinates, never below ``noise_floor``.

    One unit of precision is one observation's worth of information at the
    noise the producing estimator declared, which is what makes
    ``effective_count`` comparable to it. For a belief the in-flight
    identifier produced over the bootstrap parameterization, the collective
    block is the exception worth stating: that estimator fits the collective
    map on the integrated target and exports its information as an equivalent
    per-transition Gram, rescaled so that dividing it by the declared force
    floor gives the integrated system's precision. So one unit of collective
    precision there is one transition's worth of information at that declared
    floor, not at the residual a per-interval fit would have measured. Its
    angular block carries no such rescaling.
    """

    names: tuple[str, ...]
    precision: np.ndarray
    scale: np.ndarray
    estimable: np.ndarray
    innovation_noise: np.ndarray
    noise_floor: np.ndarray
    effective_count: float
    rank_relative_tolerance: float = 0.01
    source: str = "declared"
    rank_threshold: float | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.names)
        if not names or any(not name.strip() for name in names):
            raise ValueError("parameter-information names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("parameter-information names must be unique")
        size = len(names)
        precision = np.asarray(self.precision, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        estimable = np.asarray(self.estimable, dtype=bool)
        noise = np.asarray(self.innovation_noise, dtype=np.float64)
        floor = np.asarray(self.noise_floor, dtype=np.float64)
        if precision.shape != (size, size) or not np.all(np.isfinite(precision)):
            raise ValueError("parameter precision must be a finite square matrix")
        if not np.allclose(precision, precision.T, atol=1e-9):
            raise ValueError("parameter precision must be symmetric")
        eigenvalues = np.linalg.eigvalsh(precision)
        magnitude = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(np.min(eigenvalues)) < -1e-9 * magnitude:
            raise ValueError("parameter precision must be positive semidefinite")
        if (
            scale.shape != (size,)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("parameter scale must be finite and positive")
        if estimable.shape != (size,):
            raise ValueError("estimable mask must match the parameter names")
        if not np.any(estimable):
            raise ValueError("parameter information requires an estimable coordinate")
        if np.any(np.abs(precision[~estimable]) > 1e-10 * magnitude):
            raise ValueError(
                "coordinates outside the estimable mask carry no precision"
            )
        for values, label in ((noise, "innovation noise"), (floor, "noise floor")):
            if (
                values.shape != (TANGENT_STATE_SIZE,)
                or not np.all(np.isfinite(values))
                or np.any(values <= 0.0)
            ):
                raise ValueError(f"{label} must be twelve finite positive variances")
        if np.any(noise < floor * (1.0 - 1e-12)):
            raise ValueError("innovation noise cannot fall below its declared floor")
        if not np.isfinite(self.effective_count) or self.effective_count < 0.0:
            raise ValueError("effective evidence count cannot be negative")
        tolerance = float(self.rank_relative_tolerance)
        if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("rank tolerance must lie within (0, 1)")
        if not self.source.strip():
            raise ValueError("parameter-information source is required")
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self, "precision", _owned_array(0.5 * (precision + precision.T))
        )
        object.__setattr__(self, "scale", _owned_array(scale))
        object.__setattr__(self, "estimable", _owned_array(estimable, dtype=bool))
        object.__setattr__(self, "innovation_noise", _owned_array(noise))
        object.__setattr__(self, "noise_floor", _owned_array(floor))
        object.__setattr__(self, "effective_count", float(self.effective_count))
        object.__setattr__(self, "rank_relative_tolerance", tolerance)
        threshold = self.rank_threshold
        if threshold is None:
            largest = float(np.max(np.linalg.eigvalsh(self.normalized_precision)))
            threshold = tolerance * largest if largest > 0.0 else None
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0.0):
            raise ValueError("absolute rank threshold must be finite and nonnegative")
        object.__setattr__(self, "rank_threshold", threshold)

    @classmethod
    def unknown(
        cls,
        params: ModelParams,
        *,
        innovation_noise: np.ndarray | None = None,
        estimable: np.ndarray | None = None,
        source: str = "no_evidence",
    ) -> ParameterInformation:
        """Return the rank-zero information of a point estimate."""

        names = structured_parameter_names(params)
        floor = innovation_noise_floor()
        return cls(
            names=names,
            precision=np.zeros((len(names), len(names))),
            scale=structured_parameter_scale(params),
            estimable=(
                estimable_structured_parameters(params)
                if estimable is None
                else np.asarray(estimable, dtype=bool)
            ),
            innovation_noise=(floor if innovation_noise is None else innovation_noise),
            noise_floor=floor,
            effective_count=0.0,
            rank_relative_tolerance=default_rank_relative_tolerance(0, len(names)),
            source=source,
        )

    @classmethod
    def seeded_from_members(
        cls,
        nominal: ModelParams,
        members: Sequence[ModelParams],
        *,
        innovation_noise: np.ndarray | None = None,
        estimable: np.ndarray | None = None,
        weights: Sequence[float] | None = None,
        source: str = "member_configuration_spread",
    ) -> ParameterInformation:
        """Seed precision from the spread of several members around a nominal.

        The members' sample covariance around ``nominal`` is the only thing
        known: its supported subspace is inverted to precision and every other
        direction stays at zero precision, which is to say unknown. This is the
        whole of what a family of related vehicles or configurations can hand a
        new belief; it invents no precision on directions no member moved.
        """

        if not members:
            raise ValueError("a member seed requires at least one member")
        names = structured_parameter_names(nominal)
        center = np.asarray(structured_parameter_vector(nominal), dtype=np.float64)
        vectors = []
        for member in members:
            if structured_parameter_names(member) != names:
                raise ValueError("seed members have incompatible structure")
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
            raise ValueError("seed weights must match finite members")
        if np.any(member_weights < 0.0) or not np.any(member_weights > 0.0):
            raise ValueError("seed weights must be nonnegative and nonzero")
        member_weights = member_weights / np.sum(member_weights)
        deviations = values - center
        covariance = np.einsum(
            "n,ni,nj->ij", member_weights, deviations, deviations, optimize=True
        )
        mask = (
            estimable_structured_parameters(nominal)
            if estimable is None
            else np.asarray(estimable, dtype=bool)
        )
        precision = supported_covariance(covariance).precision
        precision[~mask, :] = 0.0
        precision[:, ~mask] = 0.0
        floor = innovation_noise_floor()
        return cls(
            names=names,
            precision=precision,
            scale=structured_parameter_scale(nominal),
            estimable=mask,
            innovation_noise=(floor if innovation_noise is None else innovation_noise),
            noise_floor=floor,
            effective_count=1.0 / float(np.sum(np.square(member_weights))),
            rank_relative_tolerance=default_rank_relative_tolerance(
                len(values), len(names)
            ),
            source=source,
        )

    @property
    def estimable_count(self) -> int:
        return int(np.count_nonzero(self.estimable))

    @property
    def normalized_precision(self) -> np.ndarray:
        """Return the precision in the unit-comparable normalized coordinates."""

        return self.scale[:, None] * self.precision * self.scale[None, :]

    def _normalized_spectrum(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Eigendecompose the estimable block of the normalized precision.

        Returns the estimable indices, the clipped eigenvalues, their
        eigenvectors, and the absolute threshold that separates resolved
        directions from unresolved ones.
        """

        indices = np.flatnonzero(self.estimable)
        eigenvalues, eigenvectors = np.linalg.eigh(
            self.normalized_precision[np.ix_(indices, indices)]
        )
        eigenvalues = np.maximum(eigenvalues, 0.0)
        largest = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
        threshold = self.rank_threshold if largest > 0.0 else np.inf
        assert threshold is not None
        return indices, eigenvalues, eigenvectors, threshold

    def resolved_rank(self) -> int:
        """Return how many directions the accumulated evidence resolves."""

        _, eigenvalues, _, threshold = self._normalized_spectrum()
        return int(np.count_nonzero(eigenvalues > threshold))

    @property
    def complete(self) -> bool:
        """Whether every estimable direction has finite supported covariance."""

        return self.resolved_rank() == self.estimable_count

    def unresolved_subspace(self) -> np.ndarray:
        """Unknown directions, in parameter coordinates; not zero-variance ones."""

        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        unresolved = eigenvalues <= threshold
        basis = np.zeros((len(self.names), int(np.count_nonzero(unresolved))))
        basis[indices] = eigenvectors[:, unresolved]
        return self.scale[:, None] * basis

    def resolved_subspace(self) -> np.ndarray:
        """Return the resolved directions as columns in parameter coordinates.

        The columns are orthonormal in the normalized coordinates the rank test
        is stated in, mapped back through ``scale``. A direction outside their
        span carries no information and receives no step.
        """

        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        resolved = eigenvalues > threshold
        basis = np.zeros((len(self.names), int(np.count_nonzero(resolved))))
        basis[indices] = eigenvectors[:, resolved]
        return self.scale[:, None] * basis

    def covariance(self) -> np.ndarray:
        """Return the pseudo-inverse of the precision on its resolved subspace.

        This is a partial covariance padded with zeros, not a zero-variance
        claim outside its support. Consumers must also use ``complete`` and
        ``unresolved_subspace``. In particular a controller cannot price this
        as a complete uncertainty bound without an explicit assumption.
        """

        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        resolved = eigenvalues > threshold
        size = len(self.names)
        if not np.any(resolved):
            return np.zeros((size, size))
        basis = eigenvectors[:, resolved]
        normalized = np.zeros((size, size))
        normalized[np.ix_(indices, indices)] = (basis / eigenvalues[resolved]) @ basis.T
        covariance = self.scale[:, None] * normalized * self.scale[None, :]
        return 0.5 * (covariance + covariance.T)

    def covariance_factor(self) -> np.ndarray:
        """Factor the resolved covariance without a second numerical-rank test.

        Small parameter variances can have large predicted effects. Select
        directions using the information's existing threshold, then invert
        those eigenvalues directly instead of thresholding the covariance.
        """

        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        retained = np.flatnonzero(eigenvalues > threshold)[::-1]
        factor = np.zeros((len(self.names), len(retained)))
        factor[indices] = eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
        return self.scale[:, None] * factor

    def authority(self, direction: np.ndarray) -> float:
        """Return how well one parameter direction is resolved, in ``[0, 1]``.

        The direction is expressed in the normalized coordinates, where the
        variance along a unit direction ``u`` is ``u' P u`` for the covariance
        ``P`` above. Authority is that variance compared against the smallest
        the evidence achieves anywhere, ``1 / lambda_max``, and scaled by the
        share of the direction that lies inside the resolved subspace. It is
        therefore one along the single best-resolved direction, ``lambda_i /
        lambda_max`` along any other resolved eigendirection, and exactly zero
        for a direction the evidence does not resolve at all. A mixed direction
        is dominated by its worst-resolved supported component, which is the
        sense in which authority is a minimum rather than an average.
        """

        values = np.asarray(direction, dtype=np.float64)
        if values.shape != (len(self.names),) or not np.all(np.isfinite(values)):
            raise ValueError("authority requires one finite parameter direction")
        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        resolved = eigenvalues > threshold
        if not np.any(resolved):
            return 0.0
        normalized = values / self.scale
        norm = float(np.linalg.norm(normalized))
        if norm <= 0.0:
            raise ValueError("authority requires a nonzero direction")
        unit = normalized[indices] / norm
        components = eigenvectors[:, resolved].T @ unit
        inside = float(np.sum(np.square(components)))
        if inside <= 0.0:
            return 0.0
        variance = float(np.sum(np.square(components) / eigenvalues[resolved]))
        largest = float(np.max(eigenvalues))
        return float(np.clip(inside * inside / (largest * variance), 0.0, 1.0))

    def information_gain_nats(self, delta: np.ndarray) -> float:
        """Return the information one precision increment adds, in nats.

        The gain is ``0.5 log det(I + Lambda^-1 V' dL V)`` on the directions
        this information already resolves, which is the expected reduction in
        differential entropy of the resolved block. Directions the increment
        newly resolves are a rank change rather than a finite gain and are
        reported by :meth:`resolved_rank` instead, so a rank-zero belief always
        reports zero gain.
        """

        increment = np.asarray(delta, dtype=np.float64)
        size = len(self.names)
        if increment.shape != (size, size) or not np.all(np.isfinite(increment)):
            raise ValueError("information gain requires a finite square increment")
        indices, eigenvalues, eigenvectors, threshold = self._normalized_spectrum()
        resolved = eigenvalues > threshold
        if not np.any(resolved):
            return 0.0
        basis = eigenvectors[:, resolved]
        normalized = self.scale[:, None] * increment * self.scale[None, :]
        projected = basis.T @ normalized[np.ix_(indices, indices)] @ basis
        matrix = (
            np.eye(int(np.count_nonzero(resolved)))
            + projected / (eigenvalues[resolved][:, None])
        )
        sign, magnitude = np.linalg.slogdet(matrix)
        if sign <= 0.0:
            raise ValueError("information gain requires a positive increment")
        return 0.5 * float(magnitude)

    def with_precision(
        self,
        precision: np.ndarray,
        *,
        innovation_noise: np.ndarray | None = None,
        effective_count: float | None = None,
        source: str | None = None,
    ) -> ParameterInformation:
        """Return the same declarations carrying a new accumulated precision."""

        return ParameterInformation(
            names=self.names,
            precision=precision,
            scale=self.scale,
            estimable=self.estimable,
            innovation_noise=(
                self.innovation_noise if innovation_noise is None else innovation_noise
            ),
            noise_floor=self.noise_floor,
            effective_count=(
                self.effective_count if effective_count is None else effective_count
            ),
            rank_relative_tolerance=self.rank_relative_tolerance,
            source=self.source if source is None else source,
            rank_threshold=self.rank_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_INFORMATION_FORMAT_VERSION,
            "kind": "structured_parameter_information",
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_scale_semantics": (
                "one_transformed_unit_or_same_axis_effective_authority"
            ),
            "names": list(self.names),
            "precision": self.precision.tolist(),
            "scale": self.scale.tolist(),
            "estimable": self.estimable.tolist(),
            "innovation_noise": self.innovation_noise.tolist(),
            "noise_floor": self.noise_floor.tolist(),
            "effective_count": self.effective_count,
            "rank_relative_tolerance": self.rank_relative_tolerance,
            "rank_threshold": self.rank_threshold,
            "parameter_uncertainty_complete": self.complete,
            "resolved_rank": self.resolved_rank(),
            "estimable_count": self.estimable_count,
            "source": self.source,
        }


def parameter_information_from_dict(
    payload: Mapping[str, Any],
) -> ParameterInformation:
    """Restore one serialized :class:`ParameterInformation`."""

    if payload.get("format_version") not in (1, PARAMETER_INFORMATION_FORMAT_VERSION):
        raise ValueError("unsupported parameter-information format")
    if payload.get("coordinate_system") != (
        "unconstrained_structured_parameter_vector"
    ):
        raise ValueError("unsupported parameter-information coordinate system")
    if payload.get("parameter_scale_semantics") != (
        "one_transformed_unit_or_same_axis_effective_authority"
    ):
        raise ValueError("unsupported parameter-information scale semantics")
    return ParameterInformation(
        names=tuple(payload["names"]),
        precision=np.asarray(payload["precision"]),
        scale=np.asarray(payload["scale"]),
        estimable=np.asarray(payload["estimable"]),
        innovation_noise=np.asarray(payload["innovation_noise"]),
        noise_floor=np.asarray(payload["noise_floor"]),
        effective_count=float(payload["effective_count"]),
        rank_relative_tolerance=float(payload["rank_relative_tolerance"]),
        source=str(payload["source"]),
        rank_threshold=payload.get("rank_threshold"),
    )
