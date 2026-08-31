"""Opinionated structured-parameter priors built from vehicle beliefs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.belief import (
    DynamicsBelief,
    LocalGaussianParameterBelief,
    UnavailableParameterEvidence,
    structured_parameter_names,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.data import TrajectorySpec
from glassbox.parameter_evidence import structured_parameter_scale

PARAMETER_PRIOR_FORMAT_VERSION = 1
PARAMETER_PRIOR_ARTIFACT_TYPE = "glassbox_structured_parameter_prior"
PARAMETER_PRIOR_METHOD = "equal_member_natural_nullspace_completion_v1"

ControlContract = tuple[str, str, str, str | None]


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _normalized_covariance(
    covariance: np.ndarray,
    natural_scale: np.ndarray,
) -> np.ndarray:
    inverse_scale = 1.0 / natural_scale
    return covariance * inverse_scale[:, None] * inverse_scale[None, :]


def _numerical_rank(eigenvalues: np.ndarray) -> int:
    tolerance = (
        np.finfo(np.float64).eps
        * len(eigenvalues)
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    return int(np.count_nonzero(eigenvalues > tolerance))


def _control_contracts(spec: TrajectorySpec) -> tuple[ControlContract, ...]:
    return tuple(
        (channel.role, channel.semantic, channel.unit, channel.frame)
        for channel in spec.controls
    )


def _validate_shared_control_contracts(
    left: Sequence[ControlContract],
    right: Sequence[ControlContract],
) -> None:
    left_by_role = {item[0]: item[1:] for item in left}
    right_by_role = {item[0]: item[1:] for item in right}
    shared = set(left_by_role) & set(right_by_role)
    if any(left_by_role[role] != right_by_role[role] for role in shared):
        raise ValueError("fleet members use incompatible control semantics")


@dataclass(frozen=True)
class StructuredParameterPrior:
    """Proper local prior with empirical and assumed uncertainty separated.

    Between-vehicle spread and any member covariances remain visible as
    evidence. Unit variance on only the unresolved natural-coordinate subspace
    makes the prior proper when the fleet is smaller than the parameter
    dimension. Completion is an explicit modeling assumption, never presented
    as observed fleet variance.
    """

    parameter_names: tuple[str, ...]
    mean: np.ndarray
    between_member_covariance: np.ndarray
    within_member_covariance: np.ndarray
    completion_covariance: np.ndarray
    natural_scale: np.ndarray
    member_labels: tuple[str, ...]
    within_member_covariance_count: int
    state_schema: str
    vehicle_family: str
    control_contracts: tuple[ControlContract, ...]
    source: str
    method: str = PARAMETER_PRIOR_METHOD

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.parameter_names)
        size = len(names)
        if not names or len(set(names)) != size or any(not value for value in names):
            raise ValueError("parameter prior requires unique nonempty names")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.natural_scale, dtype=np.float64)
        if mean.shape != (size,) or not np.all(np.isfinite(mean)):
            raise ValueError("parameter-prior mean must match finite names")
        if scale.shape != (size,) or not np.all(np.isfinite(scale)) or np.any(
            scale <= 0.0
        ):
            raise ValueError("parameter-prior natural scale must be positive")
        matrices: dict[str, np.ndarray] = {}
        for name in (
            "between_member_covariance",
            "within_member_covariance",
            "completion_covariance",
        ):
            matrix = np.asarray(getattr(self, name), dtype=np.float64)
            if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite parameter covariance")
            if not np.allclose(matrix, matrix.T, atol=1e-10):
                raise ValueError(f"{name} must be symmetric")
            eigenvalues = np.linalg.eigvalsh(matrix)
            matrix_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            if float(np.min(eigenvalues)) < -1e-9 * matrix_scale:
                raise ValueError(f"{name} must be positive semidefinite")
            matrices[name] = 0.5 * (matrix + matrix.T)
        total_eigenvalues = np.linalg.eigvalsh(
            _normalized_covariance(sum(matrices.values()), scale)
        )
        total_scale = max(float(np.max(np.abs(total_eigenvalues))), 1.0)
        if (
            np.min(total_eigenvalues)
            <= np.finfo(np.float64).eps * size * total_scale
        ):
            raise ValueError("parameter-prior total covariance must be full rank")
        labels = tuple(str(value) for value in self.member_labels)
        if not labels or len(set(labels)) != len(labels) or any(
            not value.strip() for value in labels
        ):
            raise ValueError("parameter prior requires unique member labels")
        if self.within_member_covariance_count not in {0, len(labels)}:
            raise ValueError("invalid within-member covariance count")
        if self.within_member_covariance_count == 0 and not np.allclose(
            matrices["within_member_covariance"],
            0.0,
            atol=1e-12,
        ):
            raise ValueError("within-member covariance requires every fleet member")
        if self.within_member_covariance_count == len(labels) and not np.any(
            np.diag(matrices["within_member_covariance"]) > 0.0
        ):
            raise ValueError("fleet member covariances contain no uncertainty")
        normalized_between = _normalized_covariance(
            matrices["between_member_covariance"],
            scale,
        )
        if _numerical_rank(np.linalg.eigvalsh(normalized_between)) > len(labels) - 1:
            raise ValueError("between-member covariance exceeds fleet evidence rank")
        normalized_empirical = _normalized_covariance(
            matrices["between_member_covariance"]
            + matrices["within_member_covariance"],
            scale,
        )
        empirical_eigenvalues, empirical_eigenvectors = np.linalg.eigh(
            normalized_empirical
        )
        empirical_rank = _numerical_rank(empirical_eigenvalues)
        unresolved_basis = empirical_eigenvectors[:, : size - empirical_rank]
        expected_completion = unresolved_basis @ unresolved_basis.T
        normalized_completion = _normalized_covariance(
            matrices["completion_covariance"],
            scale,
        )
        if not np.allclose(
            normalized_completion,
            expected_completion,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError(
                "parameter-prior completion must be unit covariance on only "
                "the unresolved natural-coordinate subspace"
            )
        contracts = tuple(
            (str(role), str(semantic), str(unit), frame)
            for role, semantic, unit, frame in self.control_contracts
        )
        roles = tuple(item[0] for item in contracts)
        if len(set(roles)) != len(roles) or any(
            not role or not semantic or not unit
            for role, semantic, unit, _ in contracts
        ):
            raise ValueError("parameter-prior control contracts are invalid")
        if not (
            self.state_schema.strip()
            and self.vehicle_family.strip()
            and self.source.strip()
            and self.method == PARAMETER_PRIOR_METHOD
        ):
            raise ValueError("parameter-prior provenance is invalid")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "natural_scale", scale)
        object.__setattr__(self, "member_labels", labels)
        object.__setattr__(self, "control_contracts", contracts)
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)

    @property
    def member_count(self) -> int:
        return len(self.member_labels)

    @property
    def empirical_covariance(self) -> np.ndarray:
        return self.between_member_covariance + self.within_member_covariance

    @property
    def covariance(self) -> np.ndarray:
        return self.empirical_covariance + self.completion_covariance

    @property
    def empirical_rank(self) -> int:
        eigenvalues = np.linalg.eigvalsh(
            _normalized_covariance(self.empirical_covariance, self.natural_scale)
        )
        return _numerical_rank(eigenvalues)

    @property
    def completion_fraction_in_natural_coordinates(self) -> float:
        normalized_total = _normalized_covariance(
            self.covariance,
            self.natural_scale,
        )
        normalized_completion = _normalized_covariance(
            self.completion_covariance,
            self.natural_scale,
        )
        return float(np.trace(normalized_completion)) / float(
            np.trace(normalized_total)
        )

    @classmethod
    def from_beliefs(
        cls,
        members: Sequence[DynamicsBelief],
        *,
        source: str,
        member_labels: Sequence[str] | None = None,
    ) -> StructuredParameterPrior:
        """Build one proper prior with no statistical tuning parameters."""

        if not members:
            raise ValueError("parameter-prior construction requires members")
        if not source.strip():
            raise ValueError("parameter-prior source is required")
        labels = (
            tuple(f"member_{index}" for index in range(len(members)))
            if member_labels is None
            else tuple(str(value) for value in member_labels)
        )
        if len(labels) != len(members):
            raise ValueError("member labels must match prior members")
        reference = members[0]
        names = structured_parameter_names(reference.params)
        state_schema = reference.input_spec.state_schema
        vehicle_family = reference.input_spec.vehicle.family
        contracts_by_role: dict[str, ControlContract] = {
            item[0]: item for item in _control_contracts(reference.input_spec)
        }
        vectors: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        within_covariances: list[np.ndarray] = []
        covariance_availability: list[bool] = []
        for member in members:
            if structured_parameter_names(member.params) != names:
                raise ValueError("fleet members have incompatible parameter structures")
            if member.input_spec.state_schema != state_schema:
                raise ValueError("fleet members use incompatible state schemas")
            if member.input_spec.vehicle.family != vehicle_family:
                raise ValueError("fleet members belong to different vehicle families")
            member_contracts = _control_contracts(member.input_spec)
            _validate_shared_control_contracts(
                tuple(contracts_by_role.values()),
                member_contracts,
            )
            for contract in member_contracts:
                contracts_by_role.setdefault(contract[0], contract)
            vectors.append(
                np.asarray(structured_parameter_vector(member.params), dtype=np.float64)
            )
            scales.append(structured_parameter_scale(member.params))
            has_covariance = isinstance(
                member.parameter_belief,
                LocalGaussianParameterBelief,
            )
            covariance_availability.append(has_covariance)
            if has_covariance:
                within_covariances.append(member.parameter_belief.covariance)
        if any(covariance_availability) and not all(covariance_availability):
            raise ValueError(
                "fleet members must either all provide parameter covariance or none"
            )
        values = np.asarray(vectors)
        mean = np.mean(values, axis=0)
        centered = values - mean
        between = (
            centered.T @ centered / (len(values) - 1)
            if len(values) > 1
            else np.zeros((len(names), len(names)))
        )
        within = (
            np.mean(within_covariances, axis=0)
            if within_covariances
            else np.zeros_like(between)
        )
        scale = np.median(np.asarray(scales), axis=0)
        between = _project_psd(between)
        within = _project_psd(within)
        normalized_empirical = _normalized_covariance(between + within, scale)
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_empirical)
        rank = _numerical_rank(eigenvalues)
        unresolved_basis = eigenvectors[:, : len(names) - rank]
        normalized_completion = unresolved_basis @ unresolved_basis.T
        completion = normalized_completion * scale[:, None] * scale[None, :]
        return cls(
            parameter_names=names,
            mean=mean,
            between_member_covariance=between,
            within_member_covariance=within,
            completion_covariance=completion,
            natural_scale=scale,
            member_labels=labels,
            within_member_covariance_count=len(within_covariances),
            state_schema=state_schema,
            vehicle_family=vehicle_family,
            control_contracts=tuple(contracts_by_role.values()),
            source=source,
        )

    def validate_input_spec(self, spec: TrajectorySpec) -> None:
        if spec.state_schema != self.state_schema:
            raise ValueError("parameter prior uses an incompatible state schema")
        if spec.vehicle.family != self.vehicle_family:
            raise ValueError("parameter prior uses an incompatible vehicle family")
        _validate_shared_control_contracts(
            self.control_contracts,
            _control_contracts(spec),
        )
        prior_roles = {item[0] for item in self.control_contracts}
        uncovered_roles = set(spec.control_roles) - prior_roles
        if uncovered_roles:
            raise ValueError(
                "parameter prior has no fleet evidence for target control roles: "
                + ", ".join(sorted(uncovered_roles))
            )

    def as_parameter_belief(self, *, update_count: int = 0) -> LocalGaussianParameterBelief:
        """Return the compact Gaussian consumed by runtime adaptation."""

        return LocalGaussianParameterBelief(
            parameter_names=self.parameter_names,
            covariance=self.covariance,
            source=f"parameter_prior:{self.source}",
            evidence_count=self.member_count,
            effective_sample_count=float(self.member_count),
            update_count=update_count,
        )

    def initialize_belief(self, vehicle_shell: DynamicsBelief) -> DynamicsBelief:
        """Seed a new vehicle belief while preserving its typed runtime shell.

        The shell supplies the target control/configuration contract, runtime
        envelope, residual architecture, and predictive-error model. Its point
        parameters and vehicle-local information are replaced by the family
        prior; an already fitted vehicle should use ``condition_parameter_prior``
        instead.
        """

        self.validate_input_spec(vehicle_shell.input_spec)
        if structured_parameter_names(vehicle_shell.params) != self.parameter_names:
            raise ValueError("vehicle shell and parameter prior are incompatible")
        provenance = dict(vehicle_shell.provenance)
        provenance["parameter_prior_initialization"] = {
            "prior_source": self.source,
            "prior_method": self.method,
            "prior_member_count": self.member_count,
            "prior_empirical_rank": self.empirical_rank,
            "prior_completion_fraction_in_natural_coordinates": (
                self.completion_fraction_in_natural_coordinates
            ),
            "prior_artifact": self.to_dict(),
        }
        return DynamicsBelief(
            params=with_structured_parameter_vector(
                vehicle_shell.params,
                self.mean,
            ),
            input_spec=vehicle_shell.input_spec,
            runtime_spec=vehicle_shell.runtime_spec,
            predictive_error=vehicle_shell.predictive_error,
            parameter_belief=self.as_parameter_belief(),
            parameter_evidence=UnavailableParameterEvidence(
                "family-prior initialization has no vehicle-local parameter evidence"
            ),
            predictive_error_parameter_update_count=0,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PARAMETER_PRIOR_FORMAT_VERSION,
            "artifact_type": PARAMETER_PRIOR_ARTIFACT_TYPE,
            "semantics": {
                "proper_prior": True,
                "posterior": False,
                "calibrated_distribution": False,
                "empirical_and_assumed_uncertainty_separated": True,
            },
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_names": list(self.parameter_names),
            "mean": self.mean.tolist(),
            "between_member_covariance": self.between_member_covariance.tolist(),
            "within_member_covariance": self.within_member_covariance.tolist(),
            "completion_covariance": self.completion_covariance.tolist(),
            "total_covariance": self.covariance.tolist(),
            "natural_scale": self.natural_scale.tolist(),
            "member_labels": list(self.member_labels),
            "member_count": self.member_count,
            "within_member_covariance_count": self.within_member_covariance_count,
            "empirical_rank": self.empirical_rank,
            "completion_fraction_in_natural_coordinates": (
                self.completion_fraction_in_natural_coordinates
            ),
            "completion_policy": {
                "kind": "unit_natural_coordinate_numerical_nullspace",
                "completed_dimension": len(self.parameter_names) - self.empirical_rank,
            },
            "state_schema": self.state_schema,
            "vehicle_family": self.vehicle_family,
            "control_contracts": [
                {
                    "role": role,
                    "semantic": semantic,
                    "unit": unit,
                    "frame": frame,
                }
                for role, semantic, unit, frame in self.control_contracts
            ],
            "source": self.source,
            "method": self.method,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StructuredParameterPrior:
        if payload.get("format_version") != PARAMETER_PRIOR_FORMAT_VERSION:
            raise ValueError("unsupported parameter-prior format")
        if payload.get("artifact_type") != PARAMETER_PRIOR_ARTIFACT_TYPE:
            raise ValueError("artifact is not a Glassbox parameter prior")
        if payload.get("coordinate_system") != (
            "unconstrained_structured_parameter_vector"
        ):
            raise ValueError("unsupported parameter-prior coordinate system")
        semantics = payload.get("semantics")
        if not isinstance(semantics, Mapping) or any(
            semantics.get(name) is not expected
            for name, expected in {
                "proper_prior": True,
                "posterior": False,
                "calibrated_distribution": False,
                "empirical_and_assumed_uncertainty_separated": True,
            }.items()
        ):
            raise ValueError("parameter-prior semantics are incompatible")
        prior = cls(
            parameter_names=tuple(payload["parameter_names"]),
            mean=np.asarray(payload["mean"]),
            between_member_covariance=np.asarray(
                payload["between_member_covariance"]
            ),
            within_member_covariance=np.asarray(payload["within_member_covariance"]),
            completion_covariance=np.asarray(payload["completion_covariance"]),
            natural_scale=np.asarray(payload["natural_scale"]),
            member_labels=tuple(payload["member_labels"]),
            within_member_covariance_count=int(
                payload["within_member_covariance_count"]
            ),
            state_schema=str(payload["state_schema"]),
            vehicle_family=str(payload["vehicle_family"]),
            control_contracts=tuple(
                (
                    str(item["role"]),
                    str(item["semantic"]),
                    str(item["unit"]),
                    None if item.get("frame") is None else str(item["frame"]),
                )
                for item in payload["control_contracts"]
            ),
            source=str(payload["source"]),
            method=str(payload["method"]),
        )
        if int(payload.get("member_count", prior.member_count)) != prior.member_count:
            raise ValueError("parameter-prior member count is inconsistent")
        if int(payload.get("empirical_rank", prior.empirical_rank)) != (
            prior.empirical_rank
        ):
            raise ValueError("parameter-prior empirical rank is inconsistent")
        recorded_fraction = float(
            payload.get(
                "completion_fraction_in_natural_coordinates",
                prior.completion_fraction_in_natural_coordinates,
            )
        )
        if not np.isclose(
            recorded_fraction,
            prior.completion_fraction_in_natural_coordinates,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("parameter-prior completion fraction is inconsistent")
        completion_policy = payload.get("completion_policy")
        if not isinstance(completion_policy, Mapping) or completion_policy != {
            "kind": "unit_natural_coordinate_numerical_nullspace",
            "completed_dimension": len(prior.parameter_names) - prior.empirical_rank,
        }:
            raise ValueError("parameter-prior completion policy is incompatible")
        recorded_total = payload.get("total_covariance")
        if recorded_total is not None and not np.allclose(
            np.asarray(recorded_total),
            prior.covariance,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("parameter-prior total covariance is inconsistent")
        return prior

    @classmethod
    def load(cls, path: str | Path) -> StructuredParameterPrior:
        return cls.from_dict(json.loads(Path(path).read_text()))
