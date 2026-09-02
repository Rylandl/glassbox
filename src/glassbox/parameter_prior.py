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
from glassbox.dynamics import model_family
from glassbox.parameter_evidence import structured_parameter_scale

PARAMETER_PRIOR_FORMAT_VERSION = 2
PARAMETER_PRIOR_ARTIFACT_TYPE = "glassbox_structured_parameter_prior"
PARAMETER_PRIOR_METHOD = (
    "configuration_aware_block_equal_member_natural_nullspace_completion_v2"
)

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
    parameter_control_dependencies: tuple[str | None, ...]
    parameter_member_counts: tuple[int, ...]
    member_labels: tuple[str, ...]
    member_control_roles: tuple[tuple[str, ...], ...]
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
        dependencies = tuple(
            None if value is None else str(value)
            for value in self.parameter_control_dependencies
        )
        if len(dependencies) != size:
            raise ValueError("parameter control dependencies must match names")
        member_roles = tuple(
            tuple(str(role) for role in values)
            for values in self.member_control_roles
        )
        if len(member_roles) != len(labels) or any(
            len(set(values)) != len(values) or set(values) - set(roles)
            for values in member_roles
        ):
            raise ValueError("member control-role evidence is invalid")
        member_counts = tuple(int(value) for value in self.parameter_member_counts)
        if len(member_counts) != size or any(
            value < 0 or value > len(labels) for value in member_counts
        ):
            raise ValueError("parameter member counts are invalid")
        expected_counts = tuple(
            sum(
                dependency is None or dependency in roles_for_member
                for roles_for_member in member_roles
            )
            for dependency in dependencies
        )
        if member_counts != expected_counts:
            raise ValueError(
                "parameter member counts do not match configuration evidence"
            )
        if any(
            dependency is not None
            and dependency not in roles
            and member_counts[index] != 0
            for index, dependency in enumerate(dependencies)
        ):
            raise ValueError(
                "parameter dependency claims evidence for an absent control"
            )
        between = matrices["between_member_covariance"]
        for dependency in dict.fromkeys(dependencies):
            indices = np.asarray(
                [
                    index
                    for index, value in enumerate(dependencies)
                    if value == dependency
                ],
                dtype=np.int64,
            )
            eligible_count = member_counts[int(indices[0])]
            block = between[np.ix_(indices, indices)]
            normalized_block = _normalized_covariance(block, scale[indices])
            maximum_rank = max(eligible_count - 1, 0)
            if _numerical_rank(np.linalg.eigvalsh(normalized_block)) > maximum_rank:
                raise ValueError(
                    "between-member covariance exceeds configuration evidence rank"
                )
            other = np.setdiff1d(np.arange(size), indices)
            if len(other) and any(
                not np.allclose(
                    matrices[matrix_name][np.ix_(indices, other)],
                    0.0,
                    atol=1e-12,
                )
                for matrix_name in (
                    "between_member_covariance",
                    "within_member_covariance",
                )
            ):
                raise ValueError(
                    "configuration-specific parameter blocks must be independent"
                )
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
        object.__setattr__(self, "parameter_control_dependencies", dependencies)
        object.__setattr__(self, "parameter_member_counts", member_counts)
        object.__setattr__(self, "member_labels", labels)
        object.__setattr__(self, "member_control_roles", member_roles)
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
        family = model_family(reference.params)
        dependencies = tuple(
            family.parameter_control_dependency(name) for name in names
        )
        contracts_by_role: dict[str, ControlContract] = {
            item[0]: item for item in _control_contracts(reference.input_spec)
        }
        vectors: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        member_covariances: list[np.ndarray] = []
        member_control_roles: list[tuple[str, ...]] = []
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
            member_control_roles.append(tuple(member.input_spec.control_roles))
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
                member_covariances.append(member.parameter_belief.covariance)
        if any(covariance_availability) and not all(covariance_availability):
            raise ValueError(
                "fleet members must either all provide parameter covariance or none"
            )
        values = np.asarray(vectors)
        scales_array = np.asarray(scales)
        size = len(names)
        mean = values[0].copy()
        scale = scales_array[0].copy()
        between = np.zeros((size, size), dtype=np.float64)
        within = np.zeros_like(between)
        member_counts = np.zeros(size, dtype=np.int64)
        for dependency in dict.fromkeys(dependencies):
            indices = np.asarray(
                [
                    index
                    for index, value in enumerate(dependencies)
                    if value == dependency
                ],
                dtype=np.int64,
            )
            eligible = np.asarray(
                [
                    dependency is None or dependency in roles
                    for roles in member_control_roles
                ],
                dtype=bool,
            )
            count = int(np.count_nonzero(eligible))
            member_counts[indices] = count
            if count == 0:
                continue
            block_values = values[np.ix_(eligible, indices)]
            block_mean = np.mean(block_values, axis=0)
            mean[indices] = block_mean
            scale[indices] = np.median(
                scales_array[np.ix_(eligible, indices)],
                axis=0,
            )
            centered = block_values - block_mean
            block_between = (
                centered.T @ centered / (count - 1)
                if count > 1
                else np.zeros((len(indices), len(indices)))
            )
            between[np.ix_(indices, indices)] = _project_psd(block_between)
            if member_covariances:
                covariance_values = np.asarray(member_covariances)
                block_within = np.mean(
                    covariance_values[np.ix_(eligible, indices, indices)],
                    axis=0,
                )
                within[np.ix_(indices, indices)] = _project_psd(block_within)
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
            parameter_control_dependencies=dependencies,
            parameter_member_counts=tuple(member_counts),
            member_labels=labels,
            member_control_roles=tuple(member_control_roles),
            within_member_covariance_count=len(member_covariances),
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

    def initialize_belief(
        self,
        vehicle_shell: DynamicsBelief,
        *,
        predictive_error_valid_at_prior_mean: bool = False,
    ) -> DynamicsBelief:
        """Seed a new vehicle belief while preserving its typed runtime shell.

        The shell supplies the target control/configuration contract, runtime
        envelope, residual architecture, and predictive-error model. Its point
        parameters and vehicle-local information are replaced by the family
        prior; an already fitted vehicle should use ``condition_parameter_prior``
        instead.

        Moving the parameters to the prior mean stales predictive-error
        evidence that was measured around other parameters, exactly as a commit
        or ``condition_parameter_prior`` does. The returned belief therefore
        needs ``recalibrate_predictive_error`` before plan assessment, an
        online update, or an NMPC horizon cap, unless the caller asserts with
        ``predictive_error_valid_at_prior_mean=True`` that the shell's error
        model is family-level evidence that already holds at the prior mean.
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
            "predictive_error_valid_at_prior_mean": (
                predictive_error_valid_at_prior_mean
            ),
            "predictive_error_marked_stale": (
                not predictive_error_valid_at_prior_mean
            ),
        }
        return DynamicsBelief(
            params=with_structured_parameter_vector(
                vehicle_shell.params,
                self.mean,
            ),
            input_spec=vehicle_shell.input_spec,
            runtime_spec=vehicle_shell.runtime_spec,
            predictive_error=vehicle_shell.predictive_error,
            # Moving the shell to the prior mean is one parameter update, the
            # same accounting ``condition_parameter_prior`` uses, so the error
            # evidence can be marked current or stale relative to that move.
            parameter_belief=self.as_parameter_belief(update_count=1),
            parameter_evidence=UnavailableParameterEvidence(
                "family-prior initialization has no vehicle-local parameter evidence"
            ),
            predictive_error_parameter_update_count=(
                1 if predictive_error_valid_at_prior_mean else 0
            ),
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
                "configuration_specific_coordinates_use_only_applicable_members": (
                    True
                ),
                "cross_configuration_parameter_blocks_independent": True,
            },
            "coordinate_system": "unconstrained_structured_parameter_vector",
            "parameter_names": list(self.parameter_names),
            "mean": self.mean.tolist(),
            "between_member_covariance": self.between_member_covariance.tolist(),
            "within_member_covariance": self.within_member_covariance.tolist(),
            "completion_covariance": self.completion_covariance.tolist(),
            "total_covariance": self.covariance.tolist(),
            "natural_scale": self.natural_scale.tolist(),
            "parameter_control_dependencies": list(
                self.parameter_control_dependencies
            ),
            "parameter_member_counts": list(self.parameter_member_counts),
            "member_labels": list(self.member_labels),
            "member_count": self.member_count,
            "member_control_roles": [
                list(roles) for roles in self.member_control_roles
            ],
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
                "configuration_specific_coordinates_use_only_applicable_members": (
                    True
                ),
                "cross_configuration_parameter_blocks_independent": True,
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
            parameter_control_dependencies=tuple(
                None if value is None else str(value)
                for value in payload["parameter_control_dependencies"]
            ),
            parameter_member_counts=tuple(payload["parameter_member_counts"]),
            member_labels=tuple(payload["member_labels"]),
            member_control_roles=tuple(
                tuple(str(role) for role in roles)
                for roles in payload["member_control_roles"]
            ),
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
