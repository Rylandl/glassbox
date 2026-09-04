"""Fit an effective differentiable dynamics model from a trajectory artifact."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import (
    EmpiricalErrorSample,
    ForecastErrorEnvelope,
    endpoint_error_evidence_by_horizon,
    forecast_error_from_dict,
    mean_error_by_horizon,
)
from glassbox.belief.information import (
    ParameterInformation,
    estimable_structured_parameters,
    parameter_information_from_dict,
)
from glassbox.belief.parameter_evidence import (
    MAXIMUM_PARAMETER_INFORMATION_WINDOWS,
    innovation_noise,
    parameter_information,
)
from glassbox.core.data import (
    Trajectory,
    TrajectorySpec,
    TrajectoryWindows,
    duration_to_steps,
    load_trajectory_npz,
    split_trajectory,
    trajectory_windows,
)
from glassbox.core.diagnostics import (
    aggregate_innovation_diagnostics,
    one_step_innovation_diagnostics,
)
from glassbox.core.dynamics import (
    ModelParams,
    initial_residual_parameters,
    with_response_time_constant,
)
from glassbox.core.families import (
    DynamicsModelFamily,
    family_for_platform,
)
from glassbox.core.fixedwing_synthetic import initial_fixed_wing_parameter_guess
from glassbox.core.identification import (
    MAXIMUM_TRANSITIONS_PER_HORIZON,
    MAXIMUM_WINDOWS_PER_HORIZON,
    OPTIMIZATION_POLICY_VERSION,
    WINDOW_BUDGET_POLICY,
    fit_dynamics,
    residual_initialization_statistics,
    rollout_loss_configuration,
    window_budget,
)
from glassbox.core.metrics import (
    aggregate_rollout_metrics,
    predict,
    rollout_metrics,
)
from glassbox.core.model import (
    ExecutableModel,
    ModelValidityEnvelope,
    RuntimeModelSpec,
)
from glassbox.core.model_io import parameter_dict
from glassbox.core.synthetic import initial_parameter_guess

_MODEL_CLASSES = ("structured", "structured_residual")

VOLATILE_FIT_REPORT_KEYS = frozenset({"wall_time_s"})
"""Fit-report keys whose value is wall-clock timing rather than a result.

A run's duration is worth printing and worth recording, and it is not part of
what the run produced: the same code on the same inputs writes a different
number every time. Anything that identifies a report by its content therefore
excludes these keys at every depth, and any comparison against a recorded
artifact declares them volatile.
"""


def fit_report_digest(report: Mapping[str, Any]) -> str:
    """Identify one fit report by its content, timing excluded.

    The digest is taken over the report as a canonical JSON document with
    every :data:`VOLATILE_FIT_REPORT_KEYS` entry removed, not over the file's
    bytes, so two runs of the same fit on the same inputs agree and a
    recorded provenance digest does not move with the clock.
    """

    def without_timing(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_timing(item)
                for key, item in value.items()
                if key not in VOLATILE_FIT_REPORT_KEYS
            }
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value

    payload = json.dumps(without_timing(report), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fit_on_windows(
    windows: TrajectoryWindows | tuple[TrajectoryWindows, ...],
    *,
    steps: int,
    learning_rate: float,
    fixed_motor_time_constant_s: float | None = None,
    horizon_labels: tuple[str, ...] | None = None,
    model_class: str = "structured",
    platform: str = "multirotor",
    endpoint_weight: float = 3.0,
    stability_regularization: float = 0.01,
    learn_thrust_command_offset: bool = False,
    diagonal_angular_control: bool = True,
) -> tuple[ModelParams, dict[str, Any]]:
    physics_params: ModelParams = (
        initial_fixed_wing_parameter_guess()
        if platform == "fixedwing"
        else initial_parameter_guess()
    )
    if fixed_motor_time_constant_s is not None:
        physics_params = with_response_time_constant(
            physics_params, fixed_motor_time_constant_s
        )
    diagonal_angular_control = diagonal_angular_control and platform == "multirotor"
    window_sets = windows if isinstance(windows, tuple) else (windows,)
    loss_configuration = rollout_loss_configuration(
        window_sets,
        endpoint_weight=endpoint_weight,
        stability_regularization=stability_regularization,
    )
    if model_class == "structured_residual":
        initial_params: ModelParams = initial_residual_parameters(
            physics_params,
            control_size=window_sets[0].control_size,
            exogenous_size=window_sets[0].initial_exogenous.shape[1],
            **residual_initialization_statistics(window_sets),
        )
    else:
        initial_params = physics_params
    if horizon_labels is None:
        horizon_labels = tuple(
            f"{item.controls.shape[1] * item.dt_s:g}s" for item in window_sets
        )
    if len(horizon_labels) != len(window_sets):
        raise ValueError("horizon_labels must match the supplied window sets")

    start = perf_counter()
    fit = fit_dynamics(
        window_sets,
        initial_params,
        steps=steps,
        learning_rate=learning_rate,
        fixed_motor_time_constant_s=fixed_motor_time_constant_s,
        learn_thrust_command_offset=learn_thrust_command_offset,
        diagonal_angular_control=diagonal_angular_control,
        loss_configuration=loss_configuration,
    )
    wall_time_s = perf_counter() - start
    component_losses = {}
    if (
        fit.component_initial_losses is not None
        and fit.component_final_losses is not None
    ):
        component_losses = {
            label: {
                "initial_loss": float(initial_loss),
                "final_loss": float(final_loss),
                "loss_reduction": float(initial_loss / final_loss),
            }
            for label, initial_loss, final_loss in zip(
                horizon_labels,
                fit.component_initial_losses,
                fit.component_final_losses,
            )
        }
    return fit.params, {
        "fit": {
            "initial_loss": fit.initial_loss,
            "final_loss": fit.final_loss,
            "loss_reduction": fit.initial_loss / fit.final_loss,
            "wall_time_s": wall_time_s,
            "component_losses": component_losses,
            "loss_normalizers": (
                None
                if fit.component_loss_normalizers is None
                else {
                    label: float(value)
                    for label, value in zip(
                        horizon_labels,
                        fit.component_loss_normalizers,
                    )
                }
            ),
            "rollout_loss": (
                fit.loss_configuration.to_dict()
                if fit.loss_configuration is not None
                else None
            ),
            "statistics_source": "member_training_windows",
            "optimization_data_policy": {
                "policy": fit.optimization_policy,
                "diverged": fit.diverged,
                "completed_steps": fit.completed_steps,
                "automatic_policy": OPTIMIZATION_POLICY_VERSION,
                "full_window_initial_and_final_scoring": True,
            },
        },
        "parameters": {
            "initial": parameter_dict(initial_params),
            "fitted": parameter_dict(fit.params),
        },
    }


@dataclass(frozen=True)
class EvaluationFlight:
    """One held-out flight and the control history that precedes it."""

    path: str
    trajectory: Trajectory
    control_history: np.ndarray | None = None
    source_group: str | int | None = None


def _source_groups(
    paths: Sequence[Path], trajectories: Sequence[Trajectory]
) -> list[str | int] | None:
    return _label_values(paths, trajectories, "source_group")


def _label_values(
    paths: Sequence[Path], trajectories: Sequence[Trajectory], key: str
) -> list[str | int] | None:
    """Return one label value per trajectory, or ``None`` if none carry it.

    A label that some but not all trajectories carry is an error: a split on
    it would silently mix labeled and unlabeled flights.
    """

    values = [trajectory.labels.get(key) for trajectory in trajectories]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        missing = [str(path) for path, value in zip(paths, values) if value is None]
        raise ValueError(
            f"splitting on {key} requires every trajectory to have a "
            f"{key} label; unlabeled: {', '.join(missing)}"
        )
    if any(
        not isinstance(value, (str, int))
        or (isinstance(value, str) and not value.strip())
        for value in values
    ):
        raise ValueError(f"{key} labels must be non-empty strings or integers")
    resolved = [value for value in values if isinstance(value, (str, int))]
    unique = tuple(dict.fromkeys(resolved))
    if len({str(value) for value in unique}) != len(unique):
        raise ValueError(f"{key} labels must have unique string representations")
    return resolved


@dataclass(frozen=True)
class HoldoutPlan:
    """Which flights train and which are reserved, and why.

    ``mode`` records the rule that produced the split and is copied verbatim
    into ``report["split"]["mode"]``.
    """

    mode: str
    training: tuple[Trajectory, ...]
    training_labels: tuple[str, ...]
    validation: tuple[EvaluationFlight, ...]
    training_source_groups: tuple[str | int, ...] | None = None

    @property
    def training_group_order(self) -> list[str | int]:
        """Distinct training source groups in first-appearance order."""

        if self.training_source_groups is None:
            return []
        return list(dict.fromkeys(self.training_source_groups))

    @property
    def validation_group_order(self) -> list[str | int]:
        """Distinct held-out source groups in first-appearance order."""

        return list(
            dict.fromkeys(
                flight.source_group
                for flight in self.validation
                if flight.source_group is not None
            )
        )


def _default_holdout_paths(count: int) -> list[Path]:
    return [Path(f"trajectory_{index}") for index in range(count)]


_HOLDOUT_RULES = ("label", "group", "temporal")


@dataclass(frozen=True)
class Holdout:
    """One rule for reserving evidence the fit is not allowed to see.

    There are three, and each is constructed by its own classmethod:
    :meth:`by_label` reserves every flight whose label matches,
    :meth:`by_group` reserves the final groups of a label, and
    :meth:`temporal` splits one flight chronologically.
    """

    rule: str
    key: str = "source_group"
    values: tuple[str, ...] = ()
    count: int = 1
    fraction: float = 0.70

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(str(value) for value in self.values))
        if self.rule not in _HOLDOUT_RULES:
            raise ValueError(f"holdout rule must be one of {', '.join(_HOLDOUT_RULES)}")
        if not self.key.strip():
            raise ValueError("holdout key cannot be empty")
        if self.rule == "label" and not self.values:
            raise ValueError("a label holdout needs at least one held-out value")
        if any(not value.strip() for value in self.values):
            raise ValueError("holdout values cannot be empty")
        if self.count < 1:
            raise ValueError("holdout count must be at least one")
        if not 0.0 < self.fraction < 1.0:
            raise ValueError("holdout fraction must be between zero and one")

    @classmethod
    def by_label(cls, key: str, values: Sequence[str]) -> Holdout:
        """Reserve every flight whose ``key`` label is one of ``values``."""

        return cls(rule="label", key=key, values=tuple(dict.fromkeys(values)))

    @classmethod
    def by_group(cls, count: int = 1, *, key: str = "source_group") -> Holdout:
        """Reserve the final ``count`` groups of the ``key`` label.

        When the label separates nothing, because no flight carries it or
        because they all share one value, the final ``count`` flights are
        reserved in argument order instead.
        """

        return cls(rule="group", key=key, count=count)

    @classmethod
    def temporal(cls, fraction: float = 0.70) -> Holdout:
        """Split one flight chronologically, training on the first fraction."""

        return cls(rule="temporal", fraction=fraction)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"rule": self.rule}
        if self.rule == "label":
            payload["key"] = self.key
            payload["values"] = list(self.values)
        elif self.rule == "group":
            payload["key"] = self.key
            payload["count"] = self.count
        else:
            payload["fraction"] = self.fraction
        return payload

    def plan(
        self,
        trajectories: Sequence[Trajectory],
        paths: Sequence[str | Path] | None = None,
    ) -> HoldoutPlan:
        """Decide the split without loading or fitting anything.

        ``paths`` only supplies the labels used in the report and error
        messages; it defaults to positional placeholders so the rule can be
        exercised on in-memory trajectories.
        """

        if not trajectories:
            raise ValueError("at least one trajectory is required")
        resolved_paths = (
            _default_holdout_paths(len(trajectories))
            if paths is None
            else [Path(path) for path in paths]
        )
        if len(resolved_paths) != len(trajectories):
            raise ValueError("paths must label every trajectory")
        source_groups = _source_groups(resolved_paths, trajectories)

        if self.rule == "temporal":
            if len(trajectories) != 1:
                raise ValueError(
                    "a temporal holdout splits exactly one trajectory; "
                    f"got {len(trajectories)}"
                )
            return _temporal_plan(
                trajectories[0], resolved_paths[0], self.fraction, source_groups
            )
        if len(trajectories) < 2:
            raise ValueError(
                f"a {self.rule} holdout requires multiple trajectories; "
                "use Holdout.temporal to split one flight chronologically"
            )
        if self.rule == "label":
            training_indices, validation_indices = self._label_indices(
                trajectories, resolved_paths
            )
            mode = "leave_labeled_out"
        else:
            training_indices, validation_indices, mode = self._group_indices(
                trajectories, resolved_paths
            )

        def group_of(index: int) -> str | int | None:
            return None if source_groups is None else source_groups[index]

        return HoldoutPlan(
            mode=mode,
            training=tuple(trajectories[index] for index in training_indices),
            training_labels=tuple(
                str(resolved_paths[index]) for index in training_indices
            ),
            training_source_groups=(
                None
                if source_groups is None
                else tuple(source_groups[index] for index in training_indices)
            ),
            validation=tuple(
                EvaluationFlight(
                    path=str(resolved_paths[index]),
                    trajectory=trajectories[index],
                    source_group=group_of(index),
                )
                for index in validation_indices
            ),
        )

    def _label_indices(
        self, trajectories: Sequence[Trajectory], paths: Sequence[Path]
    ) -> tuple[list[int], list[int]]:
        values = _label_values(paths, trajectories, self.key)
        if values is None:
            raise ValueError(
                f"a {self.key} holdout requires every trajectory to have a "
                f"{self.key} label"
            )
        present = {str(value) for value in values}
        missing = [value for value in self.values if value not in present]
        if missing:
            raise ValueError(
                f"held-out {self.key} values are absent: {', '.join(missing)}"
            )
        held_out = set(self.values)
        training_indices = [
            index for index, value in enumerate(values) if str(value) not in held_out
        ]
        if not training_indices:
            raise ValueError(f"a {self.key} holdout cannot reserve every trajectory")
        validation_indices = [
            index for index, value in enumerate(values) if str(value) in held_out
        ]
        return training_indices, validation_indices

    def _group_indices(
        self, trajectories: Sequence[Trajectory], paths: Sequence[Path]
    ) -> tuple[list[int], list[int], str]:
        values = _label_values(paths, trajectories, self.key)
        group_order = [] if values is None else list(dict.fromkeys(values))
        if len(group_order) < 2:
            if not 1 <= self.count < len(trajectories):
                raise ValueError(
                    "holdout count must reserve at least one but not all flights"
                )
            split = len(trajectories) - self.count
            return (
                list(range(split)),
                list(range(split, len(trajectories))),
                "leave_complete_flights_out",
            )
        if not 1 <= self.count < len(group_order):
            raise ValueError(
                "holdout count must reserve at least one but not all source groups"
            )
        held_out = set(group_order[-self.count :])
        assert values is not None
        return (
            [index for index, value in enumerate(values) if value not in held_out],
            [index for index, value in enumerate(values) if value in held_out],
            "leave_source_groups_out",
        )


DEFAULT_HOLDOUT = Holdout.by_group()


def _temporal_plan(
    trajectory: Trajectory,
    path: Path,
    fraction: float,
    source_groups: list[str | int] | None,
) -> HoldoutPlan:
    training_segment, validation_segment = split_trajectory(
        trajectory, train_fraction=fraction
    )
    return HoldoutPlan(
        mode="temporal_within_flight",
        training=(training_segment,),
        training_labels=(f"{path}#training",),
        training_source_groups=None,
        validation=(
            EvaluationFlight(
                path=f"{path}#validation",
                trajectory=validation_segment,
                control_history=training_segment.controls,
                source_group=(source_groups[0] if source_groups else None),
            ),
        ),
    )


@dataclass(frozen=True)
class LossPolicy:
    """The geometry of the rollout objective, independent of the data."""

    endpoint_weight: float = 3.0
    stability_regularization: float = 0.01
    learn_thrust_command_offset: bool = False
    diagonal_angular_control: bool = True

    def __post_init__(self) -> None:
        if self.endpoint_weight < 1.0:
            raise ValueError("endpoint_weight must be at least one")
        if self.stability_regularization < 0.0:
            raise ValueError("stability_regularization must be nonnegative")


@dataclass(frozen=True)
class WeightingPolicy:
    """How training windows share the total loss weight across flights.

    When ``balanced`` is set, each group named by the trajectories'
    ``source_group`` label carries equal total weight and windows inside a
    group are weighted uniformly; without the label, each flight is its own
    group, and when every flight declares a ``profile`` instead, each maneuver
    family carries equal weight before its replicates split it. Clearing it
    weights every window equally, so a long flight contributes in proportion to
    its duration. ``group_weights`` declares each source group's share
    explicitly and represents complete-group resampling multiplicities.
    """

    balanced: bool = True
    group_weights: Mapping[str | int, float] | None = None

    def __post_init__(self) -> None:
        if self.group_weights is None:
            return
        weights = np.asarray(list(self.group_weights.values()), dtype=np.float64)
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError(
                "group_weights values must be finite and nonnegative with at "
                "least one positive group"
            )


ABLATIONS = ("no_lag",)

_ABLATION_PROVENANCE = {"no_lag": "fixed near-zero applied-control response"}


@dataclass(frozen=True)
class FitSpec:
    """Every knob that shapes a multi-flight fit, validated on construction."""

    holdout: Holdout = DEFAULT_HOLDOUT
    horizons_s: tuple[float, ...] | None = None
    horizon_steps: int = 25
    stride: int | None = None
    steps: int = 400
    learning_rate: float = 0.02
    evaluation_horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)
    model_class: str = "structured"
    ablations: tuple[str, ...] = ()
    diagnostics: bool = False
    parameter_evidence: bool = True
    fixed_response_time_constant_s: float | None = None
    loss: LossPolicy = LossPolicy()
    weighting: WeightingPolicy = WeightingPolicy()

    def __post_init__(self) -> None:
        for name in ("evaluation_horizons_s", "horizons_s"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        object.__setattr__(self, "ablations", tuple(dict.fromkeys(self.ablations)))

        if not isinstance(self.holdout, Holdout):
            raise TypeError("holdout must be a Holdout")
        if any(seconds <= 0.0 for seconds in self.evaluation_horizons_s):
            raise ValueError("evaluation horizons must be positive")
        if self.horizons_s is not None and any(
            seconds <= 0.0 for seconds in self.horizons_s
        ):
            raise ValueError("training horizons must be positive")
        if self.model_class not in _MODEL_CLASSES:
            raise ValueError("model_class must be structured or structured_residual")
        unknown = [name for name in self.ablations if name not in ABLATIONS]
        if unknown:
            raise ValueError(f"unknown ablation(s): {', '.join(unknown)}")
        if self.fixed_response_time_constant_s is not None:
            if self.fixed_response_time_constant_s <= 0.0:
                raise ValueError("fixed_response_time_constant_s must be positive")
            if "no_lag" in self.ablations:
                raise ValueError(
                    "fixed_response_time_constant_s already pins the applied-control "
                    "response time, so the no-lag ablation does not apply"
                )

    def stride_for(self, horizon_steps: int) -> int:
        """Return the window stride used at one training horizon."""

        return horizon_steps if self.stride is None else self.stride


def _trajectory_summary(path: str, trajectory: Trajectory) -> dict[str, Any]:
    position = trajectory.states[:, 0:3]
    velocity = trajectory.states[:, 3:6]
    return {
        "path": path,
        "duration_s": float(trajectory.time_s[-1]),
        "intervals": len(trajectory.controls),
        "control_size": trajectory.control_size,
        "control_names": list(trajectory.control_names),
        "sample_rate_hz": 1.0 / trajectory.nominal_dt_s,
        "characteristics": {
            "path_length_m": float(
                np.sum(np.linalg.norm(np.diff(position, axis=0), axis=1))
            ),
            "maximum_speed_m_s": float(np.max(np.linalg.norm(velocity, axis=1))),
        },
        "spec": trajectory.spec.to_dict(),
        "labels": dict(trajectory.labels),
        "provenance": dict(trajectory.provenance),
    }


def _dataset_contract(
    paths: list[Path], trajectories: list[Trajectory]
) -> dict[str, Any]:
    """Validate canonical semantics, independent of source adapter details."""

    def consistent_value(
        name: str, values: list[Any], *, serialize: bool = False
    ) -> Any:
        comparable = [
            json.dumps(value, sort_keys=True) if serialize else value
            for value in values
        ]
        unique = list(dict.fromkeys(comparable))
        if len(unique) > 1:
            details = ", ".join(
                f"{path}={value!r}" for path, value in zip(paths, values)
            )
            raise ValueError(f"inconsistent dataset {name}: {details}")
        return values[0]

    sample_rates = [1.0 / trajectory.nominal_dt_s for trajectory in trajectories]
    reference_rate = sample_rates[0]
    if not np.allclose(sample_rates, reference_rate, atol=1e-6, rtol=0.0):
        details = ", ".join(
            f"{path}={rate:g}Hz" for path, rate in zip(paths, sample_rates)
        )
        raise ValueError(f"inconsistent dataset sample_rate_hz: {details}")

    spec_payloads = [trajectory.spec.to_dict() for trajectory in trajectories]
    spec_payload = consistent_value("trajectory_spec", spec_payloads, serialize=True)
    reference_spec = trajectories[0].spec

    source_types = []
    for trajectory in trajectories:
        adapter = trajectory.provenance.get("adapter", {})
        adapter_name = adapter.get("name") if isinstance(adapter, Mapping) else None
        source_types.append(str(adapter_name or "unknown"))
    source_type_counts = {
        source_type: source_types.count(source_type)
        for source_type in dict.fromkeys(source_types)
    }
    platform = str(spec_payload["vehicle"]["family"])

    if platform == "fixedwing":
        for path, trajectory, source_type in zip(paths, trajectories, source_types):
            if source_type != "px4_ulog":
                continue
            px4 = trajectory.provenance.get("px4", {})
            mapping = (
                px4.get("actuator_mapping", {}) if isinstance(px4, Mapping) else {}
            )
            mapping_verified = (
                mapping.get("actuator_mapping_verified")
                if isinstance(mapping, Mapping)
                else None
            )
            if mapping_verified is not True:
                raise ValueError(
                    f"fixed-wing PX4 trajectory {path} lacks a verified "
                    "canonical actuator mapping"
                )

    contract = {
        "pooling_basis": "canonical_trajectory_spec",
        "flight_count": len(trajectories),
        "sample_rate_hz": reference_rate,
        "control_size": len(reference_spec.controls),
        "control_names": list(reference_spec.control_names),
        "control_roles": list(reference_spec.control_roles),
        "control_semantics": list(reference_spec.control_semantics),
        "exogenous_size": len(reference_spec.exogenous),
        "exogenous_names": list(reference_spec.exogenous_names),
        "exogenous_roles": list(reference_spec.exogenous_roles),
        "platform": platform,
        "source_type_counts": source_type_counts,
        "state_source": spec_payload["observation_source"],
        "state_schema": spec_payload["state_schema"],
        "trajectory_spec": spec_payload,
    }
    return contract


@dataclass(frozen=True)
class DatasetResolution:
    """The validated dataset contract plus the family it selects."""

    contract: dict[str, Any]
    platform: str
    family: DynamicsModelFamily
    source_groups: list[str | int] | None


def resolve_dataset(
    paths: list[Path], trajectories: list[Trajectory], spec: FitSpec
) -> DatasetResolution:
    """Validate the pooled dataset and resolve the model family it supports."""

    source_groups = _source_groups(paths, trajectories)
    contract = _dataset_contract(paths, trajectories)
    contract["source_group_count"] = (
        len(dict.fromkeys(source_groups))
        if source_groups is not None
        else len(trajectories)
    )
    platform = str(contract["platform"])
    family = family_for_platform(platform)
    family.validate_control_schema(
        trajectories[0].control_names,
        trajectories[0].spec.control_roles,
    )
    contract["model_family"] = family.key
    if spec.model_class == "structured_residual" and not family.supports_residual:
        raise ValueError(
            f"structured_residual is not supported for platform {platform!r}"
        )
    return DatasetResolution(
        contract=contract,
        platform=platform,
        family=family,
        source_groups=source_groups,
    )


def _evaluate_model(
    params: ModelParams,
    flights: Sequence[EvaluationFlight],
    *,
    horizon_seconds: tuple[float, ...],
    diagnostics: bool,
) -> dict[str, Any]:
    per_flight: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []
    innovation_diagnostics: list[dict[str, Any]] = []
    horizon_metrics: dict[str, list[dict[str, Any]]] = {
        f"{seconds:g}s": [] for seconds in horizon_seconds
    }
    error_samples: dict[float, list[EmpiricalErrorSample]] = {
        seconds: [] for seconds in horizon_seconds
    }

    for flight in flights:
        trajectory = flight.trajectory
        full = rollout_metrics(
            predict(params, trajectory, control_history=flight.control_history)
        )
        full_metrics.append(full)
        if diagnostics:
            innovation_diagnostics.append(
                one_step_innovation_diagnostics(
                    params,
                    trajectory,
                    control_history=flight.control_history,
                )
            )
        per_horizon: dict[str, Any] = {}
        for evidence in endpoint_error_evidence_by_horizon(
            params,
            trajectory,
            horizons_s=horizon_seconds,
            source_group=str(
                flight.source_group if flight.source_group is not None else flight.path
            ),
            trajectory_id=flight.path,
        ):
            label = f"{evidence.horizon_s:g}s"
            per_horizon[label] = evidence.window_metrics
            horizon_metrics[label].append(evidence.window_metrics)
            error_samples[evidence.horizon_s].append(evidence.sample)

        flight_report: dict[str, Any] = {
            "path": flight.path,
            "duration_s": float(trajectory.time_s[-1]),
            "full_rollout": full,
            "horizon_rollouts": per_horizon,
        }
        if diagnostics:
            flight_report["one_step_innovation"] = innovation_diagnostics[-1]
        per_flight.append(flight_report)

    available_error_samples = {
        horizon: samples for horizon, samples in error_samples.items() if samples
    }
    forecast_error = (
        ForecastErrorEnvelope.from_samples(available_error_samples)
        if available_error_samples
        else None
    )

    aggregate: dict[str, Any] = {
        "flight_count": len(flights),
        "weighting": "equal_flight",
        "full_rollout": aggregate_rollout_metrics(full_metrics, weighting="equal"),
        "horizon_rollouts": {
            label: aggregate_rollout_metrics(items, weighting="equal")
            for label, items in horizon_metrics.items()
            if items
        },
        # The systematic part of the held-out error, recorded and not applied.
        # Subtracting a mean measured on other flights would move the model
        # without moving the account of what is known about it.
        "held_out_mean_tangent_error": mean_error_by_horizon(available_error_samples),
    }
    if diagnostics:
        aggregate["one_step_innovation"] = aggregate_innovation_diagnostics(
            innovation_diagnostics
        )
    return {
        "aggregate": aggregate,
        "per_flight": per_flight,
        "forecast_error": (
            None if forecast_error is None else forecast_error.to_dict()
        ),
        "innovation_noise": innovation_noise(
            params,
            [(flight.trajectory, flight.control_history) for flight in flights],
        ).tolist(),
    }


def _comparison_report(
    learned: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    metric_names = (
        "position_rmse_m",
        "velocity_rmse_m_s",
        "attitude_rmse_deg",
        "angular_velocity_rmse_rad_s",
        "final_position_error_m",
    )

    def ratios(
        learned_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for name in metric_names:
            learned_value = float(learned_metrics[name])
            result[name] = (
                float(baseline_metrics[name]) / learned_value
                if learned_value > 0.0
                else None
            )
        return result

    learned_aggregate = learned["aggregate"]
    baseline_aggregate = baseline["aggregate"]
    horizon_ratios = {}
    for label, learned_metrics in learned_aggregate["horizon_rollouts"].items():
        if label in baseline_aggregate["horizon_rollouts"]:
            horizon_ratios[label] = ratios(
                learned_metrics,
                baseline_aggregate["horizon_rollouts"][label],
            )
    return {
        "ratio_definition": (
            "no_lag_rmse / learned_lag_rmse; values above one favor learned lag"
        ),
        "aggregate_full_rollout": ratios(
            learned_aggregate["full_rollout"],
            baseline_aggregate["full_rollout"],
        ),
        "aggregate_horizon_rollouts": horizon_ratios,
    }


@dataclass(frozen=True)
class TrainingWindows:
    """Rollout windows for every training horizon, plus how they were weighted."""

    dt_s: float
    horizon_steps: tuple[int, ...]
    horizon_labels: tuple[str, ...]
    maximum_windows_by_horizon: tuple[int, ...]
    diversity_count: int
    stratification: str
    window_sets: tuple[TrajectoryWindows, ...]

    @property
    def fitting_windows(self) -> TrajectoryWindows | tuple[TrajectoryWindows, ...]:
        return self.window_sets[0] if len(self.window_sets) == 1 else self.window_sets


def _training_weights(
    plan: HoldoutPlan, spec: FitSpec
) -> tuple[
    Mapping[str, float] | None,
    Callable[[int, Trajectory], str] | None,
    str,
]:
    """Resolve the one weighting the training windows are extracted under.

    Returns the group weights, the key function that assigns each training
    flight to a group, and the name of the resulting stratification.
    """

    training = plan.training
    source_groups = plan.training_source_groups
    declared = spec.weighting.group_weights
    if declared is not None:
        if source_groups is None:
            raise ValueError("weighting group_weights requires source_group labels")
        if set(declared) != set(plan.training_group_order):
            raise ValueError(
                "weighting group_weights must contain exactly the training "
                "source groups"
            )
        return (
            {str(group): float(weight) for group, weight in declared.items()},
            None,
            "weighted_source_group",
        )
    if not spec.weighting.balanced or len(training) < 2:
        return None, None, "global_timeline"
    if source_groups is not None:
        return (
            {str(group): 1.0 for group in plan.training_group_order},
            None,
            "source_group",
        )
    profiles = [trajectory.labels.get("profile") for trajectory in training]
    if all(profile is not None for profile in profiles):
        counts = {
            profile: profiles.count(profile) for profile in dict.fromkeys(profiles)
        }
        return (
            {
                str(index): 1.0 / counts[profile]
                for index, profile in enumerate(profiles)
            },
            lambda index, _trajectory: str(index),
            "weighted_trajectory",
        )
    return (
        {str(index): 1.0 for index in range(len(training))},
        lambda index, _trajectory: str(index),
        "trajectory",
    )


def build_training_windows(plan: HoldoutPlan, spec: FitSpec) -> TrainingWindows:
    """Extract and weight the rollout windows the optimizer will consume."""

    training = list(plan.training)
    dt_s = training[0].nominal_dt_s
    if spec.horizons_s is None:
        horizon_steps = (spec.horizon_steps,)
    else:
        horizon_steps = tuple(
            dict.fromkeys(
                duration_to_steps(seconds, dt_s) for seconds in spec.horizons_s
            )
        )
    horizon_labels = tuple(f"{steps * dt_s:g}s" for steps in horizon_steps)

    weights, group_of, stratification = _training_weights(plan, spec)
    diversity_count = (
        len(plan.training_group_order)
        if plan.training_source_groups is not None
        else len(training)
    )
    maximum_windows_by_horizon = tuple(
        window_budget(steps, minimum=diversity_count) for steps in horizon_steps
    )
    window_sets = tuple(
        trajectory_windows(
            training,
            horizon=steps,
            stride=spec.stride_for(steps),
            weights=weights,
            group_of=group_of,
            maximum_windows=maximum_windows,
        )
        for steps, maximum_windows in zip(horizon_steps, maximum_windows_by_horizon)
    )
    return TrainingWindows(
        dt_s=dt_s,
        horizon_steps=horizon_steps,
        horizon_labels=horizon_labels,
        maximum_windows_by_horizon=maximum_windows_by_horizon,
        diversity_count=diversity_count,
        stratification=stratification,
        window_sets=window_sets,
    )


def evidence_independence_unit(plan: HoldoutPlan) -> str:
    """Name the unit each parameter-evidence group represents."""

    if plan.training_source_groups is not None:
        return "source_group"
    if plan.mode == "temporal_within_flight":
        return "temporal_training_segment"
    return "complete_trajectory"


def _by_horizon(windows: TrainingWindows, render: Any) -> dict[str, Any]:
    """Map every training-horizon label to ``render(window_set)``."""

    return {
        label: render(window_set)
        for label, window_set in zip(windows.horizon_labels, windows.window_sets)
    }


def _split_section(plan: HoldoutPlan, spec: FitSpec) -> dict[str, Any]:
    return {
        "mode": plan.mode,
        "independent_source_group_holdout": bool(plan.training_source_groups)
        and set(plan.training_source_groups).isdisjoint(plan.validation_group_order),
        "holdout": spec.holdout.to_dict(),
        "training_source_groups": plan.training_group_order,
        "validation_source_groups": plan.validation_group_order,
        "training_flights": [
            _trajectory_summary(label, trajectory)
            for label, trajectory in zip(plan.training_labels, plan.training)
        ],
        "validation_flights": [
            _trajectory_summary(flight.path, flight.trajectory)
            for flight in plan.validation
        ],
    }


def _training_windows_per_flight(
    plan: HoldoutPlan, windows: TrainingWindows
) -> dict[str, Any]:
    """Each training flight's share of the extracted windows, by horizon."""

    labels = plan.training_labels
    return _by_horizon(
        windows,
        lambda window_set: {
            labels[index]: int(np.sum(window_set.trajectory_indices == index))
            for index in range(len(labels))
        },
    )


def _window_selection_section(windows: TrainingWindows) -> dict[str, Any]:
    return {
        "budget_policy": WINDOW_BUDGET_POLICY,
        "maximum_windows_per_horizon": MAXIMUM_WINDOWS_PER_HORIZON,
        "maximum_transitions_per_horizon": MAXIMUM_TRANSITIONS_PER_HORIZON,
        "maximum_windows_by_horizon": dict(
            zip(windows.horizon_labels, windows.maximum_windows_by_horizon)
        ),
        "selected_windows_by_horizon": _by_horizon(
            windows, lambda window_set: len(window_set.initial_states)
        ),
        "source_group_count": windows.diversity_count,
        "stratification": windows.stratification,
    }


_TRAINING_FLIGHT_WEIGHTING = {
    "weighted_source_group": "weighted_source_group_then_equal_window",
    "source_group": "equal_source_group_then_equal_window",
    "weighted_trajectory": "equal_profile_then_equal_flight",
    "trajectory": "equal_flight",
    "global_timeline": "window_count",
}


def _configuration_section(
    *,
    spec: FitSpec,
    dataset: DatasetResolution,
    plan: HoldoutPlan,
    windows: TrainingWindows,
    independence_unit: str,
) -> dict[str, Any]:
    dt_s = windows.dt_s
    platform = dataset.platform
    return {
        "holdout_count": len(plan.validation),
        "holdout_source_group_count": len(
            {
                flight.source_group
                for flight in plan.validation
                if flight.source_group is not None
            }
        ),
        "horizon_steps": max(windows.horizon_steps),
        "training_horizon_steps": list(windows.horizon_steps),
        "training_horizons_s": [steps * dt_s for steps in windows.horizon_steps],
        "control_history_duration_s": (
            windows.window_sets[0].control_histories.shape[1] * dt_s
        ),
        "motor_history_duration_s": (
            windows.window_sets[0].control_histories.shape[1] * dt_s
            if platform == "multirotor"
            else None
        ),
        "optimization_steps_per_model": spec.steps,
        "learning_rate": spec.learning_rate,
        "endpoint_weight": spec.loss.endpoint_weight,
        "stability_regularization": spec.loss.stability_regularization,
        "multirotor_thrust_command_offset": (
            "not_applicable_fixedwing"
            if platform != "multirotor"
            else "learned"
            if spec.loss.learn_thrust_command_offset
            else "fixed_zero_reference"
        ),
        "angular_control_coupling": (
            "not_applicable_fixedwing"
            if platform != "multirotor"
            else "diagonal_mixer_reference"
            if spec.loss.diagonal_angular_control
            else "learned_cross_coupled_mixer"
        ),
        "training_windows": sum(
            len(window_set.initial_states) for window_set in windows.window_sets
        ),
        "training_windows_by_horizon": _by_horizon(
            windows, lambda window_set: len(window_set.initial_states)
        ),
        "training_window_selection": _window_selection_section(windows),
        "evaluation_horizons_s": list(spec.evaluation_horizons_s),
        "fixed_response_time_constant_s": spec.fixed_response_time_constant_s,
        "ablations": list(spec.ablations),
        "model_class": spec.model_class,
        "platform": platform,
        "model_family": dataset.family.key,
        "training_flight_weighting": _TRAINING_FLIGHT_WEIGHTING[windows.stratification],
        "training_windows_per_flight_by_horizon": _training_windows_per_flight(
            plan, windows
        ),
        "diagnostics": spec.diagnostics,
        "parameter_evidence": {
            "requested": spec.parameter_evidence,
            "method": "training_one_step_information_v1",
            "maximum_windows": MAXIMUM_PARAMETER_INFORMATION_WINDOWS,
            "independence_unit": independence_unit,
            "noise_model": "held_out_one_step_innovation_covariance",
        },
    }


def build_fit_report(
    *,
    spec: FitSpec,
    dataset: DatasetResolution,
    plan: HoldoutPlan,
    windows: TrainingWindows,
    models: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the JSON-compatible multi-flight fit report."""

    return {
        "format_version": 1,
        "dataset": dataset.contract,
        "split": _split_section(plan, spec),
        "configuration": _configuration_section(
            spec=spec,
            dataset=dataset,
            plan=plan,
            windows=windows,
            independence_unit=evidence_independence_unit(plan),
        ),
        "models": models,
        "comparison": comparison,
    }


def _information(
    model: ExecutableModel,
    model_report: Mapping[str, Any],
    *,
    plan: HoldoutPlan,
    spec: FitSpec,
    fixed_response_time: bool,
) -> ParameterInformation:
    """Build the belief's information state from this fit's own evidence.

    The noise model is measured on the held-out flights and the precision is
    accumulated from the training transitions' own Jacobians, so the belief a
    fit returns says both how wrong its one-step predictions have been and
    which parameter directions the evidence resolved. Clearing
    ``FitSpec.parameter_evidence`` leaves the point estimate with its noise
    model alone, at rank zero; no command does that.
    """

    noise = np.asarray(model_report["validation"]["innovation_noise"])
    estimable = estimable_structured_parameters(
        model.params,
        fixed_response_time=fixed_response_time,
        learn_thrust_command_offset=spec.loss.learn_thrust_command_offset,
        diagonal_angular_control=spec.loss.diagonal_angular_control,
    )
    if not spec.parameter_evidence:
        return ParameterInformation.unknown(
            model.params,
            innovation_noise=noise,
            estimable=estimable,
            source="held_out_one_step_noise_only",
        )
    groups = (
        list(plan.training_source_groups)
        if plan.training_source_groups is not None
        else list(plan.training_labels)
    )
    return parameter_information(
        model,
        plan.training,
        groups,
        innovation_noise=noise,
        estimable=estimable,
        source=f"training_one_step_information:{evidence_independence_unit(plan)}",
    )


DEFAULT_FIT_SPEC = FitSpec()


@dataclass(frozen=True)
class FitOutcome:
    """What one fit produced: a belief, the report, and any ablation beliefs."""

    belief: DynamicsBelief
    report: dict[str, Any]
    ablations: Mapping[str, DynamicsBelief] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ablations", dict(self.ablations))


def _runtime_spec(
    dataset: DatasetResolution, model_report: Mapping[str, Any]
) -> RuntimeModelSpec:
    """Build the runtime contract the fit itself resolved.

    The sample period is the pooled dataset's declared rate and the validity
    envelope is the one the rollout objective derived from the training
    windows, so a fitted model carries its own execution contract instead of
    having one recovered from its report.
    """

    return RuntimeModelSpec(
        sample_period_s=1.0 / float(dataset.contract["sample_rate_hz"]),
        validity_envelope=ModelValidityEnvelope.from_dict(
            model_report["fit"]["rollout_loss"]["dynamic_envelope"]
        ),
    )


def _belief(
    params: ModelParams,
    model_report: Mapping[str, Any],
    *,
    dataset: DatasetResolution,
    input_spec: TrajectorySpec,
    provenance: Mapping[str, Any],
) -> DynamicsBelief:
    forecast_error = model_report["validation"]["forecast_error"]
    return DynamicsBelief(
        model=ExecutableModel(params, input_spec, _runtime_spec(dataset, model_report)),
        information=parameter_information_from_dict(model_report["parameter_evidence"]),
        forecast_error=(
            None if forecast_error is None else forecast_error_from_dict(forecast_error)
        ),
        provenance=dict(provenance),
    )


def fit(
    sources: Sequence[str | Path],
    spec: FitSpec = DEFAULT_FIT_SPEC,
) -> FitOutcome:
    """Fit one vehicle's dynamics belief from canonical trajectory files.

    This is the coordinator: it loads, validates the pooled dataset, plans the
    holdout, extracts training windows, fits the model and any requested
    ablation, scores each on the reserved evidence, and returns the belief that
    scoring supports alongside the report that records how it was produced.
    """

    if not sources:
        raise ValueError("at least one trajectory path is required")
    paths = [Path(path) for path in sources]
    trajectories = [load_trajectory_npz(path) for path in paths]
    dataset = resolve_dataset(paths, trajectories, spec)
    plan = spec.holdout.plan(trajectories, paths)
    windows = build_training_windows(plan, spec)
    input_spec = trajectories[0].spec

    def fit_model(
        *, fixed_motor_time_constant_s: float | None, fixed_response_time: bool
    ) -> tuple[ModelParams, dict[str, Any]]:
        params, model_report = _fit_on_windows(
            windows.fitting_windows,
            steps=spec.steps,
            learning_rate=spec.learning_rate,
            fixed_motor_time_constant_s=fixed_motor_time_constant_s,
            horizon_labels=windows.horizon_labels,
            model_class=spec.model_class,
            platform=dataset.platform,
            endpoint_weight=spec.loss.endpoint_weight,
            stability_regularization=spec.loss.stability_regularization,
            learn_thrust_command_offset=spec.loss.learn_thrust_command_offset,
            diagonal_angular_control=spec.loss.diagonal_angular_control,
        )
        model_report["validation"] = _evaluate_model(
            params,
            plan.validation,
            horizon_seconds=spec.evaluation_horizons_s,
            diagnostics=spec.diagnostics,
        )
        model_report["parameter_evidence"] = _information(
            ExecutableModel(params, input_spec, _runtime_spec(dataset, model_report)),
            model_report,
            plan=plan,
            spec=spec,
            fixed_response_time=fixed_response_time,
        ).to_dict()
        return params, model_report

    learned_params, learned_report = fit_model(
        fixed_motor_time_constant_s=spec.fixed_response_time_constant_s,
        fixed_response_time=spec.fixed_response_time_constant_s is not None,
    )
    models: dict[str, Any] = {"learned_lag": learned_report}

    ablation_params: dict[str, ModelParams] = {}
    comparison = None
    if "no_lag" in spec.ablations:
        baseline_params, baseline_report = fit_model(
            fixed_motor_time_constant_s=0.0001, fixed_response_time=True
        )
        models["no_lag"] = baseline_report
        ablation_params["no_lag"] = baseline_params
        comparison = _comparison_report(
            learned_report["validation"], baseline_report["validation"]
        )

    report = build_fit_report(
        spec=spec,
        dataset=dataset,
        plan=plan,
        windows=windows,
        models=models,
        comparison=comparison,
    )
    provenance = {
        "training_trajectories": list(plan.training_labels),
        "validation_trajectories": [flight.path for flight in plan.validation],
    }
    return FitOutcome(
        belief=_belief(
            learned_params,
            learned_report,
            dataset=dataset,
            input_spec=input_spec,
            provenance=provenance,
        ),
        report=report,
        ablations={
            name: _belief(
                params,
                models[name],
                dataset=dataset,
                input_spec=input_spec,
                provenance={
                    **provenance,
                    "ablation": _ABLATION_PROVENANCE[name],
                },
            )
            for name, params in ablation_params.items()
        },
    )
