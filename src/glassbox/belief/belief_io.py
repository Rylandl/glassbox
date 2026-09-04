"""Serialization for first-class Glassbox dynamics beliefs."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalHorizonPredictiveError,
    LocalParameterInformation,
    parameter_belief_from_dict,
    parameter_evidence_from_dict,
    predictive_error_from_dict,
)
from glassbox.core.data import TrajectorySpec
from glassbox.core.model import ExecutableModel, RuntimeModelSpec, default_actuation
from glassbox.core.model_io import dynamics_model_from_payload, model_payload

BELIEF_FORMAT_VERSION = 4
BELIEF_ARTIFACT_TYPE = "glassbox_dynamics_belief"

# Format 3 is every belief written before the multirotor rotational-response
# branch was deleted. Its parameter evidence and parameter belief are stated
# over three coordinates the structured parameter vector no longer has.
LEGACY_BELIEF_FORMAT_VERSION = 3
DROPPED_PARAMETER_PREFIX = "log_angular_response_time_constant["


def belief_payload(belief: DynamicsBelief) -> dict[str, Any]:
    """Return a JSON-compatible predictive dynamics artifact."""

    return {
        "format_version": BELIEF_FORMAT_VERSION,
        "artifact_type": BELIEF_ARTIFACT_TYPE,
        "semantics": {
            "predictive_model": True,
            "posterior": False,
            "state_uncertainty_included": False,
            "parameter_uncertainty_included": (
                belief.parameter_belief.uncertainty_available
            ),
            "parameter_information_included": belief.parameter_evidence.available,
            "predictive_error_included": belief.predictive_error.available,
            "predictive_error_current": belief.predictive_error_current,
            "predictive_error_covariance_scope": (
                belief.predictive_error.covariance_scope.value
                if isinstance(
                    belief.predictive_error,
                    EmpiricalHorizonPredictiveError,
                )
                else None
            ),
            "parameter_evidence_covariance_scope": (
                belief.parameter_evidence.covariance_scope.value
                if isinstance(
                    belief.parameter_evidence,
                    LocalParameterInformation,
                )
                else None
            ),
        },
        "nominal_model": model_payload(
            belief.params,
            input_spec=belief.input_spec,
            runtime_spec=belief.runtime_spec,
            provenance=belief.provenance,
        ),
        "parameter_belief": belief.parameter_belief.to_dict(),
        "parameter_evidence": belief.parameter_evidence.to_dict(),
        "predictive_error": belief.predictive_error.to_dict(),
        "predictive_error_parameter_update_count": (
            belief.predictive_error_parameter_update_count
        ),
        "provenance": dict(belief.provenance),
    }


def save_dynamics_belief(belief: DynamicsBelief, path: str | Path) -> None:
    """Write one fitted dynamics belief as readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(belief_payload(belief), indent=2, allow_nan=False) + "\n"
    )


def _without_dropped_parameters(
    payload: Mapping[str, Any],
    *,
    matrix_keys: tuple[str, ...],
    row_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Remove the deleted rotational-response coordinates from legacy evidence.

    The fitter froze those coordinates on every path that ever wrote an
    artifact, so their rows and columns are exactly zero and deleting them is
    an exact projection. A belief that did carry information there cannot be
    migrated and is refused rather than quietly weakened.
    """

    names = list(payload.get("parameter_names", ()))
    keep = [
        index
        for index, name in enumerate(names)
        if not str(name).startswith(DROPPED_PARAMETER_PREFIX)
    ]
    if len(keep) == len(names):
        return dict(payload)
    dropped = [index for index in range(len(names)) if index not in set(keep)]
    trimmed = dict(payload)
    trimmed["parameter_names"] = [names[index] for index in keep]
    for key in ("center", "parameter_scale", "fitted_parameter_mask"):
        if key in trimmed:
            values = np.asarray(trimmed[key])
            trimmed[key] = values[keep].tolist()
    for key in matrix_keys:
        if key not in trimmed:
            continue
        values = np.asarray(trimmed[key], dtype=np.float64)
        if np.any(values[dropped, :] != 0.0) or np.any(values[:, dropped] != 0.0):
            raise ValueError(
                f"{key} carries information on the deleted rotational-response "
                "coordinates, so this belief cannot be read by the memoryless "
                "multirotor model; refit it"
            )
        trimmed[key] = values[np.ix_(keep, keep)].tolist()
    for key in row_keys:
        if key not in trimmed:
            continue
        values = np.asarray(trimmed[key], dtype=np.float64)
        if np.any(values[:, dropped] != 0.0):
            raise ValueError(
                f"{key} carries information on the deleted rotational-response "
                "coordinates, so this belief cannot be read by the memoryless "
                "multirotor model; refit it"
            )
        trimmed[key] = values[:, keep].tolist()
    return trimmed


def _executable_model(
    payload: Mapping[str, Any],
) -> tuple[ExecutableModel, Mapping[str, Any]]:
    """Rebuild the belief's mean from the nominal-model half of a payload.

    The artifact records no actuation map, because the map is a property of the
    command interface rather than of the fit. It is rebuilt from the declared
    control channels, which yields the identity map for an actionable model and
    no map at all for one whose inputs are observations of actuation.
    """

    params, nominal = dynamics_model_from_payload(payload)
    input_spec = TrajectorySpec.from_dict(nominal["input_spec"])
    model = ExecutableModel(
        params,
        input_spec,
        RuntimeModelSpec.from_dict(nominal["runtime_spec"]),
        default_actuation(input_spec),
    )
    return model, nominal


def _point_belief_from_model_payload(payload: Mapping[str, Any]) -> DynamicsBelief:
    """Wrap a bare nominal-model payload as a belief carrying no evidence."""

    model, nominal = _executable_model(payload)
    return DynamicsBelief(
        model=model,
        provenance=dict(nominal.get("provenance", {})),
    )


def dynamics_belief_from_payload(payload: Mapping[str, Any]) -> DynamicsBelief:
    """Restore a dynamics belief from an already decoded payload.

    A payload that is a bare nominal model rather than a belief is accepted and
    wrapped as a point belief with no predictive-error and no parameter
    evidence. This tolerance is temporary: model-only artifacts under the
    untracked ``artifacts/`` tree predate the single belief format, and it is
    removed once those artifacts are re-recorded.

    A payload written under format 3 is accepted with a warning. Its nominal
    model carries the deleted ``angular_response_time_constant``, which the
    model decoder drops, and its parameter evidence is stated over three
    coordinates the structured parameter vector no longer has, which are
    projected out here. Both tolerances go away when the recorded artifacts and
    the fitted corpus models are refitted.
    """

    if payload.get("artifact_type") != BELIEF_ARTIFACT_TYPE:
        return _point_belief_from_model_payload(payload)
    version = payload.get("format_version")
    if version not in (BELIEF_FORMAT_VERSION, LEGACY_BELIEF_FORMAT_VERSION):
        raise ValueError("unsupported dynamics-belief format")
    model, _ = _executable_model(payload["nominal_model"])
    parameter_belief = dict(payload["parameter_belief"])
    parameter_evidence = dict(payload["parameter_evidence"])
    if version == LEGACY_BELIEF_FORMAT_VERSION:
        warnings.warn(
            "reading a dynamics belief written under format "
            f"{LEGACY_BELIEF_FORMAT_VERSION}: its rotational-response "
            "coordinates are dropped from the parameter evidence to match the "
            "memoryless multirotor model; re-record the artifact",
            stacklevel=3,
        )
        parameter_belief = _without_dropped_parameters(
            parameter_belief, matrix_keys=("covariance",)
        )
        parameter_evidence = _without_dropped_parameters(
            parameter_evidence,
            matrix_keys=("information_matrix",),
            row_keys=("group_score_vectors",),
        )
    return DynamicsBelief(
        model=model,
        predictive_error=predictive_error_from_dict(payload["predictive_error"]),
        parameter_belief=parameter_belief_from_dict(parameter_belief),
        parameter_evidence=parameter_evidence_from_dict(parameter_evidence),
        predictive_error_parameter_update_count=int(
            payload["predictive_error_parameter_update_count"]
        ),
        provenance=dict(payload.get("provenance", {})),
    )


def load_dynamics_belief(path: str | Path) -> DynamicsBelief:
    """Load a belief written by :func:`save_dynamics_belief`.

    A bare nominal-model artifact is read as a point belief; see
    :func:`dynamics_belief_from_payload` for why that tolerance exists and when
    it goes away.
    """

    return dynamics_belief_from_payload(json.loads(Path(path).read_text()))
