"""Serialization for first-class Glassbox dynamics beliefs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalHorizonPredictiveError,
    LocalParameterInformation,
    parameter_belief_from_dict,
    parameter_evidence_from_dict,
    predictive_error_from_dict,
)
from glassbox.core.data import TrajectorySpec
from glassbox.core.model import RuntimeModelSpec
from glassbox.core.model_io import dynamics_model_from_payload, model_payload

BELIEF_FORMAT_VERSION = 3
BELIEF_ARTIFACT_TYPE = "glassbox_dynamics_belief"


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


def _point_belief_from_model_payload(payload: Mapping[str, Any]) -> DynamicsBelief:
    """Wrap a bare nominal-model payload as a belief carrying no evidence."""

    params, nominal = dynamics_model_from_payload(payload)
    return DynamicsBelief(
        params=params,
        input_spec=TrajectorySpec.from_dict(nominal["input_spec"]),
        runtime_spec=RuntimeModelSpec.from_dict(nominal["runtime_spec"]),
        provenance=dict(nominal.get("provenance", {})),
    )


def dynamics_belief_from_payload(payload: Mapping[str, Any]) -> DynamicsBelief:
    """Restore a dynamics belief from an already decoded payload.

    A payload that is a bare nominal model rather than a belief is accepted and
    wrapped as a point belief with no predictive-error and no parameter
    evidence. This tolerance is temporary: model-only artifacts under the
    untracked ``artifacts/`` tree predate the single belief format, and it is
    removed once those artifacts are re-recorded.
    """

    if payload.get("artifact_type") != BELIEF_ARTIFACT_TYPE:
        return _point_belief_from_model_payload(payload)
    if payload.get("format_version") != BELIEF_FORMAT_VERSION:
        raise ValueError("unsupported dynamics-belief format")
    nominal_payload = payload["nominal_model"]
    params, nominal = dynamics_model_from_payload(nominal_payload)
    return DynamicsBelief(
        params=params,
        input_spec=TrajectorySpec.from_dict(nominal["input_spec"]),
        runtime_spec=RuntimeModelSpec.from_dict(nominal["runtime_spec"]),
        predictive_error=predictive_error_from_dict(payload["predictive_error"]),
        parameter_belief=parameter_belief_from_dict(payload["parameter_belief"]),
        parameter_evidence=parameter_evidence_from_dict(payload["parameter_evidence"]),
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
