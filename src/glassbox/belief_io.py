"""Serialization for first-class Glassbox dynamics beliefs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from glassbox.belief import (
    DynamicsBelief,
    parameter_belief_from_dict,
    predictive_error_from_dict,
)
from glassbox.data import TrajectorySpec
from glassbox.model_io import dynamics_model_from_payload, model_payload
from glassbox.runtime import RuntimeModelSpec

BELIEF_FORMAT_VERSION = 1
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
            "predictive_error_included": belief.predictive_error.available,
            "predictive_error_current": belief.predictive_error_current,
        },
        "nominal_model": model_payload(
            belief.params,
            input_spec=belief.input_spec,
            runtime_spec=belief.runtime_spec,
            provenance=belief.provenance,
        ),
        "parameter_belief": belief.parameter_belief.to_dict(),
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


def dynamics_belief_from_payload(payload: Mapping[str, Any]) -> DynamicsBelief:
    """Restore a dynamics belief from an already decoded payload."""

    if payload.get("format_version") != BELIEF_FORMAT_VERSION:
        raise ValueError("unsupported dynamics-belief format")
    if payload.get("artifact_type") != BELIEF_ARTIFACT_TYPE:
        raise ValueError("artifact is not a Glassbox dynamics belief")
    nominal_payload = payload["nominal_model"]
    params, nominal = dynamics_model_from_payload(nominal_payload)
    return DynamicsBelief(
        params=params,
        input_spec=TrajectorySpec.from_dict(nominal["input_spec"]),
        runtime_spec=RuntimeModelSpec.from_dict(nominal["runtime_spec"]),
        predictive_error=predictive_error_from_dict(payload["predictive_error"]),
        parameter_belief=parameter_belief_from_dict(payload["parameter_belief"]),
        predictive_error_parameter_update_count=int(
            payload["predictive_error_parameter_update_count"]
        ),
        provenance=dict(payload.get("provenance", {})),
    )


def load_dynamics_belief(path: str | Path) -> DynamicsBelief:
    """Load a belief written by :func:`save_dynamics_belief`."""

    return dynamics_belief_from_payload(json.loads(Path(path).read_text()))
