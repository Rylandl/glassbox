"""Serialization for first-class Glassbox dynamics beliefs."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.forecast_error import (
    ForecastErrorEnvelope,
    forecast_error_from_dict,
)
from glassbox.belief.information import (
    ParameterInformation,
    parameter_information_from_dict,
    supported_covariance,
)
from glassbox.core.data import Channel, TrajectorySpec
from glassbox.core.model import (
    ActuationMap,
    DirectActuationMap,
    ExecutableModel,
    NonActionableModelError,
    RuntimeModelSpec,
    default_actuation,
)
from glassbox.core.model_io import dynamics_model_from_payload, model_payload

BELIEF_FORMAT_VERSION = 6
BELIEF_ARTIFACT_TYPE = "glassbox_dynamics_belief"

# Formats 3 and 4 are every belief written before the parameter belief, the
# rank-aware parameter evidence and the horizon predictive-error model became
# one ParameterInformation and one ForecastErrorEnvelope. Format 3 additionally
# predates the deletion of the multirotor rotational-response branch, so its
# evidence is stated over three coordinates the structured parameter vector no
# longer has.
LEGACY_BELIEF_FORMAT_VERSIONS = (3, 4)
DROPPED_PARAMETER_PREFIX = "log_angular_response_time_constant["


def belief_payload(belief: DynamicsBelief) -> dict[str, Any]:
    """Return a JSON-compatible predictive dynamics artifact."""

    assert belief.information is not None
    return {
        "format_version": BELIEF_FORMAT_VERSION,
        "artifact_type": BELIEF_ARTIFACT_TYPE,
        "semantics": {
            "predictive_model": True,
            "posterior": False,
            "state_uncertainty_included": False,
            "parameter_information_resolved_rank": (belief.information.resolved_rank()),
            "forecast_error_included": belief.forecast_error is not None,
        },
        "nominal_model": model_payload(
            belief.params,
            input_spec=belief.input_spec,
            runtime_spec=belief.runtime_spec,
            provenance=belief.provenance,
        ),
        "actuation": _actuation_payload(belief.model.actuation),
        "information": belief.information.to_dict(),
        "forecast_error": (
            None if belief.forecast_error is None else belief.forecast_error.to_dict()
        ),
        "provenance": dict(belief.provenance),
    }


def _actuation_payload(actuation: ActuationMap | None) -> dict[str, Any]:
    """Serialize identity maps; mark arbitrary code as requiring rebinding."""

    if actuation is None:
        return {"kind": "none"}
    return {
        "kind": "direct" if type(actuation) is DirectActuationMap else "external",
        "type": f"{type(actuation).__module__}.{type(actuation).__qualname__}",
        "command_channels": [
            channel.to_dict() for channel in actuation.command_channels
        ],
        "model_control_size": actuation.model_control_size,
    }


def _restore_actuation(
    payload: Mapping[str, Any] | None,
    input_spec: TrajectorySpec,
    override: ActuationMap | None,
) -> ActuationMap | None:
    if payload is None:  # Formats through 5 recorded only the model inputs.
        return default_actuation(input_spec) if override is None else override
    kind = payload.get("kind")
    if kind not in {"none", "direct", "external"}:
        raise ValueError("unsupported actuation mapping format")
    if kind == "external":
        if override is None:
            raise NonActionableModelError(
                f"artifact requires external actuation map {payload.get('type')!r}; "
                "load it with actuation=your_map"
            )
        if [channel.to_dict() for channel in override.command_channels] != payload[
            "command_channels"
        ] or override.model_control_size != payload["model_control_size"]:
            raise ValueError("rebound actuation map does not match the saved interface")
    if override is not None:
        return override
    if kind == "none":
        if default_actuation(input_spec) is not None:
            raise ValueError("absent actuation conflicts with actionable model inputs")
        return None
    restored = DirectActuationMap(
        tuple(Channel.from_dict(channel) for channel in payload["command_channels"])
    )
    if restored.model_control_size != payload["model_control_size"]:
        raise ValueError("direct actuation dimensions do not match the saved interface")
    return restored


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
            trimmed[key] = np.asarray(trimmed[key])[keep].tolist()
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
    return trimmed


def _executable_model(
    payload: Mapping[str, Any],
    *,
    actuation: ActuationMap | None = None,
    actuation_payload: Mapping[str, Any] | None = None,
) -> tuple[ExecutableModel, Mapping[str, Any]]:
    """Rebuild the mean and its saved or explicitly rebound command interface."""

    params, nominal = dynamics_model_from_payload(payload)
    input_spec = TrajectorySpec.from_dict(nominal["input_spec"])
    model = ExecutableModel(
        params,
        input_spec,
        RuntimeModelSpec.from_dict(nominal["runtime_spec"]),
        _restore_actuation(actuation_payload, input_spec, actuation),
    )
    return model, nominal


def _legacy_forecast_error(
    payload: Mapping[str, Any] | None,
) -> ForecastErrorEnvelope | None:
    """Convert a pre-format-5 predictive-error model, folding its bias back in.

    The old model carried a bias and a covariance centered on it, and applied
    the bias to every runtime forecast. Nothing applies it now, so the envelope
    has to describe the whole error: the uncentered second moment is exactly
    ``covariance + bias bias'``, which makes this conversion lossless.
    """

    if payload is None or payload.get("kind") != "empirical_horizon_tangent_moments":
        return None
    bias = np.asarray(payload["tangent_bias"], dtype=np.float64)
    covariance = np.asarray(payload["tangent_covariance"], dtype=np.float64)
    return ForecastErrorEnvelope(
        horizons_s=tuple(payload["horizons_s"]),
        tangent_covariance=covariance + np.einsum("hi,hj->hij", bias, bias),
        raw_sample_count=tuple(payload["raw_sample_count"]),
        effective_sample_count=tuple(payload["effective_sample_count"]),
        independent_group_count=tuple(payload["independent_group_count"]),
        source=str(payload.get("source", "held_out_rollout_endpoints")),
        weighting=str(
            payload.get("weighting", "equal_source_group_then_trajectory_then_endpoint")
        ),
    )


def _legacy_information(
    model: ExecutableModel,
    parameter_belief: Mapping[str, Any],
    parameter_evidence: Mapping[str, Any],
) -> ParameterInformation:
    """Convert a pre-format-5 parameter belief, or fall back to rank zero.

    A stored parameter *covariance* converts exactly: its supported subspace
    inverts to precision and every other direction stays unknown, which is the
    same conversion :meth:`ParameterInformation.seeded_from_members` performs.
    The stored rank-aware *evidence* does not: it was whitened by a
    horizon-averaged held-out forecast covariance rather than by a one-step
    innovation covariance, and neither the noise it assumed nor the number of
    independent transitions behind it can be recovered from the artifact. Such
    a belief loads at rank zero with a warning, keeping its names, scale and
    estimable mask, and is refit rather than reinterpreted.
    """

    information = ParameterInformation.unknown(model.params, source="legacy_artifact")
    if parameter_evidence.get("kind") == "local_structured_parameter_information":
        information = ParameterInformation(
            names=information.names,
            precision=np.zeros_like(information.precision),
            scale=np.asarray(parameter_evidence["parameter_scale"], dtype=np.float64),
            estimable=np.asarray(
                parameter_evidence["fitted_parameter_mask"], dtype=bool
            ),
            innovation_noise=information.innovation_noise,
            noise_floor=information.noise_floor,
            effective_count=0.0,
            rank_relative_tolerance=float(
                parameter_evidence["rank_relative_tolerance"]
            ),
            source="legacy_artifact",
        )
    if parameter_belief.get("kind") != "local_gaussian_structured_parameters":
        warnings.warn(
            "this belief's parameter evidence was whitened by held-out forecast "
            "covariance, which cannot be converted to a one-step innovation "
            "information state; it loads at rank zero, so refit it to recover "
            "what its evidence resolved",
            stacklevel=4,
        )
        return information
    covariance = np.asarray(parameter_belief["covariance"], dtype=np.float64)
    precision = supported_covariance(covariance).precision
    precision[~information.estimable, :] = 0.0
    precision[:, ~information.estimable] = 0.0
    return information.with_precision(
        precision,
        effective_count=float(parameter_belief.get("effective_sample_count", 0.0)),
        source="legacy_parameter_covariance",
    )


def _point_belief_from_model_payload(
    payload: Mapping[str, Any], *, actuation: ActuationMap | None = None
) -> DynamicsBelief:
    """Wrap a bare nominal-model payload as a belief carrying no evidence."""

    model, nominal = _executable_model(payload, actuation=actuation)
    return DynamicsBelief(
        model=model,
        provenance=dict(nominal.get("provenance", {})),
    )


def dynamics_belief_from_payload(
    payload: Mapping[str, Any], *, actuation: ActuationMap | None = None
) -> DynamicsBelief:
    """Restore a dynamics belief from an already decoded payload.

    A payload that is a bare nominal model rather than a belief is accepted and
    wrapped as a point belief with no forecast envelope and rank-zero
    information. This tolerance is temporary: model-only artifacts under the
    untracked ``artifacts/`` tree predate the single belief format, and it is
    removed once those artifacts are re-recorded.

    A payload written under format 3 or 4 is accepted, lossily and with a
    warning where the loss is real; see :func:`_legacy_information` for which
    half converts exactly and which does not.
    """

    if payload.get("artifact_type") != BELIEF_ARTIFACT_TYPE:
        return _point_belief_from_model_payload(payload, actuation=actuation)
    version = payload.get("format_version")
    if version in (5, BELIEF_FORMAT_VERSION):
        if version == BELIEF_FORMAT_VERSION and not isinstance(
            payload.get("actuation"), Mapping
        ):
            raise ValueError("belief artifact is missing its actuation contract")
        model, _ = _executable_model(
            payload["nominal_model"],
            actuation=actuation,
            actuation_payload=payload.get("actuation") if version >= 6 else None,
        )
        forecast_error = payload.get("forecast_error")
        return DynamicsBelief(
            model=model,
            information=parameter_information_from_dict(payload["information"]),
            forecast_error=(
                None
                if forecast_error is None
                else forecast_error_from_dict(forecast_error)
            ),
            provenance=dict(payload.get("provenance", {})),
        )
    if version not in LEGACY_BELIEF_FORMAT_VERSIONS:
        raise ValueError("unsupported dynamics-belief format")
    warnings.warn(
        f"reading a dynamics belief written under format {version}: its "
        "predictive-error bias is folded into the forecast envelope and its "
        "parameter evidence is converted where that is exact; re-record the "
        "artifact",
        stacklevel=3,
    )
    model, _ = _executable_model(payload["nominal_model"], actuation=actuation)
    parameter_belief = dict(payload["parameter_belief"])
    parameter_evidence = dict(payload["parameter_evidence"])
    if version == 3:
        parameter_belief = _without_dropped_parameters(
            parameter_belief, matrix_keys=("covariance",)
        )
        parameter_evidence = _without_dropped_parameters(
            parameter_evidence, matrix_keys=("information_matrix",)
        )
    return DynamicsBelief(
        model=model,
        information=_legacy_information(model, parameter_belief, parameter_evidence),
        forecast_error=_legacy_forecast_error(payload.get("predictive_error")),
        provenance=dict(payload.get("provenance", {})),
    )


def load_dynamics_belief(
    path: str | Path, *, actuation: ActuationMap | None = None
) -> DynamicsBelief:
    """Load a belief written by :func:`save_dynamics_belief`.

    A bare nominal-model artifact is read as a point belief; see
    :func:`dynamics_belief_from_payload` for why that tolerance exists and when
    it goes away.
    """

    return dynamics_belief_from_payload(
        json.loads(Path(path).read_text()), actuation=actuation
    )
