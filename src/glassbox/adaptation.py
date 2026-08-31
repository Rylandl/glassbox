"""Fast evidence-preserving updates for structured dynamics beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from glassbox.belief import (
    TANGENT_STATE_SIZE,
    DynamicsBelief,
    LocalGaussianParameterBelief,
    _regularized_covariance,
    apply_tangent_correction,
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.data import Trajectory
from glassbox.dynamics import ModelParams, control_state_after_history, step_with_latent
from glassbox.geometry import rigid_body_local_error
from glassbox.runtime import model_validity_utilization

MAXIMUM_ONLINE_UPDATE_WINDOWS = 64
ACTUATOR_HISTORY_DURATION_S = 1.0


def _endpoint_error(
    vector: Array,
    template_params: ModelParams,
    initial_state: Array,
    control_history: Array,
    controls: Array,
    target: Array,
    context: Array,
    bias: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> Array:
    params = with_structured_parameter_vector(template_params, vector)
    latent = control_state_after_history(
        params,
        control_history,
        dt_s,
        control_roles,
    )

    def transition(carry: tuple[Array, Array], inputs: tuple[Array, Array]):
        state, latent_state = carry
        control, exogenous = inputs
        return step_with_latent(
            params,
            state,
            latent_state,
            control,
            dt_s,
            control_roles,
            exogenous,
            exogenous_roles,
        ), None

    (predicted, _), _ = jax.lax.scan(
        transition,
        (initial_state, latent),
        (controls, context),
    )
    predicted_mean = apply_tangent_correction(predicted, bias)
    return rigid_body_local_error(target, predicted_mean)


def _endpoint_error_and_jacobian(
    vector: Array,
    template_params: ModelParams,
    initial_state: Array,
    control_history: Array,
    controls: Array,
    target: Array,
    context: Array,
    bias: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> tuple[Array, Array]:
    arguments = (
        vector,
        template_params,
        initial_state,
        control_history,
        controls,
        target,
        context,
        bias,
    )
    keywords = {
        "dt_s": dt_s,
        "control_roles": control_roles,
        "exogenous_roles": exogenous_roles,
    }
    value, pullback = jax.vjp(
        lambda selected: _endpoint_error(
            selected,
            *arguments[1:],
            **keywords,
        ),
        vector,
    )
    jacobian = jax.vmap(lambda basis: pullback(basis)[0])(
        jnp.eye(TANGENT_STATE_SIZE, dtype=value.dtype)
    )
    return value, jacobian


_COMPILED_ENDPOINT_ERROR = jax.jit(
    _endpoint_error,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
_COMPILED_ENDPOINT_LINEARIZATION = jax.jit(
    _endpoint_error_and_jacobian,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)


@dataclass(frozen=True)
class BeliefUpdateReport:
    """Audit record for one attempted immutable belief update."""

    applied: bool
    reason: str | None
    source_group: str
    update_horizon_s: float | None
    update_horizon_steps: int | None
    candidate_window_count: int
    used_window_count: int
    normalized_innovation_rms_before: float | None
    normalized_innovation_rms_after: float | None
    realized_local_information_gain_nats: float | None
    structured_parameter_delta_norm: float | None
    prior_covariance_trace: float | None
    posterior_covariance_trace: float | None
    maximum_validity_utilization: float
    prior_update_count: int
    posterior_update_count: int
    predictive_error_marked_stale: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "reason": self.reason,
            "source_group": self.source_group,
            "update_horizon_s": self.update_horizon_s,
            "update_horizon_steps": self.update_horizon_steps,
            "candidate_window_count": self.candidate_window_count,
            "used_window_count": self.used_window_count,
            "normalized_innovation_rms_before": (self.normalized_innovation_rms_before),
            "normalized_innovation_rms_after": (self.normalized_innovation_rms_after),
            "realized_local_information_gain_nats": (
                self.realized_local_information_gain_nats
            ),
            "structured_parameter_delta_norm": self.structured_parameter_delta_norm,
            "prior_covariance_trace": self.prior_covariance_trace,
            "posterior_covariance_trace": self.posterior_covariance_trace,
            "maximum_validity_utilization": self.maximum_validity_utilization,
            "prior_update_count": self.prior_update_count,
            "posterior_update_count": self.posterior_update_count,
            "predictive_error_marked_stale": self.predictive_error_marked_stale,
        }


def _source_group(trajectory: Trajectory) -> str:
    for key in ("source_group", "flight_id", "trajectory_id", "vehicle_id"):
        value = trajectory.labels.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unlabeled_telemetry_segment"


def _compatible_telemetry(belief: DynamicsBelief, telemetry: Trajectory) -> None:
    expected = belief.input_spec
    actual = telemetry.spec.prediction_spec()
    if actual.state_schema != expected.state_schema:
        raise ValueError("telemetry state schema does not match belief")
    if actual.vehicle.family != expected.vehicle.family:
        raise ValueError("telemetry vehicle family does not match belief")
    for attribute in (
        "control_roles",
        "control_semantics",
        "exogenous_roles",
    ):
        if getattr(actual, attribute) != getattr(expected, attribute):
            raise ValueError(f"telemetry {attribute} do not match belief")
    if tuple(channel.unit for channel in actual.controls) != tuple(
        channel.unit for channel in expected.controls
    ):
        raise ValueError("telemetry control units do not match belief")
    if tuple(channel.frame for channel in actual.controls) != tuple(
        channel.frame for channel in expected.controls
    ):
        raise ValueError("telemetry control frames do not match belief")
    if tuple(channel.unit for channel in actual.exogenous) != tuple(
        channel.unit for channel in expected.exogenous
    ):
        raise ValueError("telemetry exogenous units do not match belief")
    if tuple(channel.semantic for channel in actual.exogenous) != tuple(
        channel.semantic for channel in expected.exogenous
    ):
        raise ValueError("telemetry exogenous semantics do not match belief")
    if tuple(channel.frame for channel in actual.exogenous) != tuple(
        channel.frame for channel in expected.exogenous
    ):
        raise ValueError("telemetry exogenous frames do not match belief")


def _window_starts(candidate_count: int) -> np.ndarray:
    if candidate_count <= MAXIMUM_ONLINE_UPDATE_WINDOWS:
        return np.arange(candidate_count, dtype=np.int64)
    count = MAXIMUM_ONLINE_UPDATE_WINDOWS
    return ((2 * np.arange(count, dtype=np.int64) + 1) * candidate_count) // (2 * count)


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _unavailable_report(
    belief: DynamicsBelief,
    telemetry: Trajectory,
    *,
    reason: str,
    maximum_validity: float,
    horizon_s: float | None = None,
    horizon_steps: int | None = None,
    candidate_count: int = 0,
) -> BeliefUpdateReport:
    return BeliefUpdateReport(
        applied=False,
        reason=reason,
        source_group=_source_group(telemetry),
        update_horizon_s=horizon_s,
        update_horizon_steps=horizon_steps,
        candidate_window_count=candidate_count,
        used_window_count=0,
        normalized_innovation_rms_before=None,
        normalized_innovation_rms_after=None,
        realized_local_information_gain_nats=None,
        structured_parameter_delta_norm=None,
        prior_covariance_trace=None,
        posterior_covariance_trace=None,
        maximum_validity_utilization=maximum_validity,
        prior_update_count=belief.parameter_belief.update_count,
        posterior_update_count=belief.parameter_belief.update_count,
        predictive_error_marked_stale=False,
    )


def update_dynamics_belief(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> tuple[DynamicsBelief, BeliefUpdateReport]:
    """Apply a local Gaussian update from nonoverlapping telemetry windows.

    Window duration is selected from the shortest held-out error horizon, so
    every innovation uses an error covariance measured at the same physical
    horizon. The method intentionally exposes no optimizer or gate knobs.
    """

    if not isinstance(telemetry, Trajectory):
        raise TypeError("belief updates require one canonical Trajectory")
    _compatible_telemetry(belief, telemetry)
    maximum_validity = float(
        np.max(
            np.asarray(
                jax.vmap(
                    lambda state, context: model_validity_utilization(
                        state,
                        context,
                        belief.input_spec,
                        belief.runtime_spec.validity_envelope,
                    )
                )(
                    jnp.asarray(telemetry.states),
                    jnp.asarray(telemetry.exogenous),
                )
            )
        )
    )
    if not isinstance(belief.parameter_belief, LocalGaussianParameterBelief):
        return belief, _unavailable_report(
            belief,
            telemetry,
            reason="live adaptation requires an evidence-derived parameter covariance",
            maximum_validity=maximum_validity,
        )
    if not belief.predictive_error.available:
        return belief, _unavailable_report(
            belief,
            telemetry,
            reason="live adaptation requires predictive residual covariance",
            maximum_validity=maximum_validity,
        )
    intervals = np.diff(telemetry.time_s)
    dt_s = float(np.median(intervals))
    if not np.allclose(intervals, dt_s, atol=1e-7, rtol=0.0):
        raise ValueError("live adaptation requires fixed-rate canonical telemetry")
    shortest_horizon = float(belief.predictive_error.horizons_s[0])
    horizon_steps = max(1, round(shortest_horizon / dt_s))
    horizon_s = horizon_steps * dt_s
    candidate_count = len(telemetry.controls) // horizon_steps
    if candidate_count < 1:
        return belief, _unavailable_report(
            belief,
            telemetry,
            reason="telemetry is shorter than the maintained update horizon",
            maximum_validity=maximum_validity,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
        )

    ordinal_starts = _window_starts(candidate_count)
    starts = ordinal_starts * horizon_steps
    history_steps = max(1, int(np.ceil(ACTUATOR_HISTORY_DURATION_S / dt_s)))
    windows: list[tuple[Array, Array, Array, Array, Array, Array, Array]] = []
    for start in starts:
        stop = int(start) + horizon_steps
        history_start = max(0, int(start) - history_steps)
        history = telemetry.controls[history_start : int(start)]
        padding = np.repeat(
            telemetry.controls[0:1], history_steps - len(history), axis=0
        )
        control_history = np.concatenate((padding, history), axis=0)
        initial_state = jnp.asarray(telemetry.states[int(start)])
        controls = jnp.asarray(telemetry.controls[int(start) : stop])
        target = jnp.asarray(telemetry.states[stop])
        context = jnp.asarray(telemetry.exogenous[int(start) : stop])
        bias, covariance = belief.predictive_error.moments(
            horizon_s,
            state=initial_state,
            command=controls[-1],
            exogenous=context[-1],
        )
        windows.append(
            (
                initial_state,
                jnp.asarray(control_history),
                controls,
                target,
                context,
                bias,
                covariance,
            )
        )

    prior_vector = np.asarray(
        structured_parameter_vector(belief.params), dtype=np.float64
    )
    vector = prior_vector.copy()
    prior_covariance = np.asarray(belief.parameter_belief.covariance, dtype=np.float64)
    covariance = prior_covariance.copy()

    def normalized_rms(selected_vector: np.ndarray) -> float:
        squared: list[float] = []
        for initial, history, controls, target, context, bias, residual in windows:
            error = np.asarray(
                _COMPILED_ENDPOINT_ERROR(
                    jnp.asarray(selected_vector),
                    belief.params,
                    initial,
                    history,
                    controls,
                    target,
                    context,
                    bias,
                    dt_s=dt_s,
                    control_roles=belief.input_spec.control_roles,
                    exogenous_roles=belief.input_spec.exogenous_roles,
                ),
                dtype=np.float64,
            )
            residual_regularized = _regularized_covariance(np.asarray(residual))
            squared.append(float(error @ np.linalg.solve(residual_regularized, error)))
        return float(np.sqrt(np.mean(squared) / TANGENT_STATE_SIZE))

    innovation_before = normalized_rms(vector)
    information_gain = 0.0
    for initial, history, controls, target, context, bias, residual in windows:
        error, jacobian = _COMPILED_ENDPOINT_LINEARIZATION(
            jnp.asarray(vector),
            belief.params,
            initial,
            history,
            controls,
            target,
            context,
            bias,
            dt_s=dt_s,
            control_roles=belief.input_spec.control_roles,
            exogenous_roles=belief.input_spec.exogenous_roles,
        )
        error_np = np.asarray(error, dtype=np.float64)
        jacobian_np = np.asarray(jacobian, dtype=np.float64)
        residual_np = _regularized_covariance(np.asarray(residual))
        innovation = _regularized_covariance(
            residual_np + jacobian_np @ covariance @ jacobian_np.T
        )
        _, residual_logdet = np.linalg.slogdet(residual_np)
        _, innovation_logdet = np.linalg.slogdet(innovation)
        information_gain += max(0.0, 0.5 * float(innovation_logdet - residual_logdet))
        gain = (
            covariance
            @ jacobian_np.T
            @ np.linalg.solve(innovation, np.eye(len(error_np)))
        )
        vector -= gain @ error_np
        identity = np.eye(len(covariance))
        covariance = (identity - gain @ jacobian_np) @ covariance @ (
            identity - gain @ jacobian_np
        ).T + gain @ residual_np @ gain.T
        covariance = _project_psd(covariance)

    innovation_after = normalized_rms(vector)
    if not (
        np.all(np.isfinite(vector))
        and np.all(np.isfinite(covariance))
        and np.isfinite(innovation_after)
    ):
        return belief, _unavailable_report(
            belief,
            telemetry,
            reason="local update produced non-finite parameters or covariance",
            maximum_validity=maximum_validity,
            horizon_s=horizon_s,
            horizon_steps=horizon_steps,
            candidate_count=candidate_count,
        )

    update_count = belief.parameter_belief.update_count + 1
    updated_parameter_belief = LocalGaussianParameterBelief(
        parameter_names=belief.parameter_belief.parameter_names,
        covariance=covariance,
        source=belief.parameter_belief.source,
        evidence_count=belief.parameter_belief.evidence_count + len(windows),
        effective_sample_count=(
            belief.parameter_belief.effective_sample_count + len(windows)
        ),
        update_count=update_count,
    )
    report = BeliefUpdateReport(
        applied=True,
        reason=None,
        source_group=_source_group(telemetry),
        update_horizon_s=horizon_s,
        update_horizon_steps=horizon_steps,
        candidate_window_count=candidate_count,
        used_window_count=len(windows),
        normalized_innovation_rms_before=innovation_before,
        normalized_innovation_rms_after=innovation_after,
        realized_local_information_gain_nats=float(information_gain),
        structured_parameter_delta_norm=float(np.linalg.norm(vector - prior_vector)),
        prior_covariance_trace=float(np.trace(prior_covariance)),
        posterior_covariance_trace=float(np.trace(covariance)),
        maximum_validity_utilization=maximum_validity,
        prior_update_count=belief.parameter_belief.update_count,
        posterior_update_count=update_count,
        predictive_error_marked_stale=(
            belief.predictive_error.available and belief.predictive_error_current
        ),
    )
    provenance = dict(belief.provenance)
    provenance["online_adaptation"] = {
        "update_count": update_count,
        "last_update": report.to_dict(),
    }
    updated = DynamicsBelief(
        params=with_structured_parameter_vector(belief.params, jnp.asarray(vector)),
        input_spec=telemetry.spec.prediction_spec(),
        runtime_spec=belief.runtime_spec,
        predictive_error=belief.predictive_error,
        parameter_belief=updated_parameter_belief,
        predictive_error_parameter_update_count=(
            belief.predictive_error_parameter_update_count
        ),
        provenance=provenance,
    )
    return updated, report
