"""Recursive information assimilation with a safeguarded nonlinear mean update.

Every usable transition adds J' R^-1 J once. Its resolved inverse supplies a
parameter direction; a scale bound and backtracking on the actual nonlinear
error plus prior quadratic determine a finite displacement. Information is
retained even if no mean step is usable. Noise is measured from the actual
accepted model, not a cancellation in its linear approximation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.linearization import (
    compiled_one_step_tangent_error,
    compiled_one_step_tangent_linearization,
)
from glassbox.core.data import (
    Trajectory,
    _require_compatible_inputs,
)
from glassbox.core.dynamics import (
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.model import ExecutableModel

VALIDITY_BOUNDARY_TOLERANCE = 1e-6
MAXIMUM_NORMALIZED_PARAMETER_STEP = 1.0
MAXIMUM_BACKTRACKING_STEPS = 20


@dataclass(frozen=True)
class UpdateResult:
    """What one :meth:`DynamicsBelief.absorb` did, and on what evidence.

    ``innovation_rms_before`` and ``innovation_rms_after`` are the one-step
    innovation, whitened by the belief's own noise model before the update, of
    the incoming and the updated parameters on the same windows.
    ``information_gain_nats`` is the gain along the directions the belief
    already resolved; a belief that resolved nothing reports zero and states
    its progress through its rank instead. ``step_norm_prior_sigma`` is the
    length of the step in the metric of the prior precision, which is how many
    standard deviations of what was already known the step moved.
    """

    absorbed: bool
    reason: str | None
    window_count: int
    innovation_rms_before: float | None
    innovation_rms_after: float | None
    information_gain_nats: float | None
    step_norm_prior_sigma: float | None
    maximum_validity_utilization: float | None

    def __post_init__(self) -> None:
        if self.absorbed and self.reason is not None:
            raise ValueError("an absorbed update has no refusal reason")
        if not self.absorbed and not (self.reason or "").strip():
            raise ValueError("a refused update must say why")
        if self.window_count < 0:
            raise ValueError("window count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "absorbed": self.absorbed,
            "reason": self.reason,
            "window_count": self.window_count,
            "innovation_rms_before": self.innovation_rms_before,
            "innovation_rms_after": self.innovation_rms_after,
            "information_gain_nats": self.information_gain_nats,
            "step_norm_prior_sigma": self.step_norm_prior_sigma,
            "maximum_validity_utilization": self.maximum_validity_utilization,
        }


def _refused(reason: str, *, window_count: int = 0, validity: float | None = None):
    return UpdateResult(
        absorbed=False,
        reason=reason,
        window_count=window_count,
        innovation_rms_before=None,
        innovation_rms_after=None,
        information_gain_nats=None,
        step_norm_prior_sigma=None,
        maximum_validity_utilization=validity,
    )


def usable_one_step_transitions(
    model: ExecutableModel,
    telemetry: Trajectory,
) -> tuple[np.ndarray, np.ndarray]:
    """Return which transitions are usable and each sample's envelope use.

    Both endpoints must be finite and inside the operating envelope.
    """

    states = np.asarray(telemetry.states, dtype=np.float64)
    controls = np.asarray(telemetry.controls, dtype=np.float64)
    exogenous = np.asarray(telemetry.exogenous, dtype=np.float64)
    finite_states = np.all(np.isfinite(states), axis=1)
    finite_exogenous = np.all(np.isfinite(exogenous), axis=1)
    safe = np.where((finite_states & finite_exogenous)[:, None], states, 0.0)
    utilization = np.asarray(
        jax.vmap(model.validity_utilization)(
            jnp.asarray(safe),
            jnp.asarray(np.where(finite_exogenous[:, None], exogenous, 0.0)),
        )
    )
    per_sample = np.where(
        finite_states & finite_exogenous,
        np.max(utilization, axis=1),
        np.inf,
    )
    usable_sample = per_sample <= 1.0 + VALIDITY_BOUNDARY_TOLERANCE
    usable = (
        usable_sample[:-1] & usable_sample[1:] & np.all(np.isfinite(controls), axis=1)
    )
    return usable, per_sample


def _worst_utilization(per_sample: np.ndarray, starts: np.ndarray) -> float | None:
    """Return the envelope use of the samples an update actually read."""

    if not len(starts):
        finite = per_sample[np.isfinite(per_sample)]
        return float(np.max(finite)) if len(finite) else None
    return float(max(np.max(per_sample[starts]), np.max(per_sample[starts + 1])))


def _one_step_arguments(
    telemetry: Trajectory,
    starts: np.ndarray,
) -> tuple[jax.Array, ...]:
    """Keep the selected states and all commands leading to them."""

    stop = int(np.max(starts)) + 1
    prefix = telemetry.control_prefix
    history = prefix if prefix is not None and len(prefix) else telemetry.controls[:1]
    return (
        jnp.asarray(telemetry.states[: stop + 1]),
        jnp.asarray(history),
        jnp.asarray(telemetry.controls[:stop]),
        jnp.asarray(starts),
        jnp.asarray(telemetry.exogenous[:stop]),
    )


def one_step_linearization(
    model: ExecutableModel,
    telemetry: Trajectory,
    starts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-step innovations and their structured-parameter Jacobians.

    ``innovations`` is measured minus predicted in the twelve rigid-body local
    coordinates. ``jacobians`` differentiates the predicted endpoint tangent,
    including actuator response to all preceding commands and any control prefix.
    """

    dt_s = float(model.runtime_spec.sample_period_s)
    center = np.asarray(structured_parameter_vector(model.params), dtype=np.float64)
    errors, jacobians = compiled_one_step_tangent_linearization(
        jnp.asarray(center),
        model.params,
        *_one_step_arguments(telemetry, starts),
        dt_s=dt_s,
        control_roles=telemetry.spec.control_roles,
        exogenous_roles=telemetry.spec.exogenous_roles,
    )
    return (
        -np.asarray(errors, dtype=np.float64),
        np.array(jacobians, dtype=np.float64, copy=True),
    )


def absorb(
    belief: DynamicsBelief,
    telemetry: Trajectory,
) -> tuple[DynamicsBelief, UpdateResult]:
    """Add a fresh, nonoverlapping telemetry block and return the updated belief.

    Usable transitions add precision. The resolved inverse gives a direction,
    capped in parameter-scale coordinates and backtracked against the actual
    nonlinear error plus the prior quadratic. A rejected candidate cannot
    replace the finite mean, and unresolved directions receive no step.
    """

    if not isinstance(telemetry, Trajectory):
        raise TypeError("absorbing evidence requires one canonical Trajectory")
    _require_compatible_inputs(belief.input_spec, telemetry.spec, label="belief update")
    information = belief.information
    assert information is not None

    intervals = np.diff(np.asarray(telemetry.time_s, dtype=np.float64))
    if len(intervals) < 1:
        return belief, _refused("telemetry carries no transition")
    observed_dt_s = float(np.median(intervals))
    if not np.allclose(intervals, observed_dt_s, atol=1e-7, rtol=0.0):
        return belief, _refused("absorbing evidence requires fixed-rate telemetry")
    dt_s = float(belief.sample_period_s)
    if not np.isclose(observed_dt_s, dt_s, atol=1e-7, rtol=0.0):
        return belief, _refused(
            "telemetry sample period does not match the runtime model"
        )

    usable, per_sample = usable_one_step_transitions(belief.model, telemetry)
    starts = np.flatnonzero(usable)
    worst_validity = _worst_utilization(per_sample, starts)
    if not len(starts):
        return belief, _refused(
            "no telemetry transition is finite and inside the validity envelope",
            validity=worst_validity,
        )

    center = np.asarray(structured_parameter_vector(belief.params), dtype=np.float64)
    innovations, sensitivity = one_step_linearization(belief.model, telemetry, starts)
    if not (np.all(np.isfinite(innovations)) and np.all(np.isfinite(sensitivity))):
        return belief, _refused(
            "one-step linearization is non-finite on this telemetry",
            window_count=len(starts),
            validity=worst_validity,
        )
    sensitivity[..., ~information.estimable] = 0.0

    noise = information.innovation_noise
    weight = 1.0 / noise
    increment = np.einsum(
        "wip,i,wiq->pq", sensitivity, weight, sensitivity, optimize=True
    )
    increment = 0.5 * (increment + increment.T)
    score = np.einsum("wip,i,wi->p", sensitivity, weight, innovations, optimize=True)

    updated_information = information.with_precision(
        information.precision + increment,
        effective_count=information.effective_count + len(starts),
    )
    direction = updated_information.covariance() @ score
    normalized_length = float(np.linalg.norm(direction / information.scale))
    if not np.isfinite(normalized_length):
        return belief, _refused(
            "parameter update direction is non-finite",
            window_count=len(starts),
            validity=worst_validity,
        )
    step_scale = min(
        1.0, MAXIMUM_NORMALIZED_PARAMETER_STEP / max(normalized_length, 1e-30)
    )
    arguments = (belief.params, *_one_step_arguments(telemetry, starts))
    baseline = float(np.sum(np.square(innovations) / noise))
    updated_model = belief.model
    updated_errors = -innovations
    step = np.zeros_like(direction)
    # Information is assimilated once. Only the parameter displacement is
    # backtracked, on the actual nonlinear residual plus the prior quadratic.
    # Failure to find a descent step keeps a finite mean and still records the
    # information in this block; it never commits a divergent candidate.
    for _ in range(MAXIMUM_BACKTRACKING_STEPS):
        candidate_step = step_scale * direction
        candidate_vector = jnp.asarray(center + candidate_step)
        candidate_params = with_structured_parameter_vector(
            belief.params, candidate_vector
        )
        try:
            candidate_model = belief.model.rebind_parameters(candidate_params)
        except ValueError:
            step_scale *= 0.5
            continue
        errors = np.asarray(
            compiled_one_step_tangent_error(
                candidate_vector,
                *arguments,
                dt_s=dt_s,
                control_roles=telemetry.spec.control_roles,
                exogenous_roles=telemetry.spec.exogenous_roles,
            ),
            dtype=np.float64,
        )
        value = float(np.sum(np.square(errors) / noise))
        prior_cost = float(candidate_step @ information.precision @ candidate_step)
        if (
            np.all(np.isfinite(errors))
            and np.isfinite(prior_cost)
            and value + prior_cost <= baseline
        ):
            updated_model, updated_errors, step = (
                candidate_model,
                errors,
                candidate_step,
            )
            break
        step_scale *= 0.5

    realized_noise = np.maximum(noise, np.mean(np.square(updated_errors), axis=0))
    updated_information = updated_information.with_precision(
        updated_information.precision,
        innovation_noise=realized_noise,
    )
    prior_rank = information.resolved_rank()
    provenance: Mapping[str, Any] = {
        **belief.provenance,
        "update_count": belief.update_count + 1,
        "parameter_distance_since_measurement": (
            belief.parameter_distance_since_measurement
            + float(np.linalg.norm(step / information.scale))
        ),
    }
    updated = DynamicsBelief(
        model=updated_model,
        information=updated_information,
        forecast_error=belief.forecast_error,
        provenance=provenance,
    )
    return updated, UpdateResult(
        absorbed=True,
        reason=None,
        window_count=len(starts),
        innovation_rms_before=float(np.sqrt(np.mean(np.square(innovations) / noise))),
        innovation_rms_after=float(np.sqrt(np.mean(np.square(updated_errors) / noise))),
        information_gain_nats=information.information_gain_nats(increment),
        step_norm_prior_sigma=(
            float(np.sqrt(max(step @ information.precision @ step, 0.0)))
            if prior_rank > 0
            else None
        ),
        maximum_validity_utilization=worst_validity,
    )
