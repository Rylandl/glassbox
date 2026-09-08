"""One-step errors and structured-parameter derivatives with causal actuation."""

from __future__ import annotations

import jax
from jax import Array

from glassbox.core.dynamics import (
    ModelParams,
    control_state_after_history,
    control_state_trace,
    step_with_latent,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import rigid_body_local_error


def one_step_tangent_error(
    vector: Array,
    template_params: ModelParams,
    states: Array,
    control_history: Array,
    controls: Array,
    starts: Array,
    contexts: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> Array:
    """Predict selected transitions after scanning all preceding commands."""

    params = with_structured_parameter_vector(template_params, vector)
    initial = control_state_after_history(params, control_history, dt_s, control_roles)
    latent = control_state_trace(params, controls, dt_s, control_roles, initial)

    def predict(state: Array, applied: Array, command: Array, context: Array):
        return step_with_latent(
            params,
            state,
            applied,
            command,
            dt_s,
            control_roles,
            context,
            exogenous_roles,
        )[0]

    predicted = jax.vmap(predict)(
        states[starts], latent[starts], controls[starts], contexts[starts]
    )
    return jax.vmap(rigid_body_local_error)(states[starts + 1], predicted)


def one_step_tangent_linearization(
    vector: Array,
    *args,
    **kwargs,
) -> tuple[Array, Array]:
    """Return predicted-minus-measured errors and their parameter Jacobians."""

    def errors_with_aux(selected):
        errors = one_step_tangent_error(selected, *args, **kwargs)
        return errors, errors

    jacobians, errors = jax.jacfwd(errors_with_aux, has_aux=True)(vector)
    return errors, jacobians


compiled_one_step_tangent_error = jax.jit(
    one_step_tangent_error,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
compiled_one_step_tangent_linearization = jax.jit(
    one_step_tangent_linearization,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
