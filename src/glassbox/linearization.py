"""Shared differentiable rollout linearizations in rigid-body tangent space."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from glassbox.belief import (
    TANGENT_STATE_SIZE,
    apply_tangent_correction,
    with_structured_parameter_vector,
)
from glassbox.dynamics import ModelParams, control_state_after_history, step_with_latent
from glassbox.geometry import rigid_body_local_error


def endpoint_tangent_error(
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
    """Return one fixed-horizon endpoint error for a structured vector."""

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


def endpoint_tangent_error_and_jacobian(
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
    """Return an endpoint error and its structured-parameter Jacobian."""

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
        lambda selected: endpoint_tangent_error(
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


def batched_endpoint_tangent_error_and_jacobian(
    vector: Array,
    template_params: ModelParams,
    initial_states: Array,
    control_histories: Array,
    controls: Array,
    targets: Array,
    contexts: Array,
    biases: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> tuple[Array, Array]:
    """Vectorize endpoint errors and Jacobians across equal-horizon windows."""

    return jax.vmap(
        lambda initial, history, command, target, context, bias: (
            endpoint_tangent_error_and_jacobian(
                vector,
                template_params,
                initial,
                history,
                command,
                target,
                context,
                bias,
                dt_s=dt_s,
                control_roles=control_roles,
                exogenous_roles=exogenous_roles,
            )
        )
    )(
        initial_states,
        control_histories,
        controls,
        targets,
        contexts,
        biases,
    )


compiled_endpoint_tangent_error = jax.jit(
    endpoint_tangent_error,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
compiled_endpoint_tangent_linearization = jax.jit(
    endpoint_tangent_error_and_jacobian,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
compiled_batched_endpoint_tangent_linearization = jax.jit(
    batched_endpoint_tangent_error_and_jacobian,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
