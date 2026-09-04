"""Shared differentiable rollout linearizations in rigid-body tangent space.

Every entry point returns the endpoint error ``predicted - measured`` in the
twelve rigid-body local coordinates, and its Jacobian is therefore the
derivative of the *predicted* endpoint tangent with respect to the structured
parameters, which is what an information update accumulates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from glassbox.core.dynamics import (
    ModelParams,
    control_state_after_history,
    step_with_latent,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import TANGENT_STATE_SIZE, rigid_body_local_error


def endpoint_tangent_error(
    vector: Array,
    template_params: ModelParams,
    initial_state: Array,
    control_history: Array,
    controls: Array,
    target: Array,
    context: Array,
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
    return rigid_body_local_error(target, predicted)


def endpoint_tangent_error_and_jacobian(
    vector: Array,
    template_params: ModelParams,
    initial_state: Array,
    control_history: Array,
    controls: Array,
    target: Array,
    context: Array,
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


def batched_endpoint_tangent_error(
    vector: Array,
    template_params: ModelParams,
    initial_states: Array,
    control_histories: Array,
    controls: Array,
    targets: Array,
    contexts: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> Array:
    """Vectorize endpoint errors across equal-horizon update windows."""

    return jax.vmap(
        lambda initial, history, command, target, context: endpoint_tangent_error(
            vector,
            template_params,
            initial,
            history,
            command,
            target,
            context,
            dt_s=dt_s,
            control_roles=control_roles,
            exogenous_roles=exogenous_roles,
        )
    )(
        initial_states,
        control_histories,
        controls,
        targets,
        contexts,
    )


def batched_endpoint_tangent_error_and_jacobian(
    vector: Array,
    template_params: ModelParams,
    initial_states: Array,
    control_histories: Array,
    controls: Array,
    targets: Array,
    contexts: Array,
    *,
    dt_s: float,
    control_roles: tuple[str, ...],
    exogenous_roles: tuple[str, ...],
) -> tuple[Array, Array]:
    """Vectorize endpoint errors and Jacobians across equal-horizon windows."""

    return jax.vmap(
        lambda initial, history, command, target, context: (
            endpoint_tangent_error_and_jacobian(
                vector,
                template_params,
                initial,
                history,
                command,
                target,
                context,
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
    )


compiled_endpoint_tangent_error = jax.jit(
    endpoint_tangent_error,
    static_argnames=("dt_s", "control_roles", "exogenous_roles"),
)
compiled_batched_endpoint_tangent_error = jax.jit(
    batched_endpoint_tangent_error,
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
