"""Decompose a horizon shift; diagnostic only, no fitting or controller changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.dynamics import (
    structured_parameter_vector,
    with_structured_parameter_vector,
)
from glassbox.core.geometry import rigid_body_local_error


def scalar_probes():
    """Exact counterexamples distinguish resetting, conditioning, and overlap."""
    # x1=theta, x2=-x1+theta: the old two-step parameter sensitivity cancels.
    # Resetting x1 to its mean with the old C=1 destroys that cancellation.
    # Observing y=x1+v, Var(v)=1, instead gives Var(theta|y)=1/2.
    joint = np.ones((2, 2))  # Cov([x1, theta])
    future = np.asarray([-1.0, 1.0])
    observation = np.asarray([1.0, 0.0])
    cross = joint @ observation
    conditioned = joint - np.outer(cross, cross) / (
        observation @ joint @ observation + 1.0
    )
    exact = joint - np.outer(cross, cross) / (observation @ joint @ observation)
    return {
        "old_joint_variance": float(future @ joint @ future),
        "reset_fixed_parameter_variance": 1.0,
        "conditioned_parameter_variance_noisy_observation": float(conditioned[1, 1]),
        "conditioned_future_variance_noisy_observation": float(
            future @ conditioned @ future
        ),
        "conditioned_parameter_variance_exact_observation": float(exact[1, 1]),
        # Forecast errors caused entirely by coefficient uncertainty already
        # have second moment J C J'; adding the same term doubles it.
        "parameter_only_error_second_moment": 1.0,
        "empirical_plus_parameter_if_same_source": 2.0,
    }


def audit_plan_shift(
    plan, commands, state, latent, *, actual_state=None, actual_latent=None
):
    """Compare matching endpoints, including joint sensitivity carry-through.

    Initial state and latent are known constants. No parameter observation
    update is performed. Continued JVP is an old-prior algebra check, not a
    proposal to retain unobserved physical-state uncertainty after observation.
    """
    commands, state, latent = map(jnp.asarray, (commands, state, latent))
    context = jnp.zeros((plan.horizon_steps, plan.exogenous_size))
    values = plan.values
    old = plan.rollout_commands(commands, state, latent, context, values)
    shifted = jnp.concatenate((commands[1:], commands[-1:]))
    reset = plan.rollout_commands(
        shifted, old.mean_states[1], old.latent_states[1], context, values
    )
    center = structured_parameter_vector(values.parameters)
    factor = values.covariance_factor
    if factor is None:
        factor = jnp.zeros((len(center), 0))

    def directions(carry_state, carry_latent):
        def error(vector):
            params = with_structured_parameter_vector(values.parameters, vector)
            first_state, first_latent, _ = plan._mean_rollout_commands(
                commands[:1], state, latent, context[:1], params
            )
            initial = (
                first_state[1] if carry_state else jax.lax.stop_gradient(first_state[1])
            )
            initial_actuator = (
                first_latent[1]
                if carry_latent
                else jax.lax.stop_gradient(first_latent[1])
            )
            states, _, _ = plan._mean_rollout_commands(
                shifted, initial, initial_actuator, context, params
            )
            return jax.vmap(rigid_body_local_error)(reset.mean_states[1:], states[1:])

        return jax.jit(
            jax.vmap(lambda column: jax.jvp(error, (center,), (column,))[1])
        )(factor.T)

    reset_d = directions(False, False)
    state_d = directions(True, False) - reset_d
    latent_d = directions(False, True) - reset_d
    continued_d = directions(True, True)

    def cov(d):
        return jnp.einsum("kti,ktj->tij", d, d)

    error_cov = values.forecast_error_covariance
    old_parameter = old.tangent_covariance - error_cov
    reset_parameter = cov(reset_d)
    continued_parameter = cov(continued_d)
    utilization = jax.jit(jax.vmap(plan._robust_validity_utilization))

    def parts(states, parameter, empirical):
        zeros = jnp.zeros_like(parameter)
        return {
            name: np.asarray(utilization(states, covariance, context[: len(states)]))
            for name, covariance in (
                ("mean", zeros),
                ("parameter", parameter),
                ("empirical", empirical),
                ("both", parameter + empirical),
            )
        }

    before = parts(old.mean_states[2:], old_parameter[1:], error_cov[1:])
    after = parts(reset.mean_states[1:-1], reset_parameter[:-1], error_cov[:-1])
    index = np.unravel_index(np.argmax(after["both"]), after["both"].shape)
    stage, axis = map(int, index)
    report = {
        "known_initial_physical_and_actuator_state": True,
        "parameter_observation_update": False,
        "retained_endpoint_count": plan.horizon_steps - 1,
        "mean_matching_endpoint_max_abs_error": float(
            jnp.max(jnp.abs(old.mean_states[2:] - reset.mean_states[1:-1]))
        ),
        "joint_chain_covariance_max_abs_error": float(
            jnp.max(jnp.abs(old_parameter[1:] - continued_parameter[:-1]))
        ),
        "sensitivity_additivity_max_abs_error": float(
            jnp.max(jnp.abs(continued_d - reset_d - state_d - latent_d))
        ),
        "critical_shifted_stage_one_based": stage + 1,
        "critical_feature_axis": axis,
        "critical_matching_endpoint": {
            "old": {name: float(array[index]) for name, array in before.items()},
            "reset": {name: float(array[index]) for name, array in after.items()},
        },
        "maximum_retained_utilization": {
            "old": {name: float(array.max()) for name, array in before.items()},
            "reset": {name: float(array.max()) for name, array in after.items()},
        },
        "parameter_covariance_reset_minus_old_eigenvalue_extrema": [
            float(x)
            for x in (
                jnp.min(jnp.linalg.eigvalsh(reset_parameter[:-1] - old_parameter[1:])),
                jnp.max(jnp.linalg.eigvalsh(reset_parameter[:-1] - old_parameter[1:])),
            )
        ],
        "empirical_covariance_reset_minus_old_eigenvalue_extrema": [
            float(x)
            for x in (
                jnp.min(jnp.linalg.eigvalsh(error_cov[:-1] - error_cov[1:])),
                jnp.max(jnp.linalg.eigvalsh(error_cov[:-1] - error_cov[1:])),
            )
        ],
        "state_carried_sensitivity_norm": float(jnp.linalg.norm(state_d)),
        "actuator_carried_sensitivity_norm": float(jnp.linalg.norm(latent_d)),
        "scalar_probes": scalar_probes(),
    }
    if axis >= 3:
        tangent_axis = axis + 6
        t = np.asarray(reset_d[:, stage, tangent_axis])
        ux = np.asarray(state_d[:, stage, tangent_axis])
        ua = np.asarray(latent_d[:, stage, tangent_axis])
        u = ux + ua
        report["critical_rate_parameter_variance_decomposition"] = {
            "reset": float(t @ t),
            "carried_state": float(ux @ ux),
            "carried_actuator": float(ua @ ua),
            "carried_joint": float(u @ u),
            "twice_reset_carried_cross": float(2 * t @ u),
            "old_reconstructed": float((t + u) @ (t + u)),
        }
    if actual_state is not None:
        actual = plan.rollout_commands(
            shifted,
            jnp.asarray(actual_state),
            jnp.asarray(actual_latent),
            context,
            values,
        )
        actual_parts = parts(
            actual.mean_states[1:-1],
            (actual.tangent_covariance - error_cov)[:-1],
            error_cov[:-1],
        )
        report["actual_maximum_retained_utilization"] = {
            name: float(array.max()) for name, array in actual_parts.items()
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belief", type=Path, required=True)
    parser.add_argument(
        "--plan-arrays",
        type=Path,
        required=True,
        help="NPZ: commands, state, latent; optional actual_state and actual_latent",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from glassbox.belief.belief_io import load_dynamics_belief
    from glassbox.control.fitted import NMPCController

    plan = NMPCController(load_dynamics_belief(args.belief)).plan
    with np.load(args.plan_arrays, allow_pickle=False) as arrays:
        report = audit_plan_shift(plan, **dict(arrays))
    report["input_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (args.belief, args.plan_arrays)
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
