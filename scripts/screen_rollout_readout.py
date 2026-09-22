"""One causal 250 ms trajectory correction of the fast linear readout."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve
from screen_physical_so3 import PhysicalSO3Readout
from verify_baseline import require

from glassbox._dynamics import VehicleSequenceModel, _rollout


def with_coefficients(model, coefficient):
    f, q = len(model.params["linear"]), len(model.params["quadratic"])
    params = dict(
        model.params,
        linear=coefficient[:f],
        quadratic=coefficient[f : f + q],
        bias=coefficient[f + q],
        w2=coefficient[f + q + 1 :],
    )
    return VehicleSequenceModel(
        model.dt_s, model.history_steps, model.delay_steps, params, model.norms
    )


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def trajectory_correction(
    params,
    norms,
    mean,
    information,
    count,
    past,
    past_inputs,
    future_inputs,
    targets,
    scale,
    *,
    delay,
    dt_s,
):
    """One preconditioned Gauss-Newton step on the observed short rollout."""
    f, q = params["linear"].shape[0], params["quadratic"].shape[0]

    def residual(coefficient):
        trial = dict(
            params,
            linear=coefficient[:f],
            quadratic=coefficient[f : f + q],
            bias=coefficient[f + q],
            w2=coefficient[f + q + 1 :],
        )
        forecast = _rollout(
            trial, norms, past, past_inputs, future_inputs, delay, dt_s
        )
        return (forecast - targets) / scale

    error, pullback = jax.vjp(residual, mean)
    length = error.size
    gradient = pullback(error)[0] / length
    root = jnp.sqrt(jnp.diag(information))
    lower = jnp.linalg.cholesky(information / root[:, None] / root[None, :])
    direction = cho_solve((lower, True), gradient / root[:, None]) / root[:, None]
    effect = jax.jvp(residual, (mean,), (direction,))[1]
    curvature = jnp.sum(direction * (information @ direction))
    # Compare the completed trajectory with the one-step initialization fit;
    # the scalar step minimizes their local quadratic objective.
    step = count * jnp.sum(error * effect) / (
        length * curvature + count * jnp.sum(effect * effect)
    )
    corrected = mean - step * direction
    return corrected, step, jnp.sqrt(jnp.mean(error * error))


class ColdTrajectoryReadout(PhysicalSO3Readout):
    def __init__(self, prefix):
        super().__init__(prefix)
        model = self.session.model
        trajectory_length = round(0.25 / model.dt_s)
        size = model.history_steps + trajectory_length
        segment = prefix.segments[0]
        states = segment.states[-size - 1 :]
        inputs = segment.inputs[-size:]
        require(len(states) == size + 1, "short prefix trajectory")
        h = model.history_steps
        with jax.enable_x64(True):
            corrected, _, _ = trajectory_correction(
                model.params,
                model.norms,
                self.mean,
                self.gram,
                self.session.report["initialized_transition_count"],
                jnp.asarray(states[: h + 1])[None],
                jnp.asarray(inputs[:h])[None],
                jnp.asarray(inputs[h:])[None],
                jnp.asarray(states[h + 1 :])[None],
                jnp.asarray(self.session._scale[-1]),
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            host = np.asarray(corrected)
            require(np.isfinite(host).all(), "nonfinite initial trajectory fit")
            self.mean = corrected
            self.rhs = self.gram @ corrected
        self.session._model = with_coefficients(model, host)
