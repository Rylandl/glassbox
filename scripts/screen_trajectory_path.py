"""One bounded five-checkpoint physical-path correction to the direct readout."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from screen_cold_readout_curvature import coefficients
from screen_physical_so3 import PhysicalSO3Readout
from verify_baseline import require

from glassbox import _dynamics as core
from glassbox._dynamics import VehicleSequenceModel

TRIALS = jnp.asarray((1.0, 0.5, 0.25, 0.125))


def with_head(params, mean):
    f, q = params["linear"].shape[0], params["quadratic"].shape[0]
    return dict(
        params,
        linear=mean[:f],
        quadratic=mean[f : f + q],
        bias=mean[f + q],
        w2=mean[f + q + 1 :],
    )


def trajectory_residual(
    mean, params, norms, past, inputs, future, truth, *, delay, dt_s
):
    fitted = with_head(params, mean)
    result = core._rollout(
        fitted, norms, past[None], inputs[None], future[None], delay, dt_s
    )[0]
    checkpoints = jnp.arange(1, 6) * round(0.05 / dt_s) - 1
    prediction = result[checkpoints]
    observed = truth[checkpoints]
    residual = jnp.concatenate(
        (
            prediction[:, :6] - observed[:, :6],
            (prediction[:, 6:] - observed[:, 6:]) / jnp.sqrt(2.0),
        ),
        axis=-1,
    )
    return (residual / jnp.sqrt(5.0)).reshape(-1)


@partial(jax.jit, static_argnames=("delay", "dt_s"))
def correction(params, norms, gram, past, inputs, future, truth, *, delay, dt_s):
    mean = coefficients(params)

    def residual(value):
        return trajectory_residual(
            value,
            params,
            norms,
            past,
            inputs,
            future,
            truth,
            delay=delay,
            dt_s=dt_s,
        )

    r = residual(mean)
    jacobian = jax.jacrev(residual)(mean).reshape(75, -1)
    column_scale = jnp.repeat(1 / jnp.sqrt(jnp.maximum(1.0, jnp.diag(gram))), 6)
    whitened = jacobian * column_scale[None]
    dual = whitened @ whitened.T + jnp.eye(75, dtype=mean.dtype)
    z = -whitened.T @ jnp.linalg.solve(dual, r)
    z = z * jnp.minimum(1.0, 1.0 / jnp.maximum(jnp.linalg.norm(z), 1e-12))
    delta = (column_scale * z).reshape(mean.shape)
    baseline = jnp.linalg.norm(r)
    trial_norms = jnp.stack(
        [jnp.linalg.norm(residual(mean + alpha * delta)) for alpha in TRIALS]
    )
    improving = jnp.isfinite(trial_norms) & (trial_norms < baseline)
    index = jnp.argmax(improving)
    multiplier = jnp.where(jnp.any(improving), TRIALS[index], 0.0)
    return (
        mean + multiplier * delta,
        r,
        jacobian,
        column_scale,
        z,
        delta,
        trial_norms,
        multiplier,
        baseline,
    )


class TrajectoryPathReadout(PhysicalSO3Readout):
    def __init__(self, prefix):
        super().__init__(prefix)
        model = self.session.model
        self.initial_direct_model = model
        self.horizon = round(0.25 / model.dt_s)
        self.context_states = (
            prefix.segments[0].states[-model.history_steps - self.horizon - 1 :].copy()
        )
        self.context_inputs = (
            prefix.segments[0].inputs[-model.history_steps - self.horizon :].copy()
        )
        require(
            len(self.context_states) == model.history_steps + self.horizon + 1
            and len(self.context_inputs) == model.history_steps + self.horizon,
            "incomplete causal correction prefix",
        )
        self.initial_correction = self.refine()

    def refine(self):
        model = self.session.model
        h = model.history_steps
        past = self.context_states[: h + 1]
        inputs = self.context_inputs[:h]
        future = self.context_inputs[h:]
        truth = self.context_states[h + 1 :]
        with jax.enable_x64(True):
            values = correction(
                model.params,
                model.norms,
                self.gram,
                jnp.asarray(past),
                jnp.asarray(inputs),
                jnp.asarray(future),
                jnp.asarray(truth),
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
            saved = tuple(np.asarray(value) for value in values)
            require(
                all(np.isfinite(value).all() for value in saved),
                "nonfinite trajectory correction",
            )
        head = saved[0]
        self.session._model = VehicleSequenceModel(
            model.dt_s,
            model.history_steps,
            model.delay_steps,
            with_head(model.params, head),
            model.norms,
        )
        return saved

    def observe(self, row, command, following):
        direct = super().observe(row, command, following)
        self.context_states = np.concatenate((self.context_states[1:], following[None]))
        self.context_inputs = np.concatenate((self.context_inputs[1:], command[None]))
        refined = self.refine()
        return direct, refined
