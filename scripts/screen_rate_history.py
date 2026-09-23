"""Check whether the rate-memory screen is self-contained in public history."""

import jax
import jax.numpy as jnp
import numpy as np

from screen_jax_hybrid import JaxHybrid, JaxRateSession, joint_rollout


def rate_state(past, inputs, dt_s, command_tau):
    """Reconstruct both passive memories from only aligned observed history."""
    applied = inputs[0]
    filtered = past[0, 3:6]
    command_decay = jnp.exp(-dt_s / command_tau)
    rate_decay = jnp.exp(-dt_s / 0.1)

    def advance(carry, data):
        command, rate = data
        applied, filtered = carry
        applied = command + (applied - command) * command_decay
        filtered = rate + (filtered - rate) * rate_decay
        return (applied, filtered), None

    (applied, filtered), _ = jax.lax.scan(
        advance, (applied, filtered), (inputs, past[:-1, 3:6])
    )
    return applied, filtered


class HistorySession(JaxRateSession):
    def predict(self, past, inputs, future):
        model = self.model
        with jax.enable_x64(True):
            past, inputs, future = map(jnp.asarray, (past, inputs, future))
            applied, filtered = rate_state(
                past, inputs, model.dt_s, self.owner.tau
            )
            prediction = joint_rollout(
                {key: jnp.asarray(value) for key, value in model.params.items()},
                {key: jnp.asarray(value) for key, value in model.norms.items()},
                past,
                inputs,
                future,
                applied,
                filtered,
                jnp.asarray(self.owner.rate_coefficients),
                jnp.asarray(self.owner.tau),
                jnp.asarray(self.owner.memory_tau),
                delay=model.delay_steps,
                dt_s=model.dt_s,
            )
        return np.asarray(prediction)


class HistoryHybrid(JaxHybrid):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.session = HistorySession(self)


class PhysicalHistoryHybrid(HistoryHybrid):
    """Use the quad-derived 80 ms command time constant at every sample rate."""

    def __init__(self, prefix):
        super().__init__(prefix)
        self.tau = 0.08
        self.applied = [self.commands[0].copy()]
        for command in self.commands:
            self.applied.append(self.filter_step(self.applied[-1], command))
        self.fit_rate()
