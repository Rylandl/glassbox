"""Independent output kernels behind the same vector transition interface.

Every scalar learner sees every original state, input and context coordinate.
Only kernel/noise parameter sharing changes. Conditional variances remain
marginal variances, not a learned cross-output covariance or rollout uncertainty.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.experimental.transition_gp import (
    GaussianTransition,
    TransitionPrediction,
    TransitionSamples,
    fit_transition_gp,
)


def output_feature_order(state_size, command_size, context_size, output):
    """A bijection from vector feature order to one scalar member's order."""
    return (
        output,
        *range(state_size, state_size + command_size),
        *[i for i in range(state_size) if i != output],
        *range(state_size + command_size, state_size + command_size + context_size),
    )


@dataclass(frozen=True)
class OutputwiseTransition:
    members: tuple[GaussianTransition, ...]

    def __post_init__(self):
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members:
            raise ValueError("at least one output model is required")
        first = self.members[0]
        if first.context_size < len(self.members) - 1 or any(
            m.state_size != 1
            or (m.dt_s, m.command_size, m.context_size, m.mean_mode)
            != (first.dt_s, first.command_size, first.context_size, first.mean_mode)
            for m in self.members
        ):
            raise ValueError("output models must have compatible scalar contracts")

    @property
    def state_size(self):
        return len(self.members)

    @property
    def command_size(self):
        return self.members[0].command_size

    @property
    def context_size(self):
        return self.members[0].context_size - self.state_size + 1

    @property
    def dt_s(self):
        return self.members[0].dt_s

    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                [
                    "glassbox.experimental.outputwise_transition.v1",
                    *[m.fingerprint() for m in self.members],
                ]
            ).encode()
        ).hexdigest()

    def predict(self, states, commands, *, context=None):
        states, commands = jnp.asarray(states), jnp.asarray(commands)
        if states.ndim < 1 or states.shape[-1] != self.state_size:
            raise ValueError("state width does not match the fitted model")
        if commands.shape != (*states.shape[:-1], self.command_size):
            raise ValueError(
                "command shape does not match state batch and channel count"
            )
        if context is None:
            if self.context_size:
                raise ValueError("this model requires context at each prediction")
            context = jnp.empty((*states.shape[:-1], 0))
        context = jnp.asarray(context)
        if context.shape != (*states.shape[:-1], self.context_size):
            raise ValueError("context shape does not match the fitted model")
        predictions = []
        for i, member in enumerate(self.members):
            other = jnp.concatenate(
                (states[..., :i], states[..., i + 1 :], context), axis=-1
            )
            predictions.append(
                member.predict(states[..., i : i + 1], commands, context=other)
            )
        return TransitionPrediction(
            *[jnp.concatenate([p[k] for p in predictions], axis=-1) for k in range(3)]
        )

    def mean_rollout(self, initial_state, commands, *, context=None):
        """Compose all outputs simultaneously; caller supplies causal context."""
        commands, initial_state = jnp.asarray(commands), jnp.asarray(initial_state)
        if commands.ndim != 2 or commands.shape[1] != self.command_size:
            raise ValueError("commands must have shape (steps, command_size)")
        if initial_state.shape != (self.state_size,):
            raise ValueError("initial_state must be a single state vector")
        if context is None:
            if self.context_size:
                raise ValueError("this model requires context for the rollout")
            context = jnp.empty((len(commands), 0))
        context = jnp.asarray(context)
        if context.shape != (len(commands), self.context_size):
            raise ValueError("context must have shape (steps, context_size)")

        def advance(state, item):
            control, auxiliary = item
            mean = self.predict(state, control, context=auxiliary).mean
            return mean, mean

        _, future = jax.lax.scan(advance, initial_state, (commands, context))
        return jnp.concatenate((initial_state[None], future))

    def save(self, directory):
        """Write a new artifact directory, with its manifest written last."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for i, member in enumerate(self.members):
            member.save(directory / f"output-{i}.npz")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "glassbox.experimental.outputwise_transition.v1",
                    "outputs": self.state_size,
                    "fingerprint": self.fingerprint(),
                },
                indent=2,
            )
            + "\n"
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["format"] != "glassbox.experimental.outputwise_transition.v1":
            raise ValueError("unsupported outputwise transition format")
        count = manifest["outputs"]
        if type(count) is not int or count < 1:
            raise ValueError("invalid output count")
        model = cls(
            tuple(
                GaussianTransition.load(directory / f"output-{i}.npz")
                for i in range(count)
            )
        )
        if model.fingerprint() != manifest["fingerprint"]:
            raise ValueError("outputwise transition fingerprint does not match members")
        return model


def fit_outputwise_transition_gp(
    samples, *, kernel="matern52", steps=160, restarts=2, mean_mode="absolute"
):
    """Fit one GP per output without dropping or duplicating input coordinates."""
    members = []
    for i in range(samples.states.shape[1]):
        scalar = TransitionSamples(
            samples.states[:, i : i + 1],
            samples.commands,
            samples.next_states[:, i : i + 1],
            samples.dt_s,
            np.concatenate(
                (samples.states[:, :i], samples.states[:, i + 1 :], samples.context),
                axis=1,
            ),
        )
        members.append(
            fit_transition_gp(
                scalar,
                kernel=kernel,
                steps=steps,
                restarts=restarts,
                mean_mode=mean_mode,
            )
        )
    return OutputwiseTransition(tuple(members))
