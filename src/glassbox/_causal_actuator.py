"""Episode-fitted generic rigid-body dynamics with causal applied commands.

The fitted arrays contain no vehicle-class information. Prediction reconstructs
the applied-command state from issued commands since the start of the segment.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._learner_arrays import array_fingerprint, load_arrays, save_arrays

FORMAT = "glassbox-causal-actuator-v1"
GRAVITY = (0.0, 0.0, -9.80665)


def rotation_exp(vector):
    """SO(3) exponential with finite derivatives at zero."""
    squared = jnp.sum(vector * vector, axis=-1)
    small = squared < 1e-6
    angle = jnp.sqrt(jnp.where(small, 1.0, squared))
    a = jnp.where(
        small,
        1 - squared / 6 + squared**2 / 120 - squared**3 / 5040,
        jnp.sin(angle) / angle,
    )
    b = jnp.where(
        small,
        0.5 - squared / 24 + squared**2 / 720 - squared**3 / 40320,
        (1 - jnp.cos(angle)) / squared,
    )
    x, y, z = vector
    cross = jnp.array(((0, -z, y), (z, 0, -x), (-y, x, 0)))
    return jnp.eye(3, dtype=vector.dtype) + a * cross + b * (cross @ cross)


def latent_trace(commands, q, coeff, dt_s):
    """Shared nonlinear actuator law, initialized by the first issued command."""
    root = jnp.sqrt(q)
    # Rationalized square-root difference has the correct derivative at u=0.
    targets = commands / (
        (jnp.sqrt(jnp.abs(commands) + q) + root)
        * (jnp.sqrt(1 + q) - root)
    )

    def update(state, target):
        rising = target >= state
        c0 = jnp.where(rising, coeff[0], coeff[2])
        c1 = jnp.where(rising, coeff[1], coeff[3])
        h = dt_s / 4

        def substep(_, current):
            rate = c0 + c1 * (jnp.abs(target) + jnp.abs(current))
            midpoint = target + (current - target) * jnp.exp(-0.5 * h * rate)
            midpoint_rate = c0 + c1 * (jnp.abs(target) + jnp.abs(midpoint))
            return target + (current - target) * jnp.exp(-h * midpoint_rate)

        following = jax.lax.fori_loop(0, 4, substep, state)
        return following, following

    _, following = jax.lax.scan(update, targets[0], targets)
    return jnp.concatenate((targets[0][None, :], following), axis=0)


def _step(state, applied, following, J, Cf, Ct, dt_s):
    v, w, R = state[:3], state[3:6], state[6:].reshape(3, 3)
    m = applied.shape[0]
    H = Ct[: 3 * m].reshape(m, 3).T
    B = Ct[3 * m :].reshape(3, 2 + 2 * m)
    middle = 0.5 * (applied + following)
    applied_rate = (following - applied) / dt_s

    def angular_derivative(rate, command):
        torque = (
            B[:, 0]
            + B[:, 1 : 1 + m] @ command
            + B[:, 1 + m : 1 + 2 * m] @ (command * command)
            - B[:, -1] * rate
            + H @ applied_rate
            + jnp.cross(rate, H @ command)
            - jnp.cross(rate, J @ rate)
        )
        return jnp.linalg.solve(J, torque)

    def linear_derivative(velocity, orientation, command):
        body_velocity = orientation.T @ velocity
        speed = jnp.linalg.norm(body_velocity)
        features = jnp.concatenate(
            (
                jnp.ones(1, dtype=velocity.dtype),
                command,
                command * command,
                body_velocity,
                body_velocity * speed,
            )
        )
        return jnp.asarray(GRAVITY, dtype=velocity.dtype) + orientation @ (
            features @ Cf
        )

    w_mid = w + 0.5 * dt_s * angular_derivative(w, applied)
    R_mid = R @ rotation_exp(0.5 * dt_s * w_mid)
    v_mid = v + 0.5 * dt_s * linear_derivative(v, R, applied)
    v_next = v + dt_s * linear_derivative(v_mid, R_mid, middle)
    w_next = w + dt_s * angular_derivative(w_mid, middle)
    R_next = R @ rotation_exp(dt_s * w_mid)
    return jnp.concatenate((v_next, w_next, R_next.reshape(9)))


@dataclass(frozen=True)
class CausalActuatorModel:
    """Immutable fitted equation; commands in ``forecast`` begin at episode start."""

    dt_s: float
    q: np.ndarray
    coeff: np.ndarray
    inertia: np.ndarray
    force: np.ndarray
    torque: np.ndarray

    def __post_init__(self):
        q, coeff, inertia, force, torque = (
            np.asarray(value, dtype=np.float64)
            for value in (self.q, self.coeff, self.inertia, self.force, self.torque)
        )
        m = force.shape[0] // 2 - 3 if force.ndim == 2 else 0
        if (
            not np.isfinite(self.dt_s)
            or self.dt_s <= 0
            or q.shape != ()
            or not 1e-7 <= q <= 10
            or coeff.shape != (4,)
            or np.any(coeff <= 0)
            or inertia.shape != (3, 3)
            or force.shape != (1 + 2 * m + 6, 3)
            or torque.shape != (3 * m + 3 * (2 + 2 * m),)
            or m < 1
            or any(not np.isfinite(a).all() for a in (q, coeff, inertia, force, torque))
            or not np.allclose(inertia, inertia.T, rtol=0, atol=1e-8)
            or np.linalg.eigvalsh(inertia).min() <= 0
        ):
            raise ValueError("invalid causal actuator model")
        for name, value in (
            ("q", q),
            ("coeff", coeff),
            ("inertia", inertia),
            ("force", force),
            ("torque", torque),
        ):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "dt_s", float(self.dt_s))

    @property
    def input_count(self):
        return (len(self.force) - 7) // 2

    def forecast(self, start, past_inputs, future_inputs):
        """Predict from a state and all issued commands since the episode began."""
        x, past, future = map(jnp.asarray, (start, past_inputs, future_inputs))
        m = self.input_count
        if (
            x.shape != (15,)
            or past.ndim != 2
            or future.ndim != 2
            or past.shape[1] != m
            or future.shape[1] != m
            or len(past) < 1
            or len(future) < 1
        ):
            raise ValueError("forecast needs state and complete aligned command history")
        dtype = x.dtype
        commands = jnp.concatenate((past, future))
        trace = latent_trace(
            commands,
            jnp.asarray(self.q, dtype=dtype),
            jnp.asarray(self.coeff, dtype=dtype),
            self.dt_s,
        )
        J, Cf, Ct = (
            jnp.asarray(a, dtype=dtype)
            for a in (self.inertia, self.force, self.torque)
        )

        def advance(state, pair):
            following = _step(state, pair[0], pair[1], J, Cf, Ct, self.dt_s)
            return following, following

        _, forecast = jax.lax.scan(
            advance, x, (trace[len(past) : -1], trace[len(past) + 1 :])
        )
        return forecast

    def arrays(self):
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in ("q", "coeff", "inertia", "force", "torque")
        }

    def metadata(self):
        return {"format": FORMAT, "dt_s": self.dt_s}

    @property
    def fingerprint(self):
        return array_fingerprint(self.metadata(), self.arrays())

    def save(self, path):
        save_arrays(path, self.metadata(), self.arrays())

    @classmethod
    def load(cls, path):
        metadata, arrays = load_arrays(path)
        return cls.from_arrays(metadata, arrays)

    @classmethod
    def from_arrays(cls, metadata, arrays):
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"format", "dt_s"}
            or metadata["format"] != FORMAT
            or set(arrays) != {"q", "coeff", "inertia", "force", "torque"}
        ):
            raise ValueError("invalid causal actuator revision")
        return cls(metadata["dt_s"], **arrays)
