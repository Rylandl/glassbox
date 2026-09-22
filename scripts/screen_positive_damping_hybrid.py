"""Cold generic force readout with a small causal angular-rate model."""

import numpy as np
from scipy.spatial.transform import Rotation

from screen_rollout_readout import ColdTrajectoryReadout


class RateSession:
    def __init__(self, owner):
        self.owner = owner

    @property
    def model(self):
        return self.owner.force.session.model

    def predict(self, past, inputs, future):
        owner = self.owner
        past, inputs, future = map(lambda value: np.asarray(value, dtype=float), (past, inputs, future))
        rate = past[-1, 3:6].copy()
        applied = owner.applied[-1].copy()
        states, commands, prediction = past.copy(), inputs.copy(), []
        dt = self.model.dt_s
        for command in future:
            following = np.asarray(owner.force.session.predict(states, commands, command[None]))[0].copy()
            next_rate, applied = owner.rate_step(rate, applied, command)
            rotation = states[-1, 6:].reshape(3, 3)
            following[3:6] = next_rate
            following[6:] = (
                rotation @ Rotation.from_rotvec(0.5 * dt * (rate + next_rate)).as_matrix()
            ).reshape(9)
            prediction.append(following)
            states = np.concatenate((states[1:], following[None]))
            commands = np.concatenate((commands[1:], command[None]))
            rate = next_rate
        return np.asarray(prediction)


class PositiveDampingHybrid:
    """One fixed recipe; command count and rate coefficients come from the episode."""

    def __init__(self, prefix):
        self.force = ColdTrajectoryReadout(prefix)
        segment = prefix.segments[0]
        self.states = np.array(segment.states, copy=True)
        self.commands = np.array(segment.inputs, copy=True)
        self.applied = [self.commands[0].copy()]
        self.dt = float(segment.dt_s)
        # A fixed slow memory plus the current issued command spans prompt and
        # delayed responses; the episode determines their respective effects.
        self.tau = 8.0 * self.dt
        for command in self.commands:
            self.applied.append(self.filter_step(self.applied[-1], command))
        self.fit_rate()
        self.session = RateSession(self)

    def filter_step(self, applied, command):
        return command + (applied - command) * np.exp(-self.dt / self.tau)

    def fit_rate(self):
        count = min(25, len(self.commands))
        rows = np.arange(len(self.commands) - count, len(self.commands))
        applied = np.asarray(self.applied)
        command = self.commands[rows]
        midpoint = command + (applied[rows] - command) * np.exp(-0.5 * self.dt / self.tau)
        rate = self.states[:, 3:6]
        target = (rate[rows + 1] - rate[rows]) / self.dt
        coefficients = []
        for axis in range(3):
            design = np.column_stack(
                (np.ones(count), command, midpoint, -rate[rows, axis])
            )
            fit = np.linalg.lstsq(design, target[:, axis], rcond=None)[0]
            if fit[-1] < 0:
                fit = np.r_[
                    np.linalg.lstsq(design[:, :-1], target[:, axis], rcond=None)[0],
                    0.0,
                ]
            coefficients.append(fit)
        self.rate_coefficients = np.asarray(coefficients)

    def rate_step(self, rate, applied, command):
        matrix = self.rate_coefficients
        m = len(command)

        def acceleration(value, effect):
            return (
                matrix[:, 0]
                + matrix[:, 1 : m + 1] @ command
                + matrix[:, m + 1 : 2 * m + 1] @ effect
                - matrix[:, -1] * value
            )

        half = rate + 0.5 * self.dt * acceleration(rate, applied)
        effect_half = command + (applied - command) * np.exp(-0.5 * self.dt / self.tau)
        return (
            rate + self.dt * acceleration(half, effect_half),
            self.filter_step(applied, command),
        )

    def observe(self, row, command, following):
        self.force.observe(row, command, following)
        self.states = np.concatenate((self.states, following[None]))
        self.commands = np.concatenate((self.commands, command[None]))
        self.applied.append(self.filter_step(self.applied[-1], command))
        self.fit_rate()
