"""Episode-only closed-form angular readout for the JAX dynamics screen."""

import numpy as np

from screen_rollout_readout import ColdTrajectoryReadout


class CausalRateFit:
    """Estimate prompt and delayed command effects with positive rate damping."""

    def window_transitions(self):
        return 25

    def __init__(self, prefix):
        self.force = ColdTrajectoryReadout(prefix)
        segment = prefix.segments[0]
        self.states = np.array(segment.states, copy=True)
        self.commands = np.array(segment.inputs, copy=True)
        self.applied = [self.commands[0].copy()]
        self.dt = float(segment.dt_s)
        self.tau = 8.0 * self.dt
        for command in self.commands:
            self.applied.append(self.filter_step(self.applied[-1], command))
        self.fit_rate()

    def filter_step(self, applied, command):
        return command + (applied - command) * np.exp(-self.dt / self.tau)

    def fit_rate(self):
        count = min(self.window_transitions(), len(self.commands))
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

    def observe(self, row, command, following):
        self.force.observe(row, command, following)
        self.states = np.concatenate((self.states, following[None]))
        self.commands = np.concatenate((self.commands, command[None]))
        self.applied.append(self.filter_step(self.applied[-1], command))
        self.fit_rate()
