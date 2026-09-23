"""One causal gain direction per command with a physical-time response shape."""

import numpy as np

from screen_jax_hybrid import JaxHybrid


class CoupledLag(JaxHybrid):
    """Share prompt/delayed effect direction while fitting response time."""

    taus = (0.02, 0.05, 0.1, 0.2, 0.4)
    prompt_fractions = (0.0, 0.25, 0.5, 0.75, 1.0)

    def _applied(self, tau):
        decay = np.exp(-self.dt / tau)
        applied = [self.commands[0].copy()]
        for command in self.commands:
            applied.append(command + (applied[-1] - command) * decay)
        return np.asarray(applied)

    def _coupled_coefficients(self, rows, applied, tau, prompt):
        count = len(rows)
        command = self.commands[rows]
        midpoint = command + (applied[rows] - command) * np.exp(-0.5 * self.dt / tau)
        effective = prompt * command + (1.0 - prompt) * midpoint
        rate = self.states[:, 3:6]
        target = (rate[rows + 1] - rate[rows]) / self.dt
        m = self.commands.shape[1]
        coefficients = []
        for axis in range(3):
            design = np.column_stack((np.ones(count), effective, -rate[rows, axis]))
            fit = np.linalg.lstsq(design, target[:, axis], rcond=None)[0]
            if fit[-1] < 0:
                fit = np.r_[
                    np.linalg.lstsq(design[:, :-1], target[:, axis], rcond=None)[0],
                    0.0,
                ]
            coefficients.append(
                np.r_[fit[0], prompt * fit[1 : m + 1],
                      (1.0 - prompt) * fit[1 : m + 1], fit[-1]]
            )
        return np.asarray(coefficients)

    def _validation_error(self, coefficient, rows, applied, tau):
        m = self.commands.shape[1]
        predicted = self.states[rows[0], 3:6].copy()
        target = self.states[:, 3:6]
        error = 0.0
        for row in rows:
            command = self.commands[row]
            first = (
                coefficient[:, 0]
                + coefficient[:, 1 : m + 1] @ command
                + coefficient[:, m + 1 : 2 * m + 1] @ applied[row]
                - coefficient[:, -1] * predicted
            )
            midpoint_applied = command + (applied[row] - command) * np.exp(
                -0.5 * self.dt / tau
            )
            midpoint_rate = predicted + 0.5 * self.dt * first
            middle = (
                coefficient[:, 0]
                + coefficient[:, 1 : m + 1] @ command
                + coefficient[:, m + 1 : 2 * m + 1] @ midpoint_applied
                - coefficient[:, -1] * midpoint_rate
            )
            predicted = predicted + self.dt * middle
            error += np.sum((predicted - target[row + 1]) ** 2)
        return error / len(rows)

    def fit_rate(self):
        count = min(25, len(self.commands))
        rows = np.arange(len(self.commands) - count, len(self.commands))
        held = min(5, max(2, round(0.05 / self.dt)))
        training, validation = rows[:-held], rows[-held:]
        candidates = []
        for tau in self.taus:
            applied = self._applied(tau)
            for prompt in self.prompt_fractions:
                coefficient = self._coupled_coefficients(training, applied, tau, prompt)
                error = self._validation_error(coefficient, validation, applied, tau)
                candidates.append((error, tau, prompt, applied))
        _, self.tau, self.prompt_fraction, applied = min(
            candidates, key=lambda item: item[0]
        )
        self.applied = list(applied)
        self.rate_coefficients = self._coupled_coefficients(
            rows, applied, self.tau, self.prompt_fraction
        )
