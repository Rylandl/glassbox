"""Generic dissipative angular-rate memory in the causal JAX dynamics screen."""

import numpy as np

from screen_jax_hybrid import JaxHybrid


class RateMemory(JaxHybrid):
    """Add one positive high-frequency rate-damping term per body axis."""

    memory_tau = 0.1

    def fit_rate(self):
        rate = self.states[:, 3:6]
        decay = np.exp(-self.dt / self.memory_tau)
        if not hasattr(self, "memory_trace"):
            self.memory_trace = [rate[0].copy()]
        while len(self.memory_trace) < len(rate):
            row = len(self.memory_trace) - 1
            previous = self.memory_trace[-1]
            self.memory_trace.append(rate[row] + (previous - rate[row]) * decay)
        self.rate_memory = self.memory_trace[-1]

        count = min(25, len(self.commands))
        rows = np.arange(len(self.commands) - count, len(self.commands))
        command = self.commands[rows]
        applied = np.asarray([self.applied[row] for row in rows])
        midpoint = command + (applied - command) * np.exp(-0.5 * self.dt / self.tau)
        filtered = np.asarray([self.memory_trace[row] for row in rows])
        midpoint_filtered = rate[rows] + (filtered - rate[rows]) * np.exp(
            -0.5 * self.dt / self.memory_tau
        )
        target = (rate[rows + 1] - rate[rows]) / self.dt
        base = np.column_stack((np.ones(count), command, midpoint))
        coefficients = []
        for axis in range(3):
            design = np.column_stack(
                (base, -rate[rows, axis],
                 -(rate[rows, axis] - midpoint_filtered[:, axis]))
            )
            candidates = []
            for active in ((), (0,), (1,), (0, 1)):
                columns = list(range(base.shape[1])) + [base.shape[1] + j for j in active]
                fitted = np.linalg.lstsq(design[:, columns], target[:, axis], rcond=None)[0]
                if any(value < 0 for value in fitted[base.shape[1] :]):
                    continue
                full = np.zeros(design.shape[1])
                full[columns] = fitted
                error = np.linalg.norm(design @ full - target[:, axis]) ** 2
                candidates.append((error, full))
            coefficients.append(min(candidates, key=lambda item: item[0])[1])
        self.rate_coefficients = np.asarray(coefficients)
