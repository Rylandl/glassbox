"""Use a physical-time window for the same compact angular readout."""

from screen_jax_hybrid import JaxHybrid


class LongWindowRate(JaxHybrid):
    def window_transitions(self):
        return max(25, round(0.75 / self.dt))
