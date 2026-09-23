"""Benchmark the actual public online learner through the frozen runner."""

from glassbox import OnlineFit


class PublicOnline:
    def __init__(self, prefix):
        self.session = OnlineFit(prefix)

    def observe(self, row, command, following):
        self.session.observe(row, command, following)
