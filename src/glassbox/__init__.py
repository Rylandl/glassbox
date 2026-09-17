"""One generic dynamics learner for uniformly sampled observations and commands.

Supply a ``SequenceCollection`` with recording boundaries and channel facts to
``fit``. The returned ``LearnedDynamics`` predicts future observations, saves its
evidence, and returns a new revision from ``update``. Telemetry adapters,
evaluation workflows and control integrations live in their owning modules.
"""

from glassbox.learner import LearnedDynamics, fit
from glassbox.recordings import (
    SequenceCollection,
    SequenceSegment,
    segments_from_mask,
)

__all__ = [  # noqa: RUF022
    "fit",
    "LearnedDynamics",
    "SequenceCollection",
    "SequenceSegment",
    "segments_from_mask",
]
