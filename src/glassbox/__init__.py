"""One shared-physics dynamics learner, fitted separately to each configuration."""

from .learner import STATE_CHANNELS, LearnedDynamics, fit
from ._online_fast import FastOnlineFit as OnlineFit
from .recordings import SequenceCollection, SequenceSegment, segments_from_mask

__all__ = [
    "STATE_CHANNELS",
    "LearnedDynamics",
    "OnlineFit",
    "SequenceCollection",
    "SequenceSegment",
    "fit",
    "segments_from_mask",
]
