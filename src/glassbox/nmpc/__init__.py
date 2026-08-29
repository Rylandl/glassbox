"""Opinionated nonlinear model-predictive control for Glassbox models."""

from glassbox.nmpc.geometry import quaternion_log_error, rigid_body_local_error
from glassbox.nmpc.solver import NMPCController
from glassbox.nmpc.types import (
    NMPCDiagnostics,
    NMPCResult,
    NMPCWarmStart,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveStatus,
    TrackingTolerances,
)

__all__ = [
    "NMPCController",
    "NMPCDiagnostics",
    "NMPCResult",
    "NMPCWarmStart",
    "ReferenceTrajectory",
    "SafetyEnvelope",
    "SolveStatus",
    "TrackingTolerances",
    "quaternion_log_error",
    "rigid_body_local_error",
]
