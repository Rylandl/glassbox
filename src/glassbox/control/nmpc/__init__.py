"""Opinionated nonlinear model-predictive control for Glassbox models."""

from glassbox.control.nmpc.geometry import quaternion_log_error, rigid_body_local_error
from glassbox.control.nmpc.solver import NMPCController
from glassbox.control.nmpc.types import (
    NMPCDiagnostics,
    NMPCResult,
    NMPCWarmStart,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveStatus,
    SupportFilterMode,
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
    "SupportFilterMode",
    "TrackingTolerances",
    "quaternion_log_error",
    "rigid_body_local_error",
]
