"""The stable API for telemetry-driven differentiable dynamics identification.

Everything here is a name a reader meets in the README or in one of the
concept pages, or the type of one of their arguments or return values. The
rest of the library is not hidden, it is simply somewhere more specific:
corpus adapters in ``glassbox.io``, evaluation and benchmarks in
``glassbox.workflows``, command-line front ends in ``glassbox.cli``, and live
vehicle boundaries in ``glassbox.integrations``. Those four subpackages are
never imported by a bare ``import glassbox``, so the core import stays small
and needs no optional extra.

Names that left this list did not become private; import them from the module
that owns them, for example ``from glassbox.core.data import
load_trajectory_npz``.
"""

from glassbox.belief.belief import (
    DynamicsBelief,
    EmpiricalHorizonPredictiveError,
    LocalGaussianParameterBelief,
    LocalParameterInformation,
    PointParameterBelief,
)
from glassbox.control.fitted import NMPCController, plan_model
from glassbox.control.identifier import (
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
)
from glassbox.control.plan import (
    PlanModel,
    Prediction,
    ReferenceTrajectory,
    SafetyEnvelope,
    SolveResult,
    SolverPolicy,
    SolveStatus,
    TrackingTolerances,
)
from glassbox.control.solver import BoundedShootingSolver
from glassbox.control.supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.core.data import Channel, Trajectory, TrajectorySpec
from glassbox.core.dynamics import (
    DynamicsParams,
    FixedWingDynamicsParams,
    ModelParams,
    rollout,
    step,
)
from glassbox.core.model import ActuationMap, ExecutableModel, NonActionableModelError
from glassbox.fitting import (
    FitOutcome,
    FitSpec,
    Holdout,
    LossPolicy,
    WeightingPolicy,
    fit,
)

# Grouped in the order a reader meets these names, not alphabetically: the
# groups are the argument of the surface, and each one carries its reason.
__all__ = [  # noqa: RUF022
    # The telemetry a fit consumes, and the typed contract it carries.
    "Channel",
    "Trajectory",
    "TrajectorySpec",
    # The fit: one call, one spec, one outcome.
    "FitOutcome",
    "FitSpec",
    "Holdout",
    "LossPolicy",
    "WeightingPolicy",
    "fit",
    # The parameters a fit produces and the two functions that execute them.
    "DynamicsParams",
    "FixedWingDynamicsParams",
    "ModelParams",
    "rollout",
    "step",
    # The belief: one executable model, what the evidence resolved, and how
    # wrong the forecasts have been.
    "ActuationMap",
    "DynamicsBelief",
    "EmpiricalHorizonPredictiveError",
    "ExecutableModel",
    "LocalGaussianParameterBelief",
    "LocalParameterInformation",
    "NonActionableModelError",
    "PointParameterBelief",
    # Control: the plan-model seam, the solver behind it, and the bounded
    # result every solve returns.
    "BoundedShootingSolver",
    "NMPCController",
    "PlanModel",
    "Prediction",
    "ReferenceTrajectory",
    "SafetyEnvelope",
    "SolveResult",
    "SolveStatus",
    "SolverPolicy",
    "TrackingTolerances",
    "plan_model",
    # Learning a model in flight, and bounding the command that comes out.
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "RecursiveBootstrapConfig",
    "RecursiveBootstrapIdentifier",
    "SupervisorMode",
    "SupervisorReason",
]
