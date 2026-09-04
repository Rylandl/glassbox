"""Closed-loop control: planning, identification, and command supervision.

The layer has one seam. :mod:`glassbox.control.plan` declares ``PlanModel``,
which is everything a solver knows about a model;
:mod:`glassbox.control.solver` optimizes over one; and
:mod:`glassbox.control.fitted` is the adapter that presents a fitted dynamics
belief as one. :mod:`glassbox.control.identifier` learns a model in flight for
a vehicle that has none, and :mod:`glassbox.control.supervisor` bounds whatever
command the solver returns before it reaches a vehicle. Both are library
components with the same standing as the solver: what they compute is a
belief and a bounded command, not an experiment.
"""

from glassbox.control.identifier import (
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
    RecursiveBootstrapSampleReport,
)
from glassbox.control.supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisedCommand,
    SupervisorMode,
    SupervisorReason,
)

__all__ = [
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "RecursiveBootstrapBelief",
    "RecursiveBootstrapConfig",
    "RecursiveBootstrapIdentifier",
    "RecursiveBootstrapSampleReport",
    "SupervisedCommand",
    "SupervisorMode",
    "SupervisorReason",
]
