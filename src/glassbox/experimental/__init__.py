"""Research-grade control APIs whose contracts can change without notice.

Everything re-exported here is deliberately outside the stable ``glassbox``
surface: the recursive bootstrap identifier and the command
supervisor are still being shaped by experiments, so their names, signatures,
and semantics may change in any release. Import them from this subpackage to
make that dependency explicit.
"""

from glassbox.control.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisedCommand,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.control.online_bootstrap import (
    RecursiveBeliefValidationReport,
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
    RecursiveBootstrapSampleReport,
)

__all__ = [
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "RecursiveBeliefValidationReport",
    "RecursiveBootstrapBelief",
    "RecursiveBootstrapConfig",
    "RecursiveBootstrapIdentifier",
    "RecursiveBootstrapSampleReport",
    "SupervisedCommand",
    "SupervisorMode",
    "SupervisorReason",
]
