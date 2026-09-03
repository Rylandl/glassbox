"""Research-grade control APIs whose contracts can change without notice.

Everything re-exported here is deliberately outside the stable ``glassbox``
surface: bootstrap identification, its online controller, and the command
supervisor are still being shaped by experiments, so their names, signatures,
and semantics may change in any release. Import them from this subpackage to
make that dependency explicit.
"""

from glassbox.control.bootstrap_identification import (
    BootstrapArrestCommand,
    BootstrapExcitationConfig,
    BootstrapExcitationPlan,
    BootstrapIdentificationConfig,
    BootstrapIdentificationResult,
    BootstrapModelNotReadyError,
    BootstrapMultirotorIdentifier,
    BootstrapVelocityArrestCommand,
    plan_bootstrap_excitation,
)
from glassbox.control.flight_supervisor import (
    MultirotorFlightSupervisor,
    MultirotorSupervisorConfig,
    SupervisedCommand,
    SupervisorMode,
    SupervisorReason,
)
from glassbox.control.online_bootstrap import (
    ProgressiveBootstrapCommand,
    ProgressiveBootstrapController,
    ProgressiveBootstrapControllerConfig,
    RecursiveBeliefValidationReport,
    RecursiveBootstrapBelief,
    RecursiveBootstrapConfig,
    RecursiveBootstrapIdentifier,
    RecursiveBootstrapSampleReport,
)

__all__ = [
    "BootstrapArrestCommand",
    "BootstrapExcitationConfig",
    "BootstrapExcitationPlan",
    "BootstrapIdentificationConfig",
    "BootstrapIdentificationResult",
    "BootstrapModelNotReadyError",
    "BootstrapMultirotorIdentifier",
    "BootstrapVelocityArrestCommand",
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "ProgressiveBootstrapCommand",
    "ProgressiveBootstrapController",
    "ProgressiveBootstrapControllerConfig",
    "RecursiveBeliefValidationReport",
    "RecursiveBootstrapBelief",
    "RecursiveBootstrapConfig",
    "RecursiveBootstrapIdentifier",
    "RecursiveBootstrapSampleReport",
    "SupervisedCommand",
    "SupervisorMode",
    "SupervisorReason",
    "plan_bootstrap_excitation",
]
