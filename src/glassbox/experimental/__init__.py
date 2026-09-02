"""Research-grade control and ensemble APIs whose contracts can change without notice.

Everything re-exported here is deliberately outside the stable ``glassbox``
surface: bootstrap identification, its online controller, the command
supervisor, and the predictive-ensemble uncertainty workflow are still being
shaped by experiments, so their names, signatures, and semantics may change in
any release. Import them from this subpackage to make that dependency explicit.
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
from glassbox.experimental.dual_control import (
    DualControlConfig,
    DualControlNMPC,
    DualControlResult,
    command_information_log_determinant,
)
from glassbox.workflows.predictive_ensemble import (
    PredictiveEnsemble,
    aggregate_predictive_ensemble_metrics,
    benchmark_predictive_ensemble,
    fit_grouped_disagreement_calibration,
    grouped_bootstrap_multiplicities,
    predictive_ensemble_metrics,
    predictive_uncertainty_candidate_gate,
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
    "DualControlConfig",
    "DualControlNMPC",
    "DualControlResult",
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "PredictiveEnsemble",
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
    "aggregate_predictive_ensemble_metrics",
    "benchmark_predictive_ensemble",
    "command_information_log_determinant",
    "fit_grouped_disagreement_calibration",
    "grouped_bootstrap_multiplicities",
    "plan_bootstrap_excitation",
    "predictive_ensemble_metrics",
    "predictive_uncertainty_candidate_gate",
]
