"""Contracts for the stable ``glassbox`` surface and its subpackage boundary."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import glassbox

# Snapshot of the stable surface. Additions and removals both have to be made
# here on purpose, so a name never leaves the public API by accident.
EXPECTED_PUBLIC_API = (
    "RIGID_BODY_STATE_SCHEMA",
    "TANGENT_GROUP_ORDER",
    "TANGENT_STATE_ORDER",
    "ActuationMap",
    "BaseDynamicsParams",
    "BeliefUpdateProposal",
    "BeliefUpdateReport",
    "ControlChannel",
    "DirectActuationMap",
    "DynamicsBelief",
    "DynamicsParams",
    "EmpiricalErrorSample",
    "EmpiricalHorizonPredictiveError",
    "ErrorCovarianceScope",
    "ExogenousChannel",
    "FitResult",
    "FixedWingDynamicsParams",
    "HorizonEndpointErrorEvidence",
    "LocalGaussianParameterBelief",
    "LocalParameterInformation",
    "ModelParams",
    "ModelValidityEnvelope",
    "NMPCController",
    "NMPCDiagnostics",
    "NMPCResult",
    "NMPCWarmStart",
    "NonActionableModelError",
    "ObservationChannel",
    "PlanAssessment",
    "PointParameterBelief",
    "PredictiveTrajectory",
    "ReferenceTrajectory",
    "ResidualDynamicsParams",
    "ResolvedLocalGeometry",
    "RolloutLossConfiguration",
    "RuntimeDynamicsBelief",
    "RuntimeDynamicsModel",
    "RuntimeModelSpec",
    "SafetyEnvelope",
    "SolveStatus",
    "SolverPolicy",
    "StructuredParameterPrior",
    "SupportFilterMode",
    "TrackingTolerances",
    "Trajectory",
    "TrajectoryAdapter",
    "TrajectorySpec",
    "TrajectoryWindows",
    "UnavailableParameterEvidence",
    "UnavailablePredictiveError",
    "VehicleConfigurationSpec",
    "aggregate_rollout_metrics",
    "apply_tangent_correction",
    "duration_to_steps",
    "endpoint_error_evidence_by_horizon",
    "fit_dynamics",
    "fit_dynamics_multi_horizon",
    "load_dynamics_belief",
    "load_trajectory_npz",
    "make_trajectory_spec",
    "model_family",
    "propose_dynamics_belief_update",
    "recalibrate_predictive_error",
    "rollout",
    "rollout_divergence_metrics",
    "rollout_loss_configuration",
    "rollout_metrics",
    "rollout_with_latent",
    "runtime_spec_from_fit_report",
    "runtime_spec_from_trajectory",
    "save_dynamics_belief",
    "save_trajectory_npz",
    "split_trajectory",
    "step",
    "step_with_latent",
    "structured_parameter_names",
    "structured_parameter_vector",
    "trajectory_segment",
    "trajectory_windows",
    "update_dynamics_belief",
    "validate_and_commit_dynamics_belief_update",
    "windowed_rollout_metrics",
    "with_structured_parameter_vector",
)

# Subpackages that a bare ``import glassbox`` must never pull in: workflows and
# command-line front ends are heavy, corpus and integration adapters need
# optional extras, and the experimental surface is opt-in by design.
DEFERRED_SUBPACKAGES = (
    "cli",
    "experimental",
    "integrations",
    "io",
    "workflows",
)


def test_public_api_exports_resolve_and_are_unique() -> None:
    assert len(glassbox.__all__) == len(set(glassbox.__all__))
    assert all(hasattr(glassbox, name) for name in glassbox.__all__)


def test_public_api_matches_the_recorded_surface() -> None:
    assert tuple(glassbox.__all__) == EXPECTED_PUBLIC_API


def test_public_api_names_never_shadow_a_submodule() -> None:
    submodules = {info.name for info in pkgutil.iter_modules(glassbox.__path__)}
    assert submodules.isdisjoint(glassbox.__all__)


def test_importing_public_api_does_not_load_deferred_subpackages() -> None:
    code = f"""
import json
import sys
import glassbox

deferred = {DEFERRED_SUBPACKAGES!r}
loaded = sorted(
    name
    for name in sys.modules
    if name.startswith("glassbox.") and name.split(".")[1] in deferred
)
print(json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
