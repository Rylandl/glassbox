"""Contracts for the stable ``glassbox`` surface and its subpackage boundary."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import glassbox

# Snapshot of the stable surface, grouped in the order a reader meets it. A
# name belongs here only if the README or a docs/concepts page uses it, or if
# it is the type of a public function's argument or return value. Additions
# and removals are both deliberate edits here, so a name never enters or
# leaves the public API by accident.
EXPECTED_PUBLIC_API = (
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
    # The three parameter families and the two functions that execute them.
    "BootstrapMultirotorParams",
    "DynamicsParams",
    "FixedWingDynamicsParams",
    "ModelParams",
    "rollout",
    "step",
    # The belief: one executable model, what the evidence resolved, and how
    # wrong the forecasts have been.
    "ActuationMap",
    "DynamicsBelief",
    "ExecutableModel",
    "ForecastErrorEnvelope",
    "NonActionableModelError",
    "ParameterInformation",
    "UpdateResult",
    # Control: the plan-model seam, the solver behind it, and the bounded
    # result every solve returns.
    "BoundedShootingSolver",
    "NMPCController",
    "PlanModel",
    "PlanValues",
    "Prediction",
    "ReferenceTrajectory",
    "SafetyEnvelope",
    "SolveResult",
    "SolveStatus",
    "SolverPolicy",
    "TrackingTolerances",
    "plan_model",
    # Learning a model in flight, and bounding the command that comes out.
    "BootstrapEvidence",
    "MultirotorFlightSupervisor",
    "MultirotorSupervisorConfig",
    "RecursiveBootstrapConfig",
    "RecursiveBootstrapIdentifier",
    "SupervisorMode",
    "SupervisorReason",
)

# Subpackages that a bare ``import glassbox`` must never pull in: workflows and
# command-line front ends are heavy, and corpus and integration adapters need
# optional extras.
DEFERRED_SUBPACKAGES = (
    "cli",
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
