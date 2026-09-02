from dataclasses import replace

import numpy as np
import pytest

from glassbox.core.evaluation import (
    aggregate_innovation_diagnostics,
    one_step_innovation_diagnostics,
    state_kinematic_compatibility_diagnostics,
)
from glassbox.core.fixedwing_synthetic import (
    initial_fixed_wing_parameter_guess,
    true_fixed_wing_parameters,
)
from glassbox.core.synthetic import (
    generate_trajectory,
    initial_parameter_guess,
    true_parameters,
)

# The two vehicle-family cases below resolve their (expensive, JAX-built)
# trajectory lazily from the session-scoped conftest fixtures at test setup
# time, keyed by a plain string id, rather than building both rollouts
# eagerly inside a `@pytest.mark.parametrize` decorator at import time. That
# keeps `pytest --collect-only` fast: parametrize arguments are evaluated
# during collection, before any fixture exists to build from.
_VEHICLE_CASES = ("quadrotor", "fixedwing")


@pytest.fixture(params=_VEHICLE_CASES)
def matching_case(request):
    if request.param == "quadrotor":
        trajectory = request.getfixturevalue("quadrotor_trajectory_seed9_dur4_0s")
        return trajectory, true_parameters()
    trajectory = request.getfixturevalue("fixedwing_trajectory_seed4_dur4_0s")
    return trajectory, true_fixed_wing_parameters()


@pytest.fixture(params=_VEHICLE_CASES)
def misspecified_case(request):
    if request.param == "quadrotor":
        trajectory = request.getfixturevalue("quadrotor_trajectory_seed9_dur4_0s")
        return trajectory, initial_parameter_guess()
    trajectory = request.getfixturevalue("fixedwing_trajectory_seed4_dur4_0s")
    return trajectory, initial_fixed_wing_parameter_guess()


def test_matching_model_has_no_structured_one_step_innovation(matching_case) -> None:
    trajectory, params = matching_case
    report = one_step_innovation_diagnostics(params, trajectory)

    assert report["status"] == "ok"
    assert report["latent_actuator_state_carried"] is True
    assert report["rigid_body_state_reset_each_interval"] is True
    assert report["future_measurements_used"] is False
    assert report["summary"]["structured_innovation_detected"] is False
    assert all(
        not group["temporally_colored"] and not group["input_correlated"]
        for group in report["groups"].values()
    )


def test_misspecified_model_exposes_temporal_and_input_structure(
    misspecified_case,
) -> None:
    trajectory, params = misspecified_case
    report = one_step_innovation_diagnostics(params, trajectory)

    assert report["summary"]["structured_innovation_detected"] is True
    assert report["summary"]["temporally_colored_group_count"] >= 2
    assert report["summary"]["input_correlated_group_count"] >= 2


def test_quaternion_double_cover_does_not_create_attitude_innovation() -> None:
    trajectory = generate_trajectory(seed=3, duration_s=2.0)
    states = trajectory.states.copy()
    states[:, 6:10] *= -1.0

    report = one_step_innovation_diagnostics(
        true_parameters(), replace(trajectory, states=states)
    )

    attitude = report["groups"]["attitude"]
    assert attitude["temporally_colored"] is False
    assert attitude["input_correlated"] is False
    assert (
        max(report["channels"][f"attitude_{axis}_rad"]["rmse"] for axis in "xyz") < 1e-6
    )


def test_short_trajectory_reports_insufficient_samples() -> None:
    trajectory = generate_trajectory(seed=2, duration_s=0.1)

    report = one_step_innovation_diagnostics(true_parameters(), trajectory)

    assert report["status"] == "insufficient_samples"
    assert report["sample_count"] < report["minimum_sample_count"]


def test_state_compatibility_separates_inconsistent_pose_and_velocity() -> None:
    trajectory = generate_trajectory(seed=5, duration_s=4.0)
    clean = state_kinematic_compatibility_diagnostics(trajectory)
    states = trajectory.states.copy()
    states[:, 0] += 0.2 * np.sin(2.0 * np.pi * trajectory.time_s)
    inconsistent = state_kinematic_compatibility_diagnostics(
        replace(trajectory, states=states)
    )

    assert clean["state_observations_temporally_inconsistent"] is False
    assert inconsistent["state_observations_temporally_inconsistent"] is True
    assert inconsistent["position_velocity_compatibility"]["vector_rmse"] > 0.5
    assert inconsistent["attitude_rate_compatibility"]["vector_rmse"] < 1e-3


def test_aggregate_diagnostics_weight_flights_equally(
    quadrotor_trajectory_seed9_dur4_0s,
) -> None:
    trajectory = quadrotor_trajectory_seed9_dur4_0s
    clean = one_step_innovation_diagnostics(true_parameters(), trajectory)
    structured = one_step_innovation_diagnostics(initial_parameter_guess(), trajectory)

    report = aggregate_innovation_diagnostics([clean, structured])

    assert report["status"] == "ok"
    assert report["flight_count"] == 2
    assert report["valid_flight_count"] == 2
    assert report["flight_fraction_with_any_structured_innovation"] == pytest.approx(
        0.5
    )
    assert report["state_kinematic_compatibility"][
        "inconsistent_flight_fraction"
    ] == pytest.approx(0.0)
    assert np.isfinite(report["groups"]["velocity"]["mean_maximum_abs_autocorrelation"])
