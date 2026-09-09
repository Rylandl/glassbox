from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief.information import innovation_noise_floor
from glassbox.belief.parameter_evidence import innovation_noise
from glassbox.core.diagnostics import (
    aggregate_innovation_diagnostics,
    one_step_innovation_diagnostics,
    state_kinematic_compatibility_diagnostics,
)
from glassbox.core.fixedwing_synthetic import (
    initial_fixed_wing_parameter_guess,
    true_fixed_wing_parameters,
)
from glassbox.core.geometry import state_plus_tangent
from glassbox.core.metrics import one_step_innovations
from glassbox.core.synthetic import (
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
    report = one_step_innovation_diagnostics(
        one_step_innovations(params, trajectory), trajectory
    )

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
    report = one_step_innovation_diagnostics(
        one_step_innovations(params, trajectory), trajectory
    )

    assert report["summary"]["structured_innovation_detected"] is True
    assert report["summary"]["temporally_colored_group_count"] >= 2
    assert report["summary"]["input_correlated_group_count"] >= 2
    assert report["summary"]["nonadjacent_correlated_group_count"] >= 2


@pytest.mark.parametrize("amplitude, detected", [(1e-8, False), (1e-3, True)])
def test_input_correlation_requires_resolved_residuals(
    quadrotor_trajectory_seed9_dur4_0s, amplitude, detected
) -> None:
    trajectory = quadrotor_trajectory_seed9_dur4_0s
    innovations = amplitude * np.repeat(trajectory.controls[:, :1], 12, axis=1)
    original = innovations.copy()

    report = one_step_innovation_diagnostics(innovations, trajectory)

    assert report["summary"]["structured_innovation_detected"] is detected
    assert all(
        channel["input_correlated"] is detected
        for channel in report["channels"].values()
    )
    assert all(
        channel["strongest_past_or_current_input_correlation"] > 0.99
        for channel in report["channels"].values()
    )
    np.testing.assert_array_equal(innovations, original)


def test_observation_noise_creates_adjacent_residual_correlation(matching_case) -> None:
    trajectory, params = matching_case
    noise = np.random.default_rng(482).normal(size=(len(trajectory.states), 12))
    noise *= np.repeat((0.002, 0.02, 0.002, 0.01), 3)
    observed = replace(
        trajectory,
        states=np.asarray(
            jax.vmap(state_plus_tangent)(
                jnp.asarray(trajectory.states), jnp.asarray(noise)
            )
        ),
    )
    report = one_step_innovation_diagnostics(
        one_step_innovations(params, observed), observed
    )

    assert report["summary"]["temporally_colored_group_count"] == 4
    assert report["summary"]["nonadjacent_correlated_group_count"] == 0
    assert all(
        report["channels"][f"{group}_{axis}_{unit}"]["lag_one_autocorrelation"] < -0.3
        for group, unit in (("position", "m"), ("attitude", "rad"))
        for axis in "xyz"
    )


def test_missing_intervals_do_not_become_adjacent(quadrotor_trajectory_seed9_dur4_0s):
    trajectory = replace(
        quadrotor_trajectory_seed9_dur4_0s,
        control_prefix=quadrotor_trajectory_seed9_dur4_0s.controls[:1],
    )
    innovations = np.repeat(np.arange(len(trajectory.controls))[:, None], 12, axis=1)
    innovations = innovations.astype(float)
    innovations[1::2] = np.nan

    report = one_step_innovation_diagnostics(innovations, trajectory)

    assert report["sample_count"] == len(innovations[::2])
    assert all(
        channel["lag_one_autocorrelation"] == 0.0
        and channel["nonadjacent_autocorrelation_lag_steps"] % 2 == 0
        and channel["nonadjacent_correlated"]
        for channel in report["channels"].values()
    )


def test_slow_sampling_has_no_nonadjacent_lag_in_diagnostic_window(
    quadrotor_trajectory_seed9_dur4_0s,
):
    trajectory = replace(
        quadrotor_trajectory_seed9_dur4_0s,
        time_s=np.arange(len(quadrotor_trajectory_seed9_dur4_0s.states)) * 0.6,
    )
    report = one_step_innovation_diagnostics(
        np.ones((len(trajectory.controls), 12)),
        trajectory,
    )

    assert report["maximum_lag_steps"] == 1
    assert report["summary"]["nonadjacent_correlated_group_count"] == 0


def test_quaternion_double_cover_does_not_create_attitude_innovation(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(3, 2.0)
    states = trajectory.states.copy()
    states[:, 6:10] *= -1.0

    observed = replace(trajectory, states=states)
    report = one_step_innovation_diagnostics(
        one_step_innovations(true_parameters(), observed), observed
    )

    attitude = report["groups"]["attitude"]
    assert attitude["temporally_colored"] is False
    assert attitude["input_correlated"] is False
    assert (
        max(report["channels"][f"attitude_{axis}_rad"]["rmse"] for axis in "xyz") < 1e-6
    )


def test_short_trajectory_reports_insufficient_samples(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(2, 0.1)

    report = one_step_innovation_diagnostics(
        one_step_innovations(true_parameters(), trajectory), trajectory
    )

    assert report["status"] == "insufficient_samples"
    assert report["sample_count"] < report["minimum_sample_count"]


def test_noise_weights_flights_equally_without_subtracting_bias() -> None:
    short = np.full((2, 12), 2.0)
    long = np.full((20, 12), 4.0)
    # A partially nonfinite row contributes no coordinates. An empty flight
    # contributes no weight, and the floor applies after averaging flights.
    short[0, 3] = np.nan
    long[0, 5] = np.inf
    short[:, 0] = 0.0
    long[:, 0] = 0.0
    expected = np.full(12, (2.0**2 + 4.0**2) / 2)
    expected[0] = innovation_noise_floor()[0]

    np.testing.assert_array_equal(
        innovation_noise(iter((short, long, np.empty((0, 12))))), expected
    )
    np.testing.assert_array_equal(
        innovation_noise([np.full((3, 12), np.nan)]), innovation_noise_floor()
    )


def test_diagnostic_transient_trimming_does_not_change_noise(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(2, 1.0)
    innovations = np.zeros((len(trajectory.controls), 12))
    transient_count = len(innovations) // 10
    innovations[:transient_count, 0] = 9.0
    original = innovations.copy()
    noise = innovation_noise([innovations])

    cold = one_step_innovation_diagnostics(innovations, trajectory)
    initialized = one_step_innovation_diagnostics(
        innovations, replace(trajectory, control_prefix=trajectory.controls[:1])
    )

    assert cold["initialization_discard_steps"] == transient_count
    assert cold["channels"]["position_x_m"]["rmse"] == 0.0
    assert initialized["initialization_discard_steps"] == 0
    assert initialized["channels"]["position_x_m"]["rmse"] == pytest.approx(
        np.sqrt(noise[0])
    )
    assert noise[0] == pytest.approx(81.0 * transient_count / len(innovations))
    np.testing.assert_array_equal(innovations, original)
    np.testing.assert_array_equal(innovation_noise([innovations]), noise)


def test_state_compatibility_separates_inconsistent_pose_and_velocity(
    quadrotor_flight,
) -> None:
    trajectory = quadrotor_flight(5, 4.0)
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
    clean = one_step_innovation_diagnostics(
        one_step_innovations(true_parameters(), trajectory), trajectory
    )
    structured = one_step_innovation_diagnostics(
        one_step_innovations(initial_parameter_guess(), trajectory), trajectory
    )

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
