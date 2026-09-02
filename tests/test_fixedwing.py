import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.data import (
    Trajectory,
    make_trajectory_spec,
    save_trajectory_npz,
    trajectory_windows,
)
from glassbox.dynamics import (
    fixed_wing_trim_control,
    initial_residual_parameters,
    rollout,
    rollout_with_latent,
    state_derivative,
    step,
    step_with_latent,
)
from glassbox.evaluation import rollout_metrics
from glassbox.families import FIXED_WING_FAMILY, family_for_platform
from glassbox.fit_cli import fit_trajectory_artifacts
from glassbox.fixedwing_synthetic import (
    TRIM_AIRSPEED_M_S,
    fixed_wing_trim_state,
    generate_fixed_wing_trajectory,
    initial_fixed_wing_parameter_guess,
    true_fixed_wing_parameters,
)
from glassbox.identification import fit_dynamics
from glassbox.profile_benchmark import benchmark_profiles


def test_fixed_wing_family_declares_canonical_controls() -> None:
    family = family_for_platform("fixedwing")

    assert family is FIXED_WING_FAMILY
    assert family.control_names == ("throttle", "aileron", "elevator", "rudder")
    assert family.supports_residual is True


def test_fixed_wing_trim_is_constant_velocity_level_flight() -> None:
    params = true_fixed_wing_parameters()
    state = jnp.asarray(fixed_wing_trim_state())
    control = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S)

    next_state = step(params, state, control, 0.02)
    expected = np.asarray(state).copy()
    expected[0] += TRIM_AIRSPEED_M_S * 0.02

    np.testing.assert_allclose(next_state, expected, atol=1e-6)


def test_low_rate_step_matches_repeated_native_rate_integration() -> None:
    params = true_fixed_wing_parameters()
    state = (
        jnp.asarray(fixed_wing_trim_state())
        .at[10:13]
        .set(jnp.asarray((0.08, -0.04, 0.03)))
    )
    control = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S).at[1].add(0.04)

    coarse_state, coarse_latent = step_with_latent(params, state, control, control, 0.2)
    fine_states, fine_latent = rollout_with_latent(
        params,
        state,
        jnp.tile(control, (10, 1)),
        0.02,
        initial_motor_state=control,
    )

    np.testing.assert_allclose(coarse_state, fine_states[-1], atol=5e-5)
    np.testing.assert_allclose(coarse_latent, fine_latent[-1], atol=1e-7)


def test_initial_fixed_wing_model_stays_finite_at_five_hertz() -> None:
    params = initial_fixed_wing_parameter_guess()
    state = (
        jnp.asarray(fixed_wing_trim_state())
        .at[10:13]
        .set(jnp.asarray((0.5, -0.3, 0.2)))
    )
    controls = jnp.tile(fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S), (10, 1))

    states = rollout(params, state, controls, 0.2)

    assert bool(jnp.all(jnp.isfinite(states)))


def test_fixed_wing_aerodynamics_use_air_relative_velocity() -> None:
    params = true_fixed_wing_parameters()
    still_air_state = jnp.asarray(fixed_wing_trim_state(15.0))
    windy_state = still_air_state.at[3].set(20.0)
    control = fixed_wing_trim_control(params, 15.0)

    still_air = state_derivative(params, still_air_state, control)
    windy = state_derivative(
        params,
        windy_state,
        control,
        exogenous=jnp.asarray([5.0, 0.0]),
        exogenous_roles=("wind_north", "wind_west"),
    )

    np.testing.assert_allclose(windy[3:], still_air[3:], atol=1e-7)
    assert windy[0] == pytest.approx(20.0)


def test_fixed_wing_rollout_is_differentiable_through_all_controls() -> None:
    params = true_fixed_wing_parameters()
    state = jnp.asarray(fixed_wing_trim_state())
    trim = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S)
    controls = jnp.tile(trim, (5, 1))

    jacobian = jax.jacrev(lambda values: rollout(params, state, values, 0.02)[-1, 0:3])(
        controls
    )

    assert jacobian.shape == (3, 5, 4)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert bool(jnp.all(jnp.linalg.norm(jacobian, axis=(0, 1)) > 0.0))


def test_flying_wing_three_role_rollout_has_no_required_yaw_channel() -> None:
    params = true_fixed_wing_parameters()
    roles = ("throttle", "roll", "pitch")
    state = jnp.asarray(fixed_wing_trim_state())
    trim = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S, roles)
    controls = jnp.tile(trim, (8, 1))
    controls = controls.at[2:5, 1].add(0.08)
    controls = controls.at[5:8, 2].add(-0.06)

    states = np.asarray(rollout(params, state, controls, 0.02, roles))
    trajectory = Trajectory(
        time_s=np.arange(9) * 0.02,
        states=states,
        controls=np.asarray(controls),
        spec=make_trajectory_spec(
            ("throttle", "roll", "pitch"),
            family="fixedwing",
            observation_source="simulator_truth",
            configuration_id="synthetic_flying_wing",
        ),
    )
    metrics = rollout_metrics(params, trajectory)
    windows = trajectory_windows([trajectory], horizon=4, stride=4)
    fit = fit_dynamics(
        windows,
        initial_fixed_wing_parameter_guess(),
        steps=1,
        learning_rate=0.01,
    )
    jacobian = jax.jacrev(
        lambda values: rollout(params, state, values, 0.02, roles)[-1, 0:3]
    )(controls)

    assert trajectory.spec is not None
    assert trajectory.spec.vehicle.controlled_axes == ("roll", "pitch")
    assert metrics["position_rmse_m"] < 1e-6
    assert metrics["attitude_rmse_deg"] < 1e-5
    assert jacobian.shape == (3, 8, 3)
    assert bool(jnp.all(jnp.linalg.norm(jacobian, axis=(0, 1)) > 0.0))
    assert np.isfinite(fit.final_loss)


def test_residual_wraps_flying_wing_without_changing_kinematics() -> None:
    base = true_fixed_wing_parameters()
    roles = ("throttle", "roll", "pitch")
    residual = initial_residual_parameters(base, control_size=len(roles))
    state = jnp.asarray(fixed_wing_trim_state())
    controls = jnp.tile(fixed_wing_trim_control(base, TRIM_AIRSPEED_M_S, roles), (8, 1))
    controls = controls.at[2:5, 1].add(0.08)

    base_states = rollout(base, state, controls, 0.02, roles)
    residual_states = rollout(residual, state, controls, 0.02, roles)

    np.testing.assert_allclose(residual_states, base_states, atol=1e-7)
    assert residual.feature_mean.shape == (9,)


def test_residual_wraps_flap_equipped_layout() -> None:
    base = true_fixed_wing_parameters()._replace(
        log_flap_lift_accel_per_speed_sq=jnp.log(jnp.asarray(0.025)),
        log_flap_drag_accel_per_speed_sq=jnp.log(jnp.asarray(0.012)),
        flap_pitch_angular_accel_per_speed_sq=jnp.asarray(-0.018),
    )
    roles = ("throttle", "roll", "pitch", "yaw", "flap")
    residual = initial_residual_parameters(base, control_size=len(roles))
    state = jnp.asarray(fixed_wing_trim_state())
    control = fixed_wing_trim_control(base, TRIM_AIRSPEED_M_S, roles).at[4].set(0.3)

    base_next = step(base, state, control, 0.02, roles)
    residual_next = step(residual, state, control, 0.02, roles)

    np.testing.assert_allclose(residual_next, base_next, atol=1e-7)
    assert residual.feature_mean.shape == (11,)


def test_moving_flap_adds_lift_drag_and_pitch_effects() -> None:
    params = true_fixed_wing_parameters()._replace(
        log_flap_lift_accel_per_speed_sq=jnp.log(jnp.asarray(0.025)),
        log_flap_drag_accel_per_speed_sq=jnp.log(jnp.asarray(0.012)),
        flap_pitch_angular_accel_per_speed_sq=jnp.asarray(-0.018),
    )
    roles = ("throttle", "roll", "pitch", "yaw", "flap")
    state = jnp.asarray(fixed_wing_trim_state())
    retracted = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S, roles)
    deployed = retracted.at[4].set(0.3)

    baseline = state_derivative(params, state, retracted, roles)
    with_flap = state_derivative(params, state, deployed, roles)

    assert float(with_flap[3]) < float(baseline[3])
    assert float(with_flap[5]) > float(baseline[5])
    assert float(with_flap[11]) < float(baseline[11])


def test_lateral_surface_cross_coupling_adds_adverse_moments() -> None:
    params = true_fixed_wing_parameters()._replace(
        lateral_surface_cross_angular_accel_per_speed_sq=jnp.asarray([0.02, -0.03])
    )
    state = jnp.asarray(fixed_wing_trim_state())
    neutral = fixed_wing_trim_control(params, TRIM_AIRSPEED_M_S)
    baseline = state_derivative(params, state, neutral)
    rudder = state_derivative(params, state, neutral.at[3].add(0.1))
    aileron = state_derivative(params, state, neutral.at[1].add(0.1))

    assert float(rudder[10]) > float(baseline[10])
    assert float(aileron[12]) < float(baseline[12])


def test_fixed_wing_true_model_has_zero_rollout_error() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=4, duration_s=0.4)

    metrics = rollout_metrics(true_fixed_wing_parameters(), trajectory)

    assert metrics["position_rmse_m"] < 1e-5
    assert metrics["attitude_rmse_deg"] < 1e-5


def test_fixed_wing_fit_reduces_multistep_loss() -> None:
    trajectories = [
        generate_fixed_wing_trajectory(seed=0, duration_s=2.0),
        generate_fixed_wing_trajectory(seed=1, duration_s=2.0),
    ]
    windows = trajectory_windows(trajectories, horizon=10, stride=10)

    result = fit_dynamics(
        windows,
        initial_fixed_wing_parameter_guess(),
        steps=60,
        learning_rate=0.02,
    )

    assert result.final_loss < 0.01 * result.initial_loss


def test_fixed_wing_rollout_indexes_controls_by_semantic_role() -> None:
    trajectory = generate_fixed_wing_trajectory(seed=3, duration_s=0.4)
    reordered = trajectory.__class__(
        time_s=trajectory.time_s,
        states=trajectory.states,
        controls=trajectory.controls[:, [0, 2, 1, 3]],
        spec=make_trajectory_spec(
            ("throttle", "elevator", "aileron", "rudder"),
            family="fixedwing",
            observation_source=trajectory.spec.observation_source,
            configuration_id=trajectory.spec.vehicle.configuration_id,
            fixed_states=trajectory.spec.vehicle.fixed_states,
        ),
        labels=trajectory.labels,
        provenance=trajectory.provenance,
    )

    metrics = rollout_metrics(true_fixed_wing_parameters(), reordered)

    assert reordered.spec is not None
    assert reordered.spec.control_roles == ("throttle", "pitch", "roll", "yaw")
    assert metrics["position_rmse_m"] < 1e-5
    assert metrics["attitude_rmse_deg"] < 1e-5


def test_fixed_wing_artifacts_select_model_family_automatically(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"fixed_wing_{seed}.npz"
        save_trajectory_npz(
            generate_fixed_wing_trajectory(seed=seed, duration_s=0.6), path
        )
        paths.append(path)

    params, baseline, report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=3,
        evaluation_horizons_s=(0.1,),
        run_no_lag_ablation=False,
    )

    assert params.__class__.__name__ == "FixedWingDynamicsParams"
    assert baseline is None
    assert report["dataset"]["platform"] == "fixedwing"
    assert report["dataset"]["model_family"] == "effective_fixedwing"
    assert report["configuration"]["control_history_duration_s"] == pytest.approx(1.0)
    assert report["configuration"]["motor_history_duration_s"] is None


def test_fixed_wing_artifacts_fit_platform_neutral_residual(tmp_path) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"fixed_wing_residual_{seed}.npz"
        save_trajectory_npz(
            generate_fixed_wing_trajectory(seed=seed, duration_s=0.6), path
        )
        paths.append(path)

    params, _, report = fit_trajectory_artifacts(
        paths,
        horizon=5,
        steps=2,
        evaluation_horizons_s=(0.1,),
        run_no_lag_ablation=False,
        model_class="structured_residual",
    )

    assert params.base.__class__.__name__ == "FixedWingDynamicsParams"
    assert params.feature_mean.shape == (10,)
    assert np.all(np.asarray(params.feature_scale) > 0.0)
    assert np.all(np.asarray(params.correction_scale) > 0.0)
    assert report["configuration"]["model_class"] == "structured_residual"
    assert (
        report["models"]["learned_lag"]["parameters"]["initial"]["residual"][
            "input_features"
        ]
        == 10
    )
    rollout_loss = report["models"]["learned_lag"]["fit"]["rollout_loss"]
    assert rollout_loss["state_group_weighting"] == "equal_semantic_groups"
    assert rollout_loss["dynamic_envelope"]["body_velocity_half_width_m_s"][0] > 0.0


def test_fixed_wing_profile_benchmark_does_not_apply_multirotor_contract(
    tmp_path,
) -> None:
    paths = []
    for seed in range(3):
        path = tmp_path / f"fixed_wing_profile_{seed}.npz"
        save_trajectory_npz(
            generate_fixed_wing_trajectory(seed=seed, duration_s=0.3), path
        )
        paths.append(path)

    summary = benchmark_profiles(
        paths,
        tmp_path / "fixed_wing_benchmark",
        training_horizons_s=(0.1,),
        evaluation_horizons_s=(0.1,),
        steps=1,
    )

    assert summary["platform"] == "fixedwing"
    assert summary["acceptance"]["status"] == "not_scored"
    assert "no versioned fixed-wing" in summary["acceptance"]["reason"]
