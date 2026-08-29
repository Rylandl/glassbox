import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.dynamics import (
    MOTOR_MIXER,
    DynamicsParams,
    history_residual_from_residual,
    hover_control,
    initial_residual_parameters,
    latent_state_after_history,
    rollout,
    rollout_with_latent,
    state_derivative,
    step,
    step_with_latent,
    with_angular_dynamics_authority,
    with_constant_angular_rate,
    with_instantaneous_residual_response,
    with_instantaneous_rotational_response,
    with_thrust_command_offset,
)
from glassbox.evaluation import rollout_metrics
from glassbox.synthetic import generate_trajectory, resting_state, true_parameters


def test_hover_is_an_equilibrium() -> None:
    params = true_parameters()
    state = jnp.asarray(resting_state())
    next_state = step(params, state, hover_control(params), 0.02)

    np.testing.assert_allclose(next_state, state, atol=1e-6)


def test_shared_thrust_command_offset_preserves_hover_equilibrium() -> None:
    params = with_thrust_command_offset(true_parameters(), -0.15)
    state = jnp.asarray(resting_state())

    next_state = step(params, state, hover_control(params), 0.02)

    np.testing.assert_allclose(next_state, state, atol=1e-6)
    assert float(hover_control(params)[0]) == pytest.approx(
        float(hover_control(true_parameters())[0]) - 0.15,
        abs=1e-6,
    )


def test_thrust_command_offset_is_bounded() -> None:
    with pytest.raises(ValueError, match="strictly within"):
        with_thrust_command_offset(true_parameters(), 0.3)


def test_rollout_is_differentiable_with_respect_to_controls() -> None:
    params = true_parameters()
    state = jnp.asarray(resting_state())
    controls = jnp.tile(hover_control(params), (3, 1))

    def final_position(control_sequence: jax.Array) -> jax.Array:
        return rollout(params, state, control_sequence, 0.02)[-1, 0:3]

    jacobian = jax.jacrev(final_position)(controls)

    assert jacobian.shape == (3, 3, 4)
    assert bool(jnp.all(jnp.isfinite(jacobian)))
    assert float(jnp.linalg.norm(jacobian)) > 0.0


def test_true_model_has_zero_attitude_rollout_error() -> None:
    trajectory = generate_trajectory(seed=4, duration_s=0.2)

    metrics = rollout_metrics(true_parameters(), trajectory)

    assert metrics["attitude_rmse_deg"] < 1e-5


def test_motor_state_has_a_first_order_step_response() -> None:
    params = true_parameters()
    state = jnp.asarray(resting_state())
    hover = hover_control(params)
    command = hover + 0.1
    controls = jnp.tile(command, (5, 1))

    _, motor_states = rollout_with_latent(
        params,
        state,
        controls,
        0.02,
        initial_motor_state=hover,
    )

    assert bool(jnp.all(motor_states[1:] > motor_states[:-1]))
    assert bool(jnp.all(motor_states[-1] < command))


def test_rotational_response_can_lag_measured_motor_state() -> None:
    params = DynamicsParams.from_physical(
        thrust_accel=5.4,
        angular_accel=(18.0, 16.5, 7.5),
        linear_drag=0.18,
        angular_drag=(0.24, 0.21, 0.13),
        motor_time_constant=1e-4,
        angular_response_time_constant=(0.1, 0.1, 0.1),
    )
    state = jnp.asarray(resting_state())
    hover = hover_control(params)
    command = hover + 0.02 * MOTOR_MIXER[0]

    _, latent = step_with_latent(params, state, hover, command, 0.02)
    target = params.physical()["angular_control_matrix"] @ (MOTOR_MIXER @ command)

    assert latent.shape == (7,)
    assert float(latent[4]) > 0.0
    assert float(latent[4]) < float(target[0])
    np.testing.assert_allclose(latent[5:], 0.0, atol=1e-6)


def test_instantaneous_rotational_response_ignores_stale_latent_state() -> None:
    params = with_instantaneous_rotational_response(true_parameters())
    state = jnp.asarray(resting_state())
    applied = jnp.asarray([0.4, 0.4, 0.4, 0.4])
    command = jnp.asarray([0.8, 0.2, 0.2, 0.8])

    reference, _ = step_with_latent(params, state, applied, command, 0.01)
    stale_latent = jnp.concatenate((applied, jnp.asarray([99.0, -99.0, 50.0])))
    actual, _ = step_with_latent(params, state, stale_latent, command, 0.01)

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)


def test_rotational_control_cross_coupling_is_bounded_and_expressive() -> None:
    params = DynamicsParams.from_physical(
        thrust_accel=5.4,
        angular_accel=(18.0, 16.5, 7.5),
        linear_drag=0.18,
        angular_drag=(0.24, 0.21, 0.13),
        motor_time_constant=1e-4,
        angular_control_cross_coupling=(
            (0.0, 0.2, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )
    pitch_command = hover_control(params) + 0.02 * MOTOR_MIXER[1]
    derivative = state_derivative(params, jnp.asarray(resting_state()), pitch_command)

    assert float(derivative[10]) > 0.0
    assert float(derivative[11]) > float(derivative[10])
    assert np.max(np.abs(params.physical()["angular_control_cross_coupling"])) <= 0.5


def test_constant_rate_diagnostic_disables_angular_acceleration() -> None:
    params = with_constant_angular_rate(true_parameters())
    state = jnp.asarray(resting_state()).at[10:13].set(jnp.asarray([0.4, -0.2, 0.1]))
    derivative = state_derivative(params, state, hover_control(params))

    np.testing.assert_allclose(derivative[10:13], 0.0, atol=1e-8)
    assert float(jnp.linalg.norm(derivative[6:10])) > 0.0


def test_angular_dynamics_authority_scales_rotation_but_not_translation() -> None:
    params = true_parameters()
    half = with_angular_dynamics_authority(params, 0.5)
    state = jnp.asarray(resting_state()).at[10:13].set(jnp.asarray([0.2, -0.3, 0.1]))
    motors = hover_control(params) + 0.02 * MOTOR_MIXER[0]

    full_derivative = state_derivative(params, state, motors)
    half_derivative = state_derivative(half, state, motors)

    np.testing.assert_allclose(half_derivative[:10], full_derivative[:10])
    np.testing.assert_allclose(
        half_derivative[10:13], 0.5 * full_derivative[10:13], rtol=1e-6
    )


def test_angular_dynamics_authority_is_bounded() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        with_angular_dynamics_authority(true_parameters(), 1.1)


def test_zero_initialized_residual_matches_structured_model() -> None:
    structured = true_parameters()
    residual = initial_residual_parameters(structured)
    state = jnp.asarray(resting_state())
    controls = jnp.tile(hover_control(structured), (5, 1))

    structured_states = rollout(structured, state, controls, 0.02)
    residual_states = rollout(residual, state, controls, 0.02)

    np.testing.assert_allclose(residual_states, structured_states, atol=1e-7)


def test_instantaneous_history_residual_is_exact_nested_ablation() -> None:
    structured = true_parameters()
    residual = initial_residual_parameters(structured, hidden_units=2)
    residual = residual._replace(
        hidden_weights=jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        output_weights=jnp.asarray(
            [
                [0.3, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.2],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    observer = with_instantaneous_residual_response(
        history_residual_from_residual(residual)
    )
    state = jnp.asarray(resting_state()).at[3].set(0.8).at[10].set(0.3)
    controls = jnp.tile(hover_control(structured), (4, 1))

    expected = rollout(residual, state, controls, 0.02)
    actual = rollout(observer, state, controls, 0.02)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_history_residual_latent_depends_on_past_measured_state() -> None:
    base = true_parameters()
    residual = initial_residual_parameters(base, hidden_units=1)
    residual = residual._replace(
        hidden_weights=jnp.zeros_like(residual.hidden_weights).at[0, 0].set(2.0),
        output_weights=jnp.zeros_like(residual.output_weights).at[0, 0].set(1.0),
    )
    observer = history_residual_from_residual(
        residual, response_time_constant_s=(0.1, 0.1)
    )
    control_history = jnp.tile(hover_control(base), (2, 1))
    positive = jnp.tile(jnp.asarray(resting_state()), (3, 1)).at[:2, 3].set(1.0)
    negative = jnp.tile(jnp.asarray(resting_state()), (3, 1)).at[:2, 3].set(-1.0)

    positive_latent = latent_state_after_history(
        observer, positive, control_history, 0.02
    )
    negative_latent = latent_state_after_history(
        observer, negative, control_history, 0.02
    )
    positive_next, _ = step_with_latent(
        observer,
        positive[-1],
        positive_latent,
        control_history[-1],
        0.02,
    )
    negative_next, _ = step_with_latent(
        observer,
        negative[-1],
        negative_latent,
        control_history[-1],
        0.02,
    )

    assert float(positive_latent[-6]) < 0.0
    assert float(negative_latent[-6]) > 0.0
    assert float(positive_next[3]) < float(negative_next[3])


def test_history_residual_ignores_padded_history_intervals() -> None:
    base = true_parameters()
    observer = history_residual_from_residual(
        initial_residual_parameters(base, hidden_units=1),
        response_time_constant_s=(0.1, 0.1),
    )
    state_history = jnp.tile(jnp.asarray(resting_state()), (3, 1))
    control_history = jnp.tile(hover_control(base) + 0.1, (2, 1))

    latent = latent_state_after_history(
        observer,
        state_history,
        control_history,
        0.02,
        history_valid=jnp.asarray([False, False]),
    )

    np.testing.assert_allclose(latent[-6:], 0.0, atol=1e-8)
    np.testing.assert_allclose(latent[:4], control_history[0], atol=1e-8)


def test_estimated_wind_only_conditions_linear_residual() -> None:
    base = true_parameters()
    residual = initial_residual_parameters(base, hidden_units=1, exogenous_size=2)
    residual = residual._replace(
        hidden_weights=residual.hidden_weights.at[0, -2].set(5.0),
        output_weights=jnp.ones_like(residual.output_weights),
    )
    state = jnp.asarray(resting_state())
    control = hover_control(base)
    roles = ("estimated_wind_north", "estimated_wind_west")

    calm = state_derivative(
        residual,
        state,
        control,
        exogenous=jnp.zeros(2),
        exogenous_roles=roles,
    )
    windy = state_derivative(
        residual,
        state,
        control,
        exogenous=jnp.asarray([1.0, 0.0]),
        exogenous_roles=roles,
    )

    assert float(jnp.linalg.norm(windy[3:6] - calm[3:6])) > 0.1
    np.testing.assert_allclose(windy[10:13], calm[10:13], atol=1e-7)
