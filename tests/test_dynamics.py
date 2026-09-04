import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.dynamics import (
    MOTOR_MIXER,
    DynamicsParams,
    FixedWingDynamicsParams,
    hover_control,
    initial_residual_parameters,
    rollout,
    rollout_with_latent,
    state_derivative,
    step,
    step_with_latent,
    with_thrust_command_offset,
)
from glassbox.core.metrics import predict, rollout_metrics
from glassbox.core.synthetic import resting_state, true_parameters


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


def test_true_model_has_zero_attitude_rollout_error(quadrotor_flight) -> None:
    trajectory = quadrotor_flight(4, 0.2)

    metrics = rollout_metrics(predict(true_parameters(), trajectory))

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


def test_multirotor_latent_state_is_exactly_the_applied_controls() -> None:
    params = true_parameters()
    state = jnp.asarray(resting_state())
    applied = jnp.asarray([0.4, 0.4, 0.4, 0.4])
    command = jnp.asarray([0.8, 0.2, 0.2, 0.8])

    next_state, latent = step_with_latent(params, state, applied, command, 0.01)

    assert latent.shape == (4,)
    decay = float(jnp.exp(-0.01 / params.physical()["motor_time_constant"]))
    np.testing.assert_allclose(
        latent, command + (applied - command) * decay, rtol=1e-6, atol=1e-7
    )
    assert bool(jnp.all(jnp.isfinite(next_state)))

    with pytest.raises(ValueError, match="one applied value per control channel"):
        step_with_latent(
            params,
            state,
            jnp.concatenate((applied, jnp.zeros(3))),
            command,
            0.01,
        )


def test_control_generated_torque_follows_the_applied_control_with_no_memory() -> None:
    params = true_parameters()
    state = jnp.asarray(resting_state())
    hover = hover_control(params)
    command = hover + 0.02 * MOTOR_MIXER[0]

    applied = step_with_latent(params, state, hover, command, 0.02)[1]
    derivative = state_derivative(params, state, applied)
    expected = params.physical()["angular_control_matrix"] @ (MOTOR_MIXER @ applied)

    np.testing.assert_allclose(derivative[10:13], expected, rtol=1e-6, atol=1e-7)


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


def test_zero_initialized_residual_matches_structured_model() -> None:
    structured = true_parameters()
    residual = initial_residual_parameters(structured)
    state = jnp.asarray(resting_state())
    controls = jnp.tile(hover_control(structured), (5, 1))

    structured_states = rollout(structured, state, controls, 0.02)
    residual_states = rollout(residual, state, controls, 0.02)

    np.testing.assert_allclose(residual_states, structured_states, atol=1e-7)


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


def test_rollout_applies_per_step_exogenous_inputs(fixedwing_flight) -> None:
    from glassbox.core.dynamics import WIND_EXOGENOUS_ROLES, rollout_with_latent
    from glassbox.core.fixedwing_synthetic import (
        true_fixed_wing_parameters,
    )

    trajectory = fixedwing_flight(1, 0.3)
    params = true_fixed_wing_parameters()
    controls = jnp.asarray(trajectory.controls)
    steps = controls.shape[0]
    wind = 2.0 * np.sin(np.linspace(0.0, 3.0, 2 * steps)).reshape(steps, 2)
    dt_s = trajectory.nominal_dt_s
    roles = trajectory.spec.control_roles

    per_step_states, _ = rollout_with_latent(
        params,
        jnp.asarray(trajectory.states[0]),
        controls,
        dt_s,
        None,
        roles,
        jnp.asarray(wind),
        WIND_EXOGENOUS_ROLES,
    )

    # Chain single-step rollouts, each holding that step's wind vector.
    state = jnp.asarray(trajectory.states[0])
    latent = None
    for index in range(steps):
        states, applied = rollout_with_latent(
            params,
            state,
            controls[index : index + 1],
            dt_s,
            latent,
            roles,
            jnp.asarray(wind[index]),
            WIND_EXOGENOUS_ROLES,
        )
        state, latent = states[-1], applied[-1]
    np.testing.assert_allclose(per_step_states[-1], state, rtol=1e-5, atol=1e-6)

    held_states, _ = rollout_with_latent(
        params,
        jnp.asarray(trajectory.states[0]),
        controls,
        dt_s,
        None,
        roles,
        jnp.asarray(wind[0]),
        WIND_EXOGENOUS_ROLES,
    )
    broadcast_states, _ = rollout_with_latent(
        params,
        jnp.asarray(trajectory.states[0]),
        controls,
        dt_s,
        None,
        roles,
        jnp.asarray(np.tile(wind[0], (steps, 1))),
        WIND_EXOGENOUS_ROLES,
    )
    np.testing.assert_allclose(held_states, broadcast_states, rtol=1e-6, atol=1e-7)
    assert not np.allclose(per_step_states[-1], held_states[-1], atol=1e-4)

    with pytest.raises(ValueError, match="one row per control step"):
        rollout_with_latent(
            params,
            jnp.asarray(trajectory.states[0]),
            controls,
            dt_s,
            None,
            roles,
            jnp.asarray(wind[:-1]),
            WIND_EXOGENOUS_ROLES,
        )


def test_structured_parameters_carry_no_rotational_response_coordinate() -> None:
    from glassbox.belief.information import estimable_structured_parameters
    from glassbox.core.dynamics import structured_parameter_names

    names = structured_parameter_names(true_parameters())

    assert len(names) == 19
    assert not any("angular_response" in name for name in names)
    assert estimable_structured_parameters(true_parameters()).shape == (19,)


def test_from_physical_rejects_out_of_range_inputs() -> None:
    from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters

    with pytest.raises(
        ValueError, match="thrust_accel must be finite and strictly positive"
    ):
        DynamicsParams.from_physical(
            thrust_accel=-5.4,
            angular_accel=(18.0, 16.5, 7.5),
            linear_drag=0.18,
            angular_drag=(0.24, 0.21, 0.13),
            motor_time_constant=0.08,
        )
    with pytest.raises(ValueError, match="angular_control_cross_coupling"):
        DynamicsParams.from_physical(
            thrust_accel=5.4,
            angular_accel=(18.0, 16.5, 7.5),
            linear_drag=0.18,
            angular_drag=(0.24, 0.21, 0.13),
            motor_time_constant=0.08,
            angular_control_cross_coupling=(
                (0.0, 0.9, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
    physical = true_fixed_wing_parameters().physical()
    kwargs = {
        name: (
            tuple(float(v) for v in np.asarray(value))
            if np.ndim(value) > 0
            else float(value)
        )
        for name, value in physical.items()
    }
    with pytest.raises(ValueError, match="surface_trim must lie strictly within"):
        FixedWingDynamicsParams.from_physical(
            **{**kwargs, "surface_trim": (1.5, 0.0, 0.0)}
        )
    with pytest.raises(ValueError, match="lift_accel_per_speed_sq must be finite"):
        FixedWingDynamicsParams.from_physical(
            **{**kwargs, "lift_accel_per_speed_sq": float("nan")}
        )
    with pytest.raises(ValueError, match="thrust_command_offset must be finite"):
        DynamicsParams.from_physical(
            thrust_accel=5.4,
            thrust_command_offset=0.3,
            angular_accel=(18.0, 16.5, 7.5),
            linear_drag=0.18,
            angular_drag=(0.24, 0.21, 0.13),
            motor_time_constant=0.08,
        )
    with pytest.raises(ValueError, match="actuator_time_constant must be finite"):
        FixedWingDynamicsParams.from_physical(
            **{**kwargs, "actuator_time_constant": -0.05}
        )
