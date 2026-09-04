from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from glassbox.belief.belief import DynamicsBelief
from glassbox.belief.belief_io import save_dynamics_belief
from glassbox.core.data import Channel
from glassbox.core.dynamics import initial_residual_parameters
from glassbox.core.model import (
    DirectActuationMap,
    ExecutableModel,
    NonActionableModelError,
    runtime_spec_from_trajectory,
)
from glassbox.core.synthetic import true_parameters
from glassbox.io.nanodrone_reference import nanodrone_trajectory_spec


def _write_model(params, path, *, input_spec, runtime_spec) -> None:
    """Write the model as the point belief the library now writes."""

    save_dynamics_belief(
        DynamicsBelief(
            model=ExecutableModel(params, input_spec, runtime_spec),
        ),
        path,
    )


def test_runtime_model_loads_timing_bounds_and_latent_state(
    tmp_path, quadrotor_trajectory_seed2_dur0_2s
) -> None:
    trajectory = quadrotor_trajectory_seed2_dur0_2s
    path = tmp_path / "model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )

    runtime = ExecutableModel.load(path)
    command = jnp.full(4, 0.4)
    latent = runtime.initial_latent_state(command)
    next_state, next_latent = runtime.transition(
        jnp.asarray(trajectory.states[0]), latent, command
    )
    irregular_state, irregular_latent = runtime.transition_at_interval(
        jnp.asarray(trajectory.states[0]),
        latent,
        command,
        1.6 * trajectory.nominal_dt_s,
    )

    assert runtime.runtime_spec.sample_period_s == pytest.approx(
        trajectory.nominal_dt_s
    )
    assert runtime.command_size == 4
    assert runtime.latent_size == 4
    assert latent.shape == (4,)
    assert next_state.shape == (13,)
    assert next_latent.shape == (4,)
    assert irregular_state.shape == (13,)
    assert irregular_latent.shape == (4,)
    assert np.all(np.isfinite(next_state))
    assert np.all(np.isfinite(irregular_state))
    assert runtime.validity_utilization(next_state).shape == (6,)
    np.testing.assert_allclose(runtime.command_minimum, 0.0)
    np.testing.assert_allclose(runtime.command_maximum, 1.0)

    with pytest.raises(ValueError, match="finite and positive"):
        runtime.transition_at_interval(
            jnp.asarray(trajectory.states[0]), latent, command, 0.0
        )


def test_transition_enforces_declared_command_bounds(
    tmp_path, quadrotor_trajectory_seed2_dur0_2s
) -> None:
    trajectory = quadrotor_trajectory_seed2_dur0_2s
    path = tmp_path / "model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )
    runtime = ExecutableModel.load(path)
    state = jnp.asarray(trajectory.states[0])
    command = jnp.full(4, 0.4)
    latent = runtime.initial_latent_state(command)
    channel_name = trajectory.spec.control_names[1]

    unbounded = command.at[1].set(7.5)
    with pytest.raises(ValueError, match=f"{channel_name!r}=7.5 outside"):
        runtime.transition(state, latent, unbounded)
    with pytest.raises(ValueError, match="declared channel bounds"):
        runtime.transition_at_interval(
            state,
            latent,
            command.at[1].set(-0.5),
            trajectory.nominal_dt_s,
        )

    # A command 1e-9 outside the bound is rounding slack: it is clipped onto
    # the bound and executes as if it had been supplied there. The value is
    # carried in float64 because float32 could not represent the violation.
    slack = np.full(4, 0.4)
    slack[1] = 1.0 + 1e-9
    exact = np.full(4, 0.4)
    exact[1] = 1.0
    clipped_state, clipped_latent = runtime.transition(state, latent, slack)
    bounded_state, bounded_latent = runtime.transition(state, latent, exact)

    np.testing.assert_array_equal(clipped_state, bounded_state)
    np.testing.assert_array_equal(clipped_latent, bounded_latent)

    # The tolerance is a fraction of the channel span, not an open door.
    beyond = np.full(4, 0.4)
    beyond[1] = 1.0 + 1e-5
    with pytest.raises(ValueError, match="declared channel bounds"):
        runtime.transition(state, latent, beyond)


def test_traced_commands_stay_the_caller_s_contract(
    tmp_path, quadrotor_trajectory_seed2_dur0_2s
) -> None:
    trajectory = quadrotor_trajectory_seed2_dur0_2s
    path = tmp_path / "model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )
    runtime = ExecutableModel.load(path)
    state = jnp.asarray(trajectory.states[0])
    latent = runtime.initial_latent_state(jnp.full(4, 0.4))

    # Bound checks need concrete values, so a traced transition compiles: NMPC
    # clips its commands into the declared range before this point.
    compiled = jax.jit(
        lambda command: runtime.transition(state, latent, command)[0],
    )

    assert np.all(np.isfinite(compiled(jnp.full(4, 0.4))))


def test_measured_rotor_speed_model_loads_without_a_command_space(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    """Executable is not the same as actionable.

    A model whose inputs are measured rotor speeds still integrates, still
    reports validity, and still round-trips. It has no command space, and
    every method that needs one says so by name instead of the model refusing
    to exist.
    """

    trajectory = quadrotor_trajectory_seed0_dur0_1s
    path = tmp_path / "nanodrone_model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=nanodrone_trajectory_spec(),
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )

    model = ExecutableModel.load(path)

    assert model.actuation is None
    assert model.latent_size == 4
    assert np.all(
        np.isfinite(
            np.asarray(model.validity_utilization(jnp.asarray(trajectory.states[0])))
        )
    )
    for command_space in (
        lambda: model.command_size,
        lambda: model.command_minimum,
        lambda: model.command_maximum,
        lambda: model.initial_latent_state(jnp.zeros(4)),
    ):
        with pytest.raises(NonActionableModelError, match="no command space"):
            command_space()
    with pytest.raises(NonActionableModelError, match="command semantics"):
        DirectActuationMap(model.input_spec.controls)


@dataclass(frozen=True)
class SquaredSpeedActuation:
    command_channels: tuple[Channel, ...]
    model_control_size: int = 4

    def model_control(self, command: Array) -> Array:
        return jnp.square(command)


def test_explicit_actuation_map_can_bind_noncommand_model(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    trajectory = quadrotor_trajectory_seed0_dur0_1s
    path = tmp_path / "nanodrone_model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=nanodrone_trajectory_spec(),
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )
    command_channels = tuple(
        Channel(
            name=f"motor_{index}_command",
            role=f"motor_{index}_command",
            semantic="normalized_command",
            unit="1",
            kind="control",
            minimum=0.0,
            maximum=1.0,
        )
        for index in range(4)
    )

    runtime = ExecutableModel.load(
        path,
        actuation=SquaredSpeedActuation(command_channels),
    )

    np.testing.assert_allclose(
        runtime.actuation.model_control(jnp.asarray([0.5] * 4)),
        0.25,
    )


@dataclass(frozen=True)
class InvalidActuationOutput:
    command_channels: tuple[Channel, ...]
    model_control_size: int = 4

    def model_control(self, command: Array) -> Array:
        return jnp.zeros(3)


@dataclass(frozen=True)
class InvalidActuationBoundary:
    command_channels: tuple[Channel, ...]
    model_control_size: int = 4

    def model_control(self, command: Array) -> Array:
        return jnp.where(command > 0.9, jnp.nan, command)


def test_runtime_validates_actual_actuation_map_output(
    tmp_path, quadrotor_trajectory_seed0_dur0_1s
) -> None:
    trajectory = quadrotor_trajectory_seed0_dur0_1s
    path = tmp_path / "model.json"
    _write_model(
        true_parameters(),
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )
    command_channels = tuple(trajectory.spec.controls)

    with pytest.raises(ValueError, match="produced shape"):
        ExecutableModel.load(
            path,
            actuation=InvalidActuationOutput(command_channels),
        )

    with pytest.raises(ValueError, match="non-finite"):
        ExecutableModel.load(
            path,
            actuation=InvalidActuationBoundary(command_channels),
        )


def test_runtime_supports_structured_residual_transition(
    tmp_path, quadrotor_flight
) -> None:
    trajectory = quadrotor_flight(1, 0.1)
    path = tmp_path / "residual.json"
    params = initial_residual_parameters(true_parameters(), hidden_units=3)
    _write_model(
        params,
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )

    runtime = ExecutableModel.load(path)
    command = jnp.asarray(trajectory.controls[0])
    latent = runtime.initial_latent_state(command)
    next_state, _ = runtime.transition(
        jnp.asarray(trajectory.states[0]), latent, command
    )

    assert np.all(np.isfinite(next_state))


def test_runtime_rebinds_only_compatible_finite_parameter_numerics(
    tmp_path, quadrotor_flight
) -> None:
    trajectory = quadrotor_flight(4, 0.1)
    path = tmp_path / "model.json"
    params = true_parameters()
    _write_model(
        params,
        path,
        input_spec=trajectory.spec,
        runtime_spec=runtime_spec_from_trajectory(trajectory),
    )
    runtime = ExecutableModel.load(path)
    rebound = runtime.rebind_parameters(
        params._replace(log_linear_drag=params.log_linear_drag + 0.1)
    )

    assert rebound is not runtime
    assert rebound.input_spec is runtime.input_spec
    assert rebound.runtime_spec is runtime.runtime_spec
    assert rebound.actuation is runtime.actuation
    assert rebound.params.log_linear_drag != runtime.params.log_linear_drag

    with pytest.raises(ValueError, match="shape changed"):
        runtime.rebind_parameters(params._replace(log_angular_accel=jnp.zeros(2)))
    with pytest.raises(ValueError, match="must be finite"):
        runtime.rebind_parameters(params._replace(log_linear_drag=jnp.asarray(np.nan)))


def test_direct_actuation_requires_complete_command_bounds() -> None:
    channel = Channel(
        name="command",
        role="command",
        semantic="normalized_command",
        unit="1",
        kind="control",
    )

    with pytest.raises(NonActionableModelError, match="finite bounds"):
        DirectActuationMap((channel,))
