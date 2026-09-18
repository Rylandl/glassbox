"""Pure fixture contracts: native Crazyflow dynamics are never invoked here."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.core.dynamics import MOTOR_MIXER, quaternion_to_rotation
from glassbox.experimental import two_simulator_crazyflow as module


@pytest.fixture
def protocol():
    return json.loads(
        (
            Path(__file__).parents[1] / "docs/harness/two-simulator-flight-v1.json"
        ).read_text()
    )


@pytest.fixture
def fixture(protocol, monkeypatch):
    pytest.importorskip("crazyflow")
    monkeypatch.setenv("SCIPY_ARRAY_API", "1")
    with jax.enable_x64(True):
        plant = module.Fixture(protocol)

        def forbidden_native_dynamics(**kwargs):
            raise AssertionError("pure unit test called native simulator dynamics")

        monkeypatch.setattr(plant, "_dynamics", forbidden_native_dynamics)
        yield plant


def test_configuration_and_motor_mapping(fixture):
    config = fixture.config()
    json.dumps(config, allow_nan=False)
    assert config["configuration_id"] == module.CONFIGURATION
    assert fixture.spec.vehicle.configuration_id == module.CONFIGURATION
    assert len(config["full_state_names"]) == 17
    assert len(config["pilot_target_names"]) == 7
    assert fixture.dt == 0.01
    np.testing.assert_array_equal(
        np.asarray(fixture.params["mixing_matrix"])[:, module.GB_FROM_CF], MOTOR_MIXER
    )
    np.testing.assert_allclose(
        fixture.lower, 0.012817578393224994 / 0.12, rtol=0, atol=1e-16
    )
    for channel, lower in zip(fixture.spec.channels, fixture.lower):
        assert channel.minimum == lower and channel.maximum == 1.0
    command = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    state = np.zeros(17)
    state[6] = 1.0
    state[13:] = fixture.command_to_rpm(command) / module.RPM_SCALE
    arrays = fixture._arrays([state], [], [], [], 0, 0)
    np.testing.assert_allclose(
        arrays["actuator_outputs"][0], command, rtol=0, atol=3e-16
    )


def test_setup_failure_is_explicit(protocol, monkeypatch):
    monkeypatch.setenv("SCIPY_ARRAY_API", "1")
    with jax.enable_x64(False), pytest.raises(module.FixtureSetupError) as captured:
        module.Fixture(protocol)
    assert captured.value.detail["stage"] == "runtime"
    with jax.enable_x64(True):
        protocol["generation"]["crazyflow"]["dt_s"] = 0.05
        with pytest.raises(module.FixtureSetupError, match="configuration"):
            module.Fixture(protocol)


def test_initial_draw_order_and_heading(fixture, protocol):
    cell = dict(protocol["cells"]["crazyflow"][0], heading_nwu_deg=90, speed_m_s=4)
    state, draws = fixture._initial(1200, cell)
    rng = np.random.Generator(np.random.PCG64(1200))
    velocity = rng.uniform(-0.5, 0.5, 3)
    tangent = rng.uniform(-0.2, 0.2, 3)
    rates = rng.uniform(-0.3, 0.3, 3)
    phases = rng.uniform(0, 2 * np.pi, 3)
    frequencies = rng.uniform(0.3, 0.65, 3)
    amplitude = rng.uniform((0.2, 0.5, 0.25), (0.55, 1.1, 0.65))
    np.testing.assert_array_equal(draws["velocity_jitter"], velocity)
    np.testing.assert_array_equal(draws["attitude_tangent"], tangent)
    np.testing.assert_array_equal(draws["phases"], phases)
    np.testing.assert_array_equal(draws["frequencies_hz"], frequencies)
    np.testing.assert_array_equal(draws["base_amplitudes_rad"], amplitude)
    np.testing.assert_array_equal(draws["amplitudes_rad"], amplitude * 0.5)
    np.testing.assert_array_equal(state[10:13], rates)
    np.testing.assert_allclose(
        state[3:6], [-velocity[1], 4 + velocity[0], velocity[2]], atol=2e-15
    )
    # Heading is a left/world composition, preserving the body-frame jitter.
    rotation = np.asarray(quaternion_to_rotation(jnp.asarray(state[6:10])))
    rotation0 = np.asarray(
        quaternion_to_rotation(module._rotation_exp(jnp.asarray(tangent)))
    )
    np.testing.assert_allclose(
        rotation, np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ rotation0, atol=5e-16
    )
    np.testing.assert_array_equal(state, fixture._initial(1200, cell)[0])


def test_integer_stage_wind_and_rk4_with_stub_derivative(fixture, monkeypatch):
    # Only a synthetic derivative is integrated; no Crazyflow execution.
    def derivative(state, command, wind):
        change = jnp.zeros(17).at[0].set(state[0]).at[3:6].set(wind)
        return change, 2.0 * wind

    monkeypatch.setattr(fixture, "_derivative", derivative)
    state = jnp.zeros(17).at[0].set(1.0).at[6].set(1.0)
    following, winds, forces = fixture._advance(
        state, jnp.ones(4), jnp.asarray(50), jnp.asarray(2)
    )
    ticks = 500 + 2 * np.arange(5)[:, None] + np.asarray([0, 1, 1, 2])
    times = ticks / 1000.0
    expected = 2 * np.sin(np.pi * (times - 0.5) / 2) ** 2
    expected[times <= 0.5] = 0
    np.testing.assert_allclose(winds[..., 0], expected, rtol=1e-13, atol=1e-18)
    np.testing.assert_array_equal(forces, 2 * winds)
    h = 0.002
    np.testing.assert_allclose(
        following[0], (1 + h + h * h / 2 + h**3 / 6 + h**4 / 24) ** 5, rtol=2e-15
    )
    assert np.linalg.norm(following[6:10]) == 1.0
    np.testing.assert_array_equal(
        module.wind_at_ticks(jnp.asarray([500, 2500]), 2), np.zeros((2, 3))
    )
    np.testing.assert_array_equal(
        module.wind_at_ticks(jnp.asarray([0, 1000]), 1), [[2, 0, 0], [2, 0, 0]]
    )


def test_shared_kernel_parent_branch_contract_with_stub(fixture, protocol, monkeypatch):
    calls = []

    def fake_advance(state, command, index, mode):
        calls.append((int(index), np.asarray(command).copy()))
        following = np.asarray(state).copy()
        following[0] += float(command[0]) * 0.01
        # Preserve an invalid tail as evidence, without filtering or stopping.
        if int(index) >= 2:
            following[2] = 0.25
        ticks = index * 10 + 2 * jnp.arange(5)[:, None] + jnp.asarray([0, 1, 1, 2])
        winds = module.wind_at_ticks(ticks, mode)
        return following, winds, jnp.zeros((5, 4, 3))

    monkeypatch.setattr(fixture, "_advance", fake_advance)
    fixture.steps = 4
    entry = protocol["recordings"][0]
    cell = protocol["cells"]["crazyflow"][0]
    parent, metadata = fixture.generate(entry, cell)
    branch, branch_metadata = fixture.branch(
        parent["full_states"][1], parent["commands"][1:], 1, cell
    )
    for name in (
        "time_s",
        "states",
        "full_states",
        "commands",
        "actuator_outputs",
        "wind",
        "integration_wind",
        "integration_force",
    ):
        np.testing.assert_array_equal(branch[name], parent[name][1:])
        assert parent[name].dtype == np.float64
    assert [index for index, _ in calls] == [0, 1, 2, 3, 1, 2, 3]
    assert parent["full_states"][-1, 2] == 0.25
    assert parent["pilot_targets"].shape == (4, 7)
    np.testing.assert_array_equal(
        parent["control_prefix"], np.full((50, 4), fixture.hover)
    )
    assert metadata["seed"] == entry["seed"] and metadata["feedback_pilot"]
    assert (
        branch_metadata["origin_index"] == 1 and not branch_metadata["feedback_pilot"]
    )
    json.dumps(metadata, allow_nan=False)
    json.dumps(branch_metadata, allow_nan=False)


def test_wind_extension_and_native_argument_mapping_with_mock(fixture, monkeypatch):
    seen = {}

    def dynamics(**kwargs):
        seen.update(kwargs)
        return jnp.zeros(3), jnp.zeros(4), jnp.zeros(3), jnp.zeros(3), jnp.zeros(4)

    monkeypatch.setattr(fixture, "_dynamics", dynamics)
    state = jnp.zeros(17).at[6].set(1.0).at[13:].set(0.8)
    command = jnp.asarray([0.2, 0.4, 0.6, 0.8])
    derivative, force = fixture._derivative(
        state, command, jnp.asarray([2.0, 0.0, 0.0])
    )
    np.testing.assert_array_equal(seen["quat"], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(seen["rotor_vel"], np.full(4, 16000.0))
    np.testing.assert_array_equal(seen["cmd"], fixture.command_to_rpm(command))
    np.testing.assert_allclose(force, [2 * 0.01471782, 0, 0], rtol=0, atol=1e-16)
    np.testing.assert_allclose(
        derivative[3:6], np.asarray(force) / 0.0319, rtol=0, atol=1e-15
    )


def test_programming_errors_escape_and_bad_branch_shapes(
    fixture, protocol, monkeypatch
):
    def broken(*args):
        raise NameError("deliberate harness bug")

    monkeypatch.setattr(fixture, "_advance", broken)
    with pytest.raises(NameError, match="harness bug"):
        fixture.branch(
            np.zeros(17), np.ones((1, 4)), 0, protocol["cells"]["crazyflow"][0]
        )
    with pytest.raises(ValueError, match="full state"):
        fixture.branch(np.zeros(13), np.ones((1, 4)), 0, {})
