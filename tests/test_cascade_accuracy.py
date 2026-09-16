"""Matched prediction semantics, gradients and failure accounting for the benchmark."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def example(monkeypatch):
    pytest.importorskip("cascade")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("cascade_accuracy")


def test_observation_contract_and_reference_velocity(example):
    from glassbox.core.geometry import quaternion_to_rotation_matrices

    states = np.zeros((2, 13))
    states[:, 6] = 1
    states[:, 3:6] = [[1, 2, 3], [4, 5, 6]]
    states[:, 10:13] = [[7, 8, 9], [10, 11, 12]]
    actual = np.asarray(example.observation(states))
    np.testing.assert_array_equal(actual[:, :3], states[:, 3:6])
    np.testing.assert_array_equal(actual[:, 3:6], states[:, 10:13])
    np.testing.assert_allclose(
        actual[:, 6:], quaternion_to_rotation_matrices(states[:, 6:10]).reshape(2, 9)
    )
    derivative = jax.jacfwd(lambda t: example.reference(t)[0])(jnp.asarray(1.0))
    np.testing.assert_allclose(
        derivative, example.reference(jnp.asarray(1.0))[1], rtol=1e-6
    )


def test_known_reset_and_command_replay_match_public_plant(example):
    import cascade
    from cascade.canonical import rigid_body_to_canonical

    plant = cascade.Plant(
        cascade.skywalker_x8_spec(), cascade.PlantConfig(control_frequency_hz=20)
    )
    oracle = example.Oracle(plant.model)
    state = np.array([0, 0, 100, 18, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    command = np.array([0.4, 0.0, 0.1])
    sample = plant.reset(state, applied_control=command)
    internal = oracle.reset(sample.state, command)
    commands = np.array([[0.42, 0.02, 0.08], [0.44, -0.01, 0.11], [0.4, 0.0, 0.1]])
    predicted = np.asarray(oracle.predict(internal, commands))
    for i, u in enumerate(commands):
        sample = plant.step(u)
        internal = oracle.advance(internal, u)
        np.testing.assert_allclose(
            sample.state,
            rigid_body_to_canonical(internal.rigid_body),
            atol=2e-5,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            predicted[i], example.observation(sample.state), atol=2e-5, rtol=1e-6
        )

    # Check the differentiated dynamics against independent finite differences.
    def forecast(u):
        return oracle.predict(internal, jnp.broadcast_to(u, (5, 3)))[-1, :6]

    jacobian = np.asarray(jax.jacfwd(forecast)(jnp.asarray(command)))
    epsilon = 1e-3
    numerical = np.column_stack(
        [
            (np.asarray(forecast(command + d)) - np.asarray(forecast(command - d)))
            / (2 * epsilon)
            for d in epsilon * np.eye(3)
        ]
    )
    np.testing.assert_allclose(jacobian, numerical, atol=3e-3, rtol=3e-3)


def test_pure_predictors_are_not_contaminated_by_unused_nonfinite_branch(example):
    class Exact:
        def predict(self, state, commands):
            return jnp.broadcast_to(state, (len(commands), 15))

    class Learned:
        def predict(self, x, u, commands):
            return jnp.broadcast_to(x[-1], (len(commands), 15))

    predict = jax.jit(example.predictor(Exact(), Learned()))
    commands = jnp.zeros((5, 3))
    x, up = jnp.zeros((3, 15)), jnp.zeros((2, 3))
    exact = predict(jnp.zeros(15), x + jnp.nan, up, commands, 0.0, 0.3)
    learned = predict(jnp.full(15, jnp.nan), x, up, commands, 1.0, 0.0)
    np.testing.assert_allclose(exact[:, 2], np.arange(1, 6) * 0.3 / 5, atol=1e-7)
    np.testing.assert_array_equal(learned, 0)


def test_truncated_trial_counts_remaining_time_as_failure(example):
    import cascade

    plant = cascade.Plant(
        cascade.skywalker_x8_spec(), cascade.PlantConfig(control_frequency_hz=20)
    )
    oracle = example.Oracle(plant.model)
    state = np.array([0, 0, 100, 18, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    command = np.array([0.4, 0.0, 0.1])

    def failed(*args):
        return jnp.full(3, jnp.nan), 0.0, jnp.inf, jnp.zeros((5, 15))

    result, arrays = example.run_trial(
        plant,
        oracle,
        failed,
        state,
        command,
        seed=0,
        mix=1.0,
        bias=0.0,
        alternating=False,
        duration_s=3.0,
    )
    assert result["completed_steps"] == 0
    assert result["within_tolerance_fraction"] == 0
    assert not result["meets_requirement"]
    assert result["tracking_rmse_m"] == [None, None]
    assert arrays["errors"].shape == (60, 2) and np.isinf(arrays["errors"]).all()


def test_recording_identity_survives_adapter(example):
    state = np.zeros((10, 13))
    state[:, 6] = 1
    record = SimpleNamespace(
        states=state, controls=np.zeros((9, 3)), labels={"source_group": "reserved-80"}
    )
    collection = example.as_collection([record])
    assert collection.segments[0].recording_id == "reserved-80"
    assert collection.input_channels[1] == "aileron [rad,requested surface angle]"
