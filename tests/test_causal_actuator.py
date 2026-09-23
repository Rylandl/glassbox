"""Numerical parity and derivative contract for the selected causal equation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox._causal_actuator import CausalActuatorModel
from glassbox._causal_fit import BackgroundFit, fit_episode, fit_segments

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["fixedwing-80-row31", "highspin-085-row150"])
def test_frozen_research_forecast_and_immutable_roundtrip(name, tmp_path):
    """A public core must preserve the selected fit's saved physical rollout."""
    with np.load(FIXTURES / f"causal-{name}.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    model = CausalActuatorModel(
        float(data["dt_s"]),
        data["q"],
        data["coeff"],
        data["inertia"],
        data["force"],
        data["torque"],
    )
    with jax.enable_x64(True):
        forecast = np.asarray(
            model.forecast(data["start"], data["past_inputs"], data["future_inputs"])
        )
    np.testing.assert_allclose(forecast, data["forecast"], rtol=0, atol=2e-11)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = CausalActuatorModel.load(path)
    assert loaded.fingerprint == model.fingerprint
    with jax.enable_x64(True):
        np.testing.assert_array_equal(
            loaded.forecast(data["start"], data["past_inputs"], data["future_inputs"]),
            forecast,
        )
    data["force"][0, 0] += 100
    assert loaded.fingerprint == model.fingerprint


def test_command_jacobian_matches_symmetric_perturbation():
    with np.load(FIXTURES / "causal-highspin-085-row150.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    model = CausalActuatorModel(
        float(data["dt_s"]),
        data["q"],
        data["coeff"],
        data["inertia"],
        data["force"],
        data["torque"],
    )
    start, past, command = data["start"], data["past_inputs"], data["future_inputs"][0]
    with jax.enable_x64(True):
        response = jax.jacfwd(lambda u: model.forecast(start, past, u[None])[0, 3:6])(
            jnp.asarray(command)
        )
        epsilon = 1e-5
        difference = np.column_stack(
            [
                (
                    np.asarray(
                        model.forecast(start, past, (command + epsilon * axis)[None])
                    )[0, 3:6]
                    - np.asarray(
                        model.forecast(start, past, (command - epsilon * axis)[None])
                    )[0, 3:6]
                )
                / (2 * epsilon)
                for axis in np.eye(len(command))
            ]
        )
    assert np.isfinite(response).all()
    assert np.linalg.norm(response) > 0.1
    np.testing.assert_allclose(response, difference, rtol=1e-5, atol=1e-7)


def test_history_derivative_survives_internal_length_bucketing():
    with np.load(FIXTURES / "causal-highspin-085-row150.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    model = CausalActuatorModel(
        float(data["dt_s"]),
        data["q"],
        data["coeff"],
        data["inertia"],
        data["force"],
        data["torque"],
    )
    history = data["past_inputs"][:7]
    future = data["future_inputs"][:1]
    with jax.enable_x64(True):
        derivative = jax.jacfwd(
            lambda past: model.forecast(data["start"], past, future)[0, 3:6]
        )(jnp.asarray(history))
    assert np.isfinite(derivative).all()
    assert np.linalg.norm(derivative) > 0


def test_nonphysical_inertia_is_rejected():
    with np.load(FIXTURES / "causal-fixedwing-80-row31.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    data["inertia"] = -np.eye(3)
    with pytest.raises(ValueError, match="invalid causal"):
        CausalActuatorModel(
            float(data["dt_s"]),
            data["q"],
            data["coeff"],
            data["inertia"],
            data["force"],
            data["torque"],
        )


def test_optimizer_log_bound_roundoff_is_accepted():
    with np.load(FIXTURES / "causal-fixedwing-80-row31.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    for q in (np.exp(np.log(1e-7)), np.exp(np.log(10.0))):
        model = CausalActuatorModel(
            float(data["dt_s"]),
            q,
            data["coeff"],
            data["inertia"],
            data["force"],
            data["torque"],
        )
        assert model.q > 0


def test_short_fixedwing_fit_does_not_use_a_large_force_cancellation():
    """Frozen row-21 fit must remain usable at row 31 before another revision lands."""
    with np.load(FIXTURES / "causal-fixedwing-80-early.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    model, _ = fit_episode(
        data["training_states"], data["training_commands"], float(data["dt_s"])
    )
    with jax.enable_x64(True):
        forecast = np.asarray(
            model.forecast(data["start"], data["past_inputs"], data["future_inputs"])
        )
    assert np.linalg.norm(model.force[0]) < 2.0
    assert np.linalg.norm(forecast[-1, :3] - data["target_velocity"]) < 0.7


def _analytic_segment(reverse=False):
    dt_s, count = 0.02, 40
    time = np.arange(count) * dt_s
    commands = np.stack(
        (0.3 + 0.1 * np.sin(2 * time), 0.2 + 0.1 * np.cos(3 * time)), axis=-1
    )
    if reverse:
        commands = commands[::-1].copy()
    states = np.zeros((count + 1, 15))
    states[:, 6:] = np.eye(3).ravel()
    for row, command in enumerate(commands):
        states[row + 1, :3] = states[row, :3] + dt_s * np.array(
            [2 * command[0], command[1], -9.80665]
        )
    return states, commands, dt_s


def test_one_recipe_fits_multiple_reset_segments():
    states, commands, dt_s = _analytic_segment()
    other_states, other_commands, _ = _analytic_segment(reverse=True)
    single, _ = fit_episode(states, commands, dt_s)
    one_segment, _ = fit_segments(((states, commands, 0),), dt_s)
    joint, report = fit_segments(
        ((states, commands, 0), (other_states, other_commands, 0)), dt_s
    )
    assert single.fingerprint == one_segment.fingerprint
    assert joint.input_count == 2
    assert report["fit_segments"] == 2
    assert report["completed_transitions"] == 80


def test_background_fit_publishes_immutable_revisions():
    states, commands, dt_s = _analytic_segment()
    session = BackgroundFit(states[:31], commands[:30], dt_s)
    try:
        old = session.model
        old_fingerprint = old.fingerprint
        for row in (30, 31):
            session.observe(row, commands[row], states[row + 1])
        assert session.cursor == 32
        for _ in range(2):
            session.wait_for_publication(timeout=3)
        assert session.published_cursor == 32
        assert session.model.fingerprint != old_fingerprint
        assert old.fingerprint == old_fingerprint
        with pytest.raises(ValueError, match="observation"):
            session.observe(32, np.array([np.nan, 0.0]), states[33])
        assert session.cursor == 32
    finally:
        session.close()
