"""Numerical parity and derivative contract for the selected causal equation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox._causal_actuator import CausalActuatorModel

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
        float(data["dt_s"]), data["q"], data["coeff"], data["inertia"],
        data["force"], data["torque"],
    )
    start, past, command = data["start"], data["past_inputs"], data["future_inputs"][0]
    with jax.enable_x64(True):
        response = jax.jacfwd(
            lambda u: model.forecast(start, past, u[None])[0, 3:6]
        )(jnp.asarray(command))
        epsilon = 1e-5
        difference = np.column_stack(
            [
                (
                    np.asarray(model.forecast(start, past, (command + epsilon * axis)[None]))[0, 3:6]
                    - np.asarray(model.forecast(start, past, (command - epsilon * axis)[None]))[0, 3:6]
                ) / (2 * epsilon)
                for axis in np.eye(len(command))
            ]
        )
    assert np.isfinite(response).all()
    assert np.linalg.norm(response) > 0.1
    np.testing.assert_allclose(response, difference, rtol=1e-5, atol=1e-7)


def test_nonphysical_inertia_is_rejected():
    with np.load(FIXTURES / "causal-fixedwing-80-row31.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    data["inertia"] = -np.eye(3)
    with pytest.raises(ValueError, match="invalid causal"):
        CausalActuatorModel(
            float(data["dt_s"]), data["q"], data["coeff"], data["inertia"],
            data["force"], data["torque"],
        )
