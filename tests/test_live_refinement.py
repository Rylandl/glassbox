"""The example's plant seam preserves dynamics without exposing latent state."""

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.belief.belief import DynamicsBelief
from glassbox.core.dynamics import control_state_after_history, step_with_latent
from glassbox.core.fixedwing_synthetic import true_fixed_wing_parameters
from glassbox.core.model import ExecutableModel, runtime_spec_from_trajectory
from glassbox.core.synthetic import true_parameters


@pytest.mark.parametrize("family", ["multirotor", "fixedwing"])
def test_synthetic_plant_seam_preserves_the_original_stepping(
    family, monkeypatch, quadrotor_flight, fixedwing_flight
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    example = importlib.import_module("live_refinement")
    fixed = family == "fixedwing"
    flight = (fixedwing_flight if fixed else quadrotor_flight)(21, 0.2)
    params = true_fixed_wing_parameters() if fixed else true_parameters()
    belief = DynamicsBelief(
        ExecutableModel(params, flight.spec, runtime_spec_from_trajectory(flight))
    )
    plant = example.synthetic_plant(belief, params)
    state = plant.initial_state.copy()
    latent = control_state_after_history(
        params,
        jnp.asarray(plant.initial_command)[None],
        example.DT_S,
        flight.spec.control_roles,
    )
    original_step = jax.jit(
        lambda state, latent, command: step_with_latent(
            params, state, latent, command, example.DT_S, flight.spec.control_roles
        )
    )
    for index in range(12):
        command = plant.initial_command + 0.01 * np.sin(
            index + np.arange(len(plant.initial_command))
        )
        state, latent = original_step(state, latent, command)
        observed = plant.advance(command)
        np.testing.assert_array_equal(observed, state)
    # Consumer mutations of returned observations cannot alter the plant.
    observed[:] = 99
    state, latent = original_step(state, latent, plant.initial_command)
    np.testing.assert_array_equal(plant.advance(plant.initial_command), state)
    np.testing.assert_array_equal(
        plant.reference(np.array([0.0, 0.1])),
        example.reference_states(family, np.array([0.0, 0.1])),
    )
