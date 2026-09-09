"""Check immediate command authority and exact middle-waveform preservation."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_feedback_suffix")


def test_only_head_and_tail_are_free_and_middle_is_dynamic(experiment):
    class Mapping:
        _commands_from_normalized = staticmethod(lambda x: 0.5 * (x + 1))
        rollout_commands = staticmethod(lambda commands, *args: commands)

    def waveform(blocks, values):
        return experiment.FeedbackSuffixPlan.rollout(
            Mapping(), blocks, None, None, None, values
        )

    blocks = jnp.linspace(-1.0, 1.0, 40).reshape(10, 4)
    values = experiment.SuffixValues(None, None, None, jnp.full((20, 4), 0.37))
    kernel = jax.jit(waveform)
    first = kernel(blocks, values)
    second = kernel(blocks, values._replace(frozen_commands=jnp.full((20, 4), 0.63)))
    free = np.asarray(experiment.DESIGN["free_command_indices"])
    assert first.shape == (30, 4)
    np.testing.assert_array_equal(first[free], (blocks + 1) / 2)
    np.testing.assert_array_equal(first[4:24], values.frozen_commands)
    np.testing.assert_array_equal(second[4:24], jnp.full((20, 4), 0.63))
    np.testing.assert_array_equal(first[free], second[free])
    # Test Jacobian sparsity, including immediate control and exact endpoints.
    jacobian = np.asarray(jax.jacfwd(waveform)(blocks, values)).reshape(120, 40)
    expected = np.zeros((120, 40))
    indices = np.asarray(free)[:, None] * 4 + np.arange(4)[None, :]
    expected[indices.ravel(), np.arange(40)] = 0.5
    np.testing.assert_array_equal(jacobian, expected)


def test_shifted_seed_reconstructs_without_middle_projection(experiment):
    solver = object.__new__(experiment.FeedbackSuffixSolver)
    # Binary fractions make exact reconstruction meaningful even in float32.
    seed = jnp.asarray(np.arange(120).reshape(30, 4) % 17 / 16)
    solver.seed_commands = jnp.concatenate((seed[1:], seed[-1:]))
    solver.model = SimpleNamespace(
        command_minimum=jnp.zeros(4), command_maximum=jnp.ones(4)
    )
    blocks = solver._cold_blocks(jnp.zeros(4))
    assert blocks.shape == (10, 4)
    rebuilt = jnp.concatenate(
        ((blocks[:4] + 1) / 2, solver.seed_commands[4:24], (blocks[4:] + 1) / 2)
    )
    np.testing.assert_array_equal(rebuilt, solver.seed_commands)
    assert solver._warm_blocks(object()) is None


def test_seed_replacement_refreshes_middle_and_invalid_input_is_atomic(experiment):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Model:
        values: object
        command_size: int = 4

        @property
        def command_minimum(self):
            return jnp.zeros(4)

        @property
        def command_maximum(self):
            return jnp.ones(4)

    solver = object.__new__(experiment.FeedbackSuffixSolver)
    solver.model = Model(experiment.SuffixValues(None, None, None, None))
    solver.counts = {"evaluate": 5}
    original = jnp.linspace(0.0, 1.0, 120).reshape(30, 4)
    solver.set_seed(original)
    np.testing.assert_array_equal(solver.model.values.frozen_commands, original[4:24])
    assert not solver.counts
    updated = 1 - original
    solver.set_seed(updated)
    np.testing.assert_array_equal(solver.model.values.frozen_commands, updated[4:24])
    with pytest.raises(ValueError):
        solver.set_seed(jnp.full((30, 4), 1.01))
    np.testing.assert_array_equal(solver.seed_commands, updated)
    np.testing.assert_array_equal(solver.model.values.frozen_commands, updated[4:24])


@pytest.mark.parametrize("mismatch", [None, "latent", "cost"])
def test_independent_check_rejects_inconsistent_returned_outputs(experiment, mismatch):
    from glassbox.control.plan import NonlinearFeasibility, Prediction, SolveStatus

    commands = jnp.full((30, 4), 0.5)
    states = jnp.zeros((31, 13))
    latent = jnp.full((31, 4), 0.5)
    result = SimpleNamespace(
        status=SolveStatus.STALLED,
        message="test",
        command_usable=True,
        deadline_met=None,
        nonlinear_feasibility=NonlinearFeasibility(1, 0.0, 1e-6),
        command=commands[0],
        predicted_commands=commands,
        predicted_states=states,
        predicted_latent_states=latent + (0.01 if mismatch == "latent" else 0),
        diagnostics=SimpleNamespace(final_objective=2.0 if mismatch == "cost" else 1.0),
    )
    solver = SimpleNamespace(seed_commands=commands, counts={}, reports=[])
    prediction = Prediction(
        states, jnp.zeros((30, 12, 12)), commands, latent, jnp.zeros((30, 0))
    )

    def check(*_):
        return prediction, jnp.array(1.0), jnp.array([0.1])

    def describe():
        return experiment.describe(
            solver, result, states[0], latent[0], commands[0], check, 0
        )

    if mismatch is None:
        assert describe()["exact_frozen_middle"]
    else:
        with pytest.raises(AssertionError):
            describe()
