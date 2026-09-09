"""Falsify suffix acceptance under lag and an unrepairable retained prefix."""

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
    return importlib.import_module("investigate_terminal_suffix")


class LagPlan:
    horizon_steps = 30
    uncertainty_complete = True
    command_size = 1
    exogenous_size = 0
    command_minimum = jnp.array([-1.0])
    command_maximum = jnp.array([1.0])
    values = None
    policy = None

    def __init__(self, limit=29.5):
        self.limit = limit

    def rollout_commands(self, commands, state, latent, exogenous, values):
        assert commands.shape == (30, 1)

        def step(carry, command):
            state, actuator = carry
            actuator = 0.9 * actuator + 0.1 * command
            state = state + actuator
            return (state, actuator), (state, actuator)

        _, (states, actuators) = jax.lax.scan(step, (state, latent), commands)
        return SimpleNamespace(
            mean_states=jnp.concatenate((state[None], states)),
            tangent_covariance=jnp.zeros((30, 1, 1)),
            latent_states=jnp.concatenate((latent[None], actuators)),
            commands=commands,
        )

    def optimization_terms(self, prediction, reference, previous, policy):
        # Constant uncertainty reserve is retained at every future stage.
        margins = (self.limit - prediction.mean_states.ravel()).at[1:].add(-0.1)
        return SimpleNamespace(
            residuals=prediction.commands.ravel(), inequality_margins=margins
        )


def test_six_step_suffix_repairs_lag_that_last_command_cannot(experiment):
    plan = LagPlan()
    commands = jnp.ones((30, 1))
    state, latent = jnp.zeros(1), jnp.ones(1)
    reference = SimpleNamespace(states=jnp.zeros((31, 1)))
    final_only = plan.rollout_commands(
        commands.at[-1].set(-1), state, latent, None, None
    )
    assert (
        float(
            plan.optimization_terms(
                final_only, None, None, None
            ).inequality_margins.min()
        )
        < 0
    )
    problem = experiment.SuffixProblem(plan, reference)
    repaired, report = problem.solve(commands, state, latent, latent)
    assert report["feasible_found"]
    np.testing.assert_array_equal(repaired[:24], commands[:24])
    assert report["maximum_utilization"] <= 1 + experiment.FEASIBILITY_TOLERANCE
    assert not np.allclose(report["terminal_latent"], report["terminal_command"])


def test_frozen_prefix_violation_cannot_be_hidden_by_terminal_repair(experiment):
    plan = LagPlan(limit=0.5)
    problem = experiment.SuffixProblem(plan, SimpleNamespace(states=jnp.zeros((31, 1))))
    accepted, report = problem.solve(
        jnp.ones((30, 1)), jnp.zeros(1), jnp.ones(1), jnp.ones(1)
    )
    assert accepted is None
    assert not report["feasible_found"]


def test_acceptance_rejects_nonfinite_out_of_box_and_violated_margin(experiment):
    lower, upper = np.zeros(1), np.ones(1)
    assert not experiment.finite_feasible(
        np.array([[np.nan]]), np.ones(1), lower, upper
    )
    assert not experiment.finite_feasible(
        np.zeros((1, 1)), np.array([np.nan]), lower, upper
    )
    assert not experiment.finite_feasible(np.array([[1.001]]), np.ones(1), lower, upper)
    assert not experiment.finite_feasible(
        np.zeros((1, 1)), np.array([-1e-3]), lower, upper
    )
