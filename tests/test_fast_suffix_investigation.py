"""The suffix adapter must preserve waveform and common solve semantics."""

import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_sqp_recovery_investigation import small_controller as base_controller_fixture

from glassbox.control.plan import SolverPolicy

small_controller = base_controller_fixture


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_fast_suffix")


def test_prefix_is_dynamic_exact_and_suffix_has_independent_steps(experiment):
    # Keep normalization mapping independent of belief for this adapter test.
    class Mapping:
        _commands_from_normalized = staticmethod(lambda x: (x + 1) / 2)
        rollout_commands = staticmethod(lambda commands, *args: commands)

    mapping = Mapping()
    blocks = jnp.arange(24).reshape(6, 4) / 12 - 1
    values = experiment.SuffixValues(None, None, None, jnp.full((24, 4), 0.37))
    kernel = jax.jit(
        lambda b, v: experiment.SuffixPlan.rollout(mapping, b, None, None, None, v)
    )
    first = kernel(blocks, values)
    second = kernel(blocks, values._replace(frozen_commands=jnp.full((24, 4), 0.63)))
    np.testing.assert_array_equal(first[:24], values.frozen_commands)
    np.testing.assert_array_equal(second[:24], jnp.full((24, 4), 0.63))
    np.testing.assert_array_equal(first[24:], second[24:])
    np.testing.assert_array_equal(first[24:], (blocks + 1) / 2)


def test_bad_waveforms_are_rejected_before_optimization(experiment):
    plan = SimpleNamespace(
        command_size=4, command_minimum=np.zeros(4), command_maximum=np.ones(4)
    )
    for commands in (
        np.zeros((29, 4)),
        np.full((30, 4), np.nan),
        np.full((30, 4), 1.01),
    ):
        with pytest.raises(ValueError):
            experiment.FastSuffixSolver.validate_commands(plan, commands)


def test_real_boundary_rejects_expired_deadline_without_assessment(
    experiment, small_controller, monkeypatch
):
    import glassbox.control.solver as common

    plan = replace(
        small_controller.plan,
        policy=SolverPolicy(30, 10, allow_unresolved_parameters=True),
    )
    commands = jnp.full((30, 4), 0.5)
    estimates = importlib.import_module(
        "investigate_sqp_recovery"
    ).DEFAULT_WORK_ESTIMATES
    solver = experiment.FastSuffixSolver(plan, commands, work_estimates=estimates)
    from glassbox.core.synthetic import resting_state

    state = jnp.asarray(resting_state())
    reference = solver.hold_reference(state)
    ticks = iter((0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(
        common, "time", SimpleNamespace(perf_counter=lambda: next(ticks, 1.0))
    )
    result = solver.solve(
        state, reference, commands[0], latent_state=commands[0], deadline_s=0.02
    )
    assert not result.command_usable
    assert result.status.value == "deadline_exceeded"
    assert result.deadline_met is False
    assert result.nonlinear_feasibility.status == "not_assessed"
    assert not solver.counts
