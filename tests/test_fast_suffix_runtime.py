"""The suffix waveform's preparation is part of the command deadline."""

import importlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from glassbox.control.plan import NonlinearFeasibility, SolveStatus


@dataclass(frozen=True)
class Diagnostics:
    solve_time_s: float = 0.0


@dataclass(frozen=True)
class Result:
    command: object
    predicted_commands: object
    predicted_states: object = None
    predicted_latent_states: object = None
    diagnostics: Diagnostics = Diagnostics()
    command_usable: bool = True
    status: object = SolveStatus.STALLED
    deadline_met: bool | None = None
    nonlinear_feasibility: NonlinearFeasibility = field(
        default_factory=lambda: NonlinearFeasibility(1, 0.0, 1e-6)
    )


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = importlib.import_module("investigate_fast_suffix_runtime")
    clock = SimpleNamespace(now=0.0, preparation=0.001, solve=0.01)
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    calls = []

    def set_seed(commands):
        clock.now += clock.preparation
        solver.seed_commands = commands

    def solve(*args, **kwargs):
        calls.append(kwargs)
        clock.now += clock.solve
        return Result(solver.seed_commands[0], solver.seed_commands)

    solver = SimpleNamespace(
        set_seed=set_seed,
        solve=solve,
        model=None,
        validate_commands=lambda _, commands: commands,
        _failure_result=lambda status, *_: Result(
            np.zeros(4),
            np.zeros((30, 4)),
            command_usable=False,
            status=status,
            nonlinear_feasibility=NonlinearFeasibility(),
        ),
    )
    return module, solver, clock, calls


def request(boundary, **kwargs):
    module, solver, _, _ = boundary
    return module.solve_waveform(
        solver,
        np.arange(120).reshape(30, 4),
        None,
        None,
        np.zeros(4),
        latent_state=np.zeros(4),
        **kwargs,
    )


def test_waveform_shift_and_preparation_use_the_outer_budget(boundary):
    _, solver, _, calls = boundary
    result = request(boundary, shift_seed=True, deadline_s=0.02)
    expected = np.arange(120).reshape(30, 4)
    np.testing.assert_array_equal(
        solver.seed_commands, np.concatenate((expected[1:], expected[-1:]))
    )
    assert calls[0]["deadline_s"] == pytest.approx(0.019)
    assert result.diagnostics.solve_time_s == pytest.approx(0.011)
    assert result.command_usable and result.deadline_met is True


def test_expired_preparation_never_starts_solver(boundary):
    _, _, clock, calls = boundary
    clock.preparation = 0.02
    result = request(boundary, deadline_s=0.02)
    assert not calls
    assert result.status == SolveStatus.DEADLINE_EXCEEDED
    assert not result.command_usable and result.deadline_met is False
    assert result.nonlinear_feasibility.status == "not_assessed"


def test_outer_assembly_can_reject_an_inner_feasible_result(boundary, monkeypatch):
    module, _, clock, _ = boundary

    def slow_assembly(value, **kwargs):
        if isinstance(value, Result) and value.command_usable:
            clock.now += 0.02
        return replace(value, **kwargs)

    monkeypatch.setattr(module, "replace", slow_assembly)
    result = request(boundary, deadline_s=0.02)
    assert result.status == SolveStatus.DEADLINE_EXCEEDED
    assert not result.command_usable and result.deadline_met is False
    assert result.nonlinear_feasibility.status == "not_assessed"


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf")])
def test_invalid_deadline_is_unassessed_and_does_not_mutate_seed(boundary, deadline):
    _, solver, _, calls = boundary
    result = request(boundary, deadline_s=deadline)
    assert not calls and not hasattr(solver, "seed_commands")
    assert not result.command_usable and result.deadline_met is None
    assert result.status == SolveStatus.INVALID_INPUT


def test_invalid_seed_is_a_failed_unassessed_request(boundary):
    _, solver, _, calls = boundary

    def reject(_):
        raise ValueError("invalid waveform")

    solver.set_seed = reject
    result = request(boundary)
    assert not calls
    assert result.status == SolveStatus.INVALID_INPUT
    assert not result.command_usable and result.deadline_met is None
    assert result.nonlinear_feasibility.status == "not_assessed"
