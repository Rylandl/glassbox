"""Check that moving blocks preserve a shift without certifying its feasibility."""

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from glassbox.control.plan import NMPCWarmStart, SolverPolicy, blocks_cover_horizon


@pytest.fixture
def investigation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_horizon_shift")


def phase_solver(investigation, horizon, count, phase):
    solver = object.__new__(investigation.PhaseSolver)
    solver.policy = SolverPolicy(horizon_steps=horizon, block_count=count)
    solver.model = SimpleNamespace(
        horizon_steps=horizon,
        block_count=count,
        command_size=2,
        phase=phase,
        command_minimum=jnp.asarray([-1.0, 0.0]),
        command_maximum=jnp.asarray([1.0, 1.0]),
    )
    return solver


def test_moving_blocks_preserve_every_shift_including_phase_wrap(investigation):
    # Include equal, truncated, single-block, and one-command-per-step grids.
    for horizon in range(1, 34):
        for count in range(1, min(horizon, 10) + 1):
            if not blocks_cover_horizon(horizon, count):
                continue
            period = (horizon + count - 1) // count if count > 1 else 1
            blocks = np.stack(
                (np.linspace(-0.8, 0.8, count), np.linspace(0.1, 0.9, count)), axis=1
            )
            for phase in range(period):
                indices = investigation.block_indices(horizon, count, phase)
                commands = blocks[indices]
                warm = investigation.MovingWarmStart(commands, count, phase)
                next_phase = (phase + 1) % period
                solver = phase_solver(investigation, horizon, count, next_phase)
                normalized = solver._warm_blocks(warm)
                assert normalized is not None
                next_indices = investigation.block_indices(horizon, count, next_phase)
                recovered = solver.model.command_minimum + (
                    normalized[next_indices] + 1
                ) * 0.5 * (solver.model.command_maximum - solver.model.command_minimum)
                expected = np.concatenate((commands[1:], commands[-1:]))
                np.testing.assert_allclose(recovered, expected, atol=1e-7)
                assert normalized.shape == (count, 2)


@pytest.mark.parametrize(
    "layout",
    [(0, 1, 0), (5, 0, 0), (5, 4, 0), (6, 2, 3), (6, 2, -1), (6, 2, 0.5), (6, 1, 1)],
)
def test_invalid_moving_layout_is_rejected(investigation, layout):
    with pytest.raises(ValueError):
        investigation.block_indices(*layout)


def test_foreign_or_edited_warm_start_is_not_silently_averaged(investigation):
    commands = jnp.asarray([[0.1, 0.2]] * 3 + [[0.7, 0.8]] * 3)
    warm = investigation.MovingWarmStart(commands, 2, 0)
    solver = phase_solver(investigation, 6, 2, 1)
    assert solver._warm_blocks(warm) is not None
    assert solver._warm_blocks(NMPCWarmStart(commands)) is None
    assert solver._warm_blocks(replace(warm, phase=1)) is None
    assert solver._warm_blocks(replace(warm, block_count=3)) is None
    assert (
        solver._warm_blocks(replace(warm, commands=commands.at[2, 0].set(0.3))) is None
    )


@dataclass(frozen=True)
class Diagnostics:
    solve_time_s: float = 0.0


@dataclass(frozen=True)
class Result:
    predicted_commands: object
    warm_start: object = None
    diagnostics: Diagnostics = Diagnostics()
    command_usable: bool = True
    deadline_met: bool | None = None
    status: object = None


def fake_dispatcher(investigation):
    dispatcher = object.__new__(investigation.MovingBlockSolver)
    dispatcher.model = SimpleNamespace(horizon_steps=6, block_count=2, command_size=2)
    dispatcher.period = 3
    dispatcher.reports = []
    calls = []

    def make_solver(phase):
        def solve(*args, **kwargs):
            calls.append((phase, kwargs))
            dispatcher.reports.append({})
            return Result(jnp.ones((6, 2)))

        return SimpleNamespace(
            solve=solve,
            _failure_result=lambda status, *_: Result(
                None, command_usable=False, status=status
            ),
        )

    dispatcher.solvers = [make_solver(phase) for phase in range(3)]
    return dispatcher, calls


def test_phase_follows_input_warm_start_independent_of_report_history(investigation):
    dispatcher, calls = fake_dispatcher(investigation)
    previous = jnp.ones(2)
    first = dispatcher.solve(None, None, previous)
    assert first.warm_start.phase == 0
    second = dispatcher.solve(None, None, previous, warm_start=first.warm_start)
    assert second.warm_start.phase == 1
    dispatcher.reports.clear()
    replay = dispatcher.solve(None, None, previous, warm_start=first.warm_start)
    assert replay.warm_start.phase == 1
    third = dispatcher.solve(None, None, previous, warm_start=second.warm_start)
    assert third.warm_start.phase == 2
    wrapped = dispatcher.solve(None, None, previous, warm_start=third.warm_start)
    assert wrapped.warm_start.phase == 0
    dispatcher.solve(None, None, previous, warm_start=NMPCWarmStart(jnp.ones((6, 2))))
    assert calls[-1][0] == 0 and calls[-1][1]["warm_start"] is None


@pytest.mark.parametrize(
    "elapsed,usable", [(0.019, True), (0.02, False), (0.021, False)]
)
def test_phase_result_assembly_is_included_in_deadline(
    investigation, monkeypatch, elapsed, usable
):
    dispatcher, calls = fake_dispatcher(investigation)
    clock = iter([0.0, 0.001, elapsed])
    monkeypatch.setattr(
        investigation, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    result = dispatcher.solve(None, None, jnp.ones(2), deadline_s=0.02)
    assert calls[0][1]["deadline_s"] == pytest.approx(0.019)
    if usable:
        assert result.command_usable
        assert result.deadline_met is True
        assert result.diagnostics.solve_time_s == elapsed
    else:
        assert result.status == investigation.SolveStatus.DEADLINE_EXCEEDED
        assert result.deadline_met is False


def test_expired_phase_selection_does_not_start_an_inner_solve(
    investigation, monkeypatch
):
    dispatcher, calls = fake_dispatcher(investigation)
    clock = iter([0.0, 0.02])
    monkeypatch.setattr(
        investigation, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    result = dispatcher.solve(None, None, jnp.ones(2), deadline_s=0.02)
    assert not calls
    assert result.status == investigation.SolveStatus.DEADLINE_EXCEEDED
    assert result.deadline_met is False
