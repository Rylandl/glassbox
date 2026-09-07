"""Cold startup, disturbance injection and completion must retain their meaning."""

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_feedback_recovery")


def test_cold_request_discards_earlier_waveform_and_selects_eight_updates(
    experiment, monkeypatch
):
    from investigate_fast_suffix import FastSuffixSolver, SuffixValues

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

    solver = object.__new__(experiment.RecoverySolver)
    solver.model = Model(SuffixValues(None, None, None, None))
    solver.counts = {}
    previous = jnp.array([0.25, 0.5, 0.75, 1.0])
    earlier = jnp.arange(120).reshape(30, 4) % 17 / 16
    monkeypatch.setattr(FastSuffixSolver, "_seed_plan", lambda *_a, **_k: "seeded")
    solver.set_seed(experiment.SeedRequest(previous, earlier))
    solver._seed_plan()
    assert solver._iteration_budget == 2
    np.testing.assert_array_equal(
        solver.seed_commands, jnp.concatenate((earlier[1:], earlier[-1:]))
    )
    # A fresh cold request overwrites any setup or prior solved waveform.
    solver.set_seed(experiment.SeedRequest(previous))
    assert solver._seed_plan() == "seeded"
    assert solver._iteration_budget == 8
    np.testing.assert_array_equal(
        solver.seed_commands, jnp.repeat(previous[None, :], 30, axis=0)
    )
    solver.set_seed(experiment.SeedRequest(previous, earlier, already_shifted=True))
    solver._seed_plan()
    assert solver._iteration_budget == 2
    np.testing.assert_array_equal(solver.seed_commands, earlier)


def test_kick_changes_only_roll_at_one_absolute_tick(experiment):
    state = jnp.arange(13, dtype=float)
    for tick in (0, 59, 60, 61):
        varied, applied = experiment.kick_state(state, tick, True, 0.25)
        expected = np.asarray(state).copy()
        if tick == 60:
            expected[10] += 0.25
        np.testing.assert_array_equal(varied, expected)
        assert applied == (tick == 60)
    varied, applied = experiment.kick_state(state, 60, False, 0.25)
    np.testing.assert_array_equal(varied, state)
    assert not applied


def test_completion_score_uses_last_twenty_samples_and_separate_tolerances(experiment):
    from glassbox.core.synthetic import resting_state

    states = np.repeat(np.asarray(resting_state())[None, :], 121, axis=0)
    states[:101, 0] = 100.0
    states[101:, 0] = 2.0
    score = experiment.score(states, np.ones(12), complete=True, actual_inside=True)
    assert score["tail_normalized_tracking_rms"] == pytest.approx(np.sqrt(4 / 12))
    assert score["terminal_attitude_rate_within_tolerances"]
    assert not score["terminal_full_state_within_tolerances"]
    for complete, inside in ((False, True), (True, False)):
        rejected = experiment.score(
            states, np.ones(12), complete=complete, actual_inside=inside
        )
        assert rejected["tail_normalized_tracking_rms"] is None
        assert rejected["terminal_attitude_rate_within_tolerances"] is None


@pytest.mark.parametrize("tick,kick", [(4, False), (60, True)])
def test_rejected_solve_applies_nothing_and_cannot_complete_fixture_recovery(
    experiment, monkeypatch, tick, kick
):
    from glassbox.core.synthetic import resting_state

    state = jnp.asarray(resting_state())
    seed = jnp.full((30, 4), 0.5)
    solver = SimpleNamespace(reports=[], seed_commands=seed)
    plan = SimpleNamespace(
        sample_period_s=0.02,
        model=SimpleNamespace(
            validity_utilization=lambda _: jnp.zeros(6),
            runtime_spec=SimpleNamespace(
                validity_envelope=SimpleNamespace(
                    angular_velocity_half_width_rad_s=[1.0] * 3
                )
            ),
        ),
        tolerances=SimpleNamespace(local_state_scale=jnp.ones(12)),
    )
    monkeypatch.setattr(
        experiment,
        "solve_waveform",
        lambda *_a, **_k: SimpleNamespace(command_usable=False),
    )
    monkeypatch.setattr(experiment, "describe", lambda *_a, **_k: {"optimizer": None})

    def forbidden(*_a, **_k):
        raise AssertionError("a failed solve must never advance the plant")

    monkeypatch.setattr(experiment, "step_with_latent", forbidden)
    case, trace = experiment.simulate(
        solver,
        plan,
        None,
        None,
        None,
        {
            "state": state,
            "latent": seed[0],
            "previous": seed[0],
            "commands": seed,
            "absolute_tick": tick,
        },
        kick=kick,
    )
    assert case["applied_intervals"] == 0
    assert case["final_absolute_time_s"] == tick * 0.02
    assert case["supplied_solved_waveform"]
    assert not case["interval_limit_completed"]
    assert case["tail_normalized_tracking_rms"] is None
    assert len(trace["states"]) == 1
    if kick:
        assert trace["states"][-1, 10] == pytest.approx(0.02)
        assert case["terminal_normalized_error"][9] == pytest.approx(0.02)
        np.testing.assert_array_equal(trace["states"][-1], case["kick"]["after_state"])


def test_nonfinite_failure_diagnostics_remain_serializable(experiment):
    import json

    from glassbox.core.synthetic import resting_state

    state = np.asarray(resting_state()).copy()
    state[0] = np.nan
    result = experiment.score([state], np.ones(12), complete=False, actual_inside=False)
    result["support"] = np.inf
    stored = json.loads(json.dumps(experiment.json_finite(result), allow_nan=False))
    assert stored["tail_normalized_tracking_rms"] is None
    assert stored["terminal_normalized_error"][0] is None
    assert stored["support"] is None
