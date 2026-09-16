"""Exact witnesses separating first-order dynamics from linear feature capacity."""

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    sys.modules.pop("experiment_history_confounding", None)
    import experiment_history_confounding

    return experiment_history_confounding


@pytest.mark.parametrize("seed", [101, 202, 303])
def test_cycle_is_fully_observed_markov_with_older_nonlinear_proxy(experiment, seed):
    supplied = experiment.observed_cycle(seed, evaluation=True)
    assert len(supplied.segments) == 32
    for s in supplied.segments:
        np.testing.assert_allclose(
            experiment.cycle_step(s.states[:-1]), s.states[1:], atol=1e-15
        )
        # The next first coordinate is a cubic function of a present coordinate.
        np.testing.assert_allclose(s.states[1:, 0], s.states[:-1, 1] ** 3, atol=1e-15)
        # Lag four exposes the same nonlinear feature despite no missing state.
        np.testing.assert_array_equal(s.states[5:, 0], s.states[:-5, 0])
        assert s.start_row == 20 * int(s.segment_id.split("-")[1])
    calibration = experiment.observed_cycle(seed)
    assert not {s.recording_id for s in calibration.segments} & {
        s.recording_id for s in supplied.segments
    }


def test_short_additive_cubic_basis_contains_the_witness_residual(experiment):
    rng = np.random.default_rng(24)
    x = rng.uniform(-0.9, 0.9, (50, 5))
    features = experiment.powers(x, 3)
    coefficients = np.zeros(15)
    coefficients[0], coefficients[11] = -1, 1
    np.testing.assert_allclose(
        features @ coefficients, x[:, 1] ** 3 - x[:, 0], atol=1e-15
    )


def test_hold_witness_uses_no_future_values(experiment):
    class Model:
        def __init__(self):
            self.contract = {"name": "fixture"}
            self._seen = {"fit": "content"}
            self.history_steps = 2

        def fingerprint(self):
            return "fixture-model"

    model = experiment.HoldWitness(Model())
    states = np.arange(30.0).reshape(2, 3, 5)
    expected = np.repeat(states[:, -1:], 4, axis=1)
    np.testing.assert_array_equal(
        model.predict(states, np.zeros((2, 2, 1)), np.ones((2, 4, 1))), expected
    )
    assert model.fingerprint() == experiment.HoldWitness(Model()).fingerprint()


def test_basis_comparison_keeps_unavailable_ratios(experiment):
    zeros = np.zeros((5, 1))
    report = experiment.measures(
        zeros, {k: zeros for k in ("short", "extended", "replacement")}, np.arange(5)
    )
    assert report["extra_history_mse_reduction"] == {
        "extended": [None],
        "replacement": [None],
    }


def test_pooled_horizon_scale_uses_training_only_and_preserves_units(experiment):
    from types import SimpleNamespace

    from experiment_horizon_scaling import pooled_scale

    rng = np.random.default_rng(31)
    x, future = rng.normal(size=(7, 3, 2)), rng.normal(size=(7, 5, 2))
    future[:, -1] = x[:, -1]  # Exact return must not create a tiny final scale.
    batch = SimpleNamespace(past_states=x, future_states=future)
    scale = pooled_scale(batch, np.array([2.0, 3.0]))
    expected = np.sqrt(np.mean((future - x[:, -1:]) ** 2, axis=(0, 1)))
    np.testing.assert_allclose(scale, np.broadcast_to(expected, (5, 2)))
    factors, offset = np.array([0.01, 1000.0]), np.array([3.0, -10.0])
    encoded = SimpleNamespace(
        past_states=x * factors + offset, future_states=future * factors + offset
    )
    np.testing.assert_allclose(
        pooled_scale(encoded, np.array([2.0, 3.0]) * factors), scale * factors
    )


def test_pooled_scale_retains_positive_state_scale_floor(experiment):
    from types import SimpleNamespace

    from experiment_horizon_scaling import pooled_scale

    batch = SimpleNamespace(
        past_states=np.zeros((3, 3, 2)), future_states=np.zeros((3, 5, 2))
    )
    np.testing.assert_allclose(
        pooled_scale(batch, np.array([2.0, 3.0])), np.broadcast_to([0.02, 0.03], (5, 2))
    )
