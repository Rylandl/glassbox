"""Instrumented optimizer parity and development-only checkpoint selection."""

import copy
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.default_model import _RECIPE
from glassbox.experimental.sequence_model import SequenceBatch, fit_sequence_model


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import experiment_checkpoint_attribution

    return experiment_checkpoint_attribution


@pytest.mark.parametrize("seed", [0, 17])
def test_all_checkpoint_instrumentation_matches_library_optimizer(experiment, seed):
    rng = np.random.default_rng(901)

    def batch(n):
        past = rng.normal(size=(n, 3, 2))
        up = rng.normal(size=(n, 2, 1))
        uf = rng.normal(size=(n, 3, 1))
        target = np.empty((n, 3, 2))
        x = past[:, -1].copy()
        for t in range(3):
            x = 0.8 * x + 0.1 * np.tanh(uf[:, t])
            target[:, t] = x
        return SequenceBatch(past, up, uf, target, 0.05)

    train, dev = batch(16), batch(8)
    recipe = dict(_RECIPE, steps=13, check_every=5, width=4, batch_size=8)
    scales = np.array([[0.2, 0.3], [0.1, 0.5], [0.4, 0.7]])
    snapshots, trace = experiment.fit_checkpoints(train, dev, scales, recipe, seed)
    canonical, report = fit_sequence_model(
        train,
        dev,
        kind=recipe["kind"],
        objective="rollout",
        seed=seed,
        steps=13,
        check_every=5,
        width=4,
        batch_size=8,
        learning_rate=recipe["learning_rate"],
        ridge=recipe["ridge_fraction"] * 16 * 3,
        error_scale=scales,
    )
    assert [r["step"] for r in trace] == [0, 5, 10, 13]
    chosen = int(np.argmin([r["validation_rollout_mse"] for r in trace]))
    assert snapshots[chosen].fingerprint() == canonical.fingerprint()
    assert trace == report["trace"]
    # Snapshots are finite and own their parameters independently of later steps.
    for model in snapshots:
        assert all(np.isfinite(p).all() for p in model.params.values())
    initial = copy.deepcopy(snapshots[0].params)
    snapshots[-1].params["bias"][0] += 1
    for key, value in initial.items():
        np.testing.assert_array_equal(snapshots[0].params[key], value)


def test_select_crosses_criteria_and_reports_recording_instability(experiment):
    # Checkpoint zero wins when horizon 2 dominates; checkpoint one wins when capped.
    predicted = np.array(
        [[[[3.0], [0.0]], [[0.0], [0.0]]], [[[0.0], [1.0]], [[1.0], [0.0]]]]
    )
    target = np.zeros((2, 2, 1))
    ids = np.array(["a", "b"])
    original = experiment.select(predicted, target, np.array([[1.0], [0.1]]), ids)
    capped = experiment.select(predicted, target, np.ones((2, 1)), ids)
    assert original["selected_index"] == 0
    assert capped["selected_index"] == 1
    assert capped["leave_one_recording_out_indices"] == {"a": 0, "b": 1}
    # Ties are deterministic and cannot be broken by evaluation targets: none supplied.
    tied = experiment.select(np.zeros_like(predicted), target, np.ones((2, 1)), ids)
    assert tied["selected_index"] == 0
