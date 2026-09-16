"""Reference predictions and support thresholds must use only their declared data."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from glassbox.experimental.sequence_model import SequenceBatch


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "experiment_forecast_diagnosis.py"))


def toy_batch(offset=0):
    rng = np.random.default_rng(31)
    x = rng.normal(scale=0.01, size=(6, 4, 2))
    x[3:] += 10
    x += offset
    return SequenceBatch(
        x,
        rng.normal(size=(6, 3, 1)),
        rng.normal(size=(6, 5, 1)),
        rng.normal(size=(6, 5, 2)),
        0.02,
    )


def test_support_thresholds_exclude_same_recording_and_ignore_evaluation(experiment):
    train = toy_batch()
    batches = dict(train=train, development=toy_batch(2), evaluation=toy_batch(3))
    usage = dict(train=[dict(recording="a", windows=3), dict(recording="b", windows=3)])
    a = experiment["support_diagnostics"](batches, usage)
    b = experiment["support_diagnostics"](
        {**batches, "evaluation": toy_batch(1000)}, usage
    )
    for key in (
        "thresholds",
        "training_features",
        "feature_mean",
        "feature_scale",
        "development_distance",
    ):
        np.testing.assert_array_equal(a[key], b[key])
    assert a["training_cross_record_distance"].min() > 0.5
    assert b["evaluation_distance"].min() > a["evaluation_distance"].max()


def test_group_selection_preserves_different_development_winners(experiment):
    target = np.zeros((6, 3, 15))
    a, b = target.copy(), target.copy()
    a[:, :, :3], a[:, :, 3:6] = 1, 3
    b[:, :, :3], b[:, :, 3:6] = 2, 1
    selected = experiment["select_candidates"](dict(a=a, b=b), target, np.ones(15))
    assert all(row[:2] == ["a", "b"] for row in selected["cell_names"])
    combined = experiment["selected_cells"](dict(a=a, b=b), selected["cell_names"])
    np.testing.assert_array_equal(combined[:, :, :3], a[:, :, :3])
    np.testing.assert_array_equal(combined[:, :, 3:6], b[:, :, 3:6])


def test_trend_reference_uses_only_observed_history(experiment):
    b = toy_batch()
    history = np.broadcast_to(
        np.arange(4)[None, :, None] * np.array([2, -1]), b.past_states.shape
    )
    a = SequenceBatch(history, b.past_inputs, b.future_inputs, b.future_states, b.dt_s)
    changed = SequenceBatch(
        history, b.past_inputs, b.future_inputs, 1000 * b.future_states, b.dt_s
    )
    p = experiment["references"](a, (1, 5))
    q = experiment["references"](changed, (1, 5))
    np.testing.assert_array_equal(p["trend"], q["trend"])
    expected = np.broadcast_to(
        np.array([4, 8])[None, :, None] * np.array([2, -1]), p["trend"].shape
    )
    np.testing.assert_allclose(p["trend"], expected)
