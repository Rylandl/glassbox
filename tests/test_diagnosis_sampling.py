"""Comparison arms must receive matched observations and nested data budgets."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "experiment_transition_diagnosis.py"))


def records():
    rng = np.random.default_rng(38)
    return [
        {
            "name": f"recording-{i}",
            "role": "train",
            "states": rng.normal(size=(200, 6)),
            "controls": rng.normal(size=(199, 4)),
            "context": rng.normal(size=(200, 9)),
        }
        for i in range(6)
    ]


def test_histories_and_horizons_keep_identical_origins(experiment):
    observations = records()
    assemble = experiment["assemble"]
    baseline, _ = assemble(observations, "train", 60, 384, 1, ())
    for horizon in (1, 10, 25):
        for lags in ((), (1, 2), (5, 10), (10, 20)):
            batch, _ = assemble(observations, "train", 60, 384, horizon, lags)
            for name in ("anchors", "groups", "states", "commands"):
                np.testing.assert_array_equal(batch[name], baseline[name])
            assert len(batch["ids"]) == len(set(batch["ids"])) == 384


def test_smaller_budget_is_nested_within_each_flight(experiment):
    observations = records()
    assemble = experiment["assemble"]
    small, _ = assemble(observations, "train", 60, 192, 10, (5, 10))
    large, _ = assemble(observations, "train", 60, 384, 10, (5, 10))
    for record in observations:
        name = record["name"]
        small_mask, large_mask = small["groups"] == name, large["groups"] == name
        np.testing.assert_array_equal(
            small["ids"][small_mask], large["ids"][large_mask][:32]
        )
        np.testing.assert_array_equal(
            small["features"][small_mask], large["features"][large_mask][:32]
        )


def test_evaluation_origins_do_not_change_with_sampling_seed(experiment):
    observations = [{**r, "role": "replication"} for r in records()[:3]]
    assemble = experiment["assemble"]
    a, _ = assemble(observations, "replication", 60, None, 10, (5, 10))
    b, _ = assemble(observations, "replication", 62, None, 10, (5, 10))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
