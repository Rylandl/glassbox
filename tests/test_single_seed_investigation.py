"""Saved-request parity must refer to the exact runtime inputs and array bytes."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_single_seed")


@pytest.mark.parametrize(
    "directory,case,tick",
    [
        ("feedback-recovery", "cold_original", 0),
        ("feedback-recovery", "cold_original_kick", 60),
        ("feedback-recovery-runtime", "cold_original_kick", 10),
        ("seed-timing-history", "cold_small", 68),
    ],
)
def test_saved_request_restores_original_command_dtype(
    experiment, directory, case, tick
):
    records = Path(__file__).parents[1] / "docs/investigations"
    with np.load(records / "feedback-recovery/cold_original.npz") as trace:
        previous = trace["seed_waveforms"][0, 0]
    item = experiment.load_request(records / directory, case, tick, previous)
    assert item["state"].shape == (13,)
    assert item["latent"].shape == item["previous"].shape == (4,)
    assert item["source"]["tick"] == tick
    if tick == 0:
        assert item["request"].warm_commands is None
        np.testing.assert_array_equal(
            item["seed"], np.repeat(previous[None, :], 30, axis=0)
        )
    else:
        incoming = np.asarray(item["request"].warm_commands)
        assert incoming.dtype == np.asarray(item["seed"]).dtype
        np.testing.assert_array_equal(
            item["seed"], np.concatenate((incoming[1:], incoming[-1:]))
        )


def test_bitwise_parity_distinguishes_dtype_and_signed_zero(experiment):
    positive = np.array([0.0], dtype=np.float32)
    assert experiment.same_tree((positive,), (positive.copy(),))
    assert not experiment.same_tree((positive,), (positive.astype(np.float64),))
    assert not experiment.same_tree((positive,), (-positive,))
