"""Saved-request parity must refer to the exact runtime inputs and array bytes."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    return importlib.import_module("investigate_single_seed")


def test_bitwise_parity_distinguishes_dtype_and_signed_zero(experiment):
    positive = np.array([0.0], dtype=np.float32)
    assert experiment.same_tree((positive,), (positive.copy(),))
    assert not experiment.same_tree((positive,), (positive.astype(np.float64),))
    assert not experiment.same_tree((positive,), (-positive,))
