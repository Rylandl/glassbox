"""The acquisition comparison must not condition its random baseline on the gap."""

import runpy
from pathlib import Path

import numpy as np
import pytest

experiment = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/experiment_shape_learning.py")
)


@pytest.mark.parametrize("seed", [20, 21, 22, 23, 24])
def test_acquisition_uses_independent_pool_and_retains_the_same_base(seed):
    pool, _ = experiment["input_pool"](seed)
    base = experiment["select_rows"](pool, "outer")
    for layout in ("random-fill", "coverage-fill"):
        selected = experiment["select_rows"](pool, layout)
        assert len(selected) == len(set(selected)) == 192
        np.testing.assert_array_equal(selected[:128], base)
        assert np.all(selected[128:] >= len(pool) // 2)
    random_rows = experiment["select_rows"](pool, "random-fill")[128:]
    outside_gap = np.sum(abs(pool[random_rows, 1]) >= 0.55)
    # A wide deterministic guard catches the former all-in-gap selection bias.
    assert 10 <= outside_gap <= 50
