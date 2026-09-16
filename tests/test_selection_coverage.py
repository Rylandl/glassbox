"""Constant selection budget, nested recording pools, and fresh data identities."""

import copy
from pathlib import Path

import numpy as np


def test_recording_coverage_holds_window_budget_fixed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_horizon_generalization import generate
    from experiment_selection_coverage import pool_arrays

    matched = dict(
        input_persistence=0.65, input_innovation_weight=0.35, input_amplitude=1.0
    )
    plan = dict(
        dataset=dict(
            intervals=160,
            dt_s=0.05,
            calibration_recordings=8,
            evaluation_recordings_per_regime=4,
            rng_salt=1207,
            regimes=dict(calibration=matched, matched=matched),
        ),
        recording_counts=[2, 4, 8, 16],
        selection_rng_salt_offset=50000,
        windows=256,
    )
    before = copy.deepcopy(plan)
    pools = {n: pool_arrays(plan, 7101, 101, n) for n in plan["recording_counts"]}
    assert plan == before
    for n, pool in pools.items():
        ids, counts = np.unique(pool["recording_ids"], return_counts=True)
        assert len(ids) == n and np.all(counts == 256 // n)
        assert len(pool["targets"]) == 256
        assert all(name.startswith("selection-") for name in ids)
        assert len(set(zip(pool["recording_ids"], pool["origins"], strict=True))) == 256
        for m, smaller in pools.items():
            if m >= n:
                continue
            for record in set(smaller["recording_ids"]):
                a = pool["recording_ids"] == record
                b = smaller["recording_ids"] == record
                # More records use a shorter prefix of the same window ranking.
                np.testing.assert_array_equal(
                    pool["origins"][a], smaller["origins"][b][: 256 // n]
                )
                np.testing.assert_array_equal(
                    pool["targets"][a], smaller["targets"][b][: 256 // n]
                )
    changed = pool_arrays(plan, 7101, 202, 2)
    assert not np.array_equal(changed["targets"], pools[2]["targets"])
    supplied, _ = generate(plan["dataset"], "near_periodic", 7101, "matched")
    origin = pools[2]["origins"][0]
    assert not np.array_equal(
        pools[2]["targets"][0], supplied.segments[0].states[origin + 1 : origin + 6]
    )
