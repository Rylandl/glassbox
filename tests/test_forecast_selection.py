"""Empirical recording guards and exclusions, independent of flight data."""

import numpy as np
import pytest

from glassbox.experimental.forecast_selection import ForecastEvidence


def evidence(short_error=2.0):
    target = np.zeros((11, 2, 2))
    model = np.full_like(target, 0.1)
    model[-1] = short_error
    return ForecastEvidence.from_predictions(
        {"hold": np.ones_like(target), "model": model},
        target,
        ["long"] * 10 + ["short"],
        scale=np.ones(2),
        groups=((0,), (1,)),
    )


def test_equal_record_weighting_exposes_long_record_dominance():
    e = evidence()
    assert e.choose("pooled")["selected"] == "model"
    assert e.choose("equal_record")["selected"] == "hold"
    assert e.choose("minimax_hold")["selected"] == "hold"
    assert e.choose("guarded_hold")["eligible"] == ["hold"]
    assert e.choose("pooled")["scores"]["model"] == pytest.approx(4.1 / 11)


def test_excluded_record_outcomes_cannot_change_subset_choice():
    for policy in ("pooled", "equal_record", "minimax_hold", "guarded_hold"):
        before = evidence(2).choose(policy, recordings=("long",))
        after = evidence(1e6).choose(policy, recordings=("long",))
        assert before == after
        assert before["selected"] == "model"


def test_guard_checks_every_horizon_and_output_group():
    target = np.zeros((3, 2, 2))
    model = np.zeros_like(target)
    model[:, 1, 1] = 1.06
    e = ForecastEvidence.from_predictions(
        {"hold": np.ones_like(target), "model": model},
        target,
        ["r"] * 3,
        scale=np.ones(2),
        groups=((0,), (1,)),
    )
    assert e.choose("equal_record")["selected"] == "model"
    assert e.choose("guarded_hold")["selected"] == "hold"
    assert e.choose("guarded_hold", maximum_rmse_ratio=1.07)["selected"] == "model"


@pytest.mark.parametrize("change", ["groups", "ids", "scale", "prediction"])
def test_invalid_evidence_inputs(change):
    target = np.zeros((3, 1, 2))
    kwargs = dict(
        predictions={"hold": target.copy()},
        target=target,
        recording_ids=["r"] * 3,
        scale=np.ones(2),
        groups=((0,), (1,)),
    )
    if change == "groups":
        kwargs["groups"] = ((0, 0), (1,))
    if change == "ids":
        kwargs["recording_ids"] = ["r", "r"]
    if change == "scale":
        kwargs["scale"] = np.zeros(2)
    if change == "prediction":
        kwargs["predictions"]["hold"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ForecastEvidence.from_predictions(**kwargs)


def test_zero_reference_and_bad_subsets():
    target = np.zeros((3, 1, 1))
    e = ForecastEvidence.from_predictions(
        {"hold": target, "model": target + 1},
        target,
        ["r"] * 3,
        scale=[1],
        groups=((0,),),
    )
    assert e.choose("guarded_hold")["selected"] == "hold"
    for records in [(), ("missing",), ("r", "r")]:
        with pytest.raises(ValueError):
            e.choose("equal_record", recordings=records)


def test_evidence_owns_arrays_and_rejects_inconsistent_counts():
    source = evidence()
    mse = np.array(source.mse)
    copy = ForecastEvidence(
        source.names, source.recordings, source.counts, source.group_widths, mse
    )
    mse[:] = 0
    assert copy.choose("pooled")["selected"] == "model"
    assert not copy.mse.flags.writeable
    with pytest.raises(ValueError):
        ForecastEvidence(
            source.names, source.recordings, [10, 0], source.group_widths, source.mse
        )
