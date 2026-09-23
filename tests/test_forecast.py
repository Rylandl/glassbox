"""Generic evaluation uses declared rows and units, and never learns from them."""

import json
from dataclasses import replace

import numpy as np
import pytest

from glassbox import (
    STATE_CHANNELS,
    LearnedDynamics,
    SequenceCollection,
    SequenceSegment,
)
from glassbox.learner import _recording_content
from glassbox.workflows.forecast import evaluate


class LinearForecast(LearnedDynamics):
    """An analytic test forecast; avoids involving a fitter in score equations."""

    def __init__(self):
        self._seen = {}
        self.history_lengths = []
        self.calibrated = True
        self.absent_calibration = None

    @property
    def contract(self):
        return dict(
            configuration_id="score-test",
            state_channels=list(STATE_CHANNELS),
            input_channels=["u [command]"],
            dt_s=0.1,
        )

    @property
    def report(self):
        return {
            "envelope": {"nominal_coverage": 0.9}
            if self.calibrated
            else self.absent_calibration
        }

    @property
    def horizon_steps(self):
        return 2

    def envelope(self):
        assert self.calibrated, "uncalibrated model must not be asked for an envelope"
        return np.full((2, 15), 0.5)

    def predict(self, past_states, past_inputs, future_inputs):
        self.history_lengths.append(len(past_inputs))
        assert past_states.shape == (len(past_inputs) + 1, 15)
        assert past_inputs.shape[1:] == (1,)
        assert future_inputs.shape == (2, 1)
        increment = np.zeros((2, 15))
        increment[:, :2] = np.arange(1, 3)[:, None] * [1, 2]
        return past_states[-1:] + increment

    def fingerprint(self):
        return "analytic-fixture"

    def update(self, recordings):
        raise AssertionError("evaluation must not update")


def recording(name, rows=8, start=0, segment_id="whole"):
    t = np.arange(rows, dtype=float) + start
    states = np.zeros((rows, 15))
    states[:, :2] = np.column_stack((t, t**2))
    states[:, 6:] = np.eye(3).ravel()
    return SequenceSegment(name, segment_id, states, t[:-1, None], 0.1, start)


def collection(*segments):
    return SequenceCollection(segments, "score-test", STATE_CHANNELS, ("u [command]",))


def expected(origins):
    h = np.arange(1, 3)[None, :]
    t = np.array(origins)[:, None]
    errors = np.stack((np.zeros((len(t), 2)), 2 * h - 2 * t * h - h**2), axis=-1)
    hold = np.stack((np.broadcast_to(-h, (len(t), 2)), -2 * t * h - h**2), axis=-1)
    errors = np.pad(errors, ((0, 0), (0, 0), (0, 13)))
    hold = np.pad(hold, ((0, 0), (0, 0), (0, 13)))
    return dict(
        windows=len(t),
        rmse=np.sqrt(np.mean(errors**2, axis=0)),
        hold_current_rmse=np.sqrt(np.mean(hold**2, axis=0)),
        coverage=np.mean(np.abs(errors) <= 0.5, axis=0),
    )


def test_scores_every_origin_with_per_channel_per_horizon_equations():
    model = LinearForecast()
    report = evaluate(model, collection(recording("a", 8), recording("b", 10)))
    assert report["model_fingerprint"] == "analytic-fixture"
    assert report["contract"] == model.contract
    assert report["horizons_s"] == [0.1, 0.2]
    assert report["nominal_coverage"] == 0.9
    for actual, wanted in (
        (report["per_recording"]["a"], expected(range(1, 6))),
        (report["per_recording"]["b"], expected(range(1, 8))),
        (report["aggregate"], expected([*range(1, 6), *range(1, 8)])),
    ):
        for name, value in wanted.items():
            np.testing.assert_allclose(actual[name], value, rtol=0, atol=1e-12)
    assert model._seen == {}
    json.dumps(report, allow_nan=False)


def test_complete_histories_do_not_cross_gaps():
    model = LinearForecast()
    report = evaluate(
        model, collection(recording("a", 600), recording("a", 20, 700, "after-gap"))
    )
    assert min(model.history_lengths) == 1
    assert max(model.history_lengths) == 597
    assert len(model.history_lengths) == 597 + 17
    assert report["aggregate"]["windows"] == 614
    assert report["history_policy"] == "complete_segment"


@pytest.mark.parametrize("role", ["training", "development"])
@pytest.mark.parametrize("rename", [False, True])
def test_fit_and_development_identity_or_content_cannot_be_evaluated(role, rename):
    model = LinearForecast()
    seen = collection(recording(role))
    model._seen = _recording_content(seen)
    tested = (
        collection(replace(seen.segments[0], recording_id="alias")) if rename else seen
    )
    with pytest.raises(ValueError, match="outside fitting"):
        evaluate(model, tested)
    assert model.history_lengths == []


def test_contract_mismatch_and_empty_recordings_are_rejected():
    model = LinearForecast()
    with pytest.raises(ValueError, match="channels or sample"):
        evaluate(
            model,
            replace(collection(recording("a")), input_channels=("other command",)),
        )
    with pytest.raises(ValueError, match="no complete"):
        evaluate(model, collection(recording("too-short", 3)))


@pytest.mark.parametrize(
    "metadata", [None, {"available": False, "reason": "no calibration"}]
)
def test_uncalibrated_revision_has_no_invented_coverage(metadata):
    model = LinearForecast()
    model.calibrated = False
    model.absent_calibration = metadata
    report = evaluate(model, collection(recording("a")))
    assert report["aggregate"]["coverage"] is None
    assert report["per_recording"]["a"]["coverage"] is None
    assert report["nominal_coverage"] is None
    assert report["calibration_provenance"] is None
    assert report["known_recording_count"] == 0
    np.testing.assert_array_equal(
        report["aggregate"]["rmse"], expected(range(1, 6))["rmse"]
    )


@pytest.mark.parametrize("invalid", ["nonfinite", "shape"])
def test_unusable_predictions_are_rejected(invalid):
    model = LinearForecast()
    model.predict = lambda *args: np.full(
        (2, 15 if invalid == "nonfinite" else 14), np.nan
    )
    with pytest.raises(ValueError, match="nonfinite or misaligned"):
        evaluate(model, collection(recording("a")))
