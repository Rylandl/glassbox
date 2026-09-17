"""Generic evaluation uses declared rows and units, and never learns from them."""

import json
from dataclasses import replace

import numpy as np
import pytest

from glassbox import LearnedDynamics, SequenceCollection, SequenceSegment
from glassbox.learner import _recording_content
from glassbox.workflows.forecast import evaluate


class LinearForecast(LearnedDynamics):
    """An analytic test forecast; avoids involving a fitter in score equations."""

    def __init__(self):
        self._seen = {}
        self.batch_sizes = []

    @property
    def contract(self):
        return dict(
            configuration_id="score-test",
            state_channels=["x [m]", "y [rad]"],
            input_channels=["u [command]"],
            dt_s=0.1,
        )

    @property
    def report(self):
        return {"envelope": {"nominal_coverage": 0.9}}

    @property
    def history_steps(self):
        return 2

    @property
    def horizon_steps(self):
        return 2

    def envelope(self):
        return np.full((2, 2), 0.5)

    def predict(self, past_states, past_inputs, future_inputs):
        self.batch_sizes.append(len(past_states))
        assert past_states.shape[1:] == (3, 2)
        assert past_inputs.shape[1:] == (2, 1)
        assert future_inputs.shape[1:] == (2, 1)
        return past_states[:, -1:, :] + np.arange(1, 3)[None, :, None] * [1, 2]

    def fingerprint(self):
        return "analytic-fixture"

    def update(self, recordings):
        raise AssertionError("evaluation must not update")


def recording(name, rows=8, start=0, segment_id="whole"):
    t = np.arange(rows, dtype=float) + start
    return SequenceSegment(
        name, segment_id, np.column_stack((t, t**2)), t[:-1, None], 0.1, start
    )


def collection(*segments):
    return SequenceCollection(
        segments, "score-test", ("x [m]", "y [rad]"), ("u [command]",)
    )


def expected(origins):
    h = np.arange(1, 3)[None, :]
    t = np.array(origins)[:, None]
    errors = np.stack((np.zeros((len(t), 2)), 2 * h - 2 * t * h - h**2), axis=-1)
    hold = np.stack((np.broadcast_to(-h, (len(t), 2)), -2 * t * h - h**2), axis=-1)
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
        (report["per_recording"]["a"], expected(range(2, 6))),
        (report["per_recording"]["b"], expected(range(2, 8))),
        (report["aggregate"], expected([*range(2, 6), *range(2, 8)])),
    ):
        for name, value in wanted.items():
            np.testing.assert_allclose(actual[name], value, rtol=0, atol=1e-12)
    assert model._seen == {}
    json.dumps(report, allow_nan=False)


def test_predictions_are_bounded_and_windows_do_not_cross_gaps():
    model = LinearForecast()
    report = evaluate(
        model, collection(recording("a", 600), recording("a", 20, 700, "after-gap"))
    )
    assert max(model.batch_sizes) <= 256
    assert sum(model.batch_sizes) == 596 + 16
    assert report["aggregate"]["windows"] == 612


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
    assert model.batch_sizes == []


def test_contract_mismatch_and_empty_recordings_are_rejected():
    model = LinearForecast()
    with pytest.raises(ValueError, match="channels or sample"):
        evaluate(
            model,
            replace(collection(recording("a")), state_channels=("x [cm]", "y [rad]")),
        )
    with pytest.raises(ValueError, match="no complete"):
        evaluate(model, collection(recording("too-short", 4)))
