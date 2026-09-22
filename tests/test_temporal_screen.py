"""Short recordings must stay explicit in learning-progress comparisons."""

import numpy as np
from screen_temporal import learning_phases


def test_short_tape_keeps_all_targets_without_inventing_late_progress():
    truth = np.zeros((62, 15))
    truth[:, 6:] = np.eye(3).ravel()
    prediction = truth.copy()
    prediction[:, :6] = 0.1
    data = dict(
        row=np.arange(62),
        prediction=prediction,
        truth=truth,
        origin=truth,
        predict_s=np.ones(62),
        update_s=np.ones(62),
    )
    info = dict(
        id="short", family="quad", dt_s=0.01, parameters=1, reference_parameters=1
    )
    result = learning_phases(data, info, prediction)
    assert result["first25"]["count"] == 25
    assert result["25to100"]["count"] == 37
    assert result["100toend"] is None
    assert result["first25"]["ratios"]["primary"] == 1
    assert result["25to100"]["ratios"]["primary"] == 1
