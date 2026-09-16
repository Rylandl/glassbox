"""Boundary-safe Euclidean forecast samples with explicit temporal context.

History uses only state/input rows at or before the forecast origin. Future
actuation is an explicit conditioning input, summarized in equal time bins;
future state is used only as the target. This supports offline conditional
replay. Measured future actuation is not necessarily known during live use.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from glassbox.experimental.transition_gp import TransitionSamples, _matrix


@dataclass(frozen=True)
class ForecastWindows:
    samples: TransitionSamples
    anchors: np.ndarray
    sample_ids: tuple[str, ...]
    horizon_steps: int
    history_lags: tuple[int, ...]
    control_bin_edges: tuple[int, ...]


def forecast_windows(
    states,
    controls,
    *,
    recording_id: str,
    dt_s: float,
    horizon_steps: int,
    anchors,
    context=None,
    history_lags=(),
    control_bins=None,
) -> ForecastWindows:
    """Build samples from exactly one contiguous, uniformly sampled recording.

    Context contains origin-time auxiliary measurements. For each history lag,
    append state and control differences relative to their current values. Those
    are causal changes, not future derivatives. No state interpolation or padding
    is done. Controls have one row per interval; states include the last endpoint.
    Consumers must keep every window from one recording in one data role.
    Omitted control_bins uses up to ten bins, capped by the horizon length.
    """
    states, controls = _matrix(states, "states"), _matrix(controls, "controls")
    if len(controls) != len(states) - 1:
        raise ValueError("controls must have one row per state interval")
    if not isinstance(recording_id, str) or not recording_id:
        raise ValueError("recording_id must be a nonempty string")
    if (
        not isinstance(horizon_steps, int)
        or isinstance(horizon_steps, bool)
        or horizon_steps < 1
    ):
        raise ValueError("horizon_steps must be a positive integer")
    if control_bins is None:
        control_bins = min(10, horizon_steps)
    if (
        not isinstance(control_bins, int)
        or isinstance(control_bins, bool)
        or not 1 <= control_bins <= horizon_steps
    ):
        raise ValueError("control_bins must be between one and horizon_steps")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    lags = tuple(history_lags)
    if len(set(lags)) != len(lags) or any(
        not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 for lag in lags
    ):
        raise ValueError("history_lags must contain unique positive integers")
    indices = np.array(anchors, copy=True)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("anchors must be unique integer indices")
    if (
        len(indices) < 3
        or np.any(indices < max(lags, default=0))
        or np.any(indices + horizon_steps >= len(states))
    ):
        raise ValueError(
            "at least three complete windows within the recording are required"
        )
    auxiliary = (
        np.empty((len(states), 0)) if context is None else _matrix(context, "context")
    )
    if len(auxiliary) != len(states):
        raise ValueError("context must align with state timestamps")
    edges = np.linspace(0, horizon_steps, control_bins + 1, dtype=int)
    future = np.stack(
        [
            np.mean(controls[indices[:, None] + np.arange(lo, hi)], axis=1)
            for lo, hi in pairwise(edges)
        ],
        axis=1,
    )
    columns = [auxiliary[indices], future.reshape(len(indices), -1)]
    for lag in lags:
        columns.extend(
            (
                states[indices - lag] - states[indices],
                controls[indices - lag] - controls[indices],
            )
        )
    samples = TransitionSamples(
        states[indices],
        controls[indices],
        states[indices + horizon_steps],
        dt_s * horizon_steps,
        np.concatenate(columns, axis=1),
    )
    indices.setflags(write=False)
    return ForecastWindows(
        samples,
        indices,
        tuple(
            f"{recording_id}/anchor{int(index)}/h{horizon_steps}" for index in indices
        ),
        horizon_steps,
        lags,
        tuple(map(int, edges)),
    )
