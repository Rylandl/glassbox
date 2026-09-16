"""As-of sampling for heterogeneous telemetry streams, without interpolation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HeldSignal:
    values: np.ndarray
    valid: np.ndarray
    source_indices: np.ndarray
    age_s: np.ndarray


def causal_hold(published_s, values, query_s, *, maximum_age_s):
    """Use the last row published at/before each query, with a freshness limit.

    Duplicate publication times use the last row in input order. Times must be
    nondecreasing; values must be finite. Invalid queries return NaN values and
    source index -1, while age is infinity before the first publication. This
    asserts causality relative to the supplied clock, not physical sensor truth.
    """
    times, data, query = (
        np.asarray(a, dtype=float) for a in (published_s, values, query_s)
    )
    if (
        times.ndim != 1
        or query.ndim != 1
        or data.ndim != 2
        or len(times) != len(data)
        or not len(times)
        or not data.shape[1]
        or not all(np.isfinite(a).all() for a in (times, data, query))
        or np.any(np.diff(times) < 0)
        or not np.isfinite(maximum_age_s)
        or maximum_age_s < 0
    ):
        raise ValueError("invalid signal, timestamps, or maximum age")
    indices = np.searchsorted(times, query, side="right") - 1
    age = np.where(indices >= 0, query - times[np.maximum(indices, 0)], np.inf)
    valid = (indices >= 0) & (age <= maximum_age_s)
    held = np.where(valid[:, None], data[np.maximum(indices, 0)], np.nan)
    return HeldSignal(held, valid, np.where(valid, indices, -1), age)
