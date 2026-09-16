"""Descriptive forecast tolerances, separate from fitting and control readiness.

Consumers supply nonnegative errors in task-relevant coordinates and physical
units. The fixed readout reports an empirical 95% fraction, per recording and
over the entire forecast prefix. This is neither a confidence interval nor a
calibrated bound for new observations. Overlapping windows remain dependent.
"""

from collections.abc import Mapping

import numpy as np

_FRACTION = 0.95


def _values(array):
    """Represent unavailable/unbounded error allowances as JSON null."""
    return [float(v) if np.isfinite(v) else None for v in array]


def _empirical_quantile(values):
    # Nearest rank achieves the requested fraction of this observed sample.
    # This deliberately does not use a conformal or population-coverage claim.
    rank = int(np.ceil(_FRACTION * len(values))) - 1
    return np.sort(values, axis=0)[rank]


def assess_forecasts(errors, recording_ids, *, times_s, limits):
    """Compare recorded forecast errors against declared physical allowances.

    ``errors`` maps names (including units) to [window, future_step] nonnegative
    scalar errors, such as a vector norm. ``limits`` gives a positive allowance
    for every named error. Those allowances are consumer requirements, not
    learner tuning. ``times_s`` excludes the shared initial state.

    A window is within limits only when every metric at every earlier forecast
    step is within its allowance. The horizon readout requires at least 95% of
    observed windows in *each* recording, without pooling away poor recordings.
    Individual 95% tolerance curves are marginal; joint fractions are computed
    directly, not inferred from marginal percentiles. NaN and positive infinity
    count as unavailable forecasts and fail the affected prefix. Null reported
    allowances mean no finite allowance reaches that observed fraction.
    """
    if not isinstance(errors, Mapping) or not errors:
        raise ValueError("errors must be a nonempty mapping of named physical errors")
    if (
        not isinstance(limits, Mapping)
        or set(limits) != set(errors)
        or any(not isinstance(name, str) or not name.strip() for name in errors)
    ):
        raise ValueError("limits must name exactly the supplied errors")
    allowance = {}
    for name, limit in limits.items():
        if (
            isinstance(limit, (bool, np.bool_))
            or not np.isscalar(limit)
            or not np.isrealobj(limit)
        ):
            raise ValueError("each limit must be a finite positive scalar")
        try:
            value = float(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("each limit must be a finite positive scalar") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("each limit must be a finite positive scalar")
        allowance[name] = value
    if not np.isrealobj(times_s):
        raise ValueError("future times must be real")
    times = np.asarray(times_s, dtype=float)
    ids = np.asarray(recording_ids)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(times <= 0)
        or np.any(np.diff(times) <= 0)
        or ids.ndim != 1
        or not len(ids)
        or ids.dtype.kind != "U"
        or any(not str(r).strip() for r in ids)
    ):
        raise ValueError("positive increasing future times and recording IDs required")
    arrays = {}
    for name, value in errors.items():
        if not np.isrealobj(value):
            raise ValueError("errors must be real")
        a = np.asarray(value, dtype=float)
        if a.shape != (len(ids), len(times)) or np.any(a < 0):
            raise ValueError("errors must be nonnegative [window, future_step] arrays")
        arrays[name] = np.where(np.isfinite(a), a, np.inf)

    records = {}
    tolerance = {name: [] for name in arrays}
    fractions = []
    for recording in sorted(set(ids.tolist())):
        mask = ids == recording
        measurements = {}
        joint = np.ones((int(mask.sum()), len(times)), dtype=bool)
        for name, a in arrays.items():
            end = a[mask]
            prefix = np.maximum.accumulate(end, axis=1)
            required = _empirical_quantile(prefix)
            tolerance[name].append(required)
            within = prefix <= allowance[name]
            joint &= within
            with np.errstate(over="ignore"):
                rms = np.sqrt(np.mean(end**2, axis=0))
            measurements[name] = dict(
                endpoint_rmse=_values(rms),
                endpoint_empirical_p95=_values(_empirical_quantile(end)),
                prefix_empirical_p95=_values(required),
                prefix_observed_max=_values(prefix.max(axis=0)),
                prefix_within_limit_count=within.sum(axis=0).tolist(),
                prefix_unavailable_count=(~np.isfinite(prefix)).sum(axis=0).tolist(),
            )
        fraction = joint.mean(axis=0)
        fractions.append(fraction)
        records[recording] = dict(
            windows=int(mask.sum()),
            metrics=measurements,
            joint_prefix_within_limits_count=joint.sum(axis=0).tolist(),
            joint_prefix_within_limits_fraction=fraction.tolist(),
        )
    worst = np.min(fractions, axis=0)
    eligible = np.flatnonzero(worst >= _FRACTION)
    return dict(
        definition="observed_whole_prefix_per_recording_v1",
        empirical_fraction=_FRACTION,
        times_s=times.tolist(),
        limits=allowance,
        recordings=records,
        required_marginal_tolerance={
            name: _values(np.max(curves, axis=0)) for name, curves in tolerance.items()
        },
        minimum_recording_joint_fraction=worst.tolist(),
        observed_95pct_horizon_s=float(times[eligible[-1]]) if len(eligible) else None,
        evidence_limits=[
            "95% describes these observed windows; it is not population coverage or a confidence level",
            "recording IDs do not prove independence and overlapping windows are dependent",
            "marginal tolerances do not establish joint coverage; use measured joint counts",
            "no interpolation, extrapolation, control-readiness decision or model selection is performed",
            "null error allowances mean unavailable/unbounded, and null horizon means no measured horizon qualified",
        ],
    )
