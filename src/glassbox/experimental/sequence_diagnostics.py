"""Observational evidence about input predictability and omitted history.

Fixed diagnostic regressions are cross-fitted by whole recording. Remaining
input variation is relative to these regressors, not causal independence.
Older-history gains can reflect model error or proxy features, not only memory.
This module never updates a dynamics model or supplies an admission decision.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

_DIAGNOSTIC_RECIPE = {
    "id": "sequence-evidence-v1-prototype",
    "history_multiplier": 2,
    "maximum_windows_per_recording": 256,
    "ridge_fraction": 0.001,
    "input_predictors": ["affine", "quadratic"],
    "residual_predictor": "affine",
    "folds": "leave one whole diagnostic recording out",
    "replacement_control": "older features from other training recordings; held-out donors use training recordings only",
    "seed": 713,
}


def _features(x, quadratic):
    if not quadratic:
        return x
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((x, x[:, i] * x[:, j]))


def _scale(values):
    std = values.std(axis=0)
    return np.where(std > 0, std, 1.0)


@dataclass(frozen=True)
class _Regression:
    input_mean: np.ndarray
    input_scale: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    coefficients: np.ndarray
    quadratic: bool

    def predict(self, x):
        features = _features((x - self.input_mean) / self.input_scale, self.quadratic)
        z = (features - self.feature_mean) / self.feature_scale
        return (z @ self.coefficients) * self.target_scale + self.target_mean

    def arrays(self):
        return {name: np.asarray(value) for name, value in vars(self).items()}


def _fit(x, y, *, quadratic=False):
    """Train-only scaling and fixed ridge; no target-dependent model selection."""
    mean, scale = x.mean(axis=0), _scale(x)
    features = _features((x - mean) / scale, quadratic)
    feature_mean, feature_scale = features.mean(axis=0), _scale(features)
    z = (features - feature_mean) / feature_scale
    target_mean, target_scale = y.mean(axis=0), _scale(y)
    target = (y - target_mean) / target_scale
    penalty = _DIAGNOSTIC_RECIPE["ridge_fraction"] * len(x)
    if z.shape[1] <= len(x):
        coefficients = np.linalg.solve(
            z.T @ z + penalty * np.eye(z.shape[1]), z.T @ target
        )
    else:
        coefficients = z.T @ np.linalg.solve(z @ z.T + penalty * np.eye(len(z)), target)
    return _Regression(
        mean,
        scale,
        feature_mean,
        feature_scale,
        target_mean,
        target_scale,
        coefficients,
        quadratic,
    )


def _ratio(numerator, denominator):
    return [
        float(a / b) if b > 0 else None
        for a, b in zip(numerator, denominator, strict=True)
    ]


def _rms(values):
    return np.sqrt(np.mean(values**2, axis=0))


def _measure(arrays, mask):
    u = arrays["current_inputs"][mask]
    reference = _rms(u - arrays["input_reference"][mask])
    inputs = {}
    for kind in ("affine", "quadratic"):
        error = _rms(u - arrays[f"input_{kind}"][mask])
        inputs[kind] = dict(
            rms=error.tolist(),
            reference_rms=reference.tolist(),
            remaining_rms_fraction=_ratio(error, reference),
        )
    residual = arrays["model_residual"][mask]
    errors = {
        name: _rms(residual - arrays[f"residual_{name}"][mask])
        for name in ("short", "extended", "replacement")
    }
    reductions = {}
    for name in ("extended", "replacement"):
        fraction = _ratio(errors[name] ** 2, errors["short"] ** 2)
        reductions[name] = [None if x is None else 1 - x for x in fraction]
    return dict(
        input_predictability=inputs,
        forecast_error=dict(
            model_rms=_rms(residual).tolist(),
            **{f"after_{name}_rms": value.tolist() for name, value in errors.items()},
            extra_history_mse_reduction=reductions,
        ),
    )


def _run(model, recordings):
    from .default_model import _contract, _recording_content

    if _contract(recordings) != model.contract:
        raise ValueError(
            "diagnostic configuration, channels or sample interval differ from this model"
        )
    seen = _recording_content(recordings)
    if set(seen) & set(model._seen) or set(seen.values()) & set(model._seen.values()):
        raise ValueError(
            "diagnostics require recordings outside the model's fit and development data"
        )
    names = sorted(seen)
    if len(names) < 3:
        raise ValueError(
            "diagnostics need at least three distinct out-of-fit recordings"
        )
    p = model.history_steps
    longer = _DIAGNOSTIC_RECIPE["history_multiplier"] * p
    px, pu, older, current, targets, ids, segment_ids, origins = ([] for _ in range(8))
    for record_index, name in enumerate(names):
        candidates = [
            (s, t)
            for s in sorted(recordings.segments, key=lambda s: s.start_row)
            if s.recording_id == name
            for t in range(longer, len(s.states) - 1)
        ]
        if not candidates:
            raise ValueError(
                f"recording {name!r} has no complete {longer}-step diagnostic history"
            )
        cap = _DIAGNOSTIC_RECIPE["maximum_windows_per_recording"]
        chosen = np.linspace(
            0, len(candidates) - 1, min(cap, len(candidates)), dtype=int
        )
        for index in chosen:
            s, t = candidates[index]
            px.append(s.states[t - p : t + 1])
            pu.append(s.inputs[t - p : t])
            older.append(
                np.r_[
                    s.states[t - longer : t - p].reshape(-1),
                    s.inputs[t - longer : t - p].reshape(-1),
                ]
            )
            current.append(s.inputs[t])
            targets.append(s.states[t + 1])
            ids.append(record_index)
            segment_ids.append(s.segment_id)
            origins.append(s.start_row + t)
    px, pu, older, current, targets, ids = map(
        np.array, (px, pu, older, current, targets, ids)
    )
    predicted = np.asarray(model.predict(px, pu, current[:, None]))[:, 0]
    if not np.isfinite(predicted).all():
        raise ValueError("diagnostic model forecasts are nonfinite")
    short = np.column_stack((px.reshape(len(px), -1), pu.reshape(len(pu), -1)))
    short_error = np.column_stack((short, current))
    extended = np.column_stack((short_error, older))
    residual = targets - predicted
    arrays = dict(
        past_states=px,
        past_inputs=pu,
        older_features=older,
        short_features=short,
        current_inputs=current,
        targets=targets,
        model_prediction=predicted,
        model_residual=residual,
        record_index=ids,
        source_origins=np.array(origins),
        segment_ids=np.array(segment_ids),
        record_names=np.array(names),
    )
    for key in ("input_reference", "input_affine", "input_quadratic"):
        arrays[key] = np.empty_like(current)
    for key in ("residual_short", "residual_extended", "residual_replacement"):
        arrays[key] = np.empty_like(residual)
    for held_index, _name in enumerate(names):
        train, test = (
            np.flatnonzero(ids != held_index),
            np.flatnonzero(ids == held_index),
        )
        prefix = f"fold_{held_index}_"
        arrays[prefix + "train"] = train
        arrays[prefix + "test"] = test
        arrays["input_reference"][test] = current[train].mean(axis=0)
        rng = np.random.default_rng(_DIAGNOSTIC_RECIPE["seed"] + held_index)
        donor_train = np.array([rng.choice(train[ids[train] != ids[i]]) for i in train])
        donor_test = rng.choice(train, size=len(test))
        arrays[prefix + "donor_train"] = donor_train
        arrays[prefix + "donor_test"] = donor_test
        for label, x_train, y_train, x_test, quadratic in (
            ("input_affine", short[train], current[train], short[test], False),
            ("input_quadratic", short[train], current[train], short[test], True),
            (
                "residual_short",
                short_error[train],
                residual[train],
                short_error[test],
                False,
            ),
            (
                "residual_extended",
                extended[train],
                residual[train],
                extended[test],
                False,
            ),
            (
                "residual_replacement",
                np.column_stack((short_error[train], older[donor_train])),
                residual[train],
                np.column_stack((short_error[test], older[donor_test])),
                False,
            ),
        ):
            reg = _fit(x_train, y_train, quadratic=quadratic)
            arrays[label][test] = reg.predict(x_test)
            arrays.update(
                {prefix + label + "_" + k: v for k, v in reg.arrays().items()}
            )
    report = dict(
        recipe=copy.deepcopy(_DIAGNOSTIC_RECIPE),
        model_fingerprint=model.fingerprint(),
        contract=model.contract,
        model_history_steps=p,
        extended_history_steps=longer,
        forecast_steps=1,
        recording_content=seen,
        windows=len(ids),
        aggregate=_measure(arrays, np.ones(len(ids), dtype=bool)),
        recordings={
            name: dict(windows=int(np.sum(ids == i)), **_measure(arrays, ids == i))
            for i, name in enumerate(names)
        },
        evidence_limits=[
            "Remaining input variation is relative to fixed affine/quadratic predictors, not proof of independence, excitation sufficiency, or causal identification.",
            "Extra history can explain misspecification or proxy features; gains do not uniquely identify hidden memory or select a new dynamics history.",
            "Replacement older features are an offline negative control from other training recordings, not a deployable observation stream.",
            "Diagnostic regressions use other diagnostic recordings only. Reused or previously inspected recordings are not an untouched qualification benchmark.",
            "Ratios can exceed one and MSE reductions can be negative; None means a zero reference denominator, not zero uncertainty.",
            "No uncertainty calibration, acceptance threshold, model modification, or control-admission decision is supplied.",
        ],
    )
    return report, arrays


def diagnose(model, recordings):
    """Return cross-recording evidence using one fixed diagnostic recipe.

    Supply at least three distinct recordings outside the model's fit/development
    set, with its same declared channels and sample interval. All channels retain
    their physical units; ratios are descriptive. This is offline analysis.
    """
    return _run(model, recordings)[0]
