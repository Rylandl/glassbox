"""Independent NumPy replay of the frozen residual-only backtracking audit.

This verifier consumes authenticated parent actions and saved trial residuals. It
never imports the learner or the producer of the point summaries, and performs
no residual, derivative, conditioning or optimizer evaluation.
"""

import numpy as np

ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)
WORK = {
    "new_residual_calls": 4,
    "reused_residuals": 1,
    "cg_iterations": 0,
    "conditioning_calls": 0,
    "gradient_calls": 0,
    "initialization_calls": 0,
    "observe_calls": 0,
    "applied_updates": 0,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return float(value) if np.isfinite(value) else None


class _Checks:
    def __init__(self):
        self.count = 0

    def value(self, actual, expected, name):
        if isinstance(expected, dict):
            _require(isinstance(actual, dict), name + " must be an object")
            for key, value in expected.items():
                _require(key in actual, name + " is missing " + key)
                self.value(actual[key], value, name + "." + key)
            return
        if isinstance(expected, list):
            _require(
                isinstance(actual, list) and len(actual) == len(expected),
                name + " length differs",
            )
            for index, value in enumerate(expected):
                self.value(actual[index], value, name + "." + str(index))
            return
        if expected is None or isinstance(expected, (str, bool, int)):
            _require(
                type(actual) is type(expected) and actual == expected, name + " differs"
            )
        else:
            _require(
                isinstance(actual, (float, int)) and not isinstance(actual, bool),
                name + " must be numeric",
            )
            _require(
                np.isfinite(actual)
                and np.isclose(actual, expected, atol=1e-10, rtol=1e-8),
                name + " arithmetic differs",
            )
        self.count += 1


def _data_loss(raw, weight):
    # Each radial Huber group contributes one scalar, regardless of its width.
    result = 0.0
    for left, right in ((0, 3), (3, 6), (6, 15)):
        squared = np.sum(raw[..., left:right] ** 2, axis=-1)
        penalty = np.where(squared <= 1, 0.5 * squared, np.sqrt(squared) - 0.5)
        result += float(np.sum(weight[..., 0] * penalty))
    return result


def _comparison(left, right):
    if left < right:
        return "win"
    if left > right:
        return "loss"
    _require(left == right, "retained comparison is undefined")
    return "equal"


def _verify_point(
    summary, parent_summary, parent_arrays, trial_raw, conditioning_finite
):
    checks = _Checks()
    _require(type(conditioning_finite) is bool, "conditioning finite flag differs")
    a = {key: np.asarray(value) for key, value in parent_arrays.items()}
    theta, prior, gradient = (a[key] for key in ("theta", "prior", "gradient"))
    raw, weight, irls = (a[key] for key in ("raw", "weight", "irls"))
    delta, projected = a["probe_delta"][3], a["probe_projected"][3]
    trials = np.asarray(trial_raw)
    _require(theta.ndim == 1, "parameter shape differs")
    _require(
        theta.shape == prior.shape == gradient.shape == delta.shape,
        "parameter vector shapes differ",
    )
    _require(raw.ndim == 3 and raw.shape[-1] == 15, "parent residual shape differs")
    _require(
        weight.shape == (len(raw), 1, 1) and irls.shape == projected.shape == raw.shape,
        "parent residual weight or action shapes differ",
    )
    _require(trials.shape == (len(ALPHAS), *raw.shape), "trial residual shape differs")
    _require(np.isfinite(prior).all() and np.all(prior >= 0), "invalid parent prior")
    _require(
        np.isfinite(weight).all() and np.all(weight >= 0), "invalid parent weights"
    )
    checks.value(float(weight.sum() * raw.shape[1] * 3), 1.0, "weight normalization")
    reused = a["probe_trial_raw"][3]
    _require(
        trials[0].dtype == reused.dtype
        and trials[0].shape == reused.shape
        and trials[0].tobytes() == reused.tobytes(),
        "alpha1 residual is not byte-exact parent evidence",
    )
    checks.count += 1
    parent_probe = parent_summary["probes"][3]
    _require(type(parent_probe["finite"]) is bool, "parent finite flag differs")
    production = float(a["production_trial"])
    _require(np.isfinite(production), "original four production loss is nonfinite")
    current_data = _data_loss(raw, weight)
    current_prior = float(0.5 * (theta @ (prior * theta)))
    current = current_data + current_prior
    linear = float(gradient @ delta)
    curvature = float(np.sum(weight * irls * projected**2) + delta @ (prior * delta))
    current_summary = {
        "data_loss": _number(current_data),
        "prior_loss": _number(current_prior),
        "total_loss": _number(current),
    }
    checks.value(parent_summary["current"], current_summary, "parent.current")
    checks.value(summary["current"], current_summary, "point.current")
    rows, losses = [], []
    for index, alpha in enumerate(ALPHAS):
        step, jp = alpha * delta, alpha * projected
        candidate = theta + step
        data = _data_loss(trials[index], weight)
        penalty = float(0.5 * (candidate @ (prior * candidate)))
        total = data + penalty
        predicted = -alpha * linear - 0.5 * alpha**2 * curvature
        finite = bool(
            bool(a["finite"])
            and conditioning_finite
            and all(
                np.isfinite(value).all()
                for value in (
                    step,
                    jp,
                    candidate,
                    gradient,
                    trials[index],
                    current,
                    total,
                    predicted,
                )
            )
        )
        gain = (current - total) / predicted if finite and predicted > 0 else -np.inf
        row = {
            "alpha": alpha,
            "finite": finite,
            "would_accept": bool(finite and total < current and gain >= 0.1),
            "data_loss": _number(data),
            "prior_loss": _number(penalty),
            "total_loss": _number(total),
            "data_improvement": _number(current_data - data),
            "prior_improvement": _number(current_prior - penalty),
            "improvement": _number(current - total),
            "predicted_reduction": _number(predicted),
            "gain": _number(gain),
        }
        arithmetic = {key: value for key, value in row.items() if key != "would_accept"}
        saved = summary["steps"][index]
        _require(saved["alpha"] == alpha, "point alpha differs from frozen ladder")
        checks.value(saved, arithmetic, "point.steps." + str(index))
        # Strict decisions concern the validated saved scalars. This avoids
        # mistaking NumPy reduction-order roundoff for a different tie policy.
        saved_current = summary["current"]["total_loss"]
        saved_total, saved_predicted = saved["total_loss"], saved["predicted_reduction"]
        saved_gain = (
            (saved_current - saved_total) / saved_predicted
            if finite and saved_predicted > 0
            else -np.inf
        )
        row["would_accept"] = bool(
            finite and saved_total < saved_current and saved_gain >= 0.1
        )
        rows.append(row)
        losses.append(saved_total)
    # Conditioning is separate from the parent's direction finite flag. A failed
    # conditioning pass can disable otherwise finite parent alpha1 evidence.
    parent_alpha1 = {key: value for key, value in rows[0].items() if key != "alpha"}
    if not conditioning_finite:
        for key in ("finite", "would_accept", "gain"):
            del parent_alpha1[key]
    checks.value(parent_probe, parent_alpha1, "parent.reference_trusted")
    selected = next(
        (index for index, row in enumerate(rows) if row["would_accept"]), None
    )
    retained = production if selected is None else losses[selected]
    comparison = _comparison(retained, production)
    _require(
        summary["original_four"]["total_loss"] == production,
        "original four total is not the exact production scalar",
    )
    _require(
        summary["retained_total_loss"] == retained,
        "retained total is not the exact selected or production scalar",
    )
    _require(
        summary["selected_alpha"] == (None if selected is None else ALPHAS[selected]),
        "selected_alpha differs from frozen ladder",
    )
    expected = {
        "format": "online-backtracking-point-v1",
        "current": current_summary,
        "original_four": {
            "total_loss": production,
            "instrumented_data_loss": parent_summary["probes"][1]["data_loss"],
            "instrumented_prior_loss": parent_summary["probes"][1]["prior_loss"],
        },
        "parent_reference": parent_summary["reference"],
        "steps": rows,
        "selected_index": selected,
        "selected_alpha": None if selected is None else ALPHAS[selected],
        "selection_comparison": "no_selection" if selected is None else comparison,
        "retained_total_loss": retained,
        "retained_comparison": comparison,
        "hypothetical_residual_evaluations": 5 if selected is None else selected + 1,
        "work": WORK,
    }
    checks.value(summary, expected, "point")
    return {
        "verified": True,
        "arithmetic_checks": checks.count,
        "model_calls": 0,
        "optimizer_calls": 0,
    }


def verify_point(
    summary, parent_summary, parent_arrays, trial_raw, *, conditioning_finite
):
    """Verify losses, original acceptance and first-pass selection without a model."""
    try:
        with np.errstate(all="ignore"):
            return _verify_point(
                summary, parent_summary, parent_arrays, trial_raw, conditioning_finite
            )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid backtracking evidence schema") from error


def _empty_group():
    return {
        "count": 0,
        "selections": {},
        "comparisons": {},
        "retained_comparisons": {},
        "reference_status": {},
        "accepted_per_alpha": [0] * len(ALPHAS),
        "nonfinite_per_alpha": [0] * len(ALPHAS),
        "selected_data_decreases": 0,
        "selected_prior_decreases": 0,
        "hypothetical_residual_evaluations": 0,
    }


def _accumulate(group, result):
    group["count"] += 1
    for name, value in (
        ("selections", str(result["selected_alpha"])),
        ("comparisons", result["selection_comparison"]),
        ("retained_comparisons", result["retained_comparison"]),
        ("reference_status", result["parent_reference"]["status"]),
    ):
        counts = group[name]
        counts[value] = counts.get(value, 0) + 1
    for index, step in enumerate(result["steps"]):
        group["accepted_per_alpha"][index] += int(step["would_accept"])
        group["nonfinite_per_alpha"][index] += int(not step["finite"])
    selected = result["selected_index"]
    if selected is not None:
        step = result["steps"][selected]
        group["selected_data_decreases"] += int(step["data_improvement"] > 0)
        group["selected_prior_decreases"] += int(step["prior_improvement"] > 0)
    group["hypothetical_residual_evaluations"] += result[
        "hypothetical_residual_evaluations"
    ]


def _exact_keys(actual, expected, path):
    if isinstance(expected, dict):
        _require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            path + " keys differ",
        )
        for key in expected:
            _exact_keys(actual[key], expected[key], path + "." + key)


def verify_aggregate(actual, rows):
    """Independently count verified point results, including every rejected alpha."""
    try:
        expected = {
            "all": _empty_group(),
            "by_version": {},
            "by_version_case": {},
        }
        for row in rows:
            version, case = row["version"], row["case"]
            _require(
                isinstance(version, str) and isinstance(case, str),
                "aggregate point identity differs",
            )
            by_version = expected["by_version"].setdefault(version, _empty_group())
            by_case = expected["by_version_case"].setdefault(
                version + "/" + case, _empty_group()
            )
            for group in (expected["all"], by_version, by_case):
                _accumulate(group, row["result"])
        _exact_keys(actual, expected, "aggregate")
        checks = _Checks()
        checks.value(actual, expected, "aggregate")
        return {
            "verified": True,
            "arithmetic_checks": checks.count,
            "model_calls": 0,
            "optimizer_calls": 0,
        }
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("invalid backtracking aggregate schema") from error
