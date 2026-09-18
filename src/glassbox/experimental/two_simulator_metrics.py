"""Independent NumPy reductions for the frozen two-simulator flight baseline.

Each forecast is one planned query, including queries with no usable truth. The
caller adds simulator/scope/cell/parent/query/kind/arm identities to every row.
Aggregation never treats coordinates, horizons, or origins as independent parents.
"""

from collections import defaultdict
from itertools import groupby

import numpy as np

GROUPS = {
    "velocity_m_s": slice(0, 3),
    "body_rate_rad_s": slice(3, 6),
    "rotation_entries": slice(6, 15),
}
GROUPING = ("scope", "simulator", "kind", "arm", "horizon_s", "group", "statistic")


def _horizons(dt_s, horizons_s):
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    steps = {1}
    for horizon in horizons_s:
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("horizons must be positive and finite")
        count = round(horizon / dt_s)
        if count < 1 or not np.isclose(count * dt_s, horizon, rtol=0, atol=1e-10):
            raise ValueError("horizons must lie on the observation grid")
        steps.add(count)
    return [(step, float(round(step * dt_s, 12))) for step in sorted(steps)]


def _arrays(prediction, target, valid):
    prediction, target = (
        np.asarray(prediction, dtype=float),
        np.asarray(target, dtype=float),
    )
    valid = np.asarray(valid)
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 15:
        raise ValueError("prediction and target must have identical (H, 15) shapes")
    if valid.dtype != np.dtype(bool) or valid.shape != (len(target),):
        raise ValueError("valid must be a Boolean array of shape (H,)")
    if np.any(valid & ~np.isfinite(target).all(axis=1)):
        raise ValueError("truth marked valid contains nonfinite observations")
    return prediction, target, valid


def _mean(values):
    values = list(values)
    if not values:
        return None
    # Dividing first avoids overflow when summing large, individually finite MSEs.
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(np.asarray(values, dtype=float) / len(values)))
    return result if np.isfinite(result) else None


def _mse(errors):
    if not errors.size:
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        squared = np.square(errors)
    if not np.isfinite(squared).all():
        return None
    return _mean(squared.ravel())


def _geometry(rotation):
    matrices = rotation.reshape(-1, 3, 3)
    finite = np.isfinite(matrices).all(axis=(1, 2))
    matrices = matrices[finite]
    if not len(matrices):
        return {"finite_matrices": 0, "nonfinite_matrices": int((~finite).sum())}
    with np.errstate(over="ignore", invalid="ignore"):
        orthogonality = np.linalg.norm(
            matrices.transpose(0, 2, 1) @ matrices - np.eye(3), axis=(1, 2)
        )
        determinant = np.linalg.det(matrices)
        determinant_error = np.abs(determinant - 1)
    return {
        "finite_matrices": int(finite.sum()),
        "nonfinite_matrices": int((~finite).sum()),
        "orthogonality_frobenius_mean": _mean(orthogonality),
        "determinant_absolute_error_mean": _mean(determinant_error),
        "nonpositive_determinants": int((determinant <= 0).sum()),
    }


def score_forecast(
    prediction,
    target,
    valid,
    *,
    dt_s,
    horizons_s,
    envelope=None,
    rotation_geometry=True,
):
    """Score one (H,15) query, retaining unavailable planned horizon rows.

    ``valid`` describes truth after physical censoring, never model finiteness.
    A horizon requires a valid full truth prefix. Endpoint prediction finiteness
    concerns the selected group endpoint; cumulative finiteness concerns its full
    prefix. Coverage counts nonfinite predictions as uncovered. ``envelope`` is a
    finite nonnegative (H,15) component half-width. For branch-minus-factual arrays
    set ``rotation_geometry=False``: their last nine entries are differences.
    """
    prediction, target, valid = _arrays(prediction, target, valid)
    if envelope is not None:
        envelope = np.asarray(envelope, dtype=float)
        if envelope.shape != target.shape or not np.isfinite(envelope).all():
            raise ValueError("envelope must be finite with shape (H, 15)")
        if np.any(envelope < 0):
            raise ValueError("envelope half-widths cannot be negative")
    rows = []
    for step, horizon in _horizons(dt_s, horizons_s):
        eligible = step <= len(valid) and bool(valid[:step].all())
        for group, columns in GROUPS.items():
            for statistic in ("endpoint", "cumulative"):
                times = (
                    slice(step - 1, step) if statistic == "endpoint" else slice(0, step)
                )
                width = columns.stop - columns.start
                count = width * (1 if statistic == "endpoint" else step)
                row = {
                    "horizon_s": horizon,
                    "horizon_steps": step,
                    "group": group,
                    "statistic": statistic,
                    "truth_eligible": eligible,
                    "prediction_finite": False,
                    "mse": None,
                    "finite_subset_mse": None,
                    "finite_components": 0,
                    "components": count if eligible else 0,
                    "coverage": None,
                    "covered_components": 0,
                    "coverage_components": count
                    if eligible and envelope is not None
                    else 0,
                    "failure": None,
                }
                if eligible:
                    predicted, actual = (
                        prediction[times, columns],
                        target[times, columns],
                    )
                    finite = np.isfinite(predicted)
                    with np.errstate(over="ignore", invalid="ignore"):
                        error = predicted - actual
                    row["prediction_finite"] = bool(finite.all())
                    row["finite_components"] = int(finite.sum())
                    row["finite_subset_mse"] = _mse(error[finite])
                    if finite.all():
                        row["mse"] = _mse(error)
                    if row["mse"] is None:
                        row["failure"] = (
                            "nonfinite_prediction"
                            if not finite.all()
                            else "nonfinite_squared_error"
                        )
                    if envelope is not None:
                        covered = finite & (np.abs(error) <= envelope[times, columns])
                        row["covered_components"] = int(covered.sum())
                        row["coverage"] = float(covered.mean())
                    if rotation_geometry and group == "rotation_entries":
                        row["geometry"] = _geometry(predicted)
                rows.append(row)
    return rows


def _hierarchical_mean(rows, field):
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        value = row.get(field)
        if row["truth_eligible"] and value is not None:
            cells[row["cell"]][row["parent"]][row.get("origin", row["query"])].append(
                value
            )
    return _mean(
        _mean(
            _mean(_mean(values) for values in origins.values())
            for origins in parents.values()
        )
        for parents in cells.values()
    )


def _sqrt(value):
    return None if value is None else float(np.sqrt(value))


def _summary(rows):
    eligible = [row for row in rows if row["truth_eligible"]]
    failed = sum(row["mse"] is None for row in eligible)
    parents = {(r["cell"], r["parent"]) for r in rows}
    cells = {r["cell"] for r in rows}
    eligible_parents = {(r["cell"], r["parent"]) for r in eligible}
    eligible_cells = {r["cell"] for r in eligible}
    origins = {(r["cell"], r["parent"], r.get("origin", r["query"])) for r in rows}
    eligible_origins = {
        (r["cell"], r["parent"], r.get("origin", r["query"])) for r in eligible
    }
    status = (
        "failed"
        if failed
        else "empty"
        if not eligible
        else "incomplete"
        if len(eligible) != len(rows)
        else "complete"
    )
    mse = _hierarchical_mean(rows, "mse") if not failed else None
    return {
        "status": status,
        "planned": len(rows),
        "truth_eligible": len(eligible),
        "predicted_finite": sum(bool(row["prediction_finite"]) for row in eligible),
        "failed": failed,
        "planned_parents": len(parents),
        "eligible_parents": len(eligible_parents),
        "empty_parents": len(parents - eligible_parents),
        "planned_cells": len(cells),
        "eligible_cells": len(eligible_cells),
        "empty_cells": len(cells - eligible_cells),
        "planned_origins": len(origins),
        "eligible_origins": len(eligible_origins),
        "empty_origins": len(origins - eligible_origins),
        "mse": mse if status == "complete" else None,
        "rmse": _sqrt(mse) if status == "complete" else None,
        "available_truth_mse": mse,
        "available_truth_rmse": _sqrt(mse),
        "finite_query_subset_rmse": _sqrt(_hierarchical_mean(rows, "mse")),
        "finite_component_subset_rmse": _sqrt(
            _hierarchical_mean(rows, "finite_subset_mse")
        ),
        "coverage": _hierarchical_mean(rows, "coverage"),
        "coverage_queries": sum(row.get("coverage") is not None for row in eligible),
        "covered_components": sum(row.get("covered_components", 0) for row in eligible),
        "coverage_components": sum(
            row.get("coverage_components", 0) for row in eligible
        ),
    }


def _validate_rows(rows):
    seen = set()
    for row in rows:
        missing = set(GROUPING + ("cell", "parent", "query")) - row.keys()
        if missing:
            raise ValueError(f"metric row lacks identities: {sorted(missing)}")
        identity = tuple(row[k] for k in (*GROUPING, "cell", "parent", "query"))
        if identity in seen:
            raise ValueError("duplicate planned query metric row")
        seen.add(identity)
        for field in ("mse", "finite_subset_mse", "coverage"):
            value = row.get(field)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"invalid {field} in metric row")
        if row.get("coverage") is not None and row["coverage"] > 1:
            raise ValueError("coverage cannot exceed one")
        if not row["truth_eligible"] and row["mse"] is not None:
            raise ValueError("ineligible truth cannot carry an MSE")
        if not row["prediction_finite"] and row["mse"] is not None:
            raise ValueError("nonfinite prediction cannot carry an unqualified MSE")
        if row["kind"] == "response":
            if not isinstance(row.get("pair_nonweak"), (bool, np.bool_)):
                raise ValueError("response rows require a Boolean pair_nonweak flag")
            if row["pair_nonweak"] and not row["truth_eligible"]:
                raise ValueError("a nonweak response pair requires eligible truth")


def _ratios(rows):
    keys = tuple(key for key in GROUPING if key != "arm")
    cohorts = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = tuple(row[k] for k in keys)
        slot = tuple(row[k] for k in ("cell", "parent", "query"))
        cohorts[key][row["arm"]][slot] = row
    result = []
    for key, arms in sorted(cohorts.items()):
        if "generic" not in arms:
            continue
        for reference in ("structured", "hold"):
            if reference not in arms:
                continue
            generic, baseline = arms["generic"], arms[reference]
            common = generic.keys() & baseline.keys()
            mismatched = sum(
                generic[slot]["truth_eligible"] != baseline[slot]["truth_eligible"]
                for slot in common
            )
            origin_mismatches = sum(
                generic[slot].get("origin", generic[slot]["query"])
                != baseline[slot].get("origin", baseline[slot]["query"])
                for slot in common
            )
            is_response = dict(zip(keys, key, strict=True))["kind"] == "response"
            weak_mismatches = sum(
                generic[slot].get("pair_nonweak") != baseline[slot].get("pair_nonweak")
                for slot in common
            )
            comparable = (
                generic.keys() == baseline.keys()
                and not mismatched
                and not origin_mismatches
                and not weak_mismatches
            )
            selected_generic = [
                row
                for row in generic.values()
                if not is_response or row["pair_nonweak"]
            ]
            selected_baseline = [
                row
                for row in baseline.values()
                if not is_response or row["pair_nonweak"]
            ]
            a, b = _summary(selected_generic), _summary(selected_baseline)
            numerator, denominator = (
                a["available_truth_rmse"],
                b["available_truth_rmse"],
            )
            zero = comparable and denominator is not None and denominator == 0
            conditional_ratio = (
                numerator / denominator
                if comparable
                and numerator is not None
                and denominator is not None
                and not zero
                else None
            )
            ratio_overflow = conditional_ratio is not None and not np.isfinite(
                conditional_ratio
            )
            if ratio_overflow:
                conditional_ratio = None
            complete = (
                comparable
                and a["status"] == b["status"] == "complete"
                and all(row["truth_eligible"] for row in generic.values())
            )
            ratio = conditional_ratio if complete else None
            result.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "numerator_arm": "generic",
                    "reference_arm": reference,
                    "comparable_cohort": comparable,
                    "unmatched_queries": len(generic.keys() ^ baseline.keys()),
                    "truth_eligibility_mismatches": mismatched,
                    "origin_mismatches": origin_mismatches,
                    "weak_eligibility_mismatches": weak_mismatches,
                    "cohort": "nonweak_signed_pairs"
                    if is_response
                    else "all_truth_slots",
                    "cohort_queries": len(selected_generic),
                    "excluded_weak_queries": sum(
                        row["truth_eligible"] and not row["pair_nonweak"]
                        for row in generic.values()
                    )
                    if is_response
                    else 0,
                    "zero_reference": zero,
                    "nonfinite_ratio": bool(ratio_overflow),
                    "ratio": ratio,
                    "improvement_fraction": None if ratio is None else 1 - ratio,
                    "available_truth_ratio": conditional_ratio,
                    "available_truth_improvement_fraction": (
                        None if conditional_ratio is None else 1 - conditional_ratio
                    ),
                }
            )
    return result


def aggregate(rows):
    """Return JSON-safe summary, parent, cell, and matched-reference ratio lists.

    Every planned query must be present, including ineligible truth slots. The
    unqualified RMSE is null if truth is missing or an eligible prediction fails.
    ``available_truth_rmse`` conditions only on valid truth; finite-subset fields
    are diagnostics and never silently replace the primary score. Ratios require
    identical planned query identities, origins and eligibility masks across arms.
    Branches average within ``origin``, then origins within parent, parents within
    cell, and cells within scope. A missing origin field treats each query as one
    origin; response callers must supply it to preserve partial-origin weights.
    Response rows also require ``pair_nonweak`` from the matched lower/upper
    direction reduction. Only response ratios use that filter; raw errors retain
    every eligible weak query. The endpoint weak criterion applies to both
    endpoint and cumulative ratio rows at the corresponding horizon.
    """
    rows = list(rows)
    _validate_rows(rows)
    output = {"summaries": [], "parents": [], "cells": []}
    for name, detail in (
        ("summaries", ()),
        ("parents", ("cell", "parent")),
        ("cells", ("cell",)),
    ):
        keys = GROUPING + detail
        ordered = sorted(rows, key=lambda row: tuple(row[key] for key in keys))
        for values, members in groupby(
            ordered, key=lambda row: tuple(row[key] for key in keys)
        ):
            output[name].append(
                {**dict(zip(keys, values, strict=True)), **_summary(list(members))}
            )
    output["paired_ratios"] = _ratios(rows)
    return output


def score_response_directions(
    prediction, target, valid, *, dt_s, horizons_s, thresholds
):
    """Endpoint directions for a lower/upper pair of already-differenced responses.

    Arrays have shape (2,H,15); valid has shape (2,H). Both signed truth vectors
    must exceed the group's max-absolute-component threshold. Cosine is undefined
    for a zero predicted vector. Sign agreement uses only truth components whose
    absolute magnitude exceeds that same threshold. Weak pairs remain in rows and
    all raw-error scoring; this helper never filters the main metrics.
    """
    prediction, target, valid = (
        np.asarray(prediction),
        np.asarray(target),
        np.asarray(valid),
    )
    if prediction.ndim != 3 or prediction.shape[0] != 2 or valid.ndim != 2:
        raise ValueError("responses require (2,H,15) arrays and (2,H) validity")
    if prediction.shape != target.shape or valid.shape != prediction.shape[:2]:
        raise ValueError("paired response shapes disagree")
    pairs = [_arrays(prediction[i], target[i], valid[i]) for i in range(2)]
    for group in GROUPS:
        if (
            group not in thresholds
            or not np.isfinite(thresholds[group])
            or thresholds[group] <= 0
        ):
            raise ValueError(
                "each group requires a positive finite weak-response threshold"
            )
    rows = []
    for step, horizon in _horizons(dt_s, horizons_s):
        eligible = step <= prediction.shape[1] and bool(valid[:, :step].all())
        for group, columns in GROUPS.items():
            threshold = thresholds[group]
            nonweak = eligible and all(
                np.max(np.abs(actual[step - 1, columns])) > threshold
                for _, actual, _ in pairs
            )
            for sign, (predicted, actual, _) in zip(
                ("lower", "upper"), pairs, strict=True
            ):
                row = {
                    "horizon_s": horizon,
                    "group": group,
                    "sign": sign,
                    "truth_eligible": eligible,
                    "pair_nonweak": bool(nonweak),
                    "prediction_finite": False,
                    "cosine": None,
                    "zero_prediction": False,
                    "sign_components": 0,
                    "sign_agreements": 0,
                    "sign_agreement": None,
                }
                if eligible:
                    p, t = predicted[step - 1, columns], actual[step - 1, columns]
                    finite = bool(np.isfinite(p).all())
                    row["prediction_finite"] = finite
                    if nonweak:
                        selected = np.abs(t) > threshold
                        row["sign_components"] = int(selected.sum())
                        if finite:
                            row["sign_agreements"] = int(
                                (np.sign(p[selected]) == np.sign(t[selected])).sum()
                            )
                            row["sign_agreement"] = (
                                row["sign_agreements"] / row["sign_components"]
                            )
                            scale = np.max(np.abs(p))
                            row["zero_prediction"] = bool(scale == 0)
                            if scale:
                                pn, tn = p / scale, t / np.max(np.abs(t))
                                row["cosine"] = float(
                                    np.clip(
                                        np.dot(pn, tn)
                                        / (np.linalg.norm(pn) * np.linalg.norm(tn)),
                                        -1,
                                        1,
                                    )
                                )
                rows.append(row)
    return rows
