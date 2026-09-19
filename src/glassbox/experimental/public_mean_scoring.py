"""Fixed-reference synthetic reductions for public-mean-qualification-v1.

This module never fits or predicts. Required scores use authenticated historical
units; candidate normalization affects only explicitly labeled diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL_SHA256 = "d3a8d0eeeeab2aac923cad7f1f5fe1c8a49fc0ffa765e84b3d7c6bc80cec8f99"
_PROTOCOL = "docs/harness/public-mean-qualification-v1.json"
_SPEC = "docs/harness/public-mean-qualification-v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, expected):
    if _sha(path) != expected:
        raise ValueError(f"source anchor differs: {path}")
    return json.loads(Path(path).read_text())


def specification(repository):
    """Authenticate the frozen case/cap roster, not a caller-selected policy."""
    root = Path(repository)
    protocol = _read(root / _PROTOCOL, PROTOCOL_SHA256)
    deps = {row["path"]: row["sha256"] for row in protocol["protocol_dependencies"]}
    policy_path = f"{_SPEC}/public-mean-qualification-policy-input.json"
    anchors_path = f"{_SPEC}/public-v4-synthetic-scoring-anchors.json"
    policy = _read(root / policy_path, deps[policy_path])
    anchors = _read(root / anchors_path, deps[anchors_path])
    cases = policy["synthetic"]["cases"]
    if len(cases) != 27 or len({case["name"] for case in cases}) != 27:
        raise ValueError("frozen synthetic case roster differs")
    if sum(len(case["regimes"]) for case in cases) != 51:
        raise ValueError("frozen synthetic regime roster differs")
    if set(anchors["cases"]) != {case["name"] for case in cases}:
        raise ValueError("frozen scale anchors differ from case roster")
    return cases, anchors


def authenticate_reference_scales(repository, artifact_root):
    """Verify all 88 anchors and independently reconstruct all 27 fixed scales.

    Read the old NPZ arrays directly: the current public loader intentionally
    refuses that older recipe. No historical model is initialized or invoked.
    """
    cases, ledger = specification(repository)
    root = Path(artifact_root)
    for relative, expected in ledger["source_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"historical artifact anchor differs: {relative}")
    scales = {}
    for case in cases:
        name = case["name"]
        anchor = ledger["cases"][name]
        with np.load(root / anchor["source_path"], allow_pickle=False) as archive:
            past = archive["train_past_states"]
            future = archive["train_future_states"]
            saved = archive["norm_state_scale"]
        if past.shape[:2] != (384, 11) or future.shape != (384, 5, past.shape[-1]):
            raise ValueError(f"historical scale cache shape differs: {name}")
        current = np.concatenate((past, future), axis=1)[:, 10:-1, :]
        raw = current.std(axis=(0, 1), ddof=0)
        scale = np.where(raw > 1e-8, raw, 1.0)
        expected = np.asarray(anchor["reference_state_scale"], dtype=np.float64)
        if (
            scale.dtype.str != anchor["dtype"]
            or list(scale.shape) != anchor["shape"]
            or not np.array_equal(scale, saved)
            or not np.array_equal(scale, expected)
            or hashlib.sha256(scale.tobytes()).hexdigest()
            != anchor["array_bytes_sha256"]
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError(f"fixed reference scale does not reconstruct: {name}")
        scale.setflags(write=False)
        scales[name] = scale
    return scales


def _scale(value, width):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (width,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("scoring scales must be finite positive channel vectors")
    return value


def _reduce(error, scale):
    squared = np.mean(error**2, axis=0)
    normalized = squared / scale**2
    return {
        "horizon_rmse": np.sqrt(np.mean(normalized, axis=1)).tolist(),
        "overall_rmse": float(np.sqrt(np.mean(normalized))),
        "channel_rmse": np.sqrt(squared).tolist(),
    }


def forecast_scores(
    prediction, target, reference_scale, candidate_scale, recording_ids
):
    """Reduce saved physical forecasts in fixed units and diagnostic learned units."""
    prediction, target = (
        np.asarray(value, dtype=np.float64) for value in (prediction, target)
    )
    if (
        prediction.ndim != 3
        or prediction.shape != target.shape
        or prediction.shape[1] != 5
        or prediction.shape[0] < 1
        or prediction.shape[2] < 1
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError(
            "synthetic scores require finite complete [query,5,channel] arrays"
        )
    ids = list(recording_ids)
    if len(ids) != len(prediction) or any(not isinstance(v, str) or not v for v in ids):
        raise ValueError("synthetic recording identities must align with every query")
    reference = _scale(reference_scale, prediction.shape[2])
    candidate = _scale(candidate_scale, prediction.shape[2])
    error = prediction - target
    result = {
        "reference_state_scale": reference.tolist(),
        "candidate_state_scale": candidate.tolist(),
        "candidate_to_reference_scale": (candidate / reference).tolist(),
        "fixed_reference": _reduce(error, reference),
        "legacy_candidate_scaled": _reduce(error, candidate),
        "recordings": {},
    }
    for name in sorted(set(ids)):
        selected = np.asarray(ids) == name
        result["recordings"][name] = {
            "queries": int(selected.sum()),
            "fixed_reference": _reduce(error[selected], reference),
            "legacy_candidate_scaled": _reduce(error[selected], candidate),
        }
    # Finite inputs can still overflow a physical reduction. Such a case fails.
    try:
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ValueError("nonfinite synthetic score reduction") from exc
    return result


def verify_forecast_scores(
    saved, prediction, target, reference_scale, candidate_scale, recording_ids
):
    """Recompute every required/diagnostic reduction; reject denominator substitution."""
    actual = forecast_scores(
        prediction, target, reference_scale, candidate_scale, recording_ids
    )
    if saved != actual:
        raise ValueError("saved synthetic scores differ from fixed-reference reduction")
    return actual


def probe_scores(prediction, target, reference_scale):
    """Keep the original physical first-step requirement and separately scaled display."""
    prediction, target = (np.asarray(v, dtype=np.float64) for v in (prediction, target))
    if (
        prediction.shape != (2, 5, 1)
        or target.shape != prediction.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError(
            "probe requires two complete finite five-step scalar responses"
        )
    scale = _scale(reference_scale, 1)[0]
    physical = np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 2)))
    if not np.isfinite(physical).all():
        raise ValueError("nonfinite physical probe reduction")
    return {
        "physical_rmse": physical.tolist(),
        "physical_first_step_limit": 0.05,
        "fixed_reference_rmse": (physical / scale).tolist(),
        "fixed_reference_first_step_limit": float(0.05 / scale),
        "physical_first_step_pass": bool(physical[0] <= 0.05),
    }


def capability_decision(repository, completed_fits, rows, probes):
    """Apply only the prospective absolute-cap conjunction, with an exact roster."""
    cases, ledger = specification(repository)
    expected_cases = {case["name"] for case in cases}
    fits = list(completed_fits)
    if len(fits) != 27 or set(fits) != expected_cases:
        raise ValueError("capability needs exactly 27 completed case fits")
    caps = {
        (case["name"], regime): case["caps"][regime]
        for case in cases
        for regime in case["regimes"]
    }
    actual = [(row["case"], row["regime"]) for row in rows]
    if len(actual) != 51 or len(set(actual)) != 51 or set(actual) != set(caps):
        raise ValueError("capability needs the exact 51 case/regime identities")
    comparisons = []
    for row in rows:
        reference = np.asarray(
            row["scores"].get("reference_state_scale"), dtype=np.float64
        )
        expected = np.asarray(
            ledger["cases"][row["case"]]["reference_state_scale"], dtype=np.float64
        )
        if reference.shape != expected.shape or not np.array_equal(reference, expected):
            raise ValueError(
                "required scoring denominator differs from the frozen reference"
            )
        metrics = row["scores"]["fixed_reference"]
        values = np.asarray(metrics["horizon_rmse"], dtype=np.float64)
        overall = metrics["overall_rmse"]
        if (
            values.shape != (5,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or type(overall) not in (float, int)
            or not np.isfinite(overall)
            or overall < 0
        ):
            raise ValueError(
                "every regime needs exactly five finite nonnegative scores"
            )
        cap = caps[row["case"], row["regime"]]
        comparisons.extend(
            {
                "case": row["case"],
                "regime": row["regime"],
                "horizon_step": h + 1,
                "value": float(value),
                "limit": cap,
                "pass": bool(value <= cap),
            }
            for h, value in enumerate(values)
        )
    if len(probes) != 3 or {p["seed"] for p in probes} != {101, 202, 303}:
        raise ValueError("capability needs exactly three declared physical probes")
    probe_pass = []
    for probe in probes:
        values = np.asarray(probe["scores"]["physical_rmse"], dtype=np.float64)
        if values.shape != (5,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("physical probe roster is incomplete or nonfinite")
        probe_pass.append(bool(values[0] <= 0.05))
    return {
        "fixed_reference_absolute_capability_pass": all(
            row["pass"] for row in comparisons
        )
        and all(probe_pass),
        "case_count": len(fits),
        "regime_count": len(rows),
        "horizon_checks": comparisons,
        "probe_pass": probe_pass,
        "historical_flags_are_diagnostic": True,
    }
