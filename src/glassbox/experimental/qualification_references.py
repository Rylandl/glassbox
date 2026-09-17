"""Describe frozen evidence without changing a gate or fitting a model.

The qualification runner supplies atomically validated input bytes. This module
never opens their original paths and does not reinterpret historical acceptance.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping

import numpy as np

_CHANNELS = {"velocity_m_s": slice(0, 3), "body_rate_rad_s": slice(3, 6)}
_LEGACY_METRICS = {
    "velocity_m_s": "velocity_rmse_m_s",
    "body_rate_rad_s": "body_rate_rmse_rad_s",
}


def prefix_vector_p95(prediction, targets, recording_ids):
    """Worst recording's nearest-rank p95 of whole-prefix vector errors.

    Each window contributes its largest Euclidean three-vector error over the
    sampled horizon. Recording percentiles are computed before taking a maximum;
    neither component averaging nor pooling recordings enters this statistic.
    """
    prediction, targets = np.asarray(prediction), np.asarray(targets)
    ids = np.asarray(recording_ids)
    if (
        targets.ndim != 3
        or prediction.shape != targets.shape
        or min(targets.shape[:2]) < 1
        or targets.shape[2] < 6
        or ids.ndim != 1
        or len(ids) != len(targets)
    ):
        raise ValueError("forecasts require matching nonempty [N,H,>=6] and [N] ids")
    if prediction.dtype.kind not in "biuf" or targets.dtype.kind not in "biuf":
        raise ValueError("forecast values must be real numbers")
    if not np.isfinite(prediction).all() or not np.isfinite(targets).all():
        raise ValueError("forecast values must be finite")
    if ids.dtype.kind not in "US":
        raise ValueError("recording identities must be nonempty strings")
    if ids.dtype.kind == "S":
        ids = ids.astype(str)
    if any(not value for value in ids):
        raise ValueError("recording identities must be nonempty strings")
    error = np.asarray(prediction, dtype=float) - np.asarray(targets, dtype=float)
    result = {}
    for metric, channels in _CHANNELS.items():
        with np.errstate(over="ignore", invalid="ignore"):
            largest = np.linalg.norm(error[..., channels], axis=-1).max(axis=1)
        if not np.isfinite(largest).all():
            raise ValueError("vector error is not finite")
        recordings = {}
        for identity in sorted(set(ids.tolist())):
            values = np.sort(largest[ids == identity])
            rank = math.ceil(0.95 * len(values))
            recordings[str(identity)] = {
                "windows": len(values),
                "nearest_rank": rank,
                "prefix_vector_p95": float(values[rank - 1]),
            }
        worst = max(row["prefix_vector_p95"] for row in recordings.values())
        result[metric] = {
            "worst_recording_prefix_vector_p95": worst,
            "worst_recordings": [
                name
                for name, row in recordings.items()
                if row["prefix_vector_p95"] == worst
            ],
            "recordings": recordings,
        }
    return result


def _json(inputs, name):
    return json.loads(inputs[name])


def _legacy_decision(decision):
    return {
        key: decision[key]
        for key in (
            "manifest",
            "accepted",
            "rule_met",
            "reference_compared",
            "reference_sha256",
            "gating_rule_breaches",
        )
    }


def comparative_progress(pairs):
    """Describe both directions symmetrically; do not invent a promotion score.

    Each pair contains generic and best-structured errors in the same two
    physical metrics. A corpus win count is descriptive and ignores magnitude;
    it is not a probability of generalizing to a new system.
    """
    generic_better, structured_better, mixed = [], [], []
    generic_dominates = structured_dominates = True
    for name, (generic, structured) in sorted(pairs.items()):
        generic, structured = np.asarray(generic), np.asarray(structured)
        if generic.shape != (2,) or structured.shape != (2,):
            raise ValueError("each corpus comparison requires both physical metrics")
        if not np.isfinite(generic).all() or not np.isfinite(structured).all():
            raise ValueError("comparison errors must be finite")
        generic_dominates &= bool(np.all(generic <= structured))
        structured_dominates &= bool(np.all(structured <= generic))
        if np.all(generic < structured):
            generic_better.append(name)
        elif np.all(structured < generic):
            structured_better.append(name)
        else:
            mixed.append(name)
    return dict(
        corpora=len(pairs),
        generic_better_on_both_metrics=generic_better,
        structured_better_on_both_metrics=structured_better,
        mixed_or_tied=mixed,
        generic_no_worse_on_every_metric=bool(pairs) and generic_dominates,
        structured_no_worse_on_every_metric=bool(pairs) and structured_dominates,
        promotion_decided=False,
        meaning=(
            "Progress and universal replacement are separate. If each predictor "
            "loses somewhere, neither dominates; reversing the incumbent does not "
            "turn a one-corpus win into an all-case no-regression pass. No tradeoff "
            "weights or new promotion threshold are chosen from these known scores. "
            "The generic recipe is shared, with separately fitted parameters per "
            "system; structured comparators are fitted per corpus too."
        ),
    )


def _platform_report(plan, inputs):
    manifest = _json(inputs, "platform/manifest.json")
    rows = _json(inputs, "platform/results.json")
    decision = _json(inputs, "platform/decision.json")
    entries = {entry["name"]: entry for entry in manifest["corpora"]}
    if sorted(row["corpus"] for row in rows) != sorted(entries):
        raise ValueError("saved platform results differ from declared corpora")
    corpora, pairs = {}, {}
    for row in rows:
        name = row["corpus"]
        entry = entries[name]
        predictors = {
            "generic": "generic_prediction",
            "hold_current": "hold_prediction",
        }
        predictors.update(
            {
                f"structured:{arm}": f"structured_{arm}_prediction"
                for arm in entry["structured"]["arms"]
            }
        )
        with np.load(
            io.BytesIO(inputs[f"platform/{name}/evaluation.npz"]), allow_pickle=False
        ) as data:
            ids, targets = data["recording_ids"], data["targets"]
            expected_predictions = set(predictors.values())
            actual_predictions = {
                key for key in data.files if key.endswith("_prediction")
            }
            if actual_predictions != expected_predictions:
                raise ValueError("saved platform predictor set differs from manifest")
            measured = {
                predictor: prefix_vector_p95(data[key], targets, ids)
                for predictor, key in predictors.items()
            }
            if len(targets) != row["evaluation_rows"]:
                raise ValueError("saved platform row count differs from results")
            if targets.shape[1] != row["horizon_steps"]:
                raise ValueError("saved platform horizon differs from results")
        ceiling = entry["allowance"]
        comparisons = {}
        for metric, legacy_metric in _LEGACY_METRICS.items():
            generic = measured["generic"][metric]["worst_recording_prefix_vector_p95"]
            structured = {
                arm: values[metric]["worst_recording_prefix_vector_p95"]
                for arm, values in measured.items()
                if arm.startswith("structured:")
            }
            best = min(structured.values())
            limit = None if ceiling is None else ceiling[legacy_metric]
            comparisons[metric] = {
                "historical_descriptive_ceiling": limit,
                "generic_within_descriptive_ceiling": (
                    None if limit is None else generic <= limit
                ),
                "best_structured_prefix_vector_p95": best,
                "best_structured_arms": [
                    arm for arm, value in structured.items() if value == best
                ],
                "generic_at_or_below_structured": generic <= best,
                "legacy_pooled_final_step_component_rmse": row["generic"]["final_step"][
                    legacy_metric
                ],
            }
        corpora[name] = {
            "windows": row["evaluation_rows"],
            "horizon_steps": row["horizon_steps"],
            "horizon_s": row["horizon_s"],
            "generic_fingerprint": row["generic_fingerprint"],
            "recordings": row["recordings"],
            "predictors": measured,
            "informational_comparisons": comparisons,
        }
        pairs[name] = (
            [
                measured["generic"][metric]["worst_recording_prefix_vector_p95"]
                for metric in _CHANNELS
            ],
            [
                comparisons[metric]["best_structured_prefix_vector_p95"]
                for metric in _CHANNELS
            ],
        )
    return {
        "readout": plan["platform_readout"],
        "legacy_decision_unchanged": _legacy_decision(decision),
        "task_sufficiency_established": False,
        "comparative_progress": comparative_progress(pairs),
        "meaning": (
            "Historical ceilings describe older model errors. They are not task-derived "
            "requirements; IDF and EPFL declare none. The old gate applies them to "
            "pooled final-step component RMSE, a different statistic. These matched "
            "prefix-vector percentiles are retrospective descriptions and decide no gate."
        ),
        "corpora": corpora,
    }


def _control_report(inputs):
    manifest = _json(inputs, "control/manifest.json")
    reference = _json(inputs, "control/reference.json")
    decision = _json(inputs, "control/decision.json")
    calibration = _json(inputs, "control/calibration.json")
    trials = {}
    for repetition in range(manifest["trial"]["repetitions"]):
        for arm in ("generic", "structured"):
            row = _json(inputs, f"control/trial-{repetition}/{arm}/trial.json")
            if row["repetition"] != repetition or row["arm"] != arm:
                raise ValueError("saved control trial identity differs from its key")
            trials[f"{repetition}-{arm}"] = {
                "tracking_rmse": row["tracking_rmse"],
                "application_criterion": row["pass_criterion"],
                "controller": row["controller"],
                "model_not_ready_intervals": row["model_not_ready_intervals"],
            }
    return {
        "legacy_decision_unchanged": _legacy_decision(decision),
        "reference_provenance": {
            key: reference[key]
            for key in ("manifest", "manifest_sha256", "source", "environment")
        },
        "generic_reference_fingerprint": reference["generic_fingerprint"],
        "generic_saved_fingerprint": calibration["generic_fingerprint"],
        "application_criterion": manifest["metrics"]["pass_criterion"],
        "historical_reference_meets_matched_rule": {
            repetition: {
                metric: values["reference_meets_rule"]
                for metric, values in metrics.items()
            }
            for repetition, metrics in decision["tracking_rmse"].items()
        },
        "trials": trials,
        "meaning": (
            "Frozen reference values are historical generic tracking errors. The same-run "
            "structured comparison is a separate rule; application success is separate "
            "again and did not gate historical acceptance. Both models share the task "
            "and simulator but have different planning horizons, action block counts, "
            "warm-up and uncertainty terms. Historical results compare complete pipelines "
            "and do not isolate mean-model accuracy."
        ),
    }


def reference_report(plan: dict, inputs: Mapping[str, bytes]):
    """Report semantics and saved-model readouts from validated immutable bytes."""
    if plan["platform_readout"]["percentile"] != 0.95:
        raise ValueError("the frozen readout requires the nearest-rank 95th percentile")
    if set(inputs) != set(plan["input_sha256"]):
        raise ValueError("input snapshots differ from the frozen input set")
    roles = {
        "synthetic": (
            "reference.json",
            "Historical generic scaled RMSE and delayed-input probe scores. Absolute "
            "caps are engineering regression guards on nine fixed synthetic systems, "
            "not task tolerances or evidence of unseen system-family generalization.",
        ),
        "platform": (
            "platform-reference.json",
            "Historical generic final-step component RMSE. Same-run best structured "
            "scores and descriptive old-model ceilings are separate comparisons. "
            "The same held-out recordings and origins are shared by the current arms.",
        ),
        "control": (
            "control-reference.json",
            "Historical generic tracking RMSE under control-v5; control-v4 numerical "
            "results were retained when declared excitation changed only the report. "
            "The incumbent met no matched-structured tracking metric.",
        ),
        "live": (
            "live-reference.json",
            "First live-v3 dithered-run generic errors after the swap. The no-worse "
            "rule compares the adopting arm with its own earlier segment, not the "
            "matched frozen arm at the same times. Swap timing changes the scored "
            "segments. Refitting wall-budget overruns are informational. This audit "
            "does not rerun live trials or establish bounded live improvement.",
        ),
        "evidence": (
            "evidence-reference.json",
            "First generic-v3 empirical coverage, stored as evidence-v1 and retained "
            "under evidence-v2. Regression measures distance outside the fixed band; "
            "only previously in-band cases gate the band itself. These are marginal "
            "channel-group fractions, not joint trajectory confidence or task safety.",
        ),
    }
    result = {
        "classification": plan["classification"]["reference_audit"],
        "acceptance_recomputed": False,
        "provenance": {
            "accepted_executable_source": plan["accepted_executable_source"],
            "baseline_runs": plan["baseline_runs"],
            "source_sha256": plan["source_sha256"],
            "input_sha256": plan["input_sha256"],
            "historical_limit": (
                "Historical reference files record prose, environment and model "
                "fingerprints rather than explicit fitting-source commits. The source "
                "and input hashes here pin this audit; the accepted executable source "
                "is the qualification plan's declared baseline provenance."
            ),
        },
        "reference_roles": {
            tier: {
                "file": f"docs/harness/{filename}",
                "sha256": plan["source_sha256"][f"docs/harness/{filename}"],
                "meaning": meaning,
            }
            for tier, (filename, meaning) in roles.items()
        },
        "platform": _platform_report(plan, inputs),
        "control": _control_report(inputs),
    }
    # Fail closed if a saved summary contains nonfinite or non-JSON data.
    json.dumps(result, allow_nan=False)
    return result
