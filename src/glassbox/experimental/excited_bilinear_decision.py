"""Frozen paired residual and angular-repair decisions for bilinear ablation."""

import hashlib
import json
import warnings
from copy import deepcopy

import numpy as np

from .state_input_decision import _index, _nanmean, _ratio, _within
from .state_input_decision import reduce as paired_reduce
from .two_simulator_metrics import aggregate

_PAIRS = {
    "bilinear_public_progress": ("candidate", "baseline"),
    "quadratic_gain_retention": ("candidate", "quadratic"),
    "quadratic_public_context": ("quadratic", "baseline"),
}
_LIMITS = (
    "response_weighted_geometric_mean_ratio_max",
    "factual_weighted_geometric_mean_ratio_max",
    "each_simulator_primary_factual_and_response_geometric_mean_ratio_max",
    "each_simulator_scope_factual_and_response_geometric_mean_ratio_max",
    "each_simulator_250ms_primary_parent_rmse_p95_ratio_max",
)


def _comparison(rows, protocol, name, numerator, denominator):
    paired_protocol = deepcopy(protocol)
    policy = protocol["decision"]["comparisons"][name]
    if isinstance(policy, dict):
        if (policy["numerator"], policy["denominator"]) != (numerator, denominator):
            raise ValueError("frozen bilinear comparison arms differ")
        paired_protocol["decision"]["accept_research_candidate"].update(
            {key: policy[key] for key in _LIMITS}
        )
    paired_rows = {
        sim: [
            dict(row, arm="baseline" if row["arm"] == denominator else "candidate")
            for row in values
            if row["arm"] in (numerator, denominator)
        ]
        for sim, values in rows.items()
    }
    result = paired_reduce(
        {sim: aggregate(values) for sim, values in paired_rows.items()},
        paired_protocol,
        rows=paired_rows,
    )
    result.update(
        numerator_arm=numerator,
        denominator_arm=denominator,
        baseline_fields_mean=denominator,
        candidate_fields_mean=numerator,
        limits={
            key: paired_protocol["decision"]["accept_research_candidate"][key]
            for key in _LIMITS
        },
        remaining_promotion_requirements=protocol["decision"]["public_promotion"],
        pairwise_predictions_available=result["checks"]["finite_eligible_predictions"],
    )
    return result


def _distribution(values):
    finite = np.isfinite(values)
    return {
        "draws": [float(v) if np.isfinite(v) else None for v in values],
        "available_draws": int(finite.sum()),
        "unavailable_draws": int((~finite).sum()),
        "percentile_95_interval": None
        if not finite.any()
        else np.quantile(values[finite], [0.025, 0.975], method="linear").tolist(),
    }


def _angular_bootstrap(summaries, protocol):
    """Reuse every frozen parent index; only the reported statistic changes."""
    policy = protocol["decision"]["aggregation"]
    target = protocol["decision"]["angular_repair"]
    arms = (target["denominator"], target["numerator"])
    kinds = ("factual", "response")
    indexed = _index(
        (
            row
            for summary in summaries.values()
            for row in summary["parents"]
            if row["arm"] in arms
            and row["statistic"] == "endpoint"
            and row["horizon_s"] == target["horizon_s"]
            and row["group"] == target["group"]
        ),
        ("simulator", "scope", "cell", "parent", "arm", "kind"),
    )
    rng = np.random.default_rng(20260918)
    count = 1000
    draw_hash = hashlib.sha256()
    target_samples = []
    cell_means = []
    for sim in sorted(policy["simulator_weights"]):
        for scope in sorted(policy["scope_weights"]):
            cells = sorted(
                c["id"] for c in protocol["cells"][sim] if c["group"] == scope
            )
            for cell in cells:
                parents = sorted(
                    r["id"]
                    for r in protocol["recordings"]
                    if r["simulator"] == sim
                    and r["cell"] == cell
                    and r["role"] == "test"
                )
                chosen = rng.integers(
                    len(parents), size=(count, len(parents)), dtype=np.int64
                )
                draw_hash.update(
                    json.dumps(
                        [sim, scope, cell, parents], separators=(",", ":")
                    ).encode()
                )
                draw_hash.update(chosen.astype("<i8").tobytes())
                if (sim, scope) != (target["simulator"], target["scope"]):
                    continue
                values = np.asarray(
                    [
                        [
                            [
                                indexed[sim, scope, cell, parent, arm, kind][
                                    "available_truth_mse"
                                ]
                                for kind in kinds
                            ]
                            for arm in arms
                        ]
                        for parent in parents
                    ],
                    dtype=float,
                )
                sampled = values[chosen]
                target_samples.append(sampled)
                cell_means.append(_nanmean(sampled, axis=1))
    mean_rmse = np.sqrt(_nanmean(np.stack(cell_means), axis=0))
    # Missing-truth parents stay in the draws, including wholly unavailable draws.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parent_p95 = np.nanquantile(
            np.sqrt(np.concatenate(target_samples, axis=1)),
            0.95,
            axis=1,
            method="linear",
        )
    floor = policy["floors"][target["group"]]
    return {
        "seed": 20260918,
        "requested_draws": count,
        "parent_draws_sha256": draw_hash.hexdigest(),
        "conditional_on_available_truth": True,
        "intervals_condition_on_available_draws": True,
        "point_estimates_only_for_acceptance": True,
        "statistics": {
            f"{kind}_{statistic}": _distribution(
                np.maximum(values[:, 1, ki], floor)
                / np.maximum(values[:, 0, ki], floor)
            )
            for ki, kind in enumerate(kinds)
            for statistic, values in (
                ("aggregate_rmse", mean_rmse),
                ("parent_rmse_p95", parent_p95),
            )
        },
    }


def _angular_repair(summaries, protocol, retention):
    target = protocol["decision"]["angular_repair"]
    if (target["numerator"], target["denominator"]) != ("candidate", "quadratic"):
        raise ValueError("frozen angular repair arms differ")
    physicals = {}
    tails = {}
    checks = {
        "matching_planned_queries_and_truth": retention["checks"][
            "matching_planned_queries_and_truth"
        ],
        "finite_eligible_predictions": retention["checks"][
            "finite_eligible_predictions"
        ],
    }
    floor = protocol["decision"]["aggregation"]["floors"][target["group"]]
    for kind in ("factual", "response"):
        selected = [
            row
            for row in retention["physical_comparisons"]
            if all(
                row[key] == target[key]
                for key in ("simulator", "scope", "horizon_s", "group")
            )
            and row["kind"] == kind
        ]
        if len(selected) != 1:
            raise ValueError("missing or duplicate angular repair comparison")
        physicals[kind] = selected[0]
        selected_tails = [
            row
            for row in retention["parent_error_tails"]
            if all(
                row[key] == target[key]
                for key in ("simulator", "scope", "horizon_s", "group")
            )
            and row["kind"] == kind
            and row["statistic"] == "endpoint"
        ]
        indexed_tails = _index(selected_tails, ("arm",))
        if set(indexed_tails) != {("baseline",), ("candidate",)}:
            raise ValueError("missing angular repair parent tails")
        base, candidate = (indexed_tails[(arm,)] for arm in ("baseline", "candidate"))
        tail_ratio = _ratio(candidate["p95"], base["p95"], floor)
        tails[kind] = {
            "baseline": base,
            "candidate": candidate,
            "baseline_p95": base["p95"],
            "candidate_p95": candidate["p95"],
            "ratio": tail_ratio,
        }
        checks[f"{kind}_aggregate_rmse"] = _within(
            selected[0]["ratio"], target[f"{kind}_aggregate_rmse_ratio_max"]
        )
        checks[f"{kind}_parent_rmse_p95"] = _within(
            tail_ratio, target[f"{kind}_parent_rmse_p95_ratio_max"]
        )
    bootstrap = (
        _angular_bootstrap(summaries, protocol)
        if checks["finite_eligible_predictions"]
        else {"status": "unavailable_due_to_prediction_failure"}
    )
    return {
        "numerator_arm": "candidate",
        "denominator_arm": "quadratic",
        "baseline_fields_mean": "quadratic",
        "candidate_fields_mean": "candidate",
        "policy": deepcopy(target),
        "physical_comparisons": physicals,
        "parent_tail_comparisons": tails,
        "checks": checks,
        "residual_criteria_pass": all(checks.values()),
        "bootstrap": bootstrap,
    }


def reduce(summaries, protocol, rows=None):
    """Report pairwise progress independently; external qualification stays pending.

    Missing metric slots or mismatched truth masks are integrity errors. Failed
    predictions retain their slots and only invalidate comparisons using that arm.
    The contextual quadratic/public decision is not a mechanism-success gate.
    """
    if rows is None:
        raise ValueError("full metric rows are required for paired decisions")
    if set(summaries) != set(rows) or any(
        aggregate(values) != summaries[sim] for sim, values in rows.items()
    ):
        raise ValueError("summary differs from full metric rows")
    if set(protocol["decision"]["comparisons"]) != set(_PAIRS):
        raise ValueError("frozen comparison roster differs")
    comparisons = {
        name: _comparison(rows, protocol, name, numerator, denominator)
        for name, (numerator, denominator) in _PAIRS.items()
    }
    repair = _angular_repair(
        summaries, protocol, comparisons["quadratic_gain_retention"]
    )
    hashes = {
        name: comparison["bootstrap"].get("parent_draws_sha256")
        for name, comparison in comparisons.items()
    }
    hashes["angular_repair"] = repair["bootstrap"].get("parent_draws_sha256")
    available_hashes = {value for value in hashes.values() if value is not None}
    if len(available_hashes) > 1:
        raise ValueError("pairwise bootstrap parent draws differ")
    checks = {
        "bilinear_public_progress": comparisons["bilinear_public_progress"][
            "residual_criteria_pass"
        ],
        "quadratic_gain_retention": comparisons["quadratic_gain_retention"][
            "residual_criteria_pass"
        ],
        "angular_repair": repair["residual_criteria_pass"],
        "finite_eligible_predictions": all(
            c["checks"]["finite_eligible_predictions"] for c in comparisons.values()
        ),
        "matching_planned_queries_and_truth": all(
            c["checks"]["matching_planned_queries_and_truth"]
            for c in comparisons.values()
        ),
    }
    return {
        "protocol": protocol["id"],
        "comparisons": comparisons,
        "bilinear_public_progress": checks["bilinear_public_progress"],
        "quadratic_gain_retention": checks["quadratic_gain_retention"],
        "angular_repair": checks["angular_repair"],
        "quadratic_public_context": comparisons["quadratic_public_context"][
            "residual_criteria_pass"
        ],
        "angular_repair_comparison": repair,
        "checks": checks,
        "residual_criteria_pass": all(checks.values()),
        "diagnostic_mechanism_success": None,
        "verification_status": "requires_integrity_replay_tamper_and_focused_tests",
        "bootstrap_parent_draws_sha256": next(iter(available_hashes), None),
        "bootstrap_comparison_hashes": hashes,
        "available_bootstrap_parent_draws_match": bool(available_hashes),
        "shared_bootstrap_parent_draws_verified": all(hashes.values())
        and len(available_hashes) == 1,
        "public_promotion": False,
        "qualification": protocol["decision"]["public_promotion"],
    }
