"""Frozen five-pair decisions for fixed initial-channel loss balancing."""

from copy import deepcopy

from .excited_bilinear_decision import _angular_repair as _pinned_angular_repair
from .excited_bilinear_decision import _comparison
from .state_input_decision import validate_cohorts
from .two_simulator_metrics import aggregate

_PAIRS = {
    "balanced_public_progress": ("candidate", "baseline"),
    "anchored_gain_retention": ("candidate", "anchored"),
    "quadratic_gain_retention": ("candidate", "quadratic"),
    "anchored_public_context": ("anchored", "baseline"),
    "quadratic_public_context": ("quadratic", "baseline"),
}
_REQUIRED = (
    "balanced_public_progress",
    "anchored_gain_retention",
    "quadratic_gain_retention",
)
_ARMS = ("baseline", "quadratic", "anchored", "candidate", "hold")


def _full_cohorts(rows, protocol):
    """Validate every arm's slots and masks without gating on unrelated failures."""
    if set(rows) != set(protocol["cells"]) or set(rows) != set(
        protocol["decision"]["aggregation"]["simulator_weights"]
    ):
        raise ValueError("frozen simulator roster differs")
    for sim, values in rows.items():
        if set(row["arm"] for row in values) != set(_ARMS):
            raise ValueError(f"{sim}: frozen five-arm roster differs")
        if any(row["simulator"] != sim for row in values):
            raise ValueError(f"{sim}: metric simulator differs")
    output = {}
    for arm in _ARMS[1:]:
        paired = {
            sim: [
                dict(row, arm="candidate" if row["arm"] == arm else "baseline")
                for row in values
                if row["arm"] in ("baseline", arm)
            ]
            for sim, values in rows.items()
        }
        for sim, cohort in validate_cohorts(paired, protocol).items():
            if sim not in output:
                output[sim] = {
                    **cohort,
                    "arms": {"baseline": cohort["arms"]["baseline"]},
                }
            output[sim]["arms"][arm] = cohort["arms"]["candidate"]
    return output


def _anchored_angular_repair(rows, protocol, retention):
    """Apply the pinned statistic with a private view of the actual anchored arm."""
    target = protocol["decision"]["angular_repair"]
    if (target["numerator"], target["denominator"]) != ("candidate", "anchored"):
        raise ValueError("frozen angular repair must compare candidate to anchored")
    view = deepcopy(protocol)
    view["decision"]["angular_repair"]["denominator"] = "quadratic"
    summaries = {
        sim: aggregate(
            [
                dict(row, arm="quadratic" if row["arm"] == "anchored" else "candidate")
                for row in values
                if row["arm"] in ("candidate", "anchored")
            ]
        )
        for sim, values in rows.items()
    }
    result = _pinned_angular_repair(summaries, view, retention)
    result.update(
        denominator_arm="anchored",
        baseline_fields_mean="anchored",
        policy=deepcopy(target),
    )
    return result


def reduce(summaries, protocol, rows=None):
    """Keep five paired reports, one direct mechanism target and qualification apart.

    Raw row rosters include unavailable predictions and missing truth. Corrupt or
    omitted slots are integrity errors; prediction failure only invalidates pairs
    involving that arm. Reference/public results are context, never extra vetoes.
    """
    if rows is None:
        raise ValueError("full metric rows are required for paired decisions")
    if set(summaries) != set(rows) or any(
        aggregate(values) != summaries[sim] for sim, values in rows.items()
    ):
        raise ValueError("summary differs from full metric rows")
    if set(protocol["decision"]["comparisons"]) != set(_PAIRS):
        raise ValueError("frozen channel-balance comparison roster differs")
    cohorts = _full_cohorts(rows, protocol)
    comparisons = {
        name: _comparison(rows, protocol, name, numerator, denominator)
        for name, (numerator, denominator) in _PAIRS.items()
    }
    repair = _anchored_angular_repair(
        rows, protocol, comparisons["anchored_gain_retention"]
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
        **{name: comparisons[name]["residual_criteria_pass"] for name in _REQUIRED},
        "angular_repair": repair["residual_criteria_pass"],
        "finite_eligible_predictions": all(
            comparisons[name]["checks"]["finite_eligible_predictions"]
            for name in _REQUIRED
        ),
        "matching_planned_queries_and_truth": all(
            comparisons[name]["checks"]["matching_planned_queries_and_truth"]
            for name in _REQUIRED
        ),
    }
    return {
        "protocol": protocol["id"],
        "comparisons": comparisons,
        **{
            name: comparison["residual_criteria_pass"]
            for name, comparison in comparisons.items()
        },
        "angular_repair": checks["angular_repair"],
        "anchored_mechanism_progress": checks["anchored_gain_retention"]
        and checks["angular_repair"],
        "angular_repair_comparison": repair,
        "cohorts": cohorts,
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
