"""Frozen pairwise collection and same-data architecture decisions."""

from copy import deepcopy

from .state_input_decision import reduce as paired_reduce
from .two_simulator_metrics import aggregate

_PAIRS = {
    "public_collection": ("candidate", "baseline"),
    "quadratic_workflow": ("quadratic", "baseline"),
    "simpler_recipe_comparable": ("candidate", "quadratic"),
    "quadratic_material_value": ("quadratic", "candidate"),
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
            raise ValueError("frozen architecture comparison arms differ")
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
    # Preserve physical scores, guards, tails and uncertainty, but describe the
    # actual arms and this iteration's qualification scope explicitly.
    result["numerator_arm"], result["denominator_arm"] = numerator, denominator
    result["baseline_fields_mean"] = denominator
    result["candidate_fields_mean"] = numerator
    result["limits"] = {
        key: paired_protocol["decision"]["accept_research_candidate"][key]
        for key in _LIMITS
    }
    result["remaining_promotion_requirements"] = protocol["decision"][
        "public_promotion"
    ]
    result["pairwise_predictions_available"] = result["checks"][
        "finite_eligible_predictions"
    ]
    return result


def reduce(summaries, protocol, *, rows):
    """Keep each valid pair's evidence independent of a third model failure.

    Missing metric slots or differing truth masks are integrity errors, not an
    unavailable model. A model failure keeps its planned rows and makes only
    the comparisons involving that arm fail. Preference flags additionally
    require their respective original-data workflow to show broad progress.
    """
    if set(summaries) != set(rows) or any(
        aggregate(values) != summaries[sim] for sim, values in rows.items()
    ):
        raise ValueError("summary differs from full metric rows")
    policies = protocol["decision"]["comparisons"]
    if set(policies) != set(_PAIRS):
        raise ValueError("frozen comparison roster differs")
    if (
        policies["simpler_recipe_comparable"]["requires_public_collection_progress"]
        is not True
        or policies["quadratic_material_value"]["requires_quadratic_workflow_progress"]
        is not True
    ):
        raise ValueError("frozen preference prerequisites differ")
    comparisons = {
        name: _comparison(rows, protocol, name, numerator, denominator)
        for name, (numerator, denominator) in _PAIRS.items()
    }
    hashes = {
        name: comparison["bootstrap"].get("parent_draws_sha256")
        for name, comparison in comparisons.items()
    }
    available_hashes = {value for value in hashes.values() if value is not None}
    if len(available_hashes) > 1:
        raise ValueError("pairwise bootstrap parent draws differ")
    pair_pass = {
        name: comparison["residual_criteria_pass"]
        for name, comparison in comparisons.items()
    }
    public_progress = pair_pass["public_collection"]
    quadratic_progress = pair_pass["quadratic_workflow"]
    comparable = public_progress and pair_pass["simpler_recipe_comparable"]
    material = quadratic_progress and pair_pass["quadratic_material_value"]
    if comparable and material:
        raise ValueError("frozen reciprocal preference criteria cannot both pass")
    preference = (
        "public_recipe"
        if comparable
        else "quadratic_for_separate_qualification"
        if material
        else "unresolved_tradeoff"
    )
    return {
        "protocol": protocol["id"],
        "comparisons": comparisons,
        "public_collection_progress": public_progress,
        "quadratic_workflow_progress": quadratic_progress,
        "simpler_recipe_comparable": comparable,
        "quadratic_material_value": material,
        "preference": preference,
        "unresolved_tradeoff": not (comparable or material),
        "preference_prerequisites": {
            "simpler_recipe_comparable": {
                "public_collection_progress": public_progress,
                "pairwise_residual_criteria_pass": pair_pass[
                    "simpler_recipe_comparable"
                ],
            },
            "quadratic_material_value": {
                "quadratic_workflow_progress": quadratic_progress,
                "pairwise_residual_criteria_pass": pair_pass[
                    "quadratic_material_value"
                ],
            },
        },
        "bootstrap_parent_draws_sha256": next(iter(available_hashes), None),
        "bootstrap_comparison_hashes": hashes,
        "available_bootstrap_parent_draws_match": bool(available_hashes),
        "shared_bootstrap_parent_draws_verified": all(hashes.values())
        and len(available_hashes) == 1,
        "public_promotion": False,
        "qualification": protocol["decision"]["public_promotion"],
    }
