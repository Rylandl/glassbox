"""Frozen public-baseline and independent-excitation collection comparisons."""

from .state_input_decision import _within
from .state_input_decision import reduce as paired_reduce
from .two_simulator_metrics import aggregate


def _comparison(rows, protocol, reference):
    # A fresh mapping preserves the historical reducer and original arm labels.
    paired = {
        sim: [
            dict(row, arm="baseline" if row["arm"] == reference else "candidate")
            for row in values
            if row["arm"] in (reference, "candidate")
        ]
        for sim, values in rows.items()
    }
    return paired_reduce(
        {sim: aggregate(values) for sim, values in paired.items()},
        protocol,
        rows=paired,
    )


def reduce(summaries, protocol, *, rows):
    """Require broad gains against public and the predeclared collection repair.

    Pairwise reductions share fixed parent draws. Both comparisons retain every
    planned row, including weak responses and missing truth. The old public
    limits apply only to the public comparison. The original-data quadratic
    comparison has its own three frozen limits, without imposing the public
    limits a second time.
    """
    if set(summaries) != set(rows) or any(
        aggregate(values) != summaries[sim] for sim, values in rows.items()
    ):
        raise ValueError("summary differs from full metric rows")
    public = _comparison(rows, protocol, "baseline")
    original = _comparison(rows, protocol, "original")
    limits = protocol["decision"]["mechanism_comparison"]
    if limits["reference_arm"] != "original":
        raise ValueError("frozen mechanism reference must be original")
    targeted = [
        row
        for row in original["physical_comparisons"]
        if row["simulator"] == "crazyflow"
        and row["scope"] == "primary"
        and row["kind"] == "response"
        and row["horizon_s"] == 0.25
        and row["group"] == "body_rate_rad_s"
    ]
    if len(targeted) != 1:
        raise ValueError("missing or duplicate angular-response comparison")
    public_hash = public["bootstrap"].get("parent_draws_sha256")
    mechanism_hash = original["bootstrap"].get("parent_draws_sha256")
    if public_hash and mechanism_hash and public_hash != mechanism_hash:
        raise ValueError("pairwise bootstrap parent draws differ")
    checks = {
        **public["checks"],
        "original_finite_eligible_predictions": original["checks"][
            "finite_eligible_predictions"
        ],
        "mechanism_response_regression": _within(
            original["weighted_geometric_mean_ratios"]["response"],
            limits["response_weighted_geometric_mean_ratio_max"],
        ),
        "mechanism_factual_regression": _within(
            original["weighted_geometric_mean_ratios"]["factual"],
            limits["factual_weighted_geometric_mean_ratio_max"],
        ),
        "mechanism_angular_response_gain": _within(
            targeted[0]["ratio"],
            limits["crazyflow_primary_250ms_body_rate_response_rmse_ratio_max"],
        ),
    }
    descriptive = (
        "cohorts",
        "physical_comparisons",
        "weighted_geometric_mean_ratios",
        "scope_ratios",
        "parent_error_tails",
        "bootstrap",
    )
    return {
        **public,
        "reference_arm": "baseline",
        "checks": checks,
        "residual_criteria_pass": all(checks.values()),
        "mechanism_comparison": {
            "reference_arm": "original",
            "baseline_fields_mean": "original-data quadratic comparator, not public baseline",
            **{key: original[key] for key in descriptive},
            "targeted_angular_response": targeted[0],
        },
        "shared_bootstrap_parent_draws_verified": bool(
            public_hash and mechanism_hash and public_hash == mechanism_hash
        ),
    }
