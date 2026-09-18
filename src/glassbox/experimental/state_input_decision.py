"""Frozen residual decisions for the state-input interaction experiment.

Raw response errors include weak probes. Parent resampling preserves all query,
horizon, signal and arm dependence inside each recording. No fit or simulator
imports belong in this reducer.
"""

import hashlib
import itertools
import json
from collections import defaultdict

import numpy as np

ARMS = ("baseline", "candidate")
KINDS = ("factual", "response")
IDENTITY = (
    "simulator",
    "scope",
    "cell",
    "parent",
    "query",
    "origin",
    "kind",
    "horizon_s",
    "group",
    "statistic",
)


def _number(value):
    return value is not None and np.isfinite(value) and value >= 0


def _ratio(candidate, baseline, floor):
    if not _number(candidate) or not _number(baseline):
        return None
    return float(max(candidate, floor) / max(baseline, floor))


def _geomean(values, weights=None):
    values = list(values)
    if not values or any(v is None or not np.isfinite(v) or v <= 0 for v in values):
        return None
    return float(np.exp(np.average(np.log(values), weights=weights)))


def _within(value, limit):
    return value is not None and np.isfinite(value) and value <= limit


def _index(records, keys):
    result = {}
    for row in records:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError(f"duplicate reduction identity: {key}")
        result[key] = row
    return result


def validate_cohorts(rows, protocol):
    """Reject unmatched identities, masks or missing protocol test parents."""
    output = {}
    for sim in protocol["cells"]:
        values = rows[sim]
        indexed = {
            arm: _index((r for r in values if r["arm"] == arm), IDENTITY)
            for arm in ARMS
        }
        if (
            not indexed["baseline"]
            or indexed["baseline"].keys() != indexed["candidate"].keys()
        ):
            raise ValueError(f"{sim}: unmatched planned metric queries")
        parents = {r["parent"] for r in indexed["baseline"].values()}
        expected = {
            r["id"]
            for r in protocol["recordings"]
            if r["simulator"] == sim and r["role"] == "test"
        }
        if parents != expected:
            raise ValueError(f"{sim}: planned test-parent roster changed")
        for key, base in indexed["baseline"].items():
            other = indexed["candidate"][key]
            for field in (
                "truth_eligible",
                "pair_nonweak",
                "components",
                "horizon_steps",
            ):
                if base.get(field) != other.get(field):
                    raise ValueError(f"{sim}: truth cohort differs: {key}/{field}")
        output[sim] = {
            "matching_queries_and_truth": True,
            "metric_slots_per_arm": len(indexed["baseline"]),
            "test_parents": len(parents),
            "arms": {
                arm: {
                    "eligible_rows": sum(
                        bool(r["truth_eligible"]) for r in items.values()
                    ),
                    "failed_rows": sum(
                        bool(r["truth_eligible"])
                        and (not r["prediction_finite"] or not _number(r["mse"]))
                        for r in items.values()
                    ),
                }
                for arm, items in indexed.items()
            },
        }
    return output


def _tails(summaries):
    grouped = defaultdict(list)
    keys = ("simulator", "scope", "kind", "arm", "horizon_s", "group", "statistic")
    for summary in summaries.values():
        for parent in summary["parents"]:
            if parent["arm"] in ARMS:
                grouped[tuple(parent[k] for k in keys)].append(parent)
    result = []
    for key, parents in sorted(grouped.items()):
        values = [
            r["available_truth_rmse"]
            for r in parents
            if _number(r["available_truth_rmse"])
        ]
        result.append(
            {
                **dict(zip(keys, key, strict=True)),
                "planned_parents": len(parents),
                "available_parents": len(values),
                "missing_parents": sum(r["truth_eligible"] == 0 for r in parents),
                "failed_parents": sum(r["failed"] > 0 for r in parents),
                "p50": None
                if not values
                else float(np.quantile(values, 0.5, method="linear")),
                "p95": None
                if not values
                else float(np.quantile(values, 0.95, method="linear")),
                "max": None if not values else float(np.max(values)),
            }
        )
    return result


def _nanmean(values, axis):
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def _bootstrap(summaries, protocol, metric_keys):
    """1000 fixed paired parent draws; empty draws stay explicitly unavailable."""
    policy = protocol["decision"]["aggregation"]
    count = 1000
    rng = np.random.default_rng(20260918)
    parent_index = _index(
        (
            r
            for summary in summaries.values()
            for r in summary["parents"]
            if r["arm"] in ARMS and r["statistic"] == "endpoint"
        ),
        ("simulator", "scope", "cell", "parent", "arm", "kind", "horizon_s", "group"),
    )
    dimensions = list(itertools.product(KINDS, policy["horizons_s"], policy["groups"]))
    draw_hash = hashlib.sha256()
    ratios_by_key = {}
    for sim in sorted(policy["simulator_weights"]):
        for scope in sorted(policy["scope_weights"]):
            cells = sorted(
                c["id"] for c in protocol["cells"][sim] if c["group"] == scope
            )
            sampled_cell_mse = []
            for cell in cells:
                parents = sorted(
                    r["id"]
                    for r in protocol["recordings"]
                    if r["simulator"] == sim
                    and r["cell"] == cell
                    and r["role"] == "test"
                )
                values = np.full((len(parents), len(ARMS), len(dimensions)), np.nan)
                for pi, parent in enumerate(parents):
                    for ai, arm in enumerate(ARMS):
                        for di, (kind, horizon, group) in enumerate(dimensions):
                            record = parent_index[
                                sim, scope, cell, parent, arm, kind, horizon, group
                            ]
                            value = record["available_truth_mse"]
                            if _number(value):
                                values[pi, ai, di] = value
                chosen = rng.integers(
                    len(parents), size=(count, len(parents)), dtype=np.int64
                )
                draw_hash.update(
                    json.dumps(
                        [sim, scope, cell, parents], separators=(",", ":")
                    ).encode()
                )
                draw_hash.update(chosen.astype("<i8").tobytes())
                sampled_cell_mse.append(_nanmean(values[chosen], axis=1))
            rmse = np.sqrt(_nanmean(np.stack(sampled_cell_mse), axis=0))
            for di, (kind, horizon, group) in enumerate(dimensions):
                floor = policy["floors"][group]
                ratios_by_key[sim, scope, kind, horizon, group] = np.maximum(
                    rmse[:, 1, di], floor
                ) / np.maximum(rmse[:, 0, di], floor)
    result = {}
    for kind in KINDS:
        keys = [key for key in metric_keys if key[2] == kind]
        weights = np.asarray(
            [
                policy["simulator_weights"][k[0]] * policy["scope_weights"][k[1]]
                for k in keys
            ]
        )
        matrix = np.stack([ratios_by_key[key] for key in keys], axis=1)
        draws = np.exp(np.sum(np.log(matrix) * weights, axis=1) / weights.sum())
        finite = np.isfinite(draws)
        result[kind] = {
            "draws": [float(v) if np.isfinite(v) else None for v in draws],
            "available_draws": int(finite.sum()),
            "unavailable_draws": int((~finite).sum()),
            "percentile_95_interval": None
            if not finite.any()
            else np.quantile(draws[finite], [0.025, 0.975], method="linear").tolist(),
        }
    return {
        "seed": 20260918,
        "requested_draws": count,
        "parent_draws_sha256": draw_hash.hexdigest(),
        "conditional_on_available_truth": True,
        "intervals_condition_on_available_draws": True,
        "aggregates": result,
    }


def reduce(summaries, protocol, *, rows):
    """Derive physical-unit comparisons, tails and the frozen residual decision.

    Integrity/tamper/capability/contract checks remain separate obligations;
    ``residual_criteria_pass`` alone never promotes a public recipe.
    """
    policy = protocol["decision"]["aggregation"]
    limits = protocol["decision"]["accept_research_candidate"]
    cohorts = validate_cohorts(rows, protocol)
    keys = list(
        itertools.product(
            sorted(policy["simulator_weights"]),
            sorted(policy["scope_weights"]),
            KINDS,
            policy["horizons_s"],
            policy["groups"],
        )
    )
    indexed = _index(
        (
            r
            for summary in summaries.values()
            for r in summary["summaries"]
            if r["arm"] in ARMS and r["statistic"] == "endpoint"
        ),
        ("simulator", "scope", "kind", "horizon_s", "group", "arm"),
    )
    comparisons = []
    for key in keys:
        base, candidate = (indexed[(*key, arm)] for arm in ARMS)
        for field in (
            "planned",
            "truth_eligible",
            "planned_parents",
            "eligible_parents",
            "planned_cells",
            "eligible_cells",
            "planned_origins",
            "eligible_origins",
        ):
            if base[field] != candidate[field]:
                raise ValueError(f"aggregate cohorts differ: {key}/{field}")
        b, c = base["available_truth_rmse"], candidate["available_truth_rmse"]
        comparisons.append(
            {
                **dict(
                    zip(
                        ("simulator", "scope", "kind", "horizon_s", "group"),
                        key,
                        strict=True,
                    )
                ),
                "baseline_rmse": b,
                "candidate_rmse": c,
                "absolute_change": None
                if not _number(b) or not _number(c)
                else float(c - b),
                "ratio": _ratio(c, b, policy["floors"][key[-1]]),
                "planned": base["planned"],
                "truth_eligible": base["truth_eligible"],
                "conditional_on_incomplete_truth": base["truth_eligible"]
                != base["planned"],
            }
        )
    overall = {}
    scope_ratios = []
    for kind in KINDS:
        selected = [r for r in comparisons if r["kind"] == kind]
        overall[kind] = _geomean(
            (r["ratio"] for r in selected),
            [
                policy["simulator_weights"][r["simulator"]]
                * policy["scope_weights"][r["scope"]]
                for r in selected
            ],
        )
        for sim in sorted(policy["simulator_weights"]):
            for scope in sorted(policy["scope_weights"]):
                scope_ratios.append(
                    {
                        "simulator": sim,
                        "scope": scope,
                        "kind": kind,
                        "ratio": _geomean(
                            r["ratio"]
                            for r in selected
                            if r["simulator"] == sim and r["scope"] == scope
                        ),
                    }
                )
    tails = _tails(summaries)
    tail_index = _index(
        tails, ("simulator", "scope", "kind", "arm", "horizon_s", "group", "statistic")
    )
    tail_checks = []
    for sim, kind, group in itertools.product(
        sorted(policy["simulator_weights"]),
        limits["tail_guard_kinds"],
        limits["tail_guard_groups"],
    ):
        base, candidate = (
            tail_index[sim, "primary", kind, arm, 0.25, group, "endpoint"]
            for arm in ARMS
        )
        ratio = _ratio(candidate["p95"], base["p95"], policy["floors"][group])
        tail_checks.append(
            {
                "simulator": sim,
                "kind": kind,
                "group": group,
                "baseline_p95": base["p95"],
                "candidate_p95": candidate["p95"],
                "ratio": ratio,
                "pass": _within(
                    ratio,
                    limits["each_simulator_250ms_primary_parent_rmse_p95_ratio_max"],
                ),
            }
        )
    checks = {
        "matching_planned_queries_and_truth": True,
        "finite_eligible_predictions": all(
            arm["eligible_rows"] > 0 and arm["failed_rows"] == 0
            for sim in cohorts.values()
            for arm in sim["arms"].values()
        ),
        "response_gain": _within(
            overall["response"], limits["response_weighted_geometric_mean_ratio_max"]
        ),
        "factual_regression": _within(
            overall["factual"], limits["factual_weighted_geometric_mean_ratio_max"]
        ),
        "primary_simulator_regressions": all(
            _within(
                r["ratio"],
                limits[
                    "each_simulator_primary_factual_and_response_geometric_mean_ratio_max"
                ],
            )
            for r in scope_ratios
            if r["scope"] == "primary"
        ),
        "scope_regressions": all(
            _within(
                r["ratio"],
                limits[
                    "each_simulator_scope_factual_and_response_geometric_mean_ratio_max"
                ],
            )
            for r in scope_ratios
        ),
        "primary_parent_tail_regressions": all(r["pass"] for r in tail_checks),
    }
    return {
        "protocol": protocol["id"],
        "cohorts": cohorts,
        "physical_comparisons": comparisons,
        "weighted_geometric_mean_ratios": overall,
        "scope_ratios": scope_ratios,
        "parent_error_tails": tails,
        "tail_checks": tail_checks,
        "checks": checks,
        "residual_criteria_pass": all(checks.values()),
        "public_promotion": False,
        "remaining_promotion_requirements": "Integrity, replay, tamper and focused tests; public capability and contract qualification remain separate.",
        "bootstrap": (
            _bootstrap(summaries, protocol, keys)
            if checks["finite_eligible_predictions"]
            else {"status": "unavailable_due_to_prediction_failure"}
        ),
    }
