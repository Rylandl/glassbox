"""Frozen optimizer comparison on the original 128 saved oracle problems.

The causal replay and evidence schema are shared with the prior budget study.
Only the optimizer changes; neither a model nor a flight policy is promoted.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import scipy

from glassbox.control.solver import BoundedShootingSolver

from . import solver_budget as base
from .qualification import _write_snapshot, arrays, digest, same, write_json
from .quasi_newton import QuasiNewtonSolver
from .task_qualification import check_environment as task_environment

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-quasi-newton-v1.json"
PLAN_SHA256 = "232968d8504cb3d6a79502cec6835883a02864218195cc1fed39997d06149107"
COUNTERS = {
    "accepted_iterations": 64,
    "new_objective_evaluations": 1024,
    "seed_objective_evaluations": 2,
    "cached_seed_requests": None,
    "audit_objective_evaluations": 1,
}
FLAGS = {
    "backend_success",
    "nonfinite_evaluation",
    "backend_failure",
    "returned_seed",
}
SCORES = {"independent_projected_gradient_inf_norm", "independent_objective"}
WORK_ARRAYS = {"returned_canonical_blocks", "independent_gradient"}
STOPS = {
    "inherited_failure",
    "projected_gradient",
    "relative_improvement",
    "iteration_limit",
    "evaluation_limit",
    "nonfinite_evaluation",
    "backend_failure",
}
WORK_FIELDS = (
    set(COUNTERS)
    | FLAGS
    | SCORES
    | WORK_ARRAYS
    | {"seed", "origin", "backend_status", "backend_message", "stop_cause"}
)
# Normalize precisely the two allowed hook edits and the iteration-bound edit
# back to the byte-pinned parent. Every other edit remains detectable.
HOOK_EDITS = (
    ("def probe(plan, inputs):", "def probe(plan, inputs, *, _solve_pair=None):"),
    (
        "                pair = paired_solve(\n",
        "                pair = (paired_solve if _solve_pair is None else _solve_pair)(\n",
    ),
    (
        '        if np.any(trial["iterations"] < 0) or np.any(trial["iterations"] > [4, 64]):\n',
        '        limits = [plan["comparison"][f"{name}_maximum_iterations"] for name in ARMS]\n'
        '        if np.any(trial["iterations"] < 0) or np.any(trial["iterations"] > limits):\n',
    ),
)


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("quasi-Newton plan differs from frozen source")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"quasi-Newton inherited source changed: {name}")
    allowed = plan["allowed_harness_edit"]
    source = (ROOT / allowed["path"]).read_text()
    for before, after in HOOK_EDITS:
        if source.count(after) != 1:
            raise ValueError("solver budget replay hook differs")
        source = source.replace(after, before)
    if digest(source.encode()) != allowed["baseline_sha256"]:
        raise ValueError("solver budget changed beyond its allowed replay hook")
    return plan, raw


def check_environment(plan):
    actual = task_environment(plan) | {"scipy": scipy.__version__}
    same(actual, plan["environment"], label="pinned numerical environment")
    return actual


def task_inputs(inputs):
    return {
        key.removeprefix("inputs/"): value
        for key, value in inputs.items()
        if key.startswith("inputs/")
    }


def compare_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("quasi-Newton array fields differ")
    for key, value in expected.items():
        if value.dtype.kind in "biuU" or key == "commands":
            np.testing.assert_array_equal(actual[key], value)
        else:
            np.testing.assert_allclose(
                actual[key],
                value,
                **(base.STATE_TOLERANCE if key == "states" else base.SCORE_TOLERANCE),
            )


def historical_parity(result, trial, row):
    fresh = base.serialize_pair(dict(baseline=result, candidate=result))
    compare_arrays(
        {key: value[0] for key, value in fresh.items()},
        {key: trial[key][row, 1] for key in fresh},
    )


def paired_solve(
    model,
    policy,
    state,
    reference,
    previous_command,
    *,
    warm_start=None,
    baseline_check=None,
    historical_check=None,
    work_sink=None,
):
    args = (state, reference, previous_command)
    original = BoundedShootingSolver(model, policy).solve(*args, warm_start=warm_start)
    if baseline_check is not None:
        baseline_check(original)
    expanded = base.candidate_policy(policy)
    baseline = BoundedShootingSolver(model, expanded).solve(
        *args, warm_start=warm_start
    )
    if historical_check is not None:
        historical_check(baseline)
    solver = QuasiNewtonSolver(model, expanded)
    candidate = solver.solve(*args, warm_start=warm_start)
    for result in (baseline, candidate):
        np.testing.assert_allclose(
            original.diagnostics.initial_objective,
            result.diagnostics.initial_objective,
            **base.SCORE_TOLERANCE,
        )
        same(original.diagnostics.warm_start_used, result.diagnostics.warm_start_used)
    if work_sink is not None:
        work_sink(dict(solver.last_work))
    return dict(baseline=baseline, candidate=candidate)


def validate_work(plan, trials, work):
    base.validate_arrays(plan, trials)
    expected = [
        (s, o) for s in plan["selection"]["seeds"] for o in plan["selection"]["origins"]
    ]
    if not isinstance(work, list) or len(work) != len(expected):
        raise ValueError("quasi-Newton work roster differs")
    for index, (row, (seed, origin)) in enumerate(zip(work, expected, strict=True)):
        if not isinstance(row, dict) or set(row) != WORK_FIELDS:
            raise ValueError("quasi-Newton work fields differ")
        same([row["seed"], row["origin"]], [seed, origin])
        for key, limit in COUNTERS.items():
            value = row[key]
            if (
                type(value) is not int
                or value < 0
                or (limit is not None and value > limit)
            ):
                raise ValueError(f"quasi-Newton invalid work counter: {key}")
        if any(type(row[key]) is not bool for key in FLAGS):
            raise ValueError("quasi-Newton work flags must be booleans")
        if row["backend_status"] is not None and type(row["backend_status"]) is not int:
            raise ValueError("quasi-Newton invalid backend status")
        if type(row["backend_message"]) is not str or row["stop_cause"] not in STOPS:
            raise ValueError("quasi-Newton invalid stop diagnostic")
        trial, i = (
            trials[index // len(plan["selection"]["origins"])],
            index % len(plan["selection"]["origins"]),
        )
        for key, array_key in (
            ("independent_objective", "final_objectives"),
            ("independent_projected_gradient_inf_norm", "projected_gradient_inf_norm"),
        ):
            value = row[key]
            if value is not None and (
                type(value) not in (int, float) or not np.isfinite(value)
            ):
                raise ValueError(f"quasi-Newton invalid independent score: {key}")
            if not trial["used_fallback"][i, 1]:
                if value is None or row["audit_objective_evaluations"] != 1:
                    raise ValueError(
                        "quasi-Newton finite output lacks independent audit"
                    )
                same(value, float(trial[array_key][i, 1]))
        if not trial["used_fallback"][i, 1]:
            same(row["accepted_iterations"], int(trial["iterations"][i, 1]))
        saved_arrays = {}
        for key in WORK_ARRAYS:
            value = row[key]
            if value is None:
                if not trial["used_fallback"][i, 1]:
                    raise ValueError("quasi-Newton finite output lacks audited arrays")
                continue
            numeric = np.asarray(value)
            if (
                not isinstance(value, list)
                or numeric.shape != (base.HORIZON_STEPS, 3)
                or numeric.dtype.kind not in "fi"
                or not np.isfinite(numeric).all()
            ):
                raise ValueError(f"quasi-Newton invalid audited array: {key}")
            if key == "returned_canonical_blocks" and np.any(np.abs(numeric) > 1):
                raise ValueError("quasi-Newton canonical blocks violate bounds")
            array = numeric.astype(np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"quasi-Newton invalid float32 audited array: {key}")
            saved_arrays[key] = array
        if saved_arrays:
            if set(saved_arrays) != WORK_ARRAYS:
                raise ValueError("quasi-Newton incomplete audited arrays")
            blocks = saved_arrays["returned_canonical_blocks"]
            gradient = saved_arrays["independent_gradient"]
            residual = float(
                np.max(
                    np.abs(
                        blocks
                        - np.clip(blocks - gradient, np.float32(-1), np.float32(1))
                    )
                )
            )
            same(row["independent_projected_gradient_inf_norm"], residual)


def report_from_inputs(plan, trials, inputs, work):
    validate_work(plan, trials, work)
    result = base.report_from_inputs(plan, trials, task_inputs(inputs))
    result["plan_sha256"] = PLAN_SHA256
    limits = plan["comparison"]["qualification"]
    for row, effort in zip(result["origins"], work, strict=True):
        a, b = row["baseline"], row["candidate"]
        row["candidate_failure"] = bool(
            not b["finite_bounded_nonfallback"]
            or effort["backend_failure"]
            or effort["nonfinite_evaluation"]
        )
        row["residual_ratio"] = (
            b["projected_gradient_inf_norm"]
            / max(a["projected_gradient_inf_norm"], 0.002)
            if a["finite_bounded_nonfallback"] and b["finite_bounded_nonfallback"]
            else None
        )
        reduction = row["fractional_objective_reduction"]
        row["fractional_objective_regression"] = (
            max(0.0, -reduction) if reduction is not None else None
        )
    rows = result["origins"]
    ratios = [r["residual_ratio"] for r in rows]
    regressions = [r["fractional_objective_regression"] for r in rows]
    median_ratio = float(np.median(ratios)) if None not in ratios else None
    maximum_regression = max(regressions) if None not in regressions else None
    converged = result["pooled"]["candidate"]["below_gradient_threshold"]
    failures = sum(r["candidate_failure"] for r in rows)
    bound = max(r["candidate"]["command_bound_violation"] for r in rows)
    criteria = dict(
        enough_converged_origins=converged >= limits["minimum_converged_origins"],
        median_residual_ratio=(
            median_ratio is not None
            and median_ratio <= limits["maximum_median_residual_ratio"]
        ),
        objective_regression=(
            maximum_regression is not None
            and maximum_regression <= limits["maximum_fractional_objective_regression"]
        ),
        candidate_failures=failures <= limits["maximum_candidate_failures"],
        command_bounds=bound <= limits["maximum_command_bound_violation"],
    )

    def work_summary(selected):
        return dict(
            origins=len(selected),
            stop_causes={
                s: sum(r["stop_cause"] == s for r in selected)
                for s in sorted({r["stop_cause"] for r in selected})
            },
            backend_success_count=sum(r["backend_success"] for r in selected),
            backend_failure_count=sum(r["backend_failure"] for r in selected),
            nonfinite_evaluation_count=sum(r["nonfinite_evaluation"] for r in selected),
            returned_seed_count=sum(r["returned_seed"] for r in selected),
            **{
                key: dict(
                    total=sum(r[key] for r in selected),
                    maximum=max(r[key] for r in selected),
                )
                for key in COUNTERS
            },
        )

    result["backend_work"] = dict(
        meaning=plan["comparison"]["work_bound"],
        pooled=work_summary(work),
        per_seed=[
            dict(seed=s, **work_summary([r for r in work if r["seed"] == s]))
            for s in plan["selection"]["seeds"]
        ],
    )
    result["qualification"] = dict(
        meaning=limits["meaning"],
        criteria=criteria,
        eligible_for_future_tracking_experiment=all(criteria.values()),
        converged_origins=converged,
        median_residual_ratio=median_ratio,
        maximum_fractional_objective_regression=maximum_regression,
        candidate_failures=failures,
        maximum_command_bound_violation=bound,
        maintained_solver_promoted=False,
    )
    return result


def probe(plan, inputs):
    historical = [
        arrays(inputs[f"trial-{i}/paired.npz"])
        for i in range(len(plan["selection"]["seeds"]))
    ]
    base.validate_arrays(json.loads(inputs["manifest.json"]), historical)
    work = []
    count = len(plan["selection"]["origins"])

    def solve(*args, **kwargs):
        repetition, row = divmod(len(work), count)
        seed, origin = (
            plan["selection"]["seeds"][repetition],
            plan["selection"]["origins"][row],
        )
        return paired_solve(
            *args,
            **kwargs,
            historical_check=partial(
                historical_parity, trial=historical[repetition], row=row
            ),
            work_sink=lambda values: work.append(
                dict(seed=seed, origin=origin, **values)
            ),
        )

    trials, _ = base.probe(plan, task_inputs(inputs), _solve_pair=solve)
    return trials, work, report_from_inputs(plan, trials, inputs, work)


def artifact_names(plan):
    return base.artifact_names(plan) | {"work.json"}


def run(artifacts, output, *, _experiment=None):
    experiment = _experiment or sys.modules[__name__]
    plan, raw = experiment.frozen_plan()
    environment = experiment.check_environment(plan)
    inputs = base.input_snapshot(plan, artifacts)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    trials, work, result = experiment.probe(plan, inputs)
    write_json(output / "run.json", dict(no_fit=True, new_policy_trials=False))
    write_json(output / "environment.json", environment)
    write_json(output / "report.json", result)
    write_json(output / "work.json", work)
    for i, trial in enumerate(trials):
        directory = output / f"trial-{i}"
        directory.mkdir()
        np.savez_compressed(directory / "paired.npz", **trial)
    write_json(
        output / "files.json",
        {
            name: digest((output / name).read_bytes())
            for name in sorted(artifact_names(plan))
        },
    )
    return result


def verify(directory, *, _experiment=None):
    experiment = _experiment or sys.modules[__name__]
    plan, raw = experiment.frozen_plan()
    environment = experiment.check_environment(plan)
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved quasi-Newton plan differs")
    inputs = base.input_snapshot(plan, directory / "inputs")
    names = artifact_names(plan)
    expected = names | {"manifest.json", "files.json"} | {"inputs/" + n for n in inputs}
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    } != expected:
        raise ValueError("quasi-Newton complete file roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    if set(hashes) != names:
        raise ValueError("quasi-Newton artifact inventory differs")
    saved = {name: (directory / name).read_bytes() for name in names}
    for name, data in saved.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"altered quasi-Newton artifact: {name}")
    same(json.loads(saved["environment.json"]), environment)
    same(json.loads(saved["run.json"]), dict(no_fit=True, new_policy_trials=False))
    trials = [
        arrays(saved[f"trial-{i}/paired.npz"])
        for i in range(len(plan["selection"]["seeds"]))
    ]
    work, summary = json.loads(saved["work.json"]), json.loads(saved["report.json"])
    same(summary, experiment.report_from_inputs(plan, trials, inputs, work))
    fresh, fresh_work, result = experiment.probe(plan, inputs)
    for actual, expected in zip(trials, fresh, strict=True):
        compare_arrays(actual, expected)
    same(work, fresh_work)
    for actual, expected in zip(work, fresh_work, strict=True):
        for key in WORK_ARRAYS:
            np.testing.assert_array_equal(actual[key], expected[key])
    same(summary, result)
    return dict(
        verified=True, no_fit=True, verified_pairs=len(result["origins"]), report=result
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--artifacts", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = (
        verify(args.directory)
        if args.command == "verify"
        else run(args.artifacts, args.output)
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
