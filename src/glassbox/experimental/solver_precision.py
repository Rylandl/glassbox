"""Frozen precision comparison with common float64 audits of both plans."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np

from glassbox.control.solver import BoundedShootingSolver

from . import first_order_qualification as previous
from . import quasi_newton_qualification as shared
from . import solver_budget as base
from .precision_solver import PrecisionFactory, SeedCaptureSolver
from .qualification import _context, _write_snapshot, arrays, digest, write_json
from .solver_termination import exact, exact_arrays, task_inputs

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-precision-v1.json"
PLAN_SHA256 = "3ebbdd854b56592d2fe33686f80253c6206f2646654201e169d3d776476ee088"
check_environment = shared.check_environment


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("precision plan differs")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"precision inherited source differs: {name}")
    previous.frozen_plan()
    return plan, raw


def probe(plan, inputs):
    if jax.config.x64_enabled:
        raise ValueError("baseline replay must start in float32")
    historical = [arrays(inputs[f"trial-{i}/paired.npz"]) for i in range(4)]
    historical_work = json.loads(inputs["work.json"])
    shared.validate_work(
        json.loads(inputs["manifest.json"]), historical, historical_work
    )
    task = task_inputs(inputs)
    context = _context(shared.task_inputs(task))
    factory = PrecisionFactory(context.model)
    records = []
    count = len(plan["selection"]["origins"])

    def solve(model, policy, state, reference, command, *, warm_start, baseline_check):
        repetition, row = divmod(len(records), count)
        seed = plan["selection"]["seeds"][repetition]
        origin = plan["selection"]["origins"][row]
        arguments = (state, reference, command)
        baseline_check(
            BoundedShootingSolver(model, policy).solve(
                *arguments, warm_start=warm_start
            )
        )
        expanded = base.candidate_policy(policy)
        baseline = SeedCaptureSolver(model, expanded)
        result = baseline.solve(*arguments, warm_start=warm_start)
        shared.historical_parity(result, historical[repetition], row)
        previous.historical_work_parity(
            dict(seed=seed, origin=origin, **baseline.last_work),
            historical_work[len(records)],
        )
        candidate, work, common, precision = factory.solve(
            model,
            expanded,
            baseline.last_capture,
            baseline.last_work["returned_canonical_blocks"],
        )
        if jax.config.x64_enabled:
            raise ValueError("precision context leaked into causal replay")
        records.append(
            dict(
                seed=seed,
                origin=origin,
                capture=baseline.last_capture.record(),
                baseline_work=copy.deepcopy(baseline.last_work),
                candidate_work=work,
                common=common,
                precision=precision,
            )
        )
        return dict(baseline=result, candidate=candidate)

    trials, _ = base.probe(plan, task, _solve_pair=solve)
    return trials, records, report_from_records(plan, trials, records)


def distribution(values):
    values = [float(v) for v in values if v is not None]
    return dict(
        count=len(values),
        minimum=min(values) if values else None,
        median=float(np.median(values)) if values else None,
        maximum=max(values) if values else None,
    )


def validate_audit(audit):
    if audit is None:
        return
    if set(audit) != {
        "blocks",
        "value",
        "gradient",
        "residual",
        "commands",
        "states",
        "bound_violation",
    }:
        raise ValueError("common audit fields differ")
    for name, shape in {
        "blocks": (5, 3),
        "gradient": (5, 3),
        "commands": (5, 3),
        "states": (6, 13),
    }.items():
        array = np.asarray(audit[name])
        if (
            array.shape != shape
            or array.dtype != np.float64
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"common64 array differs: {name}")
    for name in ("value", "residual", "bound_violation"):
        if type(audit[name]) is not float or not np.isfinite(audit[name]):
            raise ValueError(f"common64 scalar differs: {name}")
    blocks, gradient = np.asarray(audit["blocks"]), np.asarray(audit["gradient"])
    if np.any(np.abs(blocks) > 1) or audit["bound_violation"] < 0:
        raise ValueError("common audit command bounds differ")
    exact(
        audit["residual"],
        float(np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1)))),
    )


def validate_candidate_work(plan, trials, work):
    """Validate native64 evidence without the historical validator's32 cast."""
    expected = [
        [s, o] for s in plan["selection"]["seeds"] for o in plan["selection"]["origins"]
    ]
    if not isinstance(work, list) or len(work) != len(expected):
        raise ValueError("precision work roster differs")
    count = len(plan["selection"]["origins"])
    for index, (row, identity) in enumerate(zip(work, expected, strict=True)):
        if not isinstance(row, dict) or set(row) != shared.WORK_FIELDS:
            raise ValueError("precision work fields differ")
        exact([row["seed"], row["origin"]], identity)
        for key, limit in (shared.COUNTERS | {"seed_objective_evaluations": 1}).items():
            value = row[key]
            if (
                type(value) is not int
                or value < 0
                or (limit is not None and value > limit)
            ):
                raise ValueError(f"precision invalid work counter: {key}")
        if any(type(row[key]) is not bool for key in shared.FLAGS):
            raise ValueError("precision work flags must be booleans")
        if row["backend_status"] is not None and type(row["backend_status"]) is not int:
            raise ValueError("precision invalid backend status")
        if (
            type(row["backend_message"]) is not str
            or row["stop_cause"] not in shared.STOPS
        ):
            raise ValueError("precision invalid stop diagnostic")
        repetition, i = divmod(index, count)
        trial = trials[repetition]
        fallback = bool(trial["used_fallback"][i, 1])
        for key, array_key in (
            ("independent_objective", "final_objectives"),
            ("independent_projected_gradient_inf_norm", "projected_gradient_inf_norm"),
        ):
            value = row[key]
            if value is not None and (
                type(value) is not float or not np.isfinite(value)
            ):
                raise ValueError(f"precision invalid independent score: {key}")
            if not fallback:
                if value is None or row["audit_objective_evaluations"] != 1:
                    raise ValueError("precision finite output lacks independent audit")
                exact(value, float(trial[array_key][i, 1]))
        if not fallback:
            exact(row["accepted_iterations"], int(trial["iterations"][i, 1]))
        saved = {}
        for key in shared.WORK_ARRAYS:
            value = row[key]
            if value is None:
                if not fallback:
                    raise ValueError("precision finite output lacks audited arrays")
                continue
            array = np.asarray(value)
            if (
                not isinstance(value, list)
                or array.shape != (base.HORIZON_STEPS, 3)
                or array.dtype != np.float64
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"precision invalid float64 audited array: {key}")
            if key == "returned_canonical_blocks" and np.any(np.abs(array) > 1):
                raise ValueError("precision canonical blocks violate bounds")
            saved[key] = array
        if saved:
            if set(saved) != shared.WORK_ARRAYS:
                raise ValueError("precision incomplete audited arrays")
            blocks, gradient = (
                saved["returned_canonical_blocks"],
                saved["independent_gradient"],
            )
            exact(
                row["independent_projected_gradient_inf_norm"],
                float(np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1)))),
            )


def report_from_records(plan, trials, records):
    base.validate_arrays(plan, trials)
    roster = [
        [s, o] for s in plan["selection"]["seeds"] for o in plan["selection"]["origins"]
    ]
    exact([[r["seed"], r["origin"]] for r in records], roster)
    candidate_work = [
        dict(seed=r["seed"], origin=r["origin"], **r["candidate_work"]) for r in records
    ]
    validate_candidate_work(plan, trials, candidate_work)
    criteria = plan["comparison"]["qualification"]
    rows = []
    count = len(plan["selection"]["origins"])
    for index, record in enumerate(records):
        if set(record) != {
            "seed",
            "origin",
            "capture",
            "baseline_work",
            "candidate_work",
            "common",
            "precision",
        }:
            raise ValueError("precision record fields differ")
        if set(record["common"]) != {"baseline", "candidate"}:
            raise ValueError("common audit arm roster differs")
        a, b = (record["common"][arm] for arm in base.ARMS)
        validate_audit(a)
        validate_audit(b)
        if a is None:
            raise ValueError("historical baseline common audit is absent")
        repetition, row = divmod(index, count)
        trial = trials[repetition]
        native = {}
        failures = {}
        for arm_index, arm in enumerate(base.ARMS):
            work = record[arm + "_work"]
            native[arm] = {
                key: trial[key][row, arm_index].item() for key in base.SCALAR_FIELDS
            }
            native[arm] = {
                k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in native[arm].items()
            }
            failures[arm] = bool(
                native[arm]["used_fallback"]
                or work["backend_failure"]
                or work["nonfinite_evaluation"]
                or record["common"][arm] is None
            )
            if record["common"][arm] is not None:
                np.testing.assert_array_equal(
                    record["common"][arm]["blocks"], work["returned_canonical_blocks"]
                )
        ratio = b["residual"] / max(a["residual"], 0.002) if b is not None else None
        reduction = a["value"] - b["value"] if b is not None else None
        denominator = max(abs(a["value"]), 1e-12)
        rows.append(
            dict(
                seed=record["seed"],
                origin=record["origin"],
                native=native,
                common={
                    arm: None
                    if record["common"][arm] is None
                    else {
                        k: record["common"][arm][k]
                        for k in ("value", "residual", "bound_violation")
                    }
                    for arm in base.ARMS
                },
                candidate_failure=failures["candidate"],
                baseline_failure=failures["baseline"],
                baseline_converged=a["residual"] <= 0.002
                and not failures["baseline"]
                and a["bound_violation"] == 0,
                candidate_converged=b is not None
                and b["residual"] <= 0.002
                and not failures["candidate"]
                and b["bound_violation"] == 0,
                residual_ratio=ratio,
                objective_reduction=reduction,
                fractional_objective_reduction=None
                if reduction is None
                else reduction / denominator,
                fractional_objective_regression=None
                if reduction is None
                else max(0.0, -reduction / denominator),
                maximum_normalized_block_change=None
                if b is None
                else float(np.max(np.abs(np.asarray(b["blocks"]) - a["blocks"]))),
                common_command_maximum_absolute_change_by_channel=None
                if b is None
                else np.max(
                    np.abs(np.asarray(b["commands"]) - a["commands"]), axis=0
                ).tolist(),
                common_first_command_absolute_change_by_channel=None
                if b is None
                else np.abs(np.asarray(b["commands"])[0] - a["commands"][0]).tolist(),
            )
        )

    def summary(selected):
        result = dict(origins=len(selected))
        for arm in base.ARMS:
            result[arm] = dict(
                converged=sum(r[arm + "_converged"] for r in selected),
                failures=sum(r[arm + "_failure"] for r in selected),
                objectives=distribution(
                    [
                        None if r["common"][arm] is None else r["common"][arm]["value"]
                        for r in selected
                    ]
                ),
                residuals=distribution(
                    [
                        None
                        if r["common"][arm] is None
                        else r["common"][arm]["residual"]
                        for r in selected
                    ]
                ),
            )
        for key in (
            "objective_reduction",
            "fractional_objective_reduction",
            "residual_ratio",
            "fractional_objective_regression",
            "maximum_normalized_block_change",
        ):
            result[key] = distribution([r[key] for r in selected])
        for key in (
            "common_command_maximum_absolute_change_by_channel",
            "common_first_command_absolute_change_by_channel",
        ):
            result[key] = [
                distribution([None if r[key] is None else r[key][i] for r in selected])
                for i in range(3)
            ]
        result["objective_gains_ties_losses"] = {
            label: sum(
                r["objective_reduction"] is not None
                and operation(r["objective_reduction"])
                for r in selected
            )
            for label, operation in [
                ("gains", lambda x: x > 0),
                ("ties", lambda x: x == 0),
                ("losses", lambda x: x < 0),
            ]
        }
        result["residual_gains_ties_losses"] = {
            label: sum(
                r["common"]["candidate"] is not None
                and operation(
                    r["common"]["baseline"]["residual"]
                    - r["common"]["candidate"]["residual"]
                )
                for r in selected
            )
            for label, operation in [
                ("gains", lambda x: x > 0),
                ("ties", lambda x: x == 0),
                ("losses", lambda x: x < 0),
            ]
        }
        return result

    pooled = summary(rows)
    converged = pooled["candidate"]["converged"]
    failures = pooled["candidate"]["failures"]
    complete = all(r["common"]["candidate"] is not None for r in rows)
    median = pooled["residual_ratio"]["median"] if complete else None
    regression = (
        pooled["fractional_objective_regression"]["maximum"] if complete else None
    )
    bound = (
        max(
            (
                r["common"]["candidate"]["bound_violation"]
                for r in rows
                if r["common"]["candidate"] is not None
            ),
            default=None,
        )
        if complete
        else None
    )
    gates = dict(
        enough_converged_origins=converged >= criteria["minimum_converged_origins"],
        median_residual_ratio=median is not None
        and median <= criteria["maximum_median_residual_ratio"],
        objective_regression=regression is not None
        and regression <= criteria["maximum_fractional_objective_regression"],
        candidate_failures=failures <= criteria["maximum_candidate_failures"],
        command_bounds=bound is not None
        and bound <= criteria["maximum_command_bound_violation"],
    )

    def work_summary(selected):
        work = {}
        for arm in base.ARMS:
            entries = [r[arm + "_work"] for r in selected]
            work[arm] = dict(
                counters={
                    k: dict(
                        total=sum(r[k] for r in entries),
                        maximum=max(r[k] for r in entries),
                    )
                    for k in shared.COUNTERS
                },
                raw_messages={
                    m: sum(r["backend_message"] == m for r in entries)
                    for m in sorted({r["backend_message"] for r in entries})
                },
                backend_failures=sum(r["backend_failure"] for r in entries),
            )
        return work

    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        new_policy_trials=False,
        primary_scoring="Common float64 planning model applied to exact returned normalized blocks; native scores are descriptive only.",
        interpretation=plan["comparison"]["interpretation"],
        pooled=pooled,
        per_seed=[
            dict(
                seed=s,
                **summary([r for r in rows if r["seed"] == s]),
                work=work_summary([r for r in records if r["seed"] == s]),
            )
            for s in plan["selection"]["seeds"]
        ],
        qualification=dict(
            meaning=criteria["meaning"],
            criteria=gates,
            eligible_for_future_tracking_experiment=all(gates.values()),
            converged_origins=converged,
            median_residual_ratio=median,
            maximum_fractional_objective_regression=regression,
            candidate_failures=failures,
            maximum_command_bound_violation=bound,
            maintained_solver_promoted=False,
        ),
        work=dict(
            meaning=plan["comparison"]["work"],
            **work_summary(records),
            common64_audit_calls=sum(
                r["common"][a] is not None for r in records for a in base.ARMS
            ),
        ),
        previously_diagnosed=[
            r
            for r in rows
            if [r["seed"], r["origin"]]
            in [[102, 134], [102, 298], [104, 186], [104, 298]]
        ],
        origins=rows,
    )


def artifact_names(plan):
    return {"run.json", "environment.json", "report.json", "records.json"} | {
        f"trial-{i}/paired.npz" for i in range(len(plan["selection"]["seeds"]))
    }


def run(artifacts, output):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = base.input_snapshot(plan, artifacts)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    trials, records, report = probe(plan, inputs)
    for name, value in [
        ("run.json", dict(no_fit=True, new_policy_trials=False)),
        ("environment.json", environment),
        ("records.json", records),
        ("report.json", report),
    ]:
        write_json(output / name, value)
    for i, trial in enumerate(trials):
        directory = output / f"trial-{i}"
        directory.mkdir()
        np.savez_compressed(directory / "paired.npz", **trial)
    write_json(
        output / "files.json",
        {n: digest((output / n).read_bytes()) for n in sorted(artifact_names(plan))},
    )
    return report


def verify(directory):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved precision plan differs")
    inputs = base.input_snapshot(plan, directory / "inputs")
    names = artifact_names(plan)
    expected = names | {"manifest.json", "files.json"} | {"inputs/" + n for n in inputs}
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    } != expected:
        raise ValueError("precision file roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    exact(sorted(hashes), sorted(names))
    saved = {n: (directory / n).read_bytes() for n in names}
    for name, value in saved.items():
        if digest(value) != hashes[name]:
            raise ValueError(f"precision artifact changed: {name}")
    exact(json.loads(saved["environment.json"]), environment)
    exact(json.loads(saved["run.json"]), dict(no_fit=True, new_policy_trials=False))
    trials = [
        arrays(saved[f"trial-{i}/paired.npz"])
        for i in range(len(plan["selection"]["seeds"]))
    ]
    records = json.loads(saved["records.json"])
    report = json.loads(saved["report.json"])
    exact(report, report_from_records(plan, trials, records))
    fresh, fresh_records, fresh_report = probe(plan, inputs)
    for actual, expected in zip(trials, fresh, strict=True):
        exact_arrays(actual, expected)
    exact(records, fresh_records)
    exact(report, fresh_report)
    return dict(verified=True, no_fit=True, verified_pairs=len(records), report=report)


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
