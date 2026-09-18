"""Frozen no-fit tracking diagnostic of existing bounded solver candidates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from glassbox.control.plan import ReferenceTrajectory

from . import quasi_newton_qualification as work_schema
from . import solver_precision as precision
from . import task_qualification as task
from .harness import control_pass_criterion, control_reference
from .qualification import (
    _context,
    _prewarm,
    _replay_trial,
    _write_snapshot,
    arrays,
    digest,
    same,
    write_json,
)
from .qualification_control import HORIZON_STEPS, WARMUP_INTERVALS, CascadeEquations
from .solver_termination import exact, exact_arrays
from .tracking_solver import TrackingOracleArm

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-tracking-v1.json"
PLAN_SHA256 = "ed0d9429e85cb9bfdeb9263eae868625fc1624033b63689369fa27fe9654c8f0"
ARMS = ("oracle_pg4", "oracle_lbfgs32", "oracle_lbfgs64")
RECORD_FIELDS = {
    "arm",
    "origin",
    "work",
    "capture",
    "precision",
    "audits",
    "seed_selection",
}


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("tracking diagnostic plan differs")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"tracking inherited source changed: {name}")
    task.frozen_plan()
    precision.frozen_plan()
    return plan, raw


def check_environment(plan):
    # Cascade provenance is checked by control_fixture against the pinned
    # control manifest; it is not a fifth Python package-version string.
    versions = {k: plan["environment"][k] for k in ("python", "jax", "numpy", "scipy")}
    return work_schema.check_environment(dict(plan, environment=versions))


input_snapshot = task.input_snapshot


def artifact_names(plan):
    names = {
        "run.json",
        "environment.json",
        "results.json",
        "report.json",
        "prewarm.json",
    }
    for i, order in enumerate(plan["control_qualification"]["arm_order"]):
        for name in order:
            names |= {
                f"trial-{i}/{name}/{file}"
                for file in (
                    "trial.json",
                    "tracking.npz",
                    "timing.npz",
                    "oracle.npz",
                    "records.json",
                )
            }
    return names


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(
        count=int(values.size),
        minimum=float(np.min(values)) if values.size else None,
        median=float(np.median(values)) if values.size else None,
        p95=float(np.percentile(values, 95)) if values.size else None,
        maximum=float(np.max(values)) if values.size else None,
    )


def validate_records(name, tracking, diagnostics, records):
    shape_arm = SimpleNamespace(plan=SimpleNamespace(command_size=3))
    count = task._diagnostic_schema(shape_arm, tracking, diagnostics)
    if not isinstance(records, list) or len(records) != count:
        raise ValueError("tracking solver record count differs")
    for i, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError("tracking solver record fields differ")
        exact(record["arm"], name)
        exact(record["origin"], i + WARMUP_INTERVALS)
        if set(record["audits"]) != {"lifted_seed64", "returned_plan64"}:
            raise ValueError("tracking audit role names differ")
        if name == ARMS[0]:
            if any(
                record[key] is not None
                for key in ("work", "capture", "precision", "seed_selection")
            ) or any(v is not None for v in record["audits"].values()):
                raise ValueError("PG4 cannot claim L-BFGS work or precision evidence")
            continue
        work = record["work"]
        if not isinstance(work, dict) or set(work) != work_schema.WORK_FIELDS - {
            "seed",
            "origin",
        }:
            raise ValueError("tracking L-BFGS work fields differ")
        for key, limit in work_schema.COUNTERS.items():
            value = work[key]
            if (
                type(value) is not int
                or value < 0
                or (limit is not None and value > limit)
            ):
                raise ValueError(f"tracking work counter differs: {key}")
        if any(type(work[key]) is not bool for key in work_schema.FLAGS):
            raise ValueError("tracking work flag type differs")
        if (
            work["backend_status"] is not None
            and type(work["backend_status"]) is not int
        ):
            raise ValueError("tracking backend status differs")
        if (
            not isinstance(work["backend_message"], str)
            or work["stop_cause"] not in work_schema.STOPS
        ):
            raise ValueError("tracking backend diagnostic differs")
        fallback = bool(diagnostics["used_fallback"][i])
        if not fallback:
            exact(work["accepted_iterations"], int(diagnostics["iterations"][i]))
            exact(work["audit_objective_evaluations"], 1)
            exact(
                work["independent_objective"], float(diagnostics["final_objectives"][i])
            )
            exact(
                work["independent_projected_gradient_inf_norm"],
                float(diagnostics["projected_gradient_inf_norm"][i]),
            )
        saved_arrays = {}
        for key in work_schema.WORK_ARRAYS:
            value = work[key]
            if value is None:
                if not fallback:
                    raise ValueError("finite tracking result lacks gradient/blocks")
                continue
            array = np.asarray(
                value, dtype=np.float32 if name == ARMS[1] else np.float64
            )
            if (
                not isinstance(value, list)
                or array.shape != (5, 3)
                or not np.isfinite(array).all()
            ):
                raise ValueError("tracking gradient/blocks shape or values differ")
            if key == "returned_canonical_blocks" and np.any(np.abs(array) > 1):
                raise ValueError("tracking canonical blocks exceed bounds")
            saved_arrays[key] = array
        if saved_arrays:
            if set(saved_arrays) != work_schema.WORK_ARRAYS:
                raise ValueError("incomplete tracking audited arrays")
            blocks, gradient = (
                saved_arrays["returned_canonical_blocks"],
                saved_arrays["independent_gradient"],
            )
            exact(
                work["independent_projected_gradient_inf_norm"],
                float(np.max(np.abs(blocks - np.clip(blocks - gradient, -1, 1)))),
            )
        if name == ARMS[1]:
            if any(
                record[key] is not None
                for key in ("capture", "precision", "seed_selection")
            ) or any(v is not None for v in record["audits"].values()):
                raise ValueError("float32 arm cannot claim64 evidence")
            continue
        selection = record["seed_selection"]
        if not isinstance(selection, dict) or set(selection) != {
            "objective_evaluations",
            "successful",
            "used_warm_start",
            "status",
            "used_fallback",
        }:
            raise ValueError("tracking seed selection schema differs")
        if (
            type(selection["objective_evaluations"]) is not int
            or not 0 <= selection["objective_evaluations"] <= 2
        ):
            raise ValueError("tracking seed selection counter differs")
        if any(
            type(selection[k]) is not bool
            for k in ("successful", "used_warm_start", "used_fallback")
        ) or not isinstance(selection["status"], str):
            raise ValueError("tracking seed selection flag differs")
        if selection["successful"]:
            if record["capture"] is None or record["precision"] is None:
                raise ValueError("successful tracking64 seed lacks evidence")
            exact(work["seed_objective_evaluations"], 1)
            if (
                record["precision"]["floating_dtype"] != "float64"
                or record["precision"]["jaxpr"]["float32_arithmetic"] != 0
            ):
                raise ValueError("tracking64 arithmetic evidence differs")
            for audit in record["audits"].values():
                precision.validate_audit(audit)
            seed_audit = record["audits"]["lifted_seed64"]
            if seed_audit is None:
                raise ValueError("tracking64 seed audit absent")
            exact(seed_audit["blocks"], record["capture"]["blocks"])
            returned = record["audits"]["returned_plan64"]
            if not fallback:
                if returned is None:
                    raise ValueError("tracking64 returned audit absent")
                exact(returned["blocks"], work["returned_canonical_blocks"])
                exact(returned["gradient"], work["independent_gradient"])
                exact(returned["value"], work["independent_objective"])
        elif (
            not fallback
            or any(record[k] is not None for k in ("capture", "precision"))
            or any(v is not None for v in record["audits"].values())
        ):
            raise ValueError(
                "failed tracking64 seed must retain fallback and absent audits"
            )
    return count


def report(plan, rows, saved):
    task.validate_rows(plan, rows)
    q = plan["control_qualification"]
    control_raw = (ROOT / "docs/harness/control-v5.json").read_bytes()
    if digest(control_raw) != plan["input_sha256"]["control/manifest.json"]:
        raise ValueError("tracking scoring manifest differs from pinned input")
    control = json.loads(control_raw)
    prewarm = json.loads(saved["prewarm.json"])
    roster = [[r["repetition"], r["arm"]] for r in rows]
    exact([[r["repetition"], r["arm"]] for r in prewarm], roster)
    for r in prewarm:
        if (
            set(r) != {"repetition", "arm", "elapsed_seconds"}
            or type(r["elapsed_seconds"]) is not float
            or not np.isfinite(r["elapsed_seconds"])
            or r["elapsed_seconds"] < 0
        ):
            raise ValueError("tracking prewarm timing differs")
    trials = []
    material = []
    for row in rows:
        prefix = row["directory"] + "/"
        tracking, diagnostics, timing = [
            arrays(saved[prefix + name])
            for name in ("tracking.npz", "oracle.npz", "timing.npz")
        ]
        records = json.loads(saved[prefix + "records.json"])
        validate_records(row["arm"], tracking, diagnostics, records)
        if set(timing) != {"tick_times_s", "solve_times_s", "deadline_assessed"}:
            raise ValueError("tracking timing fields differ")
        attempted = len(tracking["solver_used"])
        for key, size in (
            ("solve_times_s", attempted),
            ("tick_times_s", len(tracking["commands"])),
        ):
            value = timing[key]
            if (
                value.shape != (size,)
                or not np.isfinite(value).all()
                or np.any(value < 0)
            ):
                raise ValueError("tracking timing count or values differ")
        if timing["deadline_assessed"].dtype != bool:
            raise ValueError("tracking deadline dtype differs")
        np.testing.assert_array_equal(
            timing["deadline_assessed"], np.zeros(attempted, dtype=bool)
        )
        np.testing.assert_array_equal(
            timing["solve_times_s"][:WARMUP_INTERVALS],
            np.zeros(min(attempted, WARMUP_INTERVALS)),
        )
        criterion = row["pass_criterion"]
        exact(
            criterion,
            control_pass_criterion(
                tracking["states"], tracking["reference_anchor_state"], control
            ),
        )
        expected_samples = q["application"]["scored_samples_per_trial"]
        exact(criterion["scored_samples"], expected_samples)
        if (
            type(criterion["within_samples"]) is not int
            or not 0 <= criterion["within_samples"] <= expected_samples
        ):
            raise ValueError("tracking application count differs")
        exact(
            criterion["within_tolerance_fraction"],
            criterion["within_samples"] / expected_samples,
        )
        material.append((row, diagnostics, records, timing))
        trials.append(dict(row))

    def summarize(selected):
        selected_rows = [x[0] for x in selected]
        name = selected_rows[0]["arm"]
        all_records = [r for _, _, records, _ in selected for r in records]
        all_work = [r["work"] for r in all_records if r["work"] is not None]
        backend_failures = sum(w["backend_failure"] for w in all_work)
        nonfinite = sum(w["nonfinite_evaluation"] for w in all_work)
        fallback = sum(r["fallback_count"] for r in selected_rows)
        bound = max(r["maximum_command_bound_violation"] for r in selected_rows)
        adequacy = all(
            r["pass_criterion"]["met"]
            and not r["terminated"]
            and r["fallback_count"] == 0
            and r["maximum_command_bound_violation"] == 0
            for r in selected_rows
        )
        reliable = not (backend_failures or nonfinite or fallback or bound)
        solve_times = np.concatenate(
            [t["solve_times_s"][WARMUP_INTERVALS:] for _, _, _, t in selected]
        )
        tick_times = np.concatenate([t["tick_times_s"] for _, _, _, t in selected])
        native_residuals = np.concatenate(
            [d["projected_gradient_inf_norm"] for _, d, _, _ in selected]
        )
        counters = {
            k: sum(w[k] for w in all_work) if name != ARMS[0] else None
            for k in work_schema.COUNTERS
        }
        counters["accepted_iterations"] = sum(
            int(np.sum(d["iterations"])) for _, d, _, _ in selected
        )
        cost = dict(
            **counters,
            seed_selection_float32_objective_evaluations=sum(
                r["seed_selection"]["objective_evaluations"]
                for r in all_records
                if r["seed_selection"] is not None
            ),
            extra_float64_audit_calls=sum(
                a is not None for r in all_records for a in r["audits"].values()
            ),
            outer_solve_seconds=distribution(solve_times),
            tick_seconds=distribution(tick_times),
            ticks_over_sample_interval=int(
                np.count_nonzero(tick_times > q["sample_interval_s"])
            ),
            solves_over_sample_interval=int(
                np.count_nonzero(solve_times > q["sample_interval_s"])
            ),
            elapsed_seconds=sum(r["wall"]["elapsed_seconds"] for r in selected_rows),
            prewarm_seconds=sum(
                p["elapsed_seconds"]
                for p in prewarm
                if p["arm"] == name
                and any(r["repetition"] == p["repetition"] for r in selected_rows)
            ),
            objective_evaluation_count_available=name != ARMS[0],
        )
        samples = sum(r["pass_criterion"]["scored_samples"] for r in selected_rows)
        within = sum(r["pass_criterion"]["within_samples"] for r in selected_rows)
        return dict(
            trials=len(selected_rows),
            completed_trials=sum(not r["terminated"] for r in selected_rows),
            scored_samples=samples,
            within_samples=within,
            within_tolerance_fraction=within / samples,
            task_passed_trials=sum(r["pass_criterion"]["met"] for r in selected_rows),
            application_adequate=adequacy,
            backend_failure_solves=backend_failures,
            nonfinite_evaluation_solves=nonfinite,
            fallback_count=fallback,
            maximum_command_bound_violation=bound,
            solver_reliable=reliable,
            meets_application_and_reliability=adequacy and reliable,
            cost=cost,
            native_residuals=distribution(native_residuals),
            native_residual_at_threshold_solves=int(
                np.count_nonzero(
                    np.isfinite(native_residuals) & (native_residuals <= 0.002)
                )
            ),
            native_converged_solves=sum(
                bool(
                    np.isfinite(d["projected_gradient_inf_norm"][i])
                    and d["projected_gradient_inf_norm"][i] <= 0.002
                    and not d["used_fallback"][i]
                    and (
                        record["work"] is None
                        or not (
                            record["work"]["backend_failure"]
                            or record["work"]["nonfinite_evaluation"]
                        )
                    )
                )
                for _, d, records, _ in selected
                for i, record in enumerate(records)
            ),
            raw_backend_messages={
                message: sum(w["backend_message"] == message for w in all_work)
                for message in sorted({w["backend_message"] for w in all_work})
            },
        )

    trial_numerics = {}
    for entry, output in zip(material, trials, strict=True):
        summary = summarize([entry])
        numerical = {
            k: summary[k]
            for k in (
                "backend_failure_solves",
                "nonfinite_evaluation_solves",
                "native_converged_solves",
                "native_residual_at_threshold_solves",
                "native_residuals",
                "raw_backend_messages",
                "cost",
            )
        }
        output["solver_diagnostics"] = numerical
        trial_numerics[(entry[0]["repetition"], entry[0]["arm"])] = numerical
    pairs = []
    for repetition, seed in enumerate(q["seeds"]):
        selected = {r["arm"]: r for r in rows if r["repetition"] == repetition}
        comparisons = {}
        for label, a, b in [
            ("float64_minus_float32", ARMS[1], ARMS[2]),
            ("float32_minus_pg4", ARMS[0], ARMS[1]),
            ("float64_minus_pg4", ARMS[0], ARMS[2]),
        ]:
            differences = {}
            for key in (
                "within_tolerance_fraction",
                "lateral_rmse_m",
                "altitude_rmse_m",
            ):
                av, bv = (
                    selected[a]["pass_criterion"][key],
                    selected[b]["pass_criterion"][key],
                )
                differences[key] = None if av is None or bv is None else bv - av
            for key in ("backend_failure_solves", "nonfinite_evaluation_solves"):
                differences[key] = (
                    trial_numerics[(repetition, b)][key]
                    - trial_numerics[(repetition, a)][key]
                )
            differences["fallback_count"] = (
                selected[b]["fallback_count"] - selected[a]["fallback_count"]
            )
            differences["terminated_trials"] = int(selected[b]["terminated"]) - int(
                selected[a]["terminated"]
            )
            comparisons[label] = differences
        pairs.append(dict(initial_state_seed=seed, comparisons=comparisons))
    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        new_policy_trials=True,
        maintained_solver_promoted=False,
        learner_changed=False,
        historical_precision_qualification_passed=False,
        policy_decision=plan["policy_decision"],
        interpretation=q["comparison"],
        timing_meaning=q["timing"],
        per_arm={
            name: summarize([x for x in material if x[0]["arm"] == name])
            for name in ARMS
        },
        paired_differences=pairs,
        trials=trials,
    )


def verify_solves(arm, manifest, tracking, diagnostics, records):
    count = validate_records(arm.name, tracking, diagnostics, records)
    states, commands = tracking["states"], tracking["commands"]
    previous = tracking["initial_command"]
    arm.reset(states[0], previous)
    warm_start = None
    for index in range(len(tracking["solver_used"])):
        arm.observe(states[index])
        command = commands[index] if index < len(commands) else previous
        if arm.ready:
            future = (index + np.arange(HORIZON_STEPS + 1)) * manifest["trial"][
                "sample_interval_s"
            ]
            reference = ReferenceTrajectory(
                control_reference(
                    tracking["reference_anchor_state"],
                    future,
                    manifest["tracking_reference"],
                )
            )
            result = arm.solve(
                states[index], reference, previous, warm_start=warm_start
            )
            warm_start = result.warm_start
            row = index - WARMUP_INTERVALS
            if index == len(commands):
                command = diagnostics["candidate_commands"][row, 0]
            np.testing.assert_array_equal(result.command, command)
            np.testing.assert_array_equal(
                result.predicted_commands, diagnostics["candidate_commands"][row]
            )
        else:
            np.testing.assert_array_equal(command, previous)
        if index < len(commands):
            arm.command_applied(command)
            previous = command
    np.testing.assert_allclose(
        arm.equations.canonical(arm._state), states[-1], rtol=1e-5, atol=1e-5
    )
    fresh = arm.diagnostic_arrays()
    exact_arrays(fresh, diagnostics)
    exact(arm.records, records)
    statuses = {"model_not_ready": min(len(tracking["solver_used"]), WARMUP_INTERVALS)}
    for status in fresh["statuses"]:
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    return dict(
        verified_optimizer_solves=count,
        verified_oracle_forecasts=int(np.count_nonzero(~fresh["used_fallback"])),
        solver_statuses=statuses,
    )


def run(artifacts, output):
    from .harness import _control_tracking_plant, _control_trial, control_initial_state

    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = input_snapshot(plan, artifacts)
    context = _context(inputs)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    write_json(output / "run.json", dict(no_fit=True, new_policy_trials=True))
    write_json(output / "environment.json", environment)
    equations = CascadeEquations(context.model)
    q = plan["control_qualification"]

    def reference_fn(times):
        return control_reference(
            context.initial_state, times, context.manifest["tracking_reference"]
        )

    rows, prewarm = [], []
    for repetition, order in enumerate(q["arm_order"]):
        seed = q["seeds"][repetition]
        with jax.enable_x64(True):
            start = control_initial_state(context.manifest, context.initial_state, seed)
        for name in order:
            print(
                json.dumps(dict(tracking=f"{repetition}-{name}", seed=seed)), flush=True
            )
            began = time.perf_counter()
            arm = TrackingOracleArm(
                plan,
                context.manifest,
                context.learned,
                context.model,
                name,
                _equations=equations,
            )
            _prewarm(arm, context, reference_fn)
            prewarm.append(
                dict(
                    repetition=repetition,
                    arm=name,
                    elapsed_seconds=time.perf_counter() - began,
                )
            )
            plant = _control_tracking_plant(
                context.manifest, start, context.initial_command
            )
            directory = output / f"trial-{repetition}" / name
            row = _control_trial(
                context.manifest,
                arm,
                plant,
                reference_fn,
                context.initial_state,
                directory,
            )
            row = dict(
                row,
                repetition=repetition,
                initial_state_seed=seed,
                directory=str(directory.relative_to(output)),
            )
            np.savez_compressed(directory / "oracle.npz", **arm.diagnostic_arrays())
            write_json(directory / "records.json", arm.records)
            write_json(directory / "trial.json", row)
            rows.append(row)
            write_json(output / "results.json", rows)
            write_json(output / "prewarm.json", prewarm)
            print(
                json.dumps(
                    dict(
                        trial=row["directory"],
                        pass_criterion=row["pass_criterion"],
                        tracking_rmse=row["tracking_rmse"],
                    )
                ),
                flush=True,
            )
    names = artifact_names(plan)
    saved = {n: (output / n).read_bytes() for n in names - {"report.json"}}
    result = report(plan, rows, saved)
    write_json(output / "report.json", result)
    write_json(
        output / "files.json",
        {n: digest((output / n).read_bytes()) for n in sorted(names)},
    )
    return result


def verify(directory):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved tracking plan differs")
    inputs = input_snapshot(plan, directory / "inputs", saved=True)
    names = artifact_names(plan)
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    }
    if actual != names | {"manifest.json", "files.json"} | {
        "inputs/" + n for n in inputs
    }:
        raise ValueError("tracking artifact roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    exact(sorted(hashes), sorted(names))
    saved = {n: (directory / n).read_bytes() for n in names}
    for n, data in saved.items():
        if digest(data) != hashes[n]:
            raise ValueError(f"tracking artifact changed: {n}")
    exact(json.loads(saved["run.json"]), dict(no_fit=True, new_policy_trials=True))
    exact(json.loads(saved["environment.json"]), environment)
    rows = json.loads(saved["results.json"])
    result = report(plan, rows, saved)
    exact(json.loads(saved["report.json"]), result)
    context = _context(inputs)
    equations = CascadeEquations(context.model)
    checks = []
    for row in rows:
        print(json.dumps(dict(replay=row["directory"])), flush=True)
        exact(row, json.loads(saved[row["directory"] + "/trial.json"]))
        arm = TrackingOracleArm(
            plan,
            context.manifest,
            context.learned,
            context.model,
            row["arm"],
            _equations=equations,
        )
        exact(row["controller"], arm.summary())
        tracking = arrays(saved[row["directory"] + "/tracking.npz"])
        tracking["initial_command"] = context.initial_command
        diagnostics = arrays(saved[row["directory"] + "/oracle.npz"])
        records = json.loads(saved[row["directory"] + "/records.json"])
        physical = (
            task.replay_failed_prefix(plan, context, row, saved, diagnostics)
            if row["terminated"]
            else _replay_trial(plan, context, row, saved)
        )
        exact(physical["state_difference"], 0.0)
        optimizer = verify_solves(arm, context.manifest, tracking, diagnostics, records)
        same(optimizer["solver_statuses"], row["solver_statuses"])
        np.testing.assert_array_equal(
            tracking["used_fallback"][WARMUP_INTERVALS:], diagnostics["used_fallback"]
        )
        checks.append(dict(physical=physical, optimizer=optimizer))
    return dict(
        verified=True,
        no_fit=True,
        verified_trials=len(checks),
        verified_optimizer_solves=sum(
            c["optimizer"]["verified_optimizer_solves"] for c in checks
        ),
        verified_oracle_forecasts=sum(
            c["optimizer"]["verified_oracle_forecasts"] for c in checks
        ),
        maximum_state_replay_difference=max(
            c["physical"]["state_difference"] for c in checks
        ),
        report=result,
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
