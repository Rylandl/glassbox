"""Frozen four-case termination diagnosis without changing optimizer behavior."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from glassbox.control.solver import BoundedShootingSolver

from . import first_order_qualification as previous
from . import quasi_newton_qualification as shared
from . import solver_budget as base
from .qualification import _write_snapshot, arrays, digest, write_json
from .solver_trace import EXPONENTS, TracedFirstOrderSolver

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-termination-v1.json"
PLAN_SHA256 = "a6de6c119073298df9497cc5b1b59aabbbfcde94a09a6c94cdd6bb0d2e0988d7"
check_environment = shared.check_environment


def exact(actual, expected):
    """Strict JSON comparison, including scalar types and every numeric bit."""
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("diagnostic replay differs")


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("termination diagnostic plan differs")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"termination inherited source differs: {name}")
    previous.frozen_plan()
    exact(list(EXPONENTS), plan["finite_difference"]["step_exponents"])
    exact(plan["finite_difference"]["step_base"], 2)
    return plan, raw


def task_inputs(inputs):
    for _ in range(3):
        inputs = shared.task_inputs(inputs)
    return inputs


def remap_seed_inputs(inputs, original_repetition):
    prefix = f"trial-{original_repetition}/"
    result = {k: v for k, v in inputs.items() if not k.startswith("trial-")}
    result.update(
        {
            "trial-0/" + k.removeprefix(prefix): v
            for k, v in inputs.items()
            if k.startswith(prefix)
        }
    )
    if not any(k.startswith("trial-0/") for k in result):
        raise ValueError("selected original trial is absent")
    return result


def exact_arrays(actual, expected):
    if set(actual) != set(expected):
        raise ValueError("diagnostic array roster differs")
    for key in actual:
        if actual[key].dtype != expected[key].dtype:
            raise ValueError(f"diagnostic array dtype differs: {key}")
        np.testing.assert_array_equal(actual[key], expected[key])


def probe(plan, inputs):
    historical_plan = json.loads(inputs["manifest.json"])
    historical_work = json.loads(inputs["work.json"])
    historical_trials = [
        arrays(inputs[f"trial-{i}/paired.npz"])
        for i in range(len(historical_plan["selection"]["seeds"]))
    ]
    shared.validate_work(historical_plan, historical_trials, historical_work)
    saved_work = {(r["seed"], r["origin"]): r for r in historical_work}
    original_tasks = task_inputs(inputs)
    trials, records = [], []
    seeds = list(dict.fromkeys(seed for seed, _ in plan["selection"]["cases"]))
    for seed in seeds:
        repetition = historical_plan["selection"]["seeds"].index(seed)
        origins = [o for s, o in plan["selection"]["cases"] if s == seed]
        probe_plan = copy.deepcopy(historical_plan)
        probe_plan["selection"].update(seeds=[seed], origins=origins)
        index = 0

        def solve(
            model,
            policy,
            state,
            reference,
            command,
            *,
            warm_start,
            baseline_check,
            origins=tuple(origins),
            repetition=repetition,
            seed=seed,
        ):
            nonlocal index
            origin = origins[index]
            index += 1
            arguments = (state, reference, command)
            baseline_check(
                BoundedShootingSolver(model, policy).solve(
                    *arguments, warm_start=warm_start
                )
            )
            candidate_policy = base.candidate_policy(policy)
            untraced = previous.FirstOrderSolver(model, candidate_policy)
            traced = TracedFirstOrderSolver(model, candidate_policy)
            a = untraced.solve(*arguments, warm_start=warm_start)
            b = traced.solve(*arguments, warm_start=warm_start)
            row = list(historical_trials[repetition]["origins"]).index(origin)
            for solver, result in ((untraced, a), (traced, b)):
                shared.historical_parity(result, historical_trials[repetition], row)
                previous.historical_work_parity(
                    dict(seed=seed, origin=origin, **solver.last_work),
                    saved_work[seed, origin],
                )
            exact_arrays(
                base.serialize_pair(dict(baseline=a, candidate=a)),
                base.serialize_pair(dict(baseline=b, candidate=b)),
            )
            exact(untraced.last_work, traced.last_work)
            records.append(
                dict(
                    seed=seed,
                    origin=origin,
                    work=copy.deepcopy(traced.last_work),
                    trace=traced.last_trace,
                    directions=traced.last_directions,
                )
            )
            return dict(baseline=a, candidate=b)

        result, _ = base.probe(
            probe_plan, remap_seed_inputs(original_tasks, repetition), _solve_pair=solve
        )
        exact(index, len(origins))
        trials.append(result[0])
    return trials, records, report_from_records(plan, records)


def distribution(values):
    values = [float(x) for x in values if x is not None]
    return dict(
        count=len(values),
        minimum=min(values) if values else None,
        median=float(np.median(values)) if values else None,
        maximum=max(values) if values else None,
    )


def trace_summary(trace):
    requests, callbacks = trace["requests"], trace["callbacks"]
    exact([r["index"] for r in requests], list(range(len(requests))))
    exact([r["index"] for r in callbacks], list(range(len(callbacks))))
    start = callbacks[-1]["request_count"] if callbacks else 0
    terminal = requests[start:]
    origin = (
        callbacks[-1]["canonical_blocks"]
        if callbacks
        else trace["seed"]["canonical_blocks"]
    )
    canonical_hosts = {}
    collapsed_hosts = 0
    for row in terminal:
        canonical = tuple(np.asarray(row["canonical_blocks"]).ravel())
        host = tuple(np.asarray(row["host_blocks"]).ravel())
        if canonical in canonical_hosts and host not in canonical_hosts[canonical]:
            collapsed_hosts += 1
        canonical_hosts.setdefault(canonical, set()).add(host)
    return dict(
        requests=len(requests),
        accepted_callbacks=len(callbacks),
        cached_seed_requests=sum(r["cached_seed"] for r in requests),
        terminal_requests=len(terminal),
        terminal_unique_canonical_points=len(canonical_hosts),
        terminal_extra_distinct_host_proposals_sharing_canonical_points=collapsed_hosts,
        adjacent_terminal_objective_ties=sum(
            a["value"] == b["value"] for a, b in pairwise(terminal)
        ),
        terminal_maximum_canonical_movement=max(
            (
                float(np.max(np.abs(np.asarray(r["canonical_blocks"]) - origin)))
                for r in terminal
            ),
            default=0.0,
        ),
        terminal_values=distribution([r["value"] for r in terminal]),
        raw_backend=trace["raw_backend"],
    )


def report_from_records(plan, records):
    exact([[r["seed"], r["origin"]] for r in records], plan["selection"]["cases"])
    rows = []
    for row in records:
        if set(row) != {"seed", "origin", "work", "trace", "directions"}:
            raise ValueError("diagnostic record fields differ")
        work, directions = row["work"], row["directions"]
        exact(directions["step_exponents"], plan["finite_difference"]["step_exponents"])
        np.testing.assert_array_equal(
            directions["base"]["canonical_blocks"], work["returned_canonical_blocks"]
        )
        np.testing.assert_array_equal(
            directions["base"]["gradient"], work["independent_gradient"]
        )
        exact(directions["base"]["value"], work["independent_objective"])
        probes = directions["probes"]
        by_step = []
        for exponent in plan["finite_difference"]["step_exponents"]:
            selected = [p for p in probes if p["exponent"] == exponent]
            by_step.append(
                dict(
                    exponent=exponent,
                    statuses={
                        s: sum(p["status"] == s for p in selected)
                        for s in sorted({p["status"] for p in selected})
                    },
                    observed_objective_ties=sum(
                        p["observed_change"] == 0
                        for p in selected
                        if p["observed_change"] is not None
                    ),
                    predicted_change_in_spacings=distribution(
                        [p["predicted_change_in_spacings"] for p in selected]
                    ),
                    observed_predicted_ratio=distribution(
                        [p["observed_predicted_ratio"] for p in selected]
                    ),
                    absolute_linearization_discrepancy=distribution(
                        [p["absolute_linearization_discrepancy"] for p in selected]
                    ),
                )
            )
        blocks = np.asarray(work["returned_canonical_blocks"])
        rows.append(
            dict(
                seed=row["seed"],
                origin=row["origin"],
                saved_and_untraced_parity=True,
                residual=work["independent_projected_gradient_inf_norm"],
                active_bound_coordinates=int(np.sum(np.abs(blocks) == 1)),
                work=work,
                trace=trace_summary(row["trace"]),
                directional_function_calls=directions["diagnostic_function_calls"],
                directional_checks_by_step=by_step,
            )
        )
    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        new_policy_trials=False,
        new_solver_candidate=False,
        cases=rows,
        diagnostic_function_calls=sum(r["directional_function_calls"] for r in rows),
        interpretation=plan["readout"]["decision"],
        terminal_segment_meaning=plan["instrumentation"]["interpretation"],
        directional_meaning=plan["finite_difference"]["interpretation"],
    )


def artifact_names(plan):
    seeds = {seed for seed, _ in plan["selection"]["cases"]}
    return {"run.json", "environment.json", "records.json", "report.json"} | {
        f"trial-{i}/paired.npz" for i in range(len(seeds))
    }


def run(artifacts, output):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = base.input_snapshot(plan, artifacts)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    trials, records, report = probe(plan, inputs)
    write_json(
        output / "run.json",
        dict(no_fit=True, new_policy_trials=False, new_solver_candidate=False),
    )
    write_json(output / "environment.json", environment)
    write_json(output / "records.json", records)
    write_json(output / "report.json", report)
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
    return report


def verify(directory):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved diagnostic plan differs")
    inputs = base.input_snapshot(plan, directory / "inputs")
    names = artifact_names(plan)
    expected = names | {"manifest.json", "files.json"} | {"inputs/" + n for n in inputs}
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    } != expected:
        raise ValueError("diagnostic complete file roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    exact(sorted(hashes), sorted(names))
    saved = {name: (directory / name).read_bytes() for name in names}
    for name, data in saved.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"altered diagnostic artifact: {name}")
    exact(json.loads(saved["environment.json"]), environment)
    exact(
        json.loads(saved["run.json"]),
        dict(no_fit=True, new_policy_trials=False, new_solver_candidate=False),
    )
    records, report = (
        json.loads(saved["records.json"]),
        json.loads(saved["report.json"]),
    )
    exact(report, report_from_records(plan, records))
    trials = [
        arrays(saved[f"trial-{i}/paired.npz"])
        for i in range(len({s for s, _ in plan["selection"]["cases"]}))
    ]
    fresh, fresh_records, fresh_report = probe(plan, inputs)
    for actual, expected in zip(trials, fresh, strict=True):
        exact_arrays(actual, expected)
    exact(records, fresh_records)
    exact(report, fresh_report)
    return dict(verified=True, no_fit=True, verified_cases=len(records), report=report)


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
