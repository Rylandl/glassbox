"""Frozen 4-versus-64-iteration probes at original saved oracle origins.

This diagnostic replays issued commands; probe plans never advance the system
or seed another origin. It makes no model-adoption or task-success decision.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from glassbox.control.plan import NMPCWarmStart, ReferenceTrajectory
from glassbox.control.solver import BoundedShootingSolver

from .harness import control_reference
from .qualification import _context, _write_snapshot, arrays, digest, same, write_json
from .qualification_control import HORIZON_STEPS, WARMUP_INTERVALS, CascadeEquations
from .task_qualification import TaskOracleArm, _diagnostic_schema, check_environment

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/solver-budget-v1.json"
PLAN_SHA256 = "40125e9d7cf9335584b3fc166a4a60f0eee121ee45f7a3b5d27020444544cdf6"
ARMS = ("baseline", "candidate")
SCORE_TOLERANCE = dict(rtol=1e-7, atol=1e-9)
STATE_TOLERANCE = dict(rtol=1e-5, atol=1e-5)
SCALAR_FIELDS = {
    "initial_objectives": float,
    "final_objectives": float,
    "projected_gradient_inf_norm": float,
    "iterations": np.int64,
    "statuses": "U32",
    "warm_start_used": bool,
    "used_fallback": bool,
    "command_bound_violation": float,
}


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("solver budget plan differs from frozen source")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"solver budget inherited source changed: {name}")
    return plan, raw


def input_snapshot(plan, root):
    root = Path(root)
    if {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != set(
        plan["input_sha256"]
    ):
        raise ValueError("solver budget parent file roster differs")
    result = {name: (root / name).read_bytes() for name in plan["input_sha256"]}
    for name, data in result.items():
        if digest(data) != plan["input_sha256"][name]:
            raise ValueError(f"solver budget input changed: {name}")
    return result


def candidate_policy(policy):
    if policy.maximum_iterations != 4:
        raise ValueError("solver budget requires the original four-iteration policy")
    return replace(policy, maximum_iterations=64)


def original_warm_start(diagnostics, origin):
    if origin == WARMUP_INTERVALS:
        return None
    rows = np.flatnonzero(diagnostics["solve_indices"] == origin - 1)
    if len(rows) != 1 or diagnostics["used_fallback"][rows[0]]:
        raise ValueError("original previous nonfallback solve is absent")
    return NMPCWarmStart(diagnostics["candidate_commands"][rows[0]])


def paired_solve(
    model,
    policy,
    state,
    reference,
    previous_command,
    *,
    warm_start=None,
    baseline_check=None,
):
    pair = {}
    for name, p in zip(ARMS, (policy, candidate_policy(policy)), strict=True):
        pair[name] = BoundedShootingSolver(model, p).solve(
            state, reference, previous_command, warm_start=warm_start
        )
        if name == "baseline" and baseline_check is not None:
            baseline_check(pair[name])
    a, b = (pair[name] for name in ARMS)
    np.testing.assert_allclose(
        a.diagnostics.initial_objective,
        b.diagnostics.initial_objective,
        **SCORE_TOLERANCE,
    )
    same(a.diagnostics.warm_start_used, b.diagnostics.warm_start_used)
    if not a.used_fallback and not b.used_fallback:
        x, y = a.diagnostics.final_objective, b.diagnostics.final_objective
        if np.isfinite(x) and np.isfinite(y) and y > x:
            np.testing.assert_allclose(y, x, **SCORE_TOLERANCE)
    return pair


def serialize_pair(pair):
    values = {key: [] for key in SCALAR_FIELDS}
    commands, states = [], []
    for name in ARMS:
        result = pair[name]
        d = result.diagnostics
        commands.append(np.asarray(result.predicted_commands))
        states.append(
            np.zeros((HORIZON_STEPS + 1, 13))
            if result.used_fallback
            else np.asarray(result.predicted_states)
        )
        row = dict(
            initial_objectives=d.initial_objective,
            final_objectives=d.final_objective,
            projected_gradient_inf_norm=d.final_projected_gradient_inf_norm,
            iterations=d.iterations,
            statuses=str(result.status),
            warm_start_used=d.warm_start_used,
            used_fallback=result.used_fallback,
            command_bound_violation=d.maximum_command_bound_violation,
        )
        for key in values:
            values[key].append(row[key])
    return dict(
        commands=np.asarray(commands),
        states=np.asarray(states),
        **{
            key: np.asarray(value, dtype=SCALAR_FIELDS[key])
            for key, value in values.items()
        },
    )


def validate_arrays(plan, trials):
    if len(trials) != len(plan["selection"]["seeds"]):
        raise ValueError("solver budget trial roster differs")
    count = len(plan["selection"]["origins"])
    shapes = dict(
        origins=(count,),
        commands=(count, 2, HORIZON_STEPS, 3),
        states=(count, 2, HORIZON_STEPS + 1, 13),
    )
    shapes.update({key: (count, 2) for key in SCALAR_FIELDS})
    for trial in trials:
        if set(trial) != set(shapes):
            raise ValueError("solver budget array fields differ")
        for key, shape in shapes.items():
            value = np.asarray(trial[key])
            if value.shape != shape:
                raise ValueError(f"solver budget array shape differs: {key}")
            kind = np.dtype(
                SCALAR_FIELDS.get(key, np.int64 if key == "origins" else float)
            ).kind
            if value.dtype.kind != kind:
                raise ValueError(f"solver budget array dtype differs: {key}")
            if kind == "f" and (
                np.isnan(value).any()
                or (
                    key
                    not in {
                        "initial_objectives",
                        "final_objectives",
                        "projected_gradient_inf_norm",
                    }
                    and not np.isfinite(value).all()
                )
            ):
                raise ValueError(f"solver budget invalid numeric array: {key}")
        np.testing.assert_array_equal(trial["origins"], plan["selection"]["origins"])
        limits = [plan["comparison"][f"{name}_maximum_iterations"] for name in ARMS]
        if np.any(trial["iterations"] < 0) or np.any(trial["iterations"] > limits):
            raise ValueError("solver budget iteration bounds differ")
        if np.any(trial["command_bound_violation"] < 0):
            raise ValueError("solver budget negative bound violation")
        np.testing.assert_array_equal(trial["states"][trial["used_fallback"]], 0)


def report(plan, trials, command_range, gradient_tolerance):
    validate_arrays(plan, trials)
    rows = []
    for seed, trial in zip(plan["selection"]["seeds"], trials, strict=True):
        for i, origin in enumerate(trial["origins"]):
            measurements = {}
            for a, name in enumerate(ARMS):
                d = {key: trial[key][i, a].item() for key in SCALAR_FIELDS}
                finite = all(
                    np.isfinite(d[key])
                    for key in (
                        "initial_objectives",
                        "final_objectives",
                        "projected_gradient_inf_norm",
                    )
                )
                d["finite_bounded_nonfallback"] = bool(
                    finite
                    and not d["used_fallback"]
                    and d["command_bound_violation"] == 0
                )
                d["below_gradient_threshold"] = bool(
                    d["finite_bounded_nonfallback"]
                    and d["projected_gradient_inf_norm"] <= gradient_tolerance
                )
                measurements[name] = {
                    k: None if isinstance(v, float) and not np.isfinite(v) else v
                    for k, v in d.items()
                }
            a, b = (measurements[name] for name in ARMS)
            reduction = (
                a["final_objectives"] - b["final_objectives"]
                if a["finite_bounded_nonfallback"] and b["finite_bounded_nonfallback"]
                else None
            )
            gradient_change = (
                b["projected_gradient_inf_norm"] - a["projected_gradient_inf_norm"]
                if reduction is not None
                else None
            )
            rows.append(
                dict(
                    seed=seed,
                    origin=int(origin),
                    **measurements,
                    objective_reduction=reduction,
                    fractional_objective_reduction=(
                        reduction / abs(a["final_objectives"])
                        if reduction is not None and a["final_objectives"] != 0
                        else None
                    ),
                    projected_gradient_change=gradient_change,
                    first_command_change_fraction=float(
                        np.max(
                            np.abs(
                                trial["commands"][i, 1, 0] - trial["commands"][i, 0, 0]
                            )
                            / command_range
                        )
                    ),
                )
            )

    def distribution(values):
        values = [value for value in values if value is not None]
        return dict(
            count=len(values),
            minimum=min(values) if values else None,
            median=float(np.median(values)) if values else None,
            maximum=max(values) if values else None,
        )

    def summary(selected):
        result = dict(origins=len(selected))
        for name in ARMS:
            arm = [r[name] for r in selected]
            statuses = sorted({r["statuses"] for r in arm})
            result[name] = dict(
                statuses={s: sum(r["statuses"] == s for r in arm) for s in statuses},
                failure_count=sum(not r["finite_bounded_nonfallback"] for r in arm),
                below_gradient_threshold=sum(
                    r["below_gradient_threshold"] for r in arm
                ),
                iterations_total=sum(r["iterations"] for r in arm),
                iterations_maximum=max(r["iterations"] for r in arm),
                **{
                    key: distribution([r[key] for r in arm])
                    for key in (
                        "initial_objectives",
                        "final_objectives",
                        "projected_gradient_inf_norm",
                    )
                },
            )
        for key in (
            "objective_reduction",
            "fractional_objective_reduction",
            "projected_gradient_change",
            "first_command_change_fraction",
        ):
            result[key] = distribution([r[key] for r in selected])
        return result

    pooled = summary(rows)
    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        new_policy_trials=False,
        task_success_assessed=False,
        gradient_tolerance=gradient_tolerance,
        every_candidate_origin_below_threshold=(
            pooled["candidate"]["below_gradient_threshold"] == len(rows)
        ),
        interpretation=plan["comparison"]["interpretation"],
        selection_meaning=plan["selection"]["meaning"],
        pooled=pooled,
        per_seed=[
            dict(seed=s, **summary([r for r in rows if r["seed"] == s]))
            for s in plan["selection"]["seeds"]
        ],
        origins=rows,
    )


def baseline_parity(result, diagnostics, row, command):
    fresh = serialize_pair(dict(baseline=result, candidate=result))
    np.testing.assert_array_equal(result.command, command)
    np.testing.assert_array_equal(
        fresh["commands"][0], diagnostics["candidate_commands"][row]
    )
    np.testing.assert_allclose(
        fresh["states"][0], diagnostics["forecast_states"][row], **STATE_TOLERANCE
    )
    for key in set(SCALAR_FIELDS) - {"warm_start_used", "command_bound_violation"}:
        actual, expected = fresh[key][0], diagnostics[key][row]
        if np.asarray(actual).dtype.kind in "biuU":
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, **SCORE_TOLERANCE)


def probe(plan, inputs, *, _solve_pair=None):
    parent = json.loads(inputs["manifest.json"])
    context = _context(
        {
            k.removeprefix("inputs/"): v
            for k, v in inputs.items()
            if k.startswith("inputs/")
        }
    )
    equations = CascadeEquations(context.model)
    trials = []
    for repetition, seed in enumerate(plan["selection"]["seeds"]):
        print(json.dumps(dict(repetition=repetition, seed=seed)), flush=True)
        prefix = f"trial-{repetition}/{plan['selection']['arm']}/"
        row = json.loads(inputs[prefix + "trial.json"])
        same(row["initial_state_seed"], seed)
        same(row["terminated"], False)
        tracking, diagnostics = (
            arrays(inputs[prefix + file]) for file in ("tracking.npz", "oracle.npz")
        )
        arm = TaskOracleArm(
            parent,
            context.manifest,
            context.learned,
            context.model,
            plan["selection"]["arm"],
            _equations=equations,
        )
        tracking["initial_command"] = context.initial_command
        _diagnostic_schema(arm, tracking, diagnostics)
        arm.reset(tracking["states"][0], context.initial_command)
        previous, pairs = context.initial_command, []
        for origin, command in enumerate(tracking["commands"]):
            state = tracking["states"][origin]
            arm.observe(state)
            if origin in plan["selection"]["origins"]:
                times = (origin + np.arange(HORIZON_STEPS + 1)) * context.manifest[
                    "trial"
                ]["sample_interval_s"]
                reference = ReferenceTrajectory(
                    control_reference(
                        tracking["reference_anchor_state"],
                        times,
                        context.manifest["tracking_reference"],
                    )
                )
                pair = (paired_solve if _solve_pair is None else _solve_pair)(
                    arm.plan.with_causal_state(arm._state),
                    arm.policy,
                    state,
                    reference,
                    previous,
                    warm_start=original_warm_start(diagnostics, origin),
                    baseline_check=partial(
                        baseline_parity,
                        diagnostics=diagnostics,
                        row=origin - WARMUP_INTERVALS,
                        command=command,
                    ),
                )
                pairs.append(serialize_pair(pair))
            arm.command_applied(command)
            previous = command
        np.testing.assert_allclose(
            equations.canonical(arm._state), tracking["states"][-1], **STATE_TOLERANCE
        )
        trials.append(
            dict(
                origins=np.asarray(plan["selection"]["origins"], dtype=np.int64),
                **{key: np.stack([p[key] for p in pairs]) for key in pairs[0]},
            )
        )
    same(arm.policy.gradient_tolerance, 0.002)
    return trials, report_from_inputs(plan, trials, inputs)


def report_from_inputs(plan, trials, inputs):
    telemetry = json.loads(inputs["inputs/control/manifest.json"])["telemetry"]
    command_range = (
        np.asarray(telemetry["command_maximum"]) - telemetry["command_minimum"]
    )
    return report(plan, trials, command_range, 0.002)


def artifact_names(plan):
    return {"run.json", "environment.json", "report.json"} | {
        f"trial-{i}/paired.npz" for i in range(len(plan["selection"]["seeds"]))
    }


def run(artifacts, output):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = input_snapshot(plan, artifacts)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    trials, result = probe(plan, inputs)
    write_json(output / "run.json", dict(no_fit=True, new_policy_trials=False))
    write_json(output / "environment.json", environment)
    write_json(output / "report.json", result)
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


def verify(directory):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved solver budget plan differs")
    inputs = input_snapshot(plan, directory / "inputs")
    names = artifact_names(plan)
    expected = names | {"manifest.json", "files.json"} | {"inputs/" + n for n in inputs}
    if {
        str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
    } != expected:
        raise ValueError("solver budget complete file roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    if set(hashes) != names:
        raise ValueError("solver budget artifact inventory differs")
    saved = {name: (directory / name).read_bytes() for name in names}
    for name, data in saved.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"altered solver budget artifact: {name}")
    same(json.loads(saved["environment.json"]), environment)
    same(json.loads(saved["run.json"]), dict(no_fit=True, new_policy_trials=False))
    trials = [
        arrays(saved[f"trial-{i}/paired.npz"])
        for i in range(len(plan["selection"]["seeds"]))
    ]
    same(json.loads(saved["report.json"]), report_from_inputs(plan, trials, inputs))
    fresh, result = probe(plan, inputs)
    for actual, expected in zip(trials, fresh, strict=True):
        for key, value in expected.items():
            if value.dtype.kind in "biuU" or key == "commands":
                np.testing.assert_array_equal(actual[key], value)
            else:
                np.testing.assert_allclose(
                    actual[key],
                    value,
                    **(STATE_TOLERANCE if key == "states" else SCORE_TOLERANCE),
                )
    same(json.loads(saved["report.json"]), result)
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
