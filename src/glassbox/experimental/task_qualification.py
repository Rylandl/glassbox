"""Frozen task-position-scale experiment: eight paired oracle trials, no fits.

Only the candidate's lateral and altitude position scales change. The archived
learner supplies the unchanged control contract and covariance, not forecasts.
Saved evidence replays plant trajectories, forecasts and every optimizer call.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.control.plan import ReferenceTrajectory
from glassbox.control.solver import BoundedShootingSolver

from .harness import control_reference
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
from .qualification_control import (
    BLOCK_COUNT,
    HORIZON_STEPS,
    MAXIMUM_ITERATIONS,
    WARMUP_INTERVALS,
    CascadeEquations,
    OracleArm,
)

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/controller-task-v1.json"
PLAN_SHA256 = "a825a3fe5d8875b528ad9073c6362b1b8d4d6de6a14f25edec9d533f0b3029e4"
BASELINE = "oracle_baseline"
CANDIDATE = "oracle_task_scales"
ROW_FIELDS = {
    "arm",
    "completed_intervals",
    "requested_intervals",
    "terminated",
    "failure",
    "tracking_rmse",
    "pass_criterion",
    "model_not_ready_intervals",
    "fallback_count",
    "solver_statuses",
    "maximum_command_bound_violation",
    "wall",
    "controller",
    "files",
    "repetition",
    "initial_state_seed",
    "directory",
}
ORIGINAL_DIAGNOSTICS = {
    "causal_states",
    "observed_max_abs_error",
    "solve_indices",
    "candidate_commands",
    "forecast_states",
    "final_objectives",
}
EXTRA_DIAGNOSTICS = {
    "initial_objectives",
    "projected_gradient_inf_norm",
    "iterations",
    "statuses",
    "constant_covariance_objectives",
    "variable_final_objectives",
    "used_fallback",
}


def frozen_plan():
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("task qualification plan differs from frozen source")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"task qualification inherited source changed: {name}")
    if set(plan["input_paths"]) != set(plan["input_sha256"]):
        raise ValueError("task qualification input declarations disagree")
    return plan, raw


def input_snapshot(plan, root, *, saved=False):
    """Read all pinned input bytes before any plant or optimizer is constructed."""
    result = {}
    for name, relative in plan["input_paths"].items():
        raw = (Path(root) / (name if saved else relative)).read_bytes()
        if digest(raw) != plan["input_sha256"][name]:
            raise ValueError(f"task qualification input changed: {name}")
        result[name] = raw
    return result


def check_environment(plan):
    actual = dict(
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__
    )
    expected = {key: plan["environment"][key] for key in actual}
    same(actual, expected, label="pinned numerical environment")
    if jax.config.x64_enabled:
        raise ValueError("task qualification requires default float32 precision")
    return actual


def covariance_objective(plan):
    """Measure the same command-independent spread term without changing solves."""
    variance = jnp.diagonal(plan.values.forecast_error_covariance, axis1=-2, axis2=-1)
    spread = jnp.sum(variance / jnp.square(plan.tolerances.local_state_scale), axis=1)
    return float(jnp.mean(spread) + plan.policy.terminal_weight * spread[-1])


class TaskOracleArm(OracleArm):
    """Original oracle numerical path with one frozen normalization replacement."""

    def __init__(self, plan, manifest, learned, model, name, *, _equations=None):
        if name not in (BASELINE, CANDIDATE):
            raise ValueError("undeclared task qualification arm")
        super().__init__(manifest, learned, model, _equations=_equations)
        self.name = name
        q = plan["control_qualification"]
        same(list(self.plan.tolerances.position_m), q["baseline_position_m"])
        for value, key in (
            (HORIZON_STEPS, "horizon_steps"),
            (BLOCK_COUNT, "block_count"),
            (MAXIMUM_ITERATIONS, "maximum_iterations"),
            (WARMUP_INTERVALS, "warmup_intervals"),
        ):
            same(value, q[key])
        if name == CANDIDATE:
            scales = q["candidate_position_m"]
            signature = digest(
                (
                    self.plan.base.compile_signature
                    + ":task-position-scales:"
                    + json.dumps(scales)
                ).encode()
            )
            base = replace(
                self.plan.base,
                tolerances=replace(self.plan.tolerances, position_m=scales),
                compile_signature=signature,
            )
            self.plan = replace(
                self.plan,
                base=base,
                compile_signature=digest(
                    (self.plan.compile_signature + ":" + signature).encode()
                ),
            )
        self.constant_objective = covariance_objective(self.plan)

    def reset(self, initial_state, initial_command):
        super().reset(initial_state, initial_command)
        self._extras = {key: [] for key in EXTRA_DIAGNOSTICS}

    def solve(self, state, reference, previous_command, **keywords):
        if not self.ready or self._applied != self._observed - 1:
            raise ValueError("an oracle solve requires a ready, observed origin")
        plan = self.plan.with_causal_state(self._state)
        result = BoundedShootingSolver(plan, self.policy).solve(
            state, reference, previous_command, **keywords
        )
        self._indices.append(self._applied)
        self._commands.append(np.asarray(result.predicted_commands))
        # Failed solves retain their issued hold and an explicitly absent forecast.
        # A zero sentinel is never scored or interpreted as a successful forecast.
        self._forecasts.append(
            np.zeros((HORIZON_STEPS + 1, 13))
            if result.used_fallback
            else np.asarray(result.predicted_states)
        )
        d = result.diagnostics
        self._objectives.append(d.final_objective)
        measured = dict(
            initial_objectives=d.initial_objective,
            projected_gradient_inf_norm=d.final_projected_gradient_inf_norm,
            iterations=d.iterations,
            statuses=str(result.status),
            constant_covariance_objectives=self.constant_objective,
            variable_final_objectives=d.final_objective - self.constant_objective,
            used_fallback=result.used_fallback,
        )
        for key, value in measured.items():
            self._extras[key].append(value)
        return result

    def summary(self):
        return dict(
            super().summary(),
            position_scales_m=list(self.plan.tolerances.position_m),
            constant_covariance_objective=self.constant_objective,
        )

    def diagnostic_arrays(self):
        dtypes = {"statuses": "U32", "iterations": np.int64, "used_fallback": bool}
        return dict(
            super().diagnostic_arrays(),
            **{
                key: np.asarray(value, dtype=dtypes.get(key, float))
                for key, value in self._extras.items()
            },
        )


def artifact_names(plan):
    names = {"run.json", "environment.json", "results.json", "report.json"}
    for index, order in enumerate(plan["control_qualification"]["arm_order"]):
        for name in order:
            names |= {
                f"trial-{index}/{name}/{file}"
                for file in ("trial.json", "tracking.npz", "timing.npz", "oracle.npz")
            }
    return names


def validate_rows(plan, rows):
    q = plan["control_qualification"]
    expected = [(i, name) for i, order in enumerate(q["arm_order"]) for name in order]
    if (
        not isinstance(rows, list)
        or [(r.get("repetition"), r.get("arm")) for r in rows] != expected
    ):
        raise ValueError("task qualification trial roster differs")
    for row, (repetition, name) in zip(rows, expected, strict=True):
        if set(row) != ROW_FIELDS:
            raise ValueError("task qualification trial summary fields differ")
        same(row["repetition"], repetition)
        same(row["arm"], name)
        same(row["initial_state_seed"], q["seeds"][row["repetition"]])
        same(row["directory"], f"trial-{row['repetition']}/{row['arm']}")


def historical_parity(plan, inputs, rows, saved):
    """Compare historical evidence only with its corresponding baseline trials."""
    q = plan["control_qualification"]
    tolerance = q["replay"]
    state_tolerance = dict(rtol=tolerance["state_rtol"], atol=tolerance["state_atol"])
    score_tolerance = dict(rtol=tolerance["score_rtol"], atol=tolerance["score_atol"])
    checks = []
    for repetition in q["historical_baseline"]["repetitions"]:
        row = next(
            r for r in rows if r["repetition"] == repetition and r["arm"] == BASELINE
        )
        prefix = q["historical_baseline"]["path_template"].format(repetition=repetition)
        old = json.loads(inputs[prefix + "/trial.json"])
        for key in (
            "completed_intervals",
            "requested_intervals",
            "terminated",
            "failure",
            "tracking_rmse",
            "pass_criterion",
            "model_not_ready_intervals",
            "fallback_count",
            "solver_statuses",
            "maximum_command_bound_violation",
        ):
            same(
                row[key],
                old[key],
                label=f"historical {repetition}.{key}",
                **score_tolerance,
            )
        differences = {}
        for file in ("tracking.npz", "oracle.npz"):
            historical = arrays(inputs[prefix + "/" + file])
            fresh = arrays(saved[row["directory"] + "/" + file])
            if file == "tracking.npz" and set(historical) != set(fresh):
                raise ValueError("historical tracking field set differs")
            if file == "oracle.npz" and set(historical) != ORIGINAL_DIAGNOSTICS:
                raise ValueError("historical oracle field set differs")
            maximum = 0.0
            for key, value in historical.items():
                current = fresh[key]
                tol = (
                    score_tolerance
                    if key in {"final_objectives", "observed_max_abs_error"}
                    else state_tolerance
                )
                if value.dtype.kind in "biu" or key in {
                    "commands",
                    "candidate_commands",
                }:
                    np.testing.assert_array_equal(current, value)
                else:
                    np.testing.assert_allclose(current, value, **tol)
                    maximum = max(
                        maximum, float(np.max(np.abs(current - value), initial=0.0))
                    )
            differences[file] = maximum
        checks.append(
            dict(
                repetition=repetition,
                initial_state_seed=row["initial_state_seed"],
                maximum_absolute_differences=differences,
            )
        )
    return checks


def report(plan, rows, parity):
    validate_rows(plan, rows)
    candidates = [row for row in rows if row["arm"] == CANDIDATE]
    qualified = all(
        row["pass_criterion"]["met"]
        and not row["terminated"]
        and row["fallback_count"] == 0
        and row["maximum_command_bound_violation"] == 0
        for row in candidates
    )
    pairs = []
    for repetition, seed in enumerate(plan["control_qualification"]["seeds"]):
        baseline, candidate = [
            next(r for r in rows if r["repetition"] == repetition and r["arm"] == arm)
            for arm in (BASELINE, CANDIDATE)
        ]
        deltas = {}
        for key in ("within_tolerance_fraction", "lateral_rmse_m", "altitude_rmse_m"):
            a, b = baseline["pass_criterion"][key], candidate["pass_criterion"][key]
            deltas[key] = None if a is None or b is None else b - a
        pairs.append(dict(initial_state_seed=seed, candidate_minus_baseline=deltas))
    keys = (
        "arm",
        "repetition",
        "initial_state_seed",
        "terminated",
        "completed_intervals",
        "tracking_rmse",
        "pass_criterion",
        "fallback_count",
        "solver_statuses",
        "maximum_command_bound_violation",
        "controller",
    )
    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        candidate_qualifies=bool(qualified),
        learner_or_historical_acceptance_changed=False,
        qualification_rule=plan["control_qualification"]["qualification_rule"],
        historical_baseline=parity,
        paired_differences=pairs,
        trials=[{key: row[key] for key in keys} for row in rows],
    )


def _diagnostic_schema(arm, tracking, diagnostics):
    if set(diagnostics) != ORIGINAL_DIAGNOSTICS | EXTRA_DIAGNOSTICS:
        raise ValueError("task oracle diagnostic fields differ")
    completed = len(tracking["commands"])
    attempted = len(tracking.get("solver_used", tracking["commands"]))
    if attempted not in {completed, completed + 1}:
        raise ValueError("task oracle attempted interval count differs")
    count = max(0, attempted - WARMUP_INTERVALS)
    shapes = dict(
        causal_states=np.shape(tracking["states"]),
        observed_max_abs_error=(attempted,),
        solve_indices=(count,),
        candidate_commands=(count, HORIZON_STEPS, arm.plan.command_size),
        forecast_states=(count, HORIZON_STEPS + 1, 13),
        final_objectives=(count,),
    )
    shapes |= {key: (count,) for key in EXTRA_DIAGNOSTICS}
    for key, shape in shapes.items():
        if np.shape(diagnostics[key]) != shape:
            raise ValueError(f"task oracle diagnostic shape differs: {key}")
    for key in ("iterations", "solve_indices"):
        if np.asarray(diagnostics[key]).dtype.kind not in "iu":
            raise ValueError(f"task oracle diagnostic must be integer: {key}")
    if (
        np.asarray(diagnostics["used_fallback"]).dtype != bool
        or np.asarray(diagnostics["statuses"]).dtype.kind != "U"
    ):
        raise ValueError("task oracle diagnostic flag/status dtype differs")
    np.testing.assert_array_equal(
        diagnostics["solve_indices"], np.arange(WARMUP_INTERVALS, attempted)
    )
    for key in (ORIGINAL_DIAGNOSTICS | EXTRA_DIAGNOSTICS) - {"statuses"}:
        value = np.asarray(diagnostics[key])
        if np.isnan(value).any():
            raise ValueError(f"task oracle diagnostic contains NaN: {key}")
        if (
            key
            not in {
                "initial_objectives",
                "final_objectives",
                "variable_final_objectives",
                "projected_gradient_inf_norm",
            }
            and not np.isfinite(value).all()
        ):
            raise ValueError(f"task oracle diagnostic is nonfinite: {key}")
    return count


def verify_solves(plan, arm, manifest, tracking, diagnostics):
    """Replay actual states and warm starts; recorded commands are never optimized."""
    count = _diagnostic_schema(arm, tracking, diagnostics)
    tolerance = plan["control_qualification"]["replay"]
    state_tolerance = dict(rtol=tolerance["state_rtol"], atol=tolerance["state_atol"])
    score_tolerance = dict(rtol=tolerance["score_rtol"], atol=tolerance["score_atol"])
    states, commands = tracking["states"], tracking["commands"]
    previous = tracking["initial_command"]
    arm.reset(states[0], previous)
    warm_start = None
    maximum_forecast = 0.0
    attempted = len(tracking.get("solver_used", commands))
    for index in range(attempted):
        arm.observe(states[index])
        command = commands[index] if index < len(commands) else previous
        if arm.ready:
            row = index - WARMUP_INTERVALS
            candidate = diagnostics["candidate_commands"][row]
            if np.any(candidate < arm.plan.command_minimum) or np.any(
                candidate > arm.plan.command_maximum
            ):
                raise ValueError("saved oracle candidate commands exceed bounds")
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
            if index == len(commands):
                command = candidate[0]
            np.testing.assert_array_equal(result.command, command)
            np.testing.assert_array_equal(result.predicted_commands, candidate)
            if not result.used_fallback:
                local = arm.plan.with_causal_state(arm._state)
                predicted = local.rollout_commands(
                    jnp.asarray(candidate),
                    jnp.asarray(states[index]),
                    jnp.zeros(0),
                    jnp.zeros((HORIZON_STEPS, 0)),
                    local.values,
                )
                forecast = np.asarray(predicted.mean_states)
                np.testing.assert_allclose(
                    forecast, diagnostics["forecast_states"][row], **state_tolerance
                )
                maximum_forecast = max(
                    maximum_forecast,
                    float(
                        np.max(np.abs(forecast - diagnostics["forecast_states"][row]))
                    ),
                )
        else:
            np.testing.assert_array_equal(command, previous)
        if index < len(commands):
            arm.command_applied(command)
            previous = command
    np.testing.assert_allclose(
        arm.equations.canonical(arm._state), states[-1], **state_tolerance
    )
    fresh = arm.diagnostic_arrays()
    for key, value in fresh.items():
        expected = diagnostics[key]
        if value.dtype.kind in "biuU" or key == "candidate_commands":
            np.testing.assert_array_equal(value, expected)
        else:
            tol = (
                score_tolerance
                if key
                in EXTRA_DIAGNOSTICS | {"final_objectives", "observed_max_abs_error"}
                else state_tolerance
            )
            np.testing.assert_allclose(value, expected, **tol)
    statuses = {"model_not_ready": min(attempted, WARMUP_INTERVALS)}
    for status in fresh["statuses"]:
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    return dict(
        verified_optimizer_solves=count,
        verified_oracle_forecasts=int(np.count_nonzero(~fresh["used_fallback"])),
        maximum_forecast_difference=maximum_forecast,
        solver_statuses=statuses,
    )


def replay_failed_prefix(plan, context, row, saved, diagnostics):
    """Verify a finite prefix and reproduce its first nonfinite plant interval.

    The inherited loop saves an attempted solve before discovering a nonfinite
    next state. Its command/forecast therefore lives in the oracle diagnostics,
    while tracking and tick durations describe completed intervals only.
    """
    from glassbox.core.metrics import state_rmse_metrics

    from .harness import (
        SIMULATED_TIME_MEANING,
        _control_tracking_plant,
        control_initial_state,
        control_pass_criterion,
        simulated_time_wall,
    )

    manifest = context.manifest
    tracking = arrays(saved[row["directory"] + "/tracking.npz"])
    if set(tracking) != {
        "time_s",
        "states",
        "reference_states",
        "commands",
        "solver_used",
        "used_fallback",
        "initial_state",
        "reference_anchor_state",
    }:
        raise ValueError("failed task tracking fields differ")
    states, commands = tracking["states"], tracking["commands"]
    completed = len(commands)
    requested = manifest["trial"]["intervals"]
    attempted = completed + 1
    if (
        commands.shape != (completed, 3)
        or states.shape != (completed + 1, 13)
        or not 0 <= completed < requested
    ):
        raise ValueError("failed task trajectory shape or prefix length differs")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("failed task prefix must contain finite measurements")
    same(row["completed_intervals"], completed)
    same(row["requested_intervals"], requested)
    same(row["terminated"], True)
    same(row["failure"], "nonfinite plant state")
    with jax.enable_x64(True):
        initial = control_initial_state(
            manifest, context.initial_state, row["initial_state_seed"]
        )
    for value in (tracking["initial_state"], states[0]):
        np.testing.assert_array_equal(value, initial)
    np.testing.assert_array_equal(
        tracking["reference_anchor_state"], context.initial_state
    )
    minimum, maximum = (
        np.asarray(manifest["telemetry"][key])
        for key in ("command_minimum", "command_maximum")
    )
    if np.any(commands < minimum) or np.any(commands > maximum):
        raise ValueError("failed task command prefix exceeds bounds")
    same(row["maximum_command_bound_violation"], 0.0)
    held = min(completed, WARMUP_INTERVALS)
    np.testing.assert_array_equal(
        commands[:held], np.tile(context.initial_command, (held, 1))
    )
    times = np.arange(completed + 1) * manifest["trial"]["sample_interval_s"]
    np.testing.assert_array_equal(tracking["time_s"], times)
    reference = control_reference(
        context.initial_state, times, manifest["tracking_reference"]
    )
    np.testing.assert_array_equal(tracking["reference_states"], reference)
    same(
        row["tracking_rmse"],
        state_rmse_metrics(states[1:], reference[1:]) if completed else None,
    )
    same(
        row["pass_criterion"],
        control_pass_criterion(states, context.initial_state, manifest),
    )
    same(row["pass_criterion"]["met"], False)
    for key in ("solver_used", "used_fallback"):
        value = tracking[key]
        if value.shape != (attempted,) or value.dtype != bool:
            raise ValueError("failed task attempted solve flags differ")
    np.testing.assert_array_equal(
        tracking["solver_used"], np.arange(attempted) >= WARMUP_INTERVALS
    )
    warmup = min(attempted, WARMUP_INTERVALS)
    np.testing.assert_array_equal(
        tracking["used_fallback"][:warmup], np.zeros(warmup, dtype=bool)
    )
    same(row["model_not_ready_intervals"], warmup)
    same(row["fallback_count"], int(np.count_nonzero(tracking["used_fallback"])))
    plant = _control_tracking_plant(manifest, initial, context.initial_command)
    replayed = np.vstack([initial, *[plant.advance(command) for command in commands]])
    tolerance = plan["control_qualification"]["replay"]
    np.testing.assert_allclose(
        replayed, states, rtol=tolerance["state_rtol"], atol=tolerance["state_atol"]
    )
    failed_command = (
        context.initial_command
        if completed < WARMUP_INTERVALS
        else diagnostics["candidate_commands"][-1, 0]
    )
    if np.any(failed_command < minimum) or np.any(failed_command > maximum):
        raise ValueError("failed interval command exceeds bounds")
    if np.isfinite(np.asarray(plant.advance(failed_command))).all():
        raise ValueError("recorded task termination does not reproduce")
    same(
        row["files"],
        {
            name: digest(saved[row["directory"] + "/" + name])
            for name in ("tracking.npz", "timing.npz")
        },
    )
    timing = arrays(saved[row["directory"] + "/timing.npz"])
    if set(timing) != {"tick_times_s", "solve_times_s", "deadline_assessed"}:
        raise ValueError("failed task timing fields differ")
    for key, count in (("tick_times_s", completed), ("solve_times_s", attempted)):
        value = timing[key]
        if value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("failed task timing shape or values differ")
    if timing["deadline_assessed"].dtype != bool:
        raise ValueError("failed task deadline flags must be boolean")
    np.testing.assert_array_equal(
        timing["deadline_assessed"], np.zeros(attempted, dtype=bool)
    )
    np.testing.assert_array_equal(timing["solve_times_s"][:warmup], np.zeros(warmup))
    elapsed = row["wall"]["elapsed_seconds"]
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("failed task elapsed time is invalid")
    same(
        row["wall"],
        simulated_time_wall(
            meaning=SIMULATED_TIME_MEANING,
            dt_s=manifest["trial"]["sample_interval_s"],
            deadline_s=manifest["trial"]["solve_deadline_s"],
            tick_times=timing["tick_times_s"].tolist(),
            solve_times=timing["solve_times_s"].tolist(),
            elapsed_s=elapsed,
            solve_deadline_applied=False,
            deadline_assessed_intervals=0,
        ),
    )
    return dict(state_difference=float(np.max(np.abs(replayed - states))), oracle=None)


def run(artifacts, output):
    from .harness import _control_tracking_plant, _control_trial, control_initial_state

    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = input_snapshot(plan, artifacts)
    context = _context(inputs)
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    write_json(output / "run.json", dict(no_fit=True, prospective=True))
    write_json(output / "environment.json", environment)
    equations = CascadeEquations(context.model)
    q = plan["control_qualification"]

    def reference_fn(times):
        return control_reference(
            context.initial_state, times, context.manifest["tracking_reference"]
        )

    rows = []
    for repetition, order in enumerate(q["arm_order"]):
        seed = q["seeds"][repetition]
        with jax.enable_x64(True):
            start = control_initial_state(context.manifest, context.initial_state, seed)
        for name in order:
            print(
                json.dumps(dict(tracking=f"{repetition}-{name}", seed=seed)), flush=True
            )
            arm = TaskOracleArm(
                plan,
                context.manifest,
                context.learned,
                context.model,
                name,
                _equations=equations,
            )
            _prewarm(arm, context, reference_fn)
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
            write_json(directory / "trial.json", row)
            rows.append(row)
            write_json(output / "results.json", rows)
            print(
                json.dumps(
                    dict(
                        trial=f"{repetition}-{name}",
                        pass_criterion=row["pass_criterion"],
                        tracking_rmse=row["tracking_rmse"],
                    )
                ),
                flush=True,
            )
    saved = {
        name: (output / name).read_bytes()
        for name in artifact_names(plan) - {"report.json"}
    }
    result = report(plan, rows, historical_parity(plan, inputs, rows, saved))
    write_json(output / "report.json", result)
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
        raise ValueError("saved task qualification plan differs")
    inputs = input_snapshot(plan, directory / "inputs", saved=True)
    expected_names = artifact_names(plan)
    disk_names = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    if disk_names != expected_names | {"manifest.json", "files.json"} | {
        "inputs/" + key for key in inputs
    }:
        raise ValueError("task qualification complete file roster differs")
    hashes = json.loads((directory / "files.json").read_text())
    if set(hashes) != expected_names:
        raise ValueError("task qualification artifact inventory differs")
    saved = {name: (directory / name).read_bytes() for name in expected_names}
    for name, data in saved.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"altered task qualification artifact: {name}")
    same(json.loads(saved["run.json"]), dict(no_fit=True, prospective=True))
    same(json.loads(saved["environment.json"]), environment)
    rows = json.loads(saved["results.json"])
    validate_rows(plan, rows)
    result = report(plan, rows, historical_parity(plan, inputs, rows, saved))
    same(json.loads(saved["report.json"]), result)
    context = _context(inputs)
    equations = CascadeEquations(context.model)
    checks = []
    for row in rows:
        same(row, json.loads(saved[row["directory"] + "/trial.json"]))
        arm = TaskOracleArm(
            plan,
            context.manifest,
            context.learned,
            context.model,
            row["arm"],
            _equations=equations,
        )
        same(row["controller"], arm.summary())
        tracking = arrays(saved[row["directory"] + "/tracking.npz"])
        tracking["initial_command"] = context.initial_command
        diagnostics = arrays(saved[row["directory"] + "/oracle.npz"])
        _diagnostic_schema(arm, tracking, diagnostics)
        physical = (
            replay_failed_prefix(plan, context, row, saved, diagnostics)
            if row["terminated"]
            else _replay_trial(plan, context, row, saved)
        )
        replay = verify_solves(plan, arm, context.manifest, tracking, diagnostics)
        same(replay["solver_statuses"], row["solver_statuses"])
        np.testing.assert_array_equal(
            tracking["used_fallback"][WARMUP_INTERVALS:], diagnostics["used_fallback"]
        )
        checks.append(dict(physical=physical, optimizer=replay))
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
        maximum_forecast_replay_difference=max(
            c["optimizer"]["maximum_forecast_difference"] for c in checks
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
