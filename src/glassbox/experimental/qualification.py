"""Qualify frozen benchmark interpretations and the current control seam, without fits.

Run from a source checkout. ``audit`` reads the pinned existing evidence only;
``run`` also executes the four prospective controller trials. ``verify`` checks
saved inputs, rederives the report, and replays new trajectories and forecasts.
These commands do not change any historical learner acceptance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/evaluation-qualification-v1.json"
PLAN_SHA256 = "8be0ca1c5e68f276bc34a0161e23e8fe2761c785d3afb3c435fd161de7a755b0"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def arrays(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def frozen_plan():
    """Hash and parse the same bytes; check every inherited source contract."""
    from .api_migration import qualification_source_matches

    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("qualification plan differs from the frozen source")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest(
            (ROOT / name).read_bytes()
        ) != expected and not qualification_source_matches(ROOT, name, expected):
            raise ValueError(f"qualification source changed: {name}")
    return plan, raw


def input_snapshot(plan, roots):
    """Load every pinned input once and validate all before running any trial."""
    result = {}
    for name, expected in plan["input_sha256"].items():
        role, relative = name.split("/", 1)
        data = (Path(roots[role]) / relative).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"qualification input changed: {name}")
        result[name] = data
    return result


def same(actual, expected, *, rtol=1e-7, atol=1e-9, label="report"):
    """Compare complete report structure and finite numbers, including booleans."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}: fields differ")
        for name, value in expected.items():
            same(actual[name], value, rtol=rtol, atol=atol, label=f"{label}.{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: list differs")
        for index, value in enumerate(expected):
            same(actual[index], value, rtol=rtol, atol=atol, label=f"{label}[{index}]")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{label}: expected a number")
        if not np.isfinite(actual) or not np.isclose(
            actual, expected, rtol=rtol, atol=atol
        ):
            raise ValueError(f"{label}: numerical mismatch")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: value differs")


def _context(inputs):
    import jax

    from glassbox.belief.belief_io import dynamics_belief_from_payload

    from ..learner import LearnedDynamics
    from .harness import control_fixture

    if jax.config.x64_enabled:
        raise ValueError(
            "qualification trials require the pinned default float32 precision"
        )
    manifest = json.loads(inputs["control/manifest.json"])
    calibration = json.loads(inputs["control/calibration.json"])
    with jax.enable_x64(True):
        learned = LearnedDynamics.load(io.BytesIO(inputs["control/generic.npz"]))
        belief = dynamics_belief_from_payload(
            json.loads(inputs["control/structured.json"])
        )
        spec, _, _, state, command = control_fixture(manifest)
    # Match the actual tracking plant's construction, not a cast of the x64
    # trim model. Inverting inertia in float64 then casting changes coefficients
    # relative to constructing the public plant in float32.
    model = spec.to_model()
    if learned.fingerprint() != calibration["generic_fingerprint"]:
        raise ValueError("qualification generic fingerprint differs")
    np.testing.assert_allclose(
        state, calibration["initial_state"], rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        command, calibration["initial_command"], rtol=1e-8, atol=1e-9
    )
    return SimpleNamespace(
        manifest=manifest,
        learned=learned,
        belief=belief,
        model=model,
        initial_state=np.asarray(calibration["initial_state"]),
        initial_command=np.asarray(calibration["initial_command"]),
    )


def _report(plan, inputs, rows=None):
    from .qualification_references import reference_report

    report = dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        reference_audit=reference_report(plan, inputs),
    )
    if rows is not None:
        oracle = [r for r in rows if r["arm"] == "oracle_generic_seam"]
        report["control_qualification"] = dict(
            oracle_meets_application_criterion=(
                len(oracle) == len(plan["control_qualification"]["seeds"])
                and all(
                    r["pass_criterion"]["met"] and not r["terminated"] for r in oracle
                )
            ),
            rule=plan["control_qualification"]["qualification_rule"],
            historical_model_acceptance_changed=False,
            trials=[
                {
                    key: r[key]
                    for key in (
                        "arm",
                        "repetition",
                        "initial_state_seed",
                        "terminated",
                        "completed_intervals",
                        "tracking_rmse",
                        "pass_criterion",
                        "model_not_ready_intervals",
                        "fallback_count",
                        "controller",
                    )
                }
                for r in rows
            ],
            limits=plan["control_qualification"]["limits"],
        )
    return report


def _write_snapshot(output, plan_raw, inputs):
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(plan_raw)
    for name, data in inputs.items():
        target = output / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _artifact_names(plan, prospective):
    names = {"report.json", "run.json"}
    if prospective:
        names.add("results.json")
        for index, order in enumerate(plan["control_qualification"]["arm_order"]):
            for arm in order:
                prefix = f"trial-{index}/{arm}/"
                names.update(
                    prefix + name
                    for name in ("trial.json", "tracking.npz", "timing.npz")
                )
                if arm == "oracle_generic_seam":
                    names.add(prefix + "oracle.npz")
    return names


def _prewarm(arm, context, reference_fn):
    """Compile on discarded real transitions from the declared equilibrium.

    A calibration recording's first command is not its initial actuator state.
    Give the causal oracle an actual reset and issued-command history instead.
    This preparation changes neither a scored initial condition nor a command.
    """
    from glassbox.control.plan import ReferenceTrajectory

    from .harness import _control_tracking_plant

    plant = _control_tracking_plant(
        context.manifest, context.initial_state, context.initial_command
    )
    state, previous = context.initial_state, context.initial_command
    arm.reset(state, previous)
    for index in range(3):
        arm.observe(state)
        if arm.ready:
            times = (index + np.arange(arm.prediction_steps + 1)) * 0.05
            result = arm.solve(
                state, ReferenceTrajectory(reference_fn(times)), previous
            )
            np.asarray(result.command)
        state = np.asarray(plant.advance(previous))
        arm.command_applied(previous)


def run(artifacts, output, *, prospective):
    plan, raw = frozen_plan()
    inputs = input_snapshot(
        plan,
        {role: Path(artifacts) / name for role, name in plan["baseline_runs"].items()},
    )
    output = Path(output)
    _write_snapshot(output, raw, inputs)
    write_json(output / "run.json", dict(prospective=prospective, no_fit=True))
    rows = None
    if prospective:
        import jax

        from .harness import (
            _control_tracking_plant,
            _control_trial,
            control_initial_state,
            control_reference,
        )
        from .qualification_control import build_arms

        context = _context(inputs)
        manifest = context.manifest
        qualification = plan["control_qualification"]

        def reference_fn(times):
            return control_reference(
                context.initial_state, times, manifest["tracking_reference"]
            )

        rows = []
        for repetition, order in enumerate(qualification["arm_order"]):
            seed = qualification["seeds"][repetition]
            with jax.enable_x64(True):
                start = control_initial_state(manifest, context.initial_state, seed)
            for name in order:
                print(json.dumps(dict(tracking=f"{repetition}-{name}")), flush=True)
                arm = build_arms(
                    manifest, context.learned, context.belief, context.model
                )[name]
                _prewarm(arm, context, reference_fn)
                plant = _control_tracking_plant(
                    manifest, start, context.initial_command
                )
                directory = output / f"trial-{repetition}" / name
                row = _control_trial(
                    manifest, arm, plant, reference_fn, context.initial_state, directory
                )
                row.update(
                    repetition=repetition,
                    initial_state_seed=seed,
                    directory=str(directory.relative_to(output)),
                )
                if name == "oracle_generic_seam":
                    np.savez_compressed(
                        directory / "oracle.npz", **arm.diagnostic_arrays()
                    )
                write_json(directory / "trial.json", row)
                rows.append(row)
                write_json(output / "results.json", rows)
                print(
                    json.dumps(
                        dict(
                            trial=f"{repetition}-{name}",
                            tracking_rmse=row["tracking_rmse"],
                            pass_criterion=row["pass_criterion"],
                        )
                    ),
                    flush=True,
                )
        write_json(
            output / "environment.json",
            dict(
                python=platform.python_version(),
                jax=jax.__version__,
                numpy=np.__version__,
            ),
        )
    report = _report(plan, inputs, rows)
    write_json(output / "report.json", report)
    write_json(
        output / "files.json",
        {
            name: digest((output / name).read_bytes())
            for name in sorted(_artifact_names(plan, prospective))
        },
    )
    return report


def _replay_trial(plan, context, row, saved):
    import jax

    from glassbox.control.plan import SolveStatus
    from glassbox.core.metrics import state_rmse_metrics

    from .harness import (
        SIMULATED_TIME_MEANING,
        _control_tracking_plant,
        control_initial_state,
        control_pass_criterion,
        control_reference,
        simulated_time_wall,
    )

    manifest = context.manifest
    tolerance = plan["control_qualification"]["replay"]
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
        raise ValueError("qualification tracking fields differ")
    tracking["initial_command"] = context.initial_command
    states, commands = tracking["states"], tracking["commands"]
    requested = manifest["trial"]["intervals"]
    if states.shape != (len(commands) + 1, 13) or commands.shape != (requested, 3):
        raise ValueError("qualification trajectory shape or completion differs")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("qualification trajectory is nonfinite")
    minimum, maximum = (
        np.asarray(manifest["telemetry"][key])
        for key in ("command_minimum", "command_maximum")
    )
    if np.any(commands < minimum) or np.any(commands > maximum):
        raise ValueError("qualification command outside frozen bounds")
    with jax.enable_x64(True):
        initial = control_initial_state(
            manifest, context.initial_state, row["initial_state_seed"]
        )
    np.testing.assert_array_equal(tracking["initial_state"], initial)
    np.testing.assert_array_equal(states[0], initial)
    np.testing.assert_array_equal(
        tracking["reference_anchor_state"], context.initial_state
    )
    plant = _control_tracking_plant(manifest, initial, context.initial_command)
    replayed = np.vstack([initial, *[plant.advance(command) for command in commands]])
    np.testing.assert_allclose(
        replayed, states, rtol=tolerance["state_rtol"], atol=tolerance["state_atol"]
    )
    times = np.arange(requested + 1) * manifest["trial"]["sample_interval_s"]
    np.testing.assert_array_equal(tracking["time_s"], times)
    reference = control_reference(
        context.initial_state, times, manifest["tracking_reference"]
    )
    np.testing.assert_array_equal(tracking["reference_states"], reference)
    same(row["tracking_rmse"], state_rmse_metrics(states[1:], reference[1:]))
    same(
        row["pass_criterion"],
        control_pass_criterion(states, context.initial_state, manifest),
    )
    same(row["completed_intervals"], requested)
    same(row["requested_intervals"], requested)
    same(row["terminated"], False)
    same(row["failure"], None)
    used = np.arange(requested) >= plan["control_qualification"]["warmup_intervals"]
    np.testing.assert_array_equal(tracking["solver_used"], used)
    if tracking["solver_used"].dtype != bool:
        raise ValueError("qualification solver flags must be boolean")
    np.testing.assert_array_equal(
        commands[:2], np.tile(context.initial_command, (2, 1))
    )
    same(row["model_not_ready_intervals"], 2)
    same(row["fallback_count"], int(np.count_nonzero(tracking["used_fallback"])))
    if (
        tracking["used_fallback"].shape != (requested,)
        or tracking["used_fallback"].dtype != bool
    ):
        raise ValueError("qualification fallback flags differ")
    same(row["maximum_command_bound_violation"], 0.0)
    same(
        row["files"],
        {
            name: digest(saved[row["directory"] + "/" + name])
            for name in ("tracking.npz", "timing.npz")
        },
    )
    timing = arrays(saved[row["directory"] + "/timing.npz"])
    if set(timing) != {"tick_times_s", "solve_times_s", "deadline_assessed"}:
        raise ValueError("qualification timing fields differ")
    for name in ("tick_times_s", "solve_times_s"):
        values = timing[name]
        if (
            values.shape != (requested,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("qualification timing shape or values differ")
    np.testing.assert_array_equal(
        timing["deadline_assessed"], np.zeros(requested, dtype=bool)
    )
    if timing["deadline_assessed"].dtype != bool:
        raise ValueError("qualification deadline flags must be boolean")
    np.testing.assert_array_equal(timing["solve_times_s"][:2], np.zeros(2))
    np.testing.assert_array_equal(
        tracking["used_fallback"][:2], np.zeros(2, dtype=bool)
    )
    elapsed = row["wall"]["elapsed_seconds"]
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid qualification elapsed time")
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
    statuses = row["solver_statuses"]
    permitted = {str(status) for status in SolveStatus} | {"model_not_ready"}
    usable = {"converged", "iteration_limit", "stalled", "model_not_ready"}
    if (
        not isinstance(statuses, dict)
        or not set(statuses) <= permitted
        or any(type(count) is not int or count < 1 for count in statuses.values())
        or sum(statuses.values()) != requested
        or statuses.get("model_not_ready") != 2
        or statuses.get("deadline_exceeded", 0)
    ):
        raise ValueError("qualification solver status counts differ")
    same(
        sum(count for name, count in statuses.items() if name not in usable),
        row["fallback_count"],
    )
    oracle_check = None
    if row["arm"] == "oracle_generic_seam":
        from .qualification_control import verify_oracle_diagnostics

        same(row["fallback_count"], 0)
        oracle_check = verify_oracle_diagnostics(
            manifest,
            context.learned,
            context.model,
            tracking,
            arrays(saved[row["directory"] + "/oracle.npz"]),
            tolerance,
        )
        same(oracle_check["replayed_solver_statuses"], row["solver_statuses"])
        same(oracle_check["replayed_solver_calls"], requested - 2)
    return dict(
        state_difference=float(np.max(np.abs(replayed - states))), oracle=oracle_check
    )


def verify(directory):
    plan, raw = frozen_plan()
    directory = Path(directory)
    if (directory / "manifest.json").read_bytes() != raw:
        raise ValueError("saved qualification plan differs from frozen source")
    inputs = input_snapshot(
        plan, {role: directory / "inputs" / role for role in plan["baseline_runs"]}
    )
    run_raw = (directory / "run.json").read_bytes()
    run_info = json.loads(run_raw)
    if (
        set(run_info) != {"prospective", "no_fit"}
        or run_info["no_fit"] is not True
        or type(run_info["prospective"]) is not bool
    ):
        raise ValueError("invalid qualification run type")
    names = _artifact_names(plan, run_info["prospective"])
    hashes = json.loads((directory / "files.json").read_bytes())
    if set(hashes) != names:
        raise ValueError("qualification artifact inventory differs")
    saved = {
        name: (run_raw if name == "run.json" else (directory / name).read_bytes())
        for name in names
    }
    for name, data in saved.items():
        if digest(data) != hashes[name]:
            raise ValueError(f"altered qualification artifact: {name}")
    rows, errors = None, []
    if run_info["prospective"]:
        from .qualification_control import build_arms

        rows = json.loads(saved["results.json"])
        q = plan["control_qualification"]
        expected = [(i, arm) for i, order in enumerate(q["arm_order"]) for arm in order]
        if [(row["repetition"], row["arm"]) for row in rows] != expected:
            raise ValueError("qualification trial roster differs")
        context = _context(inputs)
        arms = build_arms(
            context.manifest, context.learned, context.belief, context.model
        )
        for row in rows:
            if set(row) != {
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
            }:
                raise ValueError("qualification trial summary fields differ")
            same(row["initial_state_seed"], q["seeds"][row["repetition"]])
            same(row["directory"], f"trial-{row['repetition']}/{row['arm']}")
            same(row, json.loads(saved[row["directory"] + "/trial.json"]))
            same(row["controller"], arms[row["arm"]].summary())
            errors.append(_replay_trial(plan, context, row, saved))
    report = _report(plan, inputs, rows)
    same(json.loads(saved["report.json"]), report)
    return dict(
        verified=True,
        no_fit=True,
        verified_trials=len(errors),
        maximum_state_replay_difference=max(
            (error["state_difference"] for error in errors), default=0.0
        ),
        verified_oracle_forecasts=sum(
            error["oracle"]["checked_forecasts"]
            for error in errors
            if error["oracle"] is not None
        ),
        verified_optimizer_solves=sum(
            error["oracle"]["replayed_solver_calls"]
            for error in errors
            if error["oracle"] is not None
        ),
        maximum_objective_replay_difference=max(
            (
                error["oracle"]["maximum_objective_difference"]
                for error in errors
                if error["oracle"] is not None
            ),
            default=0.0,
        ),
        report=report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "run"):
        command = commands.add_parser(name)
        command.add_argument("--artifacts", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.directory)
    else:
        result = run(args.artifacts, args.output, prospective=args.command == "run")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
