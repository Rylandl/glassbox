"""Qualify a successful nominal contact using two finer native integration grids."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from run_dart import ROOT, Journal, bind, write
from verify_baseline import (
    arrays,
    compare_contact,
    contact_score,
    digest,
    exact,
    read,
    require,
)


def authenticate_trial(trial, authority, protocol):
    require(digest(trial / "manifest.json") == authority, "trial manifest hash differs")
    inventory = read(trial / "manifest.json")
    require(
        {"binding.json", "report.json", "trajectory.npz", "prelude.npz"}
        <= inventory.keys(),
        "incomplete trial evidence",
    )
    for name, expected in inventory.items():
        require(
            Path(name).name == name and digest(trial / name) == expected,
            "trial file hash differs",
        )
    source, report = read(trial / "binding.json"), read(trial / "report.json")
    require(
        source["protocol_sha256"] == digest(protocol) == report["protocol_sha256"],
        "trial protocol differs",
    )
    require(
        report["status"] == "complete"
        and report["all_finite"]
        and report["coarse_submillimeter"]
        and all(precision_pass(report[k]) for k in ("contact", "independent_contact")),
        "coarse primary gate did not pass",
    )
    return source


def precision_pass(score):
    return bool(
        score.get("contact")
        and score.get("hit")
        and score["miss_distance_m"] < 0.001
        and score["axis_error_deg"] <= 5
        and 0.2 <= score["normal_speed_m_s"] <= 2.5
        and score["tangent_speed_m_s"] <= 0.2
        and score["time_s"] <= 1.2
    )


def expand_commands(commands, interval):
    require(interval in (0.001, 0.0005), "unexpected frozen audit grid")
    return np.repeat(commands, round(0.01 / interval), axis=0)


def convergence(scores):
    comparisons = {}
    for kind in ("canonical", "independent"):
        a, b = (row[kind] for row in scores)
        if not (a.get("contact") and b.get("contact")):
            return dict(passed=False, reason="a fine grid has no contact")
        distance = float(
            np.linalg.norm(
                np.asarray(a["contact_position_m"]) - b["contact_position_m"]
            )
        )
        time = abs(a["time_s"] - b["time_s"])
        comparisons[kind] = dict(
            contact_point_difference_m=distance,
            contact_time_difference_s=time,
            passed=bool(distance <= 0.00002 and time <= 0.00002),
        )
    return dict(
        passed=all(row["passed"] for row in comparisons.values()), **comparisons
    )


def run(baseline, dart, trial, authority, protocol, output):
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "attempt.json",
        dict(trial=str(trial), trial_manifest_sha256=authority, protocol=str(protocol)),
    )
    journal = None
    try:
        original = authenticate_trial(trial, authority, protocol)
        p, spec, (mission, _, plants), binding = bind(baseline, dart, protocol, output)
        require(
            original["baseline_manifest_sha256"] == binding["baseline_manifest_sha256"]
            and original["model_sha256"] == binding["model_sha256"]
            and original["dart_source_sha256"] == binding["dart_source_sha256"]
            and original["simulator_files"] == binding["simulator_files"],
            "trial source association differs",
        )
        trajectory = arrays(trial / "trajectory.npz")
        initial = trajectory["states"][0]
        exact(initial, arrays(trial / "prelude.npz")["states"][-1], "initial state")
        exact(
            initial,
            arrays(baseline / "dart/prelude.npz")["states"][-1],
            "nominal initial state",
        )
        exact(
            trajectory["time_s"],
            np.arange(len(trajectory["states"])) * 0.01,
            "issued command grid",
        )
        commands = trajectory["commands"]
        require(
            commands.shape == (len(trajectory["states"]) - 1, 4)
            and np.isfinite(commands).all(),
            "command tape",
        )
        grids = p["acceptance"]["dense_audit"]["sampling_intervals_s"]
        require(
            grids == [0.001, 0.0005] and p["plant"]["substeps"] == 5,
            "frozen dense integration contract",
        )
        plant, target = (
            plants.CrazyflowPlant(**p["plant"]),
            mission.Target(**p["target"]),
        )
        target_spec = dict(spec["target"], radius_m_max=target.radius_m)
        step = jax.jit(plant.step_at_interval)
        journal, scores = Journal(output), []
        for interval in grids:
            tape, states = expand_commands(commands, interval), [initial.copy()]
            try:
                for i, command in enumerate(tape):
                    args = (
                        jnp.asarray(states[-1]),
                        jnp.asarray(command),
                        jnp.asarray(interval),
                    )
                    state = journal.call(
                        f"native_{interval}",
                        i,
                        dict(zip(("state", "command", "interval"), args)),
                        step,
                        args,
                        lambda result: dict(state=result),
                    )
                    require(np.isfinite(state).all(), "nonfinite fine native state")
                    states.append(np.asarray(state))
            finally:
                times = np.arange(len(states)) * interval
                np.savez_compressed(
                    output / f"trajectory-{int(interval * 1e6)}us.npz",
                    states=np.asarray(states),
                    commands=tape[: len(states) - 1],
                    time_s=times,
                )
            canonical = mission.score_contact(np.asarray(states), times, target)
            independent = contact_score(np.asarray(states), times, target_spec)
            agreement = None
            try:
                compare_contact(independent, canonical, spec)
            except ValueError as error:
                agreement = str(error)
            scores.append(
                dict(
                    interval_s=interval,
                    native_steps=len(tape),
                    canonical=canonical,
                    independent=independent,
                    agreement_error=agreement,
                    passed=agreement is None
                    and precision_pass(canonical)
                    and precision_pass(independent),
                )
            )
        journal.close()
        agreement = convergence(scores)
        require(
            all(
                digest(ROOT / name) == value
                for name, value in binding["source_sha256"].items()
            ),
            "source changed during audit",
        )
        report = dict(
            format="glassbox-dart-resolution-audit-v1",
            trial_manifest_sha256=authority,
            protocol_sha256=digest(protocol),
            source_commit=binding["commit"],
            grids=scores,
            convergence=agreement,
            resolution_qualified=all(row["passed"] for row in scores)
            and agreement["passed"],
            native_calls=sum(row["native_steps"] for row in scores),
            model_calls=0,
            optimizer_calls=0,
            fits=0,
        )
        write(output / "report.json", report)
        write(
            output / "manifest.json",
            {f.name: digest(f) for f in sorted(output.iterdir()) if f.is_file()},
        )
        return report
    except BaseException as error:
        if journal is not None:
            journal.close()
        write(output / "failure.json", dict(status="failed", error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "trial"):
        parser.add_argument(name, type=Path)
    for name in ("dart-root", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--trial-manifest-sha256", required=True)
    a = parser.parse_args()
    result = run(
        a.baseline.resolve(),
        a.dart_root.resolve(),
        a.trial.resolve(),
        a.trial_manifest_sha256,
        a.protocol.resolve(),
        a.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
