"""Audit the fixed successful command tape using scoped float64 native arithmetic."""

import argparse
import json
import subprocess
from contextlib import contextmanager
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

SUPPLEMENT = ROOT / "docs/harness/dart-resolution-precision-v1.json"


def precision_contract(trial, authority, protocol):
    contract = read(SUPPLEMENT)
    require(
        contract["trial_manifest_sha256"] == authority
        and contract["trial_protocol_sha256"] == digest(protocol),
        "trial differs from frozen precision supplement",
    )
    subprocess.check_call(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--error-unmatch",
            str(SUPPLEMENT.relative_to(ROOT)),
        ],
        stdout=subprocess.DEVNULL,
    )
    matches = [
        path
        for path in trial.parent.glob("*/manifest.json")
        if digest(path) == contract["original_float32_audit_manifest_sha256"]
    ]
    require(
        len(matches) == 1, "original float32 audit authority is missing or ambiguous"
    )
    previous = matches[0].parent
    for name, expected in read(matches[0]).items():
        require(
            Path(name).name == name and digest(previous / name) == expected,
            "original float32 audit file differs",
        )
    report = read(previous / "report.json")
    require(
        report["trial_manifest_sha256"] == authority
        and report["protocol_sha256"] == digest(protocol)
        and report["resolution_qualified"] is False,
        "original float32 audit association differs",
    )
    return dict(
        supplemental_contract_sha256=digest(SUPPLEMENT),
        original_float32_audit_manifest_sha256=digest(matches[0]),
        original_float32_resolution_qualified=False,
        arithmetic="float64",
    )


def issued_commands(trial, trajectory):
    """Treat actual native input records as the command authority, not optimizer output."""
    commands, pending = [], None
    with (trial / "arrays.bin").open("rb") as binary:

        def value(event, name):
            binary.seek(event["arrays"][name])
            return np.load(binary, allow_pickle=False)

        with (trial / "events.jsonl").open() as events:
            for line in events:
                event = json.loads(line)
                if event.get("kind") != "native":
                    continue
                if event["phase"] == "started":
                    require(pending is None, "overlapping native calls")
                    index = len(commands)
                    command = value(event, "command")
                    exact(
                        command,
                        trajectory["commands"][index].astype(np.float32),
                        "actually issued command",
                    )
                    exact(
                        value(event, "state"),
                        trajectory["states"][index],
                        "native input state",
                    )
                    commands.append(command)
                    pending = event["call"]
                else:
                    require(
                        event["phase"] == "returned"
                        and pending == event["call"]
                        and event["finite"],
                        "incomplete original native call",
                    )
                    exact(
                        value(event, "state"),
                        trajectory["states"][len(commands)],
                        "native output state",
                    )
                    pending = None
    require(
        pending is None and len(commands) == len(trajectory["commands"]),
        "original native command count differs",
    )
    return np.asarray(commands)


@contextmanager
def native_precision(plant):
    """Promote already constructed float32 coefficients; restore scope on every exit."""
    require(not jax.config.x64_enabled, "construct the native plant in ambient float32")
    original, thrust = plant.params, plant.thrust_max
    source = {name: np.asarray(value).copy() for name, value in original.items()}
    source["thrust_max"] = np.asarray(thrust, dtype=np.float32)
    require(
        all(v.dtype == np.float32 and np.isfinite(v).all() for v in source.values()),
        "native coefficients must retain their original float32 values",
    )
    with jax.enable_x64(True):
        try:
            plant.params = {
                name: jnp.asarray(value, dtype=jnp.float64)
                for name, value in source.items()
                if name != "thrust_max"
            }
            plant.thrust_max = jnp.asarray(source["thrust_max"], dtype=jnp.float64)
            require(
                all(v.dtype == np.float64 for v in plant.params.values())
                and plant.thrust_max.dtype == np.float64,
                "native coefficient promotion failed",
            )
            yield {
                **{"source_" + k: v for k, v in source.items()},
                **{"promoted_" + k: v.astype(np.float64) for k, v in source.items()},
            }
        finally:
            plant.params, plant.thrust_max = original, thrust


def authenticate_trial(trial, authority, protocol):
    require(digest(trial / "manifest.json") == authority, "trial manifest hash differs")
    inventory = read(trial / "manifest.json")
    require(
        {
            "binding.json",
            "report.json",
            "trajectory.npz",
            "prelude.npz",
            "arrays.bin",
            "events.jsonl",
        }
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
        supplement = precision_contract(trial, authority, protocol)
        original = authenticate_trial(trial, authority, protocol)
        p, spec, (mission, _, plants), binding = bind(baseline, dart, protocol, output)
        binding["source_sha256"][str(SUPPLEMENT.relative_to(ROOT))] = digest(SUPPLEMENT)
        write(output / "precision-binding.json", supplement)
        require(
            original["baseline_manifest_sha256"] == binding["baseline_manifest_sha256"]
            and original["model_sha256"] == binding["model_sha256"]
            and original["dart_source_sha256"] == binding["dart_source_sha256"]
            and original["simulator_files"] == binding["simulator_files"],
            "trial source association differs",
        )
        trajectory = arrays(trial / "trajectory.npz")
        initial = trajectory["states"][0]
        require(initial.dtype == np.float32, "original native initial dtype differs")
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
        commands = issued_commands(trial, trajectory)
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
        journal, scores = Journal(output), []
        with native_precision(plant) as inputs:
            inputs.update(
                source_initial=initial,
                promoted_initial=initial.astype(np.float64),
                source_commands=commands,
                promoted_commands=commands.astype(np.float64),
            )
            np.savez_compressed(output / "precision-inputs.npz", **inputs)
            step = jax.jit(plant.step_at_interval)
            for interval in grids:
                tape = expand_commands(inputs["promoted_commands"], interval)
                states = [inputs["promoted_initial"].copy()]
                try:
                    for i, command in enumerate(tape):
                        args = tuple(
                            jnp.asarray(v, dtype=jnp.float64)
                            for v in (states[-1], command, interval)
                        )
                        require(
                            all(v.dtype == np.float64 for v in args),
                            "native replay input precision differs",
                        )
                        state = journal.call(
                            f"native_{interval}",
                            i,
                            dict(zip(("state", "command", "interval"), args)),
                            step,
                            args,
                            lambda result: dict(state=result),
                        )
                        require(
                            state.dtype == np.float64 and np.isfinite(state).all(),
                            "nonfinite or wrong-precision fine native state",
                        )
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
                        state_dtype=str(np.asarray(states).dtype),
                        command_dtype=str(tape.dtype),
                        canonical=canonical,
                        independent=independent,
                        agreement_error=agreement,
                        passed=agreement is None
                        and precision_pass(canonical)
                        and precision_pass(independent),
                    )
                )
        journal.close()
        require(not jax.config.x64_enabled, "ambient float32 was not restored")
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
            **supplement,
            trial_manifest_sha256=authority,
            protocol_sha256=digest(protocol),
            source_commit=binding["commit"],
            grids=scores,
            convergence=agreement,
            resolution_qualified=all(row["passed"] for row in scores)
            and agreement["passed"],
            native_calls=sum(row["native_steps"] for row in scores),
            counts=journal.counts,
            input_dtypes={k: str(v.dtype) for k, v in inputs.items()},
            ambient_x64_restored=True,
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
