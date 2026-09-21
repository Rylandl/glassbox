"""Attribute saved Dart contact error without predicting, fitting or controlling."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import verify_baseline as baseline

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / "docs/harness/dart-precision-v1.json"
_PROTOCOL_SHA256 = "290e7571f9bc2bb6659f72b72c2c31f5432d0e57cdd4e785f45f0849a184702e"
_METRICS = (
    "contact_position_m",
    "position_m",
    "velocity_m_s",
    "body_rate_rad_s",
    "contact_velocity_m_s",
)


def write(path, value):
    with Path(path).open("x") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def source_identity():
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=_ROOT).decode().strip()

    baseline.require(
        not git("status", "--porcelain"), "commit source before evaluation"
    )
    files = [
        "scripts/evaluate_dart.py",
        "scripts/verify_baseline.py",
        "src/glassbox/core/dynamics.py",
        "docs/harness/dart-precision-v1.json",
    ]
    return dict(
        root=str(_ROOT),
        commit=git("rev-parse", "HEAD"),
        files={name: baseline.digest(_ROOT / name) for name in files},
    )


def dart_sources(root, spec):
    required = {"src/crazydart/mission.py", "src/crazydart/planning.py"}
    baseline.require(required <= set(spec["source_sha256"]), "missing Dart source pins")
    for name, expected in spec["source_sha256"].items():
        baseline.require(
            baseline.digest(root / name) == expected, f"Dart source: {name}"
        )
    return dict(root=str(root), files=spec["source_sha256"])


def dart_scorer(root, target):
    sys.path.insert(0, str(root / "src"))
    modules = [
        importlib.import_module(f"crazydart.{name}") for name in ("mission", "planning")
    ]
    for module in modules:
        expected = root / "src" / (module.__name__.replace(".", "/") + ".py")
        baseline.require(
            Path(module.__file__).resolve() == expected.resolve(),
            "Dart imported from a different tree",
        )
    mission = modules[0]
    t = mission.Target(
        center=tuple(target["center_m"]),
        normal=tuple(target["normal"]),
        contact_offset=tuple(target["body_contact_offset_m"]),
        radius_m=target["radius_m_max"],
        angle_tolerance_deg=target["axis_error_deg_max"],
        maximum_tangent_speed_m_s=target["tangential_speed_m_s_max"],
        minimum_normal_speed_m_s=target["normal_speed_m_s_range"][0],
        maximum_normal_speed_m_s=target["normal_speed_m_s_range"][1],
    )
    return lambda states, times: mission.score_contact(states, times, t)


def lateral(point, target):
    normal = np.asarray(target["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    error = np.asarray(point) - target["center_m"]
    return error - normal * (error @ normal)


def prefix_error(prediction, truth, target):
    point, _, contact_velocity = baseline.kinematics(prediction, target)
    true_point, _, true_contact_velocity = baseline.kinematics(truth, target)
    deltas = (
        point - true_point,
        prediction[:3].astype(float) - truth[:3],
        prediction[3:6].astype(float) - truth[3:6],
        prediction[10:13].astype(float) - truth[10:13],
        contact_velocity - true_contact_velocity,
    )
    return {
        name: dict(vector=value.tolist(), norm=float(np.linalg.norm(value)))
        for name, value in zip(_METRICS, deltas, strict=True)
    }


def summarize(rows):
    result = {}
    for steps in (1, 2, 3):
        selected = [row for row in rows if row["steps"] == steps]
        baseline.require(len(selected) == 40, "missing executed prediction prefix")
        result[str(steps * 10) + "ms"] = dict(count=len(selected), metrics={})
        for name in _METRICS:
            vectors = np.asarray([row["errors"][name]["vector"] for row in selected])
            norms = np.linalg.norm(vectors, axis=1)
            result[str(steps * 10) + "ms"]["metrics"][name] = dict(
                rmse_norm=float(np.sqrt(np.mean(norms**2))),
                p95_norm=float(np.quantile(norms, 0.95)),
                maximum_norm=float(norms.max()),
                mean_vector=vectors.mean(axis=0).tolist(),
            )
    return result


def analyze(data, spec, scorer):
    """Reduce saved means and their exact executed prefixes; no dynamics calls."""
    selected, trajectory = data["selected"], data["trajectory"]
    target, controller = spec["target"], spec["controller"]
    dt, horizon = controller["dt_s"], controller["steps"]
    baseline.require(dt == 0.01 and horizon == 120, "frozen time contract")
    baseline.exact(
        selected["origin"], np.arange(0, 120, 3, dtype=np.int64), "all plan origins"
    )
    baseline.require(
        selected["result_states"].shape == (40, 121, 13), "saved plan shape"
    )
    audits = []

    def score(states, times, label):
        canonical = scorer(states, times)
        independent = baseline.contact_score(states, times, target)
        baseline.compare_contact(independent, canonical, spec)
        audits.append(
            dict(label=label, passed=True, canonical=canonical, independent=independent)
        )
        return canonical

    actual = score(trajectory["states"], trajectory["time_s"], "executed trajectory")
    plans, prefixes = [], []
    for index, value in enumerate(selected["origin"]):
        origin = int(value)
        end = horizon - origin
        states = selected["result_states"][index]
        baseline.exact(
            states[0], trajectory["states"][origin, :13], "selected initial state"
        )
        baseline.exact(
            selected["inputs_commands"][index, :3],
            trajectory["commands"][origin : origin + 3].astype(np.float32),
            "executed command prefix",
        )
        times = (origin + np.arange(end + 1)) * dt
        contact = score(states[: end + 1], times, f"plan origin {origin}")
        terminal = baseline.kinematics(states[end], target)[0]
        terminal_lateral = lateral(terminal, target)
        crossing_lateral = (
            lateral(contact["contact_position_m"], target)
            if contact["contact"]
            else None
        )
        row = dict(
            origin=origin,
            time_s=origin * dt,
            remaining_steps=end,
            predicted_contact=contact,
            predicted_global_crossing_interval=(
                origin + contact["interval"] if contact["contact"] else None
            ),
            terminal_contact_position_m=terminal.tolist(),
            terminal_lateral_error_m=terminal_lateral.tolist(),
            terminal_lateral_distance_m=float(np.linalg.norm(terminal_lateral)),
            crossing_lateral_error_m=(
                crossing_lateral.tolist() if crossing_lateral is not None else None
            ),
            terminal_minus_crossing_lateral_m=(
                (terminal_lateral - crossing_lateral).tolist()
                if crossing_lateral is not None
                else None
            ),
            prefixes=[],
        )
        for step in (1, 2, 3):
            entry = dict(
                origin=origin,
                steps=step,
                horizon_s=step * dt,
                absolute_time_s=(origin + step) * dt,
                errors=prefix_error(
                    states[step], trajectory["states"][origin + step], target
                ),
            )
            row["prefixes"].append(entry)
            prefixes.append(entry)
        plans.append(row)
    final_contact = plans[-1]["predicted_contact"]
    difference = None
    if final_contact["contact"] and actual["contact"]:
        baseline.require(
            actual["interval"] >= plans[-1]["origin"],
            "actual first contact predates final executed plan",
        )
        difference = {
            name: float(final_contact[name] - actual[name])
            for name in (
                "time_s",
                "miss_distance_m",
                "axis_error_deg",
                "normal_speed_m_s",
                "tangent_speed_m_s",
            )
        }
        difference["contact_position_m"] = (
            np.asarray(final_contact["contact_position_m"])
            - actual["contact_position_m"]
        ).tolist()
    return dict(
        format="glassbox-dart-saved-attribution-v1",
        plans=plans,
        final_three_origins=plans[-3:],
        prefix_summary=summarize(prefixes),
        final_prediction=dict(
            predicted=final_contact, actual=actual, predicted_minus_actual=difference
        ),
        contact_audit=dict(
            passed=True,
            comparisons=len(audits),
            comparison_budget=spec["contact_comparison"],
            records=audits,
        ),
        counts=dict(
            plans=len(plans),
            prefixes=len(prefixes),
            contact_scores=len(audits),
            no_predicted_crossing=sum(
                not p["predicted_contact"]["contact"] for p in plans
            ),
        ),
        calls=dict(fits=0, model_predictions=0, optimizer_solves=0, native_steps=0),
        interpretation="Only executed three-step prefixes are forecast/truth comparisons; earlier remaining-plan contacts are predictions.",
        new_task_success_claim=False,
        complete=True,
    )


def run(root, dart_root, output):
    root, dart_root, output = map(
        lambda p: Path(p).resolve(), (root, dart_root, output)
    )
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "request.json",
        dict(
            baseline=str(root),
            dart_root=str(dart_root),
            protocol_sha256=_PROTOCOL_SHA256,
        ),
    )
    try:
        baseline.require(
            baseline.digest(_PROTOCOL) == _PROTOCOL_SHA256, "frozen protocol differs"
        )
        protocol = baseline.read(_PROTOCOL)
        manifest = baseline.verify_integrity(root, protocol["baseline_manifest_sha256"])
        spec = baseline.read(root / "dart/spec.json")
        baseline.runtime_check(spec["runtime"])
        flags = {key: os.environ.get(key) for key in spec["runtime"]["flags"]}
        baseline.require(flags == spec["runtime"]["flags"], "runtime flags differ")
        binding = dict(
            source=source_identity(),
            dart=dart_sources(dart_root, spec),
            baseline_manifest_sha256=protocol["baseline_manifest_sha256"],
            baseline_files=len(manifest["files"]),
            protocol_sha256=_PROTOCOL_SHA256,
            runtime=spec["runtime"],
            interpreter=sys.executable,
        )
        write(output / "binding.json", binding)
        scorer = dart_scorer(dart_root, spec["target"])
        data, checked_spec, _ = baseline.dart_arrays(root)
        baseline.require(checked_spec == spec, "baseline spec changed")
        result = analyze(data, spec, scorer)
        baseline.require(
            result["final_prediction"]["actual"]
            == baseline.read(root / "dart/trial.json")["contact"],
            "canonical actual contact differs from frozen trial",
        )
        baseline.require(
            source_identity() == binding["source"],
            "local source changed during attribution",
        )
        baseline.require(
            dart_sources(dart_root, spec) == binding["dart"],
            "Dart source changed during attribution",
        )
        result["binding_sha256"] = baseline.digest(output / "binding.json")
        write(output / "result.json", result)
        return result
    except Exception as error:
        write(
            output / "failure.json",
            dict(type=type(error).__name__, message=str(error), complete=False),
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--dart-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new evidence directory; an existing path is never overwritten",
    )
    args = parser.parse_args()
    result = run(args.baseline, args.dart_root, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("counts", "prefix_summary", "final_prediction", "complete")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
