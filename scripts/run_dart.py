"""Run one frozen Dart consumer trial through the public learned model."""

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from verify_baseline import (
    arrays,
    compare_contact,
    contact_score,
    digest,
    read,
    require,
    runtime_check,
    verify_integrity,
)

ROOT = Path(__file__).resolve().parents[1]


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return str(value)
    return value.item() if isinstance(value, np.generic) else value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


class Journal:
    """Durable exact arrays before/after each call, preserving returned objects."""

    def __init__(self, output):
        self.binary = (output / "arrays.bin").open("xb")
        self.events = (output / "events.jsonl").open("x")
        self.count, self.counts = 0, {}

    def event(self, value, values):
        locations = {}
        for key, data in values.items():
            locations[key] = self.binary.tell()
            np.save(self.binary, np.asarray(data), allow_pickle=False)
        self.binary.flush()
        os.fsync(self.binary.fileno())
        self.events.write(
            json.dumps(clean(dict(value, arrays=locations)), allow_nan=False) + "\n"
        )
        self.events.flush()
        os.fsync(self.events.fileno())

    def call(self, kind, origin, inputs, function, arguments, output):
        self.count += 1
        ticket = dict(call=self.count, kind=kind, origin=int(origin))
        count = self.counts.setdefault(
            kind, dict(started=0, returned=0, raised=0, nonfinite=0)
        )
        count["started"] += 1
        self.event(dict(ticket, phase="started"), inputs)
        try:
            result = function(*arguments)
            values = output(result)
        except Exception as error:
            count["raised"] += 1
            self.event(dict(ticket, phase="raised", error=repr(error)), {})
            raise
        finite = all(np.isfinite(np.asarray(v)).all() for v in values.values())
        count["returned"] += 1
        count["nonfinite"] += int(not finite)
        self.event(dict(ticket, phase="returned", finite=bool(finite)), values)
        return result

    def close(self):
        self.binary.close()
        self.events.close()


def initial(states, commands):
    import jax
    import jax.numpy as jnp

    from glassbox.core.geometry import quaternion_to_rotation_batch

    def observed(state):
        rotation = quaternion_to_rotation_batch(state[6:10][None, :])[0]
        return jnp.concatenate((state[3:6], state[10:13], rotation.reshape(9)))

    with jax.enable_x64(True):
        past = np.asarray(jax.vmap(observed)(states[-51:, :13]))
    return jnp.asarray(states[-1, :13]), jnp.asarray(past), jnp.asarray(commands[-50:])


def initial_arrays(value):
    return dict(zip(("state", "past_states", "past_inputs"), value))


def shift_seed(plan, steps=3):
    return np.concatenate((plan[steps:], np.repeat(plan[-1:], steps, axis=0)))


def bind(baseline, dart, protocol, output):
    p, spec = read(protocol), read(baseline / "dart/spec.json")
    verify_integrity(baseline, p["baseline_manifest_sha256"])
    runtime_check(spec["runtime"])
    require(
        Path(importlib.util.find_spec("glassbox").origin).resolve()
        == ROOT / "src/glassbox/__init__.py",
        "load Glassbox from the bound checkout",
    )
    require(
        all(
            p["controller"][k] == v
            for k, v in dict(steps=120, block_steps=3, maxiter=100, dt_s=0.01).items()
        ),
        "unexpected controller work contract",
    )
    replan = p["controller"]["replan_steps"]
    require(
        type(replan) is int and 1 <= replan <= 120 and 120 % replan == 0,
        "replanning must use an integer number of native observation intervals",
    )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    require(not git("status", "--porcelain"), "commit all source before running")
    relative = protocol.resolve().relative_to(ROOT).as_posix()
    git("ls-files", "--error-unmatch", relative)
    source = {
        name: digest(ROOT / name)
        for name in git("ls-files", "src", "scripts", relative).splitlines()
    }
    sources = {**spec["source_sha256"], **p.get("dart_source_sha256", {})}
    for name, expected in sources.items():
        require(digest(dart / name) == expected, f"Dart source changed: {name}")
    package = Path(importlib.util.find_spec("crazyflow").origin).parent
    for name, expected in p["simulator_files"].items():
        require(
            digest(package.parent / name) == expected,
            f"simulator source changed: {name}",
        )
    sys.path.insert(0, str(dart / "src"))
    modules = [
        importlib.import_module("crazydart." + name)
        for name in ("mission", "planning", "plant")
    ]
    for module in modules:
        require(
            Path(module.__file__).resolve()
            == (dart / "src" / (module.__name__.replace(".", "/") + ".py")).resolve(),
            "unexpected Dart import",
        )
    binding = dict(
        commit=git("rev-parse", "HEAD"),
        source_sha256=source,
        protocol_sha256=digest(protocol),
        baseline_manifest_sha256=p["baseline_manifest_sha256"],
        model_sha256=digest(baseline / "models/dart.npz"),
        runtime=spec["runtime"],
        dart_source_sha256=sources,
        simulator_files=p["simulator_files"],
        simulator_package=str(package),
    )
    write(output / "binding.json", binding)
    return p, spec, modules, binding


def array_comparison(actual, expected):
    matching = actual.shape == expected.shape and actual.dtype == expected.dtype
    return dict(
        shape_and_dtype_equal=matching,
        exact=bool(matching and np.array_equal(actual, expected, equal_nan=True)),
        max_abs_difference=float(np.max(np.abs(actual.astype(np.float64) - expected)))
        if matching and actual.size
        else None,
    )


def matched_prefixes(selected, trajectory, replan_steps, target):
    from evaluate_dart import prefix_error

    rows = []
    for item in selected:
        origin = item["origin"]
        available = min(replan_steps, len(trajectory["commands"]) - origin)
        require(
            np.array_equal(
                item["inputs_commands"][:available],
                trajectory["commands"][origin : origin + available].astype(np.float32),
            ),
            "selected and executed prefixes differ",
        )
        for step in range(1, available + 1):
            rows.append(
                dict(
                    origin=origin,
                    horizon_steps=step,
                    errors=prefix_error(
                        item["result_states"][step],
                        trajectory["states"][origin + step],
                        target,
                    ),
                )
            )
    summary = {}
    for step in sorted({r["horizon_steps"] for r in rows}):
        sample = [r for r in rows if r["horizon_steps"] == step]
        summary[str(step * 10) + "ms"] = {}
        for metric in sample[0]["errors"]:
            errors = np.array([r["errors"][metric]["norm"] for r in sample])
            summary[str(step * 10) + "ms"][metric] = dict(
                count=len(errors),
                rmse_norm=float(np.sqrt(np.mean(errors**2))),
                p95_norm=float(np.quantile(errors, 0.95)),
                maximum_norm=float(errors.max()),
            )
    return dict(rows=rows, summary=summary)


def run(baseline, dart, output, protocol):
    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "attempt.json",
        dict(protocol=str(protocol), baseline=str(baseline), dart=str(dart)),
    )
    try:
        return _run(baseline, dart, output, protocol)
    except BaseException as error:
        write(output / "failure.json", dict(error=repr(error), status="failed"))
        raise


def _run(baseline, dart, output, protocol):
    import jax.numpy as jnp

    from glassbox import LearnedDynamics
    from glassbox.core.geometry import motion_rollout

    p, spec, (mission, planning, plants), binding = bind(
        baseline, dart, protocol, output
    )
    model, plant = (
        LearnedDynamics.load(baseline / "models/dart.npz"),
        plants.CrazyflowPlant(**p["plant"]),
    )
    target = mission.Target(**p["target"])
    prefix, seed = (
        arrays(baseline / "dart/prelude.npz"),
        arrays(baseline / "dart/seed.npz")["commands"],
    )
    for name in ("prelude.npz", "seed.npz"):
        shutil.copyfile(baseline / "dart" / name, output / name)

    def rollout(start, commands):
        return motion_rollout(model, *start, commands)

    c = p["controller"]
    planner = planning.DirectPlanner(
        rollout,
        target,
        steps=c["steps"],
        block_steps=c["block_steps"],
        minimum=c["minimum"],
        maximum=c["maximum"],
    )
    journal = Journal(output)
    grad = planner.value_gradient
    states, commands, selected, diagnostics = [prefix["states"][-1]], [], [], []
    full, issued = prefix["states"].copy(), prefix["commands"].copy()
    error, stop, started = None, "deadline", time.monotonic()

    def timeout(*_):
        raise TimeoutError("frozen trial wall-time limit")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(p["execution"]["timeout_s"])
    try:
        for origin in range(0, 120, c["replan_steps"]):
            start = initial(full, issued)
            planner.value_gradient = lambda flat, x, active, origin=origin: (
                journal.call(
                    "gradient",
                    origin,
                    dict(flat=flat, active_steps=active, **initial_arrays(x)),
                    grad,
                    (flat, x, active),
                    lambda result: dict(objective=result[0], gradient=result[1]),
                )
            )

            def select(x, u, origin=origin):
                result = journal.call(
                    "selected",
                    origin,
                    dict(commands=u, **initial_arrays(x)),
                    rollout,
                    (x, u),
                    lambda result: dict(states=result),
                )
                selected.append(
                    dict(
                        origin=origin,
                        inputs_commands=np.asarray(u),
                        result_states=np.asarray(result),
                    )
                )
                return result

            planner.rollout = select  # Objective already captures the original rollout.
            plan, _prediction, diagnostic = journal.call(
                "solve",
                origin,
                dict(
                    seed=seed,
                    active_steps=np.asarray(120 - origin),
                    **initial_arrays(start),
                ),
                planner.solve,
                (start, seed, 100, 120 - origin),
                lambda result: dict(commands=result[0], states=result[1]),
            )
            diagnostics.append(dict(origin=origin, **diagnostic))
            journal.event(
                dict(phase="diagnostic", origin=origin, diagnostic=diagnostic), {}
            )
            for command in plan[: c["replan_steps"]]:
                state = journal.call(
                    "native",
                    origin,
                    dict(state=np.asarray(states[-1]), command=jnp.asarray(command)),
                    plant.step,
                    (jnp.asarray(states[-1]), jnp.asarray(command)),
                    lambda result: dict(state=result),
                )
                require(np.isfinite(state).all(), "nonfinite native state")
                states.append(np.asarray(state))
                commands.append(command.copy())
                full = np.concatenate((full, np.asarray(state)[None]))
                issued = np.concatenate((issued, command[None]))
                if float(mission.signed_distance(state, target)) <= 0:
                    stop = "contact"
                    break
            if stop == "contact":
                break
            seed = shift_seed(plan, c["replan_steps"])
    except Exception as exc:
        error, stop = repr(exc), "exception"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        journal.close()
    trajectory = dict(
        states=np.asarray(states),
        commands=np.asarray(commands).reshape(-1, 4),
        time_s=np.arange(len(states)) * 0.01,
    )
    np.savez_compressed(output / "trajectory.npz", **trajectory)
    np.savez_compressed(
        output / "selected.npz",
        **{
            k: np.asarray([row[k] for row in selected])
            for k in ("origin", "inputs_commands", "result_states")
        },
    )
    canonical = (
        mission.score_contact(trajectory["states"], trajectory["time_s"], target)
        if commands
        else dict(contact=False, hit=False, reason="no_executed_interval")
    )
    audit_target = dict(
        center_m=p["target"]["center"],
        normal=p["target"]["normal"],
        body_contact_offset_m=p["target"]["contact_offset"],
        radius_m_max=p["target"]["radius_m"],
        axis_error_deg_max=p["target"]["angle_tolerance_deg"],
        normal_speed_m_s_range=[
            p["target"]["minimum_normal_speed_m_s"],
            p["target"]["maximum_normal_speed_m_s"],
        ],
        tangential_speed_m_s_max=p["target"]["maximum_tangent_speed_m_s"],
    )
    independent = contact_score(
        trajectory["states"], trajectory["time_s"], audit_target
    )
    compare_contact(independent, canonical, spec)
    require(
        all(
            digest(ROOT / name) == wanted
            for name, wanted in binding["source_sha256"].items()
        ),
        "source changed during trial",
    )
    finite = all(
        v["nonfinite"] == 0 and v["raised"] == 0 and v["started"] == v["returned"]
        for v in journal.counts.values()
    )
    reference = read(baseline / "dart/trial.json")
    comparison = {}
    for label in ("trajectory", "selected"):
        actual, expected = (
            arrays(output / (label + ".npz")),
            arrays(baseline / "dart" / (label + ".npz")),
        )
        comparison[label] = {
            k: array_comparison(actual[k], v)
            for k, v in expected.items()
            if k in actual
        }
    comparison["solves"] = [
        dict(
            origin=current["origin"],
            converged_equal=current["converged"] == previous["converged"],
            iterations_difference=current["iterations"] - previous["iterations"],
            objective_difference=current["objective"] - previous["objective"],
            initial_objective_difference=current["initial_objective"]
            - previous["initial_objective"],
        )
        for current in diagnostics
        for previous in reference["diagnostics"]
        if current["origin"] == previous["step"]
    ]
    comparison["solve_count_difference"] = len(diagnostics) - len(
        reference["diagnostics"]
    )
    comparison["miss_difference_m"] = (
        canonical.get("miss_distance_m", np.nan)
        - reference["contact"]["miss_distance_m"]
    )
    report = dict(
        status="complete" if error is None else "failed",
        error=error,
        stop=stop,
        elapsed_s=time.monotonic() - started,
        counts=journal.counts,
        solves=diagnostics,
        contact=canonical,
        independent_contact=independent,
        matched_prefixes=matched_prefixes(
            selected, trajectory, c["replan_steps"], audit_target
        ),
        all_finite=finite,
        task_success=bool(error is None and finite and canonical["hit"]),
        coarse_submillimeter=bool(
            error is None
            and finite
            and canonical["hit"]
            and independent["hit"]
            and canonical.get("miss_distance_m", np.inf) < 0.001
            and independent.get("miss_distance_m", np.inf) < 0.001
        ),
        dense_event_audit_performed=False,
        baseline_comparison=comparison,
        fits=0,
        source_commit=binding["commit"],
        protocol_sha256=binding["protocol_sha256"],
    )
    write(output / "report.json", report)
    write(
        output / "manifest.json",
        {f.name: digest(f) for f in sorted(output.iterdir()) if f.is_file()},
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--dart-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "docs/harness/dart-precision-native-v1.json",
    )
    args = parser.parse_args()
    result = run(
        args.baseline.resolve(),
        args.dart_root.resolve(),
        args.output.resolve(),
        args.protocol.resolve(),
    )
    print(json.dumps(clean(result), sort_keys=True, allow_nan=False), flush=True)
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
