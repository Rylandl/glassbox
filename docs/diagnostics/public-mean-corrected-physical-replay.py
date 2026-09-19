"""Frozen no-fit source bridge for corrected public inference regression."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

if not __debug__:
    raise RuntimeError("Integrity assertions require Python without optimization")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inventory(root):
    paths = sorted(Path(root).rglob("*"))
    assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix(): digest(p) for p in paths if p.is_file()}


def authenticate(args):
    assert digest(args.policy) == args.policy_sha256
    policy = read(args.policy)
    assert digest(__file__) == policy["driver_sha256"]
    from glassbox.experimental.public_mean_implementation import verify

    current = verify(args.binding, args.binding_sha256)
    assert (
        Path(__file__).resolve()
        == Path(current["public_root"]) / policy["driver_relative"]
    )
    old = verify(policy["old_binding"]["path"], policy["old_binding"]["sha256"])
    source_association(old, current, policy)
    from glassbox.experimental import public_mean_physical_evaluation as pe

    assert digest(pe.__file__) == policy["evaluator_sha256"]
    pe.operator_continuity(current)
    return policy, current, old, pe


def source_association(old, current, policy):
    core = "src/glassbox/_sequence_model.py"
    assert current["public_source_sha256"][core] == policy["corrected_core_sha256"]
    for key in (
        "runtime",
        "machine",
        "interpreter_sha256",
        "oracle_commit",
        "oracle_source_sha256",
        "consumer_source_sha256",
    ):
        assert old[key] == current[key], key
    differences = {}
    for relative, expected in old["public_source_sha256"].items():
        if relative.startswith("src/glassbox/"):
            actual = current["public_source_sha256"].get(relative)
            if actual != expected:
                differences[relative] = {"old": expected, "current": actual}
    assert differences == policy["allowed_common_source_differences"]
    return dict(
        old_commit=old["implementation_commit"],
        current_commit=current["implementation_commit"],
        declared_source_changes=differences,
    )


def old_inputs(pe, policy, simulator):
    anchor = policy["simulators"][simulator]
    seal = pe.sealed(
        anchor["evaluation_root"], "seal.json", anchor["evaluation_sha256"]
    )
    assert seal["files"] == pe.inventory(anchor["evaluation_root"])
    assert seal["status"] == "complete" and seal["simulator"] == simulator
    assert seal["inputs"]["binding_sha256"] == policy["old_binding"]["sha256"]
    inputs = {
        k: v for k, v in seal["inputs"].items() if k not in ("simulator", "replay")
    }
    assert inputs["data_sha256"] == anchor["data_sha256"]
    assert inputs["flight_fit_sha256"] == anchor["flight_fit_sha256"]
    assert (
        inputs["data_binding_sha256"]
        == inputs["flight_binding_sha256"]
        == policy["original_fit_binding_sha256"]
    )
    return Path(anchor["evaluation_root"]), inputs


def worker(args):
    policy, current, old, pe = authenticate(args)
    previous, inputs = old_inputs(pe, policy, args.simulator)
    if args.phase == "validate-old":
        assert Path(pe.__file__).resolve().is_relative_to(Path(old["public_root"]))
        report = pe.verify(
            previous,
            expected_sha256=policy["simulators"][args.simulator]["evaluation_sha256"],
            **inputs,
        )
        assert report["fits"] == report["simulations"] == 0
        write(args.output / "result.json", report)
        return

    import jax
    import numpy as np

    from glassbox import LearnedDynamics

    assert args.phase in ("public32", "public64")
    assert bool(jax.config.x64_enabled) == (args.phase == "public64")
    assert jax.default_backend() == "cpu"
    pe._imports(current, args.phase)
    pe.sealed(inputs["data_root"], "seal.json", inputs["data_sha256"])
    fit = pe.sealed(
        inputs["flight_fit_root"], "manifest.json", inputs["flight_fit_sha256"]
    )
    assert (
        fit["implementation_manifest_sha256"] == policy["original_fit_binding_sha256"]
    )
    model_path = Path(inputs["flight_fit_root"]) / "model.npz"
    model = LearnedDynamics.load(model_path)
    old_result = pe.read(previous / args.phase / "result.json")
    arm = pe.ROLES[args.phase][0]
    assert old_result["model_sha256"][arm] == digest(model_path)
    fingerprint = model.fingerprint()
    qualification = read(
        Path(current["public_root"]) / "docs/harness/public-mean-qualification-v1.json"
    )
    resolved = pe.resolved(
        inputs["reference_root"], qualification, current["public_root"]
    )
    _, queries, _ = pe._planned(inputs["data_root"], args.simulator, resolved)
    counts = dict(branch_queries=0, factual_queries=0, inferred_prefixes=0)
    arrays_compared = 0
    prediction_arrays = 0

    def emit(relative, values):
        nonlocal arrays_compared
        pe.equal(
            values,
            pe.arrays(previous / args.phase / relative),
            "corrected_prediction_regression",
        )
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **values)
        arrays_compared += len(values)

    emit("envelopes.npz", {arm: np.asarray(model.envelope())})
    calibration = None
    if args.phase == "public64":
        values, calibration = pe.calibration_evidence(model)
        emit("calibration.npz", values)
        assert calibration == old_result["calibration_reconstruction"]
    function = jax.jit(model.predict)
    for query in queries:
        values = pe.arrays(pe.under(inputs["data_root"], query["path"]))
        saved = {}
        for factual in (False, True) if query["kind"] == "response" else (False,):
            name = arm + ("_factual" if factual else "")
            saved[name] = pe.predict_query(
                function,
                query,
                values,
                dtype=np.float64 if args.phase == "public64" else np.float32,
                factual=factual,
            )
            counts["factual_queries" if factual else "branch_queries"] += 1
            commands = values["factual_inputs" if factual else "future_inputs"]
            counts["inferred_prefixes"] += int(
                query["history_eligible"] and pe.finite_command_prefix(commands) > 0
            )
            prediction_arrays += 1
        emit(pe.query_path(query), saved)
    assert counts == old_result["counters"][arm]
    assert model.fingerprint() == fingerprint
    assert bool(jax.config.x64_enabled) == (args.phase == "public64")
    policy2, current2, _, _ = authenticate(args)
    assert current2 == current and policy2 == policy
    write(
        args.output / "result.json",
        dict(
            status="complete",
            original_evaluation_sha256=policy["simulators"][args.simulator][
                "evaluation_sha256"
            ],
            old_fit_binding_sha256=policy["original_fit_binding_sha256"],
            new_inference_binding_sha256=args.binding_sha256,
            original_fit_manifest_sha256=inputs["flight_fit_sha256"],
            model_sha256=digest(model_path),
            model_fingerprint=fingerprint,
            actual_imported_sources=pe._imports(current, args.phase),
            role=args.phase,
            counters=counts,
            queries=len(queries),
            exact_arrays=arrays_compared,
            exact_prediction_arrays=prediction_arrays,
            calibration=calibration,
            fits=0,
            initializers=0,
            simulations=0,
        ),
    )


def launch(args, phase, simulator, output, execution, binding):
    execution.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        SCIPY_ARRAY_API="1",
        PYTHONPATH=str(Path(binding["public_root"]) / "src"),
    )
    if phase == "public32":
        environment.pop("JAX_ENABLE_X64", None)
    else:
        environment["JAX_ENABLE_X64"] = "1"
    command = [
        binding["interpreter"],
        str(Path(__file__).resolve()),
        str(output),
        "--phase",
        phase,
        "--simulator",
        simulator,
        "--policy",
        str(args.policy),
        "--policy-sha256",
        args.policy_sha256,
        "--binding",
        str(args.binding),
        "--binding-sha256",
        args.binding_sha256,
    ]
    write(
        execution / "command.json",
        dict(
            argv=command,
            environment={
                k: environment.get(k)
                for k in ("PYTHONPATH", "JAX_ENABLE_X64", "SCIPY_ARRAY_API", "PATH")
            },
            hard_timeout_s=14400,
        ),
    )
    started = time.monotonic()
    outcome = dict(returncode=None, hard_timeout=False)
    try:
        with (
            (execution / "stdout.log").open("x") as stdout,
            (execution / "stderr.log").open("x") as stderr,
        ):
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                outcome["returncode"] = process.wait(timeout=14400)
            except subprocess.TimeoutExpired:
                outcome["hard_timeout"] = True
                os.killpg(process.pid, signal.SIGKILL)
                outcome["returncode"] = process.wait()
        assert outcome["returncode"] == 0 and not outcome["hard_timeout"], (
            "Worker failed; no retry"
        )
    finally:
        outcome["elapsed_s"] = time.monotonic() - started
        write(execution / "exit.json", outcome)


def replay_consumer(args, policy, current, pe):
    from glassbox.experimental import public_mean_implementation as implementation
    from glassbox.experimental.public_mean_consumer_qualification import _launch

    anchors = policy["consumer"]
    packet = Path(anchors["manifest"])
    original = Path(anchors["original_directory"])
    assert digest(packet) == anchors["manifest_sha256"]
    assert digest(original / "result.json") == anchors["result_sha256"]
    old_report = read(original / "result.json")
    assert old_report["input_manifest_sha256"] == anchors["manifest_sha256"]
    assert digest(original / old_report["arrays_file"]) == old_report["arrays_sha256"]
    before = inventory(packet.parent)
    runtime = implementation.consumer_preflight(
        args.binding, args.binding_sha256, args.output / "consumer-preflight"
    )
    destination = args.output / "consumer"
    command = [
        current["interpreter"],
        "-m",
        "crazydart.glassbox_forecast",
        "--manifest",
        str(packet),
        "--expected-manifest-sha256",
        anchors["manifest_sha256"],
        "--output",
        str(destination),
    ]
    _launch(command, current, args.output / "consumer-execution")
    fresh = read(destination / "result.json")
    assert fresh["runtime"] == runtime
    assert digest(destination / fresh["arrays_file"]) == fresh["arrays_sha256"]
    ignored = {"runtime", "arrays_sha256"}
    assert {k: v for k, v in fresh.items() if k not in ignored} == {
        k: v for k, v in old_report.items() if k not in ignored
    }
    actual, expected = (
        pe.arrays(destination / fresh["arrays_file"]),
        pe.arrays(original / old_report["arrays_file"]),
    )
    pe.equal(actual, expected, "corrected_consumer_regression")
    assert len(actual) == 168 and inventory(packet.parent) == before
    result = dict(
        exact_arrays=len(actual),
        counts=fresh["counts"],
        old_result_sha256=anchors["result_sha256"],
        original_packet_sha256=anchors["manifest_sha256"],
        current_inference_binding_sha256=args.binding_sha256,
        original_packet_qualification_preserved=True,
        scope="Fresh unchanged Dart consumer with corrected public imports; exact original outputs/JVPs; no export, fit, update or repeated negative audit.",
    )
    write(args.output / "consumer-comparison.json", result)
    return result


def check_counts(reports, rows, consumer, expected):
    assert set(reports) == set(rows) == {"crazyflow", "cascade"}
    observed = dict(
        metric_rows=sum(map(len, rows.values())),
        consumer_arrays=consumer["exact_arrays"],
        consumer_means=consumer["counts"]["means"],
        consumer_responses=consumer["counts"]["responses"],
    )
    for name, actual in observed.items():
        assert actual == expected[name], name
    for phase in ("public32", "public64"):
        values = [reports[simulator][phase] for simulator in reports]
        counts = dict(
            query_slots_per_precision=sum(r["queries"] for r in values),
            prediction_arrays_per_precision=sum(
                r["exact_prediction_arrays"] for r in values
            ),
            inferred_prefixes_per_precision=sum(
                r["counters"]["inferred_prefixes"] for r in values
            ),
        )
        assert all(counts[k] == expected[k] for k in counts), phase
        assert all(
            r[k] == expected["new_" + k] == 0
            for r in values
            for k in ("fits", "initializers", "simulations")
        )
        observed[phase] = counts
    assert expected["original_fitting_operations"] == 32
    return observed


def run(args):
    policy, current, old, pe = authenticate(args)
    rows, queries, reports = {}, {}, {}
    for simulator in policy["simulators"]:
        base = args.output / simulator
        previous, inputs = old_inputs(pe, policy, simulator)
        launch(
            args,
            "validate-old",
            simulator,
            base / "old-validation",
            base / "old-validation-execution",
            old,
        )
        # This retained directory still contains its original source/result identity.
        shutil.copytree(previous / "historical", base / "historical")
        assert inventory(previous / "historical") == inventory(base / "historical")
        for phase in ("public32", "public64"):
            launch(
                args,
                phase,
                simulator,
                base / phase,
                base / (phase + "-execution"),
                current,
            )
        reports[simulator] = {
            phase: read(base / phase / "result.json")
            for phase in ("public32", "public64")
        }
        qualification = read(
            Path(current["public_root"])
            / "docs/harness/public-mean-qualification-v1.json"
        )
        resolved = pe.resolved(
            inputs["reference_root"], qualification, current["public_root"]
        )
        reduced = pe.score(inputs["data_root"], base, simulator, resolved)
        for name, value in reduced.items():
            assert value == read(previous / (name + ".json")), name
            write(base / (name + ".json"), value)
        rows[simulator] = reduced["rows"]
        _, queries[simulator], _ = pe._planned(inputs["data_root"], simulator, resolved)
    decision = pe.reduce_decision(rows, resolved, qualification, queries=queries)
    assert digest(policy["old_decision"]["path"]) == policy["old_decision"]["sha256"]
    assert decision == read(policy["old_decision"]["path"])
    write(args.output / "decision.json", decision)
    consumer = replay_consumer(args, policy, current, pe)
    counts = check_counts(reports, rows, consumer, policy["expected_counts"])
    authenticate(args)
    return dict(
        status="complete",
        consumer=consumer,
        checked_counts=counts,
        source_association=source_association(old, current, policy),
        original_fitting_operations=32,
        scope="Corrected-code regression on the original held-out cohort, not new physical accuracy evidence.",
        fits=0,
        initializers=0,
        simulations=0,
        current_binding_sha256=args.binding_sha256,
        policy_sha256=args.policy_sha256,
        original_physical_binding=policy["old_binding"],
        original_fit_binding_sha256=policy["original_fit_binding_sha256"],
        historical_predictions="Authenticated byte-for-byte copies; no new historical inference.",
        exact_metric_rows=sum(len(x) for x in rows.values()),
        physical_decision_exact=True,
        adoption_assessed=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument(
        "--phase",
        choices=("run", "validate-old", "public32", "public64"),
        default="run",
    )
    parser.add_argument("--simulator", choices=("crazyflow", "cascade"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        if args.phase == "run":
            write(args.output / "result.json", run(args))
        else:
            worker(args)
    except BaseException as error:
        write(
            args.output / "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            ),
        )
        raise
    finally:
        write(
            args.output / "seal.json",
            dict(phase=args.phase, files=inventory(args.output)),
        )


if __name__ == "__main__":
    main()
