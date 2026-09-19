"""No-fit corrected-inference bridge for saved synthetic and lifecycle evidence."""

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

if not __debug__:
    raise RuntimeError("Integrity assertions require Python without optimization")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def inventory(root, exclude=()):
    paths = sorted(Path(root).rglob("*"))
    assert not any(p.is_symlink() for p in paths)
    return {
        p.relative_to(root).as_posix(): sha(p)
        for p in paths
        if p.is_file() and p.relative_to(root).as_posix() not in exclude
    }


def historical_binding(entry):
    """Authenticate a recorded checkout through immutable original Git objects."""
    assert sha(entry["path"]) == entry["sha256"]
    old = read(entry["path"])
    revision = entry["original_commit"]
    assert old["implementation_commit"] == revision and len(revision) == 40
    root = old["public_root"]
    resolved = subprocess.check_output(
        ["git", "-C", root, "rev-parse", revision + "^{commit}"], text=True
    ).strip()
    assert resolved == revision, "original source revision"
    core = None
    assert old["public_source_sha256"]
    for relative, expected in old["public_source_sha256"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        blob = subprocess.check_output(
            ["git", "-C", root, "show", revision + ":" + relative]
        )
        assert hashlib.sha256(blob).hexdigest() == expected, (
            "original source blob " + relative
        )
        if relative == "src/glassbox/_sequence_model.py":
            core = blob
    assert core is not None
    return old, core


def source_association(old, current, policy, *, old_core=None):
    for key in (
        "format",
        "protocol_sha256",
        "runtime",
        "machine",
        "interpreter_sha256",
        "oracle_commit",
        "oracle_source_sha256",
        "consumer_source_sha256",
    ):
        assert old[key] == current[key], key
    differences = {
        path: {"old": value, "current": current["public_source_sha256"].get(path)}
        for path, value in old["public_source_sha256"].items()
        if path.startswith("src/glassbox/")
        and current["public_source_sha256"].get(path) != value
    }
    assert differences == policy["allowed_common_source_differences"]
    core = "src/glassbox/_sequence_model.py"

    # This bridge does not extend old fitting evidence to changed fitting code.
    def unchanged_body(source):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SequenceModel":
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name in (
                        "rollout",
                        "memory_state",
                    ):
                        child.body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    assert unchanged_body(
        old_core
        if old_core is not None
        else (Path(old["public_root"]) / core).read_bytes()
    ) == unchanged_body((Path(current["public_root"]) / core).read_bytes())
    return dict(
        old_commit=old["implementation_commit"],
        new_commit=current["implementation_commit"],
        old_root=old["public_root"],
        new_root=current["public_root"],
        changes=differences,
        unchanged_core_outside_two_inference_methods=True,
        original_source_authenticated_from="pinned_git_objects"
        if old_core is not None
        else "unchanged_checkout",
        old_root_is_executed=False,
    )


def authenticate(args):
    assert sha(args.policy) == args.policy_sha256
    policy = read(args.policy)
    assert sha(__file__) == policy["driver_sha256"]
    from glassbox.experimental.public_mean_implementation import verify
    from glassbox.experimental.public_v4_numerics import _imported_sources

    current = verify(args.binding, args.binding_sha256)
    assert (
        Path(__file__).resolve()
        == Path(current["public_root"]) / policy["driver_relative"]
    )
    old, association = {}, {}
    for name, entry in policy["old_bindings"].items():
        if entry["source_authentication"] == "pinned_git_objects":
            old[name], core = historical_binding(entry)
        else:
            assert entry["source_authentication"] == "unchanged_checkout"
            old[name], core = verify(entry["path"], entry["sha256"]), None
        association[name] = source_association(
            old[name], current, policy, old_core=core
        )
    _imported_sources(current["public_root"], current["public_source_sha256"])
    return policy, current, old, association


def sealed(entry):
    root = Path(entry["root"])
    assert sha(root / "run.json") == entry["run_sha256"]
    value = read(root / "run.json")
    assert value["status"] == "complete"
    assert value["files"] == inventory(root, ("run.json",))
    return root, value


def development_parity(model, directory):
    import jax
    import numpy as np

    from glassbox import learner
    from glassbox.experimental.public_mean_synthetic import _same

    fingerprint = model.fingerprint()
    ambient = bool(jax.config.x64_enabled)
    with jax.enable_x64(True):
        errors = learner._measure(model._model, model._development)
        envelope, windows, rank = learner._calibrate(model._model, model._development)
    assert bool(jax.config.x64_enabled) == ambient
    assert errors == model.report["development_errors"]
    expected = model.report["envelope"]
    assert (
        windows == expected["calibration_windows"] and rank == expected["quantile_rank"]
    )
    _same(envelope, model.envelope(), "saved development envelope")
    _same(envelope, np.asarray(expected["half_width"]), "saved report envelope")
    assert model.fingerprint() == fingerprint
    path = Path(directory) / "development-reconstruction.npz"
    np.savez_compressed(path, envelope=envelope)
    result = dict(
        development_errors=errors,
        calibration_windows=windows,
        quantile_rank=rank,
        envelope_sha256=sha(path),
        model_fingerprint=fingerprint,
    )
    write(Path(directory) / "development-reconstruction.json", result)
    return result


def synthetic(policy, current, output):
    import jax

    import glassbox
    from glassbox import _sequence_model as core
    from glassbox import learner
    from glassbox.experimental import public_mean_synthetic as synthetic

    root, saved = sealed(policy["synthetic"])
    assert saved["binding_sha256"] == policy["old_bindings"]["synthetic"]["sha256"]
    cases, _ = synthetic.scoring.specification(current["public_root"])
    synthetic._verify_data(
        root / "data", saved["generated_data_sha256"], saved["binding_sha256"], cases
    )
    reference = policy["synthetic"]["artifact_root"]
    scales = synthetic.scoring.authenticate_reference_scales(
        current["public_root"], reference
    )
    assert len(cases) == len(saved["case_results"]) == 27
    counts = dict(forbidden_fit_or_initializer_calls=0)

    def forbidden(*_args, **_kwargs):
        counts["forbidden_fit_or_initializer_calls"] += 1
        raise RuntimeError("new numerical fitting is forbidden")

    fresh_rows = []
    (output / "cases").mkdir()
    with ExitStack() as stack:
        for obj, name in (
            (glassbox, "fit"),
            (glassbox.LearnedDynamics, "update"),
            (learner, "fit_sequence_model"),
            (core, "fit_sequence_model"),
            (core, "initialize_sequence_model"),
        ):
            stack.enter_context(patch.object(obj, name, forbidden))
        for case, prior in zip(cases, saved["case_results"], strict=True):
            assert prior["status"] == "complete" and prior["fit_complete"]
            assert prior["case"] == case["name"]
            original = root / "cases" / case["name"]
            destination = output / "cases" / case["name"]
            report = synthetic._verify_case(
                original,
                destination,
                case,
                root / "data",
                reference,
                scales[case["name"]],
                prior,
            )
            model = glassbox.LearnedDynamics.load(original / "model.npz")
            report["development_reconstruction"] = development_parity(
                model, destination
            )
            row = copy.deepcopy(prior)  # Original fit completion remains old evidence.
            row["precisions"] = {
                precision: read(destination / precision / "result.json")
                for precision in ("float32", "float64")
            }
            fresh_rows.append(row)
            write(destination / "bridge.json", report)
            jax.clear_caches()
    capability, legacy = synthetic.aggregate(current["public_root"], cases, fresh_rows)
    assert capability == saved["capability"]
    assert legacy == read(root / "legacy-rows.json")
    assert counts["forbidden_fit_or_initializer_calls"] == 0
    assert inventory(root, ("run.json",)) == saved["files"]
    return dict(
        capability_from_fresh_forecasts=capability,
        original_capability=saved["capability"],
        old_fit_completion_records=27,
        regimes_per_precision=51,
        probes_per_precision=3,
        development_reports=27,
        legacy_rows_equal=True,
        legacy_decisions_preserved=saved["legacy_diagnostics"],
        new_fits=0,
        new_initializers=0,
        generated_data=0,
        **counts,
    )


def normalized_checks(rows):
    return [
        {key: value for key, value in row.items() if key != "sha256"} for row in rows
    ]


def lifecycle(policy, current, old, output, enabled):
    import jax

    from glassbox import _sequence_model as core
    from glassbox.experimental import public_mean_lifecycle as lifecycle

    root, saved = sealed(policy["lifecycle"])
    assert (
        saved["operations"] == list(lifecycle.OPERATIONS)
        and saved["successful_fit_budget"] == 3
    )
    expected = copy.deepcopy(saved["identity"])
    assert expected["package_path"] == str(Path(old["public_root"]) / "src/glassbox")
    for name, value in expected["sources"].items():
        assert value == old["public_source_sha256"]["src/glassbox/" + name]
    expected["sources"]["_sequence_model.py"] = current["public_source_sha256"][
        "src/glassbox/_sequence_model.py"
    ]
    expected["package_path"] = str(Path(current["public_root"]) / "src/glassbox")
    assert lifecycle._identity() == expected
    assert bool(jax.config.x64_enabled) == enabled
    forbidden_calls = []

    def forbidden(*_args, **_kwargs):
        forbidden_calls.append("unexpected real initializer")
        raise RuntimeError("new numerical initialization is forbidden")

    # Existing scope checks temporarily install their own throwing initializer and
    # stub fitter. They enter control flow but never perform initialization/updates.
    with patch.object(core, "initialize_sequence_model", forbidden):
        fresh = lifecycle._checks(root, output / "checks", enabled)
    assert not forbidden_calls
    previous = read(
        root / ("checks-true" if enabled else "checks-false") / "outcome.json"
    )
    for key in ("verification", "transforms"):
        assert fresh[key] == previous[key], key
    assert normalized_checks(fresh["checks"]) == normalized_checks(previous["checks"])
    assert bool(jax.config.x64_enabled) == enabled
    assert inventory(root, ("run.json",)) == saved["files"]
    return dict(
        ambient_x64=enabled,
        operations=list(lifecycle.OPERATIONS),
        original_fitting_operations=3,
        checks=len(fresh["checks"]),
        exact_verification_and_transform_reports=True,
        archive_and_scope_checks_equal=True,
        new_fits=0,
        new_initializers=0,
        scope="Existing rejected/stubbed fit/update control-flow checks; no numerical fitting or optimizer work.",
    )


def worker(args):
    policy, current, old, association = authenticate(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="failed",
        phase=args.phase,
        new_inference_binding_sha256=args.binding_sha256,
        source_association=association,
        fits=0,
        initializers=0,
    )
    try:
        if args.phase == "synthetic":
            result = synthetic(policy, current, output)
        else:
            result = lifecycle(
                policy, current, old["lifecycle"], output, args.phase == "lifecycle64"
            )
        authenticate(args)
        report.update(status="complete", result=result)
    except BaseException as error:
        report.update(
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        report["files"] = inventory(output)
        write(output / "result.json", report)


def run(args):
    _policy, current, _, association = authenticate(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed",
        new_inference_binding_sha256=args.binding_sha256,
        policy_sha256=args.policy_sha256,
        source_association=association,
        fits=0,
        initializers=0,
        simulations=0,
        stages=[],
        public_promotion=False,
    )
    try:
        for phase in ("synthetic", "lifecycle32", "lifecycle64"):
            command = [
                current["interpreter"],
                str(Path(__file__).resolve()),
                "worker",
                "--phase",
                phase,
                "--policy",
                str(Path(args.policy).resolve()),
                "--policy-sha256",
                args.policy_sha256,
                "--binding",
                str(Path(args.binding).resolve()),
                "--binding-sha256",
                args.binding_sha256,
                "--output",
                str(output / phase),
            ]
            env = dict(
                os.environ,
                PYTHONPATH=str(Path(current["public_root"]) / "src"),
                SCIPY_ARRAY_API="1",
                JAX_PLATFORMS="cpu",
            )
            env.pop("JAX_ENABLE_X64", None)
            if phase == "lifecycle64":
                env["JAX_ENABLE_X64"] = "1"
            write(
                output / (phase + "-command.json"),
                dict(
                    argv=command,
                    cwd=current["public_root"],
                    environment={
                        k: env.get(k)
                        for k in (
                            "PYTHONPATH",
                            "SCIPY_ARRAY_API",
                            "JAX_PLATFORMS",
                            "JAX_ENABLE_X64",
                        )
                    },
                    timeout_s=14400,
                ),
            )
            start = time.monotonic()
            with (output / (phase + ".log")).open("x") as log:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=current["public_root"],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=14400,
                        check=False,
                    )
                    code, timed_out = completed.returncode, False
                except subprocess.TimeoutExpired:
                    code, timed_out = None, True
            stage = dict(
                phase=phase,
                returncode=code,
                timed_out=timed_out,
                elapsed_s=time.monotonic() - start,
                log_sha256=sha(output / (phase + ".log")),
            )
            result["stages"].append(stage)
            write(output / (phase + "-exit.json"), stage)
            assert code == 0, "failed worker preserved; no automatic retry"
        authenticate(args)
        result["status"] = "complete"
        result["inference_regression_passed"] = True
    except BaseException as error:
        result.update(
            error=str(error),
            error_type=type(error).__name__,
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        result["files"] = inventory(output)
        write(output / "run.json", result)
    print(
        json.dumps(
            dict(status=result["status"], run_sha256=sha(output / "run.json"), fits=0)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker"))
    for key in ("policy", "policy-sha256", "binding", "binding-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--phase", choices=("synthetic", "lifecycle32", "lifecycle64"))
    args = parser.parse_args()
    (run if args.action == "run" else worker)(args)


if __name__ == "__main__":
    main()
