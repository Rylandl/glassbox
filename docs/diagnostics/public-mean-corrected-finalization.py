"""Finish the sealed corrected physical replay without repeating its forecasts."""

import argparse
import copy
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

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


def source_continuity(previous, current):
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
        assert previous[key] == current[key], key
    assert all(
        current["public_source_sha256"].get(path) == digest
        for path, digest in previous["public_source_sha256"].items()
    ), "unchanged corrected worker source inventory"
    return dict(
        original_prediction_commit=previous["implementation_commit"],
        finalization_commit=current["implementation_commit"],
        unchanged_prior_bound_sources=len(previous["public_source_sha256"]),
    )


def compare_decision(actual, saved, expected_stages):
    assert saved["input_stages"] == expected_stages, (
        "original decision input-stage association"
    )
    scientific = {key: value for key, value in saved.items() if key != "input_stages"}
    assert actual == scientific, "every original scientific decision field"
    return dict(
        original_input_stage_association_exact=True,
        all_scientific_decision_fields_exact=True,
        original_input_stages=copy.deepcopy(expected_stages),
        compared_fields=sorted(scientific),
    )


def authenticate(args):
    assert sha(args.policy) == args.policy_sha256
    policy = read(args.policy)
    assert sha(__file__) == policy["driver_sha256"]
    from glassbox.experimental.public_mean_implementation import verify

    current = verify(args.binding, args.binding_sha256)
    assert (
        Path(__file__).resolve()
        == Path(current["public_root"]) / policy["driver_relative"]
    )
    previous = verify(
        policy["prefix_binding"]["path"], policy["prefix_binding"]["sha256"]
    )
    association = source_continuity(previous, current)
    bridge_path = Path(current["public_root"]) / policy["physical_bridge_relative"]
    assert sha(bridge_path) == policy["physical_bridge_sha256"]
    spec = importlib.util.spec_from_file_location(
        "frozen_corrected_physical_bridge", bridge_path
    )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    bridge_args = SimpleNamespace(
        policy=Path(current["public_root"]) / policy["physical_policy_relative"],
        policy_sha256=policy["physical_policy_sha256"],
        binding=args.binding,
        binding_sha256=args.binding_sha256,
        output=args.output,
    )
    physical_policy, validated, _, pe = bridge.authenticate(bridge_args)
    assert validated == current
    return (
        policy,
        current,
        previous,
        association,
        bridge,
        bridge_args,
        physical_policy,
        pe,
    )


def prefix_evidence(policy, previous, physical_policy, pe):
    root = Path(policy["failed_prefix"]["root"])
    seal = pe.sealed(root, "seal.json", policy["failed_prefix"]["sha256"])
    assert seal["files"] == pe.inventory(root)
    assert seal["phase"] == "run" and not (root / "result.json").exists()
    assert sha(root / "failure.json") == policy["failed_prefix"]["failure_sha256"]
    assert read(root / "failure.json")["type"] == "AssertionError"
    assert not (root / "consumer").exists() and not (root / "decision.json").exists()
    reports = {}
    for sim in physical_policy["simulators"]:
        base = root / sim
        assert read(base / "old-validation-execution/exit.json")["returncode"] == 0
        reports[sim] = {}
        for role in ("public32", "public64"):
            assert read(base / (role + "-execution") / "exit.json")["returncode"] == 0
            child = pe.sealed(
                base / role, "seal.json", seal["files"][f"{sim}/{role}/seal.json"]
            )
            assert (
                child["files"] == pe.inventory(base / role) and child["phase"] == role
            )
            report = read(base / role / "result.json")
            assert report["status"] == "complete" and report["role"] == role
            assert (
                report["new_inference_binding_sha256"]
                == policy["prefix_binding"]["sha256"]
            )
            assert (
                report["old_fit_binding_sha256"]
                == physical_policy["original_fit_binding_sha256"]
            )
            assert (
                report["original_evaluation_sha256"]
                == physical_policy["simulators"][sim]["evaluation_sha256"]
            )
            assert (
                report["original_fit_manifest_sha256"]
                == physical_policy["simulators"][sim]["flight_fit_sha256"]
            )
            assert (
                report["fits"] == report["initializers"] == report["simulations"] == 0
            )
            pe.recorded_sources(
                {"imported_sources": report["actual_imported_sources"]}, previous, role
            )
            reports[sim][role] = report
    return root, reports, len(seal["files"])


def run(args):
    policy, current, previous, association, bridge, bridge_args, physical_policy, pe = (
        authenticate(args)
    )
    prefix, reports, payload_count = prefix_evidence(
        policy, previous, physical_policy, pe
    )
    rows, queries, reductions = {}, {}, {}
    verified_references = set()
    for sim in physical_policy["simulators"]:
        original, inputs = bridge.old_inputs(pe, physical_policy, sim)
        if inputs["reference_root"] not in verified_references:
            pe.sealed(inputs["reference_root"], "run.json", pe.REFERENCE_SHA256)
            verified_references.add(inputs["reference_root"])
        from glassbox.experimental.public_mean_physics import verify_data

        data = verify_data(inputs["data_root"], inputs["data_sha256"])
        assert (
            data["implementation_sha256"]
            == physical_policy["original_fit_binding_sha256"]
        )
        assert data["simulator"] == sim
        fit = pe.sealed(
            inputs["flight_fit_root"], "manifest.json", inputs["flight_fit_sha256"]
        )
        assert (
            fit["implementation_manifest_sha256"]
            == physical_policy["original_fit_binding_sha256"]
        )
        assert fit["status"] == "complete" and fit["simulator"] == sim
        for role in ("public32", "public64"):
            assert reports[sim][role]["model_sha256"] == sha(
                Path(inputs["flight_fit_root"]) / "model.npz"
            )
        qualification = read(
            Path(current["public_root"])
            / "docs/harness/public-mean-qualification-v1.json"
        )
        resolved = pe.resolved(
            inputs["reference_root"], qualification, current["public_root"]
        )
        reduced = pe.score(inputs["data_root"], prefix / sim, sim, resolved)
        for name, value in reduced.items():
            assert value == read(prefix / sim / (name + ".json")), (
                "saved corrected " + name
            )
            assert value == read(original / (name + ".json")), "saved original " + name
        rows[sim] = reduced["rows"]
        _, queries[sim], _ = pe._planned(inputs["data_root"], sim, resolved)
        reductions[sim] = dict(
            rows=len(reduced["rows"]),
            directions=len(reduced["directions"]),
            rows_directions_and_summary_exact=True,
            corrected_source_files={
                name: sha(prefix / sim / (name + ".json")) for name in reduced
            },
        )
    actual = pe.reduce_decision(rows, resolved, qualification, queries=queries)
    anchor = physical_policy["old_decision"]
    assert sha(anchor["path"]) == anchor["sha256"]
    expected_stages = {
        sim: dict(path=row["evaluation_root"], seal_sha256=row["evaluation_sha256"])
        for sim, row in physical_policy["simulators"].items()
    }
    comparison = compare_decision(actual, read(anchor["path"]), expected_stages)
    comparison.update(
        original_decision_sha256=anchor["sha256"],
        corrected_prediction_prefix_sha256=policy["failed_prefix"]["sha256"],
    )
    write(args.output / "decision.json", actual)
    write(args.output / "decision-comparison.json", comparison)
    write(args.output / "reductions.json", reductions)
    consumer = bridge.replay_consumer(bridge_args, physical_policy, current, pe)
    counts = bridge.check_counts(
        reports, rows, consumer, physical_policy["expected_counts"]
    )
    authenticate(args)
    assert pe.inventory(prefix) == read(prefix / "seal.json")["files"]
    return dict(
        status="complete",
        physical_and_consumer_regression_passed=True,
        named_checks=dict(
            prefix_external_anchor_and_inventory=True,
            unchanged_corrected_inference_source=True,
            original_decision_input_stages=True,
            all_scientific_decision_fields=True,
            saved_physical_reductions=True,
            current_consumer_exact_168_arrays=True,
        ),
        source_association=association,
        prior_failed_prefix=policy["failed_prefix"],
        prior_prediction_binding=policy["prefix_binding"],
        finalization_binding_sha256=args.binding_sha256,
        policy_sha256=args.policy_sha256,
        verified_prefix_payloads=payload_count,
        checked_counts=counts,
        consumer=consumer,
        new_fits=0,
        new_initializers=0,
        new_simulations=0,
        new_flight_forecasts=0,
        new_development_forecasts=0,
        original_fitting_operations=32,
        public_promotion=False,
        scope="Saved-only finalization of original corrected forecasts, followed by the previously unexecuted current-source Dart consumer. Original failed attempt, fitting provenance and numerical gates remain unchanged.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for key in ("policy", "policy-sha256", "binding", "binding-sha256"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    outcome = dict(status="failed", new_fits=0, new_simulations=0)
    try:
        outcome = run(args)
    except BaseException as error:
        outcome.update(
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        from glassbox.experimental.public_mean_physical_evaluation import inventory

        outcome["files"] = inventory(args.output, excluding="run.json")
        write(args.output / "run.json", outcome)


if __name__ == "__main__":
    main()
