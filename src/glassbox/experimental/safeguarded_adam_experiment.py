"""Frozen safeguarded-Adam study using pinned collection and evaluation helpers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from glassbox._learner_arrays import array_fingerprint
from glassbox.experimental import excited_public_matching as _shared_matching
from glassbox.experimental import initial_channel_balance_matching as _balanced_matching
from glassbox.experimental.excited_bilinear_decision import _angular_repair, _comparison
from glassbox.experimental.initial_channel_balance_decision import _full_cohorts
from glassbox.experimental.state_input_model import _window_fingerprint
from glassbox.experimental.two_simulator_metrics import aggregate

from . import initial_channel_balance_experiment as previous
from . import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/safeguarded-adam-v1.json"
PROTOCOL_SHA256 = "40252fccb98101143c45775615a3ef33c73018c374daa48470f19154cf4cd1c4"
SIMULATORS = ("crazyflow", "cascade")
REFERENCE_ARMS = {
    "baseline": "baseline",
    "quadratic": "quadratic",
    "balanced": "candidate",
}
ARMS = (*REFERENCE_ARMS, "candidate")
PREDICTORS = (*ARMS, "hold")
digest, read_json, write_json = (
    inherited.digest,
    inherited.read_json,
    inherited.write_json,
)
load_arrays, save_arrays = inherited.load_arrays, inherited.save_arrays
array_equal, freeze_files, verify_files = (
    inherited.array_equal,
    inherited.freeze_files,
    inherited.verify_files,
)


def delta_protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen safeguard protocol changed")
    value = read_json(PROTOCOL)
    for name in ("base_protocol", "prior_result"):
        entry = value[name]
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("historical evidence changed: " + name)
    return value


def protocol():
    """Resolve only the declared parent fields; the frozen delta is authoritative."""
    delta = delta_protocol()
    base = previous.protocol()
    previous._check_inherited_sources(base)
    value = {
        key: deepcopy(base[key]) for key in delta["resolution"]["inherit_unchanged"]
    }
    value.update(
        {
            key: deepcopy(item)
            for key, item in base.items()
            if key.startswith(("prior_", "previous_"))
        }
    )
    for key in (
        "id",
        "status",
        "base_commit",
        "named_gap",
        "scope",
        "prior_reference_bundle",
        "prior_result",
        "optimizer_evidence",
        "verification",
    ):
        value[key] = deepcopy(delta[key])
    value["counts"] = deepcopy(base["counts"])
    value["recordings"] = deepcopy(base["recordings"])
    for row in value["recordings"]:
        if row["role"] == "test":
            if row["id"].rsplit("/", 1)[-1] != "test-" + str(row["seed"]):
                raise ValueError("historical test identity is malformed")
            row["seed"] += delta["cohort"]["test_seed_increment_from_base"]
            row["id"] = row["id"].rsplit("-", 1)[0] + "-" + str(row["seed"])
    current_tests = [row for row in value["recordings"] if row["role"] == "test"]
    if len(current_tests) != delta["cohort"]["counts"]["fresh_test_parents"]:
        raise ValueError("fresh test roster count differs")
    ids, seeds = ({row[key] for row in current_tests} for key in ("id", "seed"))
    seed_identities = {(row["simulator"], row["seed"]) for row in current_tests}
    if len(ids) != len(current_tests) or len(seed_identities) != len(current_tests):
        raise ValueError("duplicate fresh test identity or simulator seed")
    historical = [base]
    for key, entry in base.items():
        if key == "previous_protocol" or (
            key.startswith("prior_") and key.endswith("_protocol")
        ):
            path = ROOT / entry["path"]
            if digest(path) != entry["sha256"]:
                raise ValueError("historical cohort protocol changed")
            historical.append(read_json(path))
    for old_protocol in historical:
        old_tests = [row for row in old_protocol["recordings"] if row["role"] == "test"]
        if ids & {row["id"] for row in old_tests} or seeds & {
            row["seed"] for row in old_tests
        }:
            raise ValueError(
                "fresh test identities or seeds overlap an inspected cohort"
            )
    value["fresh_confirmation"] = deepcopy(delta["cohort"])
    value["imported_references"] = deepcopy(delta["imported_references"])
    value["trusted_comparator_revisions"] = deepcopy(
        delta["imported_references"]["trusted_comparator_revisions"]
    )
    value["fitting"] = deepcopy(delta["fitting"])
    value["fitting"]["generic_recipe"] = deepcopy(base["fitting"]["generic_recipe"])
    value["fitting"]["normalization"] = deepcopy(base["fitting"]["normalization"])
    value["fitting"]["arms"] = list(ARMS)
    value["fitting"]["mechanism"] = {
        key: deepcopy(base["fitting"]["mechanism"][key])
        for key in ("formula", "initial_prediction", "initialization", "numerics")
    }
    value["checkpoint_diagnostics"] = {
        **deepcopy(base["checkpoint_diagnostics"]),
        **deepcopy(delta["checkpoint_diagnostics"]),
    }
    diagnostics = value["checkpoint_diagnostics"]
    diagnostics["source_binding"] = delta["verification"]["source_binding"]
    diagnostics["initialization"] = delta["fitting"]["initialization"]
    diagnostics["observer_parity"] = delta["verification"]["tests"]
    diagnostics["capture"] = (
        base["checkpoint_diagnostics"]["capture"].replace(
            "100update check", "100completed-proposal check"
        )
        + " "
        + delta["optimizer_evidence"]["observation"]
    )
    diagnostics["replay_integrity"] = delta["verification"]["replay"]
    value["decision"] = {
        key: deepcopy(base["decision"][key])
        for key in ("aggregation", "metrics", "tails", "uncertainty", "availability")
    }
    value["decision"].update(deepcopy(delta["decision"]))
    value["decision"]["accept_research_candidate"] = deepcopy(
        base["decision"]["accept_research_candidate"]
    )
    value["decision"]["comparisons"] = {
        name: {
            **deepcopy(delta["decision"]["guards_for_every_required_comparison"]),
            **deepcopy(policy),
        }
        for name, policy in delta["decision"]["comparisons"].items()
    }
    value["decision"]["angular_repair"] = deepcopy(
        delta["decision"]["angular_retention"]
    )
    value["decision"]["public_promotion"] = delta["decision"]["qualification"]
    value["inherited_source_sha256"].update(previous._ablation_sources())
    value["inherited_source_sha256"][delta["base_protocol"]["path"]] = delta[
        "base_protocol"
    ]["sha256"]
    return value


def resolved_digest():
    return hashlib.sha256(
        (
            json.dumps(protocol(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()


def resolved_protocol(output, *, create=False):
    path = Path(output) / "resolved-protocol.json"
    if create and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".resolved-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(
                (
                    json.dumps(protocol(), indent=2, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode()
            )
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    if not path.is_file() or digest(path) != resolved_digest():
        raise ValueError("resolved protocol differs from frozen parent/delta")
    return digest(path)


def _sources():
    paths = [
        ROOT / folder / (prefix + "safeguarded_adam_" + stem + ".py")
        for stem in ("model", "checkpoints", "experiment")
        for folder, prefix in (("src/glassbox/experimental", ""), ("tests", "test_"))
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("safeguard implementation or tests missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _load_private(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _implementation():
    module = _load_private(
        "glassbox.experimental._safeguarded_adam_interaction", inherited.__file__
    )
    module.PROTOCOL, module.PROTOCOL_SHA256, module.protocol = (
        PROTOCOL,
        PROTOCOL_SHA256,
        protocol,
    )
    module.ARMS, module.PREDICTORS = ARMS, PREDICTORS
    private_flight = module.flight()
    original_read = private_flight.read_json
    private_flight.read_json = lambda path: (
        protocol() if Path(path) == PROTOCOL else original_read(path)
    )
    original_check = private_flight.check_sources

    def check(p):
        if p != protocol():
            raise ValueError("resolved protocol identity differs")
        runtime = original_check(p)
        for path, expected in p["inherited_source_sha256"].items():
            if digest(ROOT / path) != expected:
                raise ValueError("inherited source changed: " + path)
        runtime["safeguarded_adam_implementation"] = _sources()
        runtime["checkpoint_source_sha256"] = previous.previous._checkpoint_sources(
            p, {**_sources(), str(PROTOCOL.relative_to(ROOT)): PROTOCOL_SHA256}
        )
        runtime["resolved_protocol_sha256"] = resolved_digest()
        return runtime

    private_flight.check_sources = check
    module._original_predict = module.Predictors.predict
    module.check_links = lambda *a, **kw: check_links(*a, **kw)
    module.Predictors = lambda *a: Predictors(*a)
    module.decision = lambda *a: decision(*a)
    return module


@lru_cache(maxsize=1)
def _runner():
    """Reuse unchanged collection/import linkage through private historical globals."""
    module = _load_private(
        "glassbox.experimental._safeguarded_adam_runner", previous.__file__
    )
    module.PROTOCOL, module.PROTOCOL_SHA256, module.protocol = (
        PROTOCOL,
        PROTOCOL_SHA256,
        protocol,
    )
    module.REFERENCE_ARMS, module.ARMS, module.PREDICTORS = (
        REFERENCE_ARMS,
        ARMS,
        PREDICTORS,
    )
    module._implementation = _implementation
    module._model_classes = _model_classes
    module.check_sources = check_sources
    module.check_reference_import = check_reference_import
    module.check_checkpoints = check_checkpoints
    module.fit_provenance = fit_provenance
    module.compare_candidate = compare_candidate
    return module


def flight():
    return _implementation().flight()


def check_sources():
    return _implementation().check_sources()


def trusted_references():
    entry = delta_protocol()["prior_reference_bundle"]
    source = Path(entry["path"])
    if digest(source / "run.json") != entry["manifest_sha256"]:
        raise ValueError("reference bundle root differs from external anchor")
    manifest = verify_files(source, "run.json")
    if manifest["protocol_sha256"] != delta_protocol()["base_protocol"]["sha256"]:
        raise ValueError("reference bundle protocol differs")
    if manifest["runtime"] != previous.check_sources():
        raise ValueError("reference historical source/runtime differs")
    return source, manifest


def _model_classes():
    from ..learner import LearnedDynamics
    from .initial_channel_balance_model import CandidateDynamics as Balanced
    from .safeguarded_adam_model import CandidateDynamics as Safeguarded
    from .state_quadratic_model import CandidateDynamics as Quadratic

    return dict(
        baseline=LearnedDynamics,
        quadratic=Quadratic,
        balanced=Balanced,
        candidate=Safeguarded,
    )


def _reference_sources(simulator):
    from .excited_public_data import _files

    source, manifest = trusted_references()
    entries = delta_protocol()["imported_references"]["trusted_comparator_revisions"][
        simulator
    ]
    result = {}
    for arm, old_arm in REFERENCE_ARMS.items():
        entry = entries[arm]
        directory = source / simulator / old_arm
        if (
            entry["source_relative_directory"] != f"{simulator}/{old_arm}"
            or entry["source_arm"] != old_arm
        ):
            raise ValueError("reference arm association differs")
        if digest(directory / "seal.json") != entry["source_seal_sha256"]:
            raise ValueError("reference arm seal differs")
        files = _files(directory, verify_files(directory, "seal.json"), exact=True)
        identities = {name: digest(directory / name) for name in files}
        if any(
            manifest["files"].get(f"{simulator}/{old_arm}/{name}") != sha
            for name, sha in identities.items()
        ):
            raise ValueError("reference payload outside frozen bundle")
        if (
            identities["model.npz"] != entry["file_sha256"]
            or identities["checkpoints/manifest.json"]
            != entry["source_checkpoint_manifest_sha256"]
        ):
            raise ValueError("reference model/checkpoint differs")
        outcome = read_json(directory / "outcome.json")
        if (
            outcome["status"] != "complete"
            or outcome["model_available"] is not True
            or outcome["reason"] is not None
        ):
            raise ValueError("frozen reference is unavailable")
        result[arm] = dict(
            source_arm=old_arm, directory=directory, files=identities, anchors=entry
        )
    return source, manifest, result


def _reference_manifest(simulator, output, runtime, source_manifest, references):
    base = Path(output) / simulator
    return dict(
        format="glassbox-safeguarded-reference-import-v1",
        simulator=simulator,
        runtime=runtime,
        historical_runtime=source_manifest["runtime"],
        source_bundle_sha256=delta_protocol()["prior_reference_bundle"][
            "manifest_sha256"
        ],
        resolved_protocol_sha256=resolved_protocol(output),
        data_seal=digest(base / "data/seal.json"),
        excitation_seal=digest(base / "excitation/seal.json"),
        excitation_reuse_sha256=digest(base / "reuse.json"),
        fresh_reference_fits=0,
        arms={
            arm: dict(
                source_arm=item["source_arm"],
                evaluation_arm=arm,
                anchors=item["anchors"],
                files=item["files"],
            )
            for arm, item in references.items()
        },
    )


def import_references(simulator, output, runtime):
    base = Path(output) / simulator
    _, manifest, references = _reference_sources(simulator)
    if (base / "reference-reuse.json").exists() or any(
        (base / arm).exists() for arm in REFERENCE_ARMS
    ):
        raise FileExistsError("reference import already exists")
    for arm, item in references.items():
        shutil.copytree(item["directory"], base / arm)
    write_json(
        base / "reference-reuse.json",
        _reference_manifest(simulator, output, runtime, manifest, references),
    )
    return check_reference_import(simulator, output, runtime)


def check_reference_import(simulator, output, runtime):
    from .excited_public_data import _files

    base = Path(output) / simulator
    _, manifest, references = _reference_sources(simulator)
    for arm, item in references.items():
        directory = base / arm
        if set(
            _files(directory, verify_files(directory, "seal.json"), exact=True)
        ) != set(item["files"]):
            raise ValueError("imported reference file roster differs")
        for name, sha in item["files"].items():
            if (
                digest(directory / name) != sha
                or (directory / name).read_bytes()
                != (item["directory"] / name).read_bytes()
            ):
                raise ValueError("imported reference bytes differ: " + arm + "/" + name)
        model = _model_classes()[arm].load(directory / "model.npz")
        if model.fingerprint() != item["anchors"]["revision_fingerprint"]:
            raise ValueError("imported reference revision differs")
        if model.report != read_json(
            directory / "report.json"
        ) or model.contract != read_json(directory / "contract.json"):
            raise ValueError("imported reference mirror differs")
    expected = _reference_manifest(simulator, output, runtime, manifest, references)
    if expected != read_json(base / "reference-reuse.json"):
        raise ValueError("reference reuse association differs")
    return expected


def generate(simulator, output):
    from .excited_public_data import import_excitation

    resolved_protocol(output, create=True)
    trusted_references()
    _implementation().generate(simulator, output)
    runtime = check_sources()
    reuse = import_excitation(simulator, output, flight(), protocol(), runtime)
    write_json(Path(output) / simulator / "reuse.json", reuse)
    return dict(
        excitation=reuse, references=import_references(simulator, output, runtime)
    )


def data_ready(output):
    resolved_protocol(output)
    return _runner().data_ready(output)


def fit_provenance(simulator, arm, output, runtime):
    if arm != "candidate":
        raise ValueError("only candidate has current fitting provenance")
    base = Path(output) / simulator
    return dict(
        runtime=runtime,
        arm=arm,
        simulator=simulator,
        resolved_protocol_sha256=resolved_protocol(output),
        data_seal=digest(base / "data/seal.json"),
        calibration_sha256=digest(base / "calibration.json"),
        imported_excitation_seal=digest(base / "excitation/seal.json"),
        reuse_sha256=digest(base / "reuse.json"),
        reference_reuse_sha256=digest(base / "reference-reuse.json"),
        reference_seals={
            name: digest(base / name / "seal.json") for name in REFERENCE_ARMS
        },
        trusted_reference_bundle_sha256=delta_protocol()["prior_reference_bundle"][
            "manifest_sha256"
        ],
        checkpoint_steps=protocol()["checkpoint_diagnostics"]["steps"],
        fitting_collection="imported_public_exact_caches",
        objective="fixed_inverse_initial_training_channel_mse",
        optimizer="bounded_full_training_loss_safeguarded_adam",
        fresh_optimizer_trajectories=1,
    )


def anchor_reference(simulator, output):
    entry = delta_protocol()["fitting"]["trusted_initial_snapshots"][simulator]
    path = Path(output) / entry["imported_relative_path"]
    outer = Path(output) / simulator / "balanced/checkpoints/manifest.json"
    inner = outer.parent / "anchored/manifest.json"
    if (
        digest(outer) != entry["source_checkpoint_manifest_sha256"]
        or digest(inner) != entry["source_inner_checkpoint_manifest_sha256"]
    ):
        raise ValueError("balanced initial manifest anchor differs")
    manifest = read_json(outer)
    relative = str(path.relative_to(outer.parent))
    if (
        manifest["files"].get(relative) != entry["snapshot_sha256"]
        or digest(path) != entry["snapshot_sha256"]
    ):
        raise ValueError("balanced initial snapshot differs")
    return dict(
        path=path,
        sha256=entry["snapshot_sha256"],
        source_bundle_sha256=delta_protocol()["prior_reference_bundle"][
            "manifest_sha256"
        ],
        source_checkpoint_manifest_sha256=entry["source_checkpoint_manifest_sha256"],
        source_relative_path=entry["source_relative_path"],
    )


def check_checkpoints(output, simulator, arm, *, replay=False):
    if arm not in ARMS:
        raise ValueError("unknown checkpoint arm")
    base = Path(output) / simulator
    if arm in REFERENCE_ARMS:
        # Byte equality is checked before calling the original checker on its own
        # source bundle. Historical arm/source/fit identities stay unchanged.
        source, _, references = _reference_sources(simulator)
        item = references[arm]
        from .excited_public_data import _files

        directory = base / arm
        if set(
            _files(directory, verify_files(directory, "seal.json"), exact=True)
        ) != set(item["files"]):
            raise ValueError("reference checkpoint file roster differs")
        if any(
            (directory / name).read_bytes() != (item["directory"] / name).read_bytes()
            for name in item["files"]
        ):
            raise ValueError("reference checkpoint import differs")
        return previous.check_checkpoints(
            source, simulator, REFERENCE_ARMS[arm], replay=replay
        )
    from . import safeguarded_adam_checkpoints as checkpoints
    from .safeguarded_adam_model import RECIPE

    directory = base / arm
    outcome = read_json(directory / "outcome.json")
    model = (
        _model_classes()[arm].load(directory / "model.npz")
        if outcome["model_available"]
        else None
    )
    expectation = _runner()._outcome_expectation(outcome, model, arm, RECIPE)
    optimization = None if model is None else model.report["optimization"]
    for expected, key in (
        ("unweighted_trace", "unweighted_development_trace"),
        ("objective", "objective"),
        ("safeguard", "safeguard"),
    ):
        expectation[expected] = None if optimization is None else optimization[key]
    reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
    runtime = check_sources()
    checker = (
        checkpoints.replay_checkpoints if replay else checkpoints.check_checkpoint_links
    )
    return checker(
        directory / "checkpoints",
        expected_sha256=outcome["checkpoint_manifest_sha256"],
        expectation=expectation,
        train=reference._train,
        development=reference._development,
        provenance=fit_provenance(simulator, arm, output, runtime),
        source_sha256=runtime["checkpoint_source_sha256"],
        selected_model=None if model is None else model._model,
        anchor_reference=anchor_reference(simulator, output),
    )


def fit_arm(simulator, arm, output):
    from ..learner import RECIPE as PUBLIC_RECIPE
    from .safeguarded_adam_checkpoints import capture_fit, save_checkpoints
    from .safeguarded_adam_model import RECIPE, CandidateFitError, fit_candidate

    if arm != "candidate" or PUBLIC_RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("only frozen candidate fitting is permitted")
    runtime = data_ready(output)
    base = Path(output) / simulator
    for reference_arm in REFERENCE_ARMS:
        check_checkpoints(output, simulator, reference_arm)
    directory = base / arm
    directory.mkdir(parents=True, exist_ok=False)
    provenance = fit_provenance(simulator, arm, output, runtime)
    write_json(directory / "start.json", provenance)
    reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
    start = time.perf_counter()
    model, reason, status = None, None, "complete"
    capture = capture_fit(
        arm,
        provenance,
        source_sha256=runtime["checkpoint_source_sha256"],
        expected_steps=tuple(protocol()["checkpoint_diagnostics"]["steps"]),
    )
    try:
        with capture as captured:
            model = fit_candidate(reference)
    except CandidateFitError as error:
        reason, status = str(error), "fit_failure"
    fit_wall_time = time.perf_counter() - start
    if model is not None:
        write_json(
            directory / "matching.json", compare_candidate(model, simulator, output)
        )
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    started = time.perf_counter()
    evidence = save_checkpoints(
        directory / "checkpoints",
        captured,
        train=reference._train,
        development=reference._development,
        recipe=RECIPE,
        selected_step=None
        if model is None
        else model.report["optimization"]["selected_step"],
        failure=reason,
        selected_model=None if model is None else model._model,
        anchor_reference=anchor_reference(simulator, output),
    )
    write_json(
        directory / "outcome.json",
        dict(
            status=status,
            model_available=model is not None,
            reason=reason,
            checkpoint_manifest_sha256=evidence["manifest_sha256"],
        ),
    )
    freeze_files(
        directory,
        "seal.json",
        dict(
            wall_time_s=time.perf_counter() - start,
            fit_and_capture_wall_time_s=fit_wall_time,
            checkpoint_diagnostics_wall_time_s=time.perf_counter() - started,
        ),
    )
    print(f"fit complete {simulator}/{arm}: status={status}", flush=True)


class Predictors:
    def __init__(self, *args):
        self._instance = _runner().Predictors(*args)

    def predict(self, *args):
        return self._instance.predict(*args)

    def envelope(self, *args):
        return self._instance.envelope(*args)


def check_links(output, simulator, *, evaluation=False):
    resolved_protocol(output)
    return _runner().check_links(output, simulator, evaluation=evaluation)


def evaluate(simulator, output, *, replay=False):
    return _implementation().evaluate(simulator, output, replay=replay)


def validate_rows(simulator, output, rows):
    return _implementation().validate_rows(simulator, output, rows)


def compare_candidate(model, simulator, output):
    base = Path(output) / simulator
    classes = _model_classes()
    return check_matched_models(
        classes["baseline"].load(base / "baseline/model.npz"),
        model,
        classes["quadratic"].load(base / "quadratic/model.npz"),
        classes["balanced"].load(base / "balanced/model.npz"),
        simulator,
    )


def decision(output):
    summaries, rows, optimizations = {}, {}, {}
    for simulator in SIMULATORS:
        check_links(output, simulator, evaluation=True)
        base = Path(output) / simulator
        summaries[simulator] = read_json(base / "evaluation/summary.json")
        rows[simulator] = read_json(base / "evaluation/rows.json")
        validate_rows(simulator, output, rows[simulator])
        outcome = read_json(base / "candidate/outcome.json")
        optimizations[simulator] = (
            read_json(base / "candidate/report.json")["optimization"]
            if outcome["model_available"]
            else None
        )
    return reduce_safeguarded(
        summaries, protocol(), rows=rows, optimization=optimizations["crazyflow"]
    )


def finalize(output):
    output = Path(output)
    if (output / "run.json").exists() or (output / "decision.json").exists():
        raise FileExistsError("frozen result already exists")
    write_json(output / "decision.json", decision(output))
    sha = freeze_files(
        output,
        "run.json",
        dict(
            protocol_sha256=PROTOCOL_SHA256,
            resolved_protocol_sha256=resolved_protocol(output),
            runtime=check_sources(),
            scope="Frozen safeguarded Adam: two fresh fits, six byte-exact imported reference fits",
        ),
    )
    print("trusted bundle SHA256: " + sha, flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    resolved_protocol(output)
    return _implementation().verify_bundle(output, expected_sha256)


def replay(simulator, output, expected_sha256):
    from .excited_public_data import replay_import

    verify_bundle(output, expected_sha256)
    result = dict(
        physics=flight().replay_data(simulator, output),
        calibration=_implementation().compare_calibration(simulator, output),
        imported_excitation=replay_import(
            simulator, output, flight(), protocol(), check_sources()
        ),
        imported_references=check_reference_import(simulator, output, check_sources()),
        checkpoints={
            arm: check_checkpoints(output, simulator, arm, replay=True) for arm in ARMS
        },
        predictions=evaluate(simulator, output, replay=True),
    )
    if decision(output) != read_json(Path(output) / "decision.json"):
        raise ValueError("fresh decision differs")
    result["decision_exact"] = True
    write_json(Path(output) / simulator / "replay.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "fit", "evaluate", "finalize", "replay")
    )
    parser.add_argument("simulator", choices=SIMULATORS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="candidate")
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()
    if args.stage == "fit":
        fit_arm(args.simulator, args.arm, args.output)
    elif args.stage == "finalize":
        finalize(args.output)
    elif args.stage == "replay":
        if not args.expected_bundle_sha:
            parser.error("replay requires --expected-bundle-sha")
        replay(args.simulator, args.output, args.expected_bundle_sha)
    else:
        globals()[args.stage](args.simulator, args.output)


_SAFEGUARD_PAIRS = {
    "safeguarded_public_progress": ("candidate", "baseline"),
    "balanced_progress": ("candidate", "balanced"),
    "quadratic_gain_retention": ("candidate", "quadratic"),
}


def _check_safeguard(optimization, p):
    value, policy = optimization["safeguard"], p["fitting"]["optimizer"]
    steps, scales = policy["proposal_attempts"], policy["scales"]
    fixed = dict(
        id=p["id"],
        acceptance="first_finite_strict_decrease",
        moment_policy="advance_on_every_finite_proposal",
        scales=scales,
        proposal_attempts=steps,
        completed_attempts=steps,
        gradient_proposal_calls=steps,
        full_training_objective_calls_max=policy[
            "extra_full_training_objective_calls_max"
        ],
        fit_wall_time_limit_s=policy["fit_wall_time_limit_s"],
    )
    if any(value.get(key) != expected for key, expected in fixed.items()):
        raise ValueError("safeguard identity, semantics or budget differs")
    counts = value["accepted_scale_counts"]
    if [row["scale"] for row in counts] != scales or any(
        set(row) != {"scale", "attempts"}
        or type(row["attempts"]) is not int
        or row["attempts"] < 0
        for row in counts
    ):
        raise ValueError("safeguard scale counts differ")
    for key in (
        "accepted_attempts",
        "rejected_attempts",
        "maximum_rejection_streak",
        "full_training_objective_calls",
    ):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError("safeguard counts must be nonnegative integers")
    accepted, rejected = value["accepted_attempts"], value["rejected_attempts"]
    calls = (
        1
        + sum((index + 1) * row["attempts"] for index, row in enumerate(counts))
        + len(scales) * rejected
    )
    if (
        accepted + rejected != steps
        or accepted != sum(row["attempts"] for row in counts)
        or value["full_training_objective_calls"] != calls
        or calls > fixed["full_training_objective_calls_max"]
    ):
        raise ValueError("safeguard attempt or objective-call accounting differs")
    streak = value["maximum_rejection_streak"]
    if not (0 <= streak <= rejected) or (rejected == 0) != (streak == 0):
        raise ValueError("safeguard rejection streak differs")
    rows = value["checkpoint_full_training_losses"]
    trace_steps = [row["step"] for row in optimization["trace"]]
    if [row["step"] for row in rows] != trace_steps or any(
        set(row) != {"step", "full_training_loss"} for row in rows
    ):
        raise ValueError("safeguard canonical training checkpoint roster differs")
    losses = np.asarray([row["full_training_loss"] for row in rows])
    selected = trace_steps.index(optimization["selected_step"])
    if (
        not np.isfinite(losses).all()
        or np.any(losses < 0)
        or np.any(np.diff(losses) > 0)
    ):
        raise ValueError("safeguard canonical training loss is invalid or increases")
    if any(
        value[key] != expected
        for key, expected in (
            ("initial_full_training_loss", losses[0]),
            ("final_full_training_loss", losses[-1]),
            ("selected_full_training_loss", losses[selected]),
        )
    ) or ((accepted == 0) != (losses[-1] == losses[0])):
        raise ValueError("safeguard canonical training loss mirrors differ")
    return deepcopy(value)


def check_matched_models(
    public, candidate, quadratic, balanced, simulator, *, candidate_module=None
):
    """Validate trusted inputs and reported mechanics; observer verifies actual execution."""
    if candidate_module is None:
        from glassbox.experimental import safeguarded_adam_model as candidate_module
    if (
        type(candidate) is not candidate_module.CandidateDynamics
        or type(candidate._model) is not candidate_module.BalancedQuadraticSequenceModel
    ):
        raise TypeError("candidate must be the frozen safeguarded Adam class")
    p = protocol()
    recipe = p["fitting"]["generic_recipe"]
    shared = _shared_matching.check_matched_models(public, quadratic)
    if (
        shared["population"] != simulator
        or shared["fitting_recipe"] != recipe
        or shared["roles"] != p["planned_automatic_roles"][simulator]
    ):
        raise ValueError("matched population, public recipe or roles differ")
    references = dict(baseline=public, quadratic=quadratic, balanced=balanced)
    anchors = p["imported_references"]["trusted_comparator_revisions"][simulator]
    for arm, model in references.items():
        if model.fingerprint() != anchors[arm]["revision_fingerprint"]:
            raise ValueError("imported reference differs from trusted revision: " + arm)
    report = candidate.report
    if (
        candidate.recipe != candidate_module.RECIPE
        or report["recipe"] != candidate_module.RECIPE
        or report.get("previous_revision") is not None
    ):
        raise ValueError("candidate recipe or initial-fit identity differs")
    if candidate.contract != public.contract or balanced.contract != public.contract:
        raise ValueError("candidate or balanced consumer contract differs")
    windows = {}
    for attribute, role in (("_train", "training"), ("_development", "development")):
        expected = _window_fingerprint(getattr(public, attribute))
        for arm, model in {**references, "candidate": candidate}.items():
            cache = getattr(model, attribute)
            if (
                _window_fingerprint(cache) != expected
                or model.report[role] != cache.coverage()
            ):
                raise ValueError(arm + " " + role + " cache or coverage differs")
        windows[role] = expected
    norms = array_fingerprint({}, balanced._model.norms)
    if array_fingerprint({}, candidate._model.norms) != norms:
        raise ValueError("candidate feature norms differ")
    shapes = {name: value.shape for name, value in balanced._model.params.items()}
    if {
        name: value.shape for name, value in candidate._model.params.items()
    } != shapes or {
        name: value.shape for name, value in quadratic._model.params.items()
    } != shapes:
        raise ValueError("candidate parameter roster or shape differs")
    for key in ("history_steps", "horizon_steps", "delay_steps"):
        if report[key] != balanced.report[key]:
            raise ValueError("candidate timing report differs")
    if (
        any(
            getattr(candidate._model, key) != getattr(balanced._model, key)
            for key in ("kind", "dt_s", "delay_steps")
        )
        or candidate.history_steps != balanced.history_steps
        or candidate.horizon_steps != balanced.horizon_steps
    ):
        raise ValueError("candidate model timing or kind differs")
    optimization = report["optimization"]
    if (
        array_fingerprint({}, {"error_scale": np.asarray(optimization["error_scale"])})
        != shared["loss_scale_fingerprint"]
    ):
        raise ValueError("candidate original hold scale differs")
    if (
        optimization["mechanism"] != p["id"]
        or optimization["minibatch_seed"] != recipe["seed"] + 10000
    ):
        raise ValueError("candidate mechanism or minibatch seed differs")
    count = sum(value.size for value in candidate._model.params.values())
    budget = _shared_matching._check_optimization(
        candidate, recipe, count, "safeguarded"
    )
    if report["matching"] != balanced.report["matching"]:
        raise ValueError("candidate matched preparation differs")
    objective = _balanced_matching._objective(candidate, recipe)
    _balanced_matching._objective(balanced, recipe)
    if optimization["objective"] != balanced.report["optimization"]["objective"]:
        raise ValueError(
            "candidate fixed objective differs from imported balanced witness"
        )
    safeguard = _check_safeguard(optimization, p)
    return dict(
        experiment=p["id"],
        population=simulator,
        candidate_fingerprint=candidate.fingerprint(),
        reference_fingerprints={
            arm: model.fingerprint() for arm, model in references.items()
        },
        window_fingerprints=windows,
        roles=shared["roles"],
        normalizations_fingerprint=norms,
        original_loss_scale_fingerprint=shared["loss_scale_fingerprint"],
        parameter_count=count,
        contract=candidate.contract,
        fitting_recipe=recipe,
        optimizer=budget,
        objective=objective,
        safeguard=safeguard,
        exact_same_data=True,
        exact_same_numeric_objective=True,
        actual_initial_identity_verified_separately_by_observer=True,
        equal_flops_claimed=False,
        equal_end_to_end_compute_claimed=False,
        scope="Trusted revisions, exact caches/norms/scales/shapes and reported budgets/objective. Prospective observer separately verifies actual initialization, weights and all proposal/acceptance evidence; no fit, gradient, solve or prediction here.",
    )


def optimization_progress(optimization, p):
    policy = p["decision"]["diagnostic_optimization_progress"]
    if optimization is None:
        return dict(
            available=False,
            passed=False,
            reason="candidate fit unavailable",
            policy=deepcopy(policy),
        )
    value = _check_safeguard(optimization, p)
    trace = optimization["trace"]
    selected = optimization["selected_step"]
    if trace[0]["step"] != 0 or len({row["step"] for row in trace}) != len(trace):
        raise ValueError("diagnostic development trace roster differs")
    indexed = {row["step"]: row["validation_rollout_mse"] for row in trace}
    initial_train, final_train = (
        value["initial_full_training_loss"],
        value["selected_full_training_loss"],
    )
    initial_dev, final_dev = indexed[0], indexed[selected]
    if (
        not np.isfinite([initial_train, final_train, initial_dev, final_dev]).all()
        or min(initial_train, initial_dev) <= 0
    ):
        return dict(
            available=False,
            passed=False,
            reason="canonical loss denominator not positive finite",
            policy=deepcopy(policy),
        )
    train_ratio, dev_ratio = final_train / initial_train, final_dev / initial_dev
    checks = dict(
        selected_after_initialization=selected
        > policy["selected_attempt_strictly_greater_than"],
        full_training_reduction=train_ratio
        <= policy["selected_full_training_weighted_loss_over_initial_max"],
        development_reduction=dev_ratio
        <= policy["selected_development_weighted_loss_over_initial_max"],
    )
    return dict(
        available=True,
        passed=all(checks.values()),
        checks=checks,
        selected_step=selected,
        initial_full_training_loss=initial_train,
        selected_full_training_loss=final_train,
        initial_development_loss=initial_dev,
        selected_development_loss=final_dev,
        canonical_training_ratio=train_ratio,
        canonical_development_ratio=dev_ratio,
        policy=deepcopy(policy),
        scope="Canonical observed acceptance and selected development losses only; independent of physical residual criteria.",
    )


def reduce_safeguarded(summaries, p, *, rows, optimization):
    """Frozen physical gates and canonical optimization diagnostic remain separate."""
    if (
        rows is None
        or set(summaries) != set(rows)
        or any(aggregate(values) != summaries[sim] for sim, values in rows.items())
    ):
        raise ValueError("summary differs from full metric rows")
    if set(p["decision"]["comparisons"]) != set(_SAFEGUARD_PAIRS):
        raise ValueError("frozen safeguarded comparison roster differs")
    if any(
        set(row["arm"] for row in values)
        != {"baseline", "quadratic", "balanced", "candidate", "hold"}
        for values in rows.values()
    ):
        raise ValueError("frozen safeguarded five-arm roster differs")
    # Reuse the unchanged five-arm roster checker through a private role view.
    cohort_rows = {
        sim: [
            dict(row, arm="anchored" if row["arm"] == "balanced" else row["arm"])
            for row in values
        ]
        for sim, values in rows.items()
    }
    cohorts = _full_cohorts(cohort_rows, p)
    for cohort in cohorts.values():
        cohort["arms"]["balanced"] = cohort["arms"].pop("anchored")
    comparisons = {
        name: _comparison(rows, p, name, *arms)
        for name, arms in _SAFEGUARD_PAIRS.items()
    }
    target = p["decision"]["angular_retention"]
    if (target["numerator"], target["denominator"]) != ("candidate", "balanced"):
        raise ValueError("angular retention must use imported balanced denominator")
    view = deepcopy(p)
    view["decision"]["angular_repair"] = dict(target, denominator="quadratic")
    angular_rows = {
        sim: [
            dict(row, arm="quadratic" if row["arm"] == "balanced" else "candidate")
            for row in values
            if row["arm"] in ("candidate", "balanced")
        ]
        for sim, values in rows.items()
    }
    angular = _angular_repair(
        {sim: aggregate(values) for sim, values in angular_rows.items()},
        view,
        comparisons["balanced_progress"],
    )
    angular.update(
        denominator_arm="balanced",
        baseline_fields_mean="balanced",
        policy=deepcopy(target),
    )
    hashes = {
        name: result["bootstrap"].get("parent_draws_sha256")
        for name, result in comparisons.items()
    }
    hashes["angular_retention"] = angular["bootstrap"].get("parent_draws_sha256")
    available = {value for value in hashes.values() if value is not None}
    if len(available) > 1:
        raise ValueError("pairwise bootstrap parent draws differ")
    checks = {
        name: result["residual_criteria_pass"] for name, result in comparisons.items()
    }
    checks.update(
        angular_retention=angular["residual_criteria_pass"],
        finite_eligible_predictions=all(
            result["checks"]["finite_eligible_predictions"]
            for result in comparisons.values()
        ),
        matching_planned_queries_and_truth=all(
            result["checks"]["matching_planned_queries_and_truth"]
            for result in comparisons.values()
        ),
    )
    diagnostic = optimization_progress(optimization, p)
    residual = all(checks.values())
    return dict(
        protocol=p["id"],
        comparisons=comparisons,
        **{
            name: result["residual_criteria_pass"]
            for name, result in comparisons.items()
        },
        angular_retention=checks["angular_retention"],
        angular_retention_comparison=angular,
        diagnostic_optimization_progress=diagnostic["passed"],
        optimization_diagnostic=diagnostic,
        residual_criteria_pass=residual,
        numerical_mechanism_criteria_pass=residual and diagnostic["passed"],
        checks=checks,
        cohorts=cohorts,
        bootstrap_parent_draws_sha256=next(iter(available), None),
        bootstrap_comparison_hashes=hashes,
        available_bootstrap_parent_draws_match=bool(available),
        shared_bootstrap_parent_draws_verified=all(hashes.values())
        and len(available) == 1,
        verification_status="requires_integrity_replay_tamper_and_focused_tests",
        public_promotion=False,
        qualification=p["decision"]["public_promotion"],
    )


if __name__ == "__main__":
    main()
