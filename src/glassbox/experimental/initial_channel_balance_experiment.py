"""Fixed initial-channel weighting with byte-exact historical reference fits."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import time
from functools import lru_cache
from pathlib import Path

import jax

from glassbox.experimental import affine_anchored_experiment as previous
from glassbox.experimental import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/initial-channel-balance-v1.json"
PROTOCOL_SHA256 = "822a28d5a20596026851ec9e5f64f9c304e09a68b67045b44bc3ee3633d056dc"
SIMULATORS = ("crazyflow", "cascade")
REFERENCE_ARMS = {
    "baseline": "baseline",
    "quadratic": "quadratic",
    "anchored": "candidate",
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


def protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen initial-channel-balance protocol changed")
    return read_json(PROTOCOL)


def _check_inherited_sources(p):
    previous._check_inherited_sources(p)
    for key in ("prior_anchored_protocol", "prior_anchored_result"):
        entry = p[key]
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("historical evidence changed: " + key)
    entry = p["diagnostic_basis"]["training_diagnosis"]
    if digest(Path(entry["path"])) != entry["sha256"]:
        raise ValueError("historical training diagnosis changed")


def _ablation_sources():
    paths = [
        ROOT / directory / (prefix + stem + ".py")
        for stem in (
            "initial_channel_balance_model",
            "initial_channel_balance_matching",
            "initial_channel_balance_checkpoints",
            "initial_channel_balance_decision",
            "initial_channel_balance_experiment",
        )
        for directory, prefix in (("src/glassbox/experimental", ""), ("tests", "test_"))
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("channel-balance implementation or tests missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    """Bind a private generator/evaluator without modifying historical globals."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._initial_channel_balance_interaction", inherited.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL, module.PROTOCOL_SHA256 = PROTOCOL, PROTOCOL_SHA256
    module.protocol = protocol
    module.ARMS, module.PREDICTORS = ARMS, PREDICTORS
    private_flight = module.flight()
    original_check = private_flight.check_sources

    def check(p):
        _check_inherited_sources(p)
        runtime = original_check(p)
        runtime["initial_channel_balance_implementation"] = _ablation_sources()
        runtime["checkpoint_source_sha256"] = previous._checkpoint_sources(
            p, runtime["initial_channel_balance_implementation"]
        )
        return runtime

    private_flight.check_sources = check
    module._original_predict = module.Predictors.predict
    module.check_links = lambda *args, **kwargs: check_links(*args, **kwargs)
    module.Predictors = lambda *args: Predictors(*args)
    module.decision = lambda *args: decision(*args)
    return module


def flight():
    return _implementation().flight()


def check_sources():
    return _implementation().check_sources()


def trusted_previous():
    return _implementation().trusted_previous()


def trusted_anchored():
    """Authenticate the full historical bundle independently of copied metadata."""
    p = protocol()
    entry = p["prior_anchored_bundle"]
    source = Path(entry["path"])
    if digest(source / "run.json") != entry["manifest_sha256"]:
        raise ValueError("prior anchored bundle differs from trusted anchor")
    manifest = verify_files(source, "run.json")
    if manifest["protocol_sha256"] != p["prior_anchored_protocol"]["sha256"]:
        raise ValueError("prior anchored protocol differs from trusted source")
    if manifest["runtime"] != previous.check_sources():
        raise ValueError("historical reference source/runtime differs")
    return source


def _model_classes():
    from glassbox.experimental.affine_anchored_model import (
        CandidateDynamics as Anchored,
    )
    from glassbox.experimental.initial_channel_balance_model import (
        CandidateDynamics as Balanced,
    )
    from glassbox.experimental.state_quadratic_model import (
        CandidateDynamics as Quadratic,
    )
    from glassbox.learner import LearnedDynamics

    return dict(
        baseline=LearnedDynamics,
        quadratic=Quadratic,
        anchored=Anchored,
        candidate=Balanced,
    )


def _reference_sources(simulator):
    from .excited_public_data import _files

    source = trusted_anchored()
    manifest = read_json(source / "run.json")
    result = {}
    for arm, old_arm in REFERENCE_ARMS.items():
        entry = protocol()["trusted_comparator_revisions"][simulator][arm]
        directory = source / simulator / old_arm
        if entry["source_relative_directory"] != f"{simulator}/{old_arm}":
            raise ValueError("historical reference association differs")
        if digest(directory / "seal.json") != entry["source_seal_sha256"]:
            raise ValueError("historical reference seal differs")
        seal = verify_files(directory, "seal.json")
        files = _files(directory, seal, exact=True)
        identities = {name: digest(directory / name) for name in files}
        for name, sha in identities.items():
            if manifest["files"].get(f"{simulator}/{old_arm}/{name}") != sha:
                raise ValueError(
                    "historical reference payload is outside trusted bundle"
                )
        if (
            entry["source_relative_path"] != f"{simulator}/{old_arm}/model.npz"
            or identities["model.npz"] != entry["file_sha256"]
            or identities["checkpoints/manifest.json"]
            != entry["source_checkpoint_manifest_sha256"]
        ):
            raise ValueError("historical model/checkpoint anchor differs")
        outcome = read_json(directory / "outcome.json")
        if (
            outcome["status"] != "complete"
            or outcome["model_available"] is not True
            or outcome["reason"] is not None
        ):
            raise ValueError("frozen historical reference is unavailable")
        result[arm] = dict(
            source_arm=old_arm, directory=directory, files=identities, anchors=entry
        )
    return source, manifest, result


def _reference_manifest(simulator, output, runtime, source_manifest, references):
    base = Path(output) / simulator
    return dict(
        format="glassbox-historical-reference-import-v1",
        simulator=simulator,
        runtime=runtime,
        historical_runtime=source_manifest["runtime"],
        source_bundle_sha256=protocol()["prior_anchored_bundle"]["manifest_sha256"],
        data_seal=digest(base / "data/seal.json"),
        excitation_seal=digest(base / "excitation/seal.json"),
        excitation_reuse_sha256=digest(base / "reuse.json"),
        arms={
            arm: dict(
                source_arm=item["source_arm"],
                evaluation_arm=arm,
                anchors=item["anchors"],
                files=item["files"],
            )
            for arm, item in references.items()
        },
        fresh_reference_fits=0,
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
    association = _reference_manifest(simulator, output, runtime, manifest, references)
    write_json(base / "reference-reuse.json", association)
    check_reference_import(simulator, output, runtime)
    return association


def check_reference_import(simulator, output, runtime):
    from .excited_public_data import _files

    base = Path(output) / simulator
    _, manifest, references = _reference_sources(simulator)
    for arm, item in references.items():
        directory = base / arm
        seal = verify_files(directory, "seal.json")
        if set(_files(directory, seal, exact=True)) != set(item["files"]):
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
            raise ValueError("imported reference fingerprint differs")
        if model.report != read_json(
            directory / "report.json"
        ) or model.contract != read_json(directory / "contract.json"):
            raise ValueError("imported reference report/contract mirror differs")
    expected = _reference_manifest(simulator, output, runtime, manifest, references)
    if expected != read_json(base / "reference-reuse.json"):
        raise ValueError("reference import association/provenance differs")
    return expected


def compare_calibration(simulator, output):
    return _implementation().compare_calibration(simulator, output)


def generate(simulator, output):
    from .excited_public_data import import_excitation

    trusted_anchored()
    _implementation().generate(simulator, output)
    runtime = check_sources()
    reuse = import_excitation(simulator, output, flight(), protocol(), runtime)
    write_json(Path(output) / simulator / "reuse.json", reuse)
    references = import_references(simulator, output, runtime)
    return dict(excitation=reuse, references=references)


def check_import(simulator, output, runtime):
    from .excited_public_data import check_import as verify_import

    result = verify_import(simulator, output, flight(), protocol(), runtime)
    if result != read_json(Path(output) / simulator / "reuse.json"):
        raise ValueError("excitation reuse evidence differs")
    return result


def data_ready(output):
    runtime = _implementation().data_ready(output)
    for simulator in SIMULATORS:
        check_import(simulator, output, runtime)
        check_reference_import(simulator, output, runtime)
    return runtime


def fit_provenance(simulator, arm, output, runtime):
    if arm != "candidate":
        raise ValueError(
            "only candidate is fitted; references retain historical provenance"
        )
    base = Path(output) / simulator
    return dict(
        runtime=runtime,
        arm=arm,
        simulator=simulator,
        data_seal=digest(base / "data/seal.json"),
        calibration_sha256=digest(base / "calibration.json"),
        imported_excitation_seal=digest(base / "excitation/seal.json"),
        reuse_sha256=digest(base / "reuse.json"),
        reference_reuse_sha256=digest(base / "reference-reuse.json"),
        reference_seals={
            name: digest(base / name / "seal.json") for name in REFERENCE_ARMS
        },
        trusted_anchored_bundle_sha256=protocol()["prior_anchored_bundle"][
            "manifest_sha256"
        ],
        checkpoint_steps=protocol()["checkpoint_diagnostics"]["steps"],
        fitting_collection="imported_public_exact_caches",
        objective="fixed_inverse_initial_training_channel_mse",
        fresh_optimizer_trajectories=1,
    )


def anchor_reference(simulator, output):
    base = Path(output) / simulator / "anchored/checkpoints"
    outer = read_json(base / "manifest.json")
    entry = protocol()["trusted_comparator_revisions"][simulator]["anchored"]
    if digest(base / "manifest.json") != entry["source_checkpoint_manifest_sha256"]:
        raise ValueError("imported anchored checkpoint anchor differs")
    row = outer["checkpoints"][0]
    if row["step"] != 0:
        raise ValueError("imported anchored initial checkpoint is missing")
    return dict(
        path=base / row["snapshot"],
        sha256=outer["files"][row["snapshot"]],
        source_bundle_sha256=protocol()["prior_anchored_bundle"]["manifest_sha256"],
        source_checkpoint_manifest_sha256=entry["source_checkpoint_manifest_sha256"],
        source_relative_path=f"{simulator}/candidate/checkpoints/{row['snapshot']}",
    )


def compare_candidate(model, simulator, output):
    from .initial_channel_balance_matching import check_matched_models

    base = Path(output) / simulator
    classes = _model_classes()
    return check_matched_models(
        classes["baseline"].load(base / "baseline/model.npz"),
        model,
        classes["quadratic"].load(base / "quadratic/model.npz"),
        classes["anchored"].load(base / "anchored/model.npz"),
    )


def _outcome_expectation(outcome, model, arm, recipe):
    available = outcome["model_available"]
    if (
        type(available) is not bool
        or available != (outcome["status"] == "complete")
        or (available and outcome["reason"] is not None)
        or (
            not available
            and (outcome["status"] != "fit_failure" or not outcome["reason"])
        )
    ):
        raise ValueError("fit outcome availability/status/reason differs")
    optimization = None if model is None else model.report["optimization"]
    return dict(
        arm=arm,
        recipe=recipe,
        expected_steps=protocol()["checkpoint_diagnostics"]["steps"],
        completed=available,
        failure=outcome["reason"],
        status=outcome["status"],
        selected_step=None if optimization is None else optimization["selected_step"],
        trace=None if optimization is None else optimization["trace"],
    )


def check_checkpoints(output, simulator, arm, *, replay=False):
    if arm not in ARMS:
        raise ValueError("unknown checkpoint arm")
    base = Path(output) / simulator
    directory = base / arm
    outcome = read_json(directory / "outcome.json")
    model = (
        _model_classes()[arm].load(directory / "model.npz")
        if outcome["model_available"]
        else None
    )
    if arm in REFERENCE_ARMS:
        from . import affine_anchored_checkpoints as checkpoints

        original_arm = REFERENCE_ARMS[arm]
        source = trusted_anchored()
        old_runtime = read_json(source / "run.json")["runtime"]
        arguments = previous._checkpoint_arguments(
            simulator, original_arm, source, old_runtime, model
        )
        expectation = _outcome_expectation(
            outcome, model, original_arm, previous._expected_recipe(original_arm)
        )
        # The copied evidence keeps the original arm, source map and fit provenance.
    else:
        from . import initial_channel_balance_checkpoints as checkpoints
        from .initial_channel_balance_model import RECIPE

        reference = _model_classes()["baseline"].load(base / "baseline/model.npz")
        runtime = check_sources()
        arguments = dict(
            train=reference._train,
            development=reference._development,
            provenance=fit_provenance(simulator, arm, output, runtime),
            source_sha256=runtime["checkpoint_source_sha256"],
            selected_model=None if model is None else model._model,
            anchor_reference=anchor_reference(simulator, output),
        )
        expectation = _outcome_expectation(outcome, model, arm, RECIPE)
        expectation["unweighted_trace"] = (
            None
            if model is None
            else model.report["optimization"]["unweighted_development_trace"]
        )
        expectation["objective"] = (
            None if model is None else model.report["optimization"]["objective"]
        )
    checker = (
        checkpoints.replay_checkpoints if replay else checkpoints.check_checkpoint_links
    )
    return checker(
        directory / "checkpoints",
        expected_sha256=outcome["checkpoint_manifest_sha256"],
        expectation=expectation,
        **arguments,
    )


def fit_arm(simulator, arm, output):
    if arm != "candidate":
        raise ValueError("only candidate may run a fresh optimizer")
    from glassbox.experimental.initial_channel_balance_model import (
        RECIPE,
        CandidateFitError,
        fit_candidate,
    )
    from glassbox.learner import RECIPE as PUBLIC_RECIPE

    from .initial_channel_balance_checkpoints import capture_fit, save_checkpoints

    if PUBLIC_RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("public recipe differs from frozen baseline")
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
    except (ValueError, CandidateFitError) as error:
        if not isinstance(
            error, CandidateFitError
        ) and not inherited._nonfinite_fit_error(error):
            raise
        reason, status = str(error), "fit_failure"
    fit_wall_time = time.perf_counter() - start
    if model is not None:
        write_json(
            directory / "matching.json", compare_candidate(model, simulator, output)
        )
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    diagnostics_start = time.perf_counter()
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
    diagnostics_wall_time = time.perf_counter() - diagnostics_start
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
            checkpoint_diagnostics_wall_time_s=diagnostics_wall_time,
        ),
    )
    print(f"fit complete {simulator}/{arm}: status={status}", flush=True)


class Predictors:
    def __init__(self, output, simulator):
        self.models = {}
        for arm, cls in _model_classes().items():
            base = Path(output) / simulator / arm
            verify_files(base, "seal.json")
            self.models[arm] = (
                cls.load(base / "model.npz")
                if read_json(base / "outcome.json")["model_available"]
                else None
            )
        self.functions = {
            arm: jax.jit(model.predict) if model is not None else None
            for arm, model in self.models.items()
        }

    def predict(self, query, arrays):
        return _implementation()._original_predict(self, query, arrays)

    def envelope(self, arm):
        model = self.models.get(arm)
        return model.envelope() if model is not None else None


def check_links(output, simulator, *, evaluation=False):
    runtime = check_sources()
    base = Path(output) / simulator
    if verify_files(base / "data", "seal.json")["runtime"] != runtime:
        raise ValueError("data runtime/source identity differs")
    if read_json(base / "calibration.json") != compare_calibration(simulator, output):
        raise ValueError("calibration comparison differs")
    check_import(simulator, output, runtime)
    check_reference_import(simulator, output, runtime)
    directory = base / "candidate"
    verify_files(directory, "seal.json")
    if read_json(directory / "start.json") != fit_provenance(
        simulator, "candidate", output, runtime
    ):
        raise ValueError("candidate fit provenance differs")
    for arm in ARMS:
        check_checkpoints(output, simulator, arm)
    if read_json(directory / "outcome.json")["model_available"]:
        model = _model_classes()["candidate"].load(directory / "model.npz")
        if model.report != read_json(
            directory / "report.json"
        ) or model.contract != read_json(directory / "contract.json"):
            raise ValueError("candidate report/contract mirror differs")
        if compare_candidate(model, simulator, output) != read_json(
            directory / "matching.json"
        ):
            raise ValueError("candidate matching evidence differs")
    if evaluation:
        seal = verify_files(base / "evaluation", "seal.json")
        if seal["data_seal"] != digest(base / "data/seal.json") or seal["models"] != {
            arm: digest(base / arm / "seal.json") for arm in ARMS
        }:
            raise ValueError("evaluation data/model provenance differs")


def validate_rows(simulator, output, rows):
    return _implementation().validate_rows(simulator, output, rows)


def evaluate(simulator, output, *, replay=False):
    return _implementation().evaluate(simulator, output, replay=replay)


def decision(output):
    from .initial_channel_balance_decision import reduce

    summaries, rows = {}, {}
    for simulator in SIMULATORS:
        check_links(output, simulator, evaluation=True)
        base = Path(output) / simulator / "evaluation"
        summaries[simulator] = read_json(base / "summary.json")
        rows[simulator] = read_json(base / "rows.json")
        validate_rows(simulator, output, rows[simulator])
        if inherited.aggregate(rows[simulator]) != summaries[simulator]:
            raise ValueError("summary differs from metric rows")
    return reduce(summaries, protocol(), rows=rows)


def finalize(output):
    output = Path(output)
    if (output / "run.json").exists() or (output / "decision.json").exists():
        raise FileExistsError("frozen result already exists")
    write_json(output / "decision.json", decision(output))
    sha = freeze_files(
        output,
        "run.json",
        dict(
            protocol_sha256=digest(PROTOCOL),
            scope="Frozen initial-channel-balance-v1; two fresh candidate fits and six imported reference fits",
            runtime=check_sources(),
        ),
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    return _implementation().verify_bundle(output, expected_sha256)


def replay(simulator, output, expected_sha256):
    from .excited_public_data import replay_import

    verify_bundle(output, expected_sha256)
    result = dict(
        physics=flight().replay_data(simulator, output),
        calibration=compare_calibration(simulator, output),
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
        raise ValueError("fresh decision or bootstrap differs")
    result["decision_exact"] = True
    write_json(Path(output) / simulator / "replay.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "fit", "evaluate", "finalize", "replay")
    )
    parser.add_argument("simulator", choices=SIMULATORS)
    parser.add_argument("--arm", choices=("candidate",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.stage == "fit":
        if args.arm is None:
            parser.error("--arm required for fit")
        fit_arm(args.simulator, args.arm, args.output)
    elif args.stage == "finalize":
        finalize(args.output)
    elif args.stage == "replay":
        replay(args.simulator, args.output, args.expected_bundle_sha)
    else:
        globals()[args.stage](args.simulator, args.output)


if __name__ == "__main__":
    main()
