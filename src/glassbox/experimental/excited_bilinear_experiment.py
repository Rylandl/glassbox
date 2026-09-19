"""Frozen bilinear ablation on identical excited recordings with observed checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import jax

from glassbox.experimental import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/excited-bilinear-ablation-v1.json"
PROTOCOL_SHA256 = "20d31d5f1553b8f7f910c5110cb52a4221613df8c6fc078d43481e45325e7a21"
SIMULATORS = ("crazyflow", "cascade")
ARMS = ("baseline", "quadratic", "candidate")
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
        raise ValueError("frozen excited bilinear ablation protocol changed")
    return read_json(PROTOCOL)


def _check_inherited_sources(p):
    for key in (
        "prior_interaction_protocol",
        "prior_quadratic_protocol",
        "prior_excitation_protocol",
        "prior_architecture_protocol",
    ):
        previous = p[key]
        if digest(ROOT / previous["path"]) != previous["sha256"]:
            raise ValueError(f"historical protocol changed: {key}")
    for path, expected in p["inherited_source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"inherited implementation changed: {path}")


def _ablation_sources():
    paths = [
        ROOT / prefix / (stem + suffix)
        for stem in (
            "fit_checkpoints",
            "excited_bilinear_matching",
            "excited_bilinear_experiment",
            "excited_bilinear_decision",
        )
        for prefix, suffix in (("src/glassbox/experimental", ".py"), ("tests", ".py"))
    ]
    # Tests use the conventional test_ prefix, unlike implementation modules.
    paths = [
        path.with_name("test_" + path.name) if path.parent.name == "tests" else path
        for path in paths
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("bilinear ablation implementation or tests are missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _checkpoint_sources(p, added):
    source_file = ROOT / p["sources"]["path"]
    if digest(source_file) != p["sources"]["sha256"]:
        raise ValueError("base checkpoint source inventory changed")
    result = {}
    for inventory in (
        read_json(source_file)["glassbox_source_sha256"],
        p["parent_source_sha256"],
        p["inherited_source_sha256"],
        added,
    ):
        for path, sha in inventory.items():
            if path in result and result[path] != sha:
                raise ValueError("conflicting checkpoint source identity: " + path)
            if digest(ROOT / path) != sha:
                raise ValueError("checkpoint source identity changed: " + path)
            result[path] = sha
    return result


@lru_cache(maxsize=1)
def _implementation():
    """Own private globals; never mutate either historical canonical module."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._excited_bilinear_interaction", inherited.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL, module.PROTOCOL_SHA256 = PROTOCOL, PROTOCOL_SHA256
    module.protocol = protocol
    module.ARMS, module.PREDICTORS = ARMS, PREDICTORS
    private_flight = module.flight()
    inherited_check = private_flight.check_sources

    def check(p):
        _check_inherited_sources(p)
        runtime = inherited_check(p)
        runtime["bilinear_ablation_implementation"] = _ablation_sources()
        runtime["checkpoint_source_sha256"] = _checkpoint_sources(
            p, runtime["bilinear_ablation_implementation"]
        )
        return runtime

    private_flight.check_sources = check
    module._original_check_links = module.check_links
    module._original_predict = module.Predictors.predict
    module.fit_provenance = lambda *args: fit_provenance(*args)
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


def trusted_excitation():
    from .excited_public_data import trusted_excitation as verified_source

    return verified_source(protocol(), flight())


def trusted_architecture():
    p = protocol()
    declared = p["prior_architecture_bundle"]
    path = Path(declared["path"])
    if digest(path / "run.json") != declared["manifest_sha256"]:
        raise ValueError("prior architecture bundle differs from trusted anchor")
    manifest = verify_files(path, "run.json")
    if manifest["protocol_sha256"] != p["prior_architecture_protocol"]["sha256"]:
        raise ValueError("prior architecture protocol differs from trusted source")
    return path


def _trusted_model(simulator, arm):
    declared = protocol()["trusted_comparator_revisions"][simulator][arm]
    path = trusted_architecture() / declared["source_relative_path"]
    if digest(path) != declared["file_sha256"]:
        raise ValueError("trusted comparator file differs from frozen revision")
    model = _model_classes()[arm].load(path)
    if model.fingerprint() != declared["revision_fingerprint"]:
        raise ValueError("trusted comparator fingerprint differs")
    return model


def trusted_baseline_model(simulator):
    return _trusted_model(simulator, "baseline")


def trusted_quadratic_model(simulator):
    return _trusted_model(simulator, "quadratic")


def compare_calibration(simulator, output):
    return _implementation().compare_calibration(simulator, output)


def generate(simulator, output):
    from .excited_public_data import import_excitation

    _implementation().generate(simulator, output)
    result = import_excitation(simulator, output, flight(), protocol(), check_sources())
    write_json(Path(output) / simulator / "reuse.json", result)
    return result


def check_import(simulator, output, runtime):
    from .excited_public_data import check_import as verify_import

    result = verify_import(simulator, output, flight(), protocol(), runtime)
    if result != read_json(Path(output) / simulator / "reuse.json"):
        raise ValueError("imported excitation reuse evidence differs")
    return result


def data_ready(output):
    runtime = _implementation().data_ready(output)
    for simulator in SIMULATORS:
        check_import(simulator, output, runtime)
    return runtime


def _same_revision(model, previous, arm):
    """Require all saved parameters, norms, caches, fit evidence and contract."""
    if model._metadata() != previous._metadata():
        raise ValueError(f"{arm} metadata, windows, contract or fit report changed")
    array_equal(model._arrays(), previous._arrays(), arm)
    current, old = model.fingerprint(), previous.fingerprint()
    if current != old:
        raise ValueError(f"{arm} fingerprint differs from trusted previous revision")
    return {
        "previous_fingerprint": old,
        "current_fingerprint": current,
        "all_saved_arrays_and_metadata_exact": True,
        "prediction_parameters_exact": True,
    }


def compare_baseline(model, simulator):
    return _same_revision(model, trusted_baseline_model(simulator), "baseline")


def compare_quadratic(model, simulator):
    return _same_revision(model, trusted_quadratic_model(simulator), "quadratic")


def compare_candidate(model, simulator):
    from .excited_bilinear_matching import check_matched_models

    # Matching remains possible if an unrelated fresh comparator fit fails.
    return check_matched_models(
        trusted_baseline_model(simulator), model, trusted_quadratic_model(simulator)
    )


def fit_provenance(simulator, arm, output, runtime):
    if arm not in ARMS:
        raise ValueError("unknown fitting arm")
    base = Path(output) / simulator
    result = {
        "runtime": runtime,
        "arm": arm,
        "simulator": simulator,
        "data_seal": digest(base / "data/seal.json"),
        "calibration_sha256": digest(base / "calibration.json"),
        "imported_excitation_seal": digest(base / "excitation/seal.json"),
        "reuse_sha256": digest(base / "reuse.json"),
        "trusted_excitation_bundle_sha256": protocol()["prior_excitation_bundle"][
            "manifest_sha256"
        ],
        "fitting_collection": "excitation",
        "trusted_architecture_bundle_sha256": protocol()["prior_architecture_bundle"][
            "manifest_sha256"
        ],
        "checkpoint_steps": protocol()["checkpoint_diagnostics"]["steps"],
    }
    if arm != "baseline":
        verify_files(base / "baseline", "seal.json")
        result["baseline_seal"] = digest(base / "baseline/seal.json")
    if arm in ("baseline", "quadratic"):
        result["trusted_revision_arm"] = (
            "candidate" if arm == "baseline" else "quadratic"
        )
    if arm == "quadratic":
        result["trusted_preparation_reference_arm"] = "original"
    if arm == "candidate":
        result["matched_data_reference_arm"] = "current_baseline_public_excited"
        result["matching_quadratic_reference_arm"] = "quadratic"
    return result


def _model_classes():
    from glassbox.experimental.state_input_model import CandidateDynamics as Bilinear
    from glassbox.experimental.state_quadratic_model import (
        CandidateDynamics as Quadratic,
    )
    from glassbox.learner import LearnedDynamics

    return dict(
        baseline=LearnedDynamics,
        quadratic=Quadratic,
        candidate=Bilinear,
    )


class PreparationUnavailable(RuntimeError):
    """A declared fit dependency failed; no substitute preparation fit is run."""


def _expected_recipe(arm):
    from glassbox.learner import RECIPE as public

    from .state_input_model import RECIPE as bilinear
    from .state_quadratic_model import RECIPE as quadratic

    return {"baseline": public, "candidate": bilinear, "quadratic": quadratic}[arm]


def _checkpoint_arguments(simulator, arm, output, runtime, model):
    reference = trusted_baseline_model(simulator)
    return dict(
        train=reference._train,
        development=reference._development,
        provenance=fit_provenance(simulator, arm, output, runtime),
        source_sha256=runtime["checkpoint_source_sha256"],
        selected_model=None if model is None else model._model,
    )


def check_checkpoints(output, simulator, arm, *, replay=False):
    from .fit_checkpoints import check_checkpoint_links, replay_checkpoints

    directory = Path(output) / simulator / arm
    outcome = read_json(directory / "outcome.json")
    model = (
        _model_classes()[arm].load(directory / "model.npz")
        if outcome["model_available"]
        else None
    )
    arguments = _checkpoint_arguments(simulator, arm, output, check_sources(), model)
    completed = outcome["model_available"]
    if (
        type(completed) is not bool
        or completed != (outcome["status"] == "complete")
        or (completed and outcome["reason"] is not None)
        or (
            not completed
            and (
                outcome["status"] not in ("fit_failure", "preparation_unavailable")
                or not outcome["reason"]
            )
        )
    ):
        raise ValueError("fit outcome status, availability or failure reason differs")
    optimization = None if model is None else model.report["optimization"]
    expectation = dict(
        arm=arm,
        recipe=_expected_recipe(arm),
        expected_steps=protocol()["checkpoint_diagnostics"]["steps"],
        completed=completed,
        failure=outcome["reason"],
        selected_step=None if optimization is None else optimization["selected_step"],
        trace=None if optimization is None else optimization["trace"],
        status=outcome["status"],
    )
    checker = replay_checkpoints if replay else check_checkpoint_links
    return checker(
        directory / "checkpoints",
        expected_sha256=outcome["checkpoint_manifest_sha256"],
        expectation=expectation,
        **arguments,
    )


def fit_arm(simulator, arm, output):
    from glassbox import fit
    from glassbox.experimental.command_excitation_fit import (
        ExcitationDataError,
        fit_excited,
    )
    from glassbox.experimental.state_input_model import (
        CandidateFitError as BilinearFitError,
    )
    from glassbox.experimental.state_input_model import fit_candidate
    from glassbox.experimental.state_quadratic_model import (
        CandidateDynamics,
        CandidateFitError,
    )
    from glassbox.learner import RECIPE, LearnedDynamics

    from .fit_checkpoints import capture_fit, save_checkpoints

    if arm not in ARMS:
        raise ValueError("unknown fitting arm")
    if RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("public recipe differs from frozen baseline")
    runtime = data_ready(output)
    base = Path(output) / simulator
    directory = base / arm
    directory.mkdir(parents=True, exist_ok=False)
    provenance = fit_provenance(simulator, arm, output, runtime)
    write_json(directory / "start.json", provenance)
    reference = trusted_baseline_model(simulator)
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
            if arm == "candidate":
                if not read_json(base / "baseline/outcome.json")["model_available"]:
                    raise PreparationUnavailable(
                        "current public-excited baseline unavailable; bilinear optimizer not attempted"
                    )
                preparation = LearnedDynamics.load(base / "baseline/model.npz")
                compare_baseline(preparation, simulator)
                model = fit_candidate(preparation)
            else:
                collection, _ = flight().collection_and_trajectories(
                    base / "excitation"
                )
                if arm == "baseline":
                    model = fit(collection)
                else:
                    model = fit_excited(
                        collection,
                        CandidateDynamics.load(
                            trusted_excitation() / simulator / "original/model.npz"
                        ),
                        roles=protocol()["planned_automatic_roles"][simulator],
                        data_seal=protocol()["imported_calibration"][
                            "excitation_seal_sha256"
                        ][simulator],
                    )
    except PreparationUnavailable as error:
        reason, status = str(error), "preparation_unavailable"
    except (
        ValueError,
        CandidateFitError,
        BilinearFitError,
        ExcitationDataError,
    ) as error:
        if not isinstance(
            error, (CandidateFitError, BilinearFitError, ExcitationDataError)
        ) and not inherited._nonfinite_fit_error(error):
            raise
        reason, status = str(error), "fit_failure"
    fit_wall_time = time.perf_counter() - start
    if model is not None:
        comparison = {
            "baseline": compare_baseline,
            "quadratic": compare_quadratic,
            "candidate": compare_candidate,
        }[arm]
        write_json(directory / "matching.json", comparison(model, simulator))
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    diagnostics_start = time.perf_counter()
    checkpoint_evidence = save_checkpoints(
        directory / "checkpoints",
        captured,
        train=reference._train,
        development=reference._development,
        recipe=_expected_recipe(arm),
        selected_step=None
        if model is None
        else model.report["optimization"]["selected_step"],
        failure=reason,
        selected_model=None if model is None else model._model,
    )
    diagnostics_wall_time = time.perf_counter() - diagnostics_start
    write_json(
        directory / "outcome.json",
        {
            "status": status,
            "model_available": model is not None,
            "reason": reason,
            "checkpoint_manifest_sha256": checkpoint_evidence["manifest_sha256"],
        },
    )
    freeze_files(
        directory,
        "seal.json",
        {
            "wall_time_s": time.perf_counter() - start,
            "fit_and_capture_wall_time_s": fit_wall_time,
            "checkpoint_diagnostics_wall_time_s": diagnostics_wall_time,
        },
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
    _implementation()._original_check_links(output, simulator, evaluation=evaluation)
    runtime = check_sources()
    check_import(simulator, output, runtime)
    base = Path(output) / simulator
    for arm, cls in _model_classes().items():
        directory = base / arm
        check_checkpoints(output, simulator, arm)
        if not read_json(directory / "outcome.json")["model_available"]:
            continue
        model = cls.load(directory / "model.npz")
        if model.report != read_json(
            directory / "report.json"
        ) or model.contract != read_json(directory / "contract.json"):
            raise ValueError("model report or contract mirror differs")
        comparison = {
            "baseline": compare_baseline,
            "quadratic": compare_quadratic,
            "candidate": compare_candidate,
        }[arm]
        if comparison(model, simulator) != read_json(directory / "matching.json"):
            raise ValueError("saved revision or same-data matching evidence differs")


def validate_rows(simulator, output, rows):
    return _implementation().validate_rows(simulator, output, rows)


def evaluate(simulator, output, *, replay=False):
    return _implementation().evaluate(simulator, output, replay=replay)


def decision(output):
    from glassbox.experimental.excited_bilinear_decision import reduce

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
        {
            "protocol_sha256": digest(PROTOCOL),
            "scope": "Frozen matched excited-bilinear-ablation-v1 comparison",
            "runtime": check_sources(),
        },
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    return _implementation().verify_bundle(output, expected_sha256)


def replay(simulator, output, expected_sha256):
    from .excited_public_data import replay_import

    verify_bundle(output, expected_sha256)
    result = {
        "physics": flight().replay_data(simulator, output),
        "calibration": compare_calibration(simulator, output),
        "imported_excitation": replay_import(
            simulator, output, flight(), protocol(), check_sources()
        ),
        "checkpoints": {
            arm: check_checkpoints(output, simulator, arm, replay=True) for arm in ARMS
        },
        "predictions": evaluate(simulator, output, replay=True),
    }
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
    parser.add_argument("--arm", choices=ARMS)
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
