"""Frozen independent command excitation experiment, reusing isolated historical mechanics."""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import jax

from glassbox.experimental import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/independent-command-excitation-v1.json"
PROTOCOL_SHA256 = "e4fd93ba3869ad23663e218cb1ec0849c2f191eb12312bd38e43f9bb1039e8c8"
SIMULATORS = ("crazyflow", "cascade")
ARMS = ("baseline", "original", "candidate")
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
        raise ValueError("frozen independent command excitation protocol changed")
    return read_json(PROTOCOL)


def _check_inherited_sources(p):
    for key in ("prior_interaction_protocol", "prior_quadratic_protocol"):
        previous = p[key]
        if digest(ROOT / previous["path"]) != previous["sha256"]:
            raise ValueError(f"historical protocol changed: {key}")
    for path, expected in p["inherited_source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"inherited implementation changed: {path}")


def _excitation_sources():
    paths = [
        ROOT / prefix / (stem + suffix)
        for stem in (
            "command_excitation_data",
            "command_excitation_fit",
            "command_excitation_experiment",
            "command_excitation_decision",
        )
        for prefix, suffix in (("src/glassbox/experimental", ".py"), ("tests", ".py"))
    ]
    # Tests use the conventional test_ prefix, unlike implementation modules.
    paths = [
        path.with_name("test_" + path.name) if path.parent.name == "tests" else path
        for path in paths
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("excitation implementation or tests are missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    """Own private globals; never mutate either historical canonical module."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._command_excitation_interaction", inherited.__file__
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
        runtime["excitation_implementation"] = _excitation_sources()
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


def trusted_quadratic():
    p = protocol()["prior_quadratic_bundle"]
    previous = Path(p["path"])
    if digest(previous / "run.json") != p["manifest_sha256"]:
        raise ValueError("prior quadratic bundle differs from trusted anchor")
    verify_files(previous, "run.json")
    return previous


def compare_calibration(simulator, output):
    return _implementation().compare_calibration(simulator, output)


def generate(simulator, output):
    from .command_excitation_data import generate_excitation

    _implementation().generate(simulator, output)
    result = generate_excitation(
        simulator, output, flight(), protocol(), check_sources()
    )
    write_json(Path(output) / simulator / "excitation.json", result)
    return result


def check_excitation(simulator, output, runtime):
    from .command_excitation_data import compare_excitation

    base = Path(output) / simulator
    seal = verify_files(base / "excitation", "seal.json")
    if seal["runtime"] != runtime:
        raise ValueError("excitation runtime/source identity differs")
    if seal["common_data_seal"] != digest(base / "data/seal.json"):
        raise ValueError("excitation common data identity differs")
    result = compare_excitation(simulator, output, flight(), protocol())
    if result != read_json(base / "excitation.json"):
        raise ValueError("excitation comparison differs")
    return result


def data_ready(output):
    runtime = _implementation().data_ready(output)
    for simulator in SIMULATORS:
        check_excitation(simulator, output, runtime)
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
    from glassbox.learner import LearnedDynamics

    previous = LearnedDynamics.load(
        trusted_previous() / simulator / "generic/model.npz"
    )
    return _same_revision(model, previous, "baseline")


def compare_original(model, simulator):
    from glassbox.experimental.state_quadratic_model import CandidateDynamics

    previous = CandidateDynamics.load(
        trusted_quadratic() / simulator / "candidate/model.npz"
    )
    return _same_revision(model, previous, "original")


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
    }
    if arm != "baseline":
        verify_files(base / "baseline", "seal.json")
        result["baseline_seal"] = digest(base / "baseline/seal.json")
    if arm in ("baseline", "original"):
        key = "previous_bundle" if arm == "baseline" else "prior_quadratic_bundle"
        result["trusted_reference_bundle_sha256"] = protocol()[key]["manifest_sha256"]
    if arm == "candidate":
        verify_files(base / "original", "seal.json")
        result.update(
            original_seal=digest(base / "original/seal.json"),
            excitation_seal=digest(base / "excitation/seal.json"),
            excitation_comparison_sha256=digest(base / "excitation.json"),
        )
    return result


def _model_classes():
    from glassbox.experimental.state_quadratic_model import CandidateDynamics
    from glassbox.learner import LearnedDynamics

    return dict(
        baseline=LearnedDynamics,
        original=CandidateDynamics,
        candidate=CandidateDynamics,
    )


def fit_arm(simulator, arm, output):
    from glassbox import fit
    from glassbox.experimental.command_excitation_fit import (
        ExcitationDataError,
        fit_excited,
    )
    from glassbox.experimental.state_quadratic_model import (
        CandidateDynamics,
        CandidateFitError,
        fit_candidate,
    )
    from glassbox.learner import RECIPE, LearnedDynamics

    if arm not in ARMS:
        raise ValueError("unknown fitting arm")
    if RECIPE != protocol()["fitting"]["generic_recipe"]:
        raise ValueError("public recipe differs from frozen baseline")
    runtime = data_ready(output)
    base = Path(output) / simulator
    directory = base / arm
    directory.mkdir(parents=True, exist_ok=False)
    write_json(
        directory / "start.json", fit_provenance(simulator, arm, output, runtime)
    )
    start = time.perf_counter()
    model, reason = None, None
    try:
        if arm == "baseline":
            collection, _ = flight().collection_and_trajectories(base / "data")
            model = fit(collection)
        elif not read_json(base / "baseline/outcome.json")["model_available"]:
            reason = "baseline_model_unavailable"
        elif arm == "original":
            model = fit_candidate(LearnedDynamics.load(base / "baseline/model.npz"))
        elif not read_json(base / "original/outcome.json")["model_available"]:
            reason = "original_quadratic_model_unavailable"
        else:
            collection, _ = flight().collection_and_trajectories(base / "excitation")
            model = fit_excited(
                collection,
                CandidateDynamics.load(base / "original/model.npz"),
                roles=protocol()["planned_automatic_roles"][simulator],
                data_seal=digest(base / "excitation/seal.json"),
            )
    except (ValueError, CandidateFitError, ExcitationDataError) as error:
        if not isinstance(
            error, (CandidateFitError, ExcitationDataError)
        ) and not inherited._nonfinite_fit_error(error):
            raise
        reason = str(error)
    if model is not None:
        if arm in ("baseline", "original"):
            comparison = compare_baseline if arm == "baseline" else compare_original
            write_json(directory / "matching.json", comparison(model, simulator))
        model.save(directory / "model.npz")
        write_json(directory / "report.json", model.report)
        write_json(directory / "contract.json", model.contract)
    write_json(
        directory / "outcome.json",
        {
            "status": "complete" if model is not None else "fit_failure",
            "model_available": model is not None,
            "reason": reason,
        },
    )
    freeze_files(directory, "seal.json", {"wall_time_s": time.perf_counter() - start})
    print(
        f"fit complete {simulator}/{arm}: model_available={model is not None}",
        flush=True,
    )


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


def _check_fit_cache(model, baseline, arm):
    if model.report["matching"]["baseline_fingerprint"] != baseline.fingerprint():
        raise ValueError(f"{arm} baseline fingerprint differs")
    for role in ("train", "development"):
        current, old = getattr(model, "_" + role), getattr(baseline, "_" + role)
        if current.keys != old.keys or current.source_origins != old.source_origins:
            raise ValueError(f"{arm} selected windows changed")
        names = ("past_states", "past_inputs", "future_inputs", "future_states")
        array_equal(
            {key: getattr(current.batch, key) for key in names},
            {key: getattr(old.batch, key) for key in names},
            f"{arm}/{role}",
        )
    array_equal(
        {key: model._model.norms[key] for key in baseline._model.norms},
        baseline._model.norms,
        f"{arm}/baseline_normalization",
    )
    if model.contract != baseline.contract:
        raise ValueError(f"{arm} signal/timing contract changed")


def check_links(output, simulator, *, evaluation=False):
    from .command_excitation_fit import check_preparation

    _implementation()._original_check_links(output, simulator, evaluation=evaluation)
    runtime = check_sources()
    check_excitation(simulator, output, runtime)
    base = Path(output) / simulator
    models = {}
    for arm, cls in _model_classes().items():
        directory = base / arm
        if not read_json(directory / "outcome.json")["model_available"]:
            continue
        model = models[arm] = cls.load(directory / "model.npz")
        if model.report != read_json(
            directory / "report.json"
        ) or model.contract != read_json(directory / "contract.json"):
            raise ValueError("model report or contract mirror differs")
        if arm in ("baseline", "original"):
            comparison = compare_baseline if arm == "baseline" else compare_original
            if comparison(model, simulator) != read_json(directory / "matching.json"):
                raise ValueError("saved comparator matching evidence differs")
        if arm == "original":
            if "baseline" not in models:
                raise ValueError("original quadratic has no baseline")
            _check_fit_cache(model, models["baseline"], arm)
        if arm == "candidate":
            if "original" not in models:
                raise ValueError("excited candidate has no original quadratic")
            collection, _ = flight().collection_and_trajectories(base / "excitation")
            check_preparation(
                model,
                collection,
                models["original"],
                roles=protocol()["planned_automatic_roles"][simulator],
                data_seal=digest(base / "excitation/seal.json"),
            )


def validate_rows(simulator, output, rows):
    return _implementation().validate_rows(simulator, output, rows)


def evaluate(simulator, output, *, replay=False):
    return _implementation().evaluate(simulator, output, replay=replay)


def decision(output):
    from glassbox.experimental.command_excitation_decision import reduce

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
            "scope": "Frozen matched independent-command-excitation-v1 mechanism experiment",
            "runtime": check_sources(),
        },
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    return _implementation().verify_bundle(output, expected_sha256)


def replay(simulator, output, expected_sha256):
    from .command_excitation_data import replay_excitation

    verify_bundle(output, expected_sha256)
    result = {
        "physics": flight().replay_data(simulator, output),
        "calibration": compare_calibration(simulator, output),
        "excitation": replay_excitation(
            simulator, output, flight(), protocol(), check_sources()
        ),
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
