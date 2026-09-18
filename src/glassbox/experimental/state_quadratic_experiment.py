"""Frozen autonomous quadratic experiment, reusing isolated historical mechanics."""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import jax

from glassbox.experimental import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/autonomous-state-quadratic-v1.json"
PROTOCOL_SHA256 = "dd7ad816039515183ee5769481d5a642a609c9ee58b39f90aa9c04c843ef233a"
SIMULATORS = ("crazyflow", "cascade")
ARMS = ("baseline", "bilinear", "candidate")
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
        raise ValueError("frozen autonomous quadratic protocol changed")
    return read_json(PROTOCOL)


def _check_inherited_sources(p):
    previous = p["prior_interaction_protocol"]
    if digest(ROOT / previous["path"]) != previous["sha256"]:
        raise ValueError("prior interaction protocol changed")
    for path, expected in p["inherited_source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"inherited interaction implementation changed: {path}")


def _quadratic_sources():
    paths = [
        ROOT / prefix / (stem + suffix)
        for stem in (
            "state_quadratic_model",
            "state_quadratic_experiment",
            "state_quadratic_decision",
        )
        for prefix, suffix in (("src/glassbox/experimental", ".py"), ("tests", ".py"))
    ]
    # Tests use the conventional test_ prefix, unlike implementation modules.
    paths = [
        path.with_name("test_" + path.name) if path.parent.name == "tests" else path
        for path in paths
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("quadratic implementation or tests are missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    """Own private globals; never mutate either historical canonical module."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._state_quadratic_interaction", inherited.__file__
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
        runtime["quadratic_implementation"] = _quadratic_sources()
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


def trusted_interaction():
    p = protocol()["prior_interaction_bundle"]
    previous = Path(p["path"])
    if digest(previous / "run.json") != p["manifest_sha256"]:
        raise ValueError("prior interaction bundle differs from trusted anchor")
    verify_files(previous, "run.json")
    return previous


def compare_calibration(simulator, output):
    return _implementation().compare_calibration(simulator, output)


def generate(simulator, output):
    return _implementation().generate(simulator, output)


def data_ready(output):
    return _implementation().data_ready(output)


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


def compare_bilinear(model, simulator):
    from glassbox.experimental.state_input_model import CandidateDynamics

    previous = CandidateDynamics.load(
        trusted_interaction() / simulator / "candidate/model.npz"
    )
    return _same_revision(model, previous, "bilinear")


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
    if arm in ("baseline", "bilinear"):
        key = "previous_bundle" if arm == "baseline" else "prior_interaction_bundle"
        result["trusted_reference_bundle_sha256"] = protocol()[key]["manifest_sha256"]
    return result


def _model_classes():
    from glassbox.experimental.state_input_model import (
        CandidateDynamics as BilinearDynamics,
    )
    from glassbox.experimental.state_quadratic_model import CandidateDynamics
    from glassbox.learner import LearnedDynamics

    return dict(
        baseline=LearnedDynamics, bilinear=BilinearDynamics, candidate=CandidateDynamics
    )


def fit_arm(simulator, arm, output):
    from glassbox import fit
    from glassbox.experimental.state_input_model import (
        CandidateFitError as BilinearFitError,
    )
    from glassbox.experimental.state_input_model import (
        fit_candidate as fit_bilinear,
    )
    from glassbox.experimental.state_quadratic_model import (
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
        elif read_json(base / "baseline/outcome.json")["model_available"]:
            baseline = LearnedDynamics.load(base / "baseline/model.npz")
            fitter = fit_bilinear if arm == "bilinear" else fit_candidate
            model = fitter(baseline)
        else:
            reason = "baseline_model_unavailable"
    except (ValueError, BilinearFitError, CandidateFitError) as error:
        if not isinstance(
            error, (BilinearFitError, CandidateFitError)
        ) and not inherited._nonfinite_fit_error(error):
            raise
        reason = str(error)
    if model is not None:
        # A comparator mismatch is a harness error, never a numerical fit failure.
        if arm in ("baseline", "bilinear"):
            comparison = compare_baseline if arm == "baseline" else compare_bilinear
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
    _implementation()._original_check_links(output, simulator, evaluation=evaluation)
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
        if arm in ("baseline", "bilinear"):
            comparison = compare_baseline if arm == "baseline" else compare_bilinear
            if comparison(model, simulator) != read_json(directory / "matching.json"):
                raise ValueError("saved comparator matching evidence differs")
        if arm != "baseline":
            if "baseline" not in models:
                raise ValueError("fitted research arm has no baseline")
            _check_fit_cache(model, models["baseline"], arm)


def validate_rows(simulator, output, rows):
    return _implementation().validate_rows(simulator, output, rows)


def evaluate(simulator, output, *, replay=False):
    return _implementation().evaluate(simulator, output, replay=replay)


def decision(output):
    from glassbox.experimental.state_quadratic_decision import reduce

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
            "scope": "Frozen matched autonomous-state-quadratic-v1 mechanism experiment",
            "runtime": check_sources(),
        },
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    return _implementation().verify_bundle(output, expected_sha256)


def replay(simulator, output, expected_sha256):
    return _implementation().replay(simulator, output, expected_sha256)


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
