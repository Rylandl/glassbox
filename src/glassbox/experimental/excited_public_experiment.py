"""Frozen public/quadratic comparison on identical independently excited recordings."""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import jax

from glassbox.experimental import state_input_experiment as inherited

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/excited-public-architecture-v1.json"
PROTOCOL_SHA256 = "6a020ae72a7bfb7a4e0453e0602f788b46d0e1aaf6319197a0743cd57c74fad4"
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
        raise ValueError("frozen excited public architecture protocol changed")
    return read_json(PROTOCOL)


def _check_inherited_sources(p):
    for key in (
        "prior_interaction_protocol",
        "prior_quadratic_protocol",
        "prior_excitation_protocol",
    ):
        previous = p[key]
        if digest(ROOT / previous["path"]) != previous["sha256"]:
            raise ValueError(f"historical protocol changed: {key}")
    for path, expected in p["inherited_source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"inherited implementation changed: {path}")


def _architecture_sources():
    paths = [
        ROOT / prefix / (stem + suffix)
        for stem in (
            "excited_public_data",
            "excited_public_matching",
            "excited_public_experiment",
            "excited_public_decision",
        )
        for prefix, suffix in (("src/glassbox/experimental", ".py"), ("tests", ".py"))
    ]
    # Tests use the conventional test_ prefix, unlike implementation modules.
    paths = [
        path.with_name("test_" + path.name) if path.parent.name == "tests" else path
        for path in paths
    ]
    if not all(path.is_file() for path in paths):
        raise ValueError("architecture comparison implementation or tests are missing")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


@lru_cache(maxsize=1)
def _implementation():
    """Own private globals; never mutate either historical canonical module."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._excited_public_interaction", inherited.__file__
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
        runtime["architecture_implementation"] = _architecture_sources()
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


def trusted_quadratic_model(simulator):
    from .state_quadratic_model import CandidateDynamics

    return CandidateDynamics.load(
        trusted_excitation() / simulator / "candidate/model.npz"
    )


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
    from glassbox.learner import LearnedDynamics

    previous = LearnedDynamics.load(
        trusted_excitation() / simulator / "baseline/model.npz"
    )
    return _same_revision(model, previous, "baseline")


def compare_quadratic(model, simulator):
    return _same_revision(model, trusted_quadratic_model(simulator), "quadratic")


def compare_candidate(model, simulator):
    from .excited_public_matching import check_matched_models

    # A trusted saved reference remains available even if the fresh quadratic
    # fit has a numerical failure. That failure must not erase public progress.
    return check_matched_models(model, trusted_quadratic_model(simulator))


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
        "fitting_collection": "data" if arm == "baseline" else "excitation",
    }
    if arm != "baseline":
        verify_files(base / "baseline", "seal.json")
        result["baseline_seal"] = digest(base / "baseline/seal.json")
    if arm in ("baseline", "quadratic"):
        result["trusted_revision_arm"] = (
            "baseline" if arm == "baseline" else "candidate"
        )
    if arm == "quadratic":
        result["trusted_preparation_reference_arm"] = "original"
    if arm == "candidate":
        result["matched_data_reference_arm"] = "candidate"
    return result


def _model_classes():
    from glassbox.experimental.state_quadratic_model import CandidateDynamics
    from glassbox.learner import LearnedDynamics

    return dict(
        baseline=LearnedDynamics,
        quadratic=CandidateDynamics,
        candidate=LearnedDynamics,
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
    )
    from glassbox.learner import RECIPE

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
        collection, _ = flight().collection_and_trajectories(
            base / ("data" if arm == "baseline" else "excitation")
        )
        if arm in ("baseline", "candidate"):
            # The ordinary public entrypoint receives no excitation metadata/options.
            model = fit(collection)
        else:
            model = fit_excited(
                collection,
                CandidateDynamics.load(
                    trusted_excitation() / simulator / "original/model.npz"
                ),
                roles=protocol()["planned_automatic_roles"][simulator],
                data_seal=protocol()["imported_calibration"]["excitation_seal_sha256"][
                    simulator
                ],
            )
    except (ValueError, CandidateFitError, ExcitationDataError) as error:
        if not isinstance(
            error, (CandidateFitError, ExcitationDataError)
        ) and not inherited._nonfinite_fit_error(error):
            raise
        reason = str(error)
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


def check_links(output, simulator, *, evaluation=False):
    _implementation()._original_check_links(output, simulator, evaluation=evaluation)
    runtime = check_sources()
    check_import(simulator, output, runtime)
    base = Path(output) / simulator
    for arm, cls in _model_classes().items():
        directory = base / arm
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
    from glassbox.experimental.excited_public_decision import reduce

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
            "scope": "Frozen matched excited-public-architecture-v1 comparison",
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
