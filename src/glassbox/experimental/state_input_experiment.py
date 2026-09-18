"""Frozen state/input interaction experiment; public learner and physics unchanged.

The private parent module only supplies the previously qualified data generator
and replay. Each new stage binds its protocol, implementation, data and models.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np

from glassbox.experimental import two_simulator_flight as parent
from glassbox.experimental.two_simulator_metrics import (
    GROUPS,
    _horizons,
    aggregate,
    score_forecast,
    score_response_directions,
)

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs/harness/state-input-interaction-v1.json"
PROTOCOL_SHA256 = "49320a597f2185e3b48ef3becd846496b503c21632e630b9ef173e4e1fa7d704"
SIMULATORS = ("crazyflow", "cascade")
ARMS = ("baseline", "candidate")
PREDICTORS = (*ARMS, "hold")
digest, read_json, write_json = parent.digest, parent.read_json, parent.write_json
load_arrays, save_arrays = parent.load_arrays, parent.save_arrays
array_equal, freeze_files, verify_files = (
    parent.array_equal,
    parent.freeze_files,
    parent.verify_files,
)


def protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen interaction protocol changed")
    return read_json(PROTOCOL)


@lru_cache(maxsize=1)
def flight():
    """Use separate globals so historical entrypoints keep their own protocol."""
    spec = importlib.util.spec_from_file_location(
        "glassbox.experimental._state_input_flight", parent.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROTOCOL = PROTOCOL
    original_check = module.check_sources

    def check_sources(p):
        if p != protocol():
            raise ValueError("interaction protocol identity differs")
        if (
            digest(ROOT / p["previous_protocol"]["path"])
            != p["previous_protocol"]["sha256"]
        ):
            raise ValueError("historical protocol changed")
        for path, sha in p["parent_source_sha256"].items():
            if digest(ROOT / path) != sha:
                raise ValueError(f"parent implementation changed: {path}")
        runtime = original_check(p)
        implementation = [
            ROOT / "src/glassbox/experimental" / name
            for name in (
                "state_input_experiment.py",
                "state_input_model.py",
                "state_input_decision.py",
            )
        ]
        tests = sorted((ROOT / "tests").glob("test_state_input_*.py"))
        if not tests:
            raise ValueError("experiment tests are missing")
        runtime["interaction_implementation"] = {
            str(path.relative_to(ROOT)): digest(path)
            for path in (*implementation, *tests)
        }
        return runtime

    module.check_sources = check_sources
    return module


def check_sources():
    return flight().check_sources(protocol())


def trusted_previous():
    p = protocol()["previous_bundle"]
    previous = Path(p["path"])
    if digest(previous / "run.json") != p["manifest_sha256"]:
        raise ValueError("historical bundle differs from trusted anchor")
    verify_files(previous, "run.json")
    return previous


def compare_calibration(simulator, output):
    """Verify unchanged raw calibration recordings and resulting admission/split."""
    previous = trusted_previous() / simulator / "data"
    current = Path(output) / simulator / "data"
    old_seal, new_seal = (
        verify_files(path, "seal.json") for path in (previous, current)
    )
    old_records, new_records = (
        [r for r in read_json(path / "records.json") if r["role"] == "calibration_pool"]
        for path in (previous, current)
    )
    if old_records != new_records:
        raise ValueError("calibration roster, setup or validity changed")
    for key in ("admitted", "roles"):
        if old_seal[key] != new_seal[key]:
            raise ValueError(f"calibration {key} changed")
    if read_json(previous / "roles.json") != read_json(current / "roles.json"):
        raise ValueError("calibration role mirror changed")
    arrays_count = 0
    for record in new_records:
        path = Path(record["prefix"] + ".npz")
        if (previous / path).exists() != (current / path).exists():
            raise ValueError("calibration physical outcome changed")
        if (current / path).exists():
            old, new = load_arrays(previous / path), load_arrays(current / path)
            array_equal(new, old, record["id"])
            arrays_count += len(new)
    return {
        "previous_bundle_sha256": protocol()["previous_bundle"]["manifest_sha256"],
        "previous_data_seal": digest(previous / "seal.json"),
        "current_data_seal": digest(current / "seal.json"),
        "calibration_parents": len(new_records),
        "arrays": arrays_count,
        "roles_and_admission_exact": True,
        "exact": True,
    }


def generate(simulator, output):
    flight().generate(simulator, output)
    result = compare_calibration(simulator, output)
    write_json(Path(output) / simulator / "calibration.json", result)
    return result


def data_ready(output):
    runtime = check_sources()
    for simulator in SIMULATORS:
        base = Path(output) / simulator
        seal = verify_files(base / "data", "seal.json")
        if seal["runtime"] != runtime:
            raise ValueError("data runtime/source identity differs")
        if read_json(base / "calibration.json") != compare_calibration(
            simulator, output
        ):
            raise ValueError("calibration comparison differs")
    return runtime


def fit_provenance(simulator, arm, output, runtime):
    base = Path(output) / simulator
    result = {
        "runtime": runtime,
        "arm": arm,
        "simulator": simulator,
        "data_seal": digest(base / "data/seal.json"),
        "calibration_sha256": digest(base / "calibration.json"),
    }
    if arm == "candidate":
        verify_files(base / "baseline", "seal.json")
        result["baseline_seal"] = digest(base / "baseline/seal.json")
    return result


def compare_baseline(model, simulator):
    from glassbox.learner import LearnedDynamics

    previous = LearnedDynamics.load(
        trusted_previous() / simulator / "generic/model.npz"
    )
    for role in ("train", "development"):
        names = ("past_states", "past_inputs", "future_inputs", "future_states")
        current = getattr(model, "_" + role)
        old = getattr(previous, "_" + role)
        array_equal(
            {key: getattr(current.batch, key) for key in names},
            {key: getattr(old.batch, key) for key in names},
            "baseline/" + role,
        )
        if current.keys != old.keys or current.source_origins != old.source_origins:
            raise ValueError("baseline selected windows changed")
    array_equal(model._model.norms, previous._model.norms, "baseline/normalization")
    array_equal(
        {"loss_scale": np.asarray(model.report["optimization"]["error_scale"])},
        {"loss_scale": np.asarray(previous.report["optimization"]["error_scale"])},
        "baseline/loss_scale",
    )
    if model.contract != previous.contract:
        raise ValueError("baseline signal/timing contract changed")
    return {
        "previous_fingerprint": previous.fingerprint(),
        "current_fingerprint": model.fingerprint(),
        "training_development_windows_and_arrays_exact": True,
        "normalization_and_loss_scale_exact": True,
        "prediction_parameters_exact": all(
            np.array_equal(v, previous._model.params[k])
            for k, v in model._model.params.items()
        ),
    }


def _nonfinite_fit_error(error):
    """Recognize the pinned public optimizer's numerical failure messages only."""
    return isinstance(error, ValueError) and (
        str(error) == "initial recursive validation loss is nonfinite"
        or str(error).startswith("nonfinite sequence training at step ")
    )


def fit_arm(simulator, arm, output):
    from glassbox import fit
    from glassbox.experimental.state_input_model import (
        CandidateDynamics,
        CandidateFitError,
        fit_candidate,
    )
    from glassbox.learner import RECIPE, LearnedDynamics

    del CandidateDynamics
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
            write_json(directory / "matching.json", compare_baseline(model, simulator))
        elif read_json(base / "baseline/outcome.json")["model_available"]:
            model = fit_candidate(LearnedDynamics.load(base / "baseline/model.npz"))
        else:
            reason = "baseline_model_unavailable"
    except (ValueError, CandidateFitError) as error:
        if not isinstance(error, CandidateFitError) and not _nonfinite_fit_error(error):
            raise
        reason = str(error)
    if model is not None:
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
        from glassbox.experimental.state_input_model import CandidateDynamics
        from glassbox.learner import LearnedDynamics

        self.models = {}
        for arm, cls in (
            ("baseline", LearnedDynamics),
            ("candidate", CandidateDynamics),
        ):
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
        h = len(arrays["future_inputs"])
        result = {arm: np.full((h, 15), np.nan) for arm in PREDICTORS}
        if not query["history_eligible"]:
            return result
        finite = np.isfinite(arrays["future_inputs"]).all(axis=1)
        bad = np.flatnonzero(~finite)
        length = int(bad[0]) if len(bad) else h
        if not length:
            return result
        commands = arrays["future_inputs"][:length]
        for arm, function in self.functions.items():
            if function is not None:
                result[arm][:length] = np.asarray(
                    function(arrays["past_states"], arrays["past_inputs"], commands)
                )
        result["hold"][:length] = arrays["past_states"][-1]
        return result

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
    for arm in ARMS:
        verify_files(base / arm, "seal.json")
        if read_json(base / arm / "start.json") != fit_provenance(
            simulator, arm, output, runtime
        ):
            raise ValueError("fit data/runtime provenance differs")
    if evaluation:
        seal = verify_files(base / "evaluation", "seal.json")
        if seal["data_seal"] != digest(base / "data/seal.json") or seal["models"] != {
            arm: digest(base / arm / "seal.json") for arm in ARMS
        }:
            raise ValueError("evaluation data/model provenance differs")


def validate_rows(simulator, output, rows):
    """Require every planned query/arm/horizon/group/statistic slot exactly once."""
    p = protocol()
    queries = read_json(Path(output) / simulator / "data/queries.json")
    horizons = _horizons(
        p["generation"][simulator]["dt_s"], p["evaluation"]["horizons_s"]
    )
    expected = {}
    for q in queries:
        for arm in PREDICTORS:
            for _, horizon in horizons:
                for group in GROUPS:
                    for statistic in ("endpoint", "cumulative"):
                        key = (q["parent"], q["id"], arm, horizon, group, statistic)
                        expected[key] = q
    if len(rows) != len(expected):
        raise ValueError("evaluation row count differs from complete planned roster")
    seen = set()
    for row in rows:
        key = tuple(
            row[k]
            for k in ("parent", "query", "arm", "horizon_s", "group", "statistic")
        )
        if key in seen or key not in expected:
            raise ValueError("duplicate or unplanned evaluation slot")
        seen.add(key)
        q = expected[key]
        for field in ("parent", "scope", "cell", "origin", "kind", "simulator"):
            if row[field] != q[field]:
                raise ValueError("evaluation query identity differs")
        if row.get("channel") != q.get("channel") or row.get("sign") != q.get("sign"):
            raise ValueError("evaluation command identity differs")


def evaluate(simulator, output, *, replay=False):
    p = protocol()
    check_links(output, simulator, evaluation=replay)
    base = Path(output) / simulator
    directory = base / "evaluation"
    if not replay:
        directory.mkdir(parents=True, exist_ok=False)
    predictors = Predictors(output, simulator)
    rows, caches, directions, lower_pairs = [], {}, [], {}
    start = time.perf_counter()
    dt = p["generation"][simulator]["dt_s"]
    queries = read_json(base / "data/queries.json")
    for query in queries:
        arrays = load_arrays(base / "data" / query["path"])
        prediction = predictors.predict(query, arrays)
        result = dict(prediction)
        target = arrays["target"]
        if query["kind"] == "response":
            key = (query["parent"], query["origin"])
            if key not in caches:
                caches[key] = predictors.predict(
                    query, dict(arrays, future_inputs=arrays["factual_inputs"])
                )
            for arm in prediction:
                result[arm + "_factual"] = caches[key][arm]
                prediction[arm] = prediction[arm] - caches[key][arm]
                result[arm + "_response"] = prediction[arm]
            target = target - arrays["factual_target"]
            pair_key = (query["parent"], query["origin"], query["channel"])
            if query["sign"] < 0:
                lower_pairs[pair_key] = (prediction, target, arrays["valid"])
            else:
                lower_prediction, lower_target, lower_valid = lower_pairs.pop(pair_key)
                for arm in prediction:
                    scored = score_response_directions(
                        np.stack((lower_prediction[arm], prediction[arm])),
                        np.stack((lower_target, target)),
                        np.stack((lower_valid, arrays["valid"])),
                        dt_s=dt,
                        horizons_s=p["evaluation"]["horizons_s"],
                        thresholds=p["evaluation"]["responses"][
                            "weak_endpoint_thresholds"
                        ],
                    )
                    directions.extend(
                        dict(
                            row,
                            simulator=simulator,
                            scope=query["scope"],
                            cell=query["cell"],
                            parent=query["parent"],
                            origin=query["origin"],
                            channel=query["channel"],
                            arm=arm,
                        )
                        for row in scored
                    )
        path = directory / (query["parent"] + "/" + query["id"] + ".npz")
        if replay:
            array_equal(result, load_arrays(path), str(path))
        else:
            save_arrays(path, result)
        for arm, predicted in prediction.items():
            scores = score_forecast(
                predicted,
                target,
                arrays["valid"],
                dt_s=dt,
                horizons_s=p["evaluation"]["horizons_s"],
                envelope=predictors.envelope(arm)
                if query["kind"] == "factual"
                else None,
                rotation_geometry=query["kind"] == "factual",
            )
            rows.extend(
                dict(
                    score,
                    simulator=simulator,
                    scope=query["scope"],
                    cell=query["cell"],
                    parent=query["parent"],
                    query=query["id"],
                    origin=query["origin"],
                    channel=query.get("channel"),
                    sign=query.get("sign"),
                    kind=query["kind"],
                    arm=arm,
                )
                for score in scores
            )
    if lower_pairs:
        raise ValueError("unpaired response query")
    nonweak = {
        (
            row["parent"],
            row["origin"],
            row["channel"],
            row["horizon_s"],
            row["group"],
        ): row["pair_nonweak"]
        for row in directions
    }
    for row in rows:
        if row["kind"] == "response":
            row["pair_nonweak"] = nonweak[
                (
                    row["parent"],
                    row["origin"],
                    row["channel"],
                    row["horizon_s"],
                    row["group"],
                )
            ]
    validate_rows(simulator, output, rows)
    summary = aggregate(rows)
    if replay:
        for name, value in (
            ("directions", directions),
            ("rows", rows),
            ("summary", summary),
        ):
            if value != read_json(directory / (name + ".json")):
                raise ValueError(f"fresh {name} differs")
        return {
            "prediction_queries": len(queries),
            "metric_rows": len(rows),
            "exact": True,
        }
    for name, value in (
        ("directions", directions),
        ("rows", rows),
        ("summary", summary),
    ):
        write_json(directory / (name + ".json"), value)
    freeze_files(
        directory,
        "seal.json",
        {
            "data_seal": digest(base / "data/seal.json"),
            "models": {arm: digest(base / arm / "seal.json") for arm in ARMS},
            "wall_time_s": time.perf_counter() - start,
        },
    )
    print(f"evaluation complete {simulator}: {len(rows)} metric rows", flush=True)


def decision(output):
    from glassbox.experimental.state_input_decision import reduce

    summaries, rows = {}, {}
    for simulator in SIMULATORS:
        check_links(output, simulator, evaluation=True)
        base = Path(output) / simulator / "evaluation"
        summaries[simulator] = read_json(base / "summary.json")
        rows[simulator] = read_json(base / "rows.json")
        validate_rows(simulator, output, rows[simulator])
        if aggregate(rows[simulator]) != summaries[simulator]:
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
            "scope": "Frozen matched state-input-interaction-v1 mechanism experiment",
            "runtime": check_sources(),
        },
    )
    print(f"trusted bundle SHA256: {sha}", flush=True)
    return sha


def verify_bundle(output, expected_sha256):
    output = Path(output)
    if not expected_sha256 or digest(output / "run.json") != expected_sha256:
        raise ValueError("bundle differs from externally trusted SHA256")
    manifest = verify_files(output, "run.json")
    if (
        manifest["protocol_sha256"] != digest(PROTOCOL)
        or manifest["runtime"] != check_sources()
    ):
        raise ValueError("run protocol/runtime identity differs")
    for simulator in SIMULATORS:
        check_links(output, simulator, evaluation=True)


def replay(simulator, output, expected_sha256):
    verify_bundle(output, expected_sha256)
    result = {
        "physics": flight().replay_data(simulator, output),
        "calibration": compare_calibration(simulator, output),
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
    if args.stage == "generate":
        generate(args.simulator, args.output)
    elif args.stage == "fit":
        if args.arm is None:
            parser.error("--arm required for fit")
        fit_arm(args.simulator, args.arm, args.output)
    elif args.stage == "evaluate":
        evaluate(args.simulator, args.output)
    elif args.stage == "finalize":
        finalize(args.output)
    else:
        replay(args.simulator, args.output, args.expected_bundle_sha)


if __name__ == "__main__":
    main()
