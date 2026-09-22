"""Offline qualification on fixed saved recording roles and physical queries."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from collect_throw import (
    authenticate,
    binding,
    check_source,
    checkpoint,
    seal,
    source_binding,
)
from evaluate_online import geometric, metrics
from run_dart import ROOT, write
from verify_baseline import arrays, digest, exact, observed, read, require

PROTOCOL = ROOT / "docs/harness/accumulator-qualification-offline-v1.json"
FIELDS = ("past_states", "past_inputs", "future_inputs", "future_states")


def start(output):
    spec, bound = read(PROTOCOL), binding(PROTOCOL)
    baseline = Path(spec["baseline"]["path"])
    authenticate(baseline, spec["baseline"]["manifest_sha256"])
    import glassbox

    model_root = Path(glassbox.__file__).resolve().parents[2]
    bound["model_source"] = source_binding(model_root, "src")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "binding.json", bound)
    write(output / "protocol.json", spec)
    return spec, bound, baseline


def windows_from_archive(meta, data):
    from glassbox._training import SequenceBatch
    from glassbox.recordings import SequenceWindows, WindowKey

    result = {}
    for role in ("train", "development"):
        keys = meta["windows"][role]
        result[role] = SequenceWindows(
            SequenceBatch(
                **{key: data[f"{role}_{key}"] for key in FIELDS},
                dt_s=meta["model"]["dt_s"],
            ),
            tuple(WindowKey(**key) for key in keys["keys"]),
            tuple(keys["source_origins"]),
        )
    return result


def fit_one(name, arm, output):
    import glassbox._dynamics as dynamics
    from glassbox import LearnedDynamics
    from glassbox._learner_arrays import load_arrays
    from glassbox.learner import _train

    spec, bound, baseline = start(output)
    require(name in spec["models"] and arm in ("baseline", "candidate"), "fit roster")
    expected = spec["model_package_files"][arm]
    require(bound["model_source"]["files"] == expected, "fit package differs")
    meta, data = load_arrays(baseline / "models" / f"{name}.npz")
    windows = windows_from_archive(meta, data)
    original = dynamics.trial_parameters
    started, attempt = time.monotonic(), 0

    def trial(*args):
        nonlocal attempt
        if args[2] == 1.0:
            attempt += 1
            if attempt == 1 or attempt % 100 == 0:
                print(
                    json.dumps(
                        dict(
                            model=name,
                            arm=arm,
                            attempt_started=attempt,
                            elapsed_s=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
        return original(*args)

    dynamics.trial_parameters = trial
    report = dict(
        model=name,
        arm=arm,
        input_model_sha256=digest(baseline / "models" / f"{name}.npz"),
    )
    try:
        model = _train(
            windows["train"], windows["development"], meta["contract"], meta["seen"]
        )
        model.save(output / "model.npz")
        restored = LearnedDynamics.load(output / "model.npz")
        require(
            model.fingerprint() == restored.fingerprint(),
            "saved fitted revision differs",
        )
        report.update(
            status="complete", fingerprint=model.fingerprint(), report=model.report
        )
    except Exception as error:
        report.update(status="failed", error=repr(error), attempts_started=attempt)
    finally:
        dynamics.trial_parameters = original
        check_source(bound)
        write(output / "result.json", report)
        print(
            json.dumps(
                dict(
                    output=str(output),
                    status=report["status"],
                    manifest_sha256=seal(output, "glassbox-accumulator-offline-fit-v1"),
                )
            ),
            flush=True,
        )
    require(report["status"] == "complete", report.get("error", "fit failed"))


def command_lengths(data, key):
    available = np.isfinite(data[key]).all(-1)
    lengths = np.cumprod(available, axis=1).sum(1)
    history_ok = np.isfinite(data["past_states"]).all((1, 2)) & np.isfinite(
        data["past_inputs"]
    ).all((1, 2))
    return np.where(history_ok, lengths, 0)


def predict_queries(model, data):
    """Fixed full-horizon batches; only command-available causal prefixes scored."""
    import jax

    kernel = jax.jit(model.predict)
    output = {}
    for suffix, key in (("", "future_inputs"), ("_factual", "factual_inputs")):
        inputs = data[key]
        lengths = command_lengths(data, key)
        predictions = np.full(data["target"].shape, np.nan, np.float32)
        eligible = np.flatnonzero(lengths)
        for start_index in range(0, len(eligible), 32):
            chosen = eligible[start_index : start_index + 32]
            index = np.pad(chosen, (0, 32 - len(chosen)), mode="edge")
            commands = inputs[index].copy()
            for i, row in enumerate(index):
                commands[i, int(lengths[row]) :] = commands[i, int(lengths[row]) - 1]
            pred = np.asarray(
                kernel(data["past_states"][index], data["past_inputs"][index], commands)
            )
            require(np.isfinite(pred).all(), "nonfinite eligible forecast")
            for i, row in enumerate(chosen):
                predictions[row, : int(lengths[row])] = pred[i, : int(lengths[row])]
        output["predicted" + suffix] = predictions
        output["lengths" + suffix] = lengths
    return output


def derivative_checks(model, data, ids):
    import jax
    import jax.numpy as jnp

    results = []
    rng = np.random.default_rng(772)
    with jax.enable_x64(True):
        for row in ids:
            past, issued, future = (
                jnp.asarray(data[k][row])
                for k in ("past_states", "past_inputs", "future_inputs")
            )
            direction = jnp.asarray(rng.normal(size=future.shape))
            direction /= jnp.linalg.norm(direction)
            kernel = jax.jit(lambda u, x=past, up=issued: model.predict(x, up, u))
            prediction, tangent = jax.jvp(kernel, (future,), (direction,))
            finite_difference = [
                (kernel(future + eps * direction) - kernel(future - eps * direction))
                / (2 * eps)
                for eps in (1e-4, 1e-5)
            ]
            results.append(
                (
                    int(row),
                    np.asarray(prediction),
                    np.asarray(tangent),
                    np.asarray(direction),
                    *[np.asarray(x) for x in finite_difference],
                )
            )
    return {
        key: np.stack([x[i] for x in results])
        for i, key in enumerate(
            ("row", "prediction", "jvp", "direction", "fd_coarse", "fd_fine")
        )
    }


def score_flights(baseline, predictions):
    from scipy.spatial.transform import Rotation

    result = []
    contracts = read(baseline / "flights/spec.json")["cohorts"]
    for cohort, values in predictions.items():
        roster = read(baseline / "flights" / f"{cohort}.json")
        data = arrays(baseline / "flights" / f"{cohort}.npz")
        contract = contracts[cohort]
        for kind in ("factual", "response"):
            for scope in sorted({q["scope"] for q in roster}):
                for step in contract["contract"]["horizons"]:
                    selected = np.array(
                        [
                            i
                            for i, q in enumerate(roster)
                            if q["kind"] == kind
                            and q["scope"] == scope
                            and q["history_eligible"]
                        ],
                        dtype=int,
                    )
                    target = data["target"][selected, step - 1]
                    prediction = values["predicted"][selected, step - 1]
                    valid = (
                        np.isfinite(target).all(-1)
                        & (values["lengths"][selected] >= step)
                        & data["valid"][selected, step - 1].astype(bool)
                    )
                    if kind == "response":
                        factual = data["factual_target"][selected, step - 1]
                        fp = values["predicted_factual"][selected, step - 1]
                        valid &= np.isfinite(factual).all(-1) & (
                            values["lengths_factual"][selected] >= step
                        )
                    require(valid.any(), "empty declared metric group")
                    require(
                        np.isfinite(prediction[valid]).all(),
                        "nonfinite scored prediction",
                    )
                    if kind == "factual":
                        scores = metrics(prediction[valid], target[valid])
                        error = prediction[valid, :6] - target[valid, :6]
                    else:
                        require(
                            np.isfinite(fp[valid]).all(),
                            "nonfinite factual response prediction",
                        )
                        error = (prediction[valid, :6] - fp[valid, :6]) - (
                            target[valid, :6] - factual[valid, :6]
                        )
                        matrices = [
                            x[valid, 6:].reshape(-1, 3, 3)
                            for x in (prediction, fp, target, factual)
                        ]
                        pr, fr, tr, tf = matrices
                        predicted_response = fr.transpose(0, 2, 1) @ pr
                        true_response = tf.transpose(0, 2, 1) @ tr
                        angle = Rotation.from_matrix(
                            predicted_response.transpose(0, 2, 1) @ true_response
                        ).magnitude()
                        scores = dict(
                            velocity_rmse_m_s=float(
                                np.sqrt(np.mean(np.sum(error[:, :3] ** 2, -1)))
                            ),
                            body_rate_rmse_rad_s=float(
                                np.sqrt(np.mean(np.sum(error[:, 3:6] ** 2, -1)))
                            ),
                            orientation_rmse_rad=float(np.sqrt(np.mean(angle**2))),
                        )
                    tail = []
                    for a, b in ((0, 3), (3, 6)):
                        squared = np.sum(error[:, a:b] ** 2, -1)
                        tail.append(
                            float(
                                np.sqrt(
                                    np.mean(
                                        np.sort(squared)[
                                            -max(1, int(np.ceil(len(squared) * 0.1))) :
                                        ]
                                    )
                                )
                            )
                        )
                    result.append(
                        dict(
                            cohort=cohort,
                            family=contract["simulator"],
                            kind=kind,
                            scope=scope,
                            horizon_steps=step,
                            horizon_s=step * contract["contract"]["dt_s"],
                            count=int(valid.sum()),
                            scores=scores,
                            tail_velocity=tail[0],
                            tail_rate=tail[1],
                        )
                    )
    return result


def dart_queries(baseline):
    spec = read(baseline / "recordings/manifest.json")
    records = [
        x
        for x in spec
        if x.get("path", "").startswith("recordings/dart/") and "test-" in x["path"]
    ]
    require(len(records) == 2, "Dart held-out roster")
    out = {k: [] for k in FIELDS}
    keys = []
    for record in records:
        data = arrays(baseline / record["path"])
        x = observed(data["states"])
        u = data["controls"]
        for origin in (50, 75, 100, 125, 150, 175):
            require(origin + 120 < len(x), "Dart horizon unavailable")
            keys.append(dict(recording=record["path"], origin=origin))
            for k, value in zip(
                FIELDS,
                (
                    x[origin - 50 : origin + 1],
                    u[origin - 50 : origin],
                    u[origin : origin + 120],
                    x[origin + 1 : origin + 121],
                ),
                strict=True,
            ):
                out[k].append(value)
    return {k: np.array(v) for k, v in out.items()}, keys


def evaluate(arm, fit_root, output):
    import jax

    from glassbox import LearnedDynamics

    spec, bound, baseline = start(output)
    require(arm in ("baseline", "candidate", "incumbent"), "evaluation arm")
    require(
        bound["model_source"]["files"]
        == spec["model_package_files"][
            "candidate" if arm == "candidate" else "baseline"
        ],
        "eval package differs",
    )
    models = {}
    for name in spec["models"]:
        path = (
            baseline / "models" / f"{name}.npz"
            if arm == "incumbent"
            else fit_root / arm / name / "model.npz"
        )
        if arm != "incumbent":
            fit = path.parent
            authenticate(fit, digest(fit / "manifest.json"))
            require(read(fit / "protocol.json") == spec, "fit protocol differs")
            require(
                read(fit / "binding.json")["model_source"]["files"]
                == bound["model_source"]["files"],
                "fit/eval implementation differs",
            )
        models[name] = LearnedDynamics.load(path)
    predictions = {}
    for cohort, contract in read(baseline / "flights/spec.json")["cohorts"].items():
        data = arrays(baseline / "flights" / f"{cohort}.npz")
        values = predict_queries(models[contract["simulator"]], data)
        predictions[cohort] = values
        checkpoint(output / f"{cohort}.npz", **values)
        eligible = np.flatnonzero(
            (values["lengths"] == data["future_inputs"].shape[1])
            & np.isfinite(data["target"]).all((1, 2))
        )[:4]
        checkpoint(
            output / f"{cohort}-derivatives.npz",
            **derivative_checks(models[contract["simulator"]], data, eligible),
        )
        print(json.dumps(dict(arm=arm, cohort=cohort, status="scored")), flush=True)
    data, keys = dart_queries(baseline)
    prediction = np.asarray(
        jax.jit(models["dart"].predict)(
            data["past_states"], data["past_inputs"], data["future_inputs"]
        )
    )
    checkpoint(
        output / "dart-heldout.npz", prediction=prediction, truth=data["future_states"]
    )
    write(output / "dart-keys.json", keys)
    checkpoint(
        output / "dart-derivatives.npz",
        **derivative_checks(models["dart"], data, [0, 1, 6, 7]),
    )
    result = dict(
        arm=arm,
        flights=score_flights(baseline, predictions),
        dart=[
            dict(
                horizon_steps=i,
                **metrics(prediction[:, i - 1], data["future_states"][:, i - 1]),
            )
            for i in (1, 25, 60, 120)
        ],
    )
    require(np.isfinite(prediction).all(), "nonfinite Dart held-out prediction")
    check_source(bound)
    write(output / "result.json", result)
    write(
        output / "models.json",
        {
            name: dict(
                path=str(
                    (baseline / "models" / f"{name}.npz")
                    if arm == "incumbent"
                    else (fit_root / arm / name / "model.npz")
                ),
                fingerprint=model.fingerprint(),
            )
            for name, model in models.items()
        },
    )
    print(
        json.dumps(
            dict(
                output=str(output),
                manifest_sha256=seal(
                    output, "glassbox-accumulator-offline-evaluation-v1"
                ),
            )
        ),
        flush=True,
    )


def compare(candidate, reference):
    require(len(candidate) == len(reference), "metric roster mismatch")
    rows = []
    for a, b in zip(candidate, reference, strict=True):
        for key in ("cohort", "kind", "scope", "horizon_steps", "count"):
            require(a[key] == b[key], "metric identity differs")
        ratios = {
            key: a["scores"][key] / max(b["scores"][key], 1e-9) for key in a["scores"]
        }
        ratios["primary"] = geometric(
            [ratios[key] for key in ("velocity_rmse_m_s", "body_rate_rmse_rad_s")]
        )
        ratios["tail"] = geometric(
            [a[key] / max(b[key], 1e-9) for key in ("tail_velocity", "tail_rate")]
        )
        rows.append(
            dict(
                family=a["family"],
                kind=a["kind"],
                cohort=a["cohort"],
                scope=a["scope"],
                horizon_steps=a["horizon_steps"],
                ratios=ratios,
            )
        )
    families = {}
    for family in sorted({r["family"] for r in rows}):
        families[family] = {
            kind: {
                key: geometric(
                    [
                        r["ratios"][key]
                        for r in rows
                        if r["family"] == family and r["kind"] == kind
                    ]
                )
                for key in rows[0]["ratios"]
            }
            for kind in ("factual", "response")
        }
    totals = {
        kind: {
            key: geometric([v[kind][key] for v in families.values()])
            for key in rows[0]["ratios"]
        }
        for kind in ("factual", "response")
    }
    return dict(rows=rows, families=families, aggregate=totals)


def qualification_checks(result, spec):
    limits = spec["checks"]
    checks = {
        "derivatives": all(
            d["passed"] for arm in result["arms"].values() for d in arm["derivatives"]
        )
    }
    for reference, comparison in result["comparisons"].items():
        aggregate = comparison["aggregate"]
        checks[reference] = {
            "forecast_primary": aggregate["factual"]["primary"]
            <= limits["forecast_primary_ratio_max"],
            "response_primary": aggregate["response"]["primary"]
            <= limits["response_primary_ratio_max"],
            "each_family_primary": all(
                geometric([scores["primary"] for scores in family.values()])
                <= limits["each_family_primary_ratio_max"]
                for family in comparison["families"].values()
            ),
            **{
                label: geometric([scores[metric] for scores in aggregate.values()])
                <= limits[limit]
                for label, metric, limit in (
                    ("rate", "body_rate_rmse_rad_s", "aggregate_rate_ratio_max"),
                    (
                        "orientation",
                        "orientation_rmse_rad",
                        "aggregate_orientation_ratio_max",
                    ),
                    ("tail", "tail", "aggregate_tail_ratio_max"),
                )
            },
        }
    return checks


def verify(output, authorities):
    spec = read(PROTOCOL)
    baseline = Path(spec["baseline"]["path"])
    authenticate(baseline, spec["baseline"]["manifest_sha256"])
    summaries = {}
    require(
        set(authorities) == set(spec["arms"]),
        "evaluation roster differs",
    )
    for arm, authority in authorities.items():
        folder = output / arm
        authenticate(folder, authority)
        require(read(folder / "protocol.json") == spec, "evaluation protocol differs")
        saved = read(folder / "result.json")
        require(saved["arm"] == arm, "evaluation arm differs")
        require(
            read(folder / "binding.json")["model_source"]["files"]
            == spec["model_package_files"][
                "candidate" if arm == "candidate" else "baseline"
            ],
            "evaluation source differs",
        )
        predictions = {
            c: arrays(folder / f"{c}.npz")
            for c in read(baseline / "flights/spec.json")["cohorts"]
        }
        for cohort, values in predictions.items():
            source = arrays(baseline / "flights" / f"{cohort}.npz")
            for suffix, key in (("", "future_inputs"), ("_factual", "factual_inputs")):
                lengths = command_lengths(source, key)
                exact(values["lengths" + suffix], lengths, "command availability")
                mask = np.arange(source[key].shape[1])[None, :] < lengths[:, None]
                require(
                    np.isfinite(values["predicted" + suffix][mask]).all(),
                    "eligible prediction missing",
                )
                require(
                    np.isnan(values["predicted" + suffix][~mask]).all(),
                    "invalid prefix predicted",
                )
        require(
            score_flights(baseline, predictions) == saved["flights"],
            "saved metrics differ",
        )
        d = arrays(folder / "dart-heldout.npz")
        truth, keys = dart_queries(baseline)
        exact(d["truth"], truth["future_states"], "Dart truth")
        require(keys == read(folder / "dart-keys.json"), "Dart roster differs")
        require(
            saved["dart"]
            == [
                dict(
                    horizon_steps=i,
                    **metrics(d["prediction"][:, i - 1], d["truth"][:, i - 1]),
                )
                for i in (1, 25, 60, 120)
            ],
            "Dart metrics differ",
        )
        derivative = []
        expected_files = {
            c + "-derivatives.npz"
            for c in read(baseline / "flights/spec.json")["cohorts"]
        } | {"dart-derivatives.npz"}
        require(
            {p.name for p in folder.glob("*-derivatives.npz")} == expected_files,
            "derivative roster differs",
        )
        for path in sorted(folder.glob("*-derivatives.npz")):
            a = arrays(path)
            if path.name == "dart-derivatives.npz":
                expected_rows = np.array([0, 1, 6, 7])
            else:
                cohort = path.name.removesuffix("-derivatives.npz")
                source = arrays(baseline / "flights" / f"{cohort}.npz")
                expected_rows = np.flatnonzero(
                    (
                        command_lengths(source, "future_inputs")
                        == source["future_inputs"].shape[1]
                    )
                    & np.isfinite(source["target"]).all((1, 2))
                )[:4]
            exact(a["row"], expected_rows, "derivative query identity")
            require(
                all(np.isfinite(v).all() for v in a.values()), "nonfinite derivative"
            )
            error = abs(a["jvp"] - a["fd_fine"])
            limit = 1e-6 + 2e-4 * abs(a["jvp"])
            derivative.append(
                dict(
                    file=path.name,
                    checks=int(error.size),
                    passed=bool(np.all(error <= limit)),
                    max_error=float(error.max()),
                    max_normalized_error=float((error / limit).max()),
                )
            )
        summaries[arm] = dict(saved, derivatives=derivative)
    references = [arm for arm in spec["arms"] if arm != "candidate"]
    result = dict(
        arms=summaries,
        comparisons={
            arm: compare(summaries["candidate"]["flights"], summaries[arm]["flights"])
            for arm in references
        },
        adopted=False,
    )
    result["checks"] = qualification_checks(result, spec)
    result["qualified"] = result["checks"]["derivatives"] and all(
        all(result["checks"][reference].values()) for reference in references
    )
    return result


def replay(index):
    """Replay the qualified candidate's saved forecasts without fitting."""
    import jax
    from verify_baseline import runtime_check

    from glassbox import LearnedDynamics

    spec = read(PROTOCOL)
    baseline = Path(spec["baseline"]["path"])
    authenticate(baseline, spec["baseline"]["manifest_sha256"])
    runtime_check(read(baseline / "dart/spec.json")["runtime"])
    fits = Path(index["fits"]["path"])
    authenticate(fits, index["fits"]["manifest_sha256"])
    evaluation = Path(index["evaluation"]["path"])
    folder = evaluation / "candidate"
    authenticate(folder, index["evaluation"]["authorities"]["candidate"])
    require(read(folder / "protocol.json") == spec, "replay protocol differs")
    models = {}
    identities = read(folder / "models.json")
    for name in spec["models"]:
        models[name] = LearnedDynamics.load(fits / "candidate" / name / "model.npz")
        require(
            models[name].fingerprint() == identities[name]["fingerprint"],
            "replay revision differs",
        )
    count = 0
    for cohort, contract in read(baseline / "flights/spec.json")["cohorts"].items():
        data = arrays(baseline / "flights" / f"{cohort}.npz")
        actual = predict_queries(models[contract["simulator"]], data)
        expected = arrays(folder / f"{cohort}.npz")
        for key in actual:
            exact(actual[key], expected[key], cohort + "/" + key)
            count += 1
    data, _ = dart_queries(baseline)
    actual = np.asarray(
        jax.jit(models["dart"].predict)(
            data["past_states"], data["past_inputs"], data["future_inputs"]
        )
    )
    exact(actual, arrays(folder / "dart-heldout.npz")["prediction"], "Dart forecasts")
    return dict(
        replayed_arrays=count + 1, models=3, fits=0, optimizer_calls=0, exact=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "evaluate", "verify", "replay"))
    parser.add_argument("--arm", choices=("baseline", "candidate", "incumbent"))
    parser.add_argument("--model", choices=("dart", "crazyflow", "cascade"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fit-root", type=Path)
    parser.add_argument("--authorities", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    if args.mode == "fit":
        fit_one(args.model, args.arm, args.output)
    elif args.mode == "evaluate":
        evaluate(args.arm, args.fit_root, args.output)
    elif args.mode == "verify":
        if args.index is not None:
            index = read(args.index)
            authenticate(Path(index["fits"]["path"]), index["fits"]["manifest_sha256"])
            result = verify(
                Path(index["evaluation"]["path"]), index["evaluation"]["authorities"]
            )
        else:
            result = verify(args.output, read(args.authorities))
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(replay(read(args.index)), indent=2))


if __name__ == "__main__":
    main()
