"""Representation ablation and a separately frozen new-recording evaluation."""

import argparse
import hashlib
import json
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import jax
import numpy as np
from experiment_forecast_diagnosis import references
from sequence_transfer_data import write_json

from glassbox.experimental.direct_forecast import (
    REPRESENTATIONS,
    DirectForecast,
    fit_direct_forecast,
    prefix_features,
)
from glassbox.experimental.sequence_model import (
    SequenceBatch,
    SequenceModel,
    _rollout,
    initialize_sequence_model,
    sequence_windows,
)

ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
RIDGES = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)


def errors(pred, target, groups):
    return np.stack(
        [
            np.sum((pred[:, :, list(g)] - target[:, :, list(g)]) ** 2, -1)
            for g in groups
        ],
        -1,
    )


def score(pred, target, scale, groups):
    return dict(
        loss=float(np.mean(((pred - target) / scale) ** 2)),
        rmse=np.sqrt(errors(pred, target, groups).mean(0)).tolist(),
    )


def predict(model, batch, horizons):
    if isinstance(model, DirectForecast):
        return np.asarray(
            model.predict(batch.past_states, batch.past_inputs, batch.future_inputs)
        )
    return np.asarray(
        RECUR(
            model.params,
            model.norms,
            model.kind,
            batch.past_states,
            batch.past_inputs,
            batch.future_inputs,
        )
    )[:, np.array(horizons) - 1]


RECUR = jax.jit(_rollout, static_argnames=("kind",))


def features(batch):
    return prefix_features(
        batch.past_states, batch.past_inputs, batch.future_inputs, 1, use_history=True
    )


def local_choice(
    training_features,
    development_features,
    development_errors,
    query_features,
    query_predictions,
    groups,
    *,
    neighbors=32,
):
    """Compare empirical squared errors among nearby development observations.

    This does not produce an uncertainty interval or a coherent step model.
    Query outcomes are deliberately absent from this interface.
    """
    mean, scale = training_features.mean(0), training_features.std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    ref = (development_features - mean) / scale
    q = (query_features - mean) / scale
    indices = np.array(
        [
            np.argsort(np.mean((ref - x) ** 2, 1), kind="stable")[
                : min(neighbors, len(ref))
            ]
            for x in q
        ]
    )
    # Errors [candidate, observation, horizon, group].
    estimated = development_errors[:, indices].mean(2).transpose(1, 0, 2, 3)
    choices = estimated.argmin(1)
    output = np.empty_like(query_predictions[0])
    for row in range(len(q)):
        for h in range(output.shape[1]):
            for g, columns in enumerate(groups):
                output[row, h, list(columns)] = query_predictions[
                    choices[row, h, g], row, h, list(columns)
                ]
    return output, dict(
        indices=indices, estimated_squared_error=estimated, choices=choices
    )


def archive_sources(output):
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__),
        root / "scripts/prepare_crazyflie_reference.py",
        root / "scripts/experiment_forecast_diagnosis.py",
        root / "scripts/sequence_transfer_data.py",
        *sorted((root / "src/glassbox/experimental").glob("*.py")),
    ]
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in files:
            z.write(p, p.relative_to(root))
    write_json(
        output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )


def plan(
    output,
    phase,
    *,
    new_data_budget="min(384, number of complete training origins), seed 60; same longest powered interval rule; no performance-based segment selection",
):
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "plan.json",
        dict(
            phase=phase,
            created_utc=datetime.now(UTC).isoformat(),
            representations=REPRESENTATIONS,
            ridge_fractions=RIDGES,
            additional_family="recursive history with identical ridge fractions",
            selection="minimum development mean squared error standardized by train observation scale, across all outputs and three horizons",
            local_selector="selected whole-model candidate, hold, trend; 32 nearest development origins in train-standardized full origin/history features; minimum local empirical squared vector error per horizon/group",
            representation_hypotheses=dict(
                relative="drop current observation levels, retain relative observation history; predicts changes equivariant to additive observation offsets",
                linear_history="fit anchored degree-one polynomial to past relative observations and commands",
                quadratic_history="anchored degree-two polynomial history",
                pca="retain 95% or 99% of training standardized-feature variance; each horizon basis uses only training inputs",
            ),
            limits=[
                "Representation options are hypotheses, not universal physical invariances.",
                "No platform equations or learned weights transfer between datasets.",
                "Local error ranking is not a calibrated interval or a coherent trajectory model.",
                "New CF outputs are accelerometer g and gyro rad/s; no velocity/attitude is inferred.",
            ],
            new_data_roles={
                "log10": "train",
                "log15": "development",
                "log16": "evaluation",
            },
            new_data_budget=new_data_budget,
        ),
    )
    archive_sources(output)


def fit_case(folder, batches, dt, horizons, groups, identity, *, evaluate=True):
    folder.mkdir(parents=True, exist_ok=False)
    train, dev = batches["train"], batches["development"]
    indices = np.array(horizons) - 1
    norm = initialize_sequence_model(train, kind="linear")
    scale = norm.norms["state_scale"]
    roles = ("development", "evaluation") if evaluate else ("development",)
    predictions = {role: references(batches[role], horizons) for role in roles}
    candidates = []
    for representation in REPRESENTATIONS:
        for ridge in RIDGES:
            name = f"{representation}-r{ridge:g}"
            model = fit_direct_forecast(
                train, horizons, representation=representation, ridge_fraction=ridge
            )
            model.save(folder / f"{name}.npz")
            for role in roles:
                predictions[role][name] = predict(model, batches[role], horizons)
            candidates.append(
                dict(
                    name=name,
                    family=representation,
                    fingerprint=model.fingerprint(),
                    head_widths=[
                        model.arrays[f"weight_{h}"].shape[0] for h in horizons
                    ],
                )
            )
    for ridge in RIDGES:
        name = f"recursive-r{ridge:g}"
        model = initialize_sequence_model(
            train,
            kind="delay",
            ridge=ridge * len(train.past_states) * train.future_states.shape[1],
        )
        model.save(folder / f"{name}.npz")
        for role in roles:
            predictions[role][name] = predict(model, batches[role], horizons)
        candidates.append(
            dict(name=name, family="recursive", fingerprint=model.fingerprint())
        )
    targets = {role: batches[role].future_states[:, indices] for role in roles}
    metrics = {
        role: {name: score(p, targets[role], scale, groups) for name, p in pred.items()}
        for role, pred in predictions.items()
    }
    losses = {n: v["loss"] for n, v in metrics["development"].items()}
    chosen = min(losses, key=losses.get)
    family = {
        f: min([c["name"] for c in candidates if c["family"] == f], key=losses.get)
        for f in [*REPRESENTATIONS, "recursive"]
    }
    report = dict(
        **identity,
        dt_s=dt,
        horizons=horizons,
        groups=groups,
        state_scale=scale.tolist(),
        candidates=candidates,
        selected=chosen,
        selected_families=family,
        scores=metrics,
    )
    np.savez_compressed(
        folder / "samples.npz",
        **{f"{r}_{k}": getattr(b, k) for r, b in batches.items() for k in ARRAYS},
    )
    np.savez_compressed(
        folder / "predictions.npz",
        **{f"{r}__{n}": p for r, ps in predictions.items() for n, p in ps.items()},
    )
    pool = list(dict.fromkeys([chosen, "hold", "trend"]))
    calibration = features(dev)
    residuals = np.array(
        [
            errors(predictions["development"][n], targets["development"], groups)
            for n in pool
        ]
    )
    np.savez_compressed(
        folder / "local-reference.npz",
        training_features=features(train),
        development_features=calibration,
        development_errors=residuals,
    )
    report["local_pool"] = pool
    if evaluate:
        means = np.array([predictions["evaluation"][n] for n in pool])
        local, detail = local_choice(
            features(train),
            calibration,
            residuals,
            features(batches["evaluation"]),
            means,
            groups,
        )
        np.savez_compressed(
            folder / "local-predictions.npz", prediction=local, **detail
        )
        report["scores"]["evaluation"]["local_selector"] = score(
            local, targets["evaluation"], scale, groups
        )
    write_json(folder / "report.json", report)
    return report


def ablation(args):
    plan(args.output, "reused-recording representation ablation")
    reports = []
    for old in sorted(args.previous.glob("*/*")):
        if not (old / "report.json").exists():
            continue
        info = json.loads((old / "report.json").read_text())
        with np.load(old / "samples.npz") as z:
            batches = {
                r: SequenceBatch(
                    **{k: z[f"{r}_{k}"] for k in ARRAYS}, dt_s=info["dt_s"]
                )
                for r in ("train", "development", "evaluation")
            }
        identity = {k: info[k] for k in ("dataset", "fold", "seed")}
        start = time.monotonic()
        r = fit_case(
            args.output / info["dataset"] / old.name,
            batches,
            info["dt_s"],
            tuple(info["horizons"]),
            ((0, 1, 2), (3, 4, 5), tuple(range(6, 15))),
            identity,
        )
        reports.append(r)
        write_json(args.output / "summary.json", reports)
        print(
            json.dumps(
                dict(
                    **identity,
                    selected=r["selected"],
                    full=r["scores"]["evaluation"][r["selected_families"]["full"]][
                        "rmse"
                    ][-1][:2],
                    chosen=r["scores"]["evaluation"][r["selected"]]["rmse"][-1][:2],
                    seconds=round(time.monotonic() - start, 2),
                )
            ),
            flush=True,
        )


def cf_windows(prepared, role):
    name = {"train": "log10", "development": "log15", "evaluation": "log16"}[role]
    meta = json.loads((prepared / f"{name}.json").read_text())
    with np.load(prepared / f"{name}.npz") as z:
        states, inputs = z["states"], z["inputs"]
    pool = np.arange(5, len(states) - 12)
    anchors = (
        np.random.default_rng(60).permutation(pool)[: min(384, len(pool))]
        if role == "train"
        else pool[::5]
    )
    return sequence_windows(
        states, inputs, anchors, history_steps=5, horizon_steps=12, dt_s=0.02
    ), dict(
        recording=name,
        role=role,
        origins=anchors.tolist(),
        windows=len(anchors),
        prepared_sha256=hashlib.sha256(
            (prepared / f"{name}.npz").read_bytes()
        ).hexdigest(),
        raw_sha256=meta["metadata"]["sha256"],
    )


def freeze_new(args):
    plan(
        args.output,
        "new Crazyflie fit and selection; no evaluation observations loaded",
    )
    data = {
        role: cf_windows(args.prepared_cf, role) for role in ("train", "development")
    }
    batches = {r: v[0] for r, v in data.items()}
    write_json(args.output / "usage.json", {r: v[1] for r, v in data.items()})
    report = fit_case(
        args.output / "fitted",
        batches,
        0.02,
        (1, 5, 12),
        ((0, 1, 2), (3, 4, 5)),
        dict(dataset="crazyflie_sensor", seed=60),
        evaluate=False,
    )
    files = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    write_json(
        args.output / "frozen.json",
        dict(
            created_utc=datetime.now(UTC).isoformat(),
            selected=report["selected"],
            fixed_evaluation_arms=[
                "selected",
                "full",
                "recursive",
                "hold",
                "trend",
                "local_selector",
            ],
            files=files,
        ),
    )
    print(
        json.dumps(
            dict(
                selected=report["selected"],
                development={
                    n: report["scores"]["development"][
                        report["selected_families"].get(n, n)
                    ]["rmse"][-1]
                    for n in ("full", "recursive", "hold", "trend")
                },
            )
        ),
        flush=True,
    )


def evaluate_new(args):
    frozen = json.loads((args.frozen / "frozen.json").read_text())
    for p, h in frozen["files"].items():
        if hashlib.sha256((args.frozen / p).read_bytes()).hexdigest() != h:
            raise ValueError("frozen artifact changed")
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.frozen / "fitted/report.json").read_text())
    batch, usage = cf_windows(args.prepared_cf, "evaluation")
    write_json(args.output / "usage.json", usage)
    np.savez_compressed(
        args.output / "samples.npz", **{k: getattr(batch, k) for k in ARRAYS}
    )
    names = {
        "selected": report["selected"],
        "full": report["selected_families"]["full"],
        "recursive": report["selected_families"]["recursive"],
        "hold": "hold",
        "trend": "trend",
    }
    means = references(batch, tuple(report["horizons"]))
    for name in set(names.values()) - set(means):
        cls = SequenceModel if name.startswith("recursive") else DirectForecast
        model = cls.load(args.frozen / "fitted" / f"{name}.npz")
        means[name] = predict(model, batch, report["horizons"])
    pool = report["local_pool"]
    with np.load(args.frozen / "fitted/local-reference.npz") as ref:
        local, detail = local_choice(
            ref["training_features"],
            ref["development_features"],
            ref["development_errors"],
            features(batch),
            np.array([means[n] for n in pool]),
            report["groups"],
        )
    result = {arm: means[name] for arm, name in names.items()}
    result["local_selector"] = local
    np.savez_compressed(args.output / "predictions.npz", **result)
    np.savez_compressed(args.output / "local-selection.npz", **detail)
    metrics = {
        arm: score(
            pred,
            batch.future_states[:, np.array(report["horizons"]) - 1],
            np.array(report["state_scale"]),
            report["groups"],
        )
        for arm, pred in result.items()
    }
    write_json(
        args.output / "report.json",
        dict(
            evaluated_utc=datetime.now(UTC).isoformat(),
            frozen_sha256=hashlib.sha256(
                (args.frozen / "frozen.json").read_bytes()
            ).hexdigest(),
            selected=report["selected"],
            horizons=report["horizons"],
            names=names,
            scores=metrics,
        ),
    )
    print(json.dumps(metrics), flush=True)


def expand_training(args):
    from prepare_crazyflie_reference import HASHES, decode

    from glassbox.experimental.causal_sampling import causal_hold

    plan(
        args.output,
        "adaptive follow-up after reserved evaluation: expand only training segment coverage",
        new_data_budget="same training-window count as frozen fit; seed 60; sample complete windows from all eligible powered intervals of log10; development and evaluation retain their original longest interval",
    )
    path = args.raw_cf / "log10"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES["log10"]
    raw = decode(path)["fixedFrequency"]
    time_s = raw["timestamp_s"]
    values = np.column_stack(
        [raw[f"acc.{c}"] for c in "xyz"]
        + [raw[f"gyro.{c}"] for c in "xyz"]
        + [raw[f"motor.m{i}"] for i in range(1, 5)]
    )
    grid = np.arange(np.ceil(time_s[0] / 0.02), np.floor(time_s[-1] / 0.02) + 1) * 0.02
    held = causal_hold(time_s, values, grid, maximum_age_s=0.01)
    valid = held.valid & (held.values[:, 6:].mean(1) / 65536 > 0.1)
    runs = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int))).reshape(
        -1, 2
    )
    eligible = []
    for first, last in runs:
        eligible.extend(
            (int(first), int(last), int(origin))
            for origin in range(first + 5, last - 12)
        )
    baseline = json.loads((args.frozen / "usage.json").read_text())["train"]["windows"]
    selection = np.random.default_rng(60).permutation(len(eligible))[:baseline]
    chosen = np.array([eligible[i] for i in selection])
    anchors = chosen[:, 2]
    data = held.values
    observations = np.column_stack((data[:, :3], np.deg2rad(data[:, 3:6])))
    commands = data[:-1, 6:] / 65536
    train = sequence_windows(
        observations, commands, anchors, history_steps=5, horizon_steps=12, dt_s=0.02
    )
    batches = {"train": train}
    for role in ("development", "evaluation"):
        batches[role] = cf_windows(args.prepared_cf, role)[0]
    write_json(
        args.output / "usage.json",
        dict(
            adaptive=True,
            raw_sha256=HASHES["log10"],
            candidate_origins=len(eligible),
            windows=len(anchors),
            segment_and_origin=chosen.tolist(),
            training_unique_state_rows=len(
                np.unique(anchors[:, None] + np.arange(-5, 13))
            ),
            development_and_evaluation="identical to first reserved test; their results have now been observed",
        ),
    )
    report = fit_case(
        args.output / "fitted",
        batches,
        0.02,
        (1, 5, 12),
        ((0, 1, 2), (3, 4, 5)),
        dict(dataset="crazyflie_sensor_expanded", seed=60),
    )
    print(
        json.dumps(
            dict(
                selected=report["selected"],
                full=report["scores"]["evaluation"][
                    report["selected_families"]["full"]
                ]["rmse"][-1],
                chosen=report["scores"]["evaluation"][report["selected"]]["rmse"][-1],
                local=report["scores"]["evaluation"]["local_selector"]["rmse"][-1],
                hold=report["scores"]["evaluation"]["hold"]["rmse"][-1],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase",
        choices=("ablation", "freeze-new", "evaluate-new", "expand-training"),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--previous",
        type=Path,
        default=Path("../artifacts/forecast-diagnosis/comparison-01"),
    )
    p.add_argument("--prepared-cf", type=Path)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--raw-cf", type=Path)
    a = p.parse_args()
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded experiments require float64")
    {
        "ablation": ablation,
        "freeze-new": freeze_new,
        "evaluate-new": evaluate_new,
        "expand-training": expand_training,
    }[a.phase](a)
