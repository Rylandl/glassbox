"""One frozen offline acceptance decision for the maintained generic fitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import jax
import numpy as np
import scipy

from glassbox.experimental._sequence_fit import (
    _BOUNDED_FIT,
    _fit_bounded_sequence_model,
)
from glassbox.experimental.sequence_model import (
    SequenceBatch,
    SequenceModel,
    initialize_sequence_model,
)

_MANIFEST_SHA256 = "c2c2f157da0c1b2d973b8d6a38d0ab08e2caeb3f8713138491cf4973738d6ae4"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_manifest(path):
    if sha(path) != _MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen M1 contract")
    return read(path)


def measure(predicted, target, scale):
    squared = (predicted - target) ** 2
    return dict(
        channel_rmse=np.sqrt(squared.mean(axis=0)).tolist(),
        horizon_scaled_rmse=np.sqrt((squared / scale**2).mean(axis=(0, 2))).tolist(),
        overall_scaled_rmse=float(np.sqrt((squared / scale**2).mean())),
    )


def recurrence(model, x, up, uf):
    """Independent NumPy replay of the frozen delay-MLP; no training code or JAX."""
    if model.kind != "delay_mlp":
        raise ValueError("M1 replay requires the frozen delay_mlp representation")
    n, p = model.norms, model.params
    history = (x - n["state_mean"]) / n["state_scale"]
    commands = (up - n["input_mean"]) / n["input_scale"]
    future = (uf - n["input_mean"]) / n["input_scale"]
    output = []
    for command in future.transpose(1, 0, 2):
        current = history[:, -1]
        features = (
            np.column_stack(
                (
                    current,
                    command,
                    (history[:, :-1] - current[:, None]).reshape(len(x), -1),
                    (commands - command[:, None]).reshape(len(x), -1),
                )
            )
            / n["feature_scale"]
        )
        delta = features @ p["linear"] + p["bias"]
        delta += np.tanh(features @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[:, 1:], predicted[:, None]), axis=1)
        commands = np.concatenate((commands[:, 1:], command[:, None]), axis=1)
    return np.stack(output, axis=1)


def decide(manifest, rows):
    required = manifest["required_families"]
    expected = {
        (f, s, False)
        for f in required + manifest["stress_families"]
        for s in manifest["data_seeds"]
    } | {
        (c["family"], c["data_seed"], True) for c in manifest["extra_regression_cases"]
    }
    keys = [(r["family"], r["data_seed"], r["extra"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(
            "acceptance requires exactly the frozen cases, without duplicates"
        )
    breaches = []
    logs = {f: [] for f in required}
    for row in rows:
        if row["status"] != "complete":
            breaches.append(dict(case=row["name"], gate="fit_complete"))
            continue
        if not row["correctness_passed"]:
            breaches.append(dict(case=row["name"], gate="correctness"))
        if set(row["regimes"]) != {"matched", "shifted"}:
            raise ValueError("acceptance requires both frozen evaluation regimes")
        usage = row["optimization"]["resource_usage"]
        for key, limit in (
            (
                "training_window_gradient_evaluations",
                manifest["compute"]["max_training_window_gradient_evaluations"],
            ),
            ("development_passes", manifest["compute"]["max_development_passes"]),
        ):
            if not isinstance(usage[key], int) or not 0 <= usage[key] <= limit:
                breaches.append(
                    dict(case=row["name"], gate=key, value=usage[key], limit=limit)
                )
        for regime, metrics in row["regimes"].items():
            horizons = np.asarray(metrics["candidate"]["horizon_scaled_rmse"])
            if horizons.ndim != 1 or not len(horizons):
                raise ValueError("acceptance requires nonempty horizon error evidence")
            cap = manifest["error_caps"][row["family"]][regime]
            if (
                not np.isfinite(horizons).all()
                or np.any(horizons < 0)
                or np.any(horizons > cap)
            ):
                breaches.append(
                    dict(
                        case=row["name"],
                        regime=regime,
                        gate="absolute_horizon_error",
                        value=float(horizons.max())
                        if np.isfinite(horizons).all()
                        else None,
                        limit=cap,
                    )
                )
            a = metrics["incumbent"]["overall_scaled_rmse"]
            b = metrics["candidate"]["overall_scaled_rmse"]
            if not np.isfinite([a, b]).all() or min(a, b) < 0:
                breaches.append(
                    dict(
                        case=row["name"], regime=regime, gate="finite_nonnegative_score"
                    )
                )
                continue
            if row["family"] in required and not row["extra"]:
                floor = manifest["primary"]["floor"]
                logs[row["family"]].append(float(np.log(max(b, floor) / max(a, floor))))
    complete = all(len(v) == len(manifest["data_seeds"]) * 2 for v in logs.values())
    ratios = {f: float(np.exp(np.mean(v))) for f, v in logs.items() if v}
    primary = (
        float(np.exp(np.mean([np.mean(v) for v in logs.values()])))
        if complete
        else None
    )
    accepted = bool(
        not breaches and complete and primary <= manifest["primary"]["maximum_ratio"]
    )
    return dict(
        manifest=manifest["id"],
        decision="replace" if accepted else "retain",
        accepted=accepted,
        hard_gate_breaches=breaches,
        primary_ratio=primary,
        required_ratio=manifest["primary"]["maximum_ratio"],
        per_family_ratios=ratios,
        meaning="Engineering regression qualification for this manifest only; no platform/control-readiness or uncertainty claim.",
    )


def verify_run(directory, manifest_path):
    """Recompute scores and fit diagnostics from preserved arrays; never optimize."""
    manifest = read(directory / "manifest.json")
    if manifest != frozen_manifest(manifest_path):
        raise ValueError("saved manifest differs from the frozen acceptance contract")
    with zipfile.ZipFile(directory / "executed-sources.zip") as archive:
        for name, digest in read(directory / "sources.json").items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"executed source hash mismatch: {name}")
    hashes = read(directory / "incumbent-artifacts.json")
    rows = read(directory / "results.json")
    losses, checks = {}, 0
    for row in rows:
        case = directory / row["name"]
        if read(case / "result.json") != row:
            raise ValueError(f"case result mismatch: {row['name']}")
        optimization = read(case / "optimization.json")
        if optimization != row["optimization"]:
            raise ValueError(f"optimization evidence mismatch: {row['name']}")
        models = {
            k: SequenceModel.load(case / f"{k}.npz") for k in ("incumbent", "candidate")
        }
        for k, model in models.items():
            if model.fingerprint() != row[f"{k}_fingerprint"]:
                raise ValueError(f"model fingerprint mismatch: {row['name']}/{k}")
        identities, objective_losses = {}, {}
        scale = np.asarray(optimization["error_scale"])
        for role in ("train", "development"):
            cache = case / f"{role}.npz"
            original = str(Path(row["source"]) / cache.name)
            if sha(cache) != hashes[original]:
                raise ValueError(f"input cache hash mismatch: {row['name']}/{role}")
            with np.load(cache, allow_pickle=False) as data:
                identities[role] = set(data["recording_ids"])
                for kind, model in models.items():
                    prediction = recurrence(
                        model,
                        data["past_states"],
                        data["past_inputs"],
                        data["future_inputs"],
                    )
                    objective_losses[f"{kind}_{role}_mse"] = float(
                        np.mean(((prediction - data["future_states"]) / scale) ** 2)
                    )
        if identities["train"] & identities["development"]:
            raise ValueError("training and development recording identities overlap")
        np.testing.assert_allclose(
            objective_losses["candidate_train_mse"],
            optimization["selected_training_mse"],
            rtol=1e-8,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            objective_losses["candidate_development_mse"],
            optimization["validation_rollout_mse"],
            rtol=1e-8,
            atol=1e-12,
        )
        losses[row["name"]] = objective_losses
        for regime in ("matched", "shifted"):
            with np.load(case / f"{regime}.npz", allow_pickle=False) as data:
                if set(data["recording_ids"]) & (
                    identities["train"] | identities["development"]
                ):
                    raise ValueError("evaluation recording identities overlap fit data")
                for kind, model in models.items():
                    prediction = recurrence(
                        model,
                        data["past_states"],
                        data["past_inputs"],
                        data["future_inputs"],
                    )
                    np.testing.assert_allclose(
                        prediction, data[kind], rtol=1e-8, atol=1e-9
                    )
                    metrics = measure(prediction, data["targets"], data["state_scale"])
                    for metric, value in metrics.items():
                        np.testing.assert_allclose(
                            value,
                            row["regimes"][regime][kind][metric],
                            rtol=1e-8,
                            atol=1e-10,
                        )
                    row["regimes"][regime][kind] = metrics
                    checks += 1
    decision = decide(manifest, rows)
    saved = read(directory / "decision.json")
    for key in ("decision", "accepted", "manifest"):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed acceptance decision differs: {key}")
    np.testing.assert_allclose(
        decision["primary_ratio"], saved["primary_ratio"], rtol=1e-10
    )
    return dict(
        verified_cases=len(rows),
        model_regime_replays=checks,
        decision=decision,
        objective_losses=losses,
        meaning="Saved evidence replay, not a new fit or independent statistical confirmation. Solver call counts and derivative checks remain executed-run evidence.",
    )


def main(args):
    root = Path(__file__).resolve().parents[1]
    manifest = frozen_manifest(args.manifest)
    for key in _BOUNDED_FIT:
        if key in manifest["candidate"]:
            if _BOUNDED_FIT[key] != manifest["candidate"][key]:
                raise ValueError(f"fitter setting differs from frozen contract: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "manifest.json", manifest)
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            precision="scoped float64",
        ),
    )
    sources = (
        sorted(root.glob("src/**/*.py"))
        + sorted(root.glob("scripts/*.py"))
        + [
            root / "pyproject.toml",
            root / "uv.lock",
            root / "tests/test_bounded_sequence_fit.py",
            root / "tests/test_generic_fit_acceptance.py",
        ]
    )
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, str(p.relative_to(root)))
    write(
        args.output / "sources.json",
        {str(p.relative_to(root)): sha(p) for p in sources},
    )
    cases = [
        dict(
            family=f,
            data_seed=s,
            extra=False,
            source_root=manifest["incumbent_sources"]["main"],
        )
        for f in manifest["required_families"] + manifest["stress_families"]
        for s in manifest["data_seeds"]
    ]
    cases += [
        dict(
            family=c["family"],
            data_seed=c["data_seed"],
            extra=True,
            source_root=manifest["incumbent_sources"]["extra"],
        )
        for c in manifest["extra_regression_cases"]
    ]
    # Freeze all incumbent inputs before the first candidate fit.
    artifacts = {}
    for case in cases:
        directory = (
            root / case["source_root"] / f"{case['family']}-{case['data_seed']}-fit0"
        ).resolve()
        for p in directory.glob("*"):
            if p.is_file():
                artifacts[str(p)] = sha(p)
    write(args.output / "incumbent-artifacts.json", artifacts)
    rows = []
    for case in cases:
        name = f"{case['family']}-{case['data_seed']}"
        source = (root / case["source_root"] / f"{name}-fit0").resolve()
        prior = read(source / "result.json")
        selected = prior["selected"]["original"]["original"]["selected_index"] * 100
        incumbent = SequenceModel.load(source / f"original-{selected:04d}.npz")
        directory = args.output / name
        directory.mkdir()
        batches = {}
        cache_ids = {}
        for role in ("train", "development"):
            shutil.copyfile(source / f"{role}.npz", directory / f"{role}.npz")
            with np.load(source / f"{role}.npz", allow_pickle=False) as data:
                batches[role] = SequenceBatch(
                    **{
                        k: data[k]
                        for k in (
                            "past_states",
                            "past_inputs",
                            "future_inputs",
                            "future_states",
                        )
                    },
                    dt_s=manifest["calibration"]["dt_s"],
                )
                cache_ids[role] = data["recording_ids"].copy()
        if set(cache_ids["train"]) & set(cache_ids["development"]):
            raise ValueError("training and development recording identities overlap")
        train, dev = batches["train"], batches["development"]
        recipe = manifest["incumbent_recipe"]
        initial = initialize_sequence_model(
            train,
            kind=recipe["kind"],
            width=recipe["width"],
            seed=recipe["seed"],
            ridge=recipe["ridge_fraction"]
            * len(train.past_states)
            * train.future_states.shape[1],
        )
        if (
            initial.fingerprint()
            != SequenceModel.load(source / "original-0000.npz").fingerprint()
        ):
            raise ValueError("candidate initialization differs from the incumbent")
        scales = np.asarray(prior["loss_scales"]["original"])
        print(json.dumps(dict(starting=name)), flush=True)
        started = time.perf_counter()
        with jax.enable_x64(True):
            candidate, optimization = _fit_bounded_sequence_model(
                train, dev, initial, scales, cache_ids["development"]
            )
            wall = time.perf_counter() - started
            candidate.save(directory / "candidate.npz")
            incumbent.save(directory / "incumbent.npz")
            write(directory / "optimization.json", optimization)
            norm = incumbent.norms["state_scale"]
            for k, v in incumbent.norms.items():
                np.testing.assert_array_equal(v, candidate.norms[k])
            regimes = {}
            replay_max = 0.0
            for regime in ("matched", "shifted"):
                with np.load(source / f"{regime}.npz", allow_pickle=False) as supplied:
                    arrays = {
                        k: supplied[k]
                        for k in (
                            "past_states",
                            "past_inputs",
                            "future_inputs",
                            "targets",
                            "recording_ids",
                            "source_origins",
                        )
                    }
                    old_prediction = supplied["original"][selected // 100]
                if set(arrays["recording_ids"]) & (
                    set(cache_ids["train"]) | set(cache_ids["development"])
                ):
                    raise ValueError("evaluation recording identities overlap fit data")
                predicted = {
                    o: np.asarray(
                        m.rollout(
                            arrays["past_states"],
                            arrays["past_inputs"],
                            arrays["future_inputs"],
                        )
                    )
                    for o, m in (("incumbent", incumbent), ("candidate", candidate))
                }
                np.testing.assert_allclose(
                    predicted["incumbent"], old_prediction, rtol=1e-9, atol=1e-10
                )
                for o, m in (("incumbent", incumbent), ("candidate", candidate)):
                    replay = recurrence(
                        m,
                        arrays["past_states"],
                        arrays["past_inputs"],
                        arrays["future_inputs"],
                    )
                    replay_max = max(
                        replay_max, float(np.max(np.abs(replay - predicted[o])))
                    )
                    np.testing.assert_allclose(
                        replay, predicted[o], rtol=1e-8, atol=1e-9
                    )
                    if not np.isfinite(predicted[o]).all():
                        raise ValueError(f"nonfinite {o} prediction")
                np.savez_compressed(
                    directory / f"{regime}.npz", **arrays, **predicted, state_scale=norm
                )
                regimes[regime] = {
                    o: measure(p, arrays["targets"], norm) for o, p in predicted.items()
                }
                regimes[regime]["recordings"] = {
                    str(r): {
                        o: measure(
                            p[arrays["recording_ids"] == r],
                            arrays["targets"][arrays["recording_ids"] == r],
                            norm,
                        )
                        for o, p in predicted.items()
                    }
                    for r in sorted(set(arrays["recording_ids"]))
                }
            # The returned mean remains differentiable and causally prefix-consistent.
            short = candidate.rollout(
                dev.past_states[:1], dev.past_inputs[:1], dev.future_inputs[:1, :2]
            )
            whole = candidate.rollout(
                dev.past_states[:1], dev.past_inputs[:1], dev.future_inputs[:1]
            )
            np.testing.assert_allclose(short, whole[:, :2], rtol=1e-9, atol=1e-10)
            derivative = jax.jacfwd(
                lambda u, m=candidate, b=dev: m.rollout(
                    b.past_states[0], b.past_inputs[0], u
                )
            )(dev.future_inputs[0])
            if not np.isfinite(derivative).all():
                raise ValueError("nonfinite candidate derivative")
            for h in range(len(dev.future_inputs[0])):
                np.testing.assert_array_equal(derivative[h, :, h + 1 :], 0)
        row = dict(
            name=name,
            family=case["family"],
            data_seed=case["data_seed"],
            extra=case["extra"],
            status="complete",
            correctness_passed=True,
            incumbent_fingerprint=incumbent.fingerprint(),
            candidate_fingerprint=candidate.fingerprint(),
            source=str(source),
            optimization=optimization,
            fit_wall_seconds=wall,
            replay_max_difference=replay_max,
            regimes=regimes,
        )
        write(directory / "result.json", row)
        rows.append(row)
        write(args.output / "results.json", rows)
        print(
            json.dumps(
                dict(
                    completed=name,
                    termination=optimization["termination"]["reason"],
                    calls=optimization["resource_usage"][
                        "training_value_gradient_calls"
                    ],
                    matched=regimes["matched"]["candidate"]["overall_scaled_rmse"],
                    shifted=regimes["shifted"]["candidate"]["overall_scaled_rmse"],
                )
            ),
            flush=True,
        )
    decision = decide(manifest, rows)
    write(args.output / "decision.json", decision)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/generic-fit-acceptance.json",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output", type=Path, help="Run the frozen candidate fits into a new directory"
    )
    mode.add_argument(
        "--verify", type=Path, help="Replay a preserved run without fitting"
    )
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_run(args.verify, args.manifest), allow_nan=False))
    else:
        main(args)
