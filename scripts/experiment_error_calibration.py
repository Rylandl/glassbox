"""Independent-reset tests of calibrated error and causally available history.

Truth generates observations and diagnostic regions only. No mechanism or
region labels enter either learner. Every observation's role is persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import jax
import numpy as np

from glassbox.experimental.error_calibration import (
    PredictionContract,
    ResidualSamples,
    fit_error_calibration,
)
from glassbox.experimental.transition_gp import TransitionSamples, fit_transition_gp

CASES = ("smooth", "localized", "memory-omitted", "memory-history", "memory-unrelated")
BUDGETS = {"train": 192, "scale": 512, "calibration": 512, "test": 4000}
NOISE_STD = 0.02


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def response(features, case):
    x, u, previous = features.T
    base = 0.6 * x + 0.3 * np.sin(2 * u)
    if case.startswith("memory"):
        return (base + (0.35 + 0.15 * u) * previous)[:, None]
    base += 0.2 * x * u
    if case == "localized":
        radius_sq = (x**2 + u**2) / 0.28**2
        inside = radius_sq < 1
        base[inside] += 0.65 * np.exp(1 - 1 / (1 - radius_sq[inside]))
    return base[:, None]


def observations(seed, role, count, domain="broad"):
    # Independent SeedSequence keys, never a conditionally consumed shared pool.
    role_key = ("train", "scale", "calibration", "test", "probe").index(role)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, role_key, int(domain == "outer")])
    )
    features = rng.uniform(-1, 1, (count, 3))
    if domain == "outer":
        features[:, 1] = rng.choice([-1, 1], count) * rng.uniform(0.55, 1, count)
    noise = rng.normal(0, NOISE_STD, (count, 1))
    unrelated = rng.uniform(-1, 1, count)
    ids = np.array([f"seed{seed}/{role}/{domain}/{i}" for i in range(count)])
    return features, noise, unrelated, ids


def inputs(raw, unrelated, case):
    if case == "memory-history":
        return raw.copy()
    if case == "memory-unrelated":
        return np.column_stack((raw[:, :2], unrelated))
    return raw[:, :2].copy()


def predict(model, features):
    return model.predict(features[:, :1], features[:, 1:2], context=features[:, 2:])


def metrics(mean, latent, observed, bounds, raw):
    masks = {
        "all": np.ones(len(raw), dtype=bool),
        "feature-interior": np.linalg.norm(raw[:, :2], axis=1) < 0.28,
        "outer": np.abs(raw[:, 1]) >= 0.55,
    }
    return {
        name: {
            "count": int(mask.sum()),
            "latent_rmse": float(np.sqrt(np.mean((mean[mask] - latent[mask]) ** 2))),
            "observation_rmse": float(
                np.sqrt(np.mean((mean[mask] - observed[mask]) ** 2))
            ),
            "joint_coverage": float(
                np.mean(
                    np.all(
                        (observed[mask] >= bounds[0][mask])
                        & (observed[mask] <= bounds[1][mask]),
                        axis=1,
                    )
                )
            ),
            "mean_width": float(np.mean(bounds[1][mask] - bounds[0][mask])),
        }
        for name, mask in masks.items()
    }


def run(output, seeds, phase, steps):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "src/glassbox/experimental/error_calibration.py",
        root / "src/glassbox/experimental/transition_gp.py",
        root / "tests/test_error_calibration.py",
    ]
    hashes = {}
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in sources:
            name = str(source.relative_to(root))
            archive.write(source, name)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "sources.json", hashes)
    write_json(
        output / "plan.json",
        {
            "phase": phase,
            "seeds": seeds,
            "cases": CASES,
            "budgets": BUDGETS,
            "kernel": "rq",
            "steps": steps,
            "restarts": 2,
            "neighbors": 24,
            "miscoverage": 0.05,
            "noise_std": NOISE_STD,
            "scope": "synthetic independent resets, Euclidean scalar transitions, dt .1",
            "data_roles": "mean fit; separate error-scale fit; separate score calibration; independent final test",
            "model_selection": "training marginal likelihood only; settings frozen before confirmation",
            "evidence_domains": "broad for every case; additional outer-only evidence for localized case",
            "mean_budget": 192,
            "additional_calibration_labels": 1024,
            "evidence_rows_per_role": [64, 128, 512],
            "budget_comparison": "nested prefixes of independent scale/calibration streams; same frozen mean",
            "interpretation": "95% marginal joint observation coverage under calibration/test exchangeability; not local or shifted coverage",
            "memory": "y=.6x+.3sin(2u)+(.35+.15u)*u_previous+noise; learner gets no equation",
            "memory_control": "same observations and labels, either omitted, causal history, or independent unrelated context",
            "memory_irreducible_latent_rmse_without_history": float(
                np.sqrt((0.35**2 + 0.15**2 / 3) / 3)
            ),
            "regions": "truth-based diagnostic masks withheld from fitting; center is r<.28",
            "aggregation": "equal seed mean and descriptive min/max; no population confidence interval",
        },
    )
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "numpy": np.__version__,
            "jax": jax.__version__,
            "x64": bool(jax.config.jax_enable_x64),
            "platform": platform.platform(),
        },
    )
    reports = []
    for seed in seeds:
        for case in CASES:
            start = time.monotonic()
            folder = output / f"{case}-seed{seed}"
            folder.mkdir()
            data = {}
            for role, count in BUDGETS.items():
                raw, noise, unrelated, ids = observations(seed, role, count)
                data[f"{role}_raw"] = raw
                data[f"{role}_features"] = inputs(raw, unrelated, case)
                data[f"{role}_latent"] = response(raw, case)
                data[f"{role}_observed"] = data[f"{role}_latent"] + noise
                data[f"{role}_ids"] = ids
            train = data["train_features"]
            model = fit_transition_gp(
                TransitionSamples(
                    train[:, :1],
                    train[:, 1:2],
                    data["train_observed"],
                    0.1,
                    train[:, 2:],
                ),
                kernel="rq",
                steps=steps,
                restarts=2,
            )
            model.save(folder / "model.npz")
            contract = PredictionContract(
                model.fingerprint(),
                ("state [a.u.]", "command [a.u.]")
                + (
                    ("previous command [a.u.]",)
                    if case == "memory-history"
                    else ("unrelated context [a.u.]",)
                    if case == "memory-unrelated"
                    else ()
                ),
                ("next state [a.u.]",),
                0.1,
            )
            test = data["test_features"]
            prediction = predict(model, test)
            mean, fvar, ovar = map(np.asarray, prediction)
            results = {
                "mean": mean,
                "function_variance": fvar,
                "observation_variance": ovar,
            }
            gaussian = (
                mean - 1.95996398454 * np.sqrt(ovar),
                mean + 1.95996398454 * np.sqrt(ovar),
            )
            results["gp_lower"], results["gp_upper"] = gaussian
            report = {
                "case": case,
                "seed": seed,
                "contract": {
                    "model_id": contract.model_id,
                    "input_names": contract.input_names,
                },
                "methods": {
                    "gp": metrics(
                        mean,
                        data["test_latent"],
                        data["test_observed"],
                        gaussian,
                        data["test_raw"],
                    )
                },
            }
            # Probe is a slice visualization only; no labels are fed to learners.
            probe = np.column_stack(
                (np.zeros(401), np.linspace(-1, 1, 401), np.zeros(401))
            )
            if case.startswith("memory"):
                probe = np.column_stack(
                    (np.full(401, 0.2), np.full(401, 0.3), np.linspace(-1, 1, 401))
                )
            probe_features = inputs(probe, np.zeros(len(probe)), case)
            probe_prediction = predict(model, probe_features)
            results.update(
                probe_raw=probe,
                probe_features=probe_features,
                probe_latent=response(probe, case),
                probe_mean=np.asarray(probe_prediction.mean),
                probe_gp_width=1.95996398454
                * np.sqrt(np.asarray(probe_prediction.observation_variance)),
            )
            domains = ("broad", "outer") if case == "localized" else ("broad",)
            for domain in domains:
                residual_data = []
                for role in ("scale", "calibration"):
                    if domain == "outer":
                        raw, noise, unrelated, ids = observations(
                            seed, role, BUDGETS[role], domain
                        )
                        prefix = f"{role}_{domain}"
                        data[f"{prefix}_raw"] = raw
                        data[f"{prefix}_features"] = inputs(raw, unrelated, case)
                        data[f"{prefix}_latent"] = response(raw, case)
                        data[f"{prefix}_observed"] = data[f"{prefix}_latent"] + noise
                        data[f"{prefix}_ids"] = ids
                    else:
                        prefix = role
                    features = data[f"{prefix}_features"]
                    data[f"{prefix}_mean"] = np.asarray(predict(model, features).mean)
                    residual_data.append(
                        ResidualSamples(
                            contract,
                            features,
                            data[f"{prefix}_observed"] - data[f"{prefix}_mean"],
                            tuple(data[f"{prefix}_ids"]),
                        )
                    )
                for size, mode in (
                    (size, mode)
                    for size in ((64, 128, 512) if domain == "broad" else (512,))
                    for mode in ("global", "local")
                ):
                    name = f"{domain}{size if size != 512 else ''}-{mode}"
                    subset = [
                        ResidualSamples(
                            item.contract,
                            item.features[:size],
                            item.residuals[:size],
                            item.sample_ids[:size],
                        )
                        for item in residual_data
                    ]
                    calibrated = fit_error_calibration(
                        *subset,
                        training_sample_ids=tuple(data["train_ids"]),
                        mode=mode,
                        neighbors=24,
                    )
                    calibrated.save(folder / f"{name}.npz")
                    bounds = calibrated.interval(test, mean, contract=contract)
                    results[f"{name}_lower"], results[f"{name}_upper"] = bounds
                    results[f"probe_{name}_width"] = (
                        calibrated.quantile
                        * calibrated.error_scale(probe_features, contract=contract)
                    )
                    report["methods"][name] = metrics(
                        mean,
                        data["test_latent"],
                        data["test_observed"],
                        bounds,
                        data["test_raw"],
                    )
            np.savez_compressed(folder / "samples.npz", **data)
            np.savez_compressed(folder / "predictions.npz", **results)
            report["elapsed_s"] = time.monotonic() - start
            write_json(folder / "report.json", report)
            reports.append(report)
            print(
                json.dumps(
                    {
                        "case": case,
                        "seed": seed,
                        "rmse": report["methods"]["gp"]["all"]["latent_rmse"],
                        "local_coverage": report["methods"]["broad-local"]["all"][
                            "joint_coverage"
                        ],
                        "elapsed_s": report["elapsed_s"],
                    }
                ),
                flush=True,
            )
    aggregates = {}
    for case in CASES:
        subset = [r for r in reports if r["case"] == case]
        aggregates[case] = {}
        for method in subset[0]["methods"]:
            aggregates[case][method] = {}
            for region in subset[0]["methods"][method]:
                aggregates[case][method][region] = {}
                for metric in subset[0]["methods"][method][region]:
                    values = [r["methods"][method][region][metric] for r in subset]
                    aggregates[case][method][region][metric] = {
                        "mean": float(np.mean(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
    write_json(output / "summary.json", {"models": len(reports), "cases": aggregates})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        parser.error("recorded experiments require JAX_ENABLE_X64=1")
    run(args.output, args.seeds, args.phase, args.steps)
