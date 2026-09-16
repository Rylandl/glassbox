"""Learn nonlinear shapes without exposing their mechanism to the learner.

This remains synthetic: the equations generate observations, not model features.
Development seeds and confirmation seeds are recorded separately.
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
import jax.numpy as jnp
import numpy as np

from glassbox.experimental.transition_gp import (
    GaussianTransition,
    TransitionSamples,
    fit_transition_gp,
)

SYSTEMS = ("smooth", "dead-zone", "saturation", "localized")
NOISE_STD = 0.02


def truth(features, system):
    state, command = features[:, 0], features[:, 1]
    if system == "dead-zone":
        # State-dependent width tests a joint response surface, not just a curve.
        width = 0.25 + 0.08 * np.sin(2 * state)
        response = np.sign(command) * np.maximum(np.abs(command) - width, 0)
        return (0.6 * state + 0.8 * response)[:, None]
    if system == "saturation":
        return (0.6 * state + 0.7 * np.clip(command + 0.2 * state, -0.45, 0.45))[
            :, None
        ]
    smooth = 0.65 * state + 0.3 * np.sin(2 * command) + 0.2 * state * command
    if system == "smooth":
        return smooth[:, None]
    if system == "localized":
        r2 = np.sum(features**2, axis=1) / 0.28**2
        bump = np.zeros(len(features))
        selected = r2 < 1
        bump[selected] = 0.65 * np.exp(1 - 1 / (1 - r2[selected]))
        return (smooth + bump)[:, None]
    raise ValueError(system)


def input_pool(seed):
    rng = np.random.default_rng(71000 + seed)
    pool = rng.uniform(-1, 1, (6000, 2))
    noise = rng.normal(0, NOISE_STD, (len(pool), 1))
    return pool, noise


def select_rows(pool, layout):
    if layout == "broad":
        return np.arange(192)
    # The candidate pool is independent of the conditioned initial sample.
    # Taking the first unselected rows of the same pool would preferentially
    # choose the missing region: earlier outer rows were consumed by the base.
    split = len(pool) // 2
    base = np.flatnonzero(np.abs(pool[:split, 1]) >= 0.55)[:128]
    if layout == "outer":
        return base
    candidates = np.arange(split, len(pool))
    if layout == "random-fill":
        return np.concatenate((base, candidates[:64]))
    if layout != "coverage-fill":
        raise ValueError(layout)
    # Batch farthest-point acquisition uses input geometry only. No target,
    # dead-zone boundary, truth derivative, or test label is used to choose rows.
    scale = np.std(pool[base], axis=0)
    distances = np.min(
        np.sum(((pool[:, None] - pool[base][None]) / scale) ** 2, axis=-1), axis=1
    )
    distances[:split] = -np.inf
    selected = list(base)
    for _ in range(64):
        index = int(np.argmax(distances))
        selected.append(index)
        distances = np.minimum(
            distances, np.sum(((pool - pool[index]) / scale) ** 2, axis=1)
        )
        distances[selected] = -np.inf
    return np.asarray(selected)


def queries(seed, system):
    rng = np.random.default_rng(81000 + seed)
    points = rng.uniform(-1, 1, (1600, 2))
    state, command = points.T
    if system == "dead-zone":
        width = 0.25 + 0.08 * np.sin(2 * state)
        center = np.abs(command) < width - 0.04
        edge = np.abs(np.abs(command) - width) < 0.06
    elif system == "saturation":
        center = np.abs(command + 0.2 * state) > 0.53
        edge = np.abs(np.abs(command + 0.2 * state) - 0.45) < 0.06
    else:
        center = np.linalg.norm(points, axis=1) < 0.28
        edge = (np.linalg.norm(points, axis=1) >= 0.20) & (
            np.linalg.norm(points, axis=1) < 0.36
        )
    masks = {
        "all": np.ones(len(points), dtype=bool),
        "shape-interior": center,
        "shape-boundary": edge,
        "missing-command-region": np.abs(command) < 0.55,
    }
    latent = truth(points, system)
    observed = latent + rng.normal(0, NOISE_STD, latent.shape)
    return points, latent, observed, masks


def score(prediction, latent, observed, masks):
    mean, fvar, ovar = map(np.asarray, prediction)
    results = {}
    for name, mask in masks.items():
        difference = mean[mask] - latent[mask]
        observation_error = mean[mask] - observed[mask]
        results[name] = {
            "count": int(mask.sum()),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "function_coverage_95": float(
                np.mean(abs(difference) <= 1.95996398454 * np.sqrt(fvar[mask]))
            ),
            "observation_coverage_95": float(
                np.mean(abs(observation_error) <= 1.95996398454 * np.sqrt(ovar[mask]))
            ),
            "mean_function_std": float(np.mean(np.sqrt(fvar[mask]))),
        }
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output, seeds, kernels, layouts, steps, phase):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "src/glassbox/experimental/transition_gp.py",
        root / "tests/test_transition_gp.py",
    ]
    source_hashes = {}
    with zipfile.ZipFile(
        output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in sources:
            name = str(source.relative_to(root))
            archive.write(source, name)
            source_hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output / "sources.json", source_hashes)
    write_json(
        output / "plan.json",
        {
            "phase": phase,
            "seeds": seeds,
            "kernels": kernels,
            "layouts": layouts,
            "systems": SYSTEMS,
            "steps": steps,
            "restarts": 2,
            "noise_std": NOISE_STD,
            "training_budgets": {
                "outer": 128,
                "broad": 192,
                "random-fill": 192,
                "coverage-fill": 192,
            },
            "data_selection": "geometry only; new labels counted in the training budget",
            "acquisition_pool": "independent second half of pool; first half supplies initial rows",
            "hyperparameter_selection": "training marginal likelihood only",
            "scope": "synthetic independent resets, full next-state prediction, fixed dt .1",
            "test_regions": "predeclared using simulator truth for evaluation only",
            "aggregation": "equal seed mean; descriptive seed range, no population coverage claim",
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
        pool, noise = input_pool(seed)
        for layout in layouts:
            ids = select_rows(pool, layout)
            features = pool[ids]
            for system in SYSTEMS:
                folder = output / f"{system}-{layout}-seed{seed}"
                folder.mkdir()
                targets = truth(features, system) + noise[ids]
                query, latent, observed, masks = queries(seed, system)
                assert not np.any(np.all(features[:, None] == query[None], axis=-1))
                np.savez_compressed(
                    folder / "samples.npz",
                    train_features=features,
                    train_targets=targets,
                    train_ids=ids,
                    query=query,
                    latent=latent,
                    observed=observed,
                    **{f"region_{name}": mask for name, mask in masks.items()},
                )
                report = {
                    "seed": seed,
                    "system": system,
                    "layout": layout,
                    "models": {},
                }
                for kernel in kernels:
                    started = time.monotonic()
                    model = fit_transition_gp(
                        TransitionSamples(
                            features[:, :1], features[:, 1:], targets, 0.1
                        ),
                        kernel=kernel,
                        steps=steps,
                        restarts=2,
                    )
                    model.save(folder / f"{kernel}-model.npz")
                    prediction = jax.jit(model.predict)(
                        jnp.asarray(query[:, :1]), jnp.asarray(query[:, 1:])
                    )
                    metrics = score(prediction, latent, observed, masks)
                    line = np.linspace(-1, 1, 401)[:, None]
                    line_prediction = jax.jit(model.predict)(
                        jnp.zeros_like(jnp.asarray(line)), jnp.asarray(line)
                    )
                    line_truth = truth(
                        np.concatenate((np.zeros_like(line), line), axis=1), system
                    )
                    line_slope = np.gradient(
                        np.asarray(line_prediction.mean)[:, 0], line[:, 0]
                    )
                    if system in ("dead-zone", "saturation"):
                        flat = (
                            abs(line[:, 0]) < 0.18
                            if system == "dead-zone"
                            else (abs(line[:, 0]) > 0.55) & (abs(line[:, 0]) < 0.95)
                        )
                        metrics["slice-plateau"] = {
                            "count": int(flat.sum()),
                            "rmse": float(
                                np.sqrt(
                                    np.mean(
                                        (
                                            np.asarray(line_prediction.mean)[flat]
                                            - line_truth[flat]
                                        )
                                        ** 2
                                    )
                                )
                            ),
                            "mean_abs_command_slope": float(
                                np.mean(abs(line_slope[flat]))
                            ),
                        }
                    np.savez_compressed(
                        folder / f"{kernel}-predictions.npz",
                        mean=prediction.mean,
                        function_variance=prediction.function_variance,
                        observation_variance=prediction.observation_variance,
                        slice_command=line,
                        slice_mean=line_prediction.mean,
                        slice_variance=line_prediction.function_variance,
                        slice_truth=line_truth,
                        slice_command_slope=line_slope,
                    )
                    restored = GaussianTransition.load(folder / f"{kernel}-model.npz")
                    actual = restored.predict(
                        jnp.asarray(query[:5, :1]), jnp.asarray(query[:5, 1:])
                    )
                    for left, right in zip(prediction, actual):
                        np.testing.assert_allclose(
                            np.asarray(left[:5]),
                            np.asarray(right),
                            atol=1e-9,
                            rtol=1e-9,
                        )
                    report["models"][kernel] = {
                        "metrics": metrics,
                        "fit": model.fit_report,
                        "seconds": time.monotonic() - started,
                    }
                    print(
                        f"{phase} {system} {layout} seed{seed} {kernel}: "
                        f"all={metrics['all']['rmse']:.4f}, shape={metrics['shape-interior']['rmse']:.4f}, "
                        f"noise={model.fit_report['noise_std_output_units'][0]:.4f}",
                        flush=True,
                    )
                write_json(folder / "report.json", report)
                reports.append(report)
    summary = []
    for system in SYSTEMS:
        for layout in layouts:
            selected = [
                report
                for report in reports
                if report["system"] == system and report["layout"] == layout
            ]
            for kernel in kernels:
                for region in selected[0]["models"][kernel]["metrics"]:
                    values = [
                        report["models"][kernel]["metrics"][region]
                        for report in selected
                    ]
                    item = {
                        "system": system,
                        "layout": layout,
                        "kernel": kernel,
                        "region": region,
                        "seeds": seeds,
                    }
                    for operation in ("mean", "min", "max"):
                        item[operation] = {
                            key: float(
                                getattr(np, operation)([value[key] for value in values])
                            )
                            for key in values[0]
                            if key != "count"
                        }
                    summary.append(item)
    write_json(output / "summary.json", summary)
    print(f"Finished {len(reports) * len(kernels)} fits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--kernels", nargs="+", default=["rbf", "matern52"])
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=["outer", "broad", "random-fill", "coverage-fill"],
        default=["broad"],
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--phase", choices=["development", "confirmation"], default="development"
    )
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        parser.error("run with JAX_ENABLE_X64=1")
    run(args.output, args.seeds, args.kernels, args.layouts, args.steps, args.phase)


if __name__ == "__main__":
    main()
