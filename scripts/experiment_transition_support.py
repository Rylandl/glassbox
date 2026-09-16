"""Controlled missing-combination study of generic transition models.

Run with the workspace Python (NumPy, JAX, Matplotlib); no simulator is used.
The plan and source archive are written before fitting. Output must be new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
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

SYSTEMS = ("smooth", "coupled", "localized")
LAYOUTS = {"full": 0.0, "gap-035": 0.35, "gap-070": 0.70, "diagonal": None}
KERNELS = ("rbf", "matern52")
NOISE_STD = 0.025
BUMP_RADIUS = 0.28


def truth(features, system):
    """Benchmark truth only; never passed to the model fitter."""
    state, command = features[:, 0], features[:, 1]
    smooth = 0.65 * state + 0.3 * np.sin(2 * command) + 0.2 * state * command
    if system == "smooth":
        return smooth[:, None]
    if system == "coupled":
        return (
            0.55 * state
            + 0.35 * np.tanh(2 * command + 0.8 * state)
            + 0.22 * np.sin(3 * state * command)
        )[:, None]
    if system != "localized":
        raise ValueError(system)
    radius_sq = np.sum(features**2, axis=1) / BUMP_RADIUS**2
    # C-infinity compact bump: exactly zero outside the hidden disk.
    inside = radius_sq < 1
    bump = np.zeros(len(features))
    bump[inside] = 0.65 * np.exp(1 - 1 / (1 - radius_sq[inside]))
    return (smooth + bump)[:, None]


def training_inputs(seed, layout, rows):
    rng = np.random.default_rng(1000 + seed)
    pool = rng.uniform(-1, 1, (10000, 2))
    if layout == "diagonal":
        pool[:, 1] = pool[:, 0] + rng.normal(0, 0.015, len(pool))
        selected = np.flatnonzero(np.max(np.abs(pool), axis=1) <= 1)[:rows]
    else:
        selected = np.flatnonzero(np.linalg.norm(pool, axis=1) >= LAYOUTS[layout])[
            :rows
        ]
    if len(selected) != rows:
        raise ValueError("not enough candidate inputs")
    noise = np.random.default_rng(2000 + seed).normal(0, NOISE_STD, (len(pool), 1))
    return pool[selected], noise[selected], selected


def test_inputs(seed):
    rng = np.random.default_rng(3000 + seed)
    # Region labels below, rather than their sampling frequency, define metrics.
    square = rng.uniform(-1, 1, (1000, 2))
    angle = rng.uniform(0, 2 * np.pi, 300)
    radius = BUMP_RADIUS * np.sqrt(rng.uniform(0, 1, 300))
    center = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    candidates = rng.uniform(-1.4, 1.4, (4000, 2))
    outside = candidates[np.max(np.abs(candidates), axis=1) > 1][:400]
    along = rng.uniform(-0.95, 0.95, (200, 2))
    along[:, 1] = along[:, 0] + rng.normal(0, 0.01, len(along))
    points = np.concatenate((square, center, outside, along))
    noise = rng.normal(0, NOISE_STD, (len(points), 1))
    return points, noise


def regions(query, layout):
    radius = np.linalg.norm(query, axis=1)
    square = np.max(np.abs(query), axis=1) <= 1
    hole = LAYOUTS[layout] or 0
    return {
        "center": radius < BUMP_RADIUS,
        "gap": (radius < hole) & square,
        "reference_region": (
            square & (np.abs(query[:, 0] - query[:, 1]) <= 0.04)
            if layout == "diagonal"
            else square & (radius >= max(hole, BUMP_RADIUS))
        ),
        "outside": ~square,
        "along_diagonal": square & (np.abs(query[:, 0] - query[:, 1]) <= 0.04),
        "off_diagonal": square & (np.abs(query[:, 0] - query[:, 1]) > 0.15),
    }


def design(features, degree):
    x, u = features.T
    columns = [np.ones(len(x)), x, u]
    if degree == 2:
        columns += [x**2, x * u, u**2]
    return np.stack(columns, axis=1)


def metrics(mean, function_variance, observation_variance, latent, observed, masks):
    result = {}
    for name, mask in masks.items():
        if not np.any(mask):
            continue
        error = mean[mask] - latent[mask]
        noisy_error = mean[mask] - observed[mask]
        item = {"count": int(mask.sum()), "rmse": float(np.sqrt(np.mean(error**2)))}
        if function_variance is not None:
            fvar = np.maximum(function_variance[mask], 1e-12)
            ovar = np.maximum(observation_variance[mask], 1e-12)
            item.update(
                {
                    "function_coverage_95": float(
                        np.mean(np.abs(error) <= 1.95996398454 * np.sqrt(fvar))
                    ),
                    "observation_coverage_95": float(
                        np.mean(np.abs(noisy_error) <= 1.95996398454 * np.sqrt(ovar))
                    ),
                    "mean_function_std": float(np.mean(np.sqrt(fvar))),
                    "mean_observation_std": float(np.mean(np.sqrt(ovar))),
                    "observation_nll": float(
                        np.mean(
                            0.5 * (np.log(2 * np.pi * ovar) + noisy_error**2 / ovar)
                        )
                    ),
                }
            )
        result[name] = item
    return result


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_snapshot(output):
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        root / "src/glassbox/experimental/transition_gp.py",
        root / "src/glassbox/experimental/__init__.py",
        root / "tests/test_transition_gp.py",
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
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "platform": platform.platform(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True
        ).strip(),
    }


def run_trial(output, system, layout, seed, rows, steps):
    features, noise, identities = training_inputs(seed, layout, rows)
    targets = truth(features, system) + noise
    query, test_noise = test_inputs(seed)
    latent = truth(query, system)
    observed = latent + test_noise
    masks = regions(query, layout)
    tag = f"{system}-{layout}-seed{seed}"
    directory = output / tag
    directory.mkdir()
    samples = TransitionSamples(features[:, :1], features[:, 1:], targets, 0.1)
    np.savez_compressed(
        directory / "samples.npz",
        train_features=features,
        train_targets=targets,
        train_noise=noise,
        train_id=identities,
        query=query,
        latent=latent,
        observed=observed,
        **{f"region_{name}": mask for name, mask in masks.items()},
    )
    # Exact row disjointness is necessary here; no temporal-neighbor issue exists
    # because this benchmark samples independent resets, not flight windows.
    assert not np.any(np.all(features[:, None] == query[None], axis=-1))
    report = {
        "system": system,
        "layout": layout,
        "seed": seed,
        "training_input_hash": digest(features),
        "training_target_hash": digest(targets),
        "test_input_hash": digest(query),
        "test_observed_hash": digest(observed),
        "coordinate_bounds": [
            features.min(axis=0).tolist(),
            features.max(axis=0).tolist(),
        ],
        "learners": {},
    }
    for degree, name in ((1, "affine"), (2, "quadratic")):
        coefficients = np.linalg.lstsq(design(features, degree), targets, rcond=None)[0]
        mean = design(query, degree) @ coefficients
        report["learners"][name] = {
            "metrics": metrics(mean, None, None, latent, observed, masks)
        }
        np.savez_compressed(
            directory / f"{name}-predictions.npz", mean=mean, coefficients=coefficients
        )
    for kernel in KERNELS:
        start = time.monotonic()
        model = fit_transition_gp(samples, kernel=kernel, steps=steps, restarts=2)
        model.save(directory / f"{kernel}-model.npz")
        prediction = jax.jit(model.predict)(
            jnp.asarray(query[:, :1]), jnp.asarray(query[:, 1:])
        )
        mean, fvar, ovar = map(np.asarray, prediction)
        neighborhood = model.support(query[:, :1], query[:, 1:])
        np.savez_compressed(
            directory / f"{kernel}-predictions.npz",
            mean=mean,
            function_variance=fvar,
            observation_variance=ovar,
            nearest_distance=neighborhood.nearest_distance,
            eigenvalues=neighborhood.eigenvalues,
            basis=neighborhood.basis,
            query_offset_in_basis=neighborhood.query_offset_in_basis,
        )
        entry = {
            "fit": model.fit_report,
            "metrics": metrics(mean, fvar, ovar, latent, observed, masks),
            "seconds": time.monotonic() - start,
            "support": {
                name: {
                    "median_nearest_distance": float(
                        np.median(neighborhood.nearest_distance[mask])
                    ),
                    "median_thinness": float(
                        np.median(
                            neighborhood.eigenvalues[mask, 0]
                            / np.maximum(neighborhood.eigenvalues[mask, -1], 1e-12)
                        )
                    ),
                }
                for name, mask in masks.items()
                if np.any(mask)
            },
        }
        if seed == 0 and system == "smooth" and layout in ("diagonal", "gap-070"):
            response = model.local_response(jnp.array([0.1]), jnp.array([-0.1]))
            entry["local_response_at_0.1_minus0.1"] = {
                name: np.asarray(value).tolist()
                for name, value in zip(response._fields, response)
            }
        # Loading a fitted model must reproduce all saved predictive quantities.
        loaded = GaussianTransition.load(directory / f"{kernel}-model.npz")
        restored = loaded.predict(
            jnp.asarray(query[:8, :1]), jnp.asarray(query[:8, 1:])
        )
        for before, after in zip(prediction, restored):
            np.testing.assert_allclose(
                np.asarray(before[:8]), np.asarray(after), atol=2e-10, rtol=2e-10
            )
        report["learners"][kernel] = entry
        print(
            f"{tag} {kernel}: center RMSE={entry['metrics']['center']['rmse']:.4f}, "
            f"95% observation coverage={entry['metrics']['center']['observation_coverage_95']:.3f}, "
            f"{entry['seconds']:.1f}s",
            flush=True,
        )
    write_json(directory / "report.json", report)
    return report


def aggregate(reports):
    result = []
    for system in SYSTEMS:
        for layout in LAYOUTS:
            selected = [
                report
                for report in reports
                if report["system"] == system and report["layout"] == layout
            ]
            if not selected:
                continue
            for learner in ("affine", "quadratic", *KERNELS):
                for region in selected[0]["learners"][learner]["metrics"]:
                    items = [
                        report["learners"][learner]["metrics"][region]
                        for report in selected
                    ]
                    values = {
                        key: [item[key] for item in items]
                        for key in items[0]
                        if key != "count"
                    }
                    result.append(
                        {
                            "system": system,
                            "layout": layout,
                            "learner": learner,
                            "region": region,
                            "seeds": [report["seed"] for report in selected],
                            "mean": {
                                key: float(np.mean(value))
                                for key, value in values.items()
                            },
                            "min": {
                                key: float(np.min(value))
                                for key, value in values.items()
                            },
                            "max": {
                                key: float(np.max(value))
                                for key, value in values.items()
                            },
                        }
                    )
    return result


def plots(output, summary, reports):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.8), layout="constrained")
    colors = {"rbf": "#187b9b", "matern52": "#a64637", "quadratic": "#777777"}
    for column, system in enumerate(SYSTEMS):
        for learner in ("rbf", "matern52", "quadratic"):
            items = [
                item
                for item in summary
                if item["system"] == system
                and item["learner"] == learner
                and item["region"] == "center"
                and item["layout"] != "diagonal"
            ]
            items.sort(key=lambda item: LAYOUTS[item["layout"]])
            radius = [LAYOUTS[item["layout"]] for item in items]
            for row, metric in enumerate(("rmse", "observation_coverage_95")):
                if row and learner == "quadratic":
                    continue
                mean = [item["mean"][metric] for item in items]
                axes[row, column].plot(
                    radius, mean, marker="o", label=learner, color=colors[learner]
                )
                axes[row, column].fill_between(
                    radius,
                    [item["min"][metric] for item in items],
                    [item["max"][metric] for item in items],
                    color=colors[learner],
                    alpha=0.12,
                )
        axes[0, column].set_title(system.capitalize())
        axes[1, column].axhline(0.95, color="#444444", linestyle="--", linewidth=1)
        axes[1, column].set_ylim(0, 1.03)
        for row in range(2):
            axes[row, column].set_xlabel("Removed disk radius (normalized inputs)")
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel("Next-state RMSE in fixed center region")
    axes[1, 0].set_ylabel("Coverage of nominal 95% observation interval")
    axes[0, 0].legend(frameon=False)
    rows = json.loads((output / "plan.json").read_text())["rows"]
    figure.suptitle(
        "General transition learning across missing combinations\n"
        f"Equal {rows}-sample budgets · lines: seed means · shading: seed range"
    )
    figure.savefig(output / "gap-results.png", dpi=180)
    figure.savefig(output / "gap-results.svg")
    plt.close(figure)

    if not all(
        (output / f"{system}-gap-070-seed0").exists()
        for system in ("smooth", "localized")
    ):
        return
    axis = np.linspace(-1, 1, 81)
    x, u = np.meshgrid(axis, axis)
    grid = np.stack((x.ravel(), u.ravel()), axis=1)
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for row, system in enumerate(("smooth", "localized")):
        directory = output / f"{system}-gap-070-seed0"
        model = GaussianTransition.load(directory / "rbf-model.npz")
        predicted = model.predict(jnp.asarray(grid[:, :1]), jnp.asarray(grid[:, 1:]))
        latent = truth(grid, system)
        train = np.load(directory / "samples.npz")["train_features"]
        fields = [
            latent,
            np.asarray(predicted.mean),
            abs(np.asarray(predicted.mean) - latent),
            np.sqrt(np.asarray(predicted.function_variance)),
        ]
        titles = [
            "True next state",
            "Learned next state",
            "Absolute error",
            "Conditional function std",
        ]
        for column, (field, title) in enumerate(zip(fields, titles)):
            panel = axes[row, column]
            max_value = (1.1, 1.1, 0.65, 0.08)[column]
            mesh = panel.pcolormesh(
                x,
                u,
                field.reshape(x.shape),
                shading="auto",
                cmap="RdBu_r" if column < 2 else "magma",
                vmin=-1.1 if column < 2 else 0,
                vmax=max_value,
            )
            panel.add_patch(
                Circle((0, 0), 0.70, fill=False, edgecolor="#333333", linestyle="--")
            )
            if column == 0:
                panel.scatter(train[:, 0], train[:, 1], s=3, color="#333333", alpha=0.5)
            panel.set_aspect("equal")
            panel.set_xlabel("State")
            panel.set_title(title)
            if column == 0:
                panel.set_ylabel(f"{system.capitalize()}\nCommand")
            figure.colorbar(mesh, ax=panel, shrink=0.75)
    figure.suptitle(
        "Identical observations outside the gap can hide different dynamics\n"
        "RBF, seed 0 · dashed circle: excluded training region · noise std 0.025"
    )
    figure.savefig(output / "hidden-dynamics.png", dpi=180)
    figure.savefig(output / "hidden-dynamics.svg")
    plt.close(figure)


def audit(output, reports):
    identical = []
    for report in reports:
        if report["system"] != "localized" or report["layout"] not in (
            "gap-035",
            "gap-070",
        ):
            continue
        match = next(
            item
            for item in reports
            if item["system"] == "smooth"
            and item["layout"] == report["layout"]
            and item["seed"] == report["seed"]
        )
        assert report["training_input_hash"] == match["training_input_hash"]
        assert report["training_target_hash"] == match["training_target_hash"]
        differences = {}
        for kernel in KERNELS:
            left = np.load(
                output
                / f"localized-{report['layout']}-seed{report['seed']}"
                / f"{kernel}-predictions.npz"
            )
            right = np.load(
                output
                / f"smooth-{report['layout']}-seed{report['seed']}"
                / f"{kernel}-predictions.npz"
            )
            differences[kernel] = float(np.max(np.abs(left["mean"] - right["mean"])))
            assert differences[kernel] == 0
        identical.append(
            {
                "layout": report["layout"],
                "seed": report["seed"],
                "prediction_max_difference": differences,
            }
        )
    write_json(
        output / "audit.json",
        {
            "passed": True,
            "checks": [
                "training and evaluation rows disjoint",
                "all model round trips reproduce predictions",
                "hidden-change training labels exactly equal to smooth labels outside bump",
                "identical inputs and labels produce identical fitted predictions",
            ],
            "identical_evidence_pairs": identical,
            "interval_claim": "conditional on selected kernel/hyperparameters; measured coverage only",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args()
    if not jax.config.jax_enable_x64:
        parser.error("run this numerical study with JAX_ENABLE_X64=1")
    args.output.mkdir(parents=True, exist_ok=False)
    plan = {
        "systems": SYSTEMS,
        "layouts": LAYOUTS,
        "kernels": KERNELS,
        "seeds": args.seeds,
        "rows": args.rows,
        "steps": args.steps,
        "restarts": 2,
        "target_noise_std": NOISE_STD,
        "dt_s": 0.1,
        "targets": "complete next state",
        "fit_inputs": ["current state", "command"],
        "hyperparameter_selection": "training marginal likelihood",
        "evaluation": "independent reset transitions; fixed center probe plus domain/diagonal regions",
        "aggregation": "equal seed mean, with min/max; no independence-based confidence intervals",
        "limitations": [
            "synthetic 1-state/1-command systems",
            "noise-free input measurements",
            "no memory or drift",
            "one-step intervals only",
            "no physical platform validation",
            "no hyperparameter uncertainty",
            "no coverage guarantee under misspecification",
        ],
    }
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "environment.json", source_snapshot(args.output))
    reports = []
    for seed in args.seeds:
        for layout in LAYOUTS:
            for system in SYSTEMS:
                reports.append(
                    run_trial(args.output, system, layout, seed, args.rows, args.steps)
                )
    summary = aggregate(reports)
    write_json(args.output / "summary.json", summary)
    audit(args.output, reports)
    plots(args.output, summary, reports)
    print(
        f"Completed {len(reports) * len(KERNELS)} GP fits; audit passed. Results: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
