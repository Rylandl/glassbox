"""Measure batch-fit parameter information and forecast errors on noisy plants."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from glassbox.core.data import save_trajectory_npz, trajectory_windows
from glassbox.core.dynamics import (
    hover_control,
    rollout_with_latent,
    structured_parameter_names,
    structured_parameter_vector,
)
from glassbox.core.geometry import (
    TANGENT_GROUP_INDICES,
    rigid_body_local_error,
    state_plus_tangent,
)
from glassbox.core.synthetic import generate_trajectory, true_parameters
from glassbox.fitting import FitSpec, Holdout, fit

CONDITIONS = {"clean": 0.0, "noisy": 1.0, "noisier": 3.0, "collective": 1.0}
HORIZONS = (0.1, 0.5, 1.0)
NOISE_STD = np.repeat((0.002, 0.02, 0.002, 0.01), 3)
DT = 0.02


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def noisy_trajectory(clean, seed, scale, role):
    noise = np.random.default_rng(seed).normal(size=(len(clean.states), 12))
    states = jax.vmap(state_plus_tangent)(
        jnp.asarray(clean.states), jnp.asarray(scale * NOISE_STD * noise)
    )
    return replace(
        clean,
        states=np.asarray(states),
        spec=replace(clean.spec, observation_source="estimated"),
        labels={"source_group": f"{role}-{seed}", "role": role},
        provenance={**clean.provenance, "measurement_noise_seed": seed},
    )


def truth_flight(seed, collective):
    trajectory = generate_trajectory(seed=seed, duration_s=6.0)
    hover = np.asarray(hover_control(true_parameters()))
    if collective:
        controls = np.repeat(trajectory.controls.mean(axis=1)[:, None], 4, axis=1)
        states, _ = rollout_with_latent(
            true_parameters(), trajectory.states[0], controls, DT, hover
        )
        trajectory = replace(trajectory, controls=controls, states=np.asarray(states))
    return replace(trajectory, control_prefix=np.repeat(hover[None], 50, axis=0))


def test_arrays(belief, measured, truth):
    # Common one-second window starts make lead-time comparisons paired.
    windows = trajectory_windows(measured, horizon=50, stride=50)
    actual = trajectory_windows(truth, horizon=50, stride=50).target_states
    indices = jnp.asarray([round(h / DT) for h in HORIZONS])

    def predict(initial, controls, history, observed, true_states):
        prediction = belief.rollout(initial, controls, command_history=history)
        observed_error = jax.vmap(rigid_body_local_error)(prediction.states, observed)
        true_error = jax.vmap(rigid_body_local_error)(prediction.states, true_states)
        true_utilization = jax.vmap(belief.model.validity_utilization)(true_states)
        utilization = jnp.maximum(prediction.validity_utilization, true_utilization)
        return (
            observed_error[indices],
            true_error[indices],
            prediction.parameter_covariance[indices],
            prediction.forecast_error_covariance[indices],
            jnp.stack([jnp.max(utilization[: round(h / DT) + 1]) for h in HORIZONS]),
        )

    return tuple(
        np.asarray(value)
        for value in jax.jit(jax.vmap(predict))(
            windows.initial_states,
            windows.controls,
            windows.control_histories,
            windows.target_states,
            actual,
        )
    )


def parameter_summary(vectors, covariances, bases, eigenvalues, truth):
    errors = vectors - truth
    centered = vectors - vectors.mean(axis=0)
    factor = len(vectors) / max(len(vectors) - 1, 1)
    total, variation = [], []
    for error, deviation, basis, values in zip(errors, centered, bases, eigenvalues):
        if len(values):
            total.append(float(np.mean((basis.T @ error) ** 2 * values)))
            variation.append(
                float(factor * np.mean((basis.T @ deviation) ** 2 * values))
            )
    return {
        "mean_error": errors.mean(axis=0).tolist(),
        "centered_sample_variance": (
            np.var(vectors, axis=0, ddof=1).tolist() if len(vectors) > 1 else None
        ),
        "mean_resolved_covariance_diagonal": np.diagonal(
            covariances.mean(axis=0)
        ).tolist(),
        "resolved_error_over_variance": total,
        "resolved_centered_variation_over_variance": variation,
    }


def summarize(output, conditions):
    truth = np.asarray(structured_parameter_vector(true_parameters()))
    report = {
        "design": json.loads((output / "design.json").read_text()),
        "conditions": {},
    }
    for condition in conditions:
        files = sorted((output / condition).glob("*/arrays.npz"))
        data = [dict(np.load(path, allow_pickle=False)) for path in files]
        rows = [json.loads(path.with_name("fit.json").read_text()) for path in files]
        vectors = np.asarray([item["theta"] for item in data])
        errors = np.asarray([item["observed_error"] for item in data])
        true_errors = np.asarray([item["true_error"] for item in data])
        p = np.asarray([item["parameter"] for item in data])
        e = np.asarray([item["empirical"] for item in data])
        inside = np.asarray([item["utilization"] for item in data]) <= 1
        counts = inside.sum(axis=1)

        def inside_mean(values, inside=inside, counts=counts):
            per_fit = (values * inside).sum(axis=1) / np.maximum(counts, 1)
            return per_fit.sum(axis=0) / np.maximum((counts > 0).sum(axis=0), 1)

        groups = {}
        for name, indices in TANGENT_GROUP_INDICES.items():
            observed_mse = np.mean(errors[..., indices] ** 2, axis=(1, 3))
            true_mse = np.mean(true_errors[..., indices] ** 2, axis=(1, 3))
            empirical = np.mean(
                np.diagonal(e, axis1=-2, axis2=-1)[..., indices], axis=(1, 3)
            )
            parameter = np.mean(
                np.diagonal(p, axis1=-2, axis2=-1)[..., indices], axis=(1, 3)
            )
            groups[name] = {
                "observed_mse": observed_mse.mean(axis=0).tolist(),
                "true_mse": true_mse.mean(axis=0).tolist(),
                "empirical_over_observed_mse": (
                    empirical.mean(axis=0) / observed_mse.mean(axis=0)
                ).tolist(),
                "combined_over_observed_mse": (
                    (empirical + parameter).mean(axis=0) / observed_mse.mean(axis=0)
                ).tolist(),
                "per_fit_empirical_over_observed_mse": (
                    empirical / observed_mse
                ).tolist(),
            }
            if np.all(counts.sum(axis=0) > 0):
                actual_inside = inside_mean(np.mean(errors[..., indices] ** 2, axis=-1))
                e_inside = inside_mean(
                    np.mean(np.diagonal(e, axis1=-2, axis2=-1)[..., indices], axis=-1)
                )
                p_inside = inside_mean(
                    np.mean(np.diagonal(p, axis1=-2, axis2=-1)[..., indices], axis=-1)
                )
                groups[name]["inside_support_empirical_over_observed_mse"] = (
                    e_inside / actual_inside
                ).tolist()
                groups[name]["inside_support_combined_over_observed_mse"] = (
                    (e_inside + p_inside) / actual_inside
                ).tolist()
        report["conditions"][condition] = {
            "fits": len(data),
            "ranks": [row["rank"] for row in rows],
            "estimable_count": rows[0]["estimable_count"],
            "complete_count": sum(row["complete"] for row in rows),
            "inside_support_fraction": np.mean(inside, axis=(0, 1)).tolist(),
            "inside_support_fit_count": (counts > 0).sum(axis=0).tolist(),
            "parameters": parameter_summary(
                vectors,
                np.asarray([item["covariance"] for item in data]),
                [item["basis"] for item in data],
                [item["eigenvalues"] for item in data],
                truth,
            ),
            "forecast_groups": groups,
        }
    write_json(output / "report.json", report)
    return report


def run(output, replicates, conditions):
    if replicates < 1:
        raise ValueError("replicates must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("study output directory must be empty")
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), *sorted((root / "src/glassbox").rglob("*.py"))]
    design = {
        "replicates": replicates,
        "conditions": {key: CONDITIONS[key] for key in conditions},
        "measurement_noise_std": NOISE_STD.tolist(),
        "noise": "independent Gaussian tangent perturbations of each observed state",
        "plant": "core.synthetic.true_parameters; no process noise",
        "training": "two independent six-second flights per fit",
        "calibration": "two separate six-second flights per fit",
        "test": "three shared six-second command designs; independent observation noise per fit",
        "independent_unit": "fit replicate; test windows and horizons remain paired",
        "collective_condition": "collective-only training, calibration and test commands",
        "sample_period_s": DT,
        "fit_steps": 400,
        "training_horizon_steps": 25,
        "forecast_horizons_s": list(HORIZONS),
        "parameter_names": list(structured_parameter_names(true_parameters())),
        "true_parameter_vector": np.asarray(
            structured_parameter_vector(true_parameters())
        ).tolist(),
        "jax_version": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    write_json(output / "design.json", design)
    for condition in conditions:
        collective = condition == "collective"
        tests = [truth_flight(300000 + i, collective) for i in range(3)]
        for replicate in range(replicates):
            destination = output / condition / f"{replicate:02d}"
            destination.mkdir(parents=True, exist_ok=True)
            start = perf_counter()
            paths = []
            for i in range(4):
                role = "train" if i < 2 else "calibration"
                # Conditions share random numbers for paired noise/excitation contrasts.
                seed = 100000 + replicate * 10 + i
                clean = truth_flight(seed, collective)
                measured = noisy_trajectory(
                    clean, seed + 100000, CONDITIONS[condition], role
                )
                path = destination / f"{role}-{i}.npz"
                save_trajectory_npz(measured, path)
                paths.append(path)
            outcome = fit(
                paths,
                FitSpec(
                    holdout=Holdout.by_label("role", ("calibration",)),
                    evaluation_horizons_s=HORIZONS,
                ),
            )
            belief = outcome.belief
            measured_tests = [
                noisy_trajectory(
                    t, 400000 + replicate * 10 + i, CONDITIONS[condition], "test"
                )
                for i, t in enumerate(tests)
            ]
            observed_error, true_error, p, e, utilization = test_arrays(
                belief, measured_tests, tests
            )
            info = belief.information
            indices, values, eigenvectors, threshold = info._normalized_spectrum()
            retained = values > threshold
            basis = np.zeros((len(info.names), int(retained.sum())))
            basis[indices] = eigenvectors[:, retained] / info.scale[indices, None]
            np.savez_compressed(
                destination / "arrays.npz",
                theta=np.asarray(structured_parameter_vector(belief.params)),
                covariance=info.covariance(),
                basis=basis,
                eigenvalues=values[retained],
                observed_error=observed_error,
                true_error=true_error,
                parameter=p,
                empirical=e,
                utilization=utilization,
            )
            belief.save(destination / "belief.json")
            write_json(
                destination / "fit.json",
                {
                    "replicate": replicate,
                    "rank": info.resolved_rank(),
                    "estimable_count": info.estimable_count,
                    "complete": info.complete,
                    "effective_count": info.effective_count,
                    "fit": outcome.report["models"]["learned_lag"]["fit"],
                    "elapsed_s": perf_counter() - start,
                },
            )
            print(
                f"{condition} {replicate + 1}/{replicates}: rank {info.resolved_rank()}/{info.estimable_count}, {perf_counter() - start:.1f}s",
                flush=True,
            )
            jax.clear_caches()
        summarize(output, [key for key in conditions if (output / key).exists()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    args = parser.parse_args()
    run(args.output, args.replicates, args.conditions)
