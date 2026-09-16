"""Frozen comparison of two loss normalizations on new generic synthetic systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from pathlib import Path

import jax
import numpy as np
from experiment_horizon_scaling import pooled_scale

from glassbox.experimental.default_model import _RECIPE, fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_model import fit_sequence_model

FAMILIES = (
    "stable_affine",
    "coupled_nonlinear",
    "deadzone_saturation",
    "hidden_hysteresis",
    "delayed_nonlinear",
    "near_periodic",
    "off_periodic",
    "noisy_observation",
)
REGIMES = ("calibration", "matched", "shifted")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dimensions(family):
    if family == "stable_affine":
        return 3, 2, 3
    if family in ("coupled_nonlinear", "near_periodic", "off_periodic"):
        return 2, 2, 2
    return (2, 1, 1) if family == "hidden_hysteresis" else (1, 1, 1)


def step(family, x, u, delayed, disturbance):
    if family == "stable_affine":
        a = np.array([[0.94, 0.06, 0], [0, 0.70, 0.05], [0.02, 0, -0.35]])
        b = np.array([[0.10, 0], [0.03, 0.18], [0, 0.15]])
        return a @ x + b @ u
    if family == "coupled_nonlinear":
        return np.array(
            [
                0.78 * x[0]
                + 0.16 * np.tanh(1.5 * u[0] + 0.7 * x[1])
                + 0.04 * x[0] * x[1],
                0.65 * x[1] + 0.20 * np.sin(1.3 * u[1]) + 0.06 * x[0] ** 2,
            ]
        )
    if family == "deadzone_saturation":
        active = np.sign(u) * np.maximum(np.abs(u) - 0.2, 0)
        return 0.88 * x + 0.22 * np.clip(active, -0.45, 0.45)
    if family == "hidden_hysteresis":
        h = np.clip(x[1] + 0.25 * u[0], -0.6, 0.6)
        return np.array([0.86 * x[0] + 0.18 * h + 0.08 * u[0], h])
    if family == "delayed_nonlinear":
        return 0.72 * x + 0.22 * np.tanh(2 * delayed)
    if family in ("near_periodic", "off_periodic"):
        angle = 2 * np.pi / (5.2 if family == "near_periodic" else 8.0)
        c, s = np.cos(angle), np.sin(angle)
        return 0.98 * np.array([[c, -s], [s, c]]) @ x + 0.04 * u
    assert family == "noisy_observation"
    return 0.92 * x + 0.14 * np.tanh(1.5 * u) + 0.03 * disturbance


def generate(plan, family, seed, regime):
    latent_width, input_width, observed_width = dimensions(family)
    segments, truth = [], {}
    n = plan["intervals"]
    count = (
        plan["calibration_recordings"]
        if regime == "calibration"
        else plan["evaluation_recordings_per_regime"]
    )
    settings = plan["regimes"][regime]
    for record in range(count):
        name = f"{regime}-{record}"
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    seed,
                    FAMILIES.index(family),
                    REGIMES.index(regime),
                    record,
                    plan["rng_salt"],
                ]
            )
        )
        latent = np.empty((n + 1, latent_width))
        latent[0] = rng.uniform(-0.4, 0.4, latent_width)
        innovations = rng.uniform(
            -settings["input_amplitude"], settings["input_amplitude"], (n, input_width)
        )
        disturbances = rng.normal(size=(n, latent_width))
        sensor_noise = rng.normal(size=(n + 1, observed_width))
        inputs = np.empty_like(innovations)
        previous = np.zeros(input_width)
        for t in range(n):
            inputs[t] = (
                settings["input_persistence"] * previous
                + settings["input_innovation_weight"] * innovations[t]
            )
            previous = inputs[t]
            delayed = inputs[t - 4] if t >= 4 else np.zeros(input_width)
            latent[t + 1] = step(family, latent[t], inputs[t], delayed, disturbances[t])
        states = latent[:, :observed_width].copy()
        if family in ("near_periodic", "off_periodic"):
            states += 0.35 * states**3
        if family == "noisy_observation":
            states += 0.04 * sensor_noise
        segments.append(SequenceSegment(name, "whole", states, inputs, plan["dt_s"]))
        truth[name] = latent
    supplied = SequenceCollection(
        tuple(segments),
        configuration_id=f"synthetic-horizon-suite-{family}",
        state_channels=tuple(
            f"x{i} [observed,unitless]" for i in range(observed_width)
        ),
        input_channels=tuple(f"u{i} [requested,unitless]" for i in range(input_width)),
    )
    return supplied, truth


def evaluate_arrays(supplied, stride):
    index = [
        (s, t) for s in supplied.segments for t in range(2, len(s.states) - 5, stride)
    ]
    return dict(
        past_states=np.stack([s.states[t - 2 : t + 1] for s, t in index]),
        past_inputs=np.stack([s.inputs[t - 2 : t] for s, t in index]),
        future_inputs=np.stack([s.inputs[t : t + 5] for s, t in index]),
        targets=np.stack([s.states[t + 1 : t + 6] for s, t in index]),
        recording_ids=np.array([s.recording_id for s, _ in index]),
        source_origins=np.array([s.start_row + t for s, t in index]),
    )


def measure(prediction, target, state_scale):
    physical = np.mean((prediction - target) ** 2, axis=0)
    scaled = physical / state_scale**2
    return dict(
        channel_rmse=np.sqrt(physical).tolist(),
        horizon_scaled_rmse=np.sqrt(np.mean(scaled, axis=1)).tolist(),
        overall_scaled_rmse=float(np.sqrt(np.mean(scaled))),
    )


def screen(base, candidate, protocol):
    return bool(
        candidate > base * (1 + protocol["relative_rmse_increase"])
        and candidate - base > protocol["absolute_scaled_rmse_increase"]
    )


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert plan["recipe"] == _RECIPE and tuple(plan["families"]) == FAMILIES
    assert plan["data_seeds"] == [4101, 4202, 4303]
    assert (plan["intervals"], plan["dt_s"], plan["evaluation_stride"]) == (
        160,
        0.05,
        5,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=True,
        ),
    )
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "src/glassbox").rglob("*.py")),
        *sorted((root / "scripts").glob("*.py")),
        root / "tests/test_horizon_generalization.py",
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sources:
            z.write(p, str(p.relative_to(root)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    results = []
    for family in FAMILIES:
        for seed in plan["data_seeds"]:
            name = f"{family}-{seed}"
            directory = args.output / name
            directory.mkdir()
            print(json.dumps(dict(starting=name)), flush=True)
            calibration, _ = generate(plan, family, seed, "calibration")
            try:
                baseline = fit(calibration)
                baseline.save(directory / "baseline.npz")
                write(directory / "baseline-report.json", baseline.report)
                original = baseline.fingerprint()
                b, dev = baseline._train.batch, baseline._development.batch
                scale = pooled_scale(b, baseline._model.norms["state_scale"])
                candidate, fit_report = fit_sequence_model(
                    b,
                    dev,
                    kind=_RECIPE["kind"],
                    objective="rollout",
                    width=_RECIPE["width"],
                    ridge=_RECIPE["ridge_fraction"]
                    * len(b.past_states)
                    * b.future_states.shape[1],
                    seed=_RECIPE["seed"],
                    steps=_RECIPE["steps"],
                    batch_size=_RECIPE["batch_size"],
                    learning_rate=_RECIPE["learning_rate"],
                    check_every=_RECIPE["check_every"],
                    error_scale=scale,
                )
                candidate.save(directory / "pooled.npz")
                write(directory / "pooled-report.json", fit_report)
                for key in candidate.norms:
                    np.testing.assert_array_equal(
                        candidate.norms[key], baseline._model.norms[key]
                    )
                assert baseline.fingerprint() == original
            except ValueError as exc:
                failure = dict(
                    family=family, seed=seed, status="failed", error=str(exc)
                )
                write(directory / "result.json", failure)
                results.append(failure)
                write(args.output / "results.json", results)
                print(json.dumps(failure), flush=True)
                continue
            scale = baseline._model.norms["state_scale"]
            training_u = b.future_inputs.reshape(-1, b.future_inputs.shape[-1])
            regimes = {}
            for regime in ("matched", "shifted"):
                supplied, _ = generate(plan, family, seed, regime)
                data = evaluate_arrays(supplied, plan["evaluation_stride"])
                x, up, uf = (
                    data[key] for key in ("past_states", "past_inputs", "future_inputs")
                )
                predicted = {
                    label: np.asarray(model.rollout(x, up, uf))
                    for label, model in (
                        ("baseline", baseline._model),
                        ("pooled", candidate),
                    )
                }
                assert all(np.isfinite(value).all() for value in predicted.values())
                predicted["hold"] = np.repeat(x[:, -1:], 5, axis=1)
                report = {
                    label: measure(value, data["targets"], scale)
                    for label, value in predicted.items()
                }
                per_recording = {
                    str(record): {
                        label: measure(
                            value[data["recording_ids"] == record],
                            data["targets"][data["recording_ids"] == record],
                            scale,
                        )
                        for label, value in predicted.items()
                    }
                    for record in sorted(set(data["recording_ids"]))
                }
                inside = (uf >= training_u.min(0)) & (uf <= training_u.max(0))
                regimes[regime] = dict(
                    windows=len(x),
                    metrics=report,
                    recordings=per_recording,
                    future_input_in_training_marginal_range_fraction=np.mean(
                        inside, axis=(0, 1)
                    ).tolist(),
                    material_regression=dict(
                        first_step=screen(
                            report["baseline"]["horizon_scaled_rmse"][0],
                            report["pooled"]["horizon_scaled_rmse"][0],
                            plan["screening"],
                        ),
                        overall=screen(
                            report["baseline"]["overall_scaled_rmse"],
                            report["pooled"]["overall_scaled_rmse"],
                            plan["screening"],
                        ),
                    ),
                )
                np.savez_compressed(
                    directory / f"{regime}.npz", **data, **predicted, state_scale=scale
                )
            result = dict(
                family=family,
                seed=seed,
                status="complete",
                baseline_fingerprint=original,
                pooled_fingerprint=candidate.fingerprint(),
                baseline_selected_step=baseline.report["optimization"]["selected_step"],
                pooled_selected_step=fit_report["selected_step"],
                training_keys=[
                    [k.recording_id, k.segment_id, k.origin]
                    for k in baseline._train.keys
                ],
                development_keys=[
                    [k.recording_id, k.segment_id, k.origin]
                    for k in baseline._development.keys
                ],
                regimes=regimes,
            )
            write(directory / "result.json", result)
            results.append(result)
            write(args.output / "results.json", results)
            print(
                json.dumps(
                    dict(
                        family=family,
                        seed=seed,
                        ratios={
                            r: {
                                "first": v["metrics"]["pooled"]["horizon_scaled_rmse"][
                                    0
                                ]
                                / v["metrics"]["baseline"]["horizon_scaled_rmse"][0],
                                "overall": v["metrics"]["pooled"]["overall_scaled_rmse"]
                                / v["metrics"]["baseline"]["overall_scaled_rmse"],
                                "screen": v["material_regression"],
                            }
                            for r, v in regimes.items()
                        },
                    )
                ),
                flush=True,
            )
    assert len(results) == 24


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
