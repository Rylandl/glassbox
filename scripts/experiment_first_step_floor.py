"""Test a fixed first-step loss-scale floor, with a frozen fresh-seed confirmation."""

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
from experiment_horizon_generalization import (
    FAMILIES,
    evaluate_arrays,
    generate,
    measure,
    screen,
    write,
)

from glassbox.experimental.default_model import _RECIPE, LearnedDynamics, fit
from glassbox.experimental.sequence_model import fit_sequence_model


def first_step_floor(scales):
    """No channel's later horizon gets more weight than its first step."""
    values = np.asarray(scales)
    return np.maximum(values, values[0])


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert tuple(plan["families"]) == FAMILIES and plan["recipe"] == _RECIPE
    assert plan["candidate_key"] == "first_step_floor"
    if plan["phase"] == "development":
        assert plan["data_seeds"] == [4101, 4202, 4303] and args.previous is not None
    else:
        assert (
            plan["phase"] == "confirmation"
            and plan["data_seeds"] == [5101, 5202, 5303]
            and args.previous is None
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
            print(json.dumps(dict(starting=name, phase=plan["phase"])), flush=True)
            if args.previous is None:
                calibration, _ = generate(plan, family, seed, "calibration")
                baseline = fit(calibration)
            else:
                path = args.previous / name / "baseline.npz"
                baseline = LearnedDynamics.load(path)
                prior = json.loads((path.parent / "result.json").read_text())
                assert baseline.fingerprint() == prior["baseline_fingerprint"]
            baseline.save(directory / "baseline.npz")
            write(directory / "baseline-report.json", baseline.report)
            before = baseline.fingerprint()
            scales = np.asarray(baseline.report["optimization"]["error_scale"])
            revised = first_step_floor(scales)
            identical = np.array_equal(scales, revised)
            b, dev = baseline._train.batch, baseline._development.batch
            if identical:
                candidate, fit_report = baseline._model, baseline.report["optimization"]
            else:
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
                    error_scale=revised,
                )
            candidate.save(directory / "first_step_floor.npz")
            write(directory / "first_step_floor-report.json", fit_report)
            for key in candidate.norms:
                np.testing.assert_array_equal(
                    candidate.norms[key], baseline._model.norms[key]
                )
            scale = baseline._model.norms["state_scale"]
            training_u = b.future_inputs.reshape(-1, b.future_inputs.shape[-1])
            regimes = {}
            for regime in ("matched", "shifted"):
                supplied, _ = generate(plan, family, seed, regime)
                data = evaluate_arrays(supplied, plan["evaluation_stride"])
                x, up, uf = (
                    data[k] for k in ("past_states", "past_inputs", "future_inputs")
                )
                predicted = {
                    label: np.asarray(model.rollout(x, up, uf))
                    for label, model in (
                        ("baseline", baseline._model),
                        ("first_step_floor", candidate),
                    )
                }
                assert all(np.isfinite(value).all() for value in predicted.values())
                predicted["hold"] = np.repeat(x[:, -1:], 5, axis=1)
                metrics = {
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
                    metrics=metrics,
                    recordings=per_recording,
                    future_input_in_training_marginal_range_fraction=np.mean(
                        inside, axis=(0, 1)
                    ).tolist(),
                    material_regression=dict(
                        first_step=screen(
                            metrics["baseline"]["horizon_scaled_rmse"][0],
                            metrics["first_step_floor"]["horizon_scaled_rmse"][0],
                            plan["screening"],
                        ),
                        overall=screen(
                            metrics["baseline"]["overall_scaled_rmse"],
                            metrics["first_step_floor"]["overall_scaled_rmse"],
                            plan["screening"],
                        ),
                    ),
                )
                np.savez_compressed(
                    directory / f"{regime}.npz", **data, **predicted, state_scale=scale
                )
            assert baseline.fingerprint() == before
            result = dict(
                family=family,
                seed=seed,
                status="complete",
                phase=plan["phase"],
                candidate_key="first_step_floor",
                baseline_fingerprint=before,
                candidate_fingerprint=candidate.fingerprint(),
                baseline_selected_step=baseline.report["optimization"]["selected_step"],
                candidate_selected_step=fit_report["selected_step"],
                identical_objective_reused=identical,
                changed_scale_entries=int(np.sum(scales != revised)),
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
                        reused=identical,
                        ratios={
                            regime: dict(
                                first=r["metrics"]["first_step_floor"][
                                    "horizon_scaled_rmse"
                                ][0]
                                / r["metrics"]["baseline"]["horizon_scaled_rmse"][0],
                                overall=r["metrics"]["first_step_floor"][
                                    "overall_scaled_rmse"
                                ]
                                / r["metrics"]["baseline"]["overall_scaled_rmse"],
                                screen=r["material_regression"],
                            )
                            for regime, r in regimes.items()
                        },
                    )
                ),
                flush=True,
            )
    assert len(results) == 24


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    main(p.parse_args())
