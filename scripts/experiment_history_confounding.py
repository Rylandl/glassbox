"""Challenge extra-history evidence with a fully observed nonlinear Markov cycle.

The additive polynomial regressions here are research comparisons. Neither the
learner nor the public diagnostic recipe is modified or selected by this script.
"""

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
from experiment_model_qualification import collection, encoding, recordings
from experiment_sequence_diagnostics import fresh_recordings

from glassbox.experimental.default_model import _RECIPE, LearnedDynamics, fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_diagnostics import _DIAGNOSTIC_RECIPE, _fit, _run


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def observed_cycle(seed, *, evaluation=False):
    segments = []
    for record in range(4 if evaluation else 8):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(evaluation), record, 941])
        )
        for episode in range(8):
            initial = rng.uniform(-0.9, 0.9, 5)
            x = np.stack([np.roll(initial, -k) for k in range(13)])
            x[:, 0] **= 3
            segments.append(
                SequenceSegment(
                    f"{'evaluation' if evaluation else 'fit'}-{record}",
                    f"episode-{episode}",
                    x,
                    np.zeros((12, 1)),
                    0.05,
                    episode * 20,
                )
            )
    return SequenceCollection(
        tuple(segments),
        configuration_id="synthetic-observed-cycle",
        state_channels=tuple(f"x{i} [unitless]" for i in range(5)),
        input_channels=("u [requested,unitless]",),
    )


def cycle_step(x):
    """Markov transition uses the present observed state only."""
    x = np.asarray(x)
    return np.concatenate((x[..., 1:2] ** 3, x[..., 2:], np.cbrt(x[..., :1])), axis=-1)


class HoldWitness:
    """Deliberately misspecified mean with the same evidence-exclusion contract."""

    def __init__(self, model):
        self.contract = model.contract
        self._seen = model._seen.copy()
        self.history_steps = model.history_steps
        self._fingerprint = hashlib.sha256(
            ("hold-observation-witness-v1:" + model.fingerprint()).encode()
        ).hexdigest()

    def fingerprint(self):
        return self._fingerprint

    def predict(self, past_states, past_inputs, future_inputs):
        x, u = np.asarray(past_states), np.asarray(future_inputs)
        return np.broadcast_to(
            x[..., -1:, :], x.shape[:-2] + (u.shape[-2], x.shape[-1])
        ).copy()


def powers(x, degree):
    return np.concatenate([x**k for k in range(1, degree + 1)], axis=1)


def measures(residual, predictions, indices):
    errors = {
        name: np.mean((residual[indices] - value[indices]) ** 2, axis=0)
        for name, value in predictions.items()
    }
    return dict(
        rms={k: np.sqrt(v).tolist() for k, v in errors.items()},
        extra_history_mse_reduction={
            name: [
                float(1 - a / b) if b > 0 else None
                for a, b in zip(errors[name], errors["short"], strict=True)
            ]
            for name in ("extended", "replacement")
        },
    )


def compare_bases(base):
    short = np.column_stack((base["short_features"], base["current_inputs"]))
    older, residual, ids = (
        base[k] for k in ("older_features", "model_residual", "record_index")
    )
    saved, reports = {}, {}
    for degree in (1, 2, 3):
        predictions = {
            name: np.empty_like(residual)
            for name in ("short", "extended", "replacement")
        }
        for held in range(len(base["record_names"])):
            old_prefix = f"fold_{held}_"
            prefix = f"degree_{degree}_{old_prefix}"
            train, test, donor_train, donor_test = (
                base[old_prefix + key]
                for key in ("train", "test", "donor_train", "donor_test")
            )
            mean, scale = short[train].mean(0), short[train].std(0)
            scale = np.where(scale > 0, scale, 1.0)
            normalized = (short - mean) / scale
            polynomial = powers(normalized, degree)
            saved[prefix + "short_mean"] = mean
            saved[prefix + "short_scale"] = scale
            for name, x, query in (
                ("short", polynomial[train], polynomial[test]),
                (
                    "extended",
                    np.column_stack((polynomial[train], older[train])),
                    np.column_stack((polynomial[test], older[test])),
                ),
                (
                    "replacement",
                    np.column_stack((polynomial[train], older[donor_train])),
                    np.column_stack((polynomial[test], older[donor_test])),
                ),
            ):
                regression = _fit(x, residual[train])
                predictions[name][test] = regression.predict(query)
                saved.update(
                    {
                        prefix + name + "_" + k: value
                        for k, value in regression.arrays().items()
                    }
                )
        if degree == 1:
            for name, value in predictions.items():
                np.testing.assert_allclose(
                    value, base["residual_" + name], rtol=1e-7, atol=1e-9
                )
        saved.update({f"degree_{degree}_{k}": v for k, v in predictions.items()})
        reports[str(degree)] = dict(
            aggregate=measures(residual, predictions, np.arange(len(ids))),
            recordings={
                str(name): measures(residual, predictions, np.flatnonzero(ids == i))
                for i, name in enumerate(base["record_names"])
            },
        )
    return reports, saved


def source_model(case, seed, output, prior):
    if case.startswith("observed_cycle"):
        path = output / f"observed_cycle-learned-{seed}" / "model.npz"
        if case.endswith("learned"):
            model = fit(observed_cycle(seed))
            model.save(path)
            write(path.with_name("fit-report.json"), model.report)
        else:
            model = LearnedDynamics.load(path)
        supplied = observed_cycle(seed, evaluation=True)
    elif case == "hidden_delay-learned":
        path = prior / f"memory-hidden-{seed}/result.json"
        path = Path(json.loads(path.read_text())["model_source"])
        model = LearnedDynamics.load(path)
        supplied = collection(
            recordings("memory", seed, evaluation=True),
            encoding("identity", 1),
            "memory",
        )
    else:
        assert case == "process_noise-learned"
        path = prior / f"process_noise-{seed}/model.npz"
        model = LearnedDynamics.load(path)
        supplied = collection(
            fresh_recordings("process_noise", seed, evaluation=True),
            encoding("identity", 1),
            "process_noise",
        )
    return model, supplied, path


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert (
        plan["learner_recipe"] == _RECIPE
        and plan["diagnostic_recipe"] == _DIAGNOSTIC_RECIPE
    )
    assert plan["seeds"] == [101, 202, 303]
    assert plan["cases"] == [
        "observed_cycle-learned",
        "observed_cycle-hold",
        "hidden_delay-learned",
        "process_noise-learned",
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            jax=jax.__version__,
            x64=True,
        ),
    )
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "src/glassbox").rglob("*.py")),
        Path(__file__),
        root / "scripts/experiment_model_qualification.py",
        root / "scripts/experiment_sequence_diagnostics.py",
        root / "tests/test_history_confounding.py",
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
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            name = f"{case}-{seed}"
            directory = args.output / name
            directory.mkdir()
            print(json.dumps(dict(starting=name)), flush=True)
            fitted, supplied, path = source_model(case, seed, args.output, args.prior)
            before = fitted.fingerprint()
            model = HoldWitness(fitted) if case.endswith("hold") else fitted
            report, base = _run(model, supplied)
            comparisons, extra = compare_bases(base)
            assert fitted.fingerprint() == before
            write(directory / "diagnostic.json", report)
            write(directory / "comparison.json", comparisons)
            np.savez_compressed(directory / "evidence.npz", **base, **extra)
            result = dict(
                case=case,
                seed=seed,
                model_source=str(path.resolve()),
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                fitted_model_fingerprint=before,
                predictor_fingerprint=model.fingerprint(),
                windows=report["windows"],
                model_rms=report["aggregate"]["forecast_error"]["model_rms"],
                degrees={k: v["aggregate"] for k, v in comparisons.items()},
            )
            write(directory / "result.json", result)
            results.append(result)
            write(args.output / "results.json", results)
            print(json.dumps(result), flush=True)
    assert len(results) == 12


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("../artifacts/sequence-diagnostics/comparison-01"),
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
