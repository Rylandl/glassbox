"""Archive optimizer paths and cross-select checkpoints on generic synthetic data.

Research instrumentation only. The copied Adam loop must pass parity checks
against the library fitter; it deliberately adds no library configuration.
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
import jax.numpy as jnp
import numpy as np
from experiment_first_step_floor import first_step_floor
from experiment_horizon_generalization import evaluate_arrays, generate, measure, write

from glassbox.experimental.default_model import (
    _RECIPE,
    LearnedDynamics,
    _extract,
    _priority,
    _recording_content,
)
from glassbox.experimental.sequence_model import (
    SequenceModel,
    _rollout,
    initialize_sequence_model,
)

OBJECTIVES = ("original", "first_step_floor")
ARRAY_NAMES = ("past_states", "past_inputs", "future_inputs", "future_states")


def prepare(recordings):
    """Use the unchanged public recipe's recording split and bounded windows."""
    names = sorted(_recording_content(recordings), key=_priority)
    count = min(len(names) - 1, max(1, int(np.ceil(len(names) / 4))))
    return (
        _extract(recordings, names[count:], _RECIPE["training_windows"]),
        _extract(recordings, names[:count], _RECIPE["development_windows"]),
    )


def fit_checkpoints(train, development, scales, recipe, seed):
    """Same initialization/update arithmetic as fit_sequence_model, saving all steps."""
    kind = recipe["kind"]
    initial = initialize_sequence_model(
        train,
        kind=kind,
        seed=seed,
        width=recipe["width"],
        ridge=recipe["ridge_fraction"]
        * len(train.past_states)
        * train.future_states.shape[1],
    )
    params, norms = jax.tree.map(jnp.asarray, (initial.params, initial.norms))
    normalization = jnp.asarray(scales)
    training = tuple(jnp.asarray(getattr(train, name)) for name in ARRAY_NAMES)
    validation = tuple(jnp.asarray(getattr(development, name)) for name in ARRAY_NAMES)
    learning_rate = recipe["learning_rate"]

    def loss(par, data):
        x, up, uf, target = data
        prediction = _rollout(par, norms, kind, x, up, uf)
        return jnp.mean(((prediction - target) / normalization) ** 2)

    @jax.jit
    def evaluate(par):
        return loss(par, validation)

    @jax.jit
    def update(par, first, second, index, indices):
        data = tuple(value[indices] for value in training)
        value, grad = jax.value_and_grad(loss)(par, data)
        norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grad)))
        grad = jax.tree.map(lambda g: g * jnp.minimum(1.0, 5.0 / (norm + 1e-12)), grad)
        first = jax.tree.map(lambda m, g: 0.9 * m + 0.1 * g, first, grad)
        second = jax.tree.map(lambda v, g: 0.999 * v + 0.001 * g * g, second, grad)
        par = jax.tree.map(
            lambda w, m, v: (
                w
                - learning_rate
                * (m / (1 - 0.9**index))
                / (jnp.sqrt(v / (1 - 0.999**index)) + 1e-8)
            ),
            par,
            first,
            second,
        )
        return par, first, second, value

    models, trace = [], []

    def record(step, batch_loss=None):
        value = float(evaluate(params))
        if not np.isfinite(value) or (
            batch_loss is not None and not np.isfinite(batch_loss)
        ):
            raise ValueError(f"nonfinite checkpoint at step {step}")
        row = dict(step=step, validation_rollout_mse=value)
        if batch_loss is not None:
            row["training_batch_mse"] = float(batch_loss)
        trace.append(row)
        models.append(
            SequenceModel(
                kind,
                train.dt_s,
                initial.history_steps,
                jax.tree.map(lambda a: np.array(a, copy=True), params),
                jax.tree.map(np.asarray, norms),
            )
        )

    record(0)
    first, second = (jax.tree.map(jnp.zeros_like, params) for _ in range(2))
    rng = np.random.default_rng(seed + 10000)
    for step in range(1, recipe["steps"] + 1):
        indices = rng.integers(
            len(train.past_states),
            size=min(recipe["batch_size"], len(train.past_states)),
        )
        params, first, second, value = update(params, first, second, step, indices)
        if step % recipe["check_every"] == 0 or step == recipe["steps"]:
            record(step, float(value))
    return models, trace


@jax.jit
def predict(params, norms, x, up, uf):
    return _rollout(params, norms, "delay_mlp", x, up, uf)


def predictions(models, arrays):
    return np.stack(
        [
            np.asarray(
                predict(
                    model.params,
                    model.norms,
                    arrays["past_states"],
                    arrays["past_inputs"],
                    arrays["future_inputs"],
                )
            )
            for model in models
        ]
    )


def select(development_predictions, targets, scales, recording_ids):
    """Select solely from development data; earliest checkpoint wins exact ties."""
    loss = np.mean(((development_predictions - targets) / scales) ** 2, axis=(1, 2, 3))
    per_record = {
        str(name): np.mean(
            (
                (
                    development_predictions[:, recording_ids == name]
                    - targets[recording_ids == name]
                )
                / scales
            )
            ** 2,
            axis=(1, 2, 3),
        ).tolist()
        for name in sorted(set(recording_ids))
    }
    return dict(
        selected_index=int(np.argmin(loss)),
        losses=loss.tolist(),
        recording_losses=per_record,
        leave_one_recording_out_indices={
            str(name): int(
                np.argmin(
                    np.mean(
                        (
                            (
                                development_predictions[:, recording_ids != name]
                                - targets[recording_ids != name]
                            )
                            / scales
                        )
                        ** 2,
                        axis=(1, 2, 3),
                    )
                )
            )
            for name in sorted(set(recording_ids))
        },
    )


def main(args):
    assert jax.config.jax_enable_x64
    plan = json.loads(args.plan.read_text())
    assert plan["recipe"] == _RECIPE
    assert (
        plan["optimization_objectives"]
        == plan["selection_criteria"]
        == list(OBJECTIVES)
    )
    assert plan["checkpoints"] == list(range(0, 1001, 100))
    stage = plan["stages"][args.stage]
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
    sources = (
        sorted(root.glob("src/**/*.py"))
        + sorted(root.glob("scripts/*.py"))
        + [root / "tests/test_checkpoint_attribution.py"]
    )
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(root)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    rows = []
    for family in stage["families"]:
        for data_seed in stage["data_seeds"]:
            supplied, _ = generate(plan["dataset"], family, data_seed, "calibration")
            train, development = prepare(supplied)
            b, dev = train.batch, development.batch
            initial = initialize_sequence_model(
                b,
                kind=_RECIPE["kind"],
                width=_RECIPE["width"],
                ridge=_RECIPE["ridge_fraction"] * len(b.past_states) * 5,
            )
            scale = initial.norms["state_scale"]
            original = np.maximum(
                np.sqrt(
                    np.mean((b.future_states - b.past_states[:, -1:]) ** 2, axis=0)
                ),
                0.01 * scale,
            )
            scales = dict(
                original=original, first_step_floor=first_step_floor(original)
            )
            same = np.array_equal(*scales.values())
            for fit_seed in stage["fit_seeds"]:
                name = f"{family}-{data_seed}-fit{fit_seed}"
                print(
                    json.dumps(
                        dict(starting=name, stage=args.stage, identical_objective=same)
                    ),
                    flush=True,
                )
                directory = args.output / name
                directory.mkdir()
                for label, windows in (("train", train), ("development", development)):
                    np.savez_compressed(
                        directory / f"{label}.npz",
                        **{k: getattr(windows.batch, k) for k in ARRAY_NAMES},
                        recording_ids=np.array([k.recording_id for k in windows.keys]),
                        segment_ids=np.array([k.segment_id for k in windows.keys]),
                        origins=np.array([k.origin for k in windows.keys]),
                    )
                paths, traces, selected, dev_predictions = {}, {}, {}, {}
                dev_arrays = {k: getattr(dev, k) for k in ARRAY_NAMES}
                dev_ids = np.array([k.recording_id for k in development.keys])
                for objective in OBJECTIVES:
                    if objective == "first_step_floor" and same:
                        paths[objective], traces[objective] = (
                            paths["original"],
                            traces["original"],
                        )
                    else:
                        paths[objective], traces[objective] = fit_checkpoints(
                            b, dev, scales[objective], _RECIPE, fit_seed
                        )
                    models = paths[objective]
                    assert [t["step"] for t in traces[objective]] == plan["checkpoints"]
                    for step, model in zip(plan["checkpoints"], models, strict=True):
                        model.save(directory / f"{objective}-{step:04d}.npz")
                    dev_predictions[objective] = predictions(models, dev_arrays)
                    selected[objective] = {
                        criterion: select(
                            dev_predictions[objective],
                            dev.future_states,
                            scales[criterion],
                            dev_ids,
                        )
                        for criterion in OBJECTIVES
                    }
                    own = selected[objective][objective]
                    assert own["selected_index"] == int(
                        np.argmin(
                            [t["validation_rollout_mse"] for t in traces[objective]]
                        )
                    )
                    np.testing.assert_allclose(
                        own["losses"],
                        [t["validation_rollout_mse"] for t in traces[objective]],
                        atol=1e-10,
                        rtol=1e-8,
                    )
                parity = {}
                if args.stage == "known" and fit_seed == 0:
                    prior_dir = args.previous / f"{family}-{data_seed}"
                    base = LearnedDynamics.load(prior_dir / "baseline.npz")
                    for role, cached in (
                        ("train", base._train),
                        ("development", base._development),
                    ):
                        target = train if role == "train" else development
                        assert target.keys == cached.keys
                        for key in ARRAY_NAMES:
                            np.testing.assert_array_equal(
                                getattr(target.batch, key), getattr(cached.batch, key)
                            )
                    for objective in OBJECTIVES:
                        previous = (
                            base._model
                            if objective == "original"
                            else SequenceModel.load(prior_dir / "first_step_floor.npz")
                        )
                        report = (
                            base.report["optimization"]
                            if objective == "original"
                            else json.loads(
                                (prior_dir / "first_step_floor-report.json").read_text()
                            )
                        )
                        chosen = paths[objective][
                            selected[objective][objective]["selected_index"]
                        ]
                        assert chosen.fingerprint() == previous.fingerprint()
                        for current, old in zip(
                            traces[objective], report["trace"], strict=True
                        ):
                            assert current.keys() == old.keys()
                            for key in current:
                                np.testing.assert_allclose(
                                    current[key], old[key], rtol=1e-10, atol=1e-12
                                )
                        parity[objective] = dict(
                            fingerprint=chosen.fingerprint(),
                            source=str(prior_dir),
                            selected_step=report["selected_step"],
                            complete_trace_reproduced=True,
                        )
                write(directory / "selection.json", selected)
                write(directory / "traces.json", traces)
                np.savez_compressed(
                    directory / "development-predictions.npz", **dev_predictions
                )
                # All selectors are frozen above before evaluation recordings are generated.
                regimes = {}
                for regime in ("matched", "shifted"):
                    evaluation, _ = generate(plan["dataset"], family, data_seed, regime)
                    arrays = evaluate_arrays(
                        evaluation, plan["dataset"]["evaluation_stride"]
                    )
                    all_predictions = {
                        key: predictions(models, arrays)
                        for key, models in paths.items()
                    }
                    np.savez_compressed(
                        directory / f"{regime}.npz",
                        **arrays,
                        **all_predictions,
                        state_scale=scale,
                    )
                    metrics = {}
                    for objective, values in all_predictions.items():
                        metrics[objective] = dict(
                            checkpoints=[
                                measure(p, arrays["targets"], scale) for p in values
                            ],
                            recordings={
                                str(record): [
                                    measure(
                                        p[arrays["recording_ids"] == record],
                                        arrays["targets"][
                                            arrays["recording_ids"] == record
                                        ],
                                        scale,
                                    )
                                    for p in values
                                ]
                                for record in sorted(set(arrays["recording_ids"]))
                            },
                        )
                    regimes[regime] = dict(
                        windows=len(arrays["targets"]), metrics=metrics
                    )
                row = dict(
                    family=family,
                    data_seed=data_seed,
                    fit_seed=fit_seed,
                    stage=args.stage,
                    status="complete",
                    identical_objective_reused=same,
                    loss_scales={k: v.tolist() for k, v in scales.items()},
                    selected=selected,
                    parity=parity,
                    regimes=regimes,
                )
                write(directory / "result.json", row)
                rows.append(row)
                write(args.output / "results.json", rows)
                print(
                    json.dumps(
                        dict(
                            completed=name,
                            selected_steps={
                                o: {
                                    s: plan["checkpoints"][v["selected_index"]]
                                    for s, v in criteria.items()
                                }
                                for o, criteria in selected.items()
                            },
                        )
                    ),
                    flush=True,
                )
    assert len(rows) == len(stage["families"]) * len(stage["data_seeds"]) * len(
        stage["fit_seeds"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("known", "fresh"), required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
