"""One frozen offline acceptance decision for the causal-memory candidate (M2).

The incumbent models, their window identities, evaluation origins, state scales
and saved predictions come from the preserved M1 run. Recordings regenerated
from the archived generator must reproduce every cached array before a fit.
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
import scipy
from check_generic_fit import measure, read, recurrence, sha, write
from experiment_horizon_generalization import generate
from experiment_model_qualification import (
    collection as witness_collection,
)
from experiment_model_qualification import (
    encoding,
    memory_probe,
    memory_rollout,
)
from experiment_model_qualification import (
    recordings as witness_recordings,
)

from glassbox.experimental.default_model import _HISTORY_RECIPE
from glassbox.experimental.default_model import _fit_history as consumer_fit
from glassbox.experimental.sequence_collection import WindowKey
from glassbox.experimental.sequence_model import (
    SequenceModel,
    fit_sequence_model,
    initialize_sequence_model,
)

_MANIFEST_SHA256 = "02ac887c28eae15efa9afc9168ed5d02998596e3e86680644b59382f7ef2b1b7"
ARRAYS = ("past_states", "past_inputs", "future_inputs", "future_states")
WITNESS = "hidden_input_delay"
REPRODUCTION_TOLERANCE = 1e-12


def frozen_manifest(path):
    if sha(path) != _MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen M2 contract")
    return read(path)


def recurrence_memory(model, x, up, uf):
    """Independent NumPy replay of the filter recurrence; no training code or JAX."""
    if model.kind != "filter_mlp":
        raise ValueError("M2 replay requires the filter_mlp representation")
    n, p, delay = model.norms, model.params, model.delay_steps
    states = (x - n["state_mean"]) / n["state_scale"]
    inputs = (up - n["input_mean"]) / n["input_scale"]
    future = (uf - n["input_mean"]) / n["input_scale"]

    def features(current, command, history, commands, hidden):
        return (
            np.column_stack(
                (
                    current,
                    command,
                    (history - current[:, None]).reshape(len(x), -1),
                    (commands - command[:, None]).reshape(len(x), -1),
                    hidden,
                )
            )
            / n["feature_scale"]
        )

    hidden = np.zeros((len(x), p["memory"].shape[1]))
    context = inputs.shape[1]
    for j in range(delay, context):
        z = features(
            states[:, j],
            inputs[:, j],
            states[:, j - delay : j],
            inputs[:, j - delay : j],
            hidden,
        )
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
    history = states[:, context - delay :]
    commands = inputs[:, context - delay :]
    output = []
    for command in future.transpose(1, 0, 2):
        current = history[:, -1]
        z = features(current, command, history[:, :-1], commands, hidden)
        delta = z @ p["linear"] + p["bias"]
        delta += np.tanh(z @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[:, 1:], predicted[:, None]), axis=1)
        commands = np.concatenate((commands[:, 1:], command[:, None]), axis=1)
    return np.stack(output, axis=1)


def lengths(recipe, dt_s):
    return {
        k: max(1, int(np.rint(recipe[f"{k}_s"] / dt_s)))
        for k in ("context", "delay", "horizon")
    }


def paired_rmse(prediction, target):
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 2)))


def context_probe(context, horizon):
    """The G05 paired probe with a full consumed context before the origin."""
    full = np.zeros((2, context + horizon, 1))
    full[:, context - 3, 0] = [-1.0, 1.0]
    states = np.stack([memory_rollout(np.zeros(1), u) for u in full])
    return (
        states[:, : context + 1],
        full[:, :context],
        full[:, context:],
        states[:, context + 1 :],
    )


def context_windows(supplied, keys, context, horizon):
    """Re-extract frozen window identities with the full context; drop short ones."""
    kept = [k for k in keys if k.origin >= context]
    windows = supplied.extract(kept, history_steps=context, horizon_steps=horizon)
    return windows, kept


def evaluation_rows(supplied, stride, context, horizon):
    """The M1 evaluation origins that have a complete consumed context."""
    index = [
        (s, t)
        for s in supplied.segments
        for t in range(2, len(s.states) - horizon, stride)
        if t >= context
    ]
    return dict(
        past_states=np.stack([s.states[t - context : t + 1] for s, t in index]),
        past_inputs=np.stack([s.inputs[t - context : t] for s, t in index]),
        future_inputs=np.stack([s.inputs[t : t + horizon] for s, t in index]),
        targets=np.stack([s.states[t + 1 : t + horizon + 1] for s, t in index]),
        recording_ids=np.array([s.recording_id for s, _ in index]),
        source_origins=np.array([s.start_row + t for s, t in index]),
    )


def save_windows(path, windows, kept):
    np.savez_compressed(
        path,
        **{k: getattr(windows.batch, k) for k in ARRAYS},
        recording_ids=np.array([k.recording_id for k in kept]),
        segment_ids=np.array([k.segment_id for k in kept]),
        origins=np.array([k.origin for k in kept]),
    )


def window_counts(keys, kept, windows, path):
    return dict(
        sha256=sha(path),
        source_windows=len(keys),
        windows=len(kept),
        dropped_short_context=len(keys) - len(kept),
        recordings={
            r: sum(k.recording_id == r for k in kept)
            for r in sorted({k.recording_id for k in keys})
        },
        coverage=windows.coverage(),
    )


def fit_candidate(train, development, recipe, steps):
    """Fit the frozen candidate recipe through the maintained sequence fitter."""
    b = train.batch
    ridge = recipe["ridge_fraction"] * len(b.past_states) * b.future_states.shape[1]
    initial = initialize_sequence_model(
        b,
        kind=recipe["kind"],
        width=recipe["width"],
        memory=recipe["memory"],
        seed=recipe["seed"],
        ridge=ridge,
        delay_steps=steps["delay"],
    )
    # The rest start reads out as zero, so checkpoint zero is the affine start.
    if np.any(initial.params["linear"][-recipe["memory"] :] != 0) or np.any(
        initial.params["w2"] != 0
    ):
        raise ValueError("initial memory readout is not zero")
    hold = np.repeat(b.past_states[:, -1:], b.future_states.shape[1], 1)
    scale = np.maximum(
        np.sqrt(np.mean((hold - b.future_states) ** 2, 0)),
        recipe["hold_scale_floor"] * initial.norms["state_scale"],
    )
    started = time.perf_counter()
    model, report = fit_sequence_model(
        b,
        development.batch,
        kind=recipe["kind"],
        objective="rollout",
        width=recipe["width"],
        memory=recipe["memory"],
        ridge=ridge,
        seed=recipe["seed"],
        steps=recipe["steps"],
        batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"],
        check_every=recipe["check_every"],
        error_scale=scale,
        delay_steps=steps["delay"],
    )
    wall = time.perf_counter() - started
    n = len(b.past_states)
    # Full-batch losses of the selected checkpoint make every cache replayable.
    losses = {}
    for role, windows in (("train", train), ("development", development)):
        w = windows.batch
        prediction = np.asarray(
            model.rollout(w.past_states, w.past_inputs, w.future_inputs)
        )
        losses[role] = float(np.mean(((prediction - w.future_states) / scale) ** 2))
    np.testing.assert_allclose(
        losses["development"], report["validation_rollout_mse"], rtol=1e-8, atol=1e-12
    )
    report.update(
        selected_training_mse=losses["train"],
        selected_development_mse=losses["development"],
        optimizer="incumbent-adam-recipe",
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        initial_fingerprint=initial.fingerprint(),
        resource_usage=dict(
            training_window_gradient_evaluations=recipe["steps"]
            * min(recipe["batch_size"], n),
            development_passes=len(report["trace"]),
            training_windows=n,
            development_windows=len(development.batch.past_states),
            context_transitions_per_window=steps["context"] - steps["delay"],
        ),
    )
    return model, report, wall


def correctness(model, windows, steps):
    """Prefix consistency, causal input derivatives, rest start, carried memory."""
    b = windows.batch
    x, up, uf = b.past_states[:1], b.past_inputs[:1], b.future_inputs[:1]
    whole = model.rollout(x, up, uf)
    np.testing.assert_allclose(
        model.rollout(x, up, uf[:, :2]), whole[:, :2], rtol=1e-9, atol=1e-10
    )
    derivative = jax.jacfwd(lambda u: model.rollout(x[0], up[0], u))(uf[0])
    if not np.isfinite(derivative).all() or not np.isfinite(whole).all():
        raise ValueError("nonfinite candidate prediction or derivative")
    for h in range(uf.shape[1]):
        np.testing.assert_array_equal(derivative[h, :, h + 1 :], 0)
    split, delay = steps["context"] // 2, steps["delay"]
    rest = np.zeros((1, model.params["memory"].shape[1]))
    np.testing.assert_array_equal(
        model.memory_state(x, up), model.memory_state(x, up, memory=rest)
    )
    carried = model.memory_state(x[:, : split + 1], up[:, :split])
    continued = model.rollout(
        x[:, split - delay :], up[:, split - delay :], uf, memory=carried
    )
    np.testing.assert_allclose(continued, whole, rtol=1e-9, atol=1e-10)
    return dict(
        prefix_consistent=True,
        causal_input_derivative=True,
        rest_start=True,
        carried_memory_split=split,
    )


def score(predicted, targets, ids, scale):
    result = {o: measure(p, targets, scale) for o, p in predicted.items()}
    result["recordings"] = {
        str(r): {
            o: measure(p[ids == r], targets[ids == r], scale)
            for o, p in predicted.items()
        }
        for r in sorted(set(ids))
    }
    return result


def replay_both(incumbent, candidate, arrays):
    h = incumbent.history_steps
    short = (arrays["past_states"][:, -h - 1 :], arrays["past_inputs"][:, -h:])
    predicted = dict(
        incumbent=np.asarray(incumbent.rollout(*short, arrays["future_inputs"])),
        candidate=np.asarray(
            candidate.rollout(
                arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
            )
        ),
    )
    replays = dict(
        incumbent=recurrence(incumbent, *short, arrays["future_inputs"]),
        candidate=recurrence_memory(
            candidate,
            arrays["past_states"],
            arrays["past_inputs"],
            arrays["future_inputs"],
        ),
    )
    worst = 0.0
    for kind in predicted:
        worst = max(worst, float(np.max(np.abs(replays[kind] - predicted[kind]))))
        np.testing.assert_allclose(replays[kind], predicted[kind], rtol=1e-8, atol=1e-9)
        if not np.isfinite(predicted[kind]).all():
            raise ValueError(f"nonfinite {kind} prediction")
    return predicted, worst


def benchmark_case(case, m1, prior, plan, recipe, steps, output):
    name = f"{case['family']}-{case['data_seed']}"
    source = m1 / name
    directory = output / name
    directory.mkdir()
    incumbent = SequenceModel.load(source / "incumbent.npz")
    if incumbent.fingerprint() != prior["incumbent_fingerprint"]:
        raise ValueError(f"incumbent fingerprint differs from the M1 record: {name}")
    incumbent.save(directory / "incumbent.npz")
    calibration, _ = generate(plan, case["family"], case["data_seed"], "calibration")
    segments = {s.recording_id: s for s in calibration.segments}
    h = incumbent.history_steps
    windows, counts, regeneration = {}, {}, 0.0
    for role in ("train", "development"):
        with np.load(source / f"{role}.npz", allow_pickle=False) as data:
            ids, segment_ids = data["recording_ids"], data["segment_ids"]
            origins = data["origins"]
            for i, (r, o) in enumerate(zip(ids, origins)):
                s = segments[str(r)]
                for cached, fresh in (
                    (data["past_states"][i], s.states[o - h : o + 1]),
                    (data["past_inputs"][i], s.inputs[o - h : o]),
                    (data["future_inputs"][i], s.inputs[o : o + steps["horizon"]]),
                    (
                        data["future_states"][i],
                        s.states[o + 1 : o + steps["horizon"] + 1],
                    ),
                ):
                    regeneration = max(
                        regeneration, float(np.max(np.abs(cached - fresh)))
                    )
            keys = [
                WindowKey(str(r), str(g), int(o))
                for r, g, o in zip(ids, segment_ids, origins)
            ]
        if regeneration > REPRODUCTION_TOLERANCE:
            raise ValueError("regenerated calibration differs from the M1 cache")
        windows[role], kept = context_windows(
            calibration, keys, steps["context"], steps["horizon"]
        )
        save_windows(directory / f"{role}.npz", windows[role], kept)
        counts[role] = window_counts(
            keys, kept, windows[role], directory / f"{role}.npz"
        )
    roles = {r: {k.recording_id for k in windows[r].keys} for r in windows}
    if roles["train"] & roles["development"]:
        raise ValueError("training and development recording identities overlap")
    print(json.dumps(dict(starting=name)), flush=True)
    candidate, optimization, wall = fit_candidate(
        windows["train"], windows["development"], recipe, steps
    )
    candidate.save(directory / "candidate.npz")
    write(directory / "optimization.json", optimization)
    checks = correctness(candidate, windows["development"], steps)
    regimes, replay_max = {}, 0.0
    for regime in ("matched", "shifted"):
        supplied, _ = generate(plan, case["family"], case["data_seed"], regime)
        arrays = evaluation_rows(
            supplied, plan["evaluation_stride"], steps["context"], steps["horizon"]
        )
        with np.load(source / f"{regime}.npz", allow_pickle=False) as saved:
            keep = saved["source_origins"] >= steps["context"]
            if list(saved["recording_ids"][keep]) != list(
                arrays["recording_ids"]
            ) or list(saved["source_origins"][keep]) != list(arrays["source_origins"]):
                raise ValueError("evaluation rows differ from the M1 origins")
            for key, fresh in (
                ("past_states", arrays["past_states"][:, -h - 1 :]),
                ("past_inputs", arrays["past_inputs"][:, -h:]),
                ("future_inputs", arrays["future_inputs"]),
                ("targets", arrays["targets"]),
            ):
                regeneration = max(
                    regeneration, float(np.max(np.abs(saved[key][keep] - fresh)))
                )
            incumbent_saved = saved["incumbent"][keep]
            state_scale = saved["state_scale"]
            dropped = int((~keep).sum())
        if regeneration > REPRODUCTION_TOLERANCE:
            raise ValueError("regenerated evaluation differs from the M1 arrays")
        ids = arrays["recording_ids"]
        if set(ids) & (roles["train"] | roles["development"]):
            raise ValueError("evaluation recording identities overlap fit data")
        predicted, worst = replay_both(incumbent, candidate, arrays)
        replay_max = max(replay_max, worst)
        np.testing.assert_allclose(
            predicted["incumbent"], incumbent_saved, rtol=1e-9, atol=1e-10
        )
        np.savez_compressed(
            directory / f"{regime}.npz", **arrays, **predicted, state_scale=state_scale
        )
        regimes[regime] = dict(
            windows=len(ids),
            dropped_short_context=dropped,
            **score(predicted, arrays["targets"], ids, state_scale),
        )
    row = dict(
        name=name,
        family=case["family"],
        data_seed=case["data_seed"],
        extra=case["extra"],
        status="complete",
        correctness_passed=True,
        correctness=checks,
        incumbent_fingerprint=incumbent.fingerprint(),
        candidate_fingerprint=candidate.fingerprint(),
        initial_fingerprint=optimization["initial_fingerprint"],
        m1_source=name,
        windows=counts,
        regeneration_max_difference=regeneration,
        replay_max_difference=replay_max,
        optimization=optimization,
        fit_wall_seconds=wall,
        regimes=regimes,
    )
    write(directory / "result.json", row)
    return row


def witness_case(seed, manifest, recipe, steps, output):
    spec = manifest[WITNESS]
    name = f"{WITNESS}-{seed}"
    directory = output / name
    directory.mkdir()
    transform = encoding("identity", 1)
    calibration = witness_collection(
        witness_recordings("memory", seed), transform, "memory"
    )
    print(json.dumps(dict(starting=name, incumbent="consumer fit")), flush=True)
    started = time.perf_counter()
    learned = consumer_fit(calibration)
    incumbent_wall = time.perf_counter() - started
    if learned.report["recipe"] != _HISTORY_RECIPE:
        raise ValueError("incumbent reproduction did not use the frozen recipe")
    incumbent = learned._model
    learned.save(directory / "incumbent-revision.npz")
    incumbent.save(directory / "incumbent.npz")
    write(directory / "incumbent-report.json", learned.report)
    px, pu, uf, target = memory_probe()
    h = incumbent.history_steps
    short = (px[:, -h - 1 :], pu[:, -h:])
    incumbent_probe = np.asarray(learned.predict(px, pu, uf))
    np.testing.assert_array_equal(incumbent_probe[0], incumbent_probe[1])
    np.testing.assert_array_equal(
        incumbent_probe, np.asarray(incumbent.rollout(*short, uf))
    )
    incumbent_paired = paired_rmse(incumbent_probe, target)
    floor = np.abs(target[1, :, 0] - target[0, :, 0]) / 2
    if np.any(incumbent_paired < floor - 1e-9):
        raise ValueError("incumbent paired-probe error is below the exact floor")
    windows, counts = {}, {}
    for role, cached in (
        ("train", learned._train),
        ("development", learned._development),
    ):
        windows[role], kept = context_windows(
            calibration, cached.keys, steps["context"], steps["horizon"]
        )
        save_windows(directory / f"{role}.npz", windows[role], kept)
        counts[role] = window_counts(
            cached.keys, kept, windows[role], directory / f"{role}.npz"
        )
    roles = {r: {k.recording_id for k in windows[r].keys} for r in windows}
    if roles["train"] & roles["development"]:
        raise ValueError("training and development recording identities overlap")
    candidate, optimization, wall = fit_candidate(
        windows["train"], windows["development"], recipe, steps
    )
    candidate.save(directory / "candidate.npz")
    write(directory / "optimization.json", optimization)
    checks = correctness(candidate, windows["development"], steps)
    state_scale = incumbent.norms["state_scale"]
    supplied = witness_collection(
        witness_recordings("memory", seed, evaluation=True), transform, "memory"
    )
    arrays = evaluation_rows(supplied, 5, steps["context"], steps["horizon"])
    ids = arrays["recording_ids"]
    if set(ids) & (roles["train"] | roles["development"]):
        raise ValueError("evaluation recording identities overlap fit data")
    predicted, replay_max = replay_both(incumbent, candidate, arrays)
    np.testing.assert_array_equal(
        predicted["incumbent"],
        np.asarray(
            learned.predict(
                arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
            )
        ),
    )
    np.savez_compressed(
        directory / "matched.npz", **arrays, **predicted, state_scale=state_scale
    )
    regimes = dict(
        matched=dict(
            windows=len(ids), **score(predicted, arrays["targets"], ids, state_scale)
        )
    )
    cpx, cpu, cuf, ctarget = context_probe(steps["context"], steps["horizon"])
    np.testing.assert_allclose(ctarget, target, atol=1e-12)
    np.testing.assert_array_equal(cpx[0], cpx[1])
    np.testing.assert_array_equal(cuf[0], cuf[1])
    if np.count_nonzero(cpu[0] != cpu[1]) != 1:
        raise ValueError("probe contexts must differ in exactly one input")
    candidate_probe = np.asarray(candidate.rollout(cpx, cpu, cuf))
    replay = recurrence_memory(candidate, cpx, cpu, cuf)
    replay_max = max(replay_max, float(np.max(np.abs(replay - candidate_probe))))
    np.testing.assert_allclose(replay, candidate_probe, rtol=1e-8, atol=1e-9)
    if not np.isfinite(candidate_probe).all():
        raise ValueError("nonfinite candidate probe prediction")
    candidate_paired = paired_rmse(candidate_probe, ctarget)
    np.savez_compressed(
        directory / "probe.npz",
        incumbent_past_states=short[0],
        incumbent_past_inputs=short[1],
        candidate_past_states=cpx,
        candidate_past_inputs=cpu,
        future_inputs=cuf,
        targets=ctarget,
        incumbent_prediction=incumbent_probe,
        candidate_prediction=candidate_probe,
    )
    probe = dict(
        incumbent_paired_rmse=incumbent_paired.tolist(),
        candidate_paired_rmse=candidate_paired.tolist(),
        minimum_possible_common_rmse=floor.tolist(),
        independent_input_first_step_noise_floor=0.2 / np.sqrt(3),
        candidate_branches_identical=bool(
            np.array_equal(candidate_probe[0], candidate_probe[1])
        ),
        meaning="Paired RMSE over the two branches per horizon; the floor binds any prediction that cannot see the withheld input.",
    )
    recorded = spec["recorded_qualification_fingerprints"][str(seed)]
    row = dict(
        name=name,
        family=WITNESS,
        data_seed=seed,
        extra=False,
        status="complete",
        correctness_passed=True,
        correctness=checks,
        incumbent_fingerprint=incumbent.fingerprint(),
        incumbent_revision_fingerprint=learned.fingerprint(),
        recorded_qualification_fingerprint=recorded,
        recorded_fingerprint_reproduced=learned.fingerprint() == recorded,
        incumbent_selected_step=learned.report["optimization"]["selected_step"],
        incumbent_development_mse=learned.report["optimization"][
            "validation_rollout_mse"
        ],
        incumbent_fit_wall_seconds=incumbent_wall,
        candidate_fingerprint=candidate.fingerprint(),
        initial_fingerprint=optimization["initial_fingerprint"],
        windows=counts,
        replay_max_difference=replay_max,
        optimization=optimization,
        fit_wall_seconds=wall,
        regimes=regimes,
        probe=probe,
    )
    write(directory / "result.json", row)
    return row


def _finite(value):
    return None if value is None or not np.isfinite(value) else float(value)


def decide(manifest, rows):
    required = manifest["required_families"]
    capability = manifest["capability_families"]
    seeds = manifest["data_seeds"]
    witness = manifest[WITNESS]
    families = required + [f for f in capability if f != WITNESS]
    expected = {
        (f, s, False) for f in families + manifest["stress_families"] for s in seeds
    }
    expected |= {(WITNESS, s, False) for s in witness["data_seeds"]}
    expected |= {
        (c["family"], c["data_seed"], True) for c in manifest["extra_regression_cases"]
    }
    keys = [(r["family"], r["data_seed"], r["extra"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(
            "acceptance requires exactly the frozen cases, without duplicates"
        )
    breaches, capability_breaches = [], []
    logs = {f: [] for f in required + capability}
    probes = {}
    floor = manifest["primary"]["floor"]
    for row in rows:
        if row["status"] != "complete":
            breaches.append(dict(case=row["name"], gate="fit_complete"))
            continue
        if not row["correctness_passed"]:
            breaches.append(dict(case=row["name"], gate="correctness"))
        regimes = (
            set(witness["regimes"])
            if row["family"] == WITNESS
            else {"matched", "shifted"}
        )
        if set(row["regimes"]) != regimes:
            raise ValueError("acceptance requires the frozen evaluation regimes")
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
                        value=_finite(horizons.max())
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
            if row["family"] in logs and not row["extra"]:
                logs[row["family"]].append(float(np.log(max(b, floor) / max(a, floor))))
        if row["family"] == WITNESS:
            values = row.get("probe", {}).get("candidate_paired_rmse") or [None]
            value = _finite(values[0]) if values[0] is not None else None
            probes[str(row["data_seed"])] = value
            limit = manifest["capability"]["paired_probe_first_step_rmse_maximum"]
            if value is None or value < 0 or value > limit:
                capability_breaches.append(
                    dict(
                        case=row["name"],
                        gate="paired_probe_first_step_rmse",
                        value=value,
                        limit=limit,
                    )
                )
    counts = {f: len(seeds) * 2 for f in families}
    counts[WITNESS] = len(witness["data_seeds"]) * len(witness["regimes"])
    complete = all(len(logs[f]) == counts[f] for f in logs)
    ratios = {f: float(np.exp(np.mean(v))) for f, v in logs.items() if v}
    for f in capability:
        limit = manifest["capability"][f"{f}_maximum_ratio"]
        if f not in ratios or ratios[f] > limit:
            capability_breaches.append(
                dict(family=f, gate=f"{f}_ratio", value=ratios.get(f), limit=limit)
            )
    ordinary = (
        float(np.exp(np.mean([np.mean(logs[f]) for f in required])))
        if complete
        else None
    )
    primary = (
        float(np.exp(np.mean([np.mean(logs[f]) for f in required + capability])))
        if complete
        else None
    )
    guard_limit = manifest["guards"]["ordinary_maximum_ratio"]
    guard_passed = ordinary is not None and ordinary <= guard_limit
    accepted = bool(
        not breaches
        and not capability_breaches
        and complete
        and guard_passed
        and primary <= manifest["primary"]["maximum_ratio"]
    )
    return dict(
        manifest=manifest["id"],
        decision="replace" if accepted else "retain",
        accepted=accepted,
        hard_gate_breaches=breaches,
        capability_gate_breaches=capability_breaches,
        ordinary_guard=dict(
            ratio=ordinary, maximum_ratio=guard_limit, passed=guard_passed
        ),
        primary_ratio=primary,
        required_ratio=manifest["primary"]["maximum_ratio"],
        per_family_ratios=ratios,
        paired_probe_first_step_rmse=probes,
        meaning="Engineering capability and regression qualification for this manifest only; no platform/control-readiness, general memory-adequacy, or uncertainty claim.",
    )


def verify_run(directory, manifest_path):
    """Recompute scores, fit losses, and the decision from preserved arrays; never fit."""
    manifest = read(directory / "manifest.json")
    if manifest != frozen_manifest(manifest_path):
        raise ValueError("saved manifest differs from the frozen acceptance contract")
    with zipfile.ZipFile(directory / "executed-sources.zip") as archive:
        for name, digest in read(directory / "sources.json").items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"executed source hash mismatch: {name}")
    rows = read(directory / "results.json")
    steps = lengths(manifest["candidate_recipe"], manifest["dataset"]["dt_s"])
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
        scale = np.asarray(optimization["error_scale"])
        identities = {}
        for role in ("train", "development"):
            if sha(case / f"{role}.npz") != row["windows"][role]["sha256"]:
                raise ValueError(f"input cache hash mismatch: {row['name']}/{role}")
            with np.load(case / f"{role}.npz", allow_pickle=False) as data:
                identities[role] = set(data["recording_ids"])
                if (
                    np.any(data["origins"] < steps["context"])
                    or data["past_states"].shape[1] != steps["context"] + 1
                ):
                    raise ValueError("cached windows lack the frozen context")
                prediction = recurrence_memory(
                    models["candidate"],
                    data["past_states"],
                    data["past_inputs"],
                    data["future_inputs"],
                )
                losses[f"{row['name']}/{role}"] = float(
                    np.mean(((prediction - data["future_states"]) / scale) ** 2)
                )
        for role, key in (
            ("development", "validation_rollout_mse"),
            ("train", "selected_training_mse"),
        ):
            np.testing.assert_allclose(
                losses[f"{row['name']}/{role}"],
                optimization[key],
                rtol=1e-8,
                atol=1e-12,
            )
        if identities["train"] & identities["development"]:
            raise ValueError("training and development recording identities overlap")
        h = models["incumbent"].history_steps
        for regime in row["regimes"]:
            with np.load(case / f"{regime}.npz", allow_pickle=False) as data:
                if set(data["recording_ids"]) & (
                    identities["train"] | identities["development"]
                ):
                    raise ValueError("evaluation recording identities overlap fit data")
                if np.any(data["source_origins"] < steps["context"]):
                    raise ValueError("evaluation rows lack the frozen context")
                replays = dict(
                    incumbent=recurrence(
                        models["incumbent"],
                        data["past_states"][:, -h - 1 :],
                        data["past_inputs"][:, -h:],
                        data["future_inputs"],
                    ),
                    candidate=recurrence_memory(
                        models["candidate"],
                        data["past_states"],
                        data["past_inputs"],
                        data["future_inputs"],
                    ),
                )
                for kind, prediction in replays.items():
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
        if row["family"] == WITNESS:
            with np.load(case / "probe.npz", allow_pickle=False) as data:
                replays = dict(
                    incumbent=recurrence(
                        models["incumbent"],
                        data["incumbent_past_states"],
                        data["incumbent_past_inputs"],
                        data["future_inputs"],
                    ),
                    candidate=recurrence_memory(
                        models["candidate"],
                        data["candidate_past_states"],
                        data["candidate_past_inputs"],
                        data["future_inputs"],
                    ),
                )
                floor = np.abs(data["targets"][1, :, 0] - data["targets"][0, :, 0]) / 2
                for kind, prediction in replays.items():
                    np.testing.assert_allclose(
                        prediction, data[f"{kind}_prediction"], rtol=1e-8, atol=1e-9
                    )
                    paired = paired_rmse(prediction, data["targets"])
                    np.testing.assert_allclose(
                        paired,
                        row["probe"][f"{kind}_paired_rmse"],
                        rtol=1e-8,
                        atol=1e-10,
                    )
                    row["probe"][f"{kind}_paired_rmse"] = paired.tolist()
                    checks += 1
                if np.any(
                    paired_rmse(replays["incumbent"], data["targets"]) < floor - 1e-9
                ):
                    raise ValueError("incumbent probe error is below the exact floor")
    decision = decide(manifest, rows)
    saved = read(directory / "decision.json")
    for key in ("decision", "accepted", "manifest"):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed acceptance decision differs: {key}")
    np.testing.assert_allclose(
        decision["primary_ratio"], saved["primary_ratio"], rtol=1e-10
    )
    np.testing.assert_allclose(
        decision["ordinary_guard"]["ratio"],
        saved["ordinary_guard"]["ratio"],
        rtol=1e-10,
    )
    return dict(
        verified_cases=len(rows),
        model_regime_replays=checks,
        decision=decision,
        objective_losses=losses,
        meaning="Saved evidence replay, not a new fit or independent statistical confirmation. Fit resource counts and derivative checks remain executed-run evidence.",
    )


def main(args):
    root = Path(__file__).resolve().parents[1]
    manifest = frozen_manifest(args.manifest)
    reference = manifest["m1_reference"]
    m1_manifest = root / "docs/generic-fit-acceptance.json"
    if sha(m1_manifest) != reference["manifest_sha256"]:
        raise ValueError("M1 manifest differs from the frozen reference")
    evidence = root / reference["evidence"]
    if sha(evidence) != reference["evidence_sha256"]:
        raise ValueError("M1 evidence archive differs from the frozen reference")
    recipe = manifest["candidate_recipe"]
    plan = manifest["dataset"]
    steps = lengths(recipe, plan["dt_s"])
    budget = manifest["information_budget"]
    if (steps["context"], steps["delay"]) != (
        budget["context_steps"],
        budget["delay_steps"],
    ) or recipe["memory"] != budget["memory_coordinates"]:
        raise ValueError("candidate recipe differs from the frozen information budget")
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
            backend=jax.default_backend(),
        ),
    )
    sources = (
        sorted(root.glob("src/**/*.py"))
        + sorted(root.glob("scripts/*.py"))
        + [
            root / "pyproject.toml",
            root / "uv.lock",
            root / "tests/test_causal_memory.py",
            root / "tests/test_generic_memory_acceptance.py",
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
    inputs = args.output / "m1-inputs"
    with zipfile.ZipFile(evidence) as archive:
        archive.extractall(inputs)
    m1 = inputs / "acceptance-01"
    if read(m1 / "manifest.json") != read(m1_manifest):
        raise ValueError("preserved M1 run does not carry the frozen M1 manifest")
    priors = {r["name"]: r for r in read(m1 / "results.json")}
    cases = [
        dict(family=f, data_seed=s, extra=False)
        for f in manifest["required_families"]
        + [f for f in manifest["capability_families"] if f != WITNESS]
        + manifest["stress_families"]
        for s in manifest["data_seeds"]
    ]
    cases += [
        dict(family=c["family"], data_seed=c["data_seed"], extra=True)
        for c in manifest["extra_regression_cases"]
    ]
    rows = []

    def progress(row):
        rows.append(row)
        write(args.output / "results.json", rows)
        summary = dict(
            completed=row["name"],
            selected_step=row["optimization"]["selected_step"],
            windows=row["windows"]["train"]["windows"],
            **{
                regime: dict(
                    incumbent=metrics["incumbent"]["overall_scaled_rmse"],
                    candidate=metrics["candidate"]["overall_scaled_rmse"],
                )
                for regime, metrics in row["regimes"].items()
            },
        )
        if "probe" in row:
            summary["paired_first_step"] = dict(
                incumbent=row["probe"]["incumbent_paired_rmse"][0],
                candidate=row["probe"]["candidate_paired_rmse"][0],
            )
        print(json.dumps(summary), flush=True)

    with jax.enable_x64(True):
        for case in cases:
            name = f"{case['family']}-{case['data_seed']}"
            progress(
                benchmark_case(case, m1, priors[name], plan, recipe, steps, args.output)
            )
        for seed in manifest[WITNESS]["data_seeds"]:
            progress(witness_case(seed, manifest, recipe, steps, args.output))
    decision = decide(manifest, rows)
    write(args.output / "decision.json", decision)
    print(json.dumps(decision), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/generic-memory-acceptance.json",
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
