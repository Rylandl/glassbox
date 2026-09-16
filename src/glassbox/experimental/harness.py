"""One harness for the generic learner: fit frozen cases, score, decide, replay.

``run`` fits the consumer recipe end to end on each case of a frozen manifest,
scores its forecasts on independent recordings, and writes a decision.
``verify`` replays every saved prediction with an independent NumPy recurrence,
recomputes the scores and the decision from the saved arrays, and rejects any
artifact that differs from what the run recorded. Neither command takes tuning
options: the manifest is frozen and its digest is a constant below.

    python -m glassbox.experimental.harness run \\
        --manifest docs/harness/v1.json --output DIR
    python -m glassbox.experimental.harness verify DIR

Synthetic results are a fast regression guard. They are not platform readiness,
control adequacy, or calibrated uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform as platform_module
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .default_model import RECIPE, LearnedDynamics, fit, steps_for
from .sequence_collection import SequenceCollection, SequenceSegment

MANIFEST_SHA256 = "1ba15b3f466e91edf548f1d459a52eb0896310e7fbbb6c1c98544ec10513d781"
"""Digest of the frozen manifest this module is allowed to run and verify."""

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
WITNESS = "hidden_input_delay"
WITNESS_DELAY = 3
REPLAY_TOLERANCE = dict(rtol=1e-8, atol=1e-9)
SCORE_TOLERANCE = dict(rtol=1e-8, atol=1e-10)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_manifest(path):
    """Load the manifest only when its bytes match the frozen digest."""
    if sha256(path) != MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen harness contract")
    manifest = read(path)
    if manifest["recipe"] != RECIPE:
        raise ValueError("manifest recipe differs from the maintained recipe")
    steps = steps_for(manifest["dataset"]["dt_s"])
    budget = manifest["information_budget"]
    if (steps["history"], steps["delay"], steps["horizon"]) != (
        budget["context_steps"],
        budget["delay_steps"],
        budget["horizon_steps"],
    ):
        raise ValueError("manifest information budget differs from the recipe")
    return manifest


# --- the eight synthetic families -------------------------------------------


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
    if family != "noisy_observation":
        raise ValueError(f"unknown family: {family}")
    return 0.92 * x + 0.14 * np.tanh(1.5 * u) + 0.03 * disturbance


def generate(plan, family, seed, regime):
    """Regenerate one regime of one family from the frozen dataset plan."""
    latent_width, input_width, observed_width = dimensions(family)
    segments = []
    n = plan["intervals"]
    count = (
        plan["calibration_recordings"]
        if regime == "calibration"
        else plan["evaluation_recordings_per_regime"]
    )
    settings = plan["regimes"][regime]
    for record in range(count):
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
        segments.append(
            SequenceSegment(f"{regime}-{record}", "whole", states, inputs, plan["dt_s"])
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id=f"synthetic-harness-{family}",
        state_channels=tuple(
            f"x{i} [observed,unitless]" for i in range(observed_width)
        ),
        input_channels=tuple(f"u{i} [requested,unitless]" for i in range(input_width)),
    )


# --- the delayed-input witness ----------------------------------------------


def witness_rollout(inputs):
    """x[k+1] = 0.8 x[k] + 0.2 u[k-3], from rest, with no input before k = 3."""
    u = np.asarray(inputs)
    x = np.zeros((len(u) + 1, u.shape[-1]))
    for k in range(len(u)):
        x[k + 1] = 0.8 * x[k] + (
            0.2 * u[k - WITNESS_DELAY] if k >= WITNESS_DELAY else 0
        )
    return x


def witness_recordings(plan, seed, *, evaluation=False):
    """The G05 witness family: independent uniform commands, one hidden delay."""
    count = (
        plan["evaluation_recordings_per_regime"]
        if evaluation
        else plan["calibration_recordings"]
    )
    segments = []
    for index in range(count):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(evaluation), index])
        )
        state = rng.uniform(-1.5, 1.5, size=1)
        xs, us = [state.copy()], []
        for k in range(plan["intervals"]):
            command = rng.uniform(-1, 1, size=1)
            state = 0.8 * state + (
                0.2 * us[k - WITNESS_DELAY] if k >= WITNESS_DELAY else 0.0
            )
            xs.append(state.copy())
            us.append(command.copy())
        segments.append(
            SequenceSegment(
                f"{'evaluation' if evaluation else 'fit'}-{index}",
                "whole",
                np.array(xs),
                np.array(us),
                plan["dt_s"],
            )
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id=f"synthetic-harness-{WITNESS}",
        state_channels=("x0 [observed,unitless]",),
        input_channels=("u0 [requested,unitless]",),
    )


def witness_probe(context, horizon):
    """A paired probe whose two contexts differ in exactly one withheld input.

    Opposite unit commands three intervals before the origin, zero everywhere
    else. The supplied observations are identical, so a forecast that cannot see
    that command must answer the same in both branches; the true futures differ
    by 0.4 * 0.8**h.
    """
    full = np.zeros((2, context + horizon, 1))
    full[:, context - WITNESS_DELAY, 0] = [-1.0, 1.0]
    states = np.stack([witness_rollout(u) for u in full])
    return (
        states[:, : context + 1],
        full[:, :context],
        full[:, context:],
        states[:, context + 1 :],
    )


# --- evaluation and scoring -------------------------------------------------


def evaluation_rows(supplied, plan, steps):
    """Every declared origin that carries a complete consumed context."""
    context, horizon = steps["history"], steps["horizon"]
    start, stride = plan["evaluation_origin_start"], plan["evaluation_stride"]
    index = [
        (s, t)
        for s in supplied.segments
        for t in range(start, len(s.states) - horizon, stride)
        if t >= context
    ]
    if not index:
        raise ValueError("no evaluation origin has a complete consumed context")
    return dict(
        past_states=np.stack([s.states[t - context : t + 1] for s, t in index]),
        past_inputs=np.stack([s.inputs[t - context : t] for s, t in index]),
        future_inputs=np.stack([s.inputs[t : t + horizon] for s, t in index]),
        targets=np.stack([s.states[t + 1 : t + horizon + 1] for s, t in index]),
        recording_ids=np.array([s.recording_id for s, _ in index]),
        source_origins=np.array([s.start_row + t for s, t in index]),
    )


def measure(prediction, target, state_scale):
    physical = np.mean((np.asarray(prediction) - np.asarray(target)) ** 2, axis=0)
    scaled = physical / np.asarray(state_scale) ** 2
    return dict(
        channel_rmse=np.sqrt(physical).tolist(),
        horizon_scaled_rmse=np.sqrt(np.mean(scaled, axis=1)).tolist(),
        overall_scaled_rmse=float(np.sqrt(np.mean(scaled))),
    )


def score(prediction, targets, ids, state_scale):
    return dict(
        windows=len(targets),
        **measure(prediction, targets, state_scale),
        recordings={
            str(r): measure(prediction[ids == r], targets[ids == r], state_scale)
            for r in sorted(set(ids))
        },
    )


def replay(model, past_states, past_inputs, future_inputs):
    """Independent NumPy recurrence: the memory filter, then the forecast.

    This shares no code with the fitted rollout and never calls JAX.
    """
    n, p, delay = model.norms, model.params, model.delay_steps
    x = (np.asarray(past_states) - n["state_mean"]) / n["state_scale"]
    up = (np.asarray(past_inputs) - n["input_mean"]) / n["input_scale"]
    uf = (np.asarray(future_inputs) - n["input_mean"]) / n["input_scale"]
    rows = len(x)

    def features(current, command, history, commands, hidden):
        return (
            np.concatenate(
                (
                    current,
                    command,
                    (history - current[:, None]).reshape(rows, -1),
                    (commands - command[:, None]).reshape(rows, -1),
                    hidden,
                ),
                axis=-1,
            )
            / n["feature_scale"]
        )

    hidden = np.zeros((rows, p["memory"].shape[1]))
    for j in range(delay, up.shape[1]):
        z = features(
            x[:, j], up[:, j], x[:, j - delay : j], up[:, j - delay : j], hidden
        )
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
    history, commands = x[:, -delay - 1 :], up[:, -delay:]
    output = []
    for t in range(uf.shape[1]):
        command = uf[:, t]
        current = history[:, -1]
        z = features(current, command, history[:, :-1], commands, hidden)
        delta = z @ p["linear"] + p["bias"] + np.tanh(z @ p["w1"] + p["b1"]) @ p["w2"]
        predicted = current + n["delta_scale"] * delta
        hidden = np.tanh(z @ p["memory"] + p["memory_bias"])
        output.append(predicted * n["state_scale"] + n["state_mean"])
        history = np.concatenate((history[:, 1:], predicted[:, None]), axis=1)
        commands = np.concatenate((commands[:, 1:], command[:, None]), axis=1)
    return np.stack(output, axis=1)


def paired_rmse(prediction, target):
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 2)))


# --- the decision -----------------------------------------------------------


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def expected_cases(manifest):
    return {
        (f, int(s)) for f in manifest["families"] for s in manifest["data_seeds"]
    } | {(WITNESS, int(s)) for s in manifest["witness"]["data_seeds"]}


def case_regimes(manifest, family):
    if family == WITNESS:
        return list(manifest["witness"]["regimes"])
    return list(manifest["evaluation_regimes"])


def decide(manifest, rows, reference=None):
    """Every gate, evaluated from recorded scores alone. Anything unclear fails."""
    expected = expected_cases(manifest)
    keys = [(r.get("family"), r.get("data_seed")) for r in rows]
    breaches = []
    for family, seed in sorted(expected - set(keys)):
        breaches.append(dict(case=f"{family}-{seed}", gate="case_present"))
    for family, seed in sorted({k for k in keys if keys.count(k) > 1}):
        breaches.append(dict(case=f"{family}-{seed}", gate="case_unique"))
    for family, seed in sorted(set(keys) - expected):
        breaches.append(dict(case=f"{family}-{seed}", gate="case_declared"))
    overall, probes = {}, {}
    for row in rows:
        name = row.get("name", f"{row.get('family')}-{row.get('data_seed')}")
        if row.get("status") != "complete":
            breaches.append(dict(case=name, gate="fit_complete"))
            continue
        if row.get("family") not in manifest["error_caps"]:
            breaches.append(dict(case=name, gate="family_declared"))
            continue
        declared = case_regimes(manifest, row["family"])
        if sorted(row.get("regimes", {})) != sorted(declared):
            breaches.append(dict(case=name, gate="regimes_declared"))
            continue
        for regime in declared:
            metrics = row["regimes"][regime]
            cap = manifest["error_caps"][row["family"]][regime]
            horizons = np.asarray(metrics["horizon_scaled_rmse"], dtype=float)
            if (
                horizons.ndim != 1
                or not horizons.size
                or not np.isfinite(horizons).all()
                or np.any(horizons < 0)
                or np.any(horizons > cap)
            ):
                breaches.append(
                    dict(
                        case=name,
                        regime=regime,
                        gate="horizon_scaled_rmse",
                        value=_finite(horizons.max()) if horizons.size else None,
                        limit=cap,
                    )
                )
            value = _finite(metrics["overall_scaled_rmse"])
            overall.setdefault(name, {})[regime] = value
            if value is None or value < 0:
                breaches.append(
                    dict(
                        case=name,
                        regime=regime,
                        gate="finite_overall_scaled_rmse",
                        value=value,
                    )
                )
        if row["family"] == WITNESS:
            paired = row.get("probe", {}).get("paired_rmse") or [float("nan")]
            first = _finite(paired[0])
            probes[name] = first
            limit = manifest["witness"]["paired_probe_first_step_rmse_maximum"]
            if first is None or not 0 <= first <= limit:
                breaches.append(
                    dict(
                        case=name,
                        gate="paired_probe_first_step_rmse",
                        value=first,
                        limit=limit,
                    )
                )
    regressions = []
    if reference is not None:
        relative = manifest["reference"]["relative_tolerance"]
        absolute = manifest["reference"]["absolute_tolerance"]
        recorded = reference.get("overall_scaled_rmse", {})
        for name in sorted(overall):
            for regime, value in sorted(overall[name].items()):
                base = recorded.get(name, {}).get(regime)
                if base is None or not np.isfinite(base):
                    regressions.append(
                        dict(case=name, regime=regime, gate="reference_present")
                    )
                    continue
                limit = float(base) * (1 + relative) + absolute
                if value is None or value > limit:
                    regressions.append(
                        dict(
                            case=name,
                            regime=regime,
                            gate="reference_overall_scaled_rmse",
                            value=value,
                            reference=float(base),
                            limit=limit,
                        )
                    )
    accepted = not breaches and not regressions
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        cases=len(rows),
        gate_breaches=breaches,
        reference_regressions=regressions,
        reference_compared=reference is not None,
        overall_scaled_rmse=overall,
        paired_probe_first_step_rmse=probes,
        meaning="A synthetic regression guard for this manifest only. Not platform readiness, control adequacy, or calibrated uncertainty.",
    )


# --- running ----------------------------------------------------------------


def _files(directory, names):
    return {name: sha256(directory / name) for name in sorted(names)}


def _evaluate(learned, supplied, plan, steps, scale, directory, regime):
    arrays = evaluation_rows(supplied, plan, steps)
    prediction = np.asarray(
        learned.predict(
            arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
        )
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"nonfinite forecast in {regime}")
    np.savez_compressed(
        directory / f"{regime}.npz", **arrays, prediction=prediction, state_scale=scale
    )
    return score(prediction, arrays["targets"], arrays["recording_ids"], scale)


def _case(manifest, family, seed, output):
    plan = manifest["dataset"]
    steps = steps_for(plan["dt_s"])
    name = f"{family}-{seed}"
    directory = output / name
    directory.mkdir()
    witness = family == WITNESS
    calibration = (
        witness_recordings(plan, seed)
        if witness
        else generate(plan, family, seed, "calibration")
    )
    started = time.perf_counter()
    learned = fit(calibration)
    wall = time.perf_counter() - started
    learned.save(directory / "model.npz")
    scale = np.asarray(learned._model.norms["state_scale"])
    fitted = set(learned.report["training"]) | set(learned.report["development"])
    regimes = {}
    for regime in case_regimes(manifest, family):
        supplied = (
            witness_recordings(plan, seed, evaluation=True)
            if witness
            else generate(plan, family, seed, regime)
        )
        if {s.recording_id for s in supplied.segments} & fitted:
            raise ValueError("evaluation recording identities overlap the fit data")
        regimes[regime] = _evaluate(
            learned, supplied, plan, steps, scale, directory, regime
        )
    names = ["model.npz", *(f"{r}.npz" for r in regimes)]
    row = dict(
        name=name,
        family=family,
        data_seed=seed,
        status="complete",
        model_fingerprint=learned.fingerprint(),
        fit_wall_seconds=wall,
        state_scale=scale.tolist(),
        report=learned.report,
        regimes=regimes,
    )
    if witness:
        px, pu, uf, target = witness_probe(steps["history"], steps["horizon"])
        prediction = np.asarray(learned.predict(px, pu, uf))
        if not np.isfinite(prediction).all():
            raise ValueError("nonfinite witness probe forecast")
        np.savez_compressed(
            directory / "probe.npz",
            past_states=px,
            past_inputs=pu,
            future_inputs=uf,
            targets=target,
            prediction=prediction,
        )
        names.append("probe.npz")
        row["probe"] = dict(
            paired_rmse=paired_rmse(prediction, target).tolist(),
            blind_floor=(np.abs(target[1, :, 0] - target[0, :, 0]) / 2).tolist(),
            branches_identical=bool(np.array_equal(prediction[0], prediction[1])),
            meaning="Paired RMSE over the two branches per horizon. Any forecast that cannot see the withheld command answers identically in both branches and cannot beat the blind floor.",
        )
    row["files"] = _files(directory, names)
    write(directory / "result.json", row)
    return row


def run(manifest_path, output):
    """Fit every frozen case, score it, and write the decision and artifacts."""
    import jax

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = frozen_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
    reference_path = manifest_path.parent / manifest["reference"]["file"]
    reference = None
    if reference_path.exists():
        shutil.copyfile(reference_path, output / "reference.json")
        reference = read(output / "reference.json")
    write(
        output / "environment.json",
        dict(
            python=sys.version,
            platform=platform_module.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=True,
        ),
    )
    rows = []
    started = time.perf_counter()
    with jax.enable_x64(True):
        for family in [*manifest["families"], WITNESS]:
            seeds = (
                manifest["witness"]["data_seeds"]
                if family == WITNESS
                else manifest["data_seeds"]
            )
            for seed in seeds:
                print(json.dumps(dict(starting=f"{family}-{seed}")), flush=True)
                rows.append(_case(manifest, family, seed, output))
                write(output / "results.json", rows)
    decision = decide(manifest, rows, reference)
    decision["wall_seconds"] = time.perf_counter() - started
    write(output / "decision.json", decision)
    return decision


# --- verifying --------------------------------------------------------------


def _check_probe(data, steps):
    """The probe is the one declared construction, not whatever was saved."""
    px, pu, uf, target = witness_probe(steps["history"], steps["horizon"])
    for key, expected in (
        ("past_states", px),
        ("past_inputs", pu),
        ("future_inputs", uf),
        ("targets", target),
    ):
        np.testing.assert_allclose(data[key], expected, rtol=0, atol=1e-12)
    if (
        not np.array_equal(px[0], px[1])
        or not np.array_equal(uf[0], uf[1])
        or np.count_nonzero(pu[0] != pu[1]) != 1
    ):
        raise ValueError("probe contexts must differ in exactly one input")


def verify(directory):
    """Recompute every score and the decision from saved arrays; never fit."""
    directory = Path(directory)
    manifest = frozen_manifest(directory / "manifest.json")
    plan = manifest["dataset"]
    steps = steps_for(plan["dt_s"])
    rows = read(directory / "results.json")
    replays, worst = 0, 0.0
    for row in rows:
        case = directory / row["name"]
        if read(case / "result.json") != row:
            raise ValueError(f"case result mismatch: {row['name']}")
        for name, digest in row["files"].items():
            if sha256(case / name) != digest:
                raise ValueError(f"altered artifact: {row['name']}/{name}")
        learned = LearnedDynamics.load(case / "model.npz")
        if learned.fingerprint() != row["model_fingerprint"]:
            raise ValueError(f"model fingerprint mismatch: {row['name']}")
        model = learned._model
        scale = np.asarray(row["state_scale"])
        np.testing.assert_array_equal(scale, model.norms["state_scale"])
        for regime in case_regimes(manifest, row["family"]):
            with np.load(case / f"{regime}.npz", allow_pickle=False) as data:
                if data["past_states"].shape[1] != steps["history"] + 1:
                    raise ValueError("saved evaluation rows lack the consumed context")
                prediction = replay(
                    model,
                    data["past_states"],
                    data["past_inputs"],
                    data["future_inputs"],
                )
                worst = max(
                    worst, float(np.max(np.abs(prediction - data["prediction"])))
                )
                np.testing.assert_allclose(
                    prediction, data["prediction"], **REPLAY_TOLERANCE
                )
                fresh = score(
                    prediction,
                    data["targets"],
                    data["recording_ids"],
                    data["state_scale"],
                )
            saved = row["regimes"][regime]
            if sorted(fresh) != sorted(saved) or fresh["windows"] != saved["windows"]:
                raise ValueError(f"score shape mismatch: {row['name']}/{regime}")
            for key in ("channel_rmse", "horizon_scaled_rmse", "overall_scaled_rmse"):
                np.testing.assert_allclose(fresh[key], saved[key], **SCORE_TOLERANCE)
            for name, metrics in fresh["recordings"].items():
                for key, value in metrics.items():
                    np.testing.assert_allclose(
                        value, saved["recordings"][name][key], **SCORE_TOLERANCE
                    )
            row["regimes"][regime] = fresh
            replays += 1
        if row["family"] == WITNESS:
            with np.load(case / "probe.npz", allow_pickle=False) as data:
                _check_probe(data, steps)
                prediction = replay(
                    model,
                    data["past_states"],
                    data["past_inputs"],
                    data["future_inputs"],
                )
                worst = max(
                    worst, float(np.max(np.abs(prediction - data["prediction"])))
                )
                np.testing.assert_allclose(
                    prediction, data["prediction"], **REPLAY_TOLERANCE
                )
                paired = paired_rmse(prediction, data["targets"])
            np.testing.assert_allclose(
                paired, row["probe"]["paired_rmse"], **SCORE_TOLERANCE
            )
            row["probe"]["paired_rmse"] = paired.tolist()
            replays += 1
    reference_path = directory / "reference.json"
    reference = read(reference_path) if reference_path.exists() else None
    decision = decide(manifest, rows, reference)
    saved = read(directory / "decision.json")
    for key in ("manifest", "decision", "accepted", "cases", "reference_compared"):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed decision differs: {key}")
    for key in ("gate_breaches", "reference_regressions"):
        if len(decision[key]) != len(saved[key]):
            raise ValueError(f"replayed decision differs: {key}")
    return dict(
        verified_cases=len(rows),
        replays=replays,
        maximum_replay_difference=worst,
        decision=decision,
        meaning="A replay of saved evidence with an independent NumPy recurrence, not a new fit or independent confirmation.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="fit every frozen case and decide")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    checker = commands.add_parser("verify", help="replay a run directory")
    checker.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.manifest, args.output)
        accepted = result["accepted"]
    else:
        result = verify(args.directory)
        accepted = result["decision"]["accepted"]
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
