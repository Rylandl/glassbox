"""One harness for the generic learner: fit frozen cases, score, decide, replay.

The harness has four tiers, each with its own frozen manifest and digest
constant. ``run`` is the synthetic tier: it fits the consumer recipe end to end
on each case of ``docs/harness/v1.json``, scores its forecasts on independent
recordings, and writes a decision. ``platform`` is the accuracy tier: it fits
the same recipe on each pinned corpus of ``docs/harness/platform-v3.json`` with
whole recordings held out, fits the structured model on exactly the same
training recordings, and scores both on exactly the same held-out rows.
``control`` is the control tier: it collects the frozen Cascade X8 calibration
of ``docs/harness/control-v5.json``, declares that calibration's own additive
command excitation to the generic learner as part of its recordings, fits both
models on it, and tracks the same reference with each of them through the
existing NMPC seam. ``live`` is the live improvement tier: it collects the same
calibration under ``docs/harness/live-v3.json``, flies the structured arm from
the first interval under a declared command dither, refits the generic recipe
on the trial's own streamed transitions -- which carry that dither as their
declared excitation -- through the existing transition buffer and refinement
worker, and hands it the controller when a predeclared held-out block gate
passes.
``verify`` replays any tier from its saved artifacts and rejects anything
that changed. No command takes tuning options.

    python -m glassbox.experimental.harness run \\
        --manifest docs/harness/v1.json --output DIR
    python -m glassbox.experimental.harness platform \\
        --manifest docs/harness/platform-v3.json --corpora ROOT --output DIR
    python -m glassbox.experimental.harness control \\
        --manifest docs/harness/control-v5.json --output DIR
    python -m glassbox.experimental.harness live \\
        --manifest docs/harness/live-v3.json --output DIR
    python -m glassbox.experimental.harness verify DIR [--reference PATH]

A run copies the manifest and the reference it compared into its output, but a
replay never takes a saved copy's word for a threshold: it anchors both to the
committed files, so a run cannot relax its own gates after the fact.

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
from collections import deque
from fnmatch import fnmatch
from itertools import pairwise
from pathlib import Path

import numpy as np

from .default_model import (
    RECIPE,
    LearnedDynamics,
    excitation_fraction,
    fit,
    steps_for,
)
from .sequence_collection import (
    SequenceCollection,
    SequenceSegment,
    segments_from_mask,
)

MANIFEST_SHA256 = "1ba15b3f466e91edf548f1d459a52eb0896310e7fbbb6c1c98544ec10513d781"
"""Digest of the frozen manifest this module is allowed to run and verify."""

COMMITTED_MANIFEST = Path(__file__).resolve().parents[3] / "docs/harness/v1.json"
"""The frozen manifest in a source checkout, and the anchor for the reference.

A run copies the manifest and the reference it compared into its output, but a
copy is an artifact like any other: ``verify`` anchors both to the committed
files rather than believing the copies. Outside a checkout these paths do not
exist, and ``verify`` then needs ``--reference`` to say where the committed
reference is; it refuses rather than trusting the copy.
"""

COMMITTED_REFERENCE = COMMITTED_MANIFEST.parent / "reference.json"
"""Where ``verify`` looks for the committed synthetic reference by default."""

PLAN_CONSTANTS = ("context_s", "delay_s", "horizon_s")
"""The recipe constants a manifest's evaluation plan is cut from.

A frozen manifest freezes the evaluation, not the candidate: pinning the whole
recipe would make it impossible to measure a changed recipe against a frozen
gate, which is the one thing the gate exists for. What a manifest does pin is
the information budget every window is cut to, so a recipe that reads a
different context, explicit delay or horizon cannot be scored on a plan built
for another one. Each manifest also records the recipe it was frozen against,
in full, as provenance, and every run records the recipe it actually fitted.
"""


def declared_plan(manifest):
    """Refuse a manifest whose evaluation plan differs from the recipe's."""
    declared = manifest["recipe"]
    if any(declared.get(name) != RECIPE[name] for name in PLAN_CONSTANTS):
        raise ValueError("manifest evaluation plan differs from the maintained recipe")
    return declared


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
    declared_plan(manifest)
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


def decide(manifest, rows, reference=None, reference_sha256=None):
    """Every gate, evaluated from recorded scores alone. Anything unclear fails.

    ``reference_sha256`` identifies the reference file the scores were compared
    against. It is recorded in the decision so a replay can check that it
    compared the same bytes, and it is not otherwise used here.
    """
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
        reference_sha256=reference_sha256,
        overall_scaled_rmse=overall,
        paired_probe_first_step_rmse=probes,
        meaning="A synthetic regression guard for this manifest only. Not platform readiness, control adequacy, or calibrated uncertainty.",
    )


# --- running ----------------------------------------------------------------


def _files(directory, names):
    return {name: sha256(directory / name) for name in sorted(names)}


def envelope_rows(learned, prediction):
    """The envelope this forecast carries, one half-width beside every predicted row.

    ``predict`` returns a ``[row, horizon, channel]`` block and ``envelope``
    returns the ``[horizon, channel]`` half-widths that go with it, the same for
    every row because the calibration is a quantile over the development
    windows rather than a function of the origin. The run saves the full block
    anyway, so a replay reads coverage out of arrays that stand on their own.
    """
    prediction = np.asarray(prediction, dtype=float)
    half = np.asarray(learned.envelope(prediction.shape[1]), dtype=float)
    return np.broadcast_to(half, prediction.shape).copy()


def _evaluate(learned, supplied, plan, steps, scale, directory, regime, groups):
    arrays = evaluation_rows(supplied, plan, steps)
    prediction = np.asarray(
        learned.predict(
            arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
        )
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"nonfinite forecast in {regime}")
    half_width = envelope_rows(learned, prediction)
    np.savez_compressed(
        directory / f"{regime}.npz",
        **arrays,
        prediction=prediction,
        envelope_half_width=half_width,
        state_scale=scale,
    )
    return (
        score(prediction, arrays["targets"], arrays["recording_ids"], scale),
        envelope_coverage(prediction, arrays["targets"], half_width, groups),
    )


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
    groups = evidence_groups("per_channel", len(scale))
    fitted = set(learned.report["training"]) | set(learned.report["development"])
    regimes, evidence = {}, {}
    for regime in case_regimes(manifest, family):
        supplied = (
            witness_recordings(plan, seed, evaluation=True)
            if witness
            else generate(plan, family, seed, regime)
        )
        if {s.recording_id for s in supplied.segments} & fitted:
            raise ValueError("evaluation recording identities overlap the fit data")
        regimes[regime], evidence[regime] = _evaluate(
            learned, supplied, plan, steps, scale, directory, regime, groups
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
        evidence=evidence,
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
    evidence_manifest, evidence_reference, evidence_digest = evidence_setup(output)
    reference_path = manifest_path.parent / manifest["reference"]["file"]
    reference, reference_digest = None, None
    if reference_path.exists():
        shutil.copyfile(reference_path, output / "reference.json")
        reference, reference_digest = read(reference_path), sha256(reference_path)
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
    decision = decide(manifest, rows, reference, reference_digest)
    with_evidence(
        decision,
        evidence_decide(
            evidence_manifest,
            "synthetic",
            evidence_table(rows, "name"),
            synthetic_evidence_cases(manifest),
            evidence_reference,
            evidence_digest,
        ),
    )
    decision["wall_seconds"] = time.perf_counter() - started
    write(output / "decision.json", decision)
    return decision


def synthetic_evidence_cases(manifest):
    """The case and regime pairs the synthetic manifest declares coverage for."""
    return {
        (f"{family}-{seed}", regime)
        for family, seed in expected_cases(manifest)
        for regime in case_regimes(manifest, family)
    }


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


def anchored_reference(
    directory, reference=None, committed=COMMITTED_REFERENCE, name="reference.json"
):
    """The committed reference a replay must compare against, never the copy.

    A run copies the reference it compared into its output. That copy is an
    artifact: loosening it would loosen the regression gate on replay. So the
    copy is only ever checked against the committed file, and the committed
    file is what the replay reads. The two must agree about existing and about
    every byte, or the replay refuses. ``committed`` is the tier's own frozen
    reference; ``reference`` overrides it when the replay does not run inside a
    checkout. ``name`` is what the run called its copy, which differs only for
    the evidence reference, whose copy sits beside the tier's own.
    """
    committed = Path(reference) if reference is not None else Path(committed)
    copied = Path(directory) / name
    if copied.exists() != committed.exists():
        present, absent = (
            (copied, committed) if copied.exists() else (committed, copied)
        )
        raise ValueError(
            f"reference mismatch: {present} exists but {absent} does not, so the "
            "run's regression gate cannot be anchored"
        )
    if not committed.exists():
        return None, None
    digest = sha256(committed)
    if sha256(copied) != digest:
        raise ValueError(
            f"the reference copied into the run differs from {committed}; a saved "
            "run cannot relax its own regression gate"
        )
    return read(committed), digest


def verify(directory, reference=None):
    """Recompute every score and the decision from saved arrays; never fit.

    Which tier a directory holds is decided by the digest of the manifest it
    copied, not by anything the copy says about itself. A manifest that matches
    neither frozen contract is refused before anything is read.
    """
    directory = Path(directory)
    digest = sha256(directory / "manifest.json")
    if digest == PLATFORM_MANIFEST_SHA256:
        return verify_platform(
            directory,
            frozen_platform_manifest(directory / "manifest.json"),
            reference,
        )
    if digest == CONTROL_MANIFEST_SHA256:
        return verify_control(
            directory,
            frozen_control_manifest(directory / "manifest.json"),
            reference,
        )
    if digest == LIVE_MANIFEST_SHA256:
        return verify_live(
            directory,
            frozen_live_manifest(directory / "manifest.json"),
            reference,
        )
    if digest != MANIFEST_SHA256:
        raise ValueError("manifest digest differs from every frozen harness contract")
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
        groups = evidence_groups("per_channel", len(scale))
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
                coverage = _replayed_coverage(
                    learned, prediction, data, groups, f"{row['name']}/{regime}"
                )
            _same_coverage(coverage, row["evidence"][regime], f"{row['name']}/{regime}")
            row["evidence"][regime] = coverage
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
    anchor, digest = anchored_reference(directory, reference)
    decision = decide(manifest, rows, anchor, digest)
    evidence_manifest, evidence_anchored, evidence_digest = evidence_anchor(directory)
    evidence = evidence_decide(
        evidence_manifest,
        "synthetic",
        evidence_table(rows, "name"),
        synthetic_evidence_cases(manifest),
        evidence_anchored,
        evidence_digest,
    )
    saved = read(directory / "decision.json")
    _same_evidence(evidence, saved.get("evidence"), "synthetic")
    with_evidence(decision, evidence)
    for key in (
        "manifest",
        "decision",
        "accepted",
        "cases",
        "reference_compared",
        "reference_sha256",
    ):
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


# --- the evidence tier: one envelope, one band ------------------------------

EVIDENCE_MANIFEST_SHA256 = (
    "c669b2ccfd6a11272f32839120fb1a33bdf3ae4bbc6120f444467c784d4dbe29"
)
"""Digest of the frozen evidence manifest every tier measures coverage under."""

COMMITTED_EVIDENCE_MANIFEST = COMMITTED_MANIFEST.parent / "evidence-v2.json"
"""The frozen evidence manifest in a source checkout.

Unlike the three tier manifests this one is not a command-line argument. It is
not a tier of its own: coverage is measured inside the synthetic, platform and
control runs, on exactly the rows those tiers already score, so each run reads
this file from the checkout, copies it into its output as
``evidence-manifest.json``, and ``verify`` checks that copy against the digest
constant above rather than against the file. A copy whose bytes differ is
refused, which is the same authority the tier manifests are held to.
"""

COMMITTED_EVIDENCE_REFERENCE = COMMITTED_MANIFEST.parent / "evidence-reference.json"
"""Where ``verify`` looks for the committed evidence reference by default."""

EVIDENCE_MANIFEST_NAME = "evidence-manifest.json"
EVIDENCE_REFERENCE_NAME = "evidence-reference.json"

RIGID_BODY_GROUPS = {
    "world_velocity": (0, 1, 2),
    "body_rate": (3, 4, 5),
    "rotation_entries": (6, 7, 8, 9, 10, 11, 12, 13, 14),
}
"""The declared channel groups of the fifteen-channel observed contract."""


def frozen_evidence_manifest(path):
    """Load the evidence manifest only when its bytes match the frozen digest."""
    if sha256(path) != EVIDENCE_MANIFEST_SHA256:
        raise ValueError("evidence manifest digest differs from the frozen contract")
    manifest = read(path)
    declared_plan(manifest)
    band = manifest["band"]
    if not 0.0 < band["minimum"] <= band["maximum"] < 1.0:
        raise ValueError("the declared coverage band is not an interval in (0, 1)")
    return manifest


def evidence_groups(kind, channels):
    """The declared channel groups for one observed contract.

    ``rigid_body_15`` is the fifteen-channel contract both the platform and the
    control tier build, grouped as the manifest declares. ``per_channel`` is the
    synthetic families, whose channels are coordinates of unrelated systems and
    are each their own group.
    """
    if kind == "rigid_body_15":
        if channels != 15:
            raise ValueError("the rigid-body contract has fifteen observed channels")
        return {name: tuple(index) for name, index in RIGID_BODY_GROUPS.items()}
    if kind == "per_channel":
        if channels < 1:
            raise ValueError("an observed contract needs at least one channel")
        return {f"channel_{index}": (index,) for index in range(channels)}
    raise ValueError(f"undeclared channel grouping: {kind}")


def envelope_coverage(prediction, targets, half_widths, groups):
    """Measured coverage per channel group and horizon step, and pooled.

    One scored ``(row, channel)`` pair is covered when its absolute forecast
    error at that horizon step is at or below the envelope half-width for that
    step and channel. Nothing here knows how the half-widths were calibrated;
    it reads the arrays the run saved beside the predictions.
    """
    error = np.abs(
        np.asarray(prediction, dtype=float) - np.asarray(targets, dtype=float)
    )
    width = np.asarray(half_widths, dtype=float)
    if error.ndim != 3 or width.shape != error.shape:
        raise ValueError("envelope half-widths must be shaped like the predictions")
    if not np.isfinite(width).all() or np.any(width < 0):
        raise ValueError("envelope half-widths must be finite and nonnegative")
    covered = error <= width
    horizons, pooled = {}, {}
    for name, index in groups.items():
        block = covered[:, :, list(index)]
        horizons[name] = np.mean(block, axis=(0, 2)).tolist()
        pooled[name] = float(np.mean(block))
    return dict(
        scored_rows=int(error.shape[0]),
        horizon_steps=int(error.shape[1]),
        channels=int(error.shape[2]),
        coverage=horizons,
        pooled_coverage=pooled,
    )


def evidence_table(rows, key, regime=None):
    """One tier's recorded coverage, as the rows the evidence decision reads.

    ``key`` is what that tier calls a case. A row that recorded no coverage at
    all contributes nothing and is missing from the table, which is exactly how
    the decision is told that a case was not measured.
    """
    table = []
    for row in rows:
        measured = row.get("evidence")
        if not isinstance(measured, dict):
            continue
        if regime is not None:
            table.append(dict(case=row.get(key), regime=regime, **measured))
            continue
        for name in sorted(measured):
            if isinstance(measured[name], dict):
                table.append(dict(case=row.get(key), regime=name, **measured[name]))
    return table


def evidence_setup(output):
    """Freeze the evidence contract into one run and read the reference it compares.

    Every tier does this the same way: the committed manifest is checked
    against the digest constant and copied into the run, and the committed
    reference, when one exists, is copied beside it. A checkout that does not
    hold the evidence manifest cannot measure the evidence it declares, so the
    run refuses rather than skipping it.
    """
    if not COMMITTED_EVIDENCE_MANIFEST.exists():
        raise ValueError(
            f"the frozen evidence manifest is not at {COMMITTED_EVIDENCE_MANIFEST}; "
            "a run cannot measure coverage against a contract it cannot read"
        )
    manifest = frozen_evidence_manifest(COMMITTED_EVIDENCE_MANIFEST)
    shutil.copyfile(COMMITTED_EVIDENCE_MANIFEST, Path(output) / EVIDENCE_MANIFEST_NAME)
    reference, digest = None, None
    if COMMITTED_EVIDENCE_REFERENCE.exists():
        shutil.copyfile(
            COMMITTED_EVIDENCE_REFERENCE, Path(output) / EVIDENCE_REFERENCE_NAME
        )
        reference = read(COMMITTED_EVIDENCE_REFERENCE)
        digest = sha256(COMMITTED_EVIDENCE_REFERENCE)
    return manifest, reference, digest


def evidence_anchor(directory):
    """The frozen evidence contract and reference a replay must decide under.

    The manifest is anchored to the digest constant in this module rather than
    to a file, which is how the three tier manifests are anchored too. The
    reference is anchored to the committed file the way every other reference
    is.
    """
    manifest = frozen_evidence_manifest(Path(directory) / EVIDENCE_MANIFEST_NAME)
    reference, digest = anchored_reference(
        directory, None, COMMITTED_EVIDENCE_REFERENCE, EVIDENCE_REFERENCE_NAME
    )
    return manifest, reference, digest


def with_evidence(decision, evidence):
    """Fold one tier's evidence decision into the decision it is measured beside.

    Unenforced, the evidence decision always accepts, so this records the whole
    coverage table and changes nothing. Enforced, a rejected band rejects the
    run that measured it.
    """
    decision["evidence"] = evidence
    if not evidence["accepted"]:
        decision["accepted"] = False
        decision["decision"] = "reject"
    return decision


def _replayed_coverage(learned, prediction, data, groups, label):
    """Rebuild the saved envelope from the model and remeasure its coverage.

    A run saves the half-widths beside its predictions, but a replay never
    takes that array's word for anything: it asks the saved model artifact for
    its own envelope and refuses an array that is not it, byte for byte, then
    recomputes coverage from the replayed prediction and the saved targets.
    """
    saved = data["envelope_half_width"]
    fresh = envelope_rows(learned, prediction)
    if saved.shape != fresh.shape or not np.array_equal(saved, fresh):
        raise ValueError(f"saved envelope is not the model's own: {label}")
    return envelope_coverage(prediction, data["targets"], fresh, groups)


def _same_coverage(fresh, saved, label):
    """A replayed coverage table must match the one the run recorded."""
    if not isinstance(saved, dict) or sorted(fresh) != sorted(saved):
        raise ValueError(f"coverage shape mismatch: {label}")
    for key in ("scored_rows", "horizon_steps", "channels"):
        if fresh[key] != saved[key]:
            raise ValueError(f"coverage shape mismatch: {label}/{key}")
    for key in ("coverage", "pooled_coverage"):
        if sorted(fresh[key]) != sorted(saved[key]):
            raise ValueError(f"coverage group mismatch: {label}/{key}")
        for group, value in fresh[key].items():
            np.testing.assert_allclose(value, saved[key][group], **SCORE_TOLERANCE)


def _same_evidence(fresh, saved, label):
    """A replayed evidence decision must match the one the run wrote."""
    if saved is None:
        raise ValueError(f"{label} recorded no evidence decision")
    for key in (
        "manifest",
        "tier",
        "decision",
        "accepted",
        "band_met",
        "band_enforced",
        "gating_band_breaches",
        "cases",
        "reference_compared",
        "reference_sha256",
    ):
        if fresh[key] != saved.get(key):
            raise ValueError(f"replayed evidence decision differs: {label}/{key}")
    for key in ("gate_breaches", "band_breaches", "reference_regressions"):
        if len(fresh[key]) != len(saved.get(key, [])):
            raise ValueError(f"replayed evidence decision differs: {label}/{key}")


def _band_excess(coverage, band):
    """How far one measured coverage lies outside the declared band.

    Zero inside it. This is the smaller-is-better quantity the reference
    allowance is stated on, exactly as the other tiers state theirs on an RMSE.
    """
    if coverage is None:
        return None
    return max(0.0, band["minimum"] - coverage, coverage - band["maximum"])


def _coverage_number(value):
    """A coverage that is a finite number in ``[0, 1]``, or None for anything else."""
    number = _number(value)
    return None if number is None or not 0.0 <= number <= 1.0 else number


def _evidence_reference(reference, tier, case, regime, group):
    """One recorded reference coverage list, or None for anything that is not one."""
    node = None if reference is None else reference.get("coverage")
    for key in (tier, case, regime, group):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, list) else None


def evidence_decide(
    manifest, tier, rows, expected, reference=None, reference_sha256=None
):
    """The band, evaluated from recorded coverage alone. Anything unclear fails.

    ``expected`` is the ``(case, regime)`` set the tier's own manifest declares,
    and its case names must be exactly the ones this manifest froze; a tier that
    measures a different set of corpora than the evidence manifest declares
    fails closed here rather than reporting a shorter table.

    The semantics are the platform and control tiers': a case is a gate only
    where the reference already meets the band, a case the reference already
    fails is reported with ``gating`` false, and structural problems always fail
    closed. ``enforced`` decides only whether the result is folded into the
    tier's own acceptance; every number and every breach is computed either way.
    """
    band = manifest["band"]
    declared_cases = set(manifest["tiers"][tier]["cases"])
    grouping = manifest["tiers"][tier]["channel_groups"]
    relative = manifest["reference"]["relative_tolerance"]
    absolute = manifest["reference"]["absolute_tolerance"]
    expected = {(str(case), str(regime)) for case, regime in expected}
    breaches, band_breaches, regressions, summary = [], [], [], {}
    if {case for case, _ in expected} != declared_cases:
        breaches.append(
            dict(
                gate="cases_declared",
                measured=sorted({case for case, _ in expected}),
                declared=sorted(declared_cases),
            )
        )
    keys = [(str(row.get("case")), str(row.get("regime"))) for row in rows]
    for case, regime in sorted(expected - set(keys)):
        breaches.append(dict(case=case, regime=regime, gate="case_present"))
    for case, regime in sorted({key for key in keys if keys.count(key) > 1}):
        breaches.append(dict(case=case, regime=regime, gate="case_unique"))
    for case, regime in sorted(set(keys) - expected):
        breaches.append(dict(case=case, regime=regime, gate="case_declared"))
    for row, key in zip(rows, keys, strict=True):
        case, regime = key
        if key not in expected or keys.count(key) > 1:
            continue
        name = f"{case}/{regime}"
        scored = row.get("scored_rows")
        if not isinstance(scored, int) or isinstance(scored, bool) or scored < 1:
            breaches.append(dict(case=case, regime=regime, gate="scored_rows"))
            continue
        steps = row.get("horizon_steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            breaches.append(dict(case=case, regime=regime, gate="horizon_steps"))
            continue
        try:
            groups = evidence_groups(grouping, row.get("channels"))
        except (TypeError, ValueError):
            breaches.append(dict(case=case, regime=regime, gate="channels_declared"))
            continue
        measured = row.get("coverage")
        if not isinstance(measured, dict) or sorted(measured) != sorted(groups):
            breaches.append(dict(case=case, regime=regime, gate="groups_declared"))
            continue
        for group in sorted(groups):
            values = measured[group]
            if not isinstance(values, list) or len(values) != steps:
                breaches.append(
                    dict(case=case, regime=regime, group=group, gate="group_horizons")
                )
                continue
            recorded = _evidence_reference(reference, tier, case, regime, group)
            for horizon, value in enumerate(values):
                coverage = _coverage_number(value)
                base = (
                    None
                    if recorded is None or len(recorded) != steps
                    else _coverage_number(recorded[horizon])
                )
                excess = _band_excess(coverage, band)
                reference_excess = _band_excess(base, band)
                meets = None if base is None else reference_excess <= 0.0
                summary.setdefault(name, {}).setdefault(group, []).append(
                    dict(
                        horizon=horizon,
                        coverage=coverage,
                        band_excess=excess,
                        reference=base,
                        reference_band_excess=reference_excess,
                        reference_meets_band=meets,
                    )
                )
                if coverage is None:
                    breaches.append(
                        dict(
                            case=case,
                            regime=regime,
                            group=group,
                            horizon=horizon,
                            gate="finite_coverage",
                            value=_number(value),
                        )
                    )
                    continue
                if reference is not None:
                    if reference_excess is None:
                        regressions.append(
                            dict(
                                case=case,
                                regime=regime,
                                group=group,
                                horizon=horizon,
                                gate="reference_present",
                            )
                        )
                    else:
                        limit = reference_excess * (1 + relative) + absolute
                        if excess > limit:
                            regressions.append(
                                dict(
                                    case=case,
                                    regime=regime,
                                    group=group,
                                    horizon=horizon,
                                    gate="reference_band_excess",
                                    value=excess,
                                    coverage=coverage,
                                    reference=reference_excess,
                                    limit=limit,
                                )
                            )
                if excess > 0.0:
                    band_breaches.append(
                        dict(
                            case=case,
                            regime=regime,
                            group=group,
                            horizon=horizon,
                            gate="coverage_band",
                            value=coverage,
                            minimum=band["minimum"],
                            maximum=band["maximum"],
                            below=coverage < band["minimum"],
                            gating=meets is True,
                        )
                    )
    regressions.sort(key=lambda entry: (entry.get("case", ""), repr(sorted(entry))))
    enforced = bool(manifest["decision"]["enforced"])
    gating = [breach for breach in band_breaches if breach["gating"]]
    band_met = not breaches and not band_breaches
    # Unenforced, every number and every breach above is still measured,
    # recorded and printed; what the manifest withholds until the next
    # candidate is only the power to reject a run.
    accepted = not enforced or not (breaches or regressions or gating)
    return dict(
        manifest=manifest["id"],
        tier=tier,
        decision="accept" if accepted else "reject",
        accepted=accepted,
        band_met=band_met,
        band_enforced=enforced,
        band=dict(minimum=band["minimum"], maximum=band["maximum"]),
        nominal_coverage=manifest["envelope"]["nominal_coverage"],
        gating_band_breaches=len(gating),
        cases=len(rows),
        gate_breaches=breaches,
        band_breaches=band_breaches,
        reference_regressions=regressions,
        reference_compared=reference is not None,
        reference_sha256=reference_sha256,
        coverage=summary,
        gates_from=manifest["decision"]["gates_from"],
        meaning=manifest["decision"]["meaning"],
    )


# --- the platform tier: the corpus adapter ----------------------------------

PLATFORM_MANIFEST_SHA256 = (
    "74722a9f23eeb3eeacc2d5faefab5bd2c126d60927525eceb306494a6cbf1d79"
)
"""Digest of the frozen platform manifest this module is allowed to run."""

COMMITTED_PLATFORM_MANIFEST = COMMITTED_MANIFEST.parent / "platform-v3.json"
"""The frozen platform manifest in a source checkout."""

COMMITTED_PLATFORM_REFERENCE = COMMITTED_MANIFEST.parent / "platform-reference.json"
"""Where ``verify`` looks for the committed platform reference by default."""

VELOCITY_ROWS = slice(3, 6)
QUATERNION_ROWS = slice(6, 10)
BODY_RATE_ROWS = slice(10, 13)
"""Where ``rigid_body_13_nwu_flu_wxyz_v1`` keeps each modeled state group.

Position, rows 0 to 2, is not modeled: the recipe forecasts rates and attitude,
and a position channel would only integrate them.
"""

OBSERVED_CHANNELS = (
    "velocity_north [m/s,world_nwu]",
    "velocity_west [m/s,world_nwu]",
    "velocity_up [m/s,world_nwu]",
    "body_rate_x [rad/s,body_flu]",
    "body_rate_y [rad/s,body_flu]",
    "body_rate_z [rad/s,body_flu]",
) + tuple(
    f"rotation_{row}{column} [unitless,body_flu_to_world_nwu]"
    for row in range(3)
    for column in range(3)
)
"""The one observed contract every corpus adapts to, in this order."""

VELOCITY_CHANNELS = (0, 1, 2)
BODY_RATE_CHANNELS = (3, 4, 5)
METRICS = ("velocity_rmse_m_s", "body_rate_rmse_rad_s")
PLATFORM_EVIDENCE_REGIME = "held_out"
"""What this tier calls the rows the evidence manifest measures coverage on."""


def frozen_platform_manifest(path):
    """Load the platform manifest only when its bytes match the frozen digest."""
    if sha256(path) != PLATFORM_MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen harness contract")
    manifest = read(path)
    declared_plan(manifest)
    for entry in manifest["corpora"]:
        steps = steps_for(entry["sample_interval_s"])
        budget = entry["information_budget"]
        if (steps["history"], steps["delay"], steps["horizon"]) != (
            budget["context_steps"],
            budget["delay_steps"],
            budget["horizon_steps"],
        ):
            raise ValueError("manifest information budget differs from the recipe")
    return manifest


def observed_from_states(states):
    """The 15 observed channels of a canonical rigid-body state array.

    World-frame velocity, body rates, then the nine body-to-world rotation
    entries in row-major order. Trailing axes only, so one state, a rollout, or
    a batch of rollouts all map the same way.
    """
    from glassbox.core.geometry import quaternion_to_rotation_matrices

    states = np.asarray(states, dtype=float)
    rotation = quaternion_to_rotation_matrices(states[..., QUATERNION_ROWS])
    return np.concatenate(
        (
            states[..., VELOCITY_ROWS],
            states[..., BODY_RATE_ROWS],
            rotation.reshape(*states.shape[:-1], 9),
        ),
        axis=-1,
    )


def observed_rows(trajectory):
    """One canonical trajectory's observed channels, checked against its spec."""
    from glassbox.core.data import RIGID_BODY_STATE_SCHEMA

    schema = trajectory.spec.state_schema
    if schema != RIGID_BODY_STATE_SCHEMA:
        raise ValueError(f"unsupported canonical state schema: {schema}")
    if trajectory.states.shape[1] != 13:
        raise ValueError("a canonical rigid-body state has thirteen rows")
    return observed_from_states(trajectory.states)


def command_channels(spec):
    """Ordered input identities from the trajectory's own spec, with units."""
    return tuple(
        f"{channel.name} [{channel.unit},{channel.role}]" for channel in spec.controls
    )


def trajectory_segments(
    recording_id, trajectory, *, dt_s, tolerance_fraction, excitation=None
):
    """One canonical trajectory as generic segments, on one uniform time grid.

    A sample interval that departs from ``dt_s`` ends a segment, and so does a
    nonfinite observed row or outgoing command. Nothing is padded or imputed:
    the split runs through ``segments_from_mask`` one uniformly sampled block at
    a time, so every retained row keeps its source row index. ``dt_s`` is the
    corpus's declared sample period; every interval is checked against it here,
    and it is what the whole corpus shares so one collection can hold it.

    ``excitation``, when the caller declares one, is the exogenous component it
    injected into each applied command, aligned with the trajectory's controls
    and cut into segments with them.
    """
    rows = observed_rows(trajectory)
    inputs = np.asarray(trajectory.controls, dtype=float)
    time_s = np.asarray(trajectory.time_s, dtype=float)
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("the declared sample interval must be finite and positive")
    if excitation is not None:
        excitation = np.asarray(excitation, dtype=float)
        if excitation.shape != inputs.shape:
            raise ValueError(
                f"the declared excitation of {recording_id} is not aligned with its commands"
            )
    uniform = np.abs(np.diff(time_s) - dt_s) <= tolerance_fraction * dt_s
    valid = np.isfinite(rows).all(axis=1)
    valid[:-1] &= np.isfinite(inputs).all(axis=1)
    bounds = np.r_[0, np.flatnonzero(~uniform) + 1, len(time_s)]
    segments = []
    for start, stop in pairwise(bounds):
        block = np.zeros(len(time_s), dtype=bool)
        block[start:stop] = True
        segments.extend(
            segments_from_mask(
                recording_id,
                rows,
                inputs,
                valid & block,
                dt_s=dt_s,
                excitation=excitation,
            )
        )
    return tuple(segments)


def corpus_paths(entry, root):
    """One corpus's recordings on disk, split by the declared held-out patterns.

    The root is a command-line argument, never a manifest fact. What the
    manifest freezes is the corpus's directory below it, the held-out name
    patterns, and how many recordings each side must hold; a tree that differs
    fails closed rather than quietly measuring a different corpus.
    """
    directory = Path(root) / entry["directory"]
    paths = sorted(directory.glob(entry["file_pattern"]))
    held_out, training = [], []
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        matched = any(fnmatch(relative, p) for p in entry["held_out_patterns"])
        (held_out if matched else training).append(path)
    declared = entry["recordings"]
    found = (len(paths), len(held_out), len(training))
    if found != (declared["total"], declared["held_out"], declared["training"]):
        raise ValueError(
            f"{entry['name']} under {directory} holds {found} total/held-out/"
            "training recordings, not what the frozen manifest declares"
        )
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError(f"{entry['name']} recording identities are not unique stems")
    return training, held_out


def corpus_recordings(entry, loaded):
    """Adapt one corpus's loaded trajectories into a generic collection."""
    channels = command_channels(loaded[0][1].spec)
    segments = []
    for name, trajectory in loaded:
        if command_channels(trajectory.spec) != channels:
            raise ValueError(f"{entry['name']} recordings declare different commands")
        segments.extend(
            trajectory_segments(
                name,
                trajectory,
                dt_s=entry["sample_interval_s"],
                tolerance_fraction=entry["sample_interval_tolerance_fraction"],
            )
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id=f"platform-{entry['name']}",
        state_channels=OBSERVED_CHANNELS,
        input_channels=channels,
    )


# --- the platform tier: the same rows for both models -----------------------


def platform_origins(segments, steps, stride):
    """Every origin at the declared stride with a full context and horizon."""
    context, horizon = steps["history"], steps["horizon"]
    return [
        (segment, row)
        for segment in segments
        for row in range(context, len(segment.states) - horizon, stride)
    ]


def platform_rows(held_out, entry, steps, motor_history_steps):
    """The evaluation arrays both models forecast from, origin for origin.

    Every origin carries the generic model's consumed context and the recorded
    future commands, and the same origin's full canonical state, command
    history and exogenous context for the structured rollout. One index, one
    set of rows, two models.
    """
    from glassbox.core.data import control_history_before

    context, horizon = steps["history"], steps["horizon"]
    stride = entry["evaluation_origin_stride_rows"]
    columns = {
        key: []
        for key in (
            "past_states",
            "past_inputs",
            "future_inputs",
            "targets",
            "initial_states",
            "control_histories",
            "controls",
            "initial_exogenous",
        )
    }
    identities, origins = [], []
    for name, trajectory, segments in held_out:
        for segment, row in platform_origins(segments, steps, stride):
            source = segment.start_row + row
            columns["past_states"].append(segment.states[row - context : row + 1])
            columns["past_inputs"].append(segment.inputs[row - context : row])
            columns["future_inputs"].append(segment.inputs[row : row + horizon])
            columns["targets"].append(segment.states[row + 1 : row + horizon + 1])
            columns["initial_states"].append(trajectory.states[source])
            columns["control_histories"].append(
                control_history_before(trajectory, source, motor_history_steps)
            )
            columns["controls"].append(trajectory.controls[source : source + horizon])
            columns["initial_exogenous"].append(trajectory.exogenous[source])
            identities.append(name)
            origins.append(source)
    if not identities:
        raise ValueError(f"{entry['name']} has no origin with a complete context")
    spec = held_out[0][1].spec
    return dict(
        **{key: np.stack(value) for key, value in columns.items()},
        recording_ids=np.array(identities),
        source_origins=np.array(origins),
        control_roles=np.array(list(spec.control_roles), dtype="<U64"),
        exogenous_roles=np.array(list(spec.exogenous_roles), dtype="<U64"),
    )


def structured_forecast(params, arrays, dt_s):
    """Roll the structured model from each origin's full canonical state.

    This is the library's own rollout, initialized the way ``predict_windows``
    initializes a fixed-horizon window: the applied-actuator state inferred
    from the real command history before the origin, and the exogenous context
    recorded at the origin held across the horizon.
    """
    import jax
    import jax.numpy as jnp

    from glassbox.core.dynamics import control_state_after_history, rollout_with_latent

    roles = tuple(str(role) for role in arrays["control_roles"])
    exogenous_roles = tuple(str(role) for role in arrays["exogenous_roles"])
    latent = jax.vmap(
        lambda history: control_state_after_history(params, history, dt_s, roles)
    )(jnp.asarray(arrays["control_histories"]))
    predicted, _ = jax.vmap(
        lambda state, controls, applied, context: rollout_with_latent(
            params, state, controls, dt_s, applied, roles, context, exogenous_roles
        )
    )(
        jnp.asarray(arrays["initial_states"]),
        jnp.asarray(arrays["controls"]),
        latent,
        jnp.asarray(arrays["initial_exogenous"]),
    )
    return observed_from_states(np.asarray(predicted, dtype=float)[:, 1:])


def hold_current(past_states, horizon):
    """The reference every corpus is read against: nothing changes."""
    return np.repeat(np.asarray(past_states)[:, -1:], horizon, axis=1)


# --- the platform tier: scoring and the decision ----------------------------


def vector_rmse(error):
    """One scalar over rows and the three components, as ``core.metrics`` does."""
    return float(np.sqrt(np.mean(np.square(np.asarray(error, dtype=float)))))


def platform_measure(prediction, targets):
    """Final-step and horizon-prefix velocity and body-rate RMSE."""
    error = np.asarray(prediction, dtype=float) - np.asarray(targets, dtype=float)
    return dict(
        final_step={
            METRICS[0]: vector_rmse(error[:, -1, VELOCITY_CHANNELS]),
            METRICS[1]: vector_rmse(error[:, -1, BODY_RATE_CHANNELS]),
        },
        horizon_prefix={
            METRICS[0]: vector_rmse(error[..., VELOCITY_CHANNELS]),
            METRICS[1]: vector_rmse(error[..., BODY_RATE_CHANNELS]),
        },
    )


def platform_score(prediction, targets, ids):
    """One model on one corpus, pooled and per recording."""
    ids = np.asarray(ids)
    return dict(
        rows=len(targets),
        **platform_measure(prediction, targets),
        recordings={
            str(name): platform_measure(prediction[ids == name], targets[ids == name])
            for name in sorted(set(ids.tolist()))
        },
    )


def _number(value):
    """A finite float, or None for anything that is not a JSON number.

    Missing, null, nonfinite, or not a number at all: a gate that cannot read a
    number fails closed rather than raising or parsing a string into one.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if np.isfinite(value) else None


def _score(recorded, metric):
    """One recorded final-step score as a finite float, or None for anything else."""
    final = recorded.get("final_step") if isinstance(recorded, dict) else None
    return _number(final.get(metric) if isinstance(final, dict) else None)


def _comparator(arms, metric):
    """The better arm on these rows, metric by metric, and its value.

    Taking the lowest value any declared arm reached is the strictest reading
    of "the better arm": the generic model has to beat whichever structured arm
    did best on that metric, not an arm chosen for it. Both arms are reported.
    """
    values = {arm: _score(score, metric) for arm, score in arms.items()}
    finite = {arm: value for arm, value in values.items() if value is not None}
    if not finite:
        return None, None, False
    best = min(finite, key=lambda arm: finite[arm])
    return best, finite[best], len(finite) == len(values)


def _reference_value(reference, section, *keys):
    """One recorded reference number, or None for anything that is not one."""
    node = None if reference is None else reference.get(section)
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return _number(node)


def _reference_meets(value, limits):
    """Whether the reference's own recorded value meets this case's rule.

    One rule on every tier: the rule is reported on every case, and it gates
    only the cases the reference already meets. A case the incumbent already
    fails is measured and reported rather than blocking every change, which is
    the whole reason an enforced rule can sit beside an unmet row.

    ``limits`` are the rule's own limits on the same rows, measured in this
    run. A reference value or a limit that cannot be read leaves the answer
    unknown, and an unknown case does not gate; the regression gate fails
    closed on the unreadable value separately.
    """
    if value is None or any(limit is None for limit in limits):
        return None
    return all(value <= limit for limit in limits)


def platform_decide(manifest, rows, reference=None, reference_sha256=None):
    """Every gate, evaluated from recorded scores alone. Anything unclear fails.

    A run is accepted when no metric regressed past its reference value times
    one plus the manifest's relative tolerance plus its absolute one, the rule
    holds on every case the reference already meets it on, and nothing
    structural failed. Structural problems always fail closed, whatever the
    reference says. The rule is reported on every case either way, and
    ``rule_met`` says whether it holds everywhere, which is the stronger
    statement the status table's row is read from.

    ``reference_sha256`` identifies the reference file the scores were compared
    against. It is recorded in the decision so a replay can check that it
    compared the same bytes, and it is not otherwise used here.
    """
    declared = {entry["name"]: entry for entry in manifest["corpora"]}
    names = [row.get("corpus") for row in rows]
    breaches, rule_breaches, regressions, summary = [], [], [], {}
    relative = manifest["reference"]["relative_tolerance"]
    absolute = manifest["reference"]["absolute_tolerance"]
    for name in sorted(set(declared) - set(names)):
        breaches.append(dict(corpus=name, gate="corpus_present"))
    for name in sorted({name for name in names if names.count(name) > 1}):
        breaches.append(dict(corpus=name, gate="corpus_unique"))
    for name in sorted(set(names) - set(declared)):
        breaches.append(dict(corpus=str(name), gate="corpus_declared"))
    for row in rows:
        name = row.get("corpus")
        entry = declared.get(name)
        if entry is None:
            continue
        if row.get("status") != "complete":
            breaches.append(dict(corpus=name, gate="fit_complete"))
            continue
        scored = row.get("evaluation_rows")
        limit = manifest["evaluation_rows_maximum"]
        if not isinstance(scored, int) or not 0 < scored <= limit:
            breaches.append(
                dict(corpus=name, gate="evaluation_rows", value=scored, limit=limit)
            )
        arms = row.get("structured", {})
        if sorted(arms) != sorted(entry["structured"]["arms"]):
            breaches.append(dict(corpus=name, gate="arms_declared"))
            continue
        allowance = entry["allowance"]
        for metric in METRICS:
            generic = _score(row.get("generic", {}), metric)
            arm, comparator, complete = _comparator(arms, metric)
            ceiling = None if allowance is None else allowance[metric]
            base = _reference_value(reference, "final_step_rmse", name, metric)
            meets = _reference_meets(
                base, [comparator] if ceiling is None else [comparator, ceiling]
            )
            summary.setdefault(name, {})[metric] = dict(
                generic=generic,
                comparator=comparator,
                comparator_arm=arm,
                allowance=ceiling,
                hold_current=_score(row.get("hold_current", {}), metric),
                reference=base,
                reference_meets_rule=meets,
            )
            if reference is not None:
                if base is None:
                    regressions.append(
                        dict(corpus=name, metric=metric, gate="reference_present")
                    )
                else:
                    limit = base * (1 + relative) + absolute
                    if generic is None or generic > limit:
                        regressions.append(
                            dict(
                                corpus=name,
                                metric=metric,
                                gate="reference_final_step_rmse",
                                value=generic,
                                reference=base,
                                limit=limit,
                            )
                        )
            if generic is None or comparator is None or not complete:
                breaches.append(
                    dict(
                        corpus=name,
                        metric=metric,
                        gate="finite_final_step_rmse",
                        value=generic,
                    )
                )
                continue
            if generic > comparator:
                rule_breaches.append(
                    dict(
                        corpus=name,
                        metric=metric,
                        gate="comparator_final_step_rmse",
                        value=generic,
                        limit=comparator,
                        arm=arm,
                        gating=meets is True,
                    )
                )
            if ceiling is not None and generic > ceiling:
                rule_breaches.append(
                    dict(
                        corpus=name,
                        metric=metric,
                        gate="allowance_final_step_rmse",
                        value=generic,
                        limit=ceiling,
                        gating=meets is True,
                    )
                )
    regressions.sort(key=lambda entry: (entry["corpus"], entry["metric"]))
    enforced = bool(manifest["decision"]["enforced"])
    gating = [breach for breach in rule_breaches if breach["gating"]]
    rule_met = not breaches and not rule_breaches
    accepted = not breaches and not regressions and (not enforced or not gating)
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        rule_met=rule_met,
        rule_enforced=enforced,
        gating_rule_breaches=len(gating),
        corpora=len(rows),
        gate_breaches=breaches,
        rule_breaches=rule_breaches,
        reference_regressions=regressions,
        reference_compared=reference is not None,
        reference_sha256=reference_sha256,
        final_step_rmse=summary,
        meaning=manifest["decision"]["meaning"],
    )


# --- the platform tier: running ---------------------------------------------


def _structured_spec(entry, arm):
    """The FitSpec one corpus's recorded validation chain fits this arm with."""
    from glassbox.fitting import FitSpec, Holdout, LossPolicy

    declared = entry["structured"]
    horizons = declared["training_horizons_s"]
    return FitSpec(
        holdout=Holdout.by_group(declared["holdout_count"]),
        horizons_s=None if horizons is None else tuple(horizons),
        horizon_steps=declared["training_horizon_steps"],
        steps=declared["optimization_steps"],
        learning_rate=declared["learning_rate"],
        evaluation_horizons_s=tuple(declared["evaluation_horizons_s"]),
        model_class=arm,
        loss=LossPolicy(
            endpoint_weight=declared["endpoint_weight"],
            stability_regularization=declared["stability_regularization"],
        ),
    )


def _structured_arm(entry, arm, training_paths, arrays, directory):
    """Fit one structured arm on exactly the training recordings, then forecast."""
    from glassbox.belief.belief_io import save_dynamics_belief
    from glassbox.fitting import fit as structured_fit

    started = time.perf_counter()
    outcome = structured_fit(
        [str(path) for path in training_paths], _structured_spec(entry, arm)
    )
    wall = time.perf_counter() - started
    split = outcome.report["split"]
    trained = [item["path"] for item in split["training_flights"]]
    reserved = [item["path"] for item in split["validation_flights"]]
    if sorted(trained + reserved) != sorted(str(path) for path in training_paths):
        raise ValueError(f"{entry['name']} {arm} fit did not read the training set")
    save_dynamics_belief(outcome.belief, directory / f"structured_{arm}.json")
    write(directory / f"structured_{arm}_report.json", outcome.report)
    prediction = structured_forecast(
        outcome.belief.params, arrays, entry["sample_interval_s"]
    )
    if not np.isfinite(prediction).all():
        raise ValueError(f"nonfinite structured forecast in {entry['name']} {arm}")
    learned = outcome.report["models"]["learned_lag"]["fit"]
    return prediction, dict(
        wall_seconds=wall,
        initial_loss=float(learned["initial_loss"]),
        final_loss=float(learned["final_loss"]),
        training_recordings=[Path(path).stem for path in trained],
        reserved_recordings=[Path(path).stem for path in reserved],
        report=f"structured_{arm}_report.json",
    )


def _platform_corpus(manifest, entry, root, output):
    """One corpus end to end: adapt, fit both models, score the same rows."""
    from glassbox.core.data import load_trajectory_npz

    name = entry["name"]
    directory = output / name
    directory.mkdir()
    dt_s = entry["sample_interval_s"]
    steps = steps_for(dt_s)
    training_paths, held_paths = corpus_paths(entry, root)
    training = [(path.stem, load_trajectory_npz(path)) for path in training_paths]
    held = [(path.stem, load_trajectory_npz(path)) for path in held_paths]

    started = time.perf_counter()
    learned = fit(corpus_recordings(entry, training))
    generic_wall = time.perf_counter() - started
    learned.save(directory / "generic.npz")
    if set(learned.report["training"]) | set(learned.report["development"]) != {
        stem for stem, _ in training
    }:
        raise ValueError(f"{name} generic fit did not read the training set")

    held_out = [
        (
            stem,
            trajectory,
            trajectory_segments(
                stem,
                trajectory,
                dt_s=dt_s,
                tolerance_fraction=entry["sample_interval_tolerance_fraction"],
            ),
        )
        for stem, trajectory in held
    ]
    motor_history_steps = max(1, int(np.rint(manifest["motor_history_s"] / dt_s)))
    arrays = platform_rows(held_out, entry, steps, motor_history_steps)
    if set(arrays["recording_ids"].tolist()) & {stem for stem, _ in training}:
        raise ValueError(f"{name} held-out identities overlap the training set")

    generic = np.asarray(
        learned.predict(
            arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
        )
    )
    if not np.isfinite(generic).all():
        raise ValueError(f"nonfinite generic forecast in {name}")
    half_width = envelope_rows(learned, generic)
    hold = hold_current(arrays["past_states"], steps["horizon"])
    predictions, fits = {}, {}
    for arm in entry["structured"]["arms"]:
        print(json.dumps(dict(fitting=f"{name}/{arm}")), flush=True)
        predictions[arm], fits[arm] = _structured_arm(
            entry, arm, training_paths, arrays, directory
        )
    np.savez_compressed(
        directory / "evaluation.npz",
        **arrays,
        generic_prediction=generic,
        generic_envelope_half_width=half_width,
        hold_prediction=hold,
        **{f"structured_{arm}_prediction": p for arm, p in predictions.items()},
    )
    targets, ids = arrays["targets"], arrays["recording_ids"]
    files = ["generic.npz", "evaluation.npz"]
    for arm in predictions:
        files += [f"structured_{arm}.json", f"structured_{arm}_report.json"]
    row = dict(
        corpus=name,
        status="complete",
        sample_interval_s=dt_s,
        context_steps=steps["history"],
        horizon_steps=steps["horizon"],
        horizon_s=steps["horizon"] * dt_s,
        evaluation_origin_stride_rows=entry["evaluation_origin_stride_rows"],
        evaluation_rows=len(targets),
        motor_history_steps=motor_history_steps,
        recordings=dict(
            training=[stem for stem, _ in training],
            held_out=[stem for stem, _ in held],
        ),
        generic_fingerprint=learned.fingerprint(),
        generic_fit_wall_seconds=generic_wall,
        generic_report=learned.report,
        generic=platform_score(generic, targets, ids),
        hold_current=platform_score(hold, targets, ids),
        structured={
            arm: platform_score(prediction, targets, ids)
            for arm, prediction in predictions.items()
        },
        structured_fits=fits,
        evidence=envelope_coverage(
            generic, targets, half_width, evidence_groups("rigid_body_15", 15)
        ),
        files=_files(directory, files),
    )
    write(directory / "result.json", row)
    return row


def platform(manifest_path, corpora_root, output):
    """Measure every pinned corpus against the structured model and decide."""
    import jax

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = frozen_platform_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
    evidence_manifest, evidence_reference, evidence_digest = evidence_setup(output)
    reference_path = manifest_path.parent / manifest["reference"]["file"]
    reference, reference_digest = None, None
    if reference_path.exists():
        shutil.copyfile(reference_path, output / "reference.json")
        reference, reference_digest = read(reference_path), sha256(reference_path)
    write(
        output / "environment.json",
        dict(
            python=sys.version,
            platform=platform_module.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=True,
            corpora_root=str(Path(corpora_root)),
        ),
    )
    rows = []
    started = time.perf_counter()
    with jax.enable_x64(True):
        for entry in manifest["corpora"]:
            print(json.dumps(dict(starting=entry["name"])), flush=True)
            rows.append(_platform_corpus(manifest, entry, corpora_root, output))
            write(output / "results.json", rows)
    decision = platform_decide(manifest, rows, reference, reference_digest)
    with_evidence(
        decision,
        evidence_decide(
            evidence_manifest,
            "platform",
            evidence_table(rows, "corpus", regime=PLATFORM_EVIDENCE_REGIME),
            {
                (entry["name"], PLATFORM_EVIDENCE_REGIME)
                for entry in manifest["corpora"]
            },
            evidence_reference,
            evidence_digest,
        ),
    )
    decision["wall_seconds"] = time.perf_counter() - started
    write(output / "decision.json", decision)
    return decision


# --- the platform tier: verifying -------------------------------------------


def _same_scores(fresh, saved, label):
    if sorted(fresh) != sorted(saved) or fresh["rows"] != saved["rows"]:
        raise ValueError(f"score shape mismatch: {label}")
    for horizon in ("final_step", "horizon_prefix"):
        for metric in METRICS:
            np.testing.assert_allclose(
                fresh[horizon][metric], saved[horizon][metric], **SCORE_TOLERANCE
            )
    if sorted(fresh["recordings"]) != sorted(saved["recordings"]):
        raise ValueError(f"recording set mismatch: {label}")
    for recording, metrics in fresh["recordings"].items():
        for horizon in ("final_step", "horizon_prefix"):
            for metric in METRICS:
                np.testing.assert_allclose(
                    metrics[horizon][metric],
                    saved["recordings"][recording][horizon][metric],
                    **SCORE_TOLERANCE,
                )


def verify_platform(directory, manifest, reference=None):
    """Recompute every platform score and the decision from saved arrays.

    The regression reference is anchored to the committed file exactly as the
    synthetic tier anchors its own: the copy inside the run is only ever checked
    against it, and a run that saved a different reference, or none, than the
    committed one cannot be replayed.
    """
    import jax

    from glassbox.belief.belief_io import load_dynamics_belief

    directory = Path(directory)
    declared = {entry["name"]: entry for entry in manifest["corpora"]}
    rows = read(directory / "results.json")
    replays, worst = 0, 0.0
    with jax.enable_x64(True):
        for row in rows:
            case = directory / row["corpus"]
            if read(case / "result.json") != row:
                raise ValueError(f"case result mismatch: {row['corpus']}")
            for name, digest in row["files"].items():
                if sha256(case / name) != digest:
                    raise ValueError(f"altered artifact: {row['corpus']}/{name}")
            entry = declared[row["corpus"]]
            steps = steps_for(entry["sample_interval_s"])
            learned = LearnedDynamics.load(case / "generic.npz")
            if learned.fingerprint() != row["generic_fingerprint"]:
                raise ValueError(f"model fingerprint mismatch: {row['corpus']}")
            with np.load(case / "evaluation.npz", allow_pickle=False) as data:
                if data["past_states"].shape[1] != steps["history"] + 1:
                    raise ValueError("saved evaluation rows lack the consumed context")
                if data["future_inputs"].shape[1] != steps["horizon"]:
                    raise ValueError("saved evaluation rows lack the declared horizon")
                if len(data["targets"]) != row["evaluation_rows"]:
                    raise ValueError(f"evaluation row count differs: {row['corpus']}")
                generic = replay(
                    learned._model,
                    data["past_states"],
                    data["past_inputs"],
                    data["future_inputs"],
                )
                worst = max(
                    worst,
                    float(np.max(np.abs(generic - data["generic_prediction"]))),
                )
                np.testing.assert_allclose(
                    generic, data["generic_prediction"], **REPLAY_TOLERANCE
                )
                hold = hold_current(data["past_states"], steps["horizon"])
                np.testing.assert_array_equal(hold, data["hold_prediction"])
                targets, ids = data["targets"], data["recording_ids"]
                fresh = dict(
                    generic=platform_score(generic, targets, ids),
                    hold_current=platform_score(hold, targets, ids),
                    structured={},
                )
                coverage = _replayed_coverage(
                    learned,
                    generic,
                    dict(
                        envelope_half_width=data["generic_envelope_half_width"],
                        targets=targets,
                    ),
                    evidence_groups("rigid_body_15", 15),
                    row["corpus"],
                )
                replays += 2
                for arm in entry["structured"]["arms"]:
                    belief = load_dynamics_belief(case / f"structured_{arm}.json")
                    prediction = structured_forecast(
                        belief.params, data, entry["sample_interval_s"]
                    )
                    saved = data[f"structured_{arm}_prediction"]
                    worst = max(worst, float(np.max(np.abs(prediction - saved))))
                    np.testing.assert_allclose(prediction, saved, **REPLAY_TOLERANCE)
                    fresh["structured"][arm] = platform_score(prediction, targets, ids)
                    replays += 1
            _same_scores(fresh["generic"], row["generic"], f"{row['corpus']}/generic")
            _same_scores(
                fresh["hold_current"], row["hold_current"], f"{row['corpus']}/hold"
            )
            if sorted(fresh["structured"]) != sorted(row["structured"]):
                raise ValueError(f"structured arm mismatch: {row['corpus']}")
            for arm, score in fresh["structured"].items():
                _same_scores(score, row["structured"][arm], f"{row['corpus']}/{arm}")
            _same_coverage(coverage, row["evidence"], row["corpus"])
            row["generic"] = fresh["generic"]
            row["hold_current"] = fresh["hold_current"]
            row["structured"] = fresh["structured"]
            row["evidence"] = coverage
    anchor, digest = anchored_reference(
        directory, reference, COMMITTED_PLATFORM_REFERENCE
    )
    decision = platform_decide(manifest, rows, anchor, digest)
    evidence_manifest, evidence_anchored, evidence_digest = evidence_anchor(directory)
    evidence = evidence_decide(
        evidence_manifest,
        "platform",
        evidence_table(rows, "corpus", regime=PLATFORM_EVIDENCE_REGIME),
        {(entry["name"], PLATFORM_EVIDENCE_REGIME) for entry in manifest["corpora"]},
        evidence_anchored,
        evidence_digest,
    )
    saved = read(directory / "decision.json")
    _same_evidence(evidence, saved.get("evidence"), "platform")
    with_evidence(decision, evidence)
    for key in (
        "manifest",
        "decision",
        "accepted",
        "rule_met",
        "gating_rule_breaches",
        "corpora",
        "reference_compared",
        "reference_sha256",
    ):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed decision differs: {key}")
    for key in ("gate_breaches", "rule_breaches", "reference_regressions"):
        if len(decision[key]) != len(saved[key]):
            raise ValueError(f"replayed decision differs: {key}")
    return dict(
        tier="platform",
        verified_corpora=len(rows),
        replays=replays,
        maximum_replay_difference=worst,
        decision=decision,
        meaning="A replay of saved evidence, not a new fit or independent confirmation.",
    )


# --- simulated time: what a clock measured, recorded and read by nothing ----

SIMULATED_TIME_MEANING = (
    "host measurements of this run only. None of them enters the trajectory, a "
    "command or any metric, and two runs of this tier differ in all of them."
)
"""What a simulated-time tier's ``wall`` block is, stated in the block itself."""


def simulated_time_wall(
    *, meaning, dt_s, deadline_s, tick_times, solve_times, elapsed_s, **extra
):
    """The host measurements of one trial computed in simulated time.

    Shared by the live tier and the control tier. Both compute their trajectory
    from the plant, the models and the reference alone, and both still measure
    how long the host took to do it. Everything here is a measurement of one
    run on one machine: it is recorded because it was measured, it is reported,
    and nothing that decides a run reads it back. ``deadline_s`` is the
    threshold the solve times are counted against, not a deadline any solve was
    given.
    """
    ticks = np.asarray(tick_times, dtype=float)
    solves = np.asarray(solve_times, dtype=float)
    return dict(
        meaning=meaning,
        **extra,
        maximum_interval_seconds=max(tick_times, default=0.0),
        maximum_solve_seconds=max(solve_times, default=0.0),
        intervals_over_sample_interval=int(np.sum(ticks > dt_s)),
        solves_over_deadline=int(np.sum(solves > deadline_s)),
        solve_deadline_s=deadline_s,
        elapsed_seconds=elapsed_s,
    )


def verify_simulated_time_wall(
    wall, *, dt_s, deadline_s, tick_times, solve_times, label
):
    """Recompute a recorded ``wall`` block's two counts from the saved times.

    They decide nothing, which is exactly why they are rechecked: the point of
    the block is that it is a measurement the trajectory did not read, and a
    replay that recomputes it from the saved per-interval times can say so.
    """
    over_interval = int(np.sum(np.asarray(tick_times, dtype=float) > dt_s))
    over_deadline = int(np.sum(np.asarray(solve_times, dtype=float) > deadline_s))
    if (
        over_interval != wall["intervals_over_sample_interval"]
        or over_deadline != wall["solves_over_deadline"]
    ):
        raise ValueError(f"recomputed host measurements differ: {label}")


# --- the control tier: one Cascade trial set, two arms ----------------------

CONTROL_MANIFEST_SHA256 = (
    "c87b40e1835c2dc9725f7a9effd47d5de225c692b06e63ea32141e632161cb40"
)
"""Digest of the frozen control manifest this module is allowed to run."""

COMMITTED_CONTROL_MANIFEST = COMMITTED_MANIFEST.parent / "control-v5.json"
"""The frozen control manifest in a source checkout."""

COMMITTED_CONTROL_REFERENCE = COMMITTED_MANIFEST.parent / "control-reference.json"
"""Where ``verify`` looks for the committed control reference by default."""

CONTROL_EVIDENCE_REGIME = "reserved"
CONTROL_EVIDENCE_PLAN = dict(evaluation_origin_start=0, evaluation_stride=1)
"""Every origin of the reserved recording with a whole context and horizon.

The evidence manifest declares that scope, and the reserved recording is the
one this tier never fits in any role, so nothing has to be held back for it.
"""

CONTROL_ARMS = ("generic", "structured")
CONTROL_METRICS = ("position_rmse_m", "attitude_rmse_deg")
"""The two tracking metrics the declared rule compares, arm against arm."""

CONTROL_REPORTED_METRICS = CONTROL_METRICS + (
    "velocity_rmse_m_s",
    "angular_velocity_rmse_rad_s",
)


def frozen_control_manifest(path):
    """Load the control manifest only when its bytes match the frozen digest."""
    if sha256(path) != CONTROL_MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen harness contract")
    manifest = read(path)
    declared_plan(manifest)
    steps = steps_for(manifest["information_budget"]["sample_interval_s"])
    budget = manifest["information_budget"]
    if (steps["history"], steps["delay"], steps["horizon"]) != (
        budget["context_steps"],
        budget["delay_steps"],
        budget["horizon_steps"],
    ):
        raise ValueError("manifest information budget differs from the recipe")
    trial = manifest["trial"]
    if trial["intervals"] != round(trial["duration_s"] / trial["sample_interval_s"]):
        raise ValueError("manifest trial duration and interval count disagree")
    # This tier's trajectory is a function of the plant, the models and the
    # reference alone, which a solve cut short for want of time would break.
    if trial["solver_deadline_applied"] is not False:
        raise ValueError("this tier's solver is given no deadline")
    # The calibration's own excitation is declared to the generic learner, and
    # the manifest has to say in what form or a run cannot rebuild it.
    if not str(manifest["calibration"]["excitation"].get("declared_form", "")).strip():
        raise ValueError("the calibration does not declare the form of its excitation")
    return manifest


def control_reference(anchor_state, times, declared):
    """The declared tracking task, rebuilt from the unperturbed trim state.

    The task of ``docs/cascade-accuracy.md``: cruise carried forward at the trim
    state's own world velocity, with a lateral position sine and an altitude
    sine about it and the world velocities that match them. The anchor is the
    trim state, not a trial's own perturbed start, so every trial is scored
    against the same task. ``verify`` rebuilds the rows from the saved anchor
    rather than believing the saved reference, so a run cannot score itself
    against a reference it invented.
    """
    anchor_state = np.asarray(anchor_state, dtype=float)
    times = np.asarray(times, dtype=float)
    states = np.tile(anchor_state, (len(times), 1))
    lateral = declared["lateral_amplitude_m"]
    lateral_rate = declared["lateral_rate_rad_s"]
    altitude = declared["altitude_amplitude_m"]
    altitude_rate = declared["altitude_rate_rad_s"]
    states[:, 0:3] += times[:, None] * anchor_state[3:6]
    states[:, 1] += lateral * np.sin(lateral_rate * times)
    states[:, 2] += altitude * np.sin(altitude_rate * times)
    states[:, 4] += lateral * lateral_rate * np.cos(lateral_rate * times)
    states[:, 5] += altitude * altitude_rate * np.cos(altitude_rate * times)
    return states


CONTROL_TRACKING_ROWS = slice(1, 3)
"""The lateral and vertical world-position rows the pass criterion reads."""


def control_initial_state(manifest, anchor_state, seed):
    """The declared per-trial perturbation of the trim state, from one seed.

    The only disturbance the trial set introduces, exactly as
    ``docs/cascade-accuracy.md`` declares it: a tangent-space offset drawn from
    this seed and applied with the library's retraction, so the quaternion stays
    on the unit sphere instead of acquiring a Euclidean displacement. Both arms
    of one repetition fly from the same perturbed state.
    """
    import jax
    import jax.numpy as jnp

    from glassbox.core.geometry import state_plus_tangent

    declared = manifest["trial"]["initial_state_perturbation"]
    generator = np.random.default_rng(seed)
    offset = np.zeros(12)
    for rows, name in (
        (slice(1, 3), "lateral_and_vertical_position_m"),
        (slice(4, 6), "lateral_and_vertical_velocity_m_s"),
        (slice(6, 9), "attitude_tangent_rad"),
        (slice(9, 12), "body_rate_rad_s"),
    ):
        width = float(declared[name])
        offset[rows] = generator.uniform(-width, width, offset[rows].size)
    # The retraction runs in x64 wherever it is called from, so a trial start
    # is the same bytes in a run and in a replay that did not enable it.
    with jax.enable_x64(True):
        moved = state_plus_tangent(
            jnp.asarray(np.asarray(anchor_state, dtype=float)), jnp.asarray(offset)
        )
    return np.asarray(moved, dtype=float)


def control_pass_criterion(states, anchor_state, manifest):
    """The page's own pass criterion, measured from one trial's saved states.

    Both absolute lateral and altitude errors at most the declared tolerance, in
    at least the declared fraction of the samples after the declared settling
    time, with no terminated trial. Every unexecuted interval counts as outside
    tolerance and nothing is discarded, which is how
    ``docs/cascade-accuracy.md`` accounts for a trial that stopped early. It is
    reported on every trial and gates nothing: the decision reads the RMSE rule
    and the regression reference.
    """
    declared = manifest["metrics"]["pass_criterion"]
    trial = manifest["trial"]
    requested = int(trial["intervals"])
    times = np.arange(requested + 1) * trial["sample_interval_s"]
    target = control_reference(anchor_state, times, manifest["tracking_reference"])
    states = np.asarray(states, dtype=float)
    executed = max(0, len(states) - 1)
    error = np.full((requested, 2), np.inf)
    if executed:
        rows = slice(1, executed + 1)
        error[:executed] = np.abs(
            states[rows, CONTROL_TRACKING_ROWS] - target[rows, CONTROL_TRACKING_ROWS]
        )
    scored = times[1:] >= declared["settled_after_s"]
    within = np.all(error <= declared["tolerance_m"], axis=1)
    fraction = float(np.mean(within[scored])) if scored.any() else 0.0
    return dict(
        scored_samples=int(np.count_nonzero(scored)),
        within_samples=int(np.count_nonzero(within[scored])),
        within_tolerance_fraction=fraction,
        lateral_rmse_m=_finite(np.sqrt(np.mean(error[scored, 0] ** 2))),
        altitude_rmse_m=_finite(np.sqrt(np.mean(error[scored, 1] ** 2))),
        terminated=bool(executed != requested),
        met=bool(fraction >= declared["minimum_fraction"] and executed == requested),
    )


def control_excitation(manifest, recordings):
    """Every command channel's excitation, per recording, as a fraction of range.

    The calibration must contain the command directions the controller is asked
    to use. That is a property of the recordings the caller supplies, measured
    here and applied identically to both arms; nothing about it reaches the
    learner, which is told the channels and nothing else.
    """
    declared = manifest["telemetry"]
    ranges = np.asarray(declared["command_maximum"], dtype=float) - np.asarray(
        declared["command_minimum"], dtype=float
    )
    if np.any(ranges <= 0):
        raise ValueError("a declared command channel has no range to be excited in")
    deviations, fractions = {}, {}
    for name, flight in recordings:
        commands = np.asarray(flight.controls, dtype=float)
        if commands.ndim != 2 or commands.shape[1] != len(ranges):
            raise ValueError(f"recording {name} does not carry the declared commands")
        deviation = commands.std(axis=0)
        deviations[name] = deviation.tolist()
        fractions[name] = (deviation / ranges).tolist()
    return dict(
        declared_range=ranges.tolist(),
        standard_deviation=deviations,
        fraction=fractions,
    )


def declared_excitation(*, amplitudes, rates_rad_s, ramp_s, seed, intervals, dt_s):
    """The known additive command excitation a protocol injects, interval by interval.

    ``ramp * amplitudes * sin(rates * elapsed + phases)`` on this tier's own
    sample grid, with ``ramp = min(1, elapsed / ramp_s)`` and the phases drawn
    from one declared seed. It is the form the calibration pilot's excitation
    already has, written once so the calibration, the live tier's trial dither
    and both replays compute the same bytes from the same declared constants.
    Nothing here is measured: a replay recomputes this table from the manifest
    and the seed rather than believing what a run wrote beside its commands.
    """
    amplitudes = np.asarray(amplitudes, dtype=float)
    rates = np.asarray(rates_rad_s, dtype=float)
    if amplitudes.shape != rates.shape or amplitudes.ndim != 1:
        raise ValueError("excitation amplitudes and rates are one per command channel")
    if not np.isfinite(ramp_s) or ramp_s <= 0 or int(intervals) < 0:
        raise ValueError("the excitation ramp and interval count must be positive")
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=len(amplitudes))
    elapsed = np.arange(int(intervals)) * float(dt_s)
    ramp = np.minimum(1.0, elapsed / float(ramp_s))
    return ramp[:, None] * amplitudes * np.sin(rates * elapsed[:, None] + phases)


def calibration_excitation(manifest, seed):
    """One calibration recording's declared additive excitation, from the manifest."""
    declared = manifest["calibration"]
    dt_s = manifest["plant"]["sample_interval_s"]
    return declared_excitation(
        amplitudes=declared["excitation_amplitudes"],
        rates_rad_s=declared["excitation_rates_rad_s"],
        ramp_s=declared["setpoint"]["ramp_s"],
        seed=seed,
        intervals=round(declared["duration_s"] / dt_s),
        dt_s=dt_s,
    )


def _command_fraction(excitation, commands):
    """Each channel's excitation standard deviation over its own command range.

    The same quantity the learner's own report measures from its recordings,
    computed here from a trial's applied commands so a run states what it
    injected in the units the contract states it in.
    """
    excitation = np.asarray(excitation, dtype=float)
    commands = np.asarray(commands, dtype=float)
    if not len(excitation) or not len(commands):
        return None
    span = commands.max(axis=0) - commands.min(axis=0)
    deviation = excitation.std(axis=0)
    return [
        None if not width > 0 else float(value / width)
        for value, width in zip(deviation, span, strict=True)
    ]


def _bounded_intervals(solved, dither, minimum, maximum):
    """How many intervals the declared command box actually held the sum in.

    The dithered command is the solved one plus the declared dither, clipped to
    the box; this counts the intervals where that clip bound on any channel, so
    the difference between what was asked for and what reached the aircraft is
    a number rather than an assumption. It is not the count of intervals where
    the recorded excitation differs from the dither: the applied command is
    recorded and the excitation read back off it, so a rounding difference of
    one unit in the last place is expected on almost every interval and means
    nothing.
    """
    solved, dither = np.asarray(solved, dtype=float), np.asarray(dither, dtype=float)
    if not len(solved):
        return 0
    requested = solved + dither
    outside = (requested < minimum) | (requested > maximum)
    return int(np.count_nonzero(np.any(outside, axis=1)))


def _same_fraction(fresh, recorded, label, **tolerance):
    """One per-channel excitation fraction against the one a run recorded.

    A channel whose command never moves has no range to be excited in and its
    fraction is ``None`` on both sides; anything else is compared as a number.
    """
    if (fresh is None) != (recorded is None):
        raise ValueError(f"the recorded excitation fraction differs: {label}")
    if fresh is None:
        return
    if len(fresh) != len(recorded):
        raise ValueError(f"the recorded excitation fraction differs: {label}")
    for measured, saved in zip(fresh, recorded, strict=True):
        if (measured is None) != (saved is None):
            raise ValueError(f"the recorded excitation fraction differs: {label}")
        if measured is not None:
            np.testing.assert_allclose(measured, saved, **tolerance)


def trial_excitation(manifest, intervals=None):
    """The live tier's declared trial dither, identical in every trial and both arms.

    The calibration's per-channel amplitudes and rates, on the trial's own
    sample grid, with the phases drawn from the manifest's declared trial seed.
    It is the same sequence in every trial, which is what makes "applied
    identically to the frozen and the adopting arm" a statement a replay can
    check rather than a claim.
    """
    declared = manifest["trial"]["excitation"]
    calibration = manifest["calibration"]
    trial = manifest["trial"]
    return declared_excitation(
        amplitudes=calibration[declared["amplitudes_from"]],
        rates_rad_s=calibration[declared["rates_from"]],
        ramp_s=declared["ramp_s"],
        seed=declared["phase_seed"],
        intervals=trial["intervals"] if intervals is None else intervals,
        dt_s=trial["sample_interval_s"],
    )


def control_excitation_shortfall(manifest, measured, names):
    """The channels of these recordings that fall short of the declared fraction."""
    required = float(
        manifest["calibration"]["excitation"]["minimum_standard_deviation_fraction"]
    )
    short = []
    for name in names:
        for channel, value in enumerate(measured["fraction"][name]):
            if not float(value) >= required:
                short.append(
                    dict(
                        recording=name,
                        channel=channel,
                        fraction=float(value),
                        minimum=required,
                    )
                )
    return short


def control_telemetry_spec(manifest):
    """The declared telemetry contract every calibration recording carries."""
    from dataclasses import replace as dataclass_replace

    from glassbox.io.x8_reference import x8_trajectory_spec

    declared = manifest["telemetry"]
    minimum = declared["command_minimum"]
    maximum = declared["command_maximum"]
    base = x8_trajectory_spec(trusted_wind=False)
    return dataclass_replace(
        base,
        observation_source=manifest["plant"]["state_source"],
        channels=tuple(
            dataclass_replace(
                channel,
                minimum=float(low),
                maximum=float(high),
                semantic=semantic,
            )
            for channel, low, high, semantic in zip(
                base.controls, minimum, maximum, declared["command_semantics"]
            )
        ),
        vehicle=dataclass_replace(
            base.vehicle,
            configuration_id=declared["configuration_id"],
            fixed_states={
                **base.vehicle.fixed_states,
                "wind_world_m_s": list(manifest["plant"]["wind_world_m_s"]),
            },
        ),
    )


def control_fixture(manifest):
    """The pinned Cascade plant, its trim, and the state and command it rests at.

    The aircraft specification hash and the installed source revision are
    checked against the frozen manifest before anything is collected, so a
    changed simulator fails closed instead of quietly measuring another plant.
    """
    from importlib import metadata as importlib_metadata

    import cascade
    from cascade.canonical import rigid_body_to_canonical

    declared = manifest["plant"]
    distribution = importlib_metadata.distribution(declared["package"])
    if distribution.version != declared["version"]:
        raise ValueError(
            f"installed {declared['package']} {distribution.version} is not the "
            f"frozen {declared['version']}"
        )
    direct_url = distribution.read_text("direct_url.json")
    revision = json.loads(direct_url)["vcs_info"]["commit_id"] if direct_url else None
    if revision != declared["source_revision"]:
        raise ValueError(
            f"installed {declared['package']} source revision {revision} is not "
            f"the frozen {declared['source_revision']}"
        )
    spec = cascade.skywalker_x8_spec()
    if cascade.spec_hash(spec) != declared["spec_hash"]:
        raise ValueError("Cascade aircraft specification differs from the frozen one")
    model = spec.to_model()
    trim = cascade.trim_straight_flight(
        model,
        cascade.StraightFlightCondition(
            airspeed_m_s=declared["trim"]["airspeed_m_s"],
            altitude_m=declared["trim"]["altitude_m"],
        ),
    )
    if not trim.success:
        raise RuntimeError(f"Cascade calibration trim failed: {trim.message}")
    state = np.asarray(rigid_body_to_canonical(trim.state.rigid_body), dtype=float)
    command = np.asarray(cascade.control_to_array(trim.control), dtype=float)
    minimum = np.asarray(manifest["telemetry"]["command_minimum"], dtype=float)
    maximum = np.asarray(manifest["telemetry"]["command_maximum"], dtype=float)
    if np.any(command < minimum) or np.any(command > maximum):
        raise ValueError("trim lies outside the declared command box")
    return spec, model, trim, state, command


def _control_plant(manifest, spec=None, model=None):
    from glassbox.integrations.cascade import CascadePlant, CascadePlantConfig

    configuration = CascadePlantConfig(
        control_frequency_hz=manifest["plant"]["control_frequency_hz"]
    )
    if spec is None:
        return CascadePlant(configuration)
    return CascadePlant(configuration, spec=spec, model=model)


def control_recording(manifest, plant, trim, initial_state, initial_command, seed):
    """One calibration recording under the declared pilot, trim and excitation.

    The same protocol ``examples/cascade_refinement.py`` collects the structured
    belief's calibration with: the published X8 stabilizer at this command
    cadence, simulator-derived trim feedforward, and small setpoint and command
    perturbations from one seed. Every constant is read from the frozen
    manifest rather than written here.

    Returns the recording and the known additive excitation that went into each
    of its applied commands, which is a declared function of the manifest and
    this seed and is what the caller may hand the learner as a data fact.
    """
    import cascade
    import jax
    import jax.numpy as jnp
    from cascade.canonical import rigid_body_from_canonical
    from cascade.control import (
        GuidanceSetpoint,
        cascade_step,
        initial_cascade_state,
        skywalker_x8_controller,
    )

    from glassbox.core.data import Trajectory

    declared = manifest["calibration"]
    dt_s = manifest["plant"]["sample_interval_s"]
    setpoint_plan = declared["setpoint"]
    periods = declared["pilot_periods"]
    tuned = skywalker_x8_controller()
    pilot = tuned._replace(
        guidance=tuned.guidance._replace(
            pitch_trim=trim.decision[1], throttle_trim=trim.control.propeller[0]
        ),
        rate_period=periods["rate"],
        attitude_period=periods["attitude"],
        guidance_period=periods["guidance"],
    )
    environment = cascade.standard_environment()
    pilot_state = initial_cascade_state(pilot, trim.state, trim.control)

    @jax.jit
    def command_for(observed, pilot_state, setpoint):
        # The pilot reads only the rigid-body observation. Other fields are
        # unused by cascade_step; they are not sampled from the running plant.
        observed_aircraft = trim.state._replace(
            rigid_body=rigid_body_from_canonical(observed)
        )
        command, updated = cascade_step(
            pilot, pilot_state, setpoint, observed_aircraft, environment, dt_s
        )
        return cascade.control_to_array(command), updated

    minimum = np.asarray(manifest["telemetry"]["command_minimum"], dtype=float)
    maximum = np.asarray(manifest["telemetry"]["command_maximum"], dtype=float)
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=3)
    injected = calibration_excitation(manifest, seed)
    sample = plant.reset(initial_state, applied_control=initial_command)
    states, commands = [sample.state.copy()], []
    for index in range(round(declared["duration_s"] / dt_s)):
        elapsed = index * dt_s
        ramp = min(1.0, elapsed / setpoint_plan["ramp_s"])
        setpoint = GuidanceSetpoint(
            airspeed_m_s=jnp.asarray(
                setpoint_plan["airspeed_m_s"]
                + ramp
                * setpoint_plan["airspeed_amplitude_m_s"]
                * np.sin(setpoint_plan["airspeed_rate_rad_s"] * elapsed + phases[0])
            ),
            altitude_m=jnp.asarray(
                setpoint_plan["altitude_m"]
                + ramp
                * setpoint_plan["altitude_amplitude_m"]
                * np.sin(setpoint_plan["altitude_rate_rad_s"] * elapsed + phases[1])
            ),
            heading_rad=jnp.asarray(
                ramp
                * setpoint_plan["heading_amplitude_rad"]
                * np.sin(setpoint_plan["heading_rate_rad_s"] * elapsed + phases[2])
            ),
        )
        raw, pilot_state = command_for(sample.state, pilot_state, setpoint)
        command = np.clip(
            np.asarray(raw) + np.r_[0.0, initial_command[1:]] + injected[index],
            minimum,
            maximum,
        )
        sample = plant.step(command)
        if not np.isfinite(sample.state).all():
            raise RuntimeError(
                f"nonfinite calibration observation at seed {seed}, interval {index}"
            )
        states.append(sample.state.copy())
        commands.append(command.copy())
    return (
        Trajectory(
            time_s=np.arange(len(states)) * dt_s,
            states=np.asarray(states),
            controls=np.asarray(commands),
            control_prefix=initial_command[None],
            spec=control_telemetry_spec(manifest),
            labels={"source_group": f"cascade-calibration-{seed}"},
            provenance={
                "plant": "cascade.skywalker_x8",
                "seed": seed,
                "calibration_pilot": declared["pilot"],
                "command_policy": declared["command_policy"],
                "initial_history_assumption": declared["initial_history_assumption"],
            },
        ),
        injected,
    )


def control_collection(manifest, loaded, excitations=None):
    """Adapt the calibration recordings into the learner's own channels.

    The same fifteen-channel contract and the same segment adapter the platform
    tier uses. The control manifest declares the channels, and a recording that
    adapts to anything else fails closed here rather than later.

    ``excitations`` maps each recording's name to the exogenous component the
    caller injected into its applied commands. It is declared for every
    recording of a collection or for none of them, which is the recording
    contract's own rule, and nothing downstream of here reads it.
    """
    from .learned_plan import OBSERVED_CHANNELS as PLAN_CHANNELS

    declared = tuple(manifest["telemetry"]["observed_channels"])
    if declared != OBSERVED_CHANNELS or declared != PLAN_CHANNELS:
        raise ValueError("the manifest's observed channels are not the one contract")
    dt_s = manifest["plant"]["sample_interval_s"]
    channels = command_channels(loaded[0][1].spec)
    excitations = {} if excitations is None else dict(excitations)
    if excitations and set(excitations) != {name for name, _ in loaded}:
        raise ValueError(
            "every recording of a collection declares its excitation, or none does"
        )
    segments = []
    for name, trajectory in loaded:
        if command_channels(trajectory.spec) != channels:
            raise ValueError("calibration recordings declare different commands")
        segments.extend(
            trajectory_segments(
                name,
                trajectory,
                dt_s=dt_s,
                tolerance_fraction=1e-6,
                excitation=excitations.get(name),
            )
        )
    return SequenceCollection(
        tuple(segments),
        configuration_id=manifest["telemetry"]["configuration_id"],
        state_channels=OBSERVED_CHANNELS,
        input_channels=channels,
    )


# --- the control tier: the two arms behind one loop -------------------------


class _StructuredArm:
    """The frozen structured belief, driving the loop exactly as the example does.

    Its actuator state is reconstructed every interval from the commands the
    loop actually applied, which is the command history the example keeps, and
    the horizon is whatever the belief's own forecast-error evidence supports.
    """

    name = "structured"

    def __init__(self, manifest, belief):
        from dataclasses import replace as dataclass_replace

        from glassbox.control.fitted import NMPCController, default_solver_policy

        declared = manifest["controller"]
        self.belief = belief
        self.controller = NMPCController(
            belief.model,
            policy=dataclass_replace(
                default_solver_policy(belief),
                maximum_iterations=declared["maximum_iterations"],
                allow_unresolved_parameters=declared["allow_unresolved_parameters"],
            ),
        )
        self.policy = self.controller.plan.policy
        self._history = None

    @property
    def prediction_steps(self):
        return self.controller.prediction_steps

    @property
    def ready(self):
        return True

    def reset(self, initial_state, initial_command):
        # The plant is reset at a constant-command actuator equilibrium and the
        # manifest declares that; this is the example's own initial history.
        self._history = deque(
            [np.asarray(initial_command, dtype=float).copy()] * _CONTROL_HISTORY_STEPS,
            maxlen=_CONTROL_HISTORY_STEPS,
        )

    def observe(self, state):
        return None

    def command_applied(self, command):
        self._history.append(np.asarray(command, dtype=float).copy())

    def solve(self, state, reference, previous_command, **keywords):
        latent = self.controller.model.initial_latent_state(np.asarray(self._history))
        return self.controller.solve(
            state, reference, previous_command, latent_state=latent, **keywords
        )

    def summary(self):
        policy = self.policy
        return dict(
            arm=self.name,
            horizon_steps=policy.horizon_steps,
            horizon_s=self.controller.prediction_horizon_s,
            block_count=policy.block_count,
            maximum_iterations=policy.maximum_iterations,
            allow_unresolved_parameters=policy.allow_unresolved_parameters,
            uncertainty_available=bool(self.controller.plan.uncertainty_available),
            uncertainty_complete=bool(self.controller.plan.uncertainty_complete),
            command_history_steps=_CONTROL_HISTORY_STEPS,
            compile_signature=self.controller.plan.compile_signature,
            meaning=(
                "the frozen structured arm, planning over the fitted mean under "
                "the seam's explicit no-evidence override"
            ),
        )


class _GenericArm:
    """The generic learner behind the same loop, carrying its own history."""

    name = "generic"

    def __init__(self, manifest, learned):
        from dataclasses import replace as dataclass_replace

        from glassbox.control.plan import SafetyEnvelope, TrackingTolerances

        from .learned_plan import LearnedPlanController, fitted_solver_policy

        declared = manifest["controller"]
        telemetry = manifest["telemetry"]
        self.learned = learned
        self.controller = LearnedPlanController(
            learned,
            TrackingTolerances.for_platform(declared["tolerances_platform"]),
            SafetyEnvelope(),
            command_minimum=telemetry["command_minimum"],
            command_maximum=telemetry["command_maximum"],
            policy=dataclass_replace(
                fitted_solver_policy(learned),
                maximum_iterations=declared["maximum_iterations"],
                allow_unresolved_parameters=declared["allow_unresolved_parameters"],
            ),
        )
        self.policy = self.controller.policy

    @property
    def prediction_steps(self):
        return self.controller.prediction_steps

    @property
    def ready(self):
        return self.controller.ready

    def reset(self, initial_state, initial_command):
        self.controller.reset()

    def observe(self, state):
        self.controller.observe(state)

    def command_applied(self, command):
        self.controller.command_applied(command)

    def solve(self, state, reference, previous_command, **keywords):
        return self.controller.solve(state, reference, previous_command, **keywords)

    def summary(self):
        policy = self.policy
        plan = self.controller.plan
        return dict(
            arm=self.name,
            horizon_steps=policy.horizon_steps,
            horizon_s=self.controller.prediction_horizon_s,
            block_count=policy.block_count,
            maximum_iterations=policy.maximum_iterations,
            allow_unresolved_parameters=policy.allow_unresolved_parameters,
            uncertainty_available=bool(plan.uncertainty_available),
            uncertainty_complete=bool(plan.uncertainty_complete),
            context_steps=plan.context_steps,
            delay_steps=plan.delay_steps,
            memory_size=plan.memory_size,
            required_observations=self.controller.required_observations,
            envelope_nominal_coverage=plan.nominal_coverage,
            maximum_tangent_standard_deviation=float(
                np.max(
                    np.sqrt(
                        np.diagonal(
                            np.asarray(plan.values.forecast_error_covariance),
                            axis1=-2,
                            axis2=-1,
                        )
                    )
                )
            ),
            compile_signature=plan.compile_signature,
            meaning=(
                "the generic learner at its own fitted horizon, charging the "
                "seam's two robustness terms with its own measured forecast-error "
                "envelope and still running under the explicit no-evidence "
                "override because it resolves no parameter direction; validity "
                "utilization is zero because no support envelope is declared, "
                "not because one was checked"
            ),
        )


_CONTROL_HISTORY_STEPS = 20
"""Applied commands the structured arm reconstructs its actuator state from.

The example's own ``HISTORY_STEPS``. It is a property of that arm's actuator
model rather than of the trial, so it is not a manifest fact.
"""


def _control_arm(manifest, name, artifacts):
    if name == "structured":
        return _StructuredArm(manifest, artifacts["structured"])
    if name == "generic":
        return _GenericArm(manifest, artifacts["generic"])
    raise ValueError(f"undeclared control arm: {name}")


def _control_prewarm(arm, manifest, warmup, reference_fn):
    """Compile every kernel the timed loop will use, on real recorded data.

    Compilation must never happen after the clock starts, and both arms pay it
    the same way the example does: on a calibration recording, with the results
    discarded. The generic arm also walks that recording one observation at a
    time, because its consumed context grows by one until it is full and each
    length is its own compiled shape.
    """
    from glassbox.control.plan import ReferenceTrajectory

    dt_s = manifest["trial"]["sample_interval_s"]
    states = np.asarray(warmup.states, dtype=float)
    commands = np.asarray(warmup.controls, dtype=float)
    arm.reset(states[0], commands[0])
    warm_start = None
    for index in range(min(len(commands), _CONTROL_HISTORY_STEPS + 2)):
        arm.observe(states[index])
        if arm.ready:
            future = (index + np.arange(arm.prediction_steps + 1)) * dt_s
            reference = ReferenceTrajectory(reference_fn(future))
            result = arm.solve(
                states[index], reference, commands[index], warm_start=warm_start
            )
            warm_start = result.warm_start
            np.asarray(result.predicted_states)
            np.asarray(result.command)
        arm.command_applied(commands[index])
    arm.reset(states[0], commands[0])


def _control_trial(manifest, arm, plant, reference_fn, anchor_state, directory):
    """One tracking trial in simulated time: one arm, one plant, one reference.

    ``anchor_state`` is the unperturbed trim state the reference is built from.
    The plant starts from its own declared perturbation of it, and both are
    saved, so a replay can rebuild the reference and recheck the perturbation.

    Nothing a clock measured may reach the trajectory, which is what the live
    tier already does and what ``control-v4`` adopted. Interval ``k`` is the
    state at ``k`` times the sample interval: the loop is not paced, the solver
    is given no deadline and therefore never falls back for want of time, and
    the command it solved is the command the plant is stepped with. Solve
    times, interval times and the count of solves over the sample interval are
    still measured, and are recorded in a separate artifact and a separate
    block of the row that nothing reads back.
    """
    from glassbox.control.plan import ReferenceTrajectory
    from glassbox.core.metrics import state_rmse_metrics

    directory.mkdir(parents=True, exist_ok=True)
    trial = manifest["trial"]
    dt_s = trial["sample_interval_s"]
    requested = trial["intervals"]
    deadline_s = trial["solve_deadline_s"]
    state = plant.initial_state.copy()
    previous = plant.initial_command.copy()
    arm.reset(state, previous)
    observed = [state.copy()]
    applied, tick_times, solve_times = [], [], []
    solver_used, fallbacks, statuses, assessed = [], [], [], []
    warm_start = None
    failure = None
    started = time.monotonic()
    for index in range(requested):
        tick = time.monotonic()
        arm.observe(state)
        if arm.ready:
            future = (index + np.arange(arm.prediction_steps + 1)) * dt_s
            reference = ReferenceTrajectory(reference_fn(future))
            # No deadline: a solve cut short by a busy host would put the wall
            # clock into the commands, and this tier's trajectory is a function
            # of the plant, the models and the reference alone.
            result = arm.solve(state, reference, previous, warm_start=warm_start)
            command = np.asarray(result.command, dtype=float)
            warm_start = result.warm_start
            solve_times.append(float(result.diagnostics.solve_time_s))
            solver_used.append(True)
            fallbacks.append(bool(result.used_fallback))
            statuses.append(str(result.status))
            # ``deadline_met`` is None exactly when no deadline was assessed,
            # so this array is the run's own record of having been given none.
            assessed.append(result.deadline_met is not None)
        else:
            # No forecast exists yet: the loop holds the command it is already
            # applying rather than fabricating the history one would need.
            command = previous.copy()
            solve_times.append(0.0)
            solver_used.append(False)
            fallbacks.append(False)
            statuses.append("model_not_ready")
            assessed.append(False)
        next_state = np.asarray(plant.advance(command), dtype=float)
        if not np.isfinite(next_state).all():
            failure = "nonfinite plant state"
            break
        arm.command_applied(command)
        observed.append(next_state.copy())
        applied.append(command.copy())
        previous, state = command, next_state
        tick_times.append(time.monotonic() - tick)
    elapsed_s = time.monotonic() - started

    states_array = np.asarray(observed)
    commands_array = (
        np.asarray(applied)
        if applied
        else np.zeros((0, len(plant.initial_command)), dtype=float)
    )
    times = np.arange(len(states_array)) * dt_s
    reference_states = reference_fn(times)
    metrics = (
        state_rmse_metrics(states_array[1:], reference_states[1:])
        if len(commands_array)
        else None
    )
    minimum = np.asarray(manifest["telemetry"]["command_minimum"], dtype=float)
    maximum = np.asarray(manifest["telemetry"]["command_maximum"], dtype=float)
    bound_violation = (
        max(
            0.0,
            float(np.max(minimum - commands_array)),
            float(np.max(commands_array - maximum)),
        )
        if len(commands_array)
        else 0.0
    )
    # The trajectory and the clock live in different files. Everything in
    # tracking.npz is a function of the plant, the models and the reference, so
    # two runs of this tier produce it byte for byte; nothing in timing.npz is,
    # and nothing reads it back.
    np.savez_compressed(
        directory / "tracking.npz",
        time_s=times,
        states=states_array,
        reference_states=reference_states,
        commands=commands_array,
        solver_used=np.asarray(solver_used, dtype=bool),
        used_fallback=np.asarray(fallbacks, dtype=bool),
        initial_state=np.asarray(plant.initial_state, dtype=float),
        reference_anchor_state=np.asarray(anchor_state, dtype=float),
    )
    np.savez_compressed(
        directory / "timing.npz",
        tick_times_s=np.asarray(tick_times, dtype=float),
        solve_times_s=np.asarray(solve_times, dtype=float),
        deadline_assessed=np.asarray(assessed, dtype=bool),
    )
    terminated = failure is not None or len(commands_array) != requested
    row = dict(
        arm=arm.name,
        completed_intervals=len(commands_array),
        requested_intervals=requested,
        terminated=bool(terminated),
        failure=failure,
        tracking_rmse=metrics,
        pass_criterion=control_pass_criterion(states_array, anchor_state, manifest),
        model_not_ready_intervals=int(statuses.count("model_not_ready")),
        fallback_count=int(sum(fallbacks)),
        solver_statuses={
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        maximum_command_bound_violation=bound_violation,
        wall=simulated_time_wall(
            meaning=SIMULATED_TIME_MEANING,
            dt_s=dt_s,
            deadline_s=deadline_s,
            tick_times=tick_times,
            solve_times=solve_times,
            elapsed_s=elapsed_s,
            solve_deadline_applied=False,
            deadline_assessed_intervals=int(sum(assessed)),
        ),
        controller=arm.summary(),
        files=_files(directory, ["tracking.npz", "timing.npz"]),
    )
    write(directory / "trial.json", row)
    return row


def control_evidence_arrays(manifest, reserved):
    """The reserved recording's forecast origins, cut the way the evidence tier reads.

    The same fifteen-channel adapter the arms are fitted through, then every
    origin that carries the recipe's whole consumed context inside one segment
    and the whole horizon after it. The reserved recording is never fitted in
    any role, so this holds nothing further back.
    """
    collection = control_collection(manifest, reserved)
    steps = steps_for(manifest["plant"]["sample_interval_s"])
    return evaluation_rows(collection, CONTROL_EVIDENCE_PLAN, steps)


def control_evidence(learned, arrays):
    """Forecast the reserved recording and measure the envelope's coverage on it."""
    prediction = np.asarray(
        learned.predict(
            arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
        )
    )
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite forecast on the reserved recording")
    half_width = envelope_rows(learned, prediction)
    return prediction, half_width, control_coverage(prediction, arrays, half_width)


def control_coverage(prediction, arrays, half_width):
    """Coverage on the reserved rows, one entry per reserved recording."""
    groups = evidence_groups("rigid_body_15", 15)
    ids = np.asarray(arrays["recording_ids"])
    return {
        str(name): envelope_coverage(
            np.asarray(prediction)[ids == name],
            np.asarray(arrays["targets"])[ids == name],
            np.asarray(half_width)[ids == name],
            groups,
        )
        for name in sorted(set(ids.tolist()))
    }


def control_evidence_table(coverage):
    """One reserved recording per declared evidence case, in that decision's shape."""
    return [
        dict(case=name, regime=CONTROL_EVIDENCE_REGIME, **measured)
        for name, measured in sorted(coverage.items())
    ]


def _control_calibrate(manifest, output):
    """Collect the frozen calibration recordings and fit both arms on them."""
    from glassbox.belief.belief_io import save_dynamics_belief
    from glassbox.core.data import save_trajectory_npz, trajectory_content_digest
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit

    declared = manifest["calibration"]
    spec, model, trim, initial_state, initial_command = control_fixture(manifest)
    plant = _control_plant(manifest, spec, model)
    recordings, names, injected = {}, [], {}
    for seed in declared["seeds"]:
        print(json.dumps(dict(collecting=f"recording-{seed}")), flush=True)
        flight, excited = control_recording(
            manifest, plant, trim, initial_state, initial_command, seed
        )
        name = f"recording-{seed}"
        save_trajectory_npz(flight, output / f"{name}.npz")
        recordings[seed] = flight
        injected[name] = excited
        names.append(f"{name}.npz")
    # The declared excitation is a data fact about these recordings, saved
    # beside them so a replay can recompute it and reject an altered one.
    np.savez_compressed(output / "calibration-excitation.npz", **injected)
    names.append("calibration-excitation.npz")
    digests = {
        f"recording-{seed}": trajectory_content_digest(flight)
        for seed, flight in recordings.items()
    }
    if len(set(digests.values())) != len(digests):
        raise ValueError("calibration recordings are not distinct")
    training = [(f"recording-{s}", recordings[s]) for s in declared["training_seeds"]]
    reserved = [f"recording-{s}" for s in declared["reserved_seeds"]]

    excitation = control_excitation(
        manifest, [(f"recording-{s}", flight) for s, flight in recordings.items()]
    )
    short = control_excitation_shortfall(
        manifest, excitation, [name for name, _ in training]
    )
    excitation["training_shortfall"] = short
    print(json.dumps(dict(command_excitation=excitation["fraction"])), flush=True)
    if short:
        # The calibration is the evidence both arms are fitted from. A channel
        # the recordings never move is a channel neither arm can be asked to
        # use, so this fails before either fit rather than after both.
        raise ValueError(f"calibration command excitation falls short: {short}")

    arm = manifest["arms"]["structured"]
    started = time.perf_counter()
    outcome = structured_fit(
        [flight for _, flight in training],
        FitSpec(
            holdout=Holdout.by_group(),
            steps=arm["optimization_steps"],
            horizons_s=tuple(arm["training_horizons_s"]),
            evaluation_horizons_s=tuple(arm["evaluation_horizons_s"]),
        ),
    )
    structured_wall = time.perf_counter() - started
    save_dynamics_belief(outcome.belief, output / "structured.json")
    write(output / "structured_report.json", outcome.report)

    # The generic arm's recordings carry what the calibration injected into
    # their commands; the structured fit above is the same fit it always was
    # and never sees it. The recipe ignores it too and records that it was
    # declared, which is why both arms' numbers are the ones control-v4 measured.
    started = time.perf_counter()
    learned = fit(
        control_collection(
            manifest,
            training,
            {name: injected[name] for name, _ in training},
        )
    )
    generic_wall = time.perf_counter() - started
    learned.save(output / "generic.npz")
    if set(learned.report["training"]) | set(learned.report["development"]) != {
        name for name, _ in training
    }:
        raise ValueError("the generic fit did not read the calibration recordings")
    if learned.report.get("excitation_declared") is not True:
        raise ValueError("the generic fit was not told what the calibration injected")

    reserved_loaded = [(name, recordings[int(name.split("-")[1])]) for name in reserved]
    evidence_arrays = control_evidence_arrays(manifest, reserved_loaded)
    if set(evidence_arrays["recording_ids"].tolist()) & {name for name, _ in training}:
        raise ValueError("the reserved evidence rows overlap the fitted recordings")
    prediction, half_width, coverage = control_evidence(learned, evidence_arrays)
    np.savez_compressed(
        output / "evidence.npz",
        **evidence_arrays,
        prediction=prediction,
        envelope_half_width=half_width,
    )

    calibration = dict(
        recordings=digests,
        training=[name for name, _ in training],
        reserved=reserved,
        evidence=coverage,
        initial_state=initial_state.tolist(),
        initial_command=initial_command.tolist(),
        trim_balance_residual=np.asarray(trim.residual).tolist(),
        cascade=dict(
            spec_hash=manifest["plant"]["spec_hash"],
            source_revision=manifest["plant"]["source_revision"],
            version=manifest["plant"]["version"],
        ),
        command_excitation=excitation,
        declared_excitation=dict(
            form=manifest["calibration"]["excitation"]["declared_form"],
            arrays="calibration-excitation.npz",
            standard_deviation_fraction=learned.report[
                "excitation_standard_deviation_fraction"
            ],
        ),
        structured_fit_wall_seconds=structured_wall,
        generic_fit_wall_seconds=generic_wall,
        generic_fingerprint=learned.fingerprint(),
        generic_report=learned.report,
        files=_files(
            output,
            names
            + [
                "structured.json",
                "structured_report.json",
                "generic.npz",
                "evidence.npz",
            ],
        ),
    )
    write(output / "calibration.json", calibration)
    return (
        calibration,
        dict(structured=outcome.belief, generic=learned),
        recordings[declared["training_seeds"][0]],
        initial_state,
        initial_command,
    )


def control_decide(manifest, rows, reference=None, reference_sha256=None):
    """Every gate, evaluated from recorded trial metrics alone. Anything unclear fails.

    The same semantics the platform tier decides by. A run is accepted when no
    metric regressed past its reference value times one plus the manifest's
    relative tolerance plus its absolute one, the rule holds on every case the
    reference already meets it on, and nothing structural failed: a trial
    missing, duplicated, undeclared, terminated, short of its declared
    intervals, or carrying a metric that is not a finite number. The rule is
    reported on every trial and metric either way, and ``rule_met`` says
    whether it holds everywhere.

    ``reference_sha256`` identifies the reference file the metrics were
    compared against. It is recorded in the decision so a replay can check that
    it compared the same bytes, and it is not otherwise used here.
    """
    repetitions = manifest["trial"]["repetitions"]
    expected = {(index, arm) for index in range(repetitions) for arm in CONTROL_ARMS}
    keys = [(row.get("repetition"), row.get("arm")) for row in rows]
    breaches, rule_breaches, regressions, summary = [], [], [], {}
    criteria = {}
    relative = manifest["reference"]["relative_tolerance"]
    absolute = manifest["reference"]["absolute_tolerance"]
    for index, arm in sorted(expected - set(keys)):
        breaches.append(dict(trial=f"{index}-{arm}", gate="trial_present"))
    for index, arm in sorted({key for key in keys if keys.count(key) > 1}):
        breaches.append(dict(trial=f"{index}-{arm}", gate="trial_unique"))
    for key in sorted(set(keys) - expected, key=repr):
        breaches.append(dict(trial=f"{key[0]}-{key[1]}", gate="trial_declared"))
    measured = {}
    for row in rows:
        key = (row.get("repetition"), row.get("arm"))
        if key not in expected or keys.count(key) > 1:
            continue
        name = f"{key[0]}-{key[1]}"
        # Reported on every declared trial, including one that did not finish:
        # the page counts an unexecuted interval as outside tolerance rather
        # than discarding the trial.
        criteria[name] = row.get("pass_criterion")
        if row.get("terminated") is not False:
            breaches.append(
                dict(
                    trial=name,
                    gate="trial_complete",
                    completed=row.get("completed_intervals"),
                    requested=row.get("requested_intervals"),
                    failure=row.get("failure"),
                )
            )
            continue
        if row.get("completed_intervals") != manifest["trial"]["intervals"]:
            breaches.append(dict(trial=name, gate="declared_intervals"))
            continue
        recorded = row.get("tracking_rmse")
        values = {}
        for metric in CONTROL_METRICS:
            value = _number(
                recorded.get(metric) if isinstance(recorded, dict) else None
            )
            if value is None or value < 0:
                breaches.append(
                    dict(trial=name, metric=metric, gate="finite_rmse", value=value)
                )
            values[metric] = value
        measured[key] = values
    for index in sorted({key[0] for key in measured}):
        generic = measured.get((index, "generic"))
        structured = measured.get((index, "structured"))
        if generic is None or structured is None:
            continue
        summary[str(index)] = {}
        for metric in CONTROL_METRICS:
            base = _reference_value(
                reference, "tracking_rmse", str(index), "generic", metric
            )
            meets = _reference_meets(base, [structured[metric]])
            summary[str(index)][metric] = dict(
                generic=generic[metric],
                structured=structured[metric],
                reference=base,
                reference_meets_rule=meets,
            )
            if reference is not None:
                if base is None:
                    regressions.append(
                        dict(
                            trial=f"{index}-generic",
                            metric=metric,
                            gate="reference_present",
                        )
                    )
                else:
                    limit = base * (1 + relative) + absolute
                    if generic[metric] is None or generic[metric] > limit:
                        regressions.append(
                            dict(
                                trial=f"{index}-generic",
                                metric=metric,
                                gate="reference_tracking_rmse",
                                value=generic[metric],
                                reference=base,
                                limit=limit,
                            )
                        )
            if generic[metric] is None or structured[metric] is None:
                continue
            if generic[metric] > structured[metric]:
                rule_breaches.append(
                    dict(
                        trial=f"{index}-generic",
                        metric=metric,
                        gate="structured_arm_rmse",
                        value=generic[metric],
                        limit=structured[metric],
                        gating=meets is True,
                    )
                )
    regressions.sort(key=lambda entry: (entry["trial"], entry["metric"]))
    enforced = bool(manifest["decision"]["enforced"])
    gating = [breach for breach in rule_breaches if breach["gating"]]
    rule_met = not breaches and not rule_breaches
    accepted = not breaches and not regressions and (not enforced or not gating)
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        rule_met=rule_met,
        rule_enforced=enforced,
        gating_rule_breaches=len(gating),
        rule=manifest["decision"]["rule"],
        gates_from=manifest["decision"]["gates_from"],
        trials=len(rows),
        gate_breaches=breaches,
        rule_breaches=rule_breaches,
        reference_regressions=regressions,
        reference_compared=reference is not None,
        reference_sha256=reference_sha256,
        tracking_rmse=summary,
        pass_criterion=criteria,
        pass_criterion_meaning=manifest["metrics"]["pass_criterion"]["meaning"],
        meaning=manifest["decision"]["meaning"],
    )


def control(manifest_path, output):
    """Run the frozen control trial set once, both arms, and write the decision."""
    import jax

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = frozen_control_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
    evidence_manifest, evidence_reference, evidence_digest = evidence_setup(output)
    reference_path = manifest_path.parent / manifest["reference"]["file"]
    reference, reference_digest = None, None
    if reference_path.exists():
        shutil.copyfile(reference_path, output / "reference.json")
        reference, reference_digest = read(reference_path), sha256(reference_path)
    write(
        output / "environment.json",
        dict(
            python=sys.version,
            platform=platform_module.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64="fits only",
        ),
    )
    started = time.perf_counter()
    with jax.enable_x64(True):
        calibration, artifacts, warmup, initial_state, initial_command = (
            _control_calibrate(manifest, output)
        )
    declared = manifest["tracking_reference"]
    trial = manifest["trial"]
    with jax.enable_x64(True):
        starts = [
            control_initial_state(manifest, initial_state, seed)
            for seed in trial["initial_state_seeds"]
        ]

    def reference_fn(times):
        return control_reference(initial_state, times, declared)

    rows = []
    for repetition in range(trial["repetitions"]):
        order = trial["arm_order"][repetition % len(trial["arm_order"])]
        start = starts[repetition]
        for name in order:
            print(json.dumps(dict(tracking=f"{repetition}-{name}")), flush=True)
            arm = _control_arm(manifest, name, artifacts)
            _control_prewarm(arm, manifest, warmup, reference_fn)
            plant = _control_tracking_plant(manifest, start, initial_command)
            trial_started = time.perf_counter()
            row = _control_trial(
                manifest,
                arm,
                plant,
                reference_fn,
                initial_state,
                output / f"trial-{repetition}" / name,
            )
            row["repetition"] = repetition
            row["initial_state_seed"] = trial["initial_state_seeds"][repetition]
            row["wall"]["trial_seconds"] = time.perf_counter() - trial_started
            row["directory"] = f"trial-{repetition}/{name}"
            write(output / row["directory"] / "trial.json", row)
            rows.append(row)
            write(output / "results.json", rows)
            print(
                json.dumps(
                    dict(
                        trial=f"{repetition}-{name}",
                        tracking_rmse=row["tracking_rmse"],
                        pass_criterion=row["pass_criterion"],
                        terminated=row["terminated"],
                        wall=row["wall"]["elapsed_seconds"],
                    )
                ),
                flush=True,
            )
    decision = control_decide(manifest, rows, reference, reference_digest)
    with_evidence(
        decision,
        evidence_decide(
            evidence_manifest,
            "control",
            control_evidence_table(calibration["evidence"]),
            {(name, CONTROL_EVIDENCE_REGIME) for name in calibration["reserved"]},
            evidence_reference,
            evidence_digest,
        ),
    )
    decision["wall_seconds"] = time.perf_counter() - started
    decision["calibration"] = dict(
        command_excitation=calibration["command_excitation"]["fraction"],
        structured_fit_wall_seconds=calibration["structured_fit_wall_seconds"],
        generic_fit_wall_seconds=calibration["generic_fit_wall_seconds"],
        generic_fingerprint=calibration["generic_fingerprint"],
        reserved=calibration["reserved"],
    )
    write(output / "decision.json", decision)
    return decision


def _control_tracking_plant(manifest, initial_state, initial_command):
    """A freshly reset, prewarmed Cascade plant behind the consumer-side seam.

    The plant is stepped once at the initial command and reset again before the
    trial clock starts, so no compilation lands inside a timed interval. Only
    this callback holds simulator internals.
    """
    from types import SimpleNamespace

    plant = _control_plant(manifest)
    plant.reset(initial_state, applied_control=initial_command)
    plant.step(initial_command)
    plant.reset(initial_state, applied_control=initial_command)
    return SimpleNamespace(
        initial_state=np.asarray(initial_state, dtype=float).copy(),
        initial_command=np.asarray(initial_command, dtype=float).copy(),
        advance=lambda command: plant.step(command).state,
        source="cascade.skywalker_x8",
    )


# --- the control tier: verifying --------------------------------------------


def _verify_calibration(directory, manifest):
    """Recheck a saved calibration: hashes, recordings, excitation, both arms.

    The control and live tiers collect the same calibration with the same
    constants and fit the same two arms on it, so they check it the same way.
    Every recorded artifact hash is recomputed, every recording's content digest
    is recomputed from the recording itself, the excitation is remeasured and
    held to the declared minimum, and the generic model's fingerprint is
    checked. The reserved recording's evidence rows are rebuilt from the
    recording rather than believed: the saved arrays have to be the declared cut
    of the trajectory whose content digest was just checked, the prediction has
    to replay through the independent NumPy recurrence, and the half-widths have
    to be the model's own envelope.
    """
    from glassbox.core.data import load_trajectory_npz, trajectory_content_digest

    directory = Path(directory)
    calibration = read(directory / "calibration.json")
    for name, digest in calibration["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"altered artifact: {name}")
    for name, digest in calibration["recordings"].items():
        fresh = trajectory_content_digest(
            load_trajectory_npz(directory / f"{name}.npz")
        )
        if fresh != digest:
            raise ValueError(f"altered calibration recording: {name}")
    if set(calibration["training"]) & set(calibration["reserved"]):
        raise ValueError("a reserved recording was also used for fitting")
    fresh = control_excitation(
        manifest,
        [
            (name, load_trajectory_npz(directory / f"{name}.npz"))
            for name in sorted(calibration["recordings"])
        ],
    )
    saved = calibration["command_excitation"]
    np.testing.assert_allclose(
        fresh["declared_range"], saved["declared_range"], rtol=0, atol=0
    )
    if sorted(fresh["fraction"]) != sorted(saved["fraction"]):
        raise ValueError("recorded command excitation covers different recordings")
    for name, values in fresh["fraction"].items():
        np.testing.assert_allclose(values, saved["fraction"][name], **SCORE_TOLERANCE)
        np.testing.assert_allclose(
            fresh["standard_deviation"][name],
            saved["standard_deviation"][name],
            **SCORE_TOLERANCE,
        )
    if control_excitation_shortfall(manifest, fresh, calibration["training"]):
        raise ValueError("the saved calibration does not meet the declared excitation")

    # The declared excitation is recomputed from the manifest and each
    # recording's own seed, not believed: an altered table is rejected here and
    # by the artifact hash above, and the fractions the fit reported have to be
    # the ones these arrays and these commands actually give.
    injected = {}
    with np.load(directory / "calibration-excitation.npz", allow_pickle=False) as data:
        if sorted(data.files) != sorted(calibration["recordings"]):
            raise ValueError("the saved excitation covers different recordings")
        for name in data.files:
            rebuilt = calibration_excitation(manifest, int(name.split("-")[1]))
            if data[name].shape != rebuilt.shape or not np.array_equal(
                data[name], rebuilt
            ):
                raise ValueError(f"the saved declared excitation differs: {name}")
            injected[name] = data[name]

    learned = LearnedDynamics.load(directory / "generic.npz")
    if learned.fingerprint() != calibration["generic_fingerprint"]:
        raise ValueError("generic model fingerprint mismatch")
    training_loaded = [
        (name, load_trajectory_npz(directory / f"{name}.npz"))
        for name in calibration["training"]
    ]
    remeasured = excitation_fraction(
        control_collection(
            manifest,
            training_loaded,
            {name: injected[name] for name, _ in training_loaded},
        )
    )
    saved_fraction = calibration["declared_excitation"]["standard_deviation_fraction"]
    if learned.report.get("excitation_declared") is not True:
        raise ValueError("the saved generic fit does not declare its excitation")
    _same_fraction(
        remeasured,
        learned.report["excitation_standard_deviation_fraction"],
        "the saved generic fit",
        rtol=0,
        atol=0,
    )
    _same_fraction(remeasured, saved_fraction, "the saved calibration", rtol=0, atol=0)

    reserved_loaded = [
        (name, load_trajectory_npz(directory / f"{name}.npz"))
        for name in calibration["reserved"]
    ]
    rebuilt = control_evidence_arrays(manifest, reserved_loaded)
    with np.load(directory / "evidence.npz", allow_pickle=False) as data:
        for key, expected in rebuilt.items():
            if data[key].shape != np.shape(expected) or not np.array_equal(
                data[key], expected
            ):
                raise ValueError(f"saved reserved evidence rows differ: {key}")
        replayed = replay(
            learned._model,
            data["past_states"],
            data["past_inputs"],
            data["future_inputs"],
        )
        np.testing.assert_allclose(replayed, data["prediction"], **REPLAY_TOLERANCE)
        half_width = envelope_rows(learned, replayed)
        if not np.array_equal(half_width, data["envelope_half_width"]):
            raise ValueError("the saved reserved envelope is not the model's own")
        coverage = control_coverage(replayed, rebuilt, half_width)
    for name, measured in coverage.items():
        _same_coverage(measured, calibration["evidence"].get(name), name)
    return calibration, learned, coverage


def verify_control(directory, manifest, reference=None):
    """Recompute every control metric and the decision from saved arrays.

    This replay never reruns the plant and never reruns the solver: rerunning
    either would be a second measurement rather than a check of this one, and
    the run whose determinism the tier claims is the one that was saved. What
    it does check is everything the decision actually read, and that the run
    was computed in simulated time as far as its own artifacts can say: the
    recorded solve times are recomputed into the counts the row reports, and a
    row claiming that a deadline was assessed, or a solver status saying one
    expired, is rejected. Every recorded artifact hash is
    recomputed, including the calibration recordings and both fitted models, so
    an altered artifact is rejected. Every metric is recomputed from the saved
    per-interval tracking arrays with the library's own metric code. The
    reference rows are rebuilt from the saved initial state and the manifest's
    declared reference, so a run cannot score itself against a reference of its
    own invention. The manifest is anchored to its frozen digest, and the
    regression reference to the committed file, exactly as the other two tiers
    anchor theirs.
    """
    import jax

    from glassbox.core.metrics import state_rmse_metrics

    directory = Path(directory)
    calibration, _learned, coverage = _verify_calibration(directory, manifest)

    trial = manifest["trial"]
    dt_s = trial["sample_interval_s"]
    declared = manifest["tracking_reference"]
    rows = read(directory / "results.json")
    checked = 0
    for row in rows:
        case = directory / row["directory"]
        if read(case / "trial.json") != row:
            raise ValueError(f"trial result mismatch: {row['directory']}")
        for name, digest in row["files"].items():
            if sha256(case / name) != digest:
                raise ValueError(f"altered artifact: {row['directory']}/{name}")
        with np.load(case / "timing.npz", allow_pickle=False) as data:
            tick_times = data["tick_times_s"]
            solve_times = data["solve_times_s"]
            deadline_assessed = data["deadline_assessed"]
        with np.load(case / "tracking.npz", allow_pickle=False) as data:
            states = data["states"]
            commands = data["commands"]
            saved_reference = data["reference_states"]
            initial_state = data["initial_state"]
            anchor_state = data["reference_anchor_state"]
            times = data["time_s"]
            if len(states) != len(commands) + 1 or len(times) != len(states):
                raise ValueError(f"saved tracking arrays disagree: {row['directory']}")
            np.testing.assert_allclose(
                times, np.arange(len(states)) * dt_s, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                saved_reference,
                control_reference(anchor_state, times, declared),
                rtol=0,
                atol=1e-12,
            )
            # A run cannot invent where a trial started either: the perturbed
            # state is the declared draw from the repetition's own seed.
            with jax.enable_x64(True):
                np.testing.assert_allclose(
                    initial_state,
                    control_initial_state(
                        manifest, anchor_state, row["initial_state_seed"]
                    ),
                    rtol=0,
                    atol=1e-12,
                )
            np.testing.assert_allclose(states[0], initial_state, rtol=0, atol=1e-12)
            fresh = (
                state_rmse_metrics(states[1:], saved_reference[1:])
                if len(commands)
                else None
            )
            criterion = control_pass_criterion(states, anchor_state, manifest)
        if criterion != row["pass_criterion"]:
            raise ValueError(f"recomputed pass criterion differs: {row['directory']}")
        if (fresh is None) != (row["tracking_rmse"] is None):
            raise ValueError(f"tracking metric shape mismatch: {row['directory']}")
        if fresh is not None:
            if sorted(fresh) != sorted(row["tracking_rmse"]):
                raise ValueError(f"tracking metric set mismatch: {row['directory']}")
            for metric, value in fresh.items():
                np.testing.assert_allclose(
                    value, row["tracking_rmse"][metric], **SCORE_TOLERANCE
                )
            row["tracking_rmse"] = fresh
        wall = row["wall"]
        verify_simulated_time_wall(
            wall,
            dt_s=dt_s,
            deadline_s=trial["solve_deadline_s"],
            tick_times=tick_times,
            solve_times=solve_times,
            label=row["directory"],
        )
        # The manifest says the solver was given no deadline, and a run that
        # recorded otherwise is rejected rather than reported. Three saved
        # facts can say it: the flag the row asserts, the per-interval record
        # of whether a deadline was assessed at all, and the solver statuses.
        # None of them can prove a deadline was absent, but each of them
        # rejects a run that claims one decided something.
        if (
            wall.get("solve_deadline_applied") is not False
            or wall.get("deadline_assessed_intervals") != 0
            or int(np.sum(deadline_assessed)) != wall.get("deadline_assessed_intervals")
            or len(deadline_assessed) != len(solve_times)
        ):
            raise ValueError(
                f"the recorded solve times claim a deadline was assessed: "
                f"{row['directory']}"
            )
        if row.get("solver_statuses", {}).get("deadline_exceeded"):
            raise ValueError(f"a solve was cut short by a deadline: {row['directory']}")
        terminated = len(commands) != row["requested_intervals"]
        if (
            terminated != bool(row["terminated"])
            or len(commands) != (row["completed_intervals"])
        ):
            raise ValueError(f"recomputed completion differs: {row['directory']}")
        checked += 1
    anchor, digest = anchored_reference(
        directory, reference, COMMITTED_CONTROL_REFERENCE
    )
    decision = control_decide(manifest, rows, anchor, digest)
    evidence_manifest, evidence_anchored, evidence_digest = evidence_anchor(directory)
    evidence = evidence_decide(
        evidence_manifest,
        "control",
        control_evidence_table(coverage),
        {(name, CONTROL_EVIDENCE_REGIME) for name in calibration["reserved"]},
        evidence_anchored,
        evidence_digest,
    )
    saved = read(directory / "decision.json")
    _same_evidence(evidence, saved.get("evidence"), "control")
    with_evidence(decision, evidence)
    for key in (
        "manifest",
        "decision",
        "accepted",
        "rule_met",
        "rule_enforced",
        "gating_rule_breaches",
        "trials",
        "reference_compared",
        "reference_sha256",
    ):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed decision differs: {key}")
    for key in ("gate_breaches", "rule_breaches", "reference_regressions"):
        if len(decision[key]) != len(saved[key]):
            raise ValueError(f"replayed decision differs: {key}")
    return dict(
        tier="control",
        verified_trials=checked,
        decision=decision,
        meaning=(
            "A replay of saved evidence: recorded hashes, metrics recomputed from "
            "the saved per-interval tracking arrays, the reference rebuilt from "
            "the saved initial state, and the recorded host measurements "
            "recomputed from the saved solve and interval times so a run cannot "
            "claim a deadline decided something. The plant and the solver are "
            "not rerun, because rerunning either would be a second measurement "
            "rather than a check of this one."
        ),
    )


# --- the live improvement tier: refit on the flight, swap on held-out evidence

LIVE_MANIFEST_SHA256 = (
    "4d39b49536f7fee4fced8702ba9f931894497d4a48f97bdb83de1986efed5fd0"
)
"""Digest of the frozen live manifest this module is allowed to run."""

COMMITTED_LIVE_MANIFEST = COMMITTED_MANIFEST.parent / "live-v3.json"
"""The frozen live manifest in a source checkout."""

COMMITTED_LIVE_REFERENCE = COMMITTED_MANIFEST.parent / "live-reference.json"
"""Where ``verify`` looks for the committed live reference by default."""

LIVE_ARMS = ("adopting", "frozen")
"""The two arms of one repetition: one that may swap and one that never does."""

LIVE_METRICS = ("position_rmse_m", "attitude_rmse_deg")
"""The two tracking metrics the declared rule compares, segment against segment."""

LIVE_SEGMENTS = ("before", "after", "whole")
"""The interval ranges a trial's tracking arrays are scored over."""

LIVE_REPORTED_METRICS = LIVE_METRICS + (
    "velocity_rmse_m_s",
    "angular_velocity_rmse_rad_s",
)


def frozen_live_manifest(path):
    """Load the live manifest only when its bytes match the frozen digest."""
    if sha256(path) != LIVE_MANIFEST_SHA256:
        raise ValueError("manifest digest differs from the frozen harness contract")
    manifest = read(path)
    declared_plan(manifest)
    steps = steps_for(manifest["information_budget"]["sample_interval_s"])
    budget = manifest["information_budget"]
    if (steps["history"], steps["delay"], steps["horizon"]) != (
        budget["context_steps"],
        budget["delay_steps"],
        budget["horizon_steps"],
    ):
        raise ValueError("manifest information budget differs from the recipe")
    trial = manifest["trial"]
    if trial["intervals"] != round(trial["duration_s"] / trial["sample_interval_s"]):
        raise ValueError("manifest trial duration and interval count disagree")
    transport = manifest["live"]["transport"]
    block = transport["block_steps"]
    if block * trial["sample_interval_s"] != transport["block_duration_s"]:
        raise ValueError("manifest block size and block duration disagree")
    # The buffer emits nothing until it holds the declared command history, so
    # the first block starts there and the declared count has to fit after it.
    if transport["first_block_start_interval"] != transport["command_history_steps"]:
        raise ValueError("the first block does not start where the history fills")
    if (
        transport["first_block_start_interval"] + block * transport["blocks_per_trial"]
        > trial["intervals"]
    ):
        raise ValueError("the manifest declares more whole blocks than the trial has")
    # The recipe refuses a new recording that cannot supply three whole windows.
    if block + 1 - steps["history"] - steps["horizon"] < 3:
        raise ValueError("a declared block cannot carry three complete recipe windows")
    # The offset is what makes the swap interval a declared constant rather than
    # a measurement of how busy the host was, so it has to be one.
    offset = transport["offer_release_offset_intervals"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        raise ValueError("the offer release offset must be a positive interval count")
    if transport["drive"] != "synchronous":
        raise ValueError("this tier's worker is driven synchronously, or not at all")
    if manifest["trial"]["solver_deadline_applied"] is not False:
        raise ValueError("this tier's solver is given no deadline")
    if not str(manifest["calibration"]["excitation"].get("declared_form", "")).strip():
        raise ValueError("the calibration does not declare the form of its excitation")
    # The trial dither is what makes this tier's streamed recordings carry
    # identifying variation, so it is a declared constant of the manifest: one
    # seed, the calibration's own per-channel amplitudes and rates, and a ramp.
    excitation = trial["excitation"]
    seed = excitation["phase_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("the trial dither needs one declared non-negative seed")
    for key in ("amplitudes_from", "rates_from"):
        if excitation[key] not in manifest["calibration"]:
            raise ValueError("the trial dither names constants the calibration lacks")
    table = trial_excitation(manifest)
    if table.shape != (
        trial["intervals"],
        len(manifest["telemetry"]["command_minimum"]),
    ):
        raise ValueError("the declared trial dither does not cover every command")
    if not np.isfinite(table).all():
        raise ValueError("the declared trial dither is not finite")
    return manifest


def live_expected_swap(blocks, offset, intervals):
    """Where a trial's swap must have happened, from the recorded gates alone.

    The first block whose gate passed decides it and the declared offset places
    it: the swap interval is that block's stop interval plus the offset, and
    there is no swap at all when no gate passed or when that interval lies past
    the end of the trial. Nothing here reads a clock, which is the point --
    a replay recomputes the interval rather than believing the one a run wrote.
    """
    first = live_first_gate_block(blocks)
    if first is None:
        return None, None
    stop = blocks[first]["stop_interval"]
    release = stop + int(offset)
    if release >= int(intervals):
        return first, None
    return first, release


def live_swap_gate(candidate, comparator):
    """The predeclared held-out swap gate, from two block scores and nothing else.

    ``candidate`` and ``comparator`` are one block's final-step forecast errors
    for the generic candidate revision and for the structured belief, measured
    on identical rows of a block neither has fitted on. The gate passes when the
    candidate is at or below the comparator on world velocity and on body rate,
    both. A score that is missing, null, nonfinite or not a number at all leaves
    that metric unknown, and an unknown metric never passes: the swap fails
    closed rather than on a number nobody can read.
    """
    metrics, passed = {}, True
    for metric in METRICS:
        mine = _number(candidate.get(metric) if isinstance(candidate, dict) else None)
        theirs = _number(
            comparator.get(metric) if isinstance(comparator, dict) else None
        )
        met = None if mine is None or theirs is None else bool(mine <= theirs)
        metrics[metric] = dict(
            candidate=mine,
            structured=theirs,
            met=met,
            margin=None if met is None else theirs - mine,
        )
        passed = passed and met is True
    return dict(passed=bool(passed), metrics=metrics)


def live_first_gate_block(blocks):
    """The index of the first block whose gate passed.

    The offer the loop is allowed to apply comes from this block and no other,
    so a replay recomputes it from the recorded per-block gates rather than
    believing what the run says it swapped on. The refit budget is measured and
    reported and deliberately does not appear here: a candidate withheld because
    the host was busy would put the host's scheduling into the trajectory.
    """
    for entry in blocks if isinstance(blocks, list) else ():
        if not isinstance(entry, dict):
            continue
        gate = entry.get("gate")
        if (gate.get("passed") if isinstance(gate, dict) else None) is True:
            return entry.get("index")
    return None


def live_segments(states, reference_states, swap_interval):
    """Tracking RMSE before a swap, after it, and over the whole trial.

    Interval ``i`` is scored at the state it ends on, ``states[i + 1]``, exactly
    as the whole-trial metric scores every interval after the shared initial
    state. ``before`` is intervals ``0`` to ``swap_interval`` exclusive and
    ``after`` is ``swap_interval`` to the last completed interval inclusive, so
    the two partition the trial and neither counts the other's rows. A segment
    with no interval in it is ``None`` rather than a number over nothing, and
    ``swap_interval`` of ``None`` leaves both segments empty: a trial that never
    swapped has no before and no after.
    """
    from glassbox.core.metrics import state_rmse_metrics

    states = np.asarray(states, dtype=float)
    reference_states = np.asarray(reference_states, dtype=float)
    if states.shape != reference_states.shape or len(states) < 1:
        raise ValueError("tracking states and reference rows must be paired")
    completed = len(states) - 1

    def over(start, stop):
        if stop <= start:
            return None
        return state_rmse_metrics(
            states[start + 1 : stop + 1], reference_states[start + 1 : stop + 1]
        )

    if swap_interval is None:
        before = after = None
    else:
        swap = int(swap_interval)
        if not 0 <= swap <= completed:
            raise ValueError("the swap interval lies outside the completed trial")
        before, after = over(0, swap), over(swap, completed)
    return dict(before=before, after=after, whole=over(0, completed))


def live_decide(manifest, rows, reference=None, reference_sha256=None):
    """Every gate, evaluated from recorded trial metrics alone. Anything unclear fails.

    The semantics the platform and control tiers decide by, with this tier's own
    rule. A run is accepted when no metric regressed past its reference value
    times one plus the manifest's relative tolerance plus its absolute one, the
    rule holds on every case the reference already meets it on, and nothing
    structural failed: a trial missing, duplicated, undeclared, terminated,
    short of its declared intervals, reporting a refinement-worker error, or
    carrying a whole-trial metric that is not a finite number.

    The rule itself has two parts and both are reported on every adopting trial
    either way: the swap happened, and the trial's position and attitude RMSE
    over the intervals after the swap are at or below its own values over the
    intervals before it within the declared allowance. The frozen arm's numbers
    over the same two interval ranges are reported beside them and gate nothing;
    they are what "before" would have been if nothing had been swapped in.
    """
    repetitions = manifest["trial"]["repetitions"]
    expected = {(index, arm) for index in range(repetitions) for arm in LIVE_ARMS}
    keys = [(row.get("repetition"), row.get("arm")) for row in rows]
    breaches, rule_breaches, regressions, summary = [], [], [], {}
    budget_breaches = []
    criteria, live = {}, {}
    relative = manifest["reference"]["relative_tolerance"]
    absolute = manifest["reference"]["absolute_tolerance"]
    for index, arm in sorted(expected - set(keys)):
        breaches.append(dict(trial=f"{index}-{arm}", gate="trial_present"))
    for index, arm in sorted({key for key in keys if keys.count(key) > 1}):
        breaches.append(dict(trial=f"{index}-{arm}", gate="trial_unique"))
    for key in sorted(set(keys) - expected, key=repr):
        breaches.append(dict(trial=f"{key[0]}-{key[1]}", gate="trial_declared"))
    measured = {}
    for row in rows:
        key = (row.get("repetition"), row.get("arm"))
        if key not in expected or keys.count(key) > 1:
            continue
        name = f"{key[0]}-{key[1]}"
        criteria[name] = row.get("pass_criterion")
        # A refit that ran over its declared wall budget is a measurement of
        # this host, reported beside the rest and gating nothing: it changed no
        # command, because the candidate was offered either way.
        for entry in row.get("blocks") or ():
            if isinstance(entry, dict) and entry.get("within_budget") is not True:
                budget_breaches.append(
                    dict(
                        trial=name,
                        block=entry.get("index"),
                        gate="refit_wall_budget",
                        value=entry.get("refit_wall_seconds"),
                        limit=entry.get("wall_budget_seconds"),
                    )
                )
        live[name] = dict(
            swapped=row.get("swapped"),
            swap_interval=row.get("swap_interval"),
            swap_time_s=row.get("swap_time_s"),
            swap_revision=row.get("swap_revision"),
            swap_scored_block=row.get("swap_scored_block"),
            swap_release_interval=row.get("swap_release_interval"),
            offer_deferred_past_trial=row.get("offer_deferred_past_trial"),
            candidate_revisions=row.get("candidate_revisions"),
            submitted_blocks=row.get("submitted_blocks"),
            dropped_blocks=row.get("dropped_blocks"),
            budget_overruns=row.get("budget_overruns"),
            wall=row.get("wall"),
            blocks=row.get("blocks"),
        )
        if row.get("terminated") is not False:
            breaches.append(
                dict(
                    trial=name,
                    gate="trial_complete",
                    completed=row.get("completed_intervals"),
                    requested=row.get("requested_intervals"),
                    failure=row.get("failure"),
                )
            )
            continue
        if row.get("completed_intervals") != manifest["trial"]["intervals"]:
            breaches.append(dict(trial=name, gate="declared_intervals"))
            continue
        if row.get("worker_error") is not None:
            breaches.append(
                dict(trial=name, gate="worker_error", error=row.get("worker_error"))
            )
            continue
        recorded = row.get("segments")
        values = {}
        for segment in LIVE_SEGMENTS:
            scored = recorded.get(segment) if isinstance(recorded, dict) else None
            entry = {}
            for metric in LIVE_METRICS:
                value = _number(
                    scored.get(metric) if isinstance(scored, dict) else None
                )
                if value is not None and value < 0:
                    value = None
                # Only the whole trial is required to carry a number: a trial
                # that never swapped has no before and no after, and that is a
                # rule breach rather than an unreadable metric.
                if segment == "whole" and value is None:
                    breaches.append(
                        dict(trial=name, metric=metric, gate="finite_rmse", value=value)
                    )
                entry[metric] = value
            values[segment] = entry
        measured[key] = values
    for index in sorted({key[0] for key in measured}):
        adopting = measured.get((index, "adopting"))
        frozen = measured.get((index, "frozen"))
        if adopting is None:
            continue
        name = f"{index}-adopting"
        swapped = live[name]["swapped"] is True
        swap_gates = (
            reference is not None
            and isinstance(reference.get("swapped"), dict)
            and reference["swapped"].get(str(index)) is True
        )
        if not swapped:
            rule_breaches.append(
                dict(trial=name, gate="swap_occurred", gating=swap_gates)
            )
        summary[str(index)] = {}
        for metric in LIVE_METRICS:
            before = adopting["before"][metric]
            after = adopting["after"][metric]
            limit = None if before is None else before * (1 + relative) + absolute
            base = _reference_value(
                reference, "tracking_rmse", str(index), "adopting", "after", metric
            )
            meets = _reference_meets(base, [limit])
            summary[str(index)][metric] = dict(
                adopting_before=before,
                adopting_after=after,
                adopting_whole=adopting["whole"][metric],
                frozen_before=None if frozen is None else frozen["before"][metric],
                frozen_after=None if frozen is None else frozen["after"][metric],
                frozen_whole=None if frozen is None else frozen["whole"][metric],
                limit=limit,
                reference=base,
                reference_meets_rule=meets,
            )
            if reference is not None:
                if base is None:
                    regressions.append(
                        dict(trial=name, metric=metric, gate="reference_present")
                    )
                else:
                    ceiling = base * (1 + relative) + absolute
                    if after is None or after > ceiling:
                        regressions.append(
                            dict(
                                trial=name,
                                metric=metric,
                                gate="reference_tracking_rmse",
                                value=after,
                                reference=base,
                                limit=ceiling,
                            )
                        )
            if not swapped:
                continue
            if before is None or after is None:
                rule_breaches.append(
                    dict(
                        trial=name,
                        metric=metric,
                        gate="segments_present",
                        gating=meets is True,
                    )
                )
                continue
            if after > limit:
                rule_breaches.append(
                    dict(
                        trial=name,
                        metric=metric,
                        gate="tracking_after_swap",
                        value=after,
                        limit=limit,
                        gating=meets is True,
                    )
                )
    regressions.sort(key=lambda entry: (entry["trial"], entry.get("metric", "")))
    enforced = bool(manifest["decision"]["enforced"])
    gating = [breach for breach in rule_breaches if breach["gating"]]
    rule_met = not breaches and not rule_breaches
    accepted = not breaches and not regressions and (not enforced or not gating)
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        rule_met=rule_met,
        rule_enforced=enforced,
        gating_rule_breaches=len(gating),
        rule=manifest["decision"]["rule"],
        gates_from=manifest["decision"]["gates_from"],
        trials=len(rows),
        gate_breaches=breaches,
        rule_breaches=rule_breaches,
        budget_breaches=budget_breaches,
        budget_breaches_meaning=manifest["live"]["refit"]["on_budget_overrun"],
        reference_regressions=regressions,
        reference_compared=reference is not None,
        reference_sha256=reference_sha256,
        tracking_rmse=summary,
        live=live,
        pass_criterion=criteria,
        pass_criterion_meaning=manifest["metrics"]["pass_criterion"]["meaning"],
        swap_gate=manifest["live"]["swap_gate"]["rule"],
        meaning=manifest["decision"]["meaning"],
    )


# --- the live tier: the transport, the refiner and one refit-and-swap trial -


class _LiveRevision:
    """One plan model inside one live session, with a stable identity.

    ``learned`` is ``None`` for the revision the trial starts on, because what
    is flying then is the frozen structured belief rather than any revision of
    the generic recipe. That distinction is what lets the very first candidate
    -- the calibration fit, which has learned nothing from the flight yet -- be
    offered like any other, instead of being silently skipped for already being
    active.
    """

    __slots__ = ("block_index", "fingerprint", "learned", "revision_id")

    def __init__(self, revision_id, learned, block_index):
        self.revision_id = revision_id
        self.learned = learned
        self.block_index = block_index
        self.fingerprint = None if learned is None else learned.fingerprint()

    def to_dict(self):
        if self.learned is None:
            return dict(
                revision_id=self.revision_id,
                model="the frozen structured belief the trial starts on",
            )
        return dict(
            revision_id=self.revision_id,
            fitted_after_block=self.block_index,
            fingerprint=self.fingerprint,
            horizon_steps=int(self.learned.horizon_steps),
            recordings=len(self.learned.report["training"]),
        )


class _LiveScore:
    """One revision and the block score it earned, in the shape the worker reads."""

    __slots__ = ("revision", "score")

    def __init__(self, revision, score):
        self.revision, self.score = revision, score


class _LiveResult:
    """One scored and absorbed block, in the shape the worker and journal read."""

    __slots__ = ("candidate_after", "candidate_score", "record")

    def __init__(self, record, candidate_score, candidate_after):
        self.record = record
        self.candidate_score = candidate_score
        self.candidate_after = candidate_after

    def to_dict(self):
        return dict(self.record)


def live_block_rows(manifest, block, collection, steps):
    """One streamed block's evaluation rows: the same origins for both models.

    Every origin of the block that carries the recipe's whole consumed context
    and its whole horizon inside the block, with the recorded future commands,
    and the same origin's full canonical state, command history and exogenous
    context for the structured rollout. One index, one set of rows, two models --
    the platform tier's arrangement, cut from a block instead of a corpus.
    """
    from glassbox.core.data import control_history_before

    if len(collection.segments) != 1:
        raise ValueError("a streamed block must adapt to one contiguous segment")
    segment = collection.segments[0]
    context, horizon = steps["history"], steps["horizon"]
    history = manifest["live"]["transport"]["command_history_steps"]
    columns = {
        key: []
        for key in (
            "past_states",
            "past_inputs",
            "future_inputs",
            "targets",
            "initial_states",
            "control_histories",
            "controls",
            "initial_exogenous",
        )
    }
    origins = list(range(context, len(segment.states) - horizon))
    if len(origins) < 1:
        raise ValueError("a streamed block carries no origin with a complete context")
    for row in origins:
        columns["past_states"].append(segment.states[row - context : row + 1])
        columns["past_inputs"].append(segment.inputs[row - context : row])
        columns["future_inputs"].append(segment.inputs[row : row + horizon])
        columns["targets"].append(segment.states[row + 1 : row + horizon + 1])
        columns["initial_states"].append(block.states[row])
        columns["control_histories"].append(control_history_before(block, row, history))
        columns["controls"].append(block.controls[row : row + horizon])
        columns["initial_exogenous"].append(block.exogenous[row])
    spec = block.spec
    return dict(
        **{key: np.stack(value) for key, value in columns.items()},
        source_origins=np.array(origins),
        control_roles=np.array(list(spec.control_roles), dtype="<U64"),
        exogenous_roles=np.array(list(spec.exogenous_roles), dtype="<U64"),
    )


def live_block_scores(manifest, learned, belief, arrays):
    """Both models' forecasts and final-step scores on one block's rows.

    The generic candidate through the recipe's own ``predict`` and the frozen
    structured belief through the library's own rollout, from identical origins
    with identical commands. Both run in x64, which is the precision every fit
    and every replay in this harness runs at, so a replay reproduces them.
    """
    import jax

    with jax.enable_x64(True):
        generic = np.asarray(
            learned.predict(
                arrays["past_states"], arrays["past_inputs"], arrays["future_inputs"]
            )
        )
        structured = structured_forecast(
            belief.params, arrays, manifest["plant"]["sample_interval_s"]
        )
    if not np.isfinite(generic).all() or not np.isfinite(structured).all():
        raise ValueError("a nonfinite block forecast cannot decide a swap")
    return (
        generic,
        structured,
        platform_measure(generic, arrays["targets"])["final_step"],
        platform_measure(structured, arrays["targets"])["final_step"],
    )


class _LiveRefiner:
    """Score and refit the generic recipe on whole streamed blocks.

    The refinement contract :class:`~glassbox.workflows.streaming.Refiner`
    declares, over the generic recipe rather than a structured belief. Each
    block is scored by the current candidate and by the frozen structured
    belief before anything is fitted on it, then absorbed with
    ``update(recordings)`` as one new recording with its own identity. The
    recipe's holdout and window sampling are the recipe's own and nothing here
    reaches into them.

    Blocks, their evaluation arrays and every revision are retained in memory
    for the run to write out after the trial. Nothing is saved from this thread:
    a refit inside a block period leaves no room for a disk write.
    """

    def __init__(
        self, manifest, learned, belief, *, recording_id, session, excitation=None
    ):
        self._manifest = manifest
        self._belief = belief
        # The trial's own realized excitation, interval by interval, appended by
        # the loop as it applies each command. A block's rows are all complete
        # before the synchronous submit that hands it over, so the slice a block
        # needs always exists by the time this reads it.
        self._excitation = excitation
        self._dt_s = manifest["plant"]["sample_interval_s"]
        self._steps = steps_for(self._dt_s)
        self._budget = manifest["live"]["refit"]["budget"]
        self.history_steps = manifest["live"]["transport"]["command_history_steps"]
        self.session, self.recording_id = session, recording_id
        self._structured = _LiveRevision(f"{session}:structured", None, None)
        base = _LiveRevision(f"{session}:0", learned, None)
        self._active, self._candidate = self._structured, base
        self._revisions = {
            self._structured.revision_id: self._structured,
            base.revision_id: base,
        }
        self._scored = set()
        self._results = []
        self.blocks = []
        self.skipped_interval_count = 0
        self.offered = False
        self._next = 1
        self._cursor = 0
        self._hold = None

    @property
    def retained_revision_count(self):
        return len(self._revisions)

    @property
    def active(self):
        return self._active

    @property
    def candidate(self):
        return self._candidate

    @property
    def results(self):
        return tuple(self._results)

    def hold_for_adoption(self, revision_id):
        if self._hold is not None:
            raise ValueError("an adoption decision is already outstanding")
        if revision_id not in self._scored:
            raise ValueError("only an evaluated revision can be held for adoption")
        self._hold = revision_id

    def release_adoption_hold(self):
        self._hold = None

    def skip(self, *, recording_id, start_interval, stop_interval, reason):
        if (
            recording_id != self.recording_id
            or start_interval != self._cursor
            or stop_interval <= start_interval
        ):
            raise ValueError("a gap must advance this recording's cursor")
        self._cursor = int(stop_interval)
        self.skipped_interval_count += int(stop_interval) - int(start_interval)
        return dict(
            recording_id=recording_id,
            start_interval=int(start_interval),
            stop_interval=int(stop_interval),
            reason=reason,
        )

    def adopt(self, revision_id, *, expected_active_revision, reason):
        if expected_active_revision != self._active.revision_id:
            raise ValueError(
                "active revision changed; reconsider the adoption decision"
            )
        if revision_id not in self._scored:
            raise ValueError(
                "adoption requires a revision scored on a subsequent block"
            )
        from glassbox.workflows.refinement import Adoption

        previous = self._active.revision_id
        self._active = self._revisions[revision_id]
        return Adoption(previous, revision_id, len(self._results), reason)

    def observe(self, telemetry, *, recording_id, start_interval):
        import jax

        if recording_id != self.recording_id or start_interval != self._cursor:
            raise ValueError("blocks must arrive in order on one recording")
        index = len(self._results)
        name = f"{recording_id}-block-{index:03d}"
        stop = int(start_interval) + len(telemetry.controls)
        injected = (
            None
            if self._excitation is None
            else np.asarray(self._excitation[int(start_interval) : stop], dtype=float)
        )
        collection = control_collection(
            self._manifest,
            [(name, telemetry)],
            None if injected is None else {name: injected},
        )
        arrays = live_block_rows(self._manifest, telemetry, collection, self._steps)
        scored = self._candidate
        started = time.monotonic()
        generic, structured, generic_score, structured_score = live_block_scores(
            self._manifest, scored.learned, self._belief, arrays
        )
        score_wall = time.monotonic() - started
        gate = live_swap_gate(generic_score, structured_score)
        started = time.monotonic()
        with jax.enable_x64(True):
            updated = scored.learned.update(collection)
        refit_wall = time.monotonic() - started
        successor = _LiveRevision(f"{self.session}:{self._next}", updated, index)
        record = dict(
            index=index,
            recording_id=name,
            start_interval=int(start_interval),
            stop_interval=stop,
            rows=len(arrays["targets"]),
            excitation_declared=bool(collection.excitation_declared),
            excitation_standard_deviation_fraction=updated.report.get(
                "excitation_standard_deviation_fraction"
            ),
            scored_revision=scored.revision_id,
            scored_fingerprint=scored.fingerprint,
            revision=successor.revision_id,
            generic=generic_score,
            structured=structured_score,
            gate=gate,
            score_wall_seconds=score_wall,
            refit_wall_seconds=refit_wall,
            fit_steps=int(self._budget["fit_steps"]),
            wall_budget_seconds=float(self._budget["wall_seconds"]),
            within_budget=bool(refit_wall <= float(self._budget["wall_seconds"])),
            training_recordings=len(updated.report["training"]),
            training_windows=int(
                sum(entry["windows"] for entry in updated.report["training"].values())
            ),
        )
        # Every piece of session state advances only after the numerical work.
        self._scored.add(scored.revision_id)
        self._revisions[successor.revision_id] = successor
        self._candidate = successor
        self._cursor = record["stop_interval"]
        self._next += 1
        result = _LiveResult(record, _LiveScore(scored, generic_score), successor)
        self._results.append(result)
        self.blocks.append(
            dict(record=record, arrays=arrays, generic=generic, structured=structured)
        )
        return result

    def revision(self, revision_id):
        """One retained revision of this session, by identity."""
        return self._revisions[revision_id]

    def summary(self):
        return dict(
            session=self.session,
            blocks=len(self._results),
            candidate_revisions=self._next - 1,
            retained_revisions=self.retained_revision_count,
            skipped_intervals=self.skipped_interval_count,
            active=self._active.to_dict(),
            candidate=self._candidate.to_dict(),
        )


def _live_prewarm(manifest, artifacts, warmup, reference_fn):
    """Compile every kernel a timed trial will use, on the calibration recording.

    The generic plan model's solver, the block scoring and the refit all compile
    on first use, and none of them may compile while a trial clock is running.
    Each one is exercised here on recorded calibration data, in the shapes the
    trial will use, and every result is discarded. The refit chain is walked on
    a throwaway learner, because the training cache grows by one block's windows
    until it reaches the recipe's cap and each size is its own compiled shape.
    """
    import jax

    transport = manifest["live"]["transport"]
    dt_s = manifest["plant"]["sample_interval_s"]
    steps = steps_for(dt_s)
    block_steps = transport["block_steps"]
    history = transport["command_history_steps"]
    _control_prewarm(
        _GenericArm(manifest, artifacts["generic"]), manifest, warmup, reference_fn
    )
    scratch = artifacts["generic"]
    states = np.asarray(warmup.states, dtype=float)
    commands = np.asarray(warmup.controls, dtype=float)
    spec = control_telemetry_spec(manifest)
    dither = trial_excitation(manifest)
    for index in range(transport["blocks_per_trial"]):
        start = history + index * block_steps
        if start + block_steps >= len(commands):
            break
        block = _live_block_trajectory(
            spec, dt_s, states, commands, start, block_steps, history
        )
        name = f"prewarm-{index}"
        collection = control_collection(
            manifest, [(name, block)], {name: dither[:block_steps]}
        )
        arrays = live_block_rows(manifest, block, collection, steps)
        live_block_scores(manifest, scratch, artifacts["structured"], arrays)
        with jax.enable_x64(True):
            scratch = scratch.update(collection)


def _live_block_trajectory(spec, dt_s, states, commands, start, block_steps, history):
    """One prewarm block cut from a recording, shaped like a streamed block."""
    from glassbox.core.data import Trajectory

    return Trajectory(
        time_s=np.arange(block_steps + 1) * dt_s,
        states=states[start : start + block_steps + 1],
        controls=commands[start : start + block_steps],
        spec=spec,
        control_prefix=commands[start - history : start],
        labels={"source_group": f"prewarm-{start}"},
    )


def _live_trial(
    manifest,
    *,
    adopting,
    artifacts,
    plant,
    warmup,
    reference_fn,
    anchor_state,
    session,
    directory,
):
    """One tracking trial with a learner behind it, computed in simulated time.

    The frozen structured arm flies from the first interval. Every interval's
    aligned transition -- the observed state it started at, the observed state
    it ended at, and the command the loop actually applied over it -- goes into
    the transition buffer, and whole blocks go to the refinement worker. Both
    arms run the same transport and the same refits, so they differ in the swap
    and in nothing else; only the adopting arm is given a preparation callback
    and an adoption policy, which is what makes an offer possible at all.

    Nothing a clock measured may reach the trajectory. The loop is not paced,
    the solver is given no deadline and therefore never falls back for want of
    time, the worker learns synchronously inside the ``submit`` that hands it a
    block, and an offer is released at a declared interval offset after the
    block it was scored on rather than whenever a refit happened to finish.
    Solve times, interval times and refit wall times are all still measured, and
    are recorded in a separate artifact that nothing reads back.
    """
    from glassbox.control.plan import ReferenceTrajectory
    from glassbox.core.metrics import state_rmse_metrics
    from glassbox.workflows.streaming import RefinementWorker, TransitionBuffer

    directory.mkdir(parents=True, exist_ok=True)
    trial = manifest["trial"]
    transport = manifest["live"]["transport"]
    dt_s = trial["sample_interval_s"]
    requested = trial["intervals"]
    deadline_s = trial["solve_deadline_s"]
    context = steps_for(manifest["plant"]["sample_interval_s"])["history"]
    history_steps = transport["command_history_steps"]
    release_offset = transport["offer_release_offset_intervals"]

    # The declared trial dither: the same seeded sequence in every trial and on
    # both arms, added to whatever command the active controller solved before
    # the plant is stepped with it. Nothing about it is measured here -- it is a
    # function of the manifest's own constants and its declared trial seed.
    dither = trial_excitation(manifest)
    minimum = np.asarray(manifest["telemetry"]["command_minimum"], dtype=float)
    maximum = np.asarray(manifest["telemetry"]["command_maximum"], dtype=float)
    solved_commands, injected = [], []

    arm = _StructuredArm(manifest, artifacts["structured"])
    _control_prewarm(arm, manifest, warmup, reference_fn)
    refiner = _LiveRefiner(
        manifest,
        artifacts["generic"],
        artifacts["structured"],
        recording_id=session,
        session=session,
        excitation=injected,
    )
    buffer = TransitionBuffer(
        control_telemetry_spec(manifest),
        dt_s,
        recording_id=session,
        block_steps=transport["block_steps"],
        history_steps=history_steps,
    )
    events = []

    def journal(record):
        events.append(record)

    def should_offer(result):
        # The budget is measured, not obeyed: a refit that ran long is recorded
        # and reported, and suppressing its candidate here would let the host's
        # scheduling decide where the aircraft flew.
        return bool(result.record["gate"]["passed"] and not refiner.offered)

    def prepare(revision, block):
        # One swap per trial, declared: the first candidate that passes.
        refiner.offered = True
        controller = _GenericArm(manifest, revision.learned)
        states = np.asarray(block.states, dtype=float)
        commands = np.asarray(block.controls, dtype=float)
        controller.reset(states[0], commands[0])
        warm = None
        for step in range(len(commands)):
            controller.observe(states[step])
            if controller.ready:
                future = (step + np.arange(controller.prediction_steps + 1)) * dt_s
                outcome = controller.solve(
                    states[step],
                    ReferenceTrajectory(reference_fn(future)),
                    commands[step],
                    warm_start=warm,
                )
                warm = outcome.warm_start
                np.asarray(outcome.command)
                np.asarray(outcome.predicted_states)
            controller.command_applied(commands[step])
        controller.reset(states[0], commands[0])
        return controller

    worker = RefinementWorker(
        refiner=refiner,
        history_steps=history_steps,
        block_steps=transport["block_steps"],
        queue_capacity=transport["queue_capacity"],
        retained_blocks=transport["retained_blocks"],
        synchronous=True,
        prepare=prepare if adopting else None,
        should_offer=should_offer if adopting else None,
        on_event=journal,
    )

    state = plant.initial_state.copy()
    previous = plant.initial_command.copy()
    arm.reset(state, previous)
    observed = [state.copy()]
    applied, tick_times, solve_times, telemetry_times = [], [], [], []
    solver_used, fallbacks, statuses, actives = [], [], [], []
    recent_states = deque(maxlen=context)
    recent_commands = deque(maxlen=context)
    swap = dict(
        swapped=False,
        swap_interval=None,
        swap_time_s=None,
        swap_revision=None,
        swap_scored_block=None,
        swap_scored_stop_interval=None,
        swap_release_interval=None,
        offers=0,
        rejected_offers=0,
        offer_deferred_past_trial=False,
    )
    warm_start, failure = None, None
    held, held_release = None, None
    started = time.monotonic()
    worker.start()
    try:
        for index in range(requested):
            if held is not None and index >= held_release:
                # The declared offset has elapsed in the trial's own interval
                # count, which is the only clock this trajectory reads.
                applicable = bool(
                    not swap["swapped"]
                    and held.expected_active_revision == refiner.active.revision_id
                    and len(recent_states) == context
                )
                if applicable:
                    # The observed history belongs to the flight, not to the
                    # revision: the prepared controller is seeded with this
                    # trial's own retained states and applied commands, and
                    # nothing is padded.
                    candidate = held.controller
                    candidate.reset(state, previous)
                    for past_state, past_command in zip(
                        recent_states, recent_commands, strict=True
                    ):
                        candidate.observe(past_state)
                        candidate.command_applied(past_command)
                    arm, warm_start = candidate, None
                    swap.update(
                        swapped=True,
                        swap_interval=index,
                        swap_time_s=index * dt_s,
                        swap_revision=held.revision.revision_id,
                    )
                else:
                    swap["rejected_offers"] += 1
                worker.acknowledge(held, applied=applicable)
                held = None
            tick = time.monotonic()
            arm.observe(state)
            if arm.ready:
                future = (index + np.arange(arm.prediction_steps + 1)) * dt_s
                # No deadline: a solve cut short by a busy host would put the
                # wall clock into the commands, and this tier's trajectory is a
                # function of the plant, the models and the reference alone.
                result = arm.solve(
                    state,
                    ReferenceTrajectory(reference_fn(future)),
                    previous,
                    warm_start=warm_start,
                )
                command = np.asarray(result.command, dtype=float)
                warm_start = result.warm_start
                solve_times.append(float(result.diagnostics.solve_time_s))
                solver_used.append(True)
                fallbacks.append(bool(result.used_fallback))
                statuses.append(str(result.status))
            else:
                command = previous.copy()
                solve_times.append(0.0)
                solver_used.append(False)
                fallbacks.append(False)
                statuses.append("model_not_ready")
            # The dither is added to the solved command and the sum is held in
            # the declared command box, because a command outside it is not one
            # this vehicle accepts. What the plant is stepped with is the
            # applied command, and what the learner is told was injected is the
            # difference between the two: the exogenous component that actually
            # reached the aircraft, not the one that was asked for.
            solved = command
            command = np.clip(solved + dither[index], minimum, maximum)
            next_state = np.asarray(plant.advance(command), dtype=float)
            if not np.isfinite(next_state).all():
                failure = "nonfinite plant state"
                break
            arm.command_applied(command)
            observed.append(next_state.copy())
            applied.append(command.copy())
            solved_commands.append(solved.copy())
            injected.append(command - solved)
            actives.append(swap["swap_revision"] or f"{session}:structured")
            recent_states.append(state.copy())
            recent_commands.append(command.copy())
            previous, state = command, next_state
            tick_times.append(time.monotonic() - tick)
            telemetry = time.monotonic()
            try:
                block = buffer.push(
                    interval=index,
                    source_time_s=index * dt_s,
                    command=command,
                    state=observed[-2],
                    next_state=next_state,
                    received_at_s=time.monotonic(),
                )
            except ValueError as error:
                failure = f"telemetry refused a transition: {error}"
                break
            if block is not None:
                # Synchronous: the learning happens here, and any offer it
                # produces exists before this call returns.
                if not worker.submit(block):
                    failure = f"the refinement worker refused a block: {worker.error}"
                    break
                offer = worker.poll_offer()
                if offer is not None:
                    swap["offers"] += 1
                    held = offer
                    held_release = offer.scored_stop_interval + release_offset
                    swap.update(
                        swap_scored_stop_interval=int(offer.scored_stop_interval),
                        swap_release_interval=int(held_release),
                    )
            telemetry_times.append(time.monotonic() - telemetry)
    finally:
        elapsed_s = time.monotonic() - started
        if held is not None:
            # Released past the end of the trial: recorded, never applied.
            swap["offer_deferred_past_trial"] = True
            swap["rejected_offers"] += 1
            worker.acknowledge(held, applied=False)
        stopped = worker.close(timeout_s=60.0)
    worker_error = worker.error
    if not stopped:
        failure = failure or "the refinement worker did not stop within its budget"
    transport_summary = worker.summary()

    states_array = np.asarray(observed)
    commands_array = (
        np.asarray(applied)
        if applied
        else np.zeros((0, len(plant.initial_command)), dtype=float)
    )
    times = np.arange(len(states_array)) * dt_s
    reference_states = reference_fn(times)
    width = len(plant.initial_command)
    solved_array = (
        np.asarray(solved_commands)
        if solved_commands
        else np.zeros((0, width), dtype=float)
    )
    injected_array = (
        np.asarray(injected) if injected else np.zeros((0, width), dtype=float)
    )
    bound_violation = (
        max(
            0.0,
            float(np.max(minimum - commands_array)),
            float(np.max(commands_array - maximum)),
        )
        if len(commands_array)
        else 0.0
    )
    # The trajectory and the clock live in different files. Everything in
    # tracking.npz is a function of the plant, the models and the reference, so
    # two runs of this tier produce it byte for byte; nothing in timing.npz is,
    # and nothing reads it back. The solved command and the excitation are
    # saved beside the applied one so a replay can recompute the declared
    # dither and check that the applied command is the solved one plus it.
    np.savez_compressed(
        directory / "tracking.npz",
        time_s=times,
        states=states_array,
        reference_states=reference_states,
        commands=commands_array,
        solved_commands=solved_array,
        excitation=injected_array,
        solver_used=np.asarray(solver_used, dtype=bool),
        used_fallback=np.asarray(fallbacks, dtype=bool),
        active_revisions=np.asarray(actives, dtype="<U64"),
        initial_state=np.asarray(plant.initial_state, dtype=float),
        reference_anchor_state=np.asarray(anchor_state, dtype=float),
    )
    np.savez_compressed(
        directory / "timing.npz",
        tick_times_s=np.asarray(tick_times, dtype=float),
        solve_times_s=np.asarray(solve_times, dtype=float),
        telemetry_times_s=np.asarray(telemetry_times, dtype=float),
    )
    names = ["tracking.npz", "timing.npz"]
    blocks = []
    for entry in refiner.blocks:
        record = dict(entry["record"])
        index = record["index"]
        np.savez_compressed(
            directory / f"block-{index:03d}.npz",
            **entry["arrays"],
            generic_prediction=entry["generic"],
            structured_prediction=entry["structured"],
        )
        model = f"block-{index:03d}-scored.npz"
        refiner.revision(record["scored_revision"]).learned.save(directory / model)
        record["arrays"] = f"block-{index:03d}.npz"
        record["scored_model"] = model
        names.extend([record["arrays"], model])
        blocks.append(record)
    swap["swap_scored_block"] = next(
        (
            entry["index"]
            for entry in blocks
            if entry["stop_interval"] == swap["swap_scored_stop_interval"]
        ),
        None,
    )
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(event, allow_nan=False) + "\n" for event in events)
    )
    names.append("events.jsonl")
    terminated = failure is not None or len(commands_array) != requested
    row = dict(
        arm="adopting" if adopting else "frozen",
        session=session,
        completed_intervals=len(commands_array),
        requested_intervals=requested,
        terminated=bool(terminated),
        failure=failure,
        worker_error=worker_error,
        tracking_rmse=(
            state_rmse_metrics(states_array[1:], reference_states[1:])
            if len(commands_array)
            else None
        ),
        pass_criterion=control_pass_criterion(states_array, anchor_state, manifest),
        excitation=dict(
            declared=True,
            phase_seed=manifest["trial"]["excitation"]["phase_seed"],
            intervals=len(injected_array),
            bounded_intervals=_bounded_intervals(
                solved_array, dither[: len(solved_array)], minimum, maximum
            ),
            standard_deviation_fraction=_command_fraction(
                injected_array, commands_array
            ),
        ),
        **swap,
        blocks=blocks,
        candidate_revisions=len(blocks),
        submitted_blocks=transport_summary["submitted_blocks"],
        processed_blocks=transport_summary["processed_blocks"],
        dropped_blocks=transport_summary["dropped_blocks"],
        dropped_intervals=transport_summary["dropped_intervals"],
        skipped_intervals=transport_summary["skipped_intervals"],
        buffer_discarded_intervals=buffer.discarded_intervals,
        partial_intervals_at_shutdown=buffer.partial_intervals,
        budget_overruns=sum(not entry["within_budget"] for entry in blocks),
        model_not_ready_intervals=int(statuses.count("model_not_ready")),
        fallback_count=int(sum(fallbacks)),
        solver_statuses={
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        maximum_command_bound_violation=bound_violation,
        wall=simulated_time_wall(
            meaning=(
                "host measurements of this run only. None of them enters the "
                "trajectory, the block scores, the swap or any metric, and two "
                "runs of this tier differ in all of them."
            ),
            dt_s=dt_s,
            deadline_s=deadline_s,
            tick_times=tick_times,
            solve_times=solve_times,
            elapsed_s=elapsed_s,
            maximum_refit_seconds=max(
                (entry["refit_wall_seconds"] for entry in blocks), default=0.0
            ),
            maximum_score_seconds=max(
                (entry["score_wall_seconds"] for entry in blocks), default=0.0
            ),
            refit_budget_seconds=float(
                manifest["live"]["refit"]["budget"]["wall_seconds"]
            ),
            maximum_telemetry_seconds=max(telemetry_times, default=0.0),
        ),
        refiner=refiner.summary(),
        transport=transport_summary,
        controller=arm.summary(),
        files=_files(directory, names),
    )
    return row


def live(manifest_path, output):
    """Run the frozen live trial set once, both arms, and write the decision."""
    import jax

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = frozen_live_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
    reference_path = manifest_path.parent / manifest["reference"]["file"]
    reference, reference_digest = None, None
    if reference_path.exists():
        shutil.copyfile(reference_path, output / "reference.json")
        reference, reference_digest = read(reference_path), sha256(reference_path)
    write(
        output / "environment.json",
        dict(
            python=sys.version,
            platform=platform_module.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64="fits, refits and block scores only",
        ),
    )
    started = time.perf_counter()
    with jax.enable_x64(True):
        calibration, artifacts, warmup, initial_state, initial_command = (
            _control_calibrate(manifest, output)
        )
    declared = manifest["tracking_reference"]
    trial = manifest["trial"]
    with jax.enable_x64(True):
        starts = [
            control_initial_state(manifest, initial_state, seed)
            for seed in trial["initial_state_seeds"]
        ]

    def reference_fn(times):
        return control_reference(initial_state, times, declared)

    prewarm_started = time.perf_counter()
    _live_prewarm(manifest, artifacts, warmup, reference_fn)
    prewarm_s = time.perf_counter() - prewarm_started

    rows = []
    for repetition in range(trial["repetitions"]):
        order = trial["arm_order"][repetition % len(trial["arm_order"])]
        start = starts[repetition]
        flown = {}
        for name in order:
            print(json.dumps(dict(tracking=f"{repetition}-{name}")), flush=True)
            trial_started = time.perf_counter()
            row = _live_trial(
                manifest,
                adopting=name == "adopting",
                artifacts=artifacts,
                plant=_control_tracking_plant(manifest, start, initial_command),
                warmup=warmup,
                reference_fn=reference_fn,
                anchor_state=initial_state,
                session=f"live-{repetition}-{name}",
                directory=output / f"trial-{repetition}" / name,
            )
            row["repetition"] = repetition
            row["initial_state_seed"] = trial["initial_state_seeds"][repetition]
            row["wall"]["trial_seconds"] = time.perf_counter() - trial_started
            row["directory"] = f"trial-{repetition}/{name}"
            flown[name] = row
            print(
                json.dumps(
                    dict(
                        trial=f"{repetition}-{name}",
                        swapped=row["swapped"],
                        swap_interval=row["swap_interval"],
                        tracking_rmse=row["tracking_rmse"],
                        terminated=row["terminated"],
                        budget_overruns=row["budget_overruns"],
                        blocks=[
                            dict(
                                index=entry["index"],
                                generic=entry["generic"],
                                structured=entry["structured"],
                                passed=entry["gate"]["passed"],
                                refit_s=entry["refit_wall_seconds"],
                            )
                            for entry in row["blocks"]
                        ],
                    ),
                    allow_nan=False,
                ),
                flush=True,
            )
        # Both arms of a repetition are segmented at the same interval, because
        # the frozen arm exists to say what "before" would have kept doing.
        swap_interval = flown["adopting"]["swap_interval"]
        for name in ("adopting", "frozen"):
            row = flown[name]
            with np.load(
                output / row["directory"] / "tracking.npz", allow_pickle=False
            ) as data:
                # A terminated trial may be shorter than the interval its
                # repetition swapped at; it has no segments and fails closed.
                bounded = (
                    swap_interval
                    if swap_interval is not None
                    and swap_interval <= len(data["states"]) - 1
                    else None
                )
                row["segments"] = live_segments(
                    data["states"], data["reference_states"], bounded
                )
            row["segment_swap_interval"] = bounded
            write(output / row["directory"] / "trial.json", row)
            rows.append(row)
        write(output / "results.json", rows)
    decision = live_decide(manifest, rows, reference, reference_digest)
    decision["wall_seconds"] = time.perf_counter() - started
    decision["prewarm_seconds"] = prewarm_s
    decision["calibration"] = dict(
        command_excitation=calibration["command_excitation"]["fraction"],
        structured_fit_wall_seconds=calibration["structured_fit_wall_seconds"],
        generic_fit_wall_seconds=calibration["generic_fit_wall_seconds"],
        generic_fingerprint=calibration["generic_fingerprint"],
        reserved=calibration["reserved"],
    )
    write(output / "decision.json", decision)
    return decision


# --- the live tier: verifying -----------------------------------------------


def verify_live(directory, manifest, reference=None):
    """Recompute every live metric and the decision from saved arrays.

    This replay never reruns the plant, the solver or a refit. None of the three
    is deterministic under a wall clock or worth a second measurement, and
    rerunning any of them would be a new run rather than a check of this one.
    What it does check is everything the decision actually read, and it does so
    from forward passes over saved arrays only. Every recorded artifact hash is
    recomputed, including the calibration recordings, both fitted arms, every
    block's evaluation arrays and every scored revision, so an altered artifact
    is rejected. The tracking reference is rebuilt from the saved anchor state
    and the perturbed start from its declared seed, so a run cannot score itself
    against a task it invented. Both models' block forecasts are recomputed --
    the generic one through the independent NumPy recurrence, the structured one
    through the library's own rollout -- and every block score and every swap
    gate is recomputed from them. The swap the run applied is recomputed from
    the recorded gates, and the before and after segments from the saved
    per-interval tracking arrays.
    """
    import jax

    directory = Path(directory)
    _calibration, _learned, _coverage = _verify_calibration(directory, manifest)

    from glassbox.belief.belief_io import load_dynamics_belief
    from glassbox.core.metrics import state_rmse_metrics

    # In x64, because that is the precision the calibration fitted it at and
    # the precision every block score was computed at. Loaded anywhere else it
    # would be a single-precision copy of the comparator, and every replayed
    # block score would miss by a rounding error nobody introduced.
    with jax.enable_x64(True):
        belief = load_dynamics_belief(directory / "structured.json")
    trial = manifest["trial"]
    transport = manifest["live"]["transport"]
    dt_s = trial["sample_interval_s"]
    steps = steps_for(manifest["plant"]["sample_interval_s"])
    declared = manifest["tracking_reference"]
    release_offset = transport["offer_release_offset_intervals"]
    dither = trial_excitation(manifest)
    rows = read(directory / "results.json")
    checked, replays, worst = 0, 0, 0.0
    swaps = {}
    for row in rows:
        case = directory / row["directory"]
        if read(case / "trial.json") != row:
            raise ValueError(f"trial result mismatch: {row['directory']}")
        for name, digest in row["files"].items():
            if sha256(case / name) != digest:
                raise ValueError(f"altered artifact: {row['directory']}/{name}")
        with np.load(case / "timing.npz", allow_pickle=False) as data:
            tick_times = data["tick_times_s"]
            solve_times = data["solve_times_s"]
        with np.load(case / "tracking.npz", allow_pickle=False) as data:
            states = data["states"]
            commands = data["commands"]
            saved_reference = data["reference_states"]
            initial_state = data["initial_state"]
            anchor_state = data["reference_anchor_state"]
            times = data["time_s"]
            actives = data["active_revisions"]
            solved = data["solved_commands"]
            excitation = data["excitation"]
            if (
                len(states) != len(commands) + 1
                or len(times) != len(states)
                or len(actives) != len(commands)
                or solved.shape != commands.shape
                or excitation.shape != commands.shape
            ):
                raise ValueError(f"saved tracking arrays disagree: {row['directory']}")
            # The declared dither is recomputed from the manifest and its
            # declared trial seed, and the applied command has to be the solved
            # one plus it, held in the declared command box. That is what makes
            # "the same excitation on both arms" checkable rather than claimed.
            _verify_trial_excitation(
                row, manifest, dither, solved, commands, excitation
            )
            np.testing.assert_allclose(
                times, np.arange(len(states)) * dt_s, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                saved_reference,
                control_reference(anchor_state, times, declared),
                rtol=0,
                atol=1e-12,
            )
            with jax.enable_x64(True):
                np.testing.assert_allclose(
                    initial_state,
                    control_initial_state(
                        manifest, anchor_state, row["initial_state_seed"]
                    ),
                    rtol=0,
                    atol=1e-12,
                )
            np.testing.assert_allclose(states[0], initial_state, rtol=0, atol=1e-12)
            fresh = (
                state_rmse_metrics(states[1:], saved_reference[1:])
                if len(commands)
                else None
            )
            segments = live_segments(
                states, saved_reference, row["segment_swap_interval"]
            )
            criterion = control_pass_criterion(states, anchor_state, manifest)
            # The recorded active revision per interval has to agree with the
            # recorded swap: structured before it, the adopted revision after.
            swapped_at = row["swap_interval"]
            expected = np.full(len(commands), f"{row['session']}:structured")
            if swapped_at is not None:
                expected[swapped_at:] = row["swap_revision"]
            if not np.array_equal(actives, expected):
                raise ValueError(
                    f"the recorded active revisions do not match the recorded "
                    f"swap: {row['directory']}"
                )
        if criterion != row["pass_criterion"]:
            raise ValueError(f"recomputed pass criterion differs: {row['directory']}")
        if (fresh is None) != (row["tracking_rmse"] is None):
            raise ValueError(f"tracking metric shape mismatch: {row['directory']}")
        if fresh is not None:
            if sorted(fresh) != sorted(row["tracking_rmse"]):
                raise ValueError(f"tracking metric set mismatch: {row['directory']}")
            for metric, value in fresh.items():
                np.testing.assert_allclose(
                    value, row["tracking_rmse"][metric], **SCORE_TOLERANCE
                )
            row["tracking_rmse"] = fresh
        _same_segments(segments, row["segments"], row["directory"])
        row["segments"] = segments
        # Host measurements. They are recomputed because they were recorded,
        # and they decide nothing: the trajectory above was computed without
        # reading either of them.
        verify_simulated_time_wall(
            row["wall"],
            dt_s=dt_s,
            deadline_s=trial["solve_deadline_s"],
            tick_times=tick_times,
            solve_times=solve_times,
            label=row["directory"],
        )
        terminated = len(commands) != row["requested_intervals"]
        if (
            terminated != bool(row["terminated"])
            or len(commands) != (row["completed_intervals"])
        ):
            raise ValueError(f"recomputed completion differs: {row['directory']}")
        block_replays, block_worst = _verify_live_blocks(
            case, row, manifest, belief, steps
        )
        replays += block_replays
        worst = max(worst, block_worst)
        swaps[(row["repetition"], row["arm"])] = row
        checked += 1
    for row in rows:
        _verify_live_swap(row, release_offset, trial["intervals"])
    for repetition in sorted({key[0] for key in swaps}):
        adopting = swaps.get((repetition, "adopting"))
        if adopting is None:
            continue
        for arm in LIVE_ARMS:
            other = swaps.get((repetition, arm))
            if other is None or other["terminated"] or adopting["terminated"]:
                continue
            if other["segment_swap_interval"] != adopting["swap_interval"]:
                raise ValueError(
                    f"trial {repetition}-{arm} is segmented at an interval the "
                    "adopting arm of its repetition never swapped at"
                )
    anchor, digest = anchored_reference(directory, reference, COMMITTED_LIVE_REFERENCE)
    decision = live_decide(manifest, rows, anchor, digest)
    saved = read(directory / "decision.json")
    for key in (
        "manifest",
        "decision",
        "accepted",
        "rule_met",
        "rule_enforced",
        "gating_rule_breaches",
        "trials",
        "reference_compared",
        "reference_sha256",
    ):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed decision differs: {key}")
    for key in ("gate_breaches", "rule_breaches", "reference_regressions"):
        if len(decision[key]) != len(saved[key]):
            raise ValueError(f"replayed decision differs: {key}")
    return dict(
        tier="live",
        verified_trials=checked,
        replays=replays,
        maximum_replay_difference=worst,
        decision=decision,
        meaning=(
            "A replay of saved evidence: recorded hashes, both models' block "
            "forecasts recomputed from the saved evaluation arrays and the saved "
            "revisions, every block score and swap gate recomputed from those "
            "forecasts, the swap recomputed from the recorded gates, and the "
            "tracking metrics and segments recomputed from the saved "
            "per-interval arrays, and the swap recomputed from the recorded "
            "gates and the declared release offset. The plant, the solver and "
            "the refits are not rerun, because rerunning any of them would be a "
            "new measurement rather than a check of this one. The host "
            "measurements in timing.npz are recomputed because they were "
            "recorded; nothing in the trajectory, the scores or the decision "
            "reads them."
        ),
    )


def _verify_trial_excitation(row, manifest, dither, solved, commands, excitation):
    """Recompute one trial's declared dither and what it actually injected.

    The dither is a function of the manifest's own constants and its declared
    trial seed, so it is rebuilt rather than believed. The applied command has
    to be the solved command plus that dither, held in the declared command
    box, and the recorded excitation has to be the difference between the two:
    what reached the aircraft, rather than what was asked for. Both arms of
    every repetition are checked against the same rebuilt sequence, which is
    what "applied identically to the frozen and the adopting arm" means here.
    """
    label = row["directory"]
    minimum = np.asarray(manifest["telemetry"]["command_minimum"], dtype=float)
    maximum = np.asarray(manifest["telemetry"]["command_maximum"], dtype=float)
    executed = len(commands)
    if executed > len(dither):
        raise ValueError(f"the trial ran past its declared dither: {label}")
    expected = np.clip(solved + dither[:executed], minimum, maximum)
    if not np.array_equal(expected, commands):
        raise ValueError(
            f"the applied commands are not the solved ones plus the declared dither: {label}"
        )
    if not np.array_equal(commands - solved, excitation):
        raise ValueError(f"the recorded excitation is not what was injected: {label}")
    recorded = row.get("excitation")
    if not isinstance(recorded, dict) or recorded.get("declared") is not True:
        raise ValueError(f"the trial does not declare its excitation: {label}")
    if recorded.get("phase_seed") != manifest["trial"]["excitation"]["phase_seed"]:
        raise ValueError(f"the trial declares another dither seed: {label}")
    if recorded.get("intervals") != executed:
        raise ValueError(f"the declared excitation covers other intervals: {label}")
    if recorded.get("bounded_intervals") != _bounded_intervals(
        solved, dither[:executed], minimum, maximum
    ):
        raise ValueError(f"the recorded bounded interval count differs: {label}")
    _same_fraction(
        _command_fraction(excitation, commands),
        recorded.get("standard_deviation_fraction"),
        label,
        **SCORE_TOLERANCE,
    )
    # And that it reached the learner: every streamed block declared it, and
    # the fraction its refit reported is the one this block's own rows give.
    for record in row.get("blocks") or ():
        start, stop = record["start_interval"], record["stop_interval"]
        block = f"{label}/block-{record['index']:03d}"
        if record.get("excitation_declared") is not True:
            raise ValueError(f"a streamed block carried no excitation: {block}")
        if not 0 <= start < stop <= executed:
            raise ValueError(f"a streamed block lies outside the trial: {block}")
        _same_fraction(
            _command_fraction(excitation[start:stop], commands[start:stop]),
            record.get("excitation_standard_deviation_fraction"),
            block,
            **SCORE_TOLERANCE,
        )


def _same_segments(fresh, saved, label):
    """One trial's recomputed before/after/whole metrics against its recorded ones."""
    if sorted(fresh) != sorted(saved or {}):
        raise ValueError(f"segment set mismatch: {label}")
    for segment, measured in fresh.items():
        recorded = saved[segment]
        if (measured is None) != (recorded is None):
            raise ValueError(f"segment shape mismatch: {label}/{segment}")
        if measured is None:
            continue
        if sorted(measured) != sorted(recorded):
            raise ValueError(f"segment metric set mismatch: {label}/{segment}")
        for metric, value in measured.items():
            np.testing.assert_allclose(value, recorded[metric], **SCORE_TOLERANCE)


def _verify_live_blocks(case, row, manifest, belief, steps):
    """Recompute both models' forecasts, scores and gates on every saved block.

    In x64, because that is the precision the run scored these blocks at: a
    replay at another precision would be comparing two different computations
    rather than checking one.
    """
    import jax

    replays, worst = 0, 0.0
    with jax.enable_x64(True):
        return _live_block_replays(case, row, manifest, belief, steps, replays, worst)


def _live_block_replays(case, row, manifest, belief, steps, replays, worst):
    for record in row["blocks"]:
        label = f"{row['directory']}/{record['arrays']}"
        learned = LearnedDynamics.load(case / record["scored_model"])
        if learned.fingerprint() != record["scored_fingerprint"]:
            raise ValueError(f"scored revision fingerprint mismatch: {label}")
        with np.load(case / record["arrays"], allow_pickle=False) as data:
            if data["past_states"].shape[1] != steps["history"] + 1:
                raise ValueError(f"saved block rows lack the consumed context: {label}")
            if data["future_inputs"].shape[1] != steps["horizon"]:
                raise ValueError(f"saved block rows lack the declared horizon: {label}")
            if len(data["targets"]) != record["rows"]:
                raise ValueError(f"saved block row count differs: {label}")
            generic = replay(
                learned._model,
                data["past_states"],
                data["past_inputs"],
                data["future_inputs"],
            )
            worst = max(
                worst, float(np.max(np.abs(generic - data["generic_prediction"])))
            )
            np.testing.assert_allclose(
                generic, data["generic_prediction"], **REPLAY_TOLERANCE
            )
            structured = structured_forecast(
                belief.params, data, manifest["plant"]["sample_interval_s"]
            )
            worst = max(
                worst,
                float(np.max(np.abs(structured - data["structured_prediction"]))),
            )
            np.testing.assert_allclose(
                structured, data["structured_prediction"], **REPLAY_TOLERANCE
            )
            targets = data["targets"]
        fresh = dict(
            generic=platform_measure(generic, targets)["final_step"],
            structured=platform_measure(structured, targets)["final_step"],
        )
        for arm, score in fresh.items():
            for metric in METRICS:
                np.testing.assert_allclose(
                    score[metric], record[arm][metric], **SCORE_TOLERANCE
                )
        gate = live_swap_gate(fresh["generic"], fresh["structured"])
        if gate["passed"] != record["gate"]["passed"]:
            raise ValueError(f"recomputed swap gate differs: {label}")
        replays += 2
    return replays, worst


def _verify_live_swap(row, release_offset, intervals):
    """Recompute the swap from the recorded gates and the declared offset alone.

    Under this tier there is exactly one answer and no clock in it: the first
    block whose gate passed, released the declared number of intervals after
    that block ended, unless that lands past the end of the trial. A run that
    recorded any other swap -- or none where one was due -- is rejected.
    """
    label = row["directory"]
    if row["arm"] == "frozen":
        expected_block, expected_interval = None, None
        if row["swapped"]:
            raise ValueError(f"a frozen arm recorded a swap: {label}")
    else:
        expected_block, expected_interval = live_expected_swap(
            row["blocks"], release_offset, intervals
        )
    if bool(row["swapped"]) != (expected_interval is not None):
        raise ValueError(
            f"the recorded swap does not match the one the gates and the declared "
            f"offset require: {label}"
        )
    if row["swapped"]:
        if row["swap_interval"] != expected_interval:
            raise ValueError(
                f"the recorded swap interval is not the block's stop interval plus "
                f"the declared offset: {label}"
            )
        if row["swap_scored_block"] != expected_block:
            raise ValueError(
                f"the recorded swap did not come from the first block whose gate "
                f"passed: {label}"
            )
        block = row["blocks"][expected_block]
        if row["swap_scored_stop_interval"] != block["stop_interval"]:
            raise ValueError(f"the recorded swap names another block: {label}")
        if row["swap_revision"] != block["scored_revision"]:
            raise ValueError(
                f"the recorded swap adopted a revision that block did not score: "
                f"{label}"
            )
    if row["candidate_revisions"] != len(row["blocks"]):
        raise ValueError(f"recorded revision count differs from the blocks: {label}")
    if row["budget_overruns"] != sum(
        not entry["within_budget"] for entry in row["blocks"]
    ):
        raise ValueError(f"recorded budget overruns differ from the blocks: {label}")
    # The queue bound cannot bite a synchronous drive, so a dropped block would
    # mean the run was not the one this manifest declares.
    if row["dropped_blocks"]:
        raise ValueError(f"a synchronously driven worker dropped a block: {label}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="fit every frozen case and decide")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    measurer = commands.add_parser("platform", help="measure every pinned corpus")
    measurer.add_argument("--manifest", type=Path, required=True)
    measurer.add_argument("--corpora", type=Path, required=True)
    measurer.add_argument("--output", type=Path, required=True)
    tracker = commands.add_parser("control", help="track the frozen Cascade trial set")
    tracker.add_argument("--manifest", type=Path, required=True)
    tracker.add_argument("--output", type=Path, required=True)
    refiner = commands.add_parser(
        "live", help="refit on the flight and swap on held-out evidence"
    )
    refiner.add_argument("--manifest", type=Path, required=True)
    refiner.add_argument("--output", type=Path, required=True)
    checker = commands.add_parser("verify", help="replay a run directory")
    checker.add_argument("directory", type=Path)
    checker.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "the committed reference the run's regression gate is anchored to "
            f"(default: {COMMITTED_REFERENCE} for a synthetic run, "
            f"{COMMITTED_PLATFORM_REFERENCE} for a platform run, "
            f"{COMMITTED_CONTROL_REFERENCE} for a control run and "
            f"{COMMITTED_LIVE_REFERENCE} for a live run)"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.manifest, args.output)
        accepted = result["accepted"]
    elif args.command == "platform":
        result = platform(args.manifest, args.corpora, args.output)
        accepted = result["accepted"]
    elif args.command == "control":
        result = control(args.manifest, args.output)
        accepted = result["accepted"]
    elif args.command == "live":
        result = live(args.manifest, args.output)
        accepted = result["accepted"]
    else:
        result = verify(args.directory, args.reference)
        accepted = result["decision"]["accepted"]
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
