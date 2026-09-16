"""One harness for the generic learner: fit frozen cases, score, decide, replay.

The harness has three tiers, each with its own frozen manifest and digest
constant. ``run`` is the synthetic tier: it fits the consumer recipe end to end
on each case of ``docs/harness/v1.json``, scores its forecasts on independent
recordings, and writes a decision. ``platform`` is the accuracy tier: it fits
the same recipe on each pinned corpus of ``docs/harness/platform-v2.json`` with
whole recordings held out, fits the structured model on exactly the same
training recordings, and scores both on exactly the same held-out rows.
``control`` is the control tier: it collects the frozen Cascade X8 calibration
of ``docs/harness/control-v2.json``, fits both models on it, and tracks the
same reference with each of them through the existing NMPC seam.
``verify`` replays any tier from its saved artifacts and rejects anything
that changed. No command takes tuning options.

    python -m glassbox.experimental.harness run \\
        --manifest docs/harness/v1.json --output DIR
    python -m glassbox.experimental.harness platform \\
        --manifest docs/harness/platform-v2.json --corpora ROOT --output DIR
    python -m glassbox.experimental.harness control \\
        --manifest docs/harness/control-v2.json --output DIR
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

from .default_model import RECIPE, LearnedDynamics, fit, steps_for
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


def anchored_reference(directory, reference=None, committed=COMMITTED_REFERENCE):
    """The committed reference a replay must compare against, never the copy.

    A run copies the reference it compared into its output. That copy is an
    artifact: loosening it would loosen the regression gate on replay. So the
    copy is only ever checked against the committed file, and the committed
    file is what the replay reads. The two must agree about existing and about
    every byte, or the replay refuses. ``committed`` is the tier's own frozen
    reference; ``reference`` overrides it when the replay does not run inside a
    checkout.
    """
    committed = Path(reference) if reference is not None else Path(committed)
    copied = Path(directory) / "reference.json"
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
        if reference is not None or (directory / "reference.json").exists():
            raise ValueError(
                "the control tier declares no regression reference; this is its "
                "first measurement and there is nothing to anchor"
            )
        return verify_control(
            directory, frozen_control_manifest(directory / "manifest.json")
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
    anchor, digest = anchored_reference(directory, reference)
    decision = decide(manifest, rows, anchor, digest)
    saved = read(directory / "decision.json")
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


# --- the platform tier: the corpus adapter ----------------------------------

PLATFORM_MANIFEST_SHA256 = (
    "f4796e1a1bc2120aa0c00067853f13bf2004cf00ca9c5b34322daf0b7ce2f3ac"
)
"""Digest of the frozen platform manifest this module is allowed to run."""

COMMITTED_PLATFORM_MANIFEST = COMMITTED_MANIFEST.parent / "platform-v2.json"
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


def trajectory_segments(recording_id, trajectory, *, dt_s, tolerance_fraction):
    """One canonical trajectory as generic segments, on one uniform time grid.

    A sample interval that departs from ``dt_s`` ends a segment, and so does a
    nonfinite observed row or outgoing command. Nothing is padded or imputed:
    the split runs through ``segments_from_mask`` one uniformly sampled block at
    a time, so every retained row keeps its source row index. ``dt_s`` is the
    corpus's declared sample period; every interval is checked against it here,
    and it is what the whole corpus shares so one collection can hold it.
    """
    rows = observed_rows(trajectory)
    inputs = np.asarray(trajectory.controls, dtype=float)
    time_s = np.asarray(trajectory.time_s, dtype=float)
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("the declared sample interval must be finite and positive")
    uniform = np.abs(np.diff(time_s) - dt_s) <= tolerance_fraction * dt_s
    valid = np.isfinite(rows).all(axis=1)
    valid[:-1] &= np.isfinite(inputs).all(axis=1)
    bounds = np.r_[0, np.flatnonzero(~uniform) + 1, len(time_s)]
    segments = []
    for start, stop in pairwise(bounds):
        block = np.zeros(len(time_s), dtype=bool)
        block[start:stop] = True
        segments.extend(
            segments_from_mask(recording_id, rows, inputs, valid & block, dt_s=dt_s)
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


def platform_decide(manifest, rows, reference=None, reference_sha256=None):
    """Every gate, evaluated from recorded scores alone. Anything unclear fails.

    Structural problems are gate breaches and always fail closed. The accuracy
    rule itself is reported separately, and becomes a gate once the manifest
    says it is enforced. The regression reference, when one was compared, is
    always a gate: no corpus may exceed its recorded generic final-step RMSE by
    more than the manifest's allowance on either metric.

    ``reference_sha256`` identifies the reference file the scores were compared
    against. It is recorded in the decision so a replay can check that it
    compared the same bytes, and it is not otherwise used here.
    """
    declared = {entry["name"]: entry for entry in manifest["corpora"]}
    names = [row.get("corpus") for row in rows]
    breaches, rule_breaches, summary = [], [], {}
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
            summary.setdefault(name, {})[metric] = dict(
                generic=generic,
                comparator=comparator,
                comparator_arm=arm,
                allowance=None if allowance is None else allowance[metric],
                hold_current=_score(row.get("hold_current", {}), metric),
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
                    )
                )
            if allowance is not None and generic > allowance[metric]:
                rule_breaches.append(
                    dict(
                        corpus=name,
                        metric=metric,
                        gate="allowance_final_step_rmse",
                        value=generic,
                        limit=allowance[metric],
                    )
                )
    regressions = []
    if reference is not None:
        relative = manifest["reference"]["relative_tolerance"]
        absolute = manifest["reference"]["absolute_tolerance"]
        recorded = reference.get("final_step_rmse", {})
        for name in sorted(summary):
            for metric in METRICS:
                base = _number(recorded.get(name, {}).get(metric))
                value = summary[name][metric]["generic"]
                if base is None:
                    regressions.append(
                        dict(corpus=name, metric=metric, gate="reference_present")
                    )
                    continue
                limit = base * (1 + relative) + absolute
                if value is None or value > limit:
                    regressions.append(
                        dict(
                            corpus=name,
                            metric=metric,
                            gate="reference_final_step_rmse",
                            value=value,
                            reference=base,
                            limit=limit,
                        )
                    )
    enforced = bool(manifest["decision"]["enforced"])
    rule_met = not breaches and not rule_breaches
    accepted = not breaches and not regressions and (rule_met or not enforced)
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        rule_met=rule_met,
        rule_enforced=enforced,
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
            row["generic"] = fresh["generic"]
            row["hold_current"] = fresh["hold_current"]
            row["structured"] = fresh["structured"]
    anchor, digest = anchored_reference(
        directory, reference, COMMITTED_PLATFORM_REFERENCE
    )
    decision = platform_decide(manifest, rows, anchor, digest)
    saved = read(directory / "decision.json")
    for key in (
        "manifest",
        "decision",
        "accepted",
        "rule_met",
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


# --- the control tier: one Cascade trial set, two arms ----------------------

CONTROL_MANIFEST_SHA256 = (
    "d5f446238a042438eb8c75ba3d432150c27e7ca6b4ecd72eed11fcfaa26e5787"
)
"""Digest of the frozen control manifest this module is allowed to run."""

COMMITTED_CONTROL_MANIFEST = COMMITTED_MANIFEST.parent / "control-v2.json"
"""The frozen control manifest in a source checkout."""

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
    return manifest


def control_reference(initial_state, times, declared):
    """The declared cruise reference, rebuilt from one initial state.

    A straight cruise carried forward at the initial state's own world velocity,
    with a small cosine altitude variation and the climb rate that matches it.
    ``verify`` rebuilds it from the saved initial state rather than believing
    the saved reference rows, so a run cannot score itself against a reference
    it invented.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    times = np.asarray(times, dtype=float)
    states = np.tile(initial_state, (len(times), 1))
    rate = declared["rate_rad_s"]
    states[:, 0:3] += times[:, None] * initial_state[3:6]
    states[:, 2] += declared["altitude_amplitude_m"] * (1.0 - np.cos(rate * times))
    states[:, 5] += declared["climb_rate_amplitude_m_s"] * np.sin(rate * times)
    return states


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
    amplitudes = np.asarray(declared["excitation_amplitudes"], dtype=float)
    rates = np.asarray(declared["excitation_rates_rad_s"], dtype=float)
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=3)
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
        excitation = ramp * amplitudes * np.sin(rates * elapsed + phases)
        command = np.clip(
            np.asarray(raw) + np.r_[0.0, initial_command[1:]] + excitation,
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
    return Trajectory(
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
    )


def control_collection(manifest, loaded):
    """Adapt the calibration recordings into the learner's own channels.

    The same fifteen-channel contract and the same segment adapter the platform
    tier uses. The control manifest declares the channels, and a recording that
    adapts to anything else fails closed here rather than later.
    """
    from .learned_plan import OBSERVED_CHANNELS as PLAN_CHANNELS

    declared = tuple(manifest["telemetry"]["observed_channels"])
    if declared != OBSERVED_CHANNELS or declared != PLAN_CHANNELS:
        raise ValueError("the manifest's observed channels are not the one contract")
    dt_s = manifest["plant"]["sample_interval_s"]
    channels = command_channels(loaded[0][1].spec)
    segments = []
    for name, trajectory in loaded:
        if command_channels(trajectory.spec) != channels:
            raise ValueError("calibration recordings declare different commands")
        segments.extend(
            trajectory_segments(name, trajectory, dt_s=dt_s, tolerance_fraction=1e-6)
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
            compile_signature=plan.compile_signature,
            meaning=(
                "the generic learner at its own fitted horizon, claiming no "
                "covariance and running under the seam's explicit no-evidence "
                "override; validity utilization is zero because no support "
                "envelope is declared, not because one was checked"
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


def _control_trial(manifest, arm, plant, reference_fn, directory):
    """One paced tracking trial: one arm, one freshly reset plant, one reference."""
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
    applied, tick_times, lags, solve_times = [], [], [], []
    solver_used, fallbacks, statuses = [], [], []
    warm_start = None
    failure = None
    started = time.monotonic()
    for index in range(requested):
        scheduled = started + index * dt_s
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        tick = time.monotonic()
        lags.append(max(0.0, tick - scheduled))
        arm.observe(state)
        if arm.ready:
            future = (index + np.arange(arm.prediction_steps + 1)) * dt_s
            reference = ReferenceTrajectory(reference_fn(future))
            result = arm.solve(
                state,
                reference,
                previous,
                warm_start=warm_start,
                deadline_s=deadline_s,
            )
            command = np.asarray(result.command, dtype=float)
            warm_start = result.warm_start
            solve_times.append(float(result.diagnostics.solve_time_s))
            solver_used.append(True)
            fallbacks.append(bool(result.used_fallback))
            statuses.append(str(result.status))
        else:
            # No forecast exists yet: the loop holds the command it is already
            # applying rather than fabricating the history one would need.
            command = previous.copy()
            solve_times.append(0.0)
            solver_used.append(False)
            fallbacks.append(False)
            statuses.append("model_not_ready")
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
    np.savez_compressed(
        directory / "tracking.npz",
        time_s=times,
        states=states_array,
        reference_states=reference_states,
        commands=commands_array,
        tick_times_s=np.asarray(tick_times, dtype=float),
        source_clock_lags_s=np.asarray(lags, dtype=float),
        solve_times_s=np.asarray(solve_times, dtype=float),
        solver_used=np.asarray(solver_used, dtype=bool),
        used_fallback=np.asarray(fallbacks, dtype=bool),
        initial_state=np.asarray(plant.initial_state, dtype=float),
    )
    terminated = failure is not None or len(commands_array) != requested
    row = dict(
        arm=arm.name,
        completed_intervals=len(commands_array),
        requested_intervals=requested,
        terminated=bool(terminated),
        failure=failure,
        tracking_rmse=metrics,
        deadline_misses=int(np.sum(np.asarray(tick_times, dtype=float) > dt_s)),
        solve_deadline_misses=int(
            np.sum(np.asarray(solve_times, dtype=float) > deadline_s)
        ),
        model_not_ready_intervals=int(statuses.count("model_not_ready")),
        fallback_count=int(sum(fallbacks)),
        solver_statuses={
            status: statuses.count(status) for status in sorted(set(statuses))
        },
        maximum_command_bound_violation=bound_violation,
        maximum_source_clock_lag_s=max(lags, default=0.0),
        maximum_tick_time_s=max(tick_times, default=0.0),
        control_elapsed_s=elapsed_s,
        controller=arm.summary(),
        files=_files(directory, ["tracking.npz"]),
    )
    write(directory / "trial.json", row)
    return row


def _control_calibrate(manifest, output):
    """Collect the frozen calibration recordings and fit both arms on them."""
    from glassbox.belief.belief_io import save_dynamics_belief
    from glassbox.core.data import save_trajectory_npz, trajectory_content_digest
    from glassbox.fitting import FitSpec, Holdout
    from glassbox.fitting import fit as structured_fit

    declared = manifest["calibration"]
    spec, model, trim, initial_state, initial_command = control_fixture(manifest)
    plant = _control_plant(manifest, spec, model)
    recordings, names = {}, []
    for seed in declared["seeds"]:
        print(json.dumps(dict(collecting=f"recording-{seed}")), flush=True)
        flight = control_recording(
            manifest, plant, trim, initial_state, initial_command, seed
        )
        name = f"recording-{seed}"
        save_trajectory_npz(flight, output / f"{name}.npz")
        recordings[seed] = flight
        names.append(f"{name}.npz")
    digests = {
        f"recording-{seed}": trajectory_content_digest(flight)
        for seed, flight in recordings.items()
    }
    if len(set(digests.values())) != len(digests):
        raise ValueError("calibration recordings are not distinct")
    training = [(f"recording-{s}", recordings[s]) for s in declared["training_seeds"]]
    reserved = [f"recording-{s}" for s in declared["reserved_seeds"]]

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

    started = time.perf_counter()
    learned = fit(control_collection(manifest, training))
    generic_wall = time.perf_counter() - started
    learned.save(output / "generic.npz")
    if set(learned.report["training"]) | set(learned.report["development"]) != {
        name for name, _ in training
    }:
        raise ValueError("the generic fit did not read the calibration recordings")

    calibration = dict(
        recordings=digests,
        training=[name for name, _ in training],
        reserved=reserved,
        initial_state=initial_state.tolist(),
        initial_command=initial_command.tolist(),
        trim_balance_residual=np.asarray(trim.residual).tolist(),
        cascade=dict(
            spec_hash=manifest["plant"]["spec_hash"],
            source_revision=manifest["plant"]["source_revision"],
            version=manifest["plant"]["version"],
        ),
        structured_fit_wall_seconds=structured_wall,
        generic_fit_wall_seconds=generic_wall,
        generic_fingerprint=learned.fingerprint(),
        generic_report=learned.report,
        files=_files(
            output, names + ["structured.json", "structured_report.json", "generic.npz"]
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


def control_decide(manifest, rows):
    """Every gate, evaluated from recorded trial metrics alone. Anything unclear fails.

    Structural problems always fail closed: a trial missing, duplicated,
    undeclared, terminated, or carrying a metric that is not a finite number.
    The accuracy rule itself is reported separately and becomes a gate only
    when the manifest says it is enforced.
    """
    repetitions = manifest["trial"]["repetitions"]
    expected = {(index, arm) for index in range(repetitions) for arm in CONTROL_ARMS}
    keys = [(row.get("repetition"), row.get("arm")) for row in rows]
    breaches, rule_breaches, summary = [], [], {}
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
            summary[str(index)][metric] = dict(
                generic=generic[metric], structured=structured[metric]
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
                    )
                )
    enforced = bool(manifest["decision"]["enforced"])
    rule_met = not breaches and not rule_breaches
    accepted = not breaches and (rule_met or not enforced)
    return dict(
        manifest=manifest["id"],
        decision="accept" if accepted else "reject",
        accepted=accepted,
        rule_met=rule_met,
        rule_enforced=enforced,
        rule=manifest["decision"]["rule"],
        gates_from=manifest["decision"]["gates_from"],
        trials=len(rows),
        gate_breaches=breaches,
        rule_breaches=rule_breaches,
        tracking_rmse=summary,
        meaning=manifest["decision"]["meaning"],
    )


def control(manifest_path, output):
    """Run the frozen control trial set once, both arms, and write the decision."""
    import jax

    manifest_path, output = Path(manifest_path), Path(output)
    manifest = frozen_control_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
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

    def reference_fn(times):
        return control_reference(initial_state, times, declared)

    rows = []
    for repetition in range(manifest["trial"]["repetitions"]):
        order = manifest["trial"]["arm_order"][
            repetition % len(manifest["trial"]["arm_order"])
        ]
        for name in order:
            print(json.dumps(dict(tracking=f"{repetition}-{name}")), flush=True)
            arm = _control_arm(manifest, name, artifacts)
            _control_prewarm(arm, manifest, warmup, reference_fn)
            plant = _control_tracking_plant(manifest, initial_state, initial_command)
            trial_started = time.perf_counter()
            row = _control_trial(
                manifest,
                arm,
                plant,
                reference_fn,
                output / f"trial-{repetition}" / name,
            )
            row["repetition"] = repetition
            row["trial_wall_seconds"] = time.perf_counter() - trial_started
            row["directory"] = f"trial-{repetition}/{name}"
            write(output / row["directory"] / "trial.json", row)
            rows.append(row)
            write(output / "results.json", rows)
            print(
                json.dumps(
                    dict(
                        trial=f"{repetition}-{name}",
                        tracking_rmse=row["tracking_rmse"],
                        terminated=row["terminated"],
                        deadline_misses=row["deadline_misses"],
                    )
                ),
                flush=True,
            )
    decision = control_decide(manifest, rows)
    decision["wall_seconds"] = time.perf_counter() - started
    decision["calibration"] = dict(
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


def verify_control(directory, manifest):
    """Recompute every control metric and the decision from saved arrays.

    This replay never reruns the plant and never reruns the solver: neither one
    is deterministic under a wall clock, and rerunning either would be a new
    measurement rather than a check of this one. What it does check is
    everything the decision actually read. Every recorded artifact hash is
    recomputed, including the calibration recordings and both fitted models, so
    an altered artifact is rejected. Every metric is recomputed from the saved
    per-interval tracking arrays with the library's own metric code. The
    reference rows are rebuilt from the saved initial state and the manifest's
    declared reference, so a run cannot score itself against a reference of its
    own invention. The manifest is anchored to its frozen digest.
    """
    from glassbox.core.data import load_trajectory_npz, trajectory_content_digest
    from glassbox.core.metrics import state_rmse_metrics

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

    learned = LearnedDynamics.load(directory / "generic.npz")
    if learned.fingerprint() != calibration["generic_fingerprint"]:
        raise ValueError("generic model fingerprint mismatch")

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
        with np.load(case / "tracking.npz", allow_pickle=False) as data:
            states = data["states"]
            commands = data["commands"]
            saved_reference = data["reference_states"]
            tick_times = data["tick_times_s"]
            solve_times = data["solve_times_s"]
            initial_state = data["initial_state"]
            times = data["time_s"]
            if len(states) != len(commands) + 1 or len(times) != len(states):
                raise ValueError(f"saved tracking arrays disagree: {row['directory']}")
            np.testing.assert_allclose(
                times, np.arange(len(states)) * dt_s, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                saved_reference,
                control_reference(initial_state, times, declared),
                rtol=0,
                atol=1e-12,
            )
            fresh = (
                state_rmse_metrics(states[1:], saved_reference[1:])
                if len(commands)
                else None
            )
            misses = int(np.sum(tick_times > dt_s))
            solve_misses = int(np.sum(solve_times > trial["solve_deadline_s"]))
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
        if (
            misses != row["deadline_misses"]
            or solve_misses != (row["solve_deadline_misses"])
        ):
            raise ValueError(f"recomputed deadline misses differ: {row['directory']}")
        terminated = len(commands) != row["requested_intervals"]
        if (
            terminated != bool(row["terminated"])
            or len(commands) != (row["completed_intervals"])
        ):
            raise ValueError(f"recomputed completion differs: {row['directory']}")
        checked += 1
    decision = control_decide(manifest, rows)
    saved = read(directory / "decision.json")
    for key in (
        "manifest",
        "decision",
        "accepted",
        "rule_met",
        "rule_enforced",
        "trials",
    ):
        if decision[key] != saved[key]:
            raise ValueError(f"replayed decision differs: {key}")
    for key in ("gate_breaches", "rule_breaches"):
        if len(decision[key]) != len(saved[key]):
            raise ValueError(f"replayed decision differs: {key}")
    return dict(
        tier="control",
        verified_trials=checked,
        decision=decision,
        meaning=(
            "A replay of saved evidence: recorded hashes, metrics recomputed from "
            "the saved per-interval tracking arrays, and the reference rebuilt "
            "from the saved initial state. The plant and the solver are not "
            "rerun, because rerunning either would be a new measurement rather "
            "than a check of this one."
        ),
    )


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
    checker = commands.add_parser("verify", help="replay a run directory")
    checker.add_argument("directory", type=Path)
    checker.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "the committed reference the run's regression gate is anchored to "
            f"(default: {COMMITTED_REFERENCE} for a synthetic run and "
            f"{COMMITTED_PLATFORM_REFERENCE} for a platform run)"
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
    else:
        result = verify(args.directory, args.reference)
        accepted = result["decision"]["accepted"]
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
