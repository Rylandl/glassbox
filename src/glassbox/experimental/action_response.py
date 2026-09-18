"""Frozen, no-fit diagnostic of the adopted learner's command response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .harness import control_initial_state
from .learned_plan import observed_from_state
from .qualification import _context, arrays, digest, write_json
from .qualification_control import CascadeEquations
from .quasi_newton_qualification import check_environment
from .solver_termination import exact, exact_arrays
from .task_qualification import input_snapshot

ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = ROOT / "docs/harness/action-response-v1.json"
PLAN_SHA256 = "4b40ee76aa5e0e7b1b5a2eafdbce90b95566ba45dcc769e5c3c04453df52dba6"
PROBES = (
    "baseline",
    "command_0_minus",
    "command_0_plus",
    "command_1_minus",
    "command_1_plus",
    "command_2_minus",
    "command_2_plus",
)


def frozen_plan():
    """Check this protocol and inherited bytes without historical AST gates."""
    raw = PLAN_PATH.read_bytes()
    if digest(raw) != PLAN_SHA256:
        raise ValueError("action-response frozen plan differs")
    plan = json.loads(raw)
    for name, expected in plan["source_sha256"].items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f"action-response inherited source differs: {name}")
    if set(plan["input_paths"]) != set(plan["input_sha256"]):
        raise ValueError("action-response input declarations disagree")
    exact(plan["probes"]["order"], list(PROBES))
    return plan, raw


def command_probes(baseline, minimum, maximum, fraction=0.05):
    """Seven feasible float32 tapes, changing just one physical channel."""
    baseline, minimum, maximum = (
        np.asarray(value, dtype=np.float32) for value in (baseline, minimum, maximum)
    )
    if (
        baseline.ndim != 2
        or baseline.shape[1] != 3
        or baseline.shape[0] == 0
        or minimum.shape != (3,)
        or maximum.shape != (3,)
        or not all(np.isfinite(a).all() for a in (baseline, minimum, maximum))
        or not np.all(minimum < maximum)
        or not np.isfinite(fraction)
        or not 0 <= fraction <= 1
        or np.any(baseline < minimum)
        or np.any(baseline > maximum)
    ):
        raise ValueError("invalid baseline, command bounds or probe fraction")
    result = np.repeat(baseline[None], 7, axis=0)
    weight = np.float32(fraction)
    for channel in range(3):
        for sign, bound in enumerate((minimum, maximum)):
            result[1 + 2 * channel + sign, :, channel] = (
                np.float32(1) - weight
            ) * baseline[:, channel] + weight * bound[channel]
    if np.any(result < minimum) or np.any(result > maximum):
        raise ValueError("float32 probe exceeds its declared bounds")
    return result


def _vectors(predicted, true):
    predicted, true = (
        np.asarray(value, dtype=np.float64) for value in (predicted, true)
    )
    if predicted.shape != true.shape or predicted.ndim < 1 or not predicted.shape[-1]:
        raise ValueError(
            "prediction and truth require identical nonempty vector shapes"
        )
    predicted, true = (a.reshape(-1, a.shape[-1]) for a in (predicted, true))
    valid = np.isfinite(predicted).all(axis=1) & np.isfinite(true).all(axis=1)
    counts = dict(
        total_count=int(valid.size),
        valid_count=int(valid.sum()),
        invalid_count=int((~valid).sum()),
    )
    return predicted[valid], true[valid], counts


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return dict(
        count=int(values.size),
        median=float(np.median(values)) if values.size else None,
        p10=float(np.percentile(values, 10)) if values.size else None,
        p90=float(np.percentile(values, 90)) if values.size else None,
    )


def absolute_statistics(predicted, true):
    """Vector-complete valid subset, retaining the failed-vector denominator."""
    predicted, true, counts = _vectors(predicted, true)
    error = predicted - true
    return counts | dict(
        component_rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        mean_vector_error=float(np.mean(np.linalg.norm(error, axis=1)))
        if len(error)
        else None,
    )


def response_statistics(predicted_delta, true_delta, threshold=1e-6):
    """Direction and gain in one group's units; weak responses stay counted."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("weak-response threshold must be finite and nonnegative")
    predicted, true, counts = _vectors(predicted_delta, true_delta)
    pn, tn = (np.linalg.norm(a, axis=1) for a in (predicted, true))
    error = np.linalg.norm(predicted - true, axis=1)
    nonweak = tn > threshold
    direction = nonweak & (pn > threshold)
    dot = np.sum(predicted * true, axis=1)
    gain = dot[nonweak] / tn[nonweak] ** 2
    return counts | dict(
        weak_true_count=int((~nonweak).sum()),
        weak_predicted_count=int((pn <= threshold).sum()),
        normalizable_count=int(nonweak.sum()),
        cosine_count=int(direction.sum()),
        rms_vector_error=float(np.sqrt(np.mean(error**2))) if len(error) else None,
        true_response_norm=distribution(tn),
        predicted_response_norm=distribution(pn),
        cosine=distribution(dot[direction] / (pn[direction] * tn[direction])),
        magnitude_ratio=distribution(pn[nonweak] / tn[nonweak]),
        projected_gain=distribution(gain),
        relative_response_error=distribution(error[nonweak] / tn[nonweak]),
        negative_projected_gain_fraction=float(np.mean(gain < 0))
        if len(gain)
        else None,
    )


def _forecast(call, shape, *, origin, probe_name, predictor, failures):
    """Keep a failed complete sequence in its original slot as NaNs."""
    try:
        prediction = np.asarray(call(), dtype=np.float32)
        if prediction.shape != shape:
            raise ValueError(f"forecast shape {prediction.shape} differs from {shape}")
    except Exception as error:
        failures.append(
            dict(
                origin=int(origin),
                probe=probe_name,
                predictor=predictor,
                kind="exception",
                exception_type=type(error).__name__,
                message=str(error),
            )
        )
        return np.full(shape, np.nan, dtype=np.float32), False
    if not np.isfinite(prediction).all():
        failures.append(
            dict(
                origin=int(origin),
                probe=probe_name,
                predictor=predictor,
                kind="nonfinite",
            )
        )
        return np.full(shape, np.nan, dtype=np.float32), False
    return prediction, True


def _source_trial(plan, inputs, context, index):
    row = json.loads(inputs[f"trial-{index}/trial.json"])
    tracking = arrays(inputs[f"trial-{index}/tracking.npz"])
    count = plan["population"]["intervals"]
    seed = plan["population"]["initial_state_seeds"][index]
    exact(row["arm"], plan["population"]["source_arm"])
    exact(row["initial_state_seed"], seed)
    exact(row["repetition"], index)
    exact(row["completed_intervals"], count)
    exact(row["requested_intervals"], count)
    exact(row["terminated"], False)
    exact(row["failure"], None)
    exact(row["files"]["tracking.npz"], digest(inputs[f"trial-{index}/tracking.npz"]))
    states, commands = tracking["states"], tracking["commands"]
    if states.shape != (count + 1, 13) or commands.shape != (count, 3):
        raise ValueError("source trajectory shape differs")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("source trajectory is nonfinite")
    initial = control_initial_state(context.manifest, context.initial_state, seed)
    np.testing.assert_array_equal(tracking["initial_state"], initial)
    np.testing.assert_array_equal(states[0], initial)
    np.testing.assert_array_equal(
        tracking["reference_anchor_state"], context.initial_state
    )
    minimum, maximum = (
        np.asarray(context.manifest["telemetry"][key], dtype=np.float64)
        for key in ("command_minimum", "command_maximum")
    )
    if np.any(commands < minimum) or np.any(commands > maximum):
        raise ValueError("source commands exceed declared bounds")
    return states, commands.astype(np.float32), initial


def _trial(plan, inputs, context, equations, index, *, predict=None):
    predict = context.learned.predict if predict is None else predict
    population = plan["population"]
    origins = np.asarray(population["origins"], dtype=np.int64)
    history, horizon = population["history_steps"], population["horizon_steps"]
    states, commands, initial = _source_trial(plan, inputs, context, index)
    if (
        len(set(origins.tolist())) != len(origins)
        or np.any(origins < history)
        or np.any(origins + horizon > len(commands))
    ):
        raise ValueError("declared origins lack complete histories or futures")
    state = equations.reset(jnp.asarray(initial), jnp.asarray(context.initial_command))
    causal = {}
    # The historical plant saves the declared float64 initial observation, then
    # the float32 canonical output after every interval. Reproduce that boundary.
    replayed = [initial]
    for k, command in enumerate(commands):
        if k in origins:
            causal[k] = state
        state = equations.advance(state, jnp.asarray(command))
        replayed.append(np.asarray(equations.canonical(state)))
    replayed = np.asarray(replayed, dtype=np.float64)
    np.testing.assert_array_equal(replayed, states)
    observed = np.stack([np.asarray(observed_from_state(row)) for row in states])
    minimum, maximum = (
        context.manifest["telemetry"][key]
        for key in ("command_minimum", "command_maximum")
    )
    result = dict(
        origins=origins,
        observations=np.stack([observed[k - history : k + 1] for k in origins]),
        past_commands=np.stack([commands[k - history : k] for k in origins]),
        future_commands=np.stack(
            [
                command_probes(
                    commands[k : k + horizon],
                    minimum,
                    maximum,
                    plan["probes"]["fraction_toward_bound"],
                )
                for k in origins
            ]
        ),
        replayed_states=replayed,
    )
    result["command_deltas"] = (
        result["future_commands"] - result["future_commands"][:, :1]
    )
    shape = (len(origins), len(PROBES), horizon)
    for key, width in (("learned", 15), ("reference", 15), ("canonical", 13)):
        result[key] = np.full((*shape, width), np.nan, dtype=np.float32)
    for key in ("learner_valid", "reference_valid"):
        result[key] = np.zeros(shape[:2], dtype=bool)
    failures = []
    for i, origin in enumerate(origins):
        for p, probe_name in enumerate(PROBES):
            future = result["future_commands"][i, p]

            def reference_forecast(origin=origin, future=future):
                current, canonical = causal[origin], []
                for command in future:
                    current = equations.advance(current, jnp.asarray(command))
                    canonical.append(np.asarray(equations.canonical(current)))
                return np.stack(canonical)

            canonical, valid = _forecast(
                reference_forecast,
                (horizon, 13),
                origin=origin,
                probe_name=probe_name,
                predictor="reference",
                failures=failures,
            )
            result["canonical"][i, p] = canonical
            result["reference_valid"][i, p] = valid
            if p == 0:
                if not valid:
                    raise ValueError("factual public-equation replay failed")
                np.testing.assert_array_equal(
                    canonical, states[origin + 1 : origin + horizon + 1]
                )
            if valid:
                result["reference"][i, p] = np.stack(
                    [np.asarray(observed_from_state(row)) for row in canonical]
                )
            prediction, valid = _forecast(
                lambda i=i, future=future: predict(
                    result["observations"][i], result["past_commands"][i], future
                ),
                (horizon, 15),
                origin=origin,
                probe_name=probe_name,
                predictor="learner",
                failures=failures,
            )
            result["learned"][i, p] = prediction
            result["learner_valid"][i, p] = valid
    if jax.config.x64_enabled:
        raise ValueError("action-response changed the default numerical precision")
    return result, failures


def probe(plan, inputs):
    """Freshly reconstruct every origin and execute the complete forecast roster."""
    context = _context(inputs)
    population = plan["population"]
    exact(context.learned.history_steps, population["history_steps"])
    exact(context.learned.horizon_steps, population["horizon_steps"])
    exact(context.learned.contract["dt_s"], population["sample_interval_s"])
    exact(
        context.manifest["plant"]["sample_interval_s"], population["sample_interval_s"]
    )
    metadata = dict(
        fingerprint=context.learned.fingerprint(),
        archive_sha256=digest(inputs["control/generic.npz"]),
        recipe=context.learned.recipe,
        contract=context.learned.contract,
        history_steps=context.learned.history_steps,
        horizon_steps=context.learned.horizon_steps,
        source_manifest_recipe=json.loads(inputs["control/manifest.json"])["recipe"],
        recipe_meaning="actual loaded learner recipe; historical source manifest is provenance only",
    )
    equations = CascadeEquations(context.model)
    predict = jax.jit(context.learned.predict)
    trials, failures = [], []
    for index in range(population["trials"]):
        print(f"action-response trial-{index}: replay and forecasts", flush=True)
        trial, failed = _trial(plan, inputs, context, equations, index, predict=predict)
        trials.append(trial)
        failures.append(failed)
    return trials, failures, metadata


def _validate_trial(plan, trial, failures):
    population = plan["population"]
    n, h, p = (
        len(population["origins"]),
        population["horizon_steps"],
        population["history_steps"],
    )
    shapes = dict(
        origins=((n,), np.int64),
        observations=((n, p + 1, 15), np.float32),
        past_commands=((n, p, 3), np.float32),
        future_commands=((n, 7, h, 3), np.float32),
        command_deltas=((n, 7, h, 3), np.float32),
        learned=((n, 7, h, 15), np.float32),
        reference=((n, 7, h, 15), np.float32),
        canonical=((n, 7, h, 13), np.float32),
        learner_valid=((n, 7), bool),
        reference_valid=((n, 7), bool),
        replayed_states=((population["intervals"] + 1, 13), np.float64),
    )
    if set(trial) != set(shapes):
        raise ValueError("action-response trial array roster differs")
    for key, (shape, dtype) in shapes.items():
        if trial[key].shape != shape or trial[key].dtype != np.dtype(dtype):
            raise ValueError(f"action-response trial array shape/dtype differs: {key}")
    np.testing.assert_array_equal(trial["origins"], population["origins"])
    np.testing.assert_array_equal(
        trial["command_deltas"],
        trial["future_commands"] - trial["future_commands"][:, :1],
    )
    for key in (
        "observations",
        "past_commands",
        "future_commands",
        "command_deltas",
        "replayed_states",
    ):
        if not np.isfinite(trial[key]).all():
            raise ValueError(f"action-response input array is nonfinite: {key}")
    expected_failures = set()
    for predictor, key, validity in (
        ("learner", "learned", "learner_valid"),
        ("reference", "reference", "reference_valid"),
    ):
        for i, origin in enumerate(trial["origins"]):
            for p, name in enumerate(PROBES):
                output = trial[key][i, p]
                if predictor == "reference":
                    canonical = trial["canonical"][i, p]
                    if trial[validity][i, p] and not np.isfinite(canonical).all():
                        raise ValueError("valid reference canonical state is nonfinite")
                    if not trial[validity][i, p] and not np.isnan(canonical).all():
                        raise ValueError("failed reference canonical state must be NaN")
                if trial[validity][i, p]:
                    if not np.isfinite(output).all():
                        raise ValueError("valid forecast is nonfinite")
                else:
                    if not np.isnan(output).all():
                        raise ValueError(
                            "failed forecast must preserve a complete NaN slot"
                        )
                    expected_failures.add((int(origin), name, predictor))
    if not isinstance(failures, list):
        raise ValueError("forecast failures must be a list")
    actual_failures = []
    for failure in failures:
        if not isinstance(failure, dict) or failure.get("kind") not in (
            "exception",
            "nonfinite",
        ):
            raise ValueError("forecast failure schema differs")
        fields = {"origin", "probe", "predictor", "kind"}
        if failure["kind"] == "exception":
            fields |= {"exception_type", "message"}
            if not all(
                isinstance(failure.get(k), str) for k in ("exception_type", "message")
            ):
                raise ValueError("forecast exception record differs")
        if set(failure) != fields or type(failure["origin"]) is not int:
            raise ValueError("forecast failure fields differ")
        actual_failures.append(
            (failure["origin"], failure["probe"], failure["predictor"])
        )
    if (
        len(set(actual_failures)) != len(actual_failures)
        or set(actual_failures) != expected_failures
    ):
        raise ValueError("forecast failure roster differs")
    if not trial["reference_valid"][:, 0].all():
        raise ValueError("factual reference must replay completely")
    for i, origin in enumerate(trial["origins"]):
        np.testing.assert_array_equal(
            trial["canonical"][i, 0],
            trial["replayed_states"][origin + 1 : origin + h + 1],
        )


def _action_size(deltas):
    return dict(
        sequences=int(np.prod(deltas.shape[:-2])),
        fully_inactive_sequences=int(np.all(deltas == 0, axis=(-2, -1)).sum()),
        maximum_absolute_by_channel=np.max(
            np.abs(deltas), axis=tuple(range(deltas.ndim - 1))
        )
        .astype(float)
        .tolist(),
        rms_by_channel=np.sqrt(
            np.mean(deltas.astype(float) ** 2, axis=tuple(range(deltas.ndim - 1)))
        ).tolist(),
    )


def _summary(plan, trial):
    learned, reference = (
        trial[key].astype(np.float64) for key in ("learned", "reference")
    )
    groups = {}
    h = plan["population"]["horizon_steps"]
    threshold = plan["probes"]["weak_response_vector_norm"]
    for group, (start, stop) in plan["metrics"]["groups"].items():
        predicted, true = learned[..., start:stop], reference[..., start:stop]
        pd, td = predicted[:, 1:] - predicted[:, :1], true[:, 1:] - true[:, :1]
        absolute = {
            label: [
                absolute_statistics(a[..., step, :], b[..., step, :])
                for step in range(h)
            ]
            for label, a, b in (
                ("baseline", predicted[:, 0], true[:, 0]),
                ("all_sequences", predicted, true),
            )
        }
        response = {}
        for label, a, b in [
            ("pooled_six_probes", pd, td),
            *[
                (name, pd[:, i : i + 1], td[:, i : i + 1])
                for i, name in enumerate(PROBES[1:])
            ],
        ]:
            response[label] = dict(
                by_horizon=[
                    response_statistics(a[..., step, :], b[..., step, :], threshold)
                    for step in range(h)
                ],
                full_horizon=response_statistics(
                    a.reshape(*a.shape[:-2], -1),
                    b.reshape(*b.shape[:-2], -1),
                    threshold,
                ),
            )
        groups[group] = dict(
            units=plan["metrics"]["group_units"][group],
            absolute=absolute,
            response=response,
        )
    return dict(
        origins=len(trial["origins"]),
        forecast_slots=int(trial["learner_valid"].size),
        learner_valid=int(trial["learner_valid"].sum()),
        learner_invalid=int((~trial["learner_valid"]).sum()),
        reference_valid=int(trial["reference_valid"].sum()),
        reference_invalid=int((~trial["reference_valid"]).sum()),
        action_size={
            "pooled_six_probes": _action_size(trial["command_deltas"][:, 1:]),
            **{
                name: _action_size(trial["command_deltas"][:, i])
                for i, name in enumerate(PROBES)
                if i
            },
        },
        groups=groups,
    )


def report(plan, trials, failures, model_metadata):
    population = plan["population"]
    if len(trials) != population["trials"] or len(failures) != len(trials):
        raise ValueError("action-response trial roster differs")
    for trial, failed in zip(trials, failures, strict=True):
        _validate_trial(plan, trial, failed)
    if sum(len(t["origins"]) for t in trials) != population["total_origins"]:
        raise ValueError("action-response origin count differs")
    if (
        sum(t["learner_valid"].size for t in trials)
        != plan["probes"]["forecast_sequences"]
    ):
        raise ValueError("action-response sequence count differs")
    pooled = {
        key: np.concatenate([trial[key] for trial in trials]) for key in trials[0]
    }
    return dict(
        plan=plan["id"],
        plan_sha256=PLAN_SHA256,
        no_fit=True,
        learner_changed=False,
        controller_changed=False,
        application_qualified=False,
        meaning="fixed correlated origins of four oracle-driven simulator trajectories; descriptive diagnostics, not independent systems or a controller qualification",
        model=model_metadata,
        horizon_seconds=[
            (step + 1) * population["sample_interval_s"]
            for step in range(population["horizon_steps"])
        ],
        trials=[
            dict(
                index=i,
                seed=population["initial_state_seeds"][i],
                failures=failures[i],
                **_summary(plan, trial),
            )
            for i, trial in enumerate(trials)
        ],
        pooled=_summary(plan, pooled),
    )


def artifact_names(plan):
    return {
        "manifest.json",
        "run.json",
        "environment.json",
        "model.json",
        "report.json",
        *("inputs/" + name for name in plan["input_paths"]),
        *(
            f"trial-{i}/{name}"
            for i in range(plan["population"]["trials"])
            for name in ("arrays.npz", "failures.json")
        ),
    }


def run(artifacts, output):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    inputs = input_snapshot(plan, artifacts)
    output = Path(output)
    if output.exists():
        raise ValueError("action-response output must not already exist")
    trials, failures, metadata = probe(plan, inputs)
    result = report(plan, trials, failures, metadata)
    output.mkdir(parents=True)
    (output / "manifest.json").write_bytes(raw)
    for name, data in inputs.items():
        path = output / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for i, (trial, failed) in enumerate(zip(trials, failures, strict=True)):
        directory = output / f"trial-{i}"
        directory.mkdir()
        np.savez_compressed(directory / "arrays.npz", **trial)
        write_json(directory / "failures.json", failed)
    write_json(output / "environment.json", environment)
    write_json(output / "model.json", metadata)
    write_json(output / "report.json", result)
    write_json(
        output / "run.json",
        dict(
            plan_sha256=PLAN_SHA256,
            files={
                name: digest((output / name).read_bytes())
                for name in sorted(artifact_names(plan) - {"run.json"})
            },
        ),
    )
    return result


def verify(directory):
    plan, raw = frozen_plan()
    environment = check_environment(plan)
    directory = Path(directory)
    names = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    if names != artifact_names(plan):
        raise ValueError("action-response artifact roster differs")
    saved = {name: (directory / name).read_bytes() for name in names}
    if saved["manifest.json"] != raw:
        raise ValueError("action-response saved plan differs")
    exact(
        json.loads(saved["run.json"]),
        dict(
            plan_sha256=PLAN_SHA256,
            files={name: digest(saved[name]) for name in sorted(names - {"run.json"})},
        ),
    )
    exact(json.loads(saved["environment.json"]), environment)
    inputs = input_snapshot(plan, directory / "inputs", saved=True)
    trials = [
        arrays(saved[f"trial-{i}/arrays.npz"])
        for i in range(plan["population"]["trials"])
    ]
    failures = [
        json.loads(saved[f"trial-{i}/failures.json"]) for i in range(len(trials))
    ]
    metadata = json.loads(saved["model.json"])
    result = report(plan, trials, failures, metadata)
    exact(json.loads(saved["report.json"]), result)
    fresh, fresh_failures, fresh_metadata = probe(plan, inputs)
    exact(metadata, fresh_metadata)
    exact(failures, fresh_failures)
    for saved_trial, fresh_trial in zip(trials, fresh, strict=True):
        exact_arrays(saved_trial, fresh_trial)
    exact(result, report(plan, fresh, fresh_failures, fresh_metadata))
    return dict(
        verified_trials=len(trials),
        verified_origins=plan["population"]["total_origins"],
        verified_forecast_sequences=plan["probes"]["forecast_sequences"],
        exact_physical_replay=True,
        exact_learner_replay=True,
        report=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--artifacts", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    replay = commands.add_parser("verify")
    replay.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = (
        run(args.artifacts, args.output)
        if args.command == "run"
        else verify(args.directory)
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
