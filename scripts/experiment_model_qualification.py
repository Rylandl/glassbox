"""Falsify a frozen generic learner on platform-neutral synthetic systems.

No vehicle dynamics, controller optimization, or learner tuning. Every result
is decoded to common physical coordinates before comparison. Constants are
research protocol definitions, not additions to the consumer fitting API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

from glassbox.experimental.default_model import _HISTORY_RECIPE as _RECIPE
from glassbox.experimental.default_model import _fit_history as fit
from glassbox.experimental.sequence_collection import (
    SequenceCollection,
    SequenceSegment,
)
from glassbox.experimental.sequence_model import SequenceBatch, sequence_windows

DT = 0.05
STEPS = 160
SEEDS = (101, 202, 303)
EXCITATION = (0.0, 0.02, 0.25)
ENCODINGS = ("identity", "units_offset", "orthogonal", "permuted", "duplicate_first")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@dataclass(frozen=True)
class Encoding:
    name: str
    state_map: np.ndarray
    state_inverse: np.ndarray
    offset: np.ndarray
    input_map: np.ndarray

    def states(self, x):
        return x @ self.state_map + self.offset

    def inputs(self, u):
        return u @ self.input_map

    def decode(self, y):
        return (y - self.offset) @ self.state_inverse


def encoding(name, width=2):
    e, d, f = np.eye(width), np.eye(width), np.eye(width)
    offset = np.zeros(width)
    if name == "units_offset":
        e, d = np.diag([1000.0, 0.001]), np.diag([0.001, 1000.0])
        f, offset = np.diag([0.1, 100.0]), np.array([100.0, -3.0])
    elif name == "orthogonal":
        c, s = np.cos(0.73), np.sin(0.73)
        e = np.array([[c, -s], [s, c]])
        d = e.T
    elif name == "permuted":
        e = d = np.array([[0.0, 1.0], [1.0, 0.0]])
    elif name == "duplicate_first":
        e = np.column_stack((np.eye(2), np.repeat([[1.0], [0.0]], 6, axis=1)))
        d = np.zeros((8, 2))
        d[:2] = np.eye(2)  # Score the original two outputs, without averaging copies.
        offset = np.zeros(8)
    elif name != "identity":
        raise ValueError(name)
    return Encoding(name, e, d, offset, f)


def linear_rollout(initial, inputs, a=1.08, b=0.2):
    state, outputs = np.array(initial, copy=True), []
    for k in range(inputs.shape[1]):
        state = a * state + b * inputs[:, k]
        outputs.append(state.copy())
    return np.stack(outputs, axis=1)


def memory_rollout(initial, inputs, delay=3):
    state, outputs = np.array(initial, copy=True), [np.array(initial, copy=True)]
    for k in range(len(inputs)):
        state = 0.8 * state + (0.2 * inputs[k - delay] if k >= delay else 0.0)
        outputs.append(state.copy())
    return np.stack(outputs)


def nonlinear_step(state, command):
    return np.array(
        [
            0.90 * state[0] + 0.10 * np.tanh(1.5 * command[0]) + 0.025 * state[1],
            0.86 * state[1] + 0.08 * command[1] + 0.04 * np.tanh(state[0]),
        ]
    )


def recordings(family, seed, *, evaluation=False, excitation=0.0):
    result = []
    for index in range(4 if evaluation else 8):
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, int(evaluation), index])
        )
        width = 2 if family == "encoding" else 1
        state = rng.uniform(-1.5, 1.5, size=width)
        xs, us = [state.copy()], []
        previous = np.zeros(width)
        for k in range(STEPS):
            if family == "command":
                command = -0.5 * state + excitation * rng.uniform(-1, 1, size=1)
                state = 1.08 * state + 0.2 * command
            elif family == "memory":
                command = rng.uniform(-1, 1, size=1)
                state = 0.8 * state + (0.2 * us[k - 3] if k >= 3 else 0.0)
            elif family == "encoding":
                command = np.clip(0.7 * previous + 0.35 * rng.normal(size=2), -1, 1)
                state = nonlinear_step(state, command)
            else:
                raise ValueError(family)
            previous = command
            xs.append(state.copy())
            us.append(command.copy())
        result.append(
            SequenceSegment(
                f"{'evaluation' if evaluation else 'fit'}-{index}",
                "whole",
                np.array(xs),
                np.array(us),
                DT,
            )
        )
    return tuple(result)


def collection(rows, transform, family):
    return SequenceCollection(
        tuple(
            SequenceSegment(
                r.recording_id,
                r.segment_id,
                transform.states(r.states),
                transform.inputs(r.inputs),
                r.dt_s,
            )
            for r in rows
        ),
        configuration_id=f"synthetic-qualification-{family}-{transform.name}",
        state_channels=tuple(
            f"encoded_x{i} [{transform.name},unitless]"
            for i in range(transform.state_map.shape[1])
        ),
        input_channels=tuple(
            f"encoded_u{i} [{transform.name},requested input]"
            for i in range(transform.input_map.shape[1])
        ),
    )


def windows(rows):
    batches = []
    for row in rows:
        origins = np.arange(2, len(row.states) - 5, 5)
        batches.append(
            sequence_windows(
                row.states,
                row.inputs,
                origins,
                history_steps=2,
                horizon_steps=5,
                dt_s=DT,
            )
        )
    return SequenceBatch(
        **{
            k: np.concatenate([getattr(b, k) for b in batches])
            for k in ("past_states", "past_inputs", "future_inputs", "future_states")
        },
        dt_s=DT,
    )


def predict(model, transform, batch, inputs=None):
    future = batch.future_inputs if inputs is None else inputs
    raw = np.asarray(
        model.predict(
            transform.states(batch.past_states),
            transform.inputs(batch.past_inputs),
            transform.inputs(future),
        )
    )
    return raw, transform.decode(raw)


def rmse(prediction, target):
    return np.sqrt(
        np.mean(np.sum((prediction - target) ** 2, axis=-1), axis=0)
    ).tolist()


def memory_probe():
    # The +/- pulse lies just outside the recipe's supplied command history.
    full_inputs = np.zeros((2, 8, 1))
    full_inputs[:, 0, 0] = [-1.0, 1.0]
    states = np.stack([memory_rollout(np.zeros(1), u) for u in full_inputs])
    return states[:, :4], full_inputs[:, :3], full_inputs[:, 3:], states[:, 4:]


def case(family, seed, variant, output):
    directory = output / f"{family}-{seed}-{variant}"
    directory.mkdir()
    excitation = float(variant) if family == "command" else 0.0
    transform = (
        encoding(variant, 2) if family == "encoding" else encoding("identity", 1)
    )
    calibration = recordings(family, seed, excitation=excitation)
    model = fit(collection(calibration, transform, family))
    assert model.report["recipe"] == _RECIPE
    model.save(directory / "model.npz")
    write(directory / "fit-report.json", model.report)
    evaluated = windows(
        recordings(family, seed, evaluation=True, excitation=excitation)
    )
    encoded, predicted = predict(model, transform, evaluated)
    arrays = dict(
        past_states=evaluated.past_states,
        past_inputs=evaluated.past_inputs,
        future_inputs=evaluated.future_inputs,
        targets=evaluated.future_states,
        encoded_prediction=encoded,
        prediction=predicted,
        state_map=transform.state_map,
        state_inverse=transform.state_inverse,
        offset=transform.offset,
        input_map=transform.input_map,
    )
    report = dict(
        family=family,
        seed=seed,
        variant=variant,
        status="complete",
        fingerprint=model.fingerprint(),
        recipe=model.report["recipe"]["id"],
        selected_step=model.report["optimization"]["selected_step"],
        forecast_rmse=rmse(predicted, evaluated.future_states),
        development_mse=model.report["optimization"]["validation_rollout_mse"],
        training_keys=[
            (k.recording_id, k.segment_id, k.origin) for k in model._train.keys
        ],
        development_keys=[
            (k.recording_id, k.segment_id, k.origin) for k in model._development.keys
        ],
    )
    if family == "command":
        query = windows(recordings("command", seed, evaluation=True, excitation=0.0))
        _, nominal = predict(model, transform, query)
        paired, truths, inputs = [], [], []
        for sign in (-1, 1):
            u = query.future_inputs.copy()
            u[:, 0] += sign * 0.05
            _, prediction = predict(model, transform, query, u)
            paired.append(prediction)
            truths.append(linear_rollout(query.past_states[:, -1], u))
            inputs.append(u)
        paired, truths, inputs = map(np.array, (paired, truths, inputs))
        slope = (paired[1] - paired[0]) / 0.1
        true_slope = (truths[1] - truths[0]) / 0.1
        slope_error = np.asarray(rmse(slope, true_slope)) / np.asarray(
            rmse(true_slope, 0)
        )
        sampled_u = model._train.batch.future_inputs
        in_range = (inputs[:, :, 0] >= sampled_u.min((0, 1))) & (
            inputs[:, :, 0] <= sampled_u.max((0, 1))
        )
        report.update(
            common_history_forecast_rmse=rmse(nominal, query.future_states),
            perturbed_forecast_rmse=rmse(
                paired.reshape(-1, 5, 1), truths.reshape(-1, 5, 1)
            ),
            command_slope_relative_rmse=slope_error.tolist(),
            perturbed_commands_in_marginal_training_range_fraction=float(
                in_range.all(-1).mean()
            ),
        )
        arrays.update(
            query_past_states=query.past_states,
            query_past_inputs=query.past_inputs,
            query_future_inputs=inputs,
            query_prediction=paired,
            query_targets=truths,
            nominal_prediction=nominal,
            nominal_targets=query.future_states,
        )
    elif family == "memory":
        px, pu, uf, target = memory_probe()
        full = np.asarray(model.predict(px, pu, uf))
        short = np.asarray(model.predict(px[:, -3:], pu[:, -2:], uf))
        np.testing.assert_array_equal(full, short)
        np.testing.assert_array_equal(full[0], full[1])
        floor = np.abs(target[1, :, 0] - target[0, :, 0]) / 2
        measured = np.array(rmse(full, target))
        assert np.all(measured >= floor - 1e-12)
        report.update(
            paired_probe_rmse=measured.tolist(),
            paired_probe_minimum_possible_rmse=floor.tolist(),
            independent_input_first_step_noise_floor=0.2 / np.sqrt(3),
            longer_supplied_history_discarded=True,
        )
        arrays.update(
            probe_past_states=px,
            probe_past_inputs=pu,
            probe_future_inputs=uf,
            probe_targets=target,
            probe_prediction=full,
        )
    np.savez_compressed(directory / "evaluation.npz", **arrays)
    write(directory / "result.json", report)
    return report


def validate_plan(plan):
    assert plan["recipe"] == _RECIPE and plan["dataset_seeds"] == list(SEEDS)
    assert plan["calibration_recordings"] == 8 and plan["evaluation_recordings"] == 4
    assert plan["steps_per_recording"] == STEPS and plan["dt_s"] == DT
    assert plan["evaluation_origin_stride"] == 5
    a, b, c = (
        plan["experiments"][n]
        for n in ("command_response", "hidden_memory", "encoding")
    )
    assert (a["a"], a["b"], a["feedback_gain"]) == (1.08, 0.2, -0.5)
    assert a["excitation_uniform_half_widths"] == list(EXCITATION)
    assert a["query_first_command_offsets"] == [-0.05, 0.05]
    assert a["alternative_b_witnesses"] == [0.05, 0.2, 0.5]
    assert (b["output_decay"], b["input_gain"], b["input_delay_steps"]) == (0.8, 0.2, 3)
    assert c["variants"] == list(ENCODINGS)
    assert c["state_unit_scales"] == [1000.0, 0.001] and c["state_offsets"] == [
        100.0,
        -3.0,
    ]
    assert c["input_unit_scales"] == [0.1, 100.0] and c["rotation_rad"] == 0.73
    assert c["extra_first_channel_copies"] == 6


def main(args):
    if not jax.config.jax_enable_x64:
        raise ValueError("recorded benchmark requires JAX_ENABLE_X64=1")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", plan)
    root = Path(__file__).resolve().parents[1]
    files = [
        *sorted((root / "src/glassbox").rglob("*.py")),
        Path(__file__),
        root / "tests/test_model_qualification.py",
    ]
    with zipfile.ZipFile(
        args.output / "executed-sources.zip", "x", zipfile.ZIP_DEFLATED
    ) as z:
        for path in files:
            z.write(path, str(path.relative_to(root)))
    write(
        args.output / "sources.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    write(
        args.output / "environment.json",
        dict(
            python=sys.version,
            platform=platform.platform(),
            jax=jax.__version__,
            numpy=np.__version__,
            x64=jax.config.jax_enable_x64,
            backend=jax.default_backend(),
        ),
    )
    reports = []
    for family, variants in (
        ("command", tuple(map(str, EXCITATION))),
        ("memory", ("hidden",)),
        ("encoding", ENCODINGS),
    ):
        for seed in SEEDS:
            for variant in variants:
                print(json.dumps(dict(starting=[family, seed, variant])), flush=True)
                try:
                    result = case(family, seed, variant, args.output)
                except (
                    ValueError,
                    FloatingPointError,
                    AssertionError,
                    np.linalg.LinAlgError,
                ) as error:
                    result = dict(
                        family=family,
                        seed=seed,
                        variant=variant,
                        status="failed",
                        error=repr(error),
                    )
                    write(
                        args.output / f"{family}-{seed}-{variant}" / "result.json",
                        result,
                    )
                reports.append(result)
                write(args.output / "results.json", reports)
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in ("training_keys", "development_keys")
                        }
                    ),
                    flush=True,
                )
    assert len(reports) == 27
    witnesses = []
    for seed in SEEDS:
        for b in (0.05, 0.2, 0.5):
            maximum = 0.0
            for row in recordings("command", seed, evaluation=True, excitation=0.0):
                predicted_next = (0.98 + 0.5 * b) * row.states[:-1] + b * row.inputs
                maximum = max(
                    maximum, float(np.max(np.abs(predicted_next - row.states[1:])))
                )
            witnesses.append(
                dict(
                    seed=seed,
                    a=0.98 + 0.5 * b,
                    b=b,
                    maximum_recorded_transition_difference=maximum,
                )
            )
    write(args.output / "identifiability-witnesses.json", witnesses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
