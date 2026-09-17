"""Run the public generic fit, predict, save/load and update workflow.

A synthetic two-signal system supplies distinct recordings. This demonstrates
the interface and data roles, not performance on an unseen physical system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from glassbox import LearnedDynamics, SequenceCollection, SequenceSegment, fit
from glassbox.io.recordings import save_recordings
from glassbox.workflows.forecast import evaluate

CONFIGURATION_ID = "onboarding-demo-v1"
STATE_CHANNELS = ("signal_a [unitless]", "signal_b [unitless]")
INPUT_CHANNELS = ("command [unitless,requested]",)


def recording(seed: int) -> SequenceSegment:
    generator = np.random.default_rng(seed)
    inputs = generator.uniform(-1.0, 1.0, size=(160, 1))
    states = np.zeros((len(inputs) + 1, 2))
    states[0] = generator.normal(scale=0.1, size=2)
    for index, command in enumerate(inputs[:, 0]):
        a, b = states[index]
        states[index + 1] = (
            0.90 * a + 0.08 * command + 0.015 * b,
            0.85 * b + 0.06 * np.tanh(a) + 0.04 * command,
        )
    return SequenceSegment(f"recording-{seed}", "whole", states, inputs, dt_s=0.05)


def collection(segments) -> SequenceCollection:
    return SequenceCollection(
        tuple(segments),
        configuration_id=CONFIGURATION_ID,
        state_channels=STATE_CHANNELS,
        input_channels=INPUT_CHANNELS,
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    recordings = [recording(seed) for seed in range(6)]
    calibration = collection(recordings[:3])
    initial_evaluation = collection(recordings[3:4])
    fresh = collection(recordings[4:5])
    evaluation = collection(recordings[5:6])
    for name, data in (
        ("calibration", calibration),
        ("initial-evaluation", initial_evaluation),
        ("new-recordings", fresh),
        ("evaluation", evaluation),
    ):
        save_recordings(data, output / f"{name}.npz")

    model = fit(calibration)
    original_fingerprint = model.fingerprint()
    model.save(output / "model.npz")
    write_json(output / "fit.json", model.report)
    loaded = LearnedDynamics.load(output / "model.npz")
    if loaded.fingerprint() != original_fingerprint:
        raise AssertionError("saved model identity changed on load")

    # An untouched recording supplies actual history and a known command query.
    query = recordings[3]
    p, h = model.history_steps, model.horizon_steps
    past_states, past_inputs = query.states[: p + 1], query.inputs[:p]
    future_inputs = query.inputs[p : p + h]
    forecast = np.asarray(loaded.predict(past_states, past_inputs, future_inputs))
    np.testing.assert_array_equal(
        forecast, model.predict(past_states, past_inputs, future_inputs)
    )
    np.savez_compressed(
        output / "prediction.npz",
        past_states=past_states,
        past_inputs=past_inputs,
        future_inputs=future_inputs,
        targets=query.states[p + 1 : p + h + 1],
        prediction=forecast,
        envelope_half_width=loaded.envelope(h),
    )
    initial_report = evaluate(loaded, initial_evaluation)
    write_json(output / "initial-evaluation.json", initial_report)

    revision = loaded.update(fresh)
    if loaded.fingerprint() != original_fingerprint:
        raise AssertionError("update changed the original model")
    if revision.report["previous_revision"] != original_fingerprint:
        raise AssertionError("updated model lost its predecessor identity")
    revision.save(output / "updated-model.npz")
    write_json(output / "update-fit.json", revision.report)

    # Both models see identical untouched rows; neither learns from this check.
    before = evaluate(loaded, evaluation)
    after = evaluate(revision, evaluation)
    write_json(output / "before-update.json", before)
    write_json(output / "after-update.json", after)
    summary = {
        "purpose": "generic_synthetic_interface_walkthrough",
        "configuration_id": CONFIGURATION_ID,
        "recipe": model.recipe["id"],
        "data_roles": {
            "training": sorted(model.report["training"]),
            "development": sorted(model.report["development"]),
            "initial_evaluation": ["recording-3"],
            "update": ["recording-4"],
            "common_evaluation": ["recording-5"],
        },
        "original_fingerprint": original_fingerprint,
        "updated_fingerprint": revision.fingerprint(),
        "updated_previous_revision": revision.report["previous_revision"],
        "original_unchanged": loaded.fingerprint() == original_fingerprint,
        "prediction_shape": list(forecast.shape),
        "common_evaluation": {
            "metric": "final-step RMSE per observation channel over every complete window",
            "channels": list(STATE_CHANNELS),
            "before": before["aggregate"]["rmse"][-1],
            "after": after["aggregate"]["rmse"][-1],
            "automatic_adoption": False,
        },
        "limits": "Synthetic interface fixture, not new-system or control validation.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/onboarding"))
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(
        json.dumps(
            {
                "recipe": summary["recipe"],
                "original_unchanged": summary["original_unchanged"],
                "common_evaluation": summary["common_evaluation"],
                "summary": str(arguments.output / "summary.json"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
