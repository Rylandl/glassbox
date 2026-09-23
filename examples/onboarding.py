"""Use the public learner on analytic rigid-body motion, without a simulator.

This executes the real fixed fit and update budgets. The generated system is an
API demonstration, not evidence of performance on a physical platform.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from glassbox import (
    STATE_CHANNELS,
    LearnedDynamics,
    SequenceCollection,
    SequenceSegment,
    fit,
)
from glassbox.io.recordings import save_recordings
from glassbox.workflows import evaluate


def recording(name: str, seed: int) -> SequenceSegment:
    rng = np.random.default_rng(seed)
    dt, steps = 0.02, 120
    t = np.arange(steps) * dt
    phases = rng.uniform(-np.pi, np.pi, 6)
    commands = np.sin(t[:, None] * np.linspace(0.7, 1.8, 6) + phases)
    commands[:, :3] *= 0.3
    commands[:, 2] += 9.81
    commands[:, 3:] *= 0.1
    velocity = np.zeros(3)
    omega = np.zeros(3)
    rotation = np.eye(3)
    states = [np.r_[velocity, omega, rotation.ravel()]]
    for command in commands:
        acceleration = rotation @ command[:3] + [0.0, 0.0, -9.81] - 0.25 * velocity
        next_omega = omega + dt * (command[3:] - 0.4 * omega)
        rotation = (
            rotation @ Rotation.from_rotvec(dt * (omega + next_omega) / 2).as_matrix()
        )
        velocity = velocity + dt * acceleration
        omega = next_omega
        states.append(np.r_[velocity, omega, rotation.ravel()])
    return SequenceSegment(name, "whole", np.asarray(states), commands, dt)


def collection(*segments: SequenceSegment) -> SequenceCollection:
    return SequenceCollection(
        segments,
        "analytic-onboarding-v1",
        STATE_CHANNELS,
        tuple(f"command_{index} [issued,unitless]" for index in range(6)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/onboarding"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training = collection(*(recording(f"training-{i}", i) for i in range(3)))
    held_out = collection(recording("held-out", 3))
    fresh = collection(recording("new-recording", 4))
    for name, data in (
        ("training", training),
        ("held-out", held_out),
        ("new-recordings", fresh),
    ):
        save_recordings(data, args.output / f"{name}.npz")

    model = fit(training)
    model.save(args.output / "model.npz")
    model = LearnedDynamics.load(args.output / "model.npz")
    segment = held_out.segments[0]
    p, h = len(segment.inputs) // 2, model.horizon_steps
    prediction = model.predict(
        segment.states[: p + 1], segment.inputs[:p], segment.inputs[p : p + h]
    )
    np.savez_compressed(
        args.output / "prediction.npz",
        prediction=np.asarray(prediction),
        truth=segment.states[p + 1 : p + h + 1],
    )
    before = model.fingerprint()
    revision = model.update(fresh)
    assert model.fingerprint() == before
    revision.save(args.output / "updated-model.npz")
    report = {
        "initial": evaluate(model, held_out),
        "updated": evaluate(revision, held_out),
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Wrote recordings, model revisions, prediction and evaluation to {args.output}"
    )


if __name__ == "__main__":
    main()
