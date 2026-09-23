# Glassbox

Glassbox learns differentiable rigid-body dynamics from observed motion and issued
commands. One procedure fits each configuration without a vehicle type, mass,
inertia, mixer, actuator layout, prior trained model or tuning menu. Shared gravity,
frame transforms and rotation kinematics surround fitted force, torque, inertia
and hidden command response. The model is intended for use by controllers,
planners, estimators and analysis code in other projects.

## Install

From this repository with Python 3.12–3.13:

```bash
uv sync --dev
```

The package depends on JAX, NumPy and SciPy. Supply uniformly sampled recordings
with the [signal and timing contract](docs/learner.md).

## Use

```python
from glassbox import LearnedDynamics, fit
from glassbox.io.recordings import load_recordings

model = fit(load_recordings("training.npz"))
recording = load_recordings("held-out.npz").segments[0]
origin = len(recording.inputs) // 2
prediction = model.predict(
    recording.states[: origin + 1],
    recording.inputs[:origin],
    recording.inputs[origin : origin + model.horizon_steps],
)
model.save("model.npz")
revision = LearnedDynamics.load("model.npz").update(
    load_recordings("new-recordings.npz")
)
revision.save("updated-model.npz")
```

Prediction needs every issued command since the segment began to reconstruct
hidden actuator state. `update` returns a new immutable revision and leaves the
original unchanged. Evaluate improvements on independent recordings.

For a live contiguous episode, `OnlineFit(prefix)` collects completed transitions
and fits in the background, periodically publishing immutable revisions. The
published model can lag the newest observation; query `published_cursor` to see
which data it used. Initialization and publication time are part of fresh-start
performance. See [streaming identification](docs/learner.md#streaming-identification).

The [onboarding example](examples/onboarding.py) constructs analytic rigid-body
recordings, runs the actual fit/update procedure, persists revisions and evaluates
on a separate recording:

```bash
uv run python examples/onboarding.py --output artifacts/onboarding
uv run glassbox fit artifacts/onboarding/training.npz --model model.npz
uv run glassbox evaluate model.npz artifacts/onboarding/held-out.npz
```

The example is an API walkthrough, not physical validation. The
[current state and measured limits](docs/status.md) distinguish fully refitted
prefix accuracy from the models actually available during background fitting.

[Development guide](CONTRIBUTING.md) · [Documentation](docs/README.md) ·
[Apache-2.0 license](LICENSE)
