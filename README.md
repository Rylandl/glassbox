# Glassbox

Glassbox learns differentiable rigid-body dynamics from recordings of motion and
issued commands. One fixed learning procedure fits each platform or configuration;
the user supplies no vehicle type, mass, inertia, mixer, actuator layout or tuning
choices. Shared gravity and rigid-body kinematics combine with learned effective
accelerations and command memory.

The learned model is the product. Use its predictions and derivatives in your own
controller, planner, estimator or analysis code.

## Install

From this repository, with Python 3.12–3.13:

```bash
uv sync --dev
```

The package depends on JAX, NumPy and SciPy. There are no simulator or telemetry
extras. Supply uniformly sampled recordings with the [documented signal and timing
contract](docs/learner.md).

## Use

```python
from glassbox import LearnedDynamics, fit
from glassbox.io.recordings import load_recordings

model = fit(load_recordings("training.npz"))
recording = load_recordings("held-out.npz").segments[0]
p, h = model.history_steps, model.horizon_steps
prediction = model.predict(
    recording.states[: p + 1],
    recording.inputs[:p],
    recording.inputs[p : p + h],
)
model.save("model.npz")
revision = LearnedDynamics.load("model.npz").update(
    load_recordings("new-recordings.npz")
)
revision.save("updated-model.npz")
```

`predict` returns future states conditioned on the supplied commands. `update`
returns a new revision and leaves the original unchanged; improvement must be
measured on separate recordings.

For a contiguous live stream, `OnlineFit(prefix)` retains optimizer state and
assimilates one completed transition at a time, with bounded replay and fitting
work. It uses the same dynamics formulation. See the
[streaming contract](docs/learner.md#streaming-identification) and current measured
limits in [status](docs/status.md).

The [onboarding example](examples/onboarding.py) shows recording construction,
fit, prediction, persistence, held-out evaluation and update using analytic
rigid-body motion:

```bash
uv run python examples/onboarding.py --output artifacts/onboarding
uv run glassbox fit artifacts/onboarding/training.npz --model model.npz
uv run glassbox evaluate model.npz artifacts/onboarding/held-out.npz
```

The example runs the real fixed fitting procedure twice; it is an API walkthrough,
not a quick test or physical validation.

## Current evidence

The single maintained model combines shared rigid-body physics, full linear lag
response, compact nonlinear history and eight learned stable accumulators.
Each configuration gets its own fit; there is no vehicle-family selector or
smaller-model option. Four-command / 10 ms models now use **5,450 parameters**,
down from 8,714.

The matched architectural comparison preserves aggregate online accuracy and
improves Crazyflow command-response error **4.31%** and Dart 250 ms forecast
error **14.99%**. Crazyflow forecast error is **4.20% higher**, and quad whole
updates are **6.10% slower** on the measured CPU (**31.9 → 33.9 ms**). Fixed-wing
results are unchanged in the tested two-lag configuration. Compactness did not
produce a CPU speedup; real-time fitting and other hardware remain unqualified.

No new Dart controller trial was run for this architecture. The earlier
**3.669 mm** full-history accumulator result and **0.720 mm** refined-v8 result
belong to different revisions. Held-out forecast improvements do not establish
submillimeter control. Long-horizon fidelity, independent error coverage and live
closed-loop identification remain open.

See [the temporal comparison](docs/nonlinear-temporal.md) for matched fits,
physical errors and archive provenance, [status](docs/status.md) for remaining
gaps, and [the network review](docs/network-review.md) for the next priority.
Earlier migration and projection results remain historical evidence; current
full-stream evaluation corrects their combined accuracy interpretation.

[Development guide](CONTRIBUTING.md) · [Documentation](docs/README.md) ·
[Apache-2.0 license](LICENSE)
