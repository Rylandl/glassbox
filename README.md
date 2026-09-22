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

The accumulator is the single maintained dynamics formulation. It retains shared
rigid-body physics, all 100 ms lag inputs, a nonlinear acceleration head and
learned memory time constants. A parallel history reduction replaces nonlinear
recurrent memory; there is no vehicle-family selector or smaller-model option.

Across six known quad/fixed-wing streams and 3,137 causal updates, it improves
aggregate prediction error **4.11% versus v8**, while reducing quad median/p95
update time **47.63% / 49.76%**. Quad updates still take roughly 40 ms against
10 ms observations, so real-time fitting remains unqualified.

With equal fresh fitting budgets, aggregate offline forecast/command-response
errors improve **8.29% / 9.78%** versus v8. The unchanged Dart task misses by **3.67 mm
with the accumulator and 3.60 mm with v8**. Both retain valid gradients and pass
attitude/speed limits; both miss the strict 1 mm target. The historical **0.720 mm**
result used a separately refined v8 revision and is not current accumulator
performance. Closing that fitting-pipeline gap is distinct from choosing the
memory architecture.

See [the migration evidence](docs/accumulator-migration.md) for the offline
forecast/response comparisons and adoption decision, [status](docs/status.md) for
remaining gaps, and [the network review](docs/network-review.md) for the next
optimization. Generalization, independent error coverage and live closed-loop
identification remain limited.

[Development guide](CONTRIBUTING.md) · [Documentation](docs/README.md) ·
[Apache-2.0 license](LICENSE)
