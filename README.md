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

For a contiguous live stream, `OnlineFit(prefix)` retains causal fitting state
and assimilates one completed transition at a time, with bounded replay and
fitting work. It uses the same dynamics formulation. See the
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

The single maintained model combines shared rigid-body physics, a learned
three-axis force readout, an episode-fitted angular-rate response, compact
nonlinear history and eight stable accumulators. Each configuration gets its
own fit; there is no vehicle-family selector or smaller-model option.
Four-command / 10 ms models use **4,340 parameters**, down from 5,450 in the
preceding model.

On the frozen 4,187-update public online benchmark, 250 ms body-rate error was
0.345/0.597 rad/s on two fixed-wing recordings and 0.841/0.551/6.237 rad/s on
three quad arm conditions. The hard arm condition remains inaccurate. Warm CPU
updates took about 0.92 ms fixed-wing and 1.7 ms quad, but cold compilation
and first updates remain much slower. The first underexcited command-response
probe still has 1.071 relative error.

No new live Throw or Dart controller trial has been run for this revision. Two
newly collected Crazyflow arm configurations show that the 1.40 arm predicts
well, while the 0.85 arm
has a severe 250 ms high-spin rate error. The earlier **3.669 mm**
full-history accumulator result and **0.720 mm** refined-v8 result belong to
different revisions. See the [current result](docs/public-rate-memory.md) for
known-recording evidence, the [held-out arm result](docs/heldout-quad.md) for
new-configuration evidence, and [status](docs/status.md) for the next gap.
Earlier temporal, migration and projection results remain historical evidence.

[Development guide](CONTRIBUTING.md) · [Documentation](docs/README.md) ·
[Apache-2.0 license](LICENSE)
