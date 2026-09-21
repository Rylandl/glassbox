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

The supported shared-physics learner is the single maintained implementation. Its
saved Dart revision now reaches **0.720 mm nominal contact miss** through a
radius-aware lateral objective and 10 ms feedback, with **0.93° axis error** and
all **9,874 objective/gradient evaluations finite**. The learned model is unchanged.
Fine-grid float64 replay confirms **0.756 mm** with submicrometer convergence;
the original float32 convergence failure remains documented. This is one known
seeded simulated task, not broad control reliability or real-time qualification.
See [the full precision result](docs/dart-precision.md), including every attempt.

The streaming fitter reduces aggregate one-step velocity/rate error **41% versus
the previous working online version** across four quad and two fixed-wing tapes
(73% versus frozen startup fits). All twelve primary case/metric comparisons
improve with one shared procedure. Fixed-wing forecast spikes, one quad orientation
regression and quad update latency remain open; live controller adoption is
unqualified. See [the online result](docs/online-fitting.md).

Forecast and response errors across Crazyflow and Cascade remain improvement
work, including wind cases and longer horizons. Error-envelope calibration and
physical derivative accuracy remain limited. See [status](docs/status.md) for the
measured gaps and [the learner contract](docs/learner.md) for model scope.

[Development guide](CONTRIBUTING.md) · [Documentation](docs/README.md) ·
[Apache-2.0 license](LICENSE)
