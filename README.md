# Glassbox

Glassbox learns differentiable vehicle dynamics from recorded state and
actuator telemetry, keeps an explicit account of what the fitted model does not
know, and uses that belief for bounded online adaptation and model-predictive
control. It targets multirotors and fixed-wing aircraft flown by PX4, with
adapters for several published system-identification datasets and two
simulators.

The fitted artifact is a **dynamics belief**: one executable rigid-body model
with learned actuator lag and an optional compact residual, the accumulated
information saying which of its coefficients the evidence has resolved and how
precisely, and the forecast-error envelope saying how wrong its predictions
have been on flights it did not see. Evaluation, live updates, and the NMPC
controller all consume that one artifact.

## Install

```bash
uv sync --dev
```

Python 3.11 to 3.13. JAX runs on the CPU backend in float32 by default.
The core package depends only on JAX and NumPy. There are three optional
extras: `px4` for PX4 ULog ingestion and SITL recording (`pyulog`,
`pymavlink`), `ros` for the EPFL rosbag adapter (`rosbags`), and `cascade` for
the Cascade fixed-wing plant, which installs from GitHub. `uv sync --dev` installs the telemetry extras because the default
test suite exercises them.

## Quickstart

Turn telemetry into canonical trajectories. From a PX4 ULog:

```bash
uv run glassbox extract flight.ulg flight.npz --rate 50
```

The [PX4 ULog guide](docs/guides/px4-ulog.md) covers fixed-wing logs,
ground-truth versus estimated states, and gap handling. The pinned reference
corpora (`glassbox corpus list`, then `glassbox corpus prepare NAME DIR`)
produce the same NPZ format.

Fit a belief on several flights. The final source group, or the final flight
when the flights are not grouped, is held out for validation; add
`--ablation no-lag` to fit the near-zero-lag comparison beside the model:

```bash
uv run glassbox fit flights/*.npz \
  --model artifacts/belief.json --report artifacts/report.json
```

The same fit in Python, where the belief is the return value rather than a
file:

```python
from pathlib import Path

from glassbox import FitSpec, fit

outcome = fit(sorted(Path("flights").glob("*.npz")), FitSpec(steps=400))
belief = outcome.belief
```

Use the belief for control:

```python
import jax.numpy as jnp

from glassbox import DynamicsBelief, NMPCController, SafetyEnvelope, TrackingTolerances
from glassbox.core.dynamics import hover_control

belief = DynamicsBelief.load("artifacts/belief.json")
controller = NMPCController(
    belief,
    TrackingTolerances.for_platform(belief.input_spec.vehicle.family),
    SafetyEnvelope(
        minimum_position_m=(-100.0, -100.0, -20.0),
        maximum_position_m=(100.0, 100.0, 100.0),
    ),
)

# NWU position and velocity, WXYZ quaternion, FLU body rates.
state = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = controller.solve(
    state, controller.hold_reference(state), hover_control(belief.params)
)
command = result.command  # bounded even when result.command_usable is False
```

Update the belief from recent telemetry with `belief.absorb(trajectory)`,
which returns a new belief and an `UpdateResult`. Every usable one-step
transition adds `J' R^-1 J` to the accumulated precision and the step is that
precision's pseudo-inverse applied to the whitened innovation, so a
well-resolved coefficient moves less than a poorly resolved one and a
direction the telemetry does not excite does not move at all. There is no
proposal, no validation split and no acceptance threshold; information
accumulates and is never discounted. See
[dynamics beliefs](docs/concepts/dynamics-beliefs.md) and
[NMPC](docs/concepts/nmpc.md) for the full contracts.

## Layout

The package is a set of subpackages under `src/glassbox`. `import glassbox`
loads only `core`, `belief`, `control`, and the one public `fitting` module;
workflows, command-line front ends, corpus adapters, and integrations are
imported on demand.

| Subpackage | Modules |
| --- | --- |
| root | `fitting` (`fit`, `FitSpec`, `Holdout`, the fit report) |
| `core` | `data`, `dynamics`, `families`, `geometry`, `identification`, `metrics`, `diagnostics`, `model`, `model_io`, `synthetic`, `fixedwing_synthetic` |
| `belief` | `belief`, `information`, `forecast_error`, `update`, `parameter_evidence`, `belief_io`, `linearization` |
| `control` | `plan`, `solver`, `fitted`, `identifier`, `supervisor` |
| `io` | `corpus`, `px4_ulog`, `px4_frames`, `pinned_download`, `sitl_profile`, `arp_reference`, `idf_reference`, `nanodrone_reference`, `x8_reference`, `epfl_reference` |
| `workflows` | `evaluate`, `holdout`, `record_results`, `benchmarks/` (`nmpc`, `recovery`, `cascade_x8`) |
| `cli` | one module per command, plus the static `_tree` |
| `integrations` | `loop`, `px4`, `px4_nmpc_shadow`, `cascade` |

### The stable surface

`glassbox.__all__` is these names. A name is here because the README or a
concept page uses it, or because it is the type of one of their arguments or
return values; everything else is imported from the module that owns it, for
example `from glassbox.core.data import load_trajectory_npz`.

| Group | Names |
| --- | --- |
| Telemetry | `Channel`, `Trajectory`, `TrajectorySpec` |
| The fit | `fit`, `FitSpec`, `FitOutcome`, `Holdout`, `LossPolicy`, `WeightingPolicy` |
| Parameters and rollout | `ModelParams`, `DynamicsParams`, `FixedWingDynamicsParams`, `rollout`, `step` |
| The belief | `DynamicsBelief`, `ParameterInformation`, `ForecastErrorEnvelope`, `UpdateResult`, `ExecutableModel`, `ActuationMap`, `NonActionableModelError` |
| Control | `PlanModel`, `plan_model`, `BoundedShootingSolver`, `SolverPolicy`, `SolveResult`, `SolveStatus`, `NMPCController`, `ReferenceTrajectory`, `SafetyEnvelope`, `TrackingTolerances`, `Prediction` |
| In-flight identification and supervision | `RecursiveBootstrapIdentifier`, `RecursiveBootstrapConfig`, `MultirotorFlightSupervisor`, `MultirotorSupervisorConfig`, `SupervisorMode`, `SupervisorReason` |

The canonical state is 13 wide: NWU position and velocity, a WXYZ unit
quaternion from body to world, and FLU body rates. Commands are normalized
actuator inputs in the order the trajectory spec declares.

## The `glassbox` command

Every workflow lives behind one console command. `glassbox --help` lists the
whole tree, and every leaf prints its own flags with `--help`.

```text
glassbox extract      PX4 ULogs to canonical trajectory NPZ files          [px4]
glassbox corpus       the pinned reference corpora
    list | fetch | prepare
glassbox synthetic    synthetic trajectories for either vehicle family
glassbox fit          fit a dynamics belief and report from NPZ flights
glassbox evaluate     score models on held-out flight under one protocol
glassbox benchmark    the maintained closed-loop and corpus benchmarks
    nmpc | recovery | cascade-x8                        (cascade-x8: [cascade])
glassbox record-results  regenerate the recorded artifacts under docs/results/
glassbox sitl-profile    fly one bounded PX4 SITL maneuver profile         [px4]
glassbox px4-shadow      passive NMPC shadow on live PX4; never sends      [px4]
```

A bracketed name is the optional extra a command needs, for example
`uv run --extra px4 glassbox px4-shadow ...`.

## Tests

```bash
uv run pytest
```

The default suite takes several minutes; `-m "not slow"` skips the three
benchmark-scale tests. Tests marked `cascade` need that extra; deselect them
with `-m "not cascade"` when the extra is not installed. The PX4 SITL contract
tests are opt-in:

```bash
GLASSBOX_RUN_PX4_SITL=1 uv run pytest -m px4_sitl tests/integration/test_px4_sitl.py -v
```

Lint with `uv run ruff check src tests`.

## Documentation

The [documentation index](docs/README.md) lists every page. The main entries:

- [Scope and current boundary](docs/scope.md)
- Concepts: [dynamics beliefs](docs/concepts/dynamics-beliefs.md),
  [NMPC](docs/concepts/nmpc.md),
  [flight supervisor](docs/concepts/flight-supervisor.md),
  [bootstrap identification](docs/concepts/bootstrap-identification.md)
- Experiments: one page per corpus, simulator diagnostic, and gate under
  [`docs/experiments/`](docs/experiments/), each citing its recorded artifact
  in [`docs/results/`](docs/results/)
- [Literature review](docs/literature-review.md)

## Status

Both vehicle families have differentiable rollout, fitting, serialization, and
PX4 ULog ingestion. Fixed-wing short and medium rollouts transfer across
maneuver families, recording sessions, and two airframe configurations;
multirotor results are competitive with a published structured-residual
reference on one airframe and not yet consistently better than kinematic
persistence on another. Long rollouts on multi-minute sessions remain
unstable, parameters do not transfer between airframes, and every controller
result is a bounded simulation diagnostic rather than a flight-safety claim.
See [scope](docs/scope.md) for the full boundary.

[glassbox-throw](https://github.com/Rylandl/glassbox-throw) is a demo built on
this package: a simulated quadrotor thrown with no prior model, learned in
flight and recovered.

## License and citation

Glassbox is released under the [Apache License, Version 2.0](LICENSE). The
reference flight corpora it can download (Nano-Quadrotor, ARP, IDF-DS,
Skywalker X8, EPFL TOPOPlane2) are distributed by their authors under their own
terms and are not covered by this license.

If you use Glassbox in academic work, please cite it. GitHub renders a
ready-made citation from [CITATION.cff](CITATION.cff); the BibTeX form is:

```bibtex
@software{lillibridge2026glassbox,
  author  = {Lillibridge, Ryland},
  title   = {Glassbox: telemetry-driven differentiable vehicle dynamics identification},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Rylandl/glassbox},
  license = {Apache-2.0}
}
```
