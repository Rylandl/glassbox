# Glassbox

Glassbox turns vehicle telemetry into one canonical flight object, fits a
differentiable rigid-body model whose free parameters are the map from
actuator behaviour to forces and torques, keeps that model current from live
telemetry by absorbing information rather than by passing a gate, and turns it
into a bounded command every control interval. It does this for a vehicle with
a fitted model and for a vehicle with none, through one solver and one belief
type. It targets multirotors and fixed-wing aircraft flown by PX4, with
adapters for five published system-identification corpora and two simulators.

The product object is a `DynamicsBelief`: one executable model, the
accumulated `ParameterInformation` saying which coefficients the evidence
resolved and how precisely, and the `ForecastErrorEnvelope` saying how wrong
predictions of a given length have been on flights the fit did not see.
Evaluation, live updates and the controller all consume that one artifact.

## Install

```bash
uv sync --dev
```

Python 3.11 to 3.13. JAX runs on the CPU backend in float32 by default, and
the core package depends only on JAX and NumPy. Three optional extras add
telemetry and simulator support: `px4` for PX4 ULog ingestion and SITL
recording, `ros` for the EPFL rosbag adapter, and `cascade` for the Cascade
fixed-wing plant. `uv sync --dev` installs the telemetry extras because the
default test suite exercises them.

## Quickstart

Turn telemetry into canonical trajectories. From a PX4 ULog:

```bash
uv run glassbox extract flight.ulg flight.npz --rate 50
```

The [PX4 ULog guide](docs/guides/px4-ulog.md) covers fixed-wing logs,
gap handling and the fit flags. The pinned reference corpora
(`glassbox corpus list`, then `glassbox corpus prepare NAME DIR`) produce the
same NPZ format, and `glassbox synthetic DIR` writes a closed-world corpus for
either family.

Fit a belief on several flights. The final source group, or the final flight
when the flights are not grouped, is held out for validation:

```bash
uv run glassbox fit flights/*.npz \
  --model artifacts/belief.json --report artifacts/report.json
```

The same fit in Python, where the belief is the return value rather than a
file. `parameter_evidence` additionally accumulates the fit's own one-step
information, so the belief starts with a resolved rank instead of at zero:

```python
from pathlib import Path

from glassbox import FitSpec, fit

outcome = fit(
    sorted(Path("flights").glob("*.npz")),
    FitSpec(parameter_evidence=True),
)
belief = outcome.belief
belief.save("artifacts/belief.json")
```

Update the belief from recent telemetry. Every usable one-step transition adds
`J' R^-1 J` to the accumulated precision and the step is that precision's
pseudo-inverse applied to the whitened innovation, so a direction the
telemetry does not excite does not move at all. There is no proposal, no
validation split and no acceptance threshold:

```python
from glassbox.core.data import load_trajectory_npz

telemetry = load_trajectory_npz("flights/recent.npz")
belief, update = belief.absorb(telemetry)
print(update.absorbed, update.window_count, update.information_gain_nats)
```

Use the belief for control:

```python
import jax.numpy as jnp

from glassbox import DynamicsBelief, NMPCController
from glassbox.core.dynamics import hover_control

belief = DynamicsBelief.load("artifacts/belief.json")
controller = NMPCController(belief)

# NWU position and velocity, WXYZ quaternion, FLU body rates.
state = jnp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
result = controller.solve(
    state, controller.hold_reference(state), hover_control(belief.params)
)
command = result.command  # bounded even when result.command_usable is False
```

See [dynamics beliefs](docs/concepts/dynamics-beliefs.md) and
[NMPC](docs/concepts/nmpc.md) for the full contracts.

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

## Layout

The package is a set of subpackages under `src/glassbox`. `import glassbox`
loads only `core`, `belief`, `control` and the one public `fitting` module;
workflows, command-line front ends, corpus adapters and integrations are
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

`glassbox.__all__` is these 41 names. A name is here because the README or a
concept page uses it, or because it is the type of one of their arguments or
return values; everything else is imported from the module that owns it, for
example `from glassbox.core.data import load_trajectory_npz`.

| Group | Names |
| --- | --- |
| Telemetry | `Channel`, `Trajectory`, `TrajectorySpec` |
| The fit | `fit`, `FitSpec`, `FitOutcome`, `Holdout`, `LossPolicy`, `WeightingPolicy` |
| Parameters and rollout | `ModelParams`, `DynamicsParams`, `FixedWingDynamicsParams`, `BootstrapMultirotorParams`, `rollout`, `step` |
| The belief | `DynamicsBelief`, `ParameterInformation`, `ForecastErrorEnvelope`, `UpdateResult`, `ExecutableModel`, `ActuationMap`, `NonActionableModelError` |
| Control | `PlanModel`, `plan_model`, `BoundedShootingSolver`, `SolverPolicy`, `SolveResult`, `SolveStatus`, `NMPCController`, `ReferenceTrajectory`, `SafetyEnvelope`, `TrackingTolerances`, `Prediction`, `PlanValues` |
| In-flight identification and supervision | `RecursiveBootstrapIdentifier`, `RecursiveBootstrapConfig`, `BootstrapEvidence`, `MultirotorFlightSupervisor`, `MultirotorSupervisorConfig`, `SupervisorMode`, `SupervisorReason` |

The canonical state is 13 wide: NWU position and velocity, a WXYZ unit
quaternion from body to world, and FLU body rates. Commands are normalized
actuator inputs in the order the trajectory spec declares.

## Tests

```bash
uv run pytest
```

The default suite takes several minutes; `-m "not slow"` skips the
benchmark-scale tests, `-m "not cascade"` the ones needing that extra, and the
PX4 SITL contract tests are opt-in behind `GLASSBOX_RUN_PX4_SITL=1`. Lint with
`uv run ruff check src tests`. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the
full check list and the recorded-artifact procedure.

## Documentation

Eight pages, listed in the [documentation index](docs/README.md):

- [Scope and current boundary](docs/scope.md)
- [Validation](docs/validation.md): every recorded number, with the artifact
  it comes from
- Concepts: [dynamics beliefs](docs/concepts/dynamics-beliefs.md),
  [NMPC and the flight supervisor](docs/concepts/nmpc.md),
  [bootstrap identification](docs/concepts/bootstrap-identification.md)
- [PX4 ULogs](docs/guides/px4-ulog.md): ingestion and SITL recording
- [Literature review](docs/literature-review.md): the negative-result record

## Status

Both vehicle families have differentiable rollout, fitting, serialization and
PX4 ULog ingestion. Fixed-wing short and medium rollouts transfer across
maneuver families, sessions and two airframe configurations; multirotor
results sit behind a published structured-residual reference on one airframe
and behind kinematic persistence on the other. Long rollouts on multi-minute
sessions remain unstable, parameters do not transfer between airframes, and
every controller result is a bounded simulation diagnostic rather than a
flight-safety claim. See [scope](docs/scope.md) for the boundary and
[validation](docs/validation.md) for the evidence.

[glassbox-throw](https://github.com/Rylandl/glassbox-throw) is a demo built on
this package: a simulated quadrotor thrown with no prior model, learned in
flight and recovered. It is pinned to glassbox `d10bb24`, the revision before
this refactor, and stays there until it is resynced once. Resyncing means
planning through `plan_model` over the bootstrap belief instead of a
reimplemented solver, and driving the identifier through its current API; at
that point most of the demo's own 3,258-line controller is glassbox.

## License and citation

Glassbox is released under the [Apache License, Version 2.0](LICENSE). The
reference corpora it can download are distributed by their authors under their
own terms and are not covered by this license.

If you use Glassbox in academic work, please cite it. GitHub renders a
citation from [CITATION.cff](CITATION.cff); the BibTeX form is:

```bibtex
@software{lillibridge2026glassbox,
  author  = {Lillibridge, Ryland},
  title   = {Glassbox: telemetry-driven differentiable vehicle dynamics identification},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/Rylandl/glassbox},
  license = {Apache-2.0}
}
```
