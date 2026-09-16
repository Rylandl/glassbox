# Glassbox

Glassbox fits differentiable quadrotor and fixed-wing dynamics from telemetry.
It provides rollout prediction, evaluation against declared baselines, and
incremental parameter updates, with PX4 ingestion and adapters for public
flight datasets.

The platform-level goal is to reduce the calibration and engineering effort
needed to bring a new airframe or hardware revision into a shared control and
autonomy workflow. Glassbox delivers the model and its evidence; downstream
applications supply behaviors. This is an onboarding objective, with the
current evidence recorded in the validation results below.

A `DynamicsBelief` combines the executable model, local parameter information,
and measured forecast error. Start with the workflow below; [scope](docs/scope.md)
describes the design and [validation](docs/validation.md) records the results.
The [platform onboarding proposal](docs/platform-onboarding.md) describes the
next interface improvements and how to evaluate them.

The intended experience is an opinionated fit, predict, and update workflow.
Start with `fit(telemetry)` and its defaults. Model architecture, feature
representations, regularization, and model-selection policies belong to the
maintained learning recipe. The research investigations explore those choices;
they are not a configuration guide for onboarding a platform. The
[design contract](docs/scope.md#opinionated-onboarding) distinguishes this target
from the current structured models and experimental generic learners.

## Install

```bash
uv sync --dev
```

Python 3.11 to 3.13. JAX runs on the CPU backend in float32 by default, and
the core package depends only on JAX and NumPy. Optional extras add telemetry
support: `px4` for PX4 ULog ingestion and SITL recording, and `ros` for the EPFL
rosbag adapter. `uv sync --dev` installs both for testing. To install the Cascade
simulator from its Git source, use `uv sync --dev --group cascade`.

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

The same fit in Python returns a belief directly. The defaults include
accumulating parameter information from the fitted data:

```python
from pathlib import Path

from glassbox import fit

outcome = fit(sorted(Path("flights").glob("*.npz")))
belief = outcome.belief
belief.save("artifacts/belief.json")
```

`fit` also accepts `Trajectory` objects directly, or a mixture of trajectories
and paths. Arrays use the same coordinator, holdout rules, and evidence
calculation as files. The [onboarding walkthrough](docs/guides/platform-onboarding.md)
runs fit, independent evaluation, prediction, and an update for both families
without writing calibration trajectories to disk.

Evaluate on a separate set of flights reserved for this comparison:

```bash
uv run glassbox evaluate artifacts/belief.json unseen/*.npz \
  --protocol windowed --report artifacts/evaluation.json
```

Inspect prediction errors against the baseline at the horizons you need. The
fit report records the data used to calibrate forecast error.

Update with fresh, nonoverlapping telemetry:

```python
from glassbox.core.data import load_trajectory_npz

telemetry = load_trajectory_npz("flights/recent.npz")
belief, update = belief.absorb(telemetry)
print(update.absorbed, update.window_count, update.information_gain_nats)
```

See [dynamics beliefs](docs/concepts/dynamics-beliefs.md) for the evidence and
update API, and [NMPC](docs/concepts/nmpc.md) for using a belief in control.
The experimental [streaming replay](docs/guides/platform-onboarding.md#streaming-refinement)
scores each fresh telemetry block before learning, keeps the active model pinned,
and lets the application adopt a previously evaluated revision explicitly.
The [live refinement experiment](docs/streaming-refinement.md) runs that learner
beside simulated tracking with bounded telemetry queues and prepared controller
handoffs, comparing frozen and adopting models for both families.
The [Cascade follow-up](docs/cascade-refinement.md) tests the same workflow on
independently implemented Skywalker X8 dynamics.

## The `glassbox` command

The command-line workflows share one console command. `glassbox --help` lists the
whole tree, and every leaf prints its own flags with `--help`.

```text
glassbox extract      PX4 ULogs to canonical trajectory NPZ files          [px4]
glassbox corpus       the pinned reference corpora
    list | fetch | prepare
glassbox synthetic    synthetic trajectories for either vehicle family
glassbox fit          fit a dynamics belief and report from NPZ flights
glassbox evaluate     score models on held-out flight under one protocol
glassbox benchmark    the maintained closed-loop and corpus benchmarks
    nmpc | recovery | cascade-x8                        (cascade-x8 needs Cascade)
glassbox record-results  regenerate the recorded artifacts under docs/results/
glassbox sitl-profile    fly one bounded PX4 SITL maneuver profile         [px4]
glassbox px4-shadow      passive NMPC shadow on live PX4; never sends      [px4]
```

A bracketed name is the optional extra a command needs, for example
`uv run --extra px4 glassbox px4-shadow ...`.

## Code organization

- `core`: telemetry, dynamics, fitting numerics and prediction metrics
- `belief` and `fitting.py`: fitted artifacts, parameter evidence and updates
- `io`, `workflows` and `cli`: ingestion, evaluation and command-line entry points
- `control` and `integrations`: planning and vehicle links

The package root exports the main API in
[`glassbox.__init__`](src/glassbox/__init__.py). Other functions are imported
from their owning module.

## Tests

```bash
uv run pytest
```

The default suite takes several minutes; `-m "not slow"` skips the
benchmark-scale tests, `-m "not cascade"` the ones needing the simulator, and the
PX4 SITL contract tests are opt-in behind `GLASSBOX_RUN_PX4_SITL=1`. Lint with
`uv run ruff check src tests`. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the
full check list and the recorded-artifact procedure.

## Documentation

The [documentation index](docs/README.md) links the concepts, telemetry guide,
recorded results and investigations. Control and bootstrap identification are
experimental; PX4 integration currently runs in passive shadow mode.

## License and citation

Glassbox is released under the [Apache License, Version 2.0](LICENSE). The
reference corpora it can download are distributed by their authors under their
own terms and are not covered by this license.

If you use Glassbox in academic work, please cite it using
[CITATION.cff](CITATION.cff).
