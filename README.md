# Glassbox

Glassbox learns differentiable dynamics from recordings of observed signals
and commands. One maintained recipe provides `fit`, `predict` and `update`;
callers supply signal identities, units, timing, recording boundaries and a
configuration ID.

The learned model is the product. Use its forecasts, derivatives and saved
revisions with your own controller, planner, estimator or analysis code.
Glassbox's experimental controller is an optional reference consumer.

The generic learner is the adopted development baseline. The same recipe is
fitted separately to each system. Current evidence covers five flight corpora:
it beats the structured comparators on four and loses on ARP. Two isolated
research learners now meet a declared simulator tracking task; the adopted
public model has not yet been qualified with that improved controller.
Error-envelope coverage and reliable updates remain open problems.
[Status](docs/status.md) records the measurements and the proposed JSBSim model
benchmark. Performance across arbitrary systems remains unproved.

## Install

```bash
uv sync --dev
```

Python 3.11–3.13 is supported. Core numerical dependencies are JAX, NumPy and
SciPy. Optional `px4` and `ros` extras support telemetry ingestion; `uv sync
--dev` installs their test dependencies. Install the optional Cascade simulator
with `uv sync --dev --group cascade`.

## Quickstart

Run a complete example with synthetic observations, saved forecasts, held-out
evaluation and an immutable update:

```bash
uv run python examples/platform_onboarding.py --output artifacts/onboarding
```

The example writes generic recording archives that also work with the CLI:

```bash
uv run glassbox fit artifacts/onboarding/calibration.npz \
  --model artifacts/onboarding/cli-model.npz \
  --report artifacts/onboarding/cli-fit.json

uv run glassbox evaluate artifacts/onboarding/cli-model.npz \
  artifacts/onboarding/evaluation.npz \
  --report artifacts/onboarding/cli-evaluation.json
```

In Python, `fit` returns the model directly:

```python
from glassbox import LearnedDynamics, fit
from glassbox.io.recordings import load_recordings

model = fit(load_recordings("artifacts/onboarding/calibration.npz"))
recording = load_recordings("artifacts/onboarding/evaluation.npz").segments[0]
p, h = model.history_steps, model.horizon_steps

future = model.predict(
    recording.states[: p + 1],
    recording.inputs[:p],
    recording.inputs[p : p + h],
)
half_width = model.envelope(h)
model.save("artifacts/onboarding/model.npz")

revision = LearnedDynamics.load("artifacts/onboarding/model.npz").update(
    load_recordings("artifacts/onboarding/new-recordings.npz")
)
revision.save("artifacts/onboarding/updated-model.npz")
```

A forecast is conditional on the supplied future commands. `update` returns a
new revision and leaves the original unchanged. Evaluate both on the same
untouched recordings before claiming an improvement.

For your own arrays, construct a `SequenceCollection` of `SequenceSegment`
objects. Each segment holds `N + 1` observations and `N` commands at one
uniform sample interval; command `k` acts between observations `k` and
`k + 1`. Fit needs at least two distinct recording identities. See the
[onboarding guide](docs/guides/platform-onboarding.md) for the complete example
and [learner contract](docs/learner.md) for shapes, history, splits and evidence.

## Recording files and telemetry

`glassbox.io.recordings` saves and loads generic recording NPZ files, including
their signal contract and segment boundaries. PX4 extraction and corpus
preparation produce canonical flight-trajectory NPZ files. Convert those
explicitly with `from_trajectories` before generic fitting; the formats serve
different input contracts. The [PX4 guide](docs/guides/px4-ulog.md) shows the
conversion. The adapter preserves command names, units, frames, semantics and
roles in the channel identities; converted data must match the loaded model's
contract.

```bash
uv run glassbox extract flight.ulg artifacts/flight.npz --rate 50
uv run glassbox corpus list
```

## The `glassbox` command

`glassbox --help` lists the complete tree. The primary fitting and evaluation
commands have no model-family, optimizer, split-policy or horizon options.

| Command | Purpose |
| --- | --- |
| `fit RECORDINGS.npz... --model MODEL.npz [--report REPORT.json]` | Fit the generic recipe from generic recording archives. |
| `evaluate MODEL.npz RECORDINGS.npz... [--report REPORT.json]` | Measure held-out per-channel forecast errors and envelope coverage. |
| `extract`, `corpus` | Ingest telemetry and prepare canonical flight datasets. |
| `synthetic` | Generate canonical flight fixtures for retained benchmarks. |
| `benchmark`, `record-results` | Run the retained research comparisons. |
| `sitl-profile`, `px4-shadow` | PX4 integration tools; see their own contracts and optional dependencies. |

Evaluation compares the model with hold-current on every complete forecast
window. It reports errors in each channel's units and envelope coverage, with
no automatic task-success decision.

## Code organization

- `learner.py`: the generic `fit` and `LearnedDynamics` implementation.
- `recordings.py`: observation segments and recording collections.
- `io.recordings` and `workflows.forecast`: generic files, conversion and evaluation.
- `control` and `integrations`: experimental planning and transport consumers.
- `core`, `belief`, `fitting` and `workflows`: retained structured benchmarks,
  telemetry contracts and unmigrated consumers.

The package root exports `fit`, `LearnedDynamics`, `SequenceCollection`,
`SequenceSegment` and `segments_from_mask`. Structured implementations are
imported explicitly by their remaining consumers.

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

Use `-m "not slow"` to skip benchmark-scale tests and `-m "not cascade"` to
exclude simulator contracts. PX4 SITL tests require
`GLASSBOX_RUN_PX4_SITL=1`. [Contributing](CONTRIBUTING.md) describes the full
checks and recorded-artifact procedure.

## Documentation

Start with the [documentation index](docs/README.md), [scope](docs/scope.md),
[charter](docs/charter.md) and [current status](docs/status.md).

## License and citation

Glassbox is released under the [Apache License, Version 2.0](LICENSE).
Reference corpora retain their authors' licenses. Cite the project using
[CITATION.cff](CITATION.cff).
