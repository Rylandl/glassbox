# Contributing

Read the [charter](docs/charter.md) and [current status](docs/status.md) before
changing the learner. Glassbox maintains one generic rigid-body formulation.
Every fitted quantity comes from the current configuration's observations; no
vehicle-family branch, pretrained dynamics, actuator metadata requirement or
consumer tuning option belongs in the model.

## Development

```bash
uv sync --dev
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest -q
uv build
```

Tests cover analytic motion, recording boundaries, derivatives, persistence,
immutable updates and scoring. Frozen physical evidence supplements these tests;
one good forecast is not a controller or reliability result.

## Iteration discipline

Work on one named gap in an isolated worktree. Freeze and commit the measurement
runner before a full fit. State exactly which observations and published revision
each prediction uses, including initialization and background fitting time.
Compare full-state velocity, body rate, orientation and command response in their
physical units. Keep fully refitted-prefix accuracy separate from the revisions
that would actually have been available in a live episode. Commit completed
scientific code and verify saved forecasts without refitting. Explain regressions
in the overall engineering decision rather than using every cell as a veto.

Delete superseded implementations and unused tools. Git preserves historical
sources and evidence. Applications own controllers, simulation, telemetry decoding,
coordinate conversion and clock alignment.

## Current package boundaries

- `src/glassbox/_causal_actuator.py`: differentiable equation and immutable fitted parameters.
- `src/glassbox/_causal_fit.py`: one episode fitting procedure and background worker.
- `src/glassbox/learner.py`: public fit, predict, update and saved revisions.
- `src/glassbox/online.py`: causal observation collection and intermittent publication.
- `src/glassbox/recordings.py` and `src/glassbox/io/`: recording semantics and archives.
- `src/glassbox/core/`: generic geometry and motion adapter.
- `src/glassbox/workflows/forecast.py` and `src/glassbox/cli/`: held-out scoring and CLI.

The frozen matched-prefix and publication runners are
`scripts/qualify_causal_public.py` and
`scripts/qualify_causal_publication.py`. Their source recordings and incumbent
forecasts are identified in `docs/online-readout-benchmark.json`; the current
result and limitations are in [status](docs/status.md). Historical experiments
require their recorded Git revisions, not compatibility code in the current
package.
