# Contributing

Read [the charter](docs/charter.md) and [current status](docs/status.md) before
starting an iteration. Glassbox maintains one learner and one fixed recipe.
Fitted weights vary by configuration; consumer flags and vehicle-family branches
do not select different dynamics implementations.

## Development

```bash
uv sync --dev
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest -q
uv build
```

Tests exercise analytic motion, recording boundaries, gradients, persistence,
immutable updates and forecast scoring. Lifecycle tests use explicitly shortened
internal training budgets; they do not claim accuracy for a production fit. CI
also installs the wheel into a separate environment and runs the CLI, evaluation
and learner tests outside the checkout.

## Evaluation and changes

Work on one named gap. Freeze and commit the evaluation protocol before fitting
or collecting new results. Use held-out recordings and report physical errors,
large-error cases and all attempted conditions. Model accuracy, envelope coverage,
computable derivatives and controller task success are separate claims.

For a behavior-preserving change, replay saved inputs against their saved outputs
before replacing the implementation. Verify that an altered artifact is rejected.
Record numerical results and remaining limits in the current status and the
iteration's report. Historical verdicts are not rewritten after a policy decision
or a later success.

Keep active code small: delete superseded implementations, interfaces, tests and
one-off experiment tooling. Preserve the evidence needed to support current claims
and the recordings needed for the next iteration before deleting their obsolete
containers. Git retains committed history.

## Package boundaries

- `learner.py` owns fit, predict, update and saved model revisions.
- `_dynamics.py` implements the shared motion formulation and fitting arithmetic.
- `recordings.py` owns timing, segment boundaries and extraction.
- `io/recordings.py` stores and loads recording archives.
- `workflows/forecast.py` scores independent recordings without learning.
- `cli/` exposes only fit and evaluate.

Examples and documentation use the public model API. Keep telemetry meaning and
coordinate conversion at the application boundary; never silently invent absent
units, frames, timing or command semantics. Controllers and simulators belong to
their own projects.
