# Contributing

Read [the charter](docs/charter.md) and [current status](docs/status.md) before
starting an iteration. Glassbox maintains one generic learner and one fixed
recipe. Fitted weights vary by configuration; consumer flags and vehicle-family
branches do not select different dynamics implementations.

## Development

```bash
uv sync --dev
uv run ruff check src tests examples scripts
uv run ruff format --check src tests examples scripts
uv run pytest -q
uv build
```

Tests exercise analytic motion, recording boundaries, gradients, persistence,
immutable updates and forecast scoring. Lifecycle tests use shortened internal
training budgets; they do not establish production-fit accuracy. CI also tests
the installed wheel's CLI, forecast and learner lifecycle outside the checkout.

## Decide with evidence

Work on one named gap. Commit the measurement contract before fitting or
collecting results. Distinguish model accuracy, calibration, mathematical
derivatives, controller outcomes and runtime. Compare changed architectures
under matched fitting conditions; retain deployed revisions as a separate
practical reference.

Preserve frozen outcomes and every attempted condition. Interpret regressions
in the overall decision: generality, accuracy, runtime and maintainability all
matter. An isolated loss is not automatically a veto. Broken contracts, invalid
numerics and substantial consistent capability losses need resolution. Explain
what a comparison actually establishes before adding another experiment.

For an algebraic refactor, first use saved inputs to compare predictions,
derivatives and genuine update snapshots. Measure the whole update; component
compiler timings are not additive CPU shares. Refit or rerun a consumer only
when changed behavior or unresolved evidence warrants it.

Delete superseded implementations instead of retaining compatibility branches.
Preserve current evidence and useful recordings; Git holds historical source.
Controllers, simulators and telemetry conversion belong to their own projects.

## Verify current evidence

The [current public-rate report](docs/public-rate-memory.md) records the frozen
online benchmark, exact source and artifact identities, physical errors and
saved-data verification commands. No fitting is needed to audit scores. The
[held-out arm report](docs/heldout-quad.md) does the same for two new Crazyflow
geometries. The [temporal comparison](docs/nonlinear-temporal.md) is historical
evidence for the preceding model; use its recorded source for original archive
replay.

## Verify historical accumulator evidence

The historical [migration index](docs/accumulator-migration.json) identifies the
saved fitted revisions, online comparison, offline predictions and Dart trials.
Evidence packs are outside Git and the Python distribution. Keep their recorded
paths, or update paths in a local copy of the index without changing authorities.
Do not alter payloads or their manifests.

The saved-data verification authenticates the fit pack and evaluation packs,
recomputes physical errors, and checks saved finite-difference evidence without
fitting or calling a model:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python \
  scripts/qualify_accumulator.py verify --index docs/accumulator-migration.json
```

The subsequent [projection-reuse comparison](docs/history-projection-reuse.md)
checks saved-model predictions, derivatives and whole-update snapshots without
refitting. Its report includes the saved-data verification command and records
strict numerical flags separately from the adoption decision.

For the migration's historical bitwise prediction replay, use source `6e29c4b`
and run without fitting:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python \
  scripts/qualify_accumulator.py replay --index docs/accumulator-migration.json
```

The projection refactor changes floating-point grouping, so current source does
not promise bitwise reproduction of the migration's predictions. Historical exact
replay uses the recorded CPU/arm64 runtime: CPython 3.12.12,
JAX/jaxlib 0.11.1, NumPy 2.5.3 and SciPy 1.18.1, with ambient JAX float32 and
float64 fitting. Ordinary tests cover supported installations; bitwise replay
on other runtimes is not promised.

The online comparison also has an independent saved-array audit:

```bash
PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 uv run python \
  scripts/screen_accumulator.py verify \
  --output artifacts/accumulator-migration-v1/online \
  --manifest-sha256 65be54facc67666bdc2a60d55e414b5b47574eed85523dd09c9efc04fac73024
```

Its paired timings used exclusive alternating blocks; the offline fits and Dart
trials ran alongside other work and do not support a latency comparison.

## Reproduce experiments deliberately

The migration report records scientific commits and protocols. Use those
checkouts to reproduce a historical fit or trial, with a fresh output directory.
The historical six-fit accumulator comparison imports v8 from a separate
historical checkout; the maintained package contains only the temporal learner. A benchmark
protocol is an internal measurement contract, not a product tuning interface.

The Dart runner requires an explicit model-bound protocol. For the historical
accumulator trial, use source `19a0221` with the preserved external controller
snapshot (its archives are incompatible with the current temporal recipe):

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/run_dart.py artifacts/baseline \
  --dart-root artifacts/dart-precision-v1/dart-lateral \
  --output artifacts/accumulator-dart-reproduction \
  --protocol docs/harness/accumulator-dart-qualification-v1.json
```

This runs optimization and native simulation with the saved revision, not a fit.
The fresh accumulator trial missed by 3.67 mm; it did not meet the strict 1 mm
benchmark. The 0.720 mm result belongs to a different, historically refined v8
revision. Do not present it as current accumulator performance.

## Historical evidence

`artifacts/baseline` and [its index](docs/baseline.json) preserve the old v8
models, recordings and exact replay evidence. Those model/session archives are
not compatible with the accumulator. Use source `7b118d6` for the old
`verify_baseline.py` model replay and the source revisions recorded in
[online evidence](docs/online-fitting.md), [cost profiling](docs/online-cost-profile.md)
and [Dart precision](docs/dart-precision.md) for their experiments. Saved-data
verification can still audit historical outcomes without maintaining old dynamics
inside the package.

## Package boundaries

- `learner.py`: fit, predict, update and immutable saved revisions.
- `_dynamics.py`: shared mechanics, learned acceleration and memory.
- `_rate.py`: generic episode-fitted angular response and passive memory.
- `online.py`: causal streaming ingestion, bounded replay and fitting state.
- `recordings.py`: timing, segment boundaries and window extraction.
- `io/recordings.py`: recording archive persistence.
- `workflows/forecast.py`: independent evaluation without learning.
- `cli/`: fit and evaluate only.

Applications own signal meaning, frames, units and clock alignment. Do not infer
absent telemetry semantics or silently substitute actuator states for issued
commands.
