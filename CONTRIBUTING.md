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

## Replay the adopted baseline

The local `artifacts/baseline` directory contains the three adopted models,
retained recordings and saved evaluation inputs/outputs. It is deliberately
outside Git and the Python distribution. For another checkout, copy this whole
directory from the preserved evidence pack; its manifest must match the hash in
[docs/baseline.json](docs/baseline.json).

```bash
env -u JAX_ENABLE_X64 SCIPY_ARRAY_API=1 uv run python \
  scripts/verify_baseline.py artifacts/baseline --dart-root /path/to/dart
```

Exact replay requires the recorded runtime: CPython 3.12.12, JAX 0.11.1,
NumPy 2.5.3 and SciPy 1.18.1, on CPU/arm64 with JAX default32. The verifier
checks runtime and payload hashes before predictions. A different supported
installation can run the ordinary tests; bitwise baseline equivalence across
other backends or versions has not been established.

The optional Dart path supplies the unchanged external objective and contact
scorer; their source hashes are checked. A complete run reproduces all 12,768
saved flight arrays, 40 selected Dart trajectories and 4,144 objective/gradient
evaluations without fitting, optimizing or simulating. Omitting `--dart-root`
still checks the models, flight arrays, selected trajectories and independent
contact reconstruction, but reports `complete: false` and zero gradient replays.
The result is a preservation check on existing evidence, not a fresh flight trial.

## Reproduce Dart precision

The preserved `artifacts/dart-precision-v1/dart-original` snapshot supplies the
old objective for the baseline verifier after Dart's live objective changes.
The winning consumer is `artifacts/dart-precision-v1/dart-lateral`. The evidence
pack and its identities are described in [the precision result](docs/dart-precision.md).
Use the same pinned runtime as the baseline, with Dart/Crazyflow installed.
The scripts import Glassbox from this checkout and require a clean Git commit.

A fresh nominal run uses the one retained frozen controller protocol:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/run_dart.py artifacts/baseline \
  --dart-root artifacts/dart-precision-v1/dart-lateral \
  --output artifacts/dart-precision-reproduction
```

The output directory must not exist. This performs controller optimization and
native simulation with the existing learned model; it does not fit a model.
Compare the issued commands, native states and numerical scores with
`native-lateral`, rather than runtime-dependent journal/manifest bytes.
The original scientific run used commit `ddbefa793f4b0be0e19f6e996130612a749298b8`.
Earlier protocols and the completed attribution tool remain in their recorded
Git commits, not as maintained alternatives.

The arithmetic-precision audit independently replays the **preserved winning
command tape**, without model or controller calls. Its supplemental protocol
pins that exact trial and does not automatically certify a new run:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/audit_dart_resolution.py artifacts/baseline \
  artifacts/dart-precision-v1/native-lateral \
  --dart-root artifacts/dart-precision-v1/dart-lateral \
  --protocol docs/harness/dart-lateral-precision-v1.json \
  --trial-manifest-sha256 e44c9794c859cbefa1290a4dbdfac45176be8004ffc5b429ff01ffcda61eaa91 \
  --output artifacts/dart-resolution-reproduction
```

The original float32 convergence failure remains in
`artifacts/dart-precision-v1/lateral-resolution-audit`. A fresh trial, target or
control change requires a new prospective measurement contract.

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
