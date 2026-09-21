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

## Reproduce streaming fitting

The active [v6 physical-curvature comparison](docs/harness/online-fit-v6.json) reuses the
exact sealed v1 collection and two known Cascade recordings. It intentionally
refuses changed inputs. The old Throw controller collected issued commands and
observations; it is not the candidate or a matched-input accuracy comparator.
See [the v1 result](docs/online-fit-v1.json) for its failed result and identities.

Commit source and choose a new output directory. With the preserved v1 pack:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/evaluate_online.py run artifacts/online-fit-v1/collection \
  --collection-sha256 276ba8d5c49250bf0cc8c242884f50e42c2566b660619958c53ca285651486c4 \
  --protocol docs/harness/online-fit-v6.json \
  --reference artifacts/online-fit-v4/evaluation \
  --output artifacts/online-fit-v6-reproduction
```

Verify errors and the causal journal without fitting, using the evaluation
manifest hash printed by the run. The copied, authenticated v4 reference pack (including v2)
is verified too; no old optimizer implementation is needed:

```bash
PYTHONPATH=src:scripts python scripts/evaluate_online.py verify \
  artifacts/online-fit-v6-reproduction \
  --manifest-sha256 EVALUATION_MANIFEST_SHA256
```

The rejected v5 solver-consistency experiment is reproducible from source
`5db9059`, using its frozen `online-fit-v5.json` protocol and the v4 evaluation
as `--reference`. Current verification still audits its sealed pack and nested
v4/v2 references with zero fits; current runs use the maintained v6 fitter.
See [the failed primary result](docs/online-fit-v5.json).

The rejected [v7 domain experiment](docs/online-fit-v7.md) is reproducible from
source `3ca9498`, with protocol `online-fit-v7.json` and the adopted v6 evaluation
as `--reference`. Current saved-array verification preserves its v7 domain
arithmetic and 23 causal captures without loading or maintaining a v7 learner.
The separate diagnostic driver in its checks directory invokes the unchanged
snapshot diagnostics from that committed source after authenticating the v7
pack; its sealed result authority is in [the index](docs/online-fit-v7.json).

The preceding v4 result is reproducible from source `8053938`; its sealed pack
is the v6 primary comparator. V6 scientific source is `2e465a4`.

The v2 optimizer result is reproducible from source `a43d2dc`; its sealed pack
supplies the v4 primary comparator. The failed coordinate-only v3 run is
reproducible from source `492a521` and retains its original verdict. The original v1 procedure and native
collection are reproducible from source
commit `ecf488e`, using the original Throw virtualenv for `collect_throw.py` and
the current-runtime dependencies for fitting. Do not run the new optimizer under
the old candidate protocol. A fresh collection requires its own frozen input
contract. These replays establish neither candidate-controlled recovery nor
blind generalization across platforms.

## Reproduce the causal fixed-wing diagnosis

The current [v2 trace](docs/harness/online-causal-trace-v2.json) uses adopted v6
and recovers 23 snapshots across all 450 fixed-wing updates. Source `7dee1b4`
reproduces it with the commands below, replacing the input with
`artifacts/online-fit-v6/evaluation` and explicitly passing
`--protocol docs/harness/online-causal-trace-v2.json` to `run`. Its read-only
verification also checks angular head increments and physical Jacobian blocks.
See [the result](docs/online-angular-response.md).

The [frozen causal trace](docs/harness/online-causal-trace-v1.json) diagnoses the
working v4 fitter on its authenticated saved fixed-wing inputs. It replays all
450 updates to recover the exact contemporaneous model state at 20 declared
origins; this is a diagnostic replay, not a new candidate or blind evaluation.
Use source `5648d13` for this historical v4 replay and choose an output
directory that does not exist:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/trace_online.py run artifacts/online-fit-v4/evaluation \
  --output artifacts/online-causal-trace-reproduction
```

Use the same pinned runtime as the original online evaluation. Verification
loads the captured sessions, checks their causal caches and original predictions,
and recomputes every integration diagnostic **without optimizer updates**:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/trace_online.py verify artifacts/online-causal-trace-reproduction \
  --manifest-sha256 TRACE_MANIFEST_SHA256
```

The scientific source is `5648d13`. The learner source inventory must match the
authenticated v4 binding exactly; future changes require the recorded source
checkout for replay. See [the findings](docs/online-causal-trace.md).

## Reproduce physical-head support diagnosis

From frozen source `724fc02` and the authenticated causal-trace pack:

```bash
env -u JAX_ENABLE_X64 PYTHONPATH=src:scripts SCIPY_ARRAY_API=1 python \
  scripts/diagnose_online_support.py \
  --reference artifacts/online-causal-trace-v1/diagnosis \
  --output artifacts/online-response-support-reproduction
```

This performs no fits: it transforms saved coefficients and features, computes
support/nullspace diagnostics, and crosses two observed histories/current commands
under identical weights. Counterfactual pairs have no truth score. See
[the bounded interpretation and authority](docs/online-response-support.md).

## Package boundaries

- `learner.py` owns fit, predict, update and saved model revisions.
- `_dynamics.py` implements the shared motion formulation and fitting arithmetic.
- `online.py` owns causal streaming ingestion, bounded replay and persistent fitting state.
- `recordings.py` owns timing, segment boundaries and extraction.
- `io/recordings.py` stores and loads recording archives.
- `workflows/forecast.py` scores independent recordings without learning.
- `cli/` exposes only fit and evaluate.

Examples and documentation use the public model API. Keep telemetry meaning and
coordinate conversion at the application boundary; never silently invent absent
units, frames, timing or command semantics. Controllers and simulators belong to
their own projects.
