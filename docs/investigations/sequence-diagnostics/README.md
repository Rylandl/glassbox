# Sequence diagnostic evidence

The [report](../../sequence-diagnostics.md) describes the new experimental
`model.diagnose(recordings)` API and 24 platform-neutral synthetic cases. The
diagnostics are read-only; the dynamics fitting and prediction recipe did not
change. There are 15 reused models and nine fresh fits, all reported.

## Contents

- [plan.json](plan.json): the protocol frozen before fitting, with the fixed
  learner/diagnostic recipes, cases, seeds, and observation budgets.
- [summary.json](summary.json): per-channel ranges over three seeds.
- [diagnostic-results.zip](diagnostic-results.zip): all 24 reports and results,
  plus the nine fresh models' fit reports. Each diagnostic report includes
  per-recording measurements and interpretation limits.
- [audit.json](audit.json): independent indexing of 14,976 windows, NumPy model
  replay, 480 augmented least-squares fits, and 6,216 numerical comparisons.
- [sequence-diagnostics.png](sequence-diagnostics.png): positive controls and
  the deterministic cubic policy blind spot.
- [environment.json](environment.json): numerical run environment, using
  float64 on CPU. [validation.json](validation.json) records source and wheel
  checks, the unchanged learner recipe, and source differences.
- [source-pytest.xml](source-pytest.xml) and [wheel-pytest.xml](wheel-pytest.xml):
  50 focused tests passing in each environment. The wheel suite ran outside
  the checkout with JAX float32. The full repository suite was not run.
- [wheel-verification.json](wheel-verification.json): wheel hash, verified
  import paths, and byte agreement for all 82 packaged Python files.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json):
  the library, experiment, generator, and diagnostic tests at execution time.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, focused tests, and build configuration, including the independent
  auditor. The executed test snapshot predates a NumPy `row_stack` → `vstack`
  compatibility fix and added dual-solve/gap test coverage. The numerical
  library and experiment are identical between the executed and final sources.
- [starting-hashes.json](starting-hashes.json): hashes of 81 preexisting library
  files and the documentation index. The only change to preexisting library
  code is adding `LearnedDynamics.diagnose`; the diagnostic module is new.
- [run-artifacts.json](run-artifacts.json): paths, sizes, and hashes of the raw
  model/evidence arrays and package artifacts, including the 15 reused models.
  These remain in the local artifact tree instead of being duplicated here.
- [manifest.json](manifest.json): bundle hashes and sizes, excluding itself.

## Replay

From the Glassbox checkout with the existing environment:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python scripts/experiment_sequence_diagnostics.py --plan docs/investigations/sequence-diagnostics/plan.json --previous ../artifacts/model-qualification/comparison-01 --output ../artifacts/sequence-diagnostics/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_sequence_diagnostics.py --run ../artifacts/sequence-diagnostics/comparison-replay --output ../artifacts/sequence-diagnostics/report-replay
```

The recorded trees are `../artifacts/sequence-diagnostics/comparison-01` and
`../artifacts/sequence-diagnostics/report-01`. Both scripts require a new output
directory. The experiment needs the earlier saved models; the auditor uses
the model paths and hashes recorded in each result. Raw evidence arrays include
window provenance, model predictions, diagnostic features, fold membership,
donor identities, regression normalizers/coefficients, and held-out predictions.
Synthetic source recordings regenerate from the archived generators and seeds.

To test the final source implementation:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python -m pytest -q tests/test_sequence_diagnostics.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py tests/test_model_qualification.py
../.venv/bin/ruff check src/glassbox/experimental/default_model.py src/glassbox/experimental/sequence_diagnostics.py scripts/experiment_sequence_diagnostics.py scripts/report_sequence_diagnostics.py tests/test_sequence_diagnostics.py
```

The arithmetic audit shares data generators and the model artifact loader,
but reconstructs window indexing, regression fits, and scores independently.
It does not establish independence of inputs, uniquely identify hidden memory,
calibrate uncertainty, or validate extrapolation. The prior synthetic recordings
were already inspected; all results are diagnostic development evidence. Three
data seeds share one learner initialization seed and are not uncertainty bounds.
