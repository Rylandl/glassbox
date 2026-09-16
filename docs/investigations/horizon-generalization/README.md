# Generic horizon-normalization evidence

The [report](../../horizon-generalization.md) records three synthetic comparison
stages on 2026-09-14. Pooling worsens first-step error in 19 of 24 cases in each
evaluation regime. A subsequent first-step weight cap preserves 20 of 24 models
exactly in fresh-seed confirmation, but one changed model regresses. Neither
candidate changes the library recipe or adds consumer configuration.

## Contents

- [pooled-plan.json](pooled-plan.json),
  [first-step-development-plan.json](first-step-development-plan.json), and
  [first-step-confirmation-plan.json](first-step-confirmation-plan.json): frozen
  protocols, exact equations, data seeds, fitting settings, and screen constants.
  The two floor plans were chosen after pooled results but frozen together before
  any floor fits. Their inherited `purpose` text mentions pooling; the
  `candidate`, `candidate_key`, `phase`, and `provenance` fields specify the
  actual follow-up experiment.
- [results.zip](results.zip): every aggregate result, per-case result, and
  baseline/candidate fit report for all three stages. All channels, horizons,
  recordings, and regressions are retained, together with model fingerprints
  and training/development window identities.
- [pooled-summary.json](pooled-summary.json),
  [development-summary.json](development-summary.json), and
  [confirmation-summary.json](confirmation-summary.json): counts and
  per-family absolute/relative error changes. Three-seed ranges are not
  confidence intervals; a median can hide a regressing seed.
- [pooled-audit.json](pooled-audit.json),
  [development-audit.json](development-audit.json), and
  [confirmation-audit.json](confirmation-audit.json): independent indexing,
  model replay, loss reconstruction, and score checks. Each checks 15,360
  calibration/development windows and 5,952 evaluation windows; the development
  stage reuses the earlier baseline data. Audits share the generator, artifact
  loaders, and an earlier independent NumPy recurrence, without refitting.
- [pooled.png](pooled.png), [development.png](development.png), and
  [confirmation.png](confirmation.png): median error changes, supplemented by
  the report's explicit confirmation failure and all-seed results.
- `pooled-`, `development-`, and `confirmation-` prefixed `environment.json`,
  `sources.json`, and `executed-sources.zip`: exact fitting-stage source
  snapshots and runtime versions. The pooled snapshot predates the added
  first-step-floor test; its numerical sources are unchanged. The two later
  snapshots match every listed source. The final auditor is captured below.
- [final-implementation.zip](final-implementation.zip) and
  [implementation-sources.json](implementation-sources.json): all 82 library
  Python files, scripts, the seven tested modules and test configuration, plus
  build/dependency metadata. No library source changed during this study.
- [source-pytest.xml](source-pytest.xml), [validation.json](validation.json),
  and [starting-hashes.json](starting-hashes.json): 72 passing focused tests,
  lint/format/link checks, source comparisons, and numerical audit results.
  No wheel or full repository test run was repeated for research-only changes.
- [run-artifacts.json](run-artifacts.json): paths, sizes, and SHA256 hashes for
  local raw runs and reports, including model and evaluation NPZ files. Those
  large arrays are not duplicated in this compact bundle; replay regenerates
  them. [manifest.json](manifest.json) hashes every bundle file except itself.

## Replay

From the Glassbox checkout with its existing virtual environment, use new output
directories; each command refuses to overwrite an existing output. Numerical
work used CPU float64, Python 3.12, JAX 0.11.1, and NumPy 2.5.3; full versions
are in the environment files. The snapshots pin the dirty checkout's actual
sources independently of the branch name. The final implementation snapshot
includes both runners and the auditor needed below. Bitwise optimizer
reproducibility across runtime or hardware changes is not promised.

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_horizon_generalization.py --plan docs/investigations/horizon-generalization/pooled-plan.json --output ../artifacts/horizon-generalization/pooled-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_first_step_floor.py --plan docs/investigations/horizon-generalization/first-step-development-plan.json --previous ../artifacts/horizon-generalization/pooled-replay --output ../artifacts/horizon-generalization/development-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_first_step_floor.py --plan docs/investigations/horizon-generalization/first-step-confirmation-plan.json --output ../artifacts/horizon-generalization/confirmation-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_horizon_generalization.py --run ../artifacts/horizon-generalization/pooled-replay --output ../artifacts/horizon-generalization/pooled-replay-report
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_horizon_generalization.py --run ../artifacts/horizon-generalization/development-replay --output ../artifacts/horizon-generalization/development-replay-report
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_horizon_generalization.py --run ../artifacts/horizon-generalization/confirmation-replay --output ../artifacts/horizon-generalization/confirmation-replay-report
```

The focused test command was:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python -m pytest -q tests/test_horizon_generalization.py tests/test_history_confounding.py tests/test_sequence_diagnostics.py tests/test_model_qualification.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py
```

These are 72 paired case comparisons across stages, 144 regime comparisons,
and 79 optimizer runs. Development reuses 24 existing baselines; 41 candidate
models reuse exact baseline parameters under identical objectives. Confirmation
uses new seeds on the same eight equation families, not new physical systems.
This bundle supports a limited research decision about loss normalization.
