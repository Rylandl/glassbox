# History confounding and objective evidence

The [report](../../history-confounding.md) records two platform-neutral synthetic
experiments on 2026-09-14: 12 diagnostic cases comparing feature capacity and
history, followed by nine matched loss-normalization fits. The cycle contributes
three new baseline models; the delay/noise baselines are reused. All 82 library
Python files remain unchanged, and no consumer options were introduced.

## Contents

- [plan.json](plan.json): diagnostic protocol frozen before fitting the cycle.
- [diagnostic-results.zip](diagnostic-results.zip): all case reports, every
  basis/recording/channel comparison, and the three new baseline fit reports.
- [summary.json](summary.json), [audit.json](audit.json): ranges, 5,280 scored
  windows, and independent reconstruction of 432 auxiliary fits. Hold and
  learned cycle predictors share their input windows.
- [history-confounding.png](history-confounding.png): short nonlinear features
  versus older observations on the fitted models.
- [objective_probe.py](objective_probe.py), [objective-audit.json](objective-audit.json):
  post-hoc decomposition of the original cycle loss, before candidate fitting.
  The helper was made relocatable afterward; its arithmetic was rerun unchanged.
- [objective-ablation-plan.json](objective-ablation-plan.json): subsequent
  matched-comparison protocol frozen before the nine candidate fits.
- [objective-results.zip](objective-results.zip): all nine fit reports/results,
  every horizon/channel error, both model fingerprints, and paired evaluation
  counts. Candidates are research `SequenceModel` artifacts, not promoted
  `LearnedDynamics` revisions under the existing recipe identifier.
- [objective-comparison-summary.json](objective-comparison-summary.json) and
  [objective-comparison-audit.json](objective-comparison-audit.json): all-horizon
  accuracy changes and independent replay/indexing of 4,272 evaluation windows.
- [horizon-scaling.png](horizon-scaling.png): paired forecasts under original
  versus pooled horizon normalization, including the regressions.
- [environment.json](environment.json), [ablation-environment.json](ablation-environment.json):
  CPU/Python/JAX/NumPy versions; numerical work used float64.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json)
  and [ablation-executed-sources.zip](ablation-executed-sources.zip) /
  [ablation-sources.json](ablation-sources.json): executed source snapshots for
  each stage. The original diagnostic snapshot predates two added pooled-scale
  tests. Numerical library and experiment sources did not change after fitting.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, relevant tests, and build configuration, including the auditor.
- [final-source-pytest.xml](final-source-pytest.xml), [validation.json](validation.json):
  58 passing focused tests, lint/format checks, unchanged library hashes, and
  audit results. No new wheel build or full repository suite was run because
  this work changed research scripts/tests/docs only.
- [starting-hashes.json](starting-hashes.json): pre-experiment library and
  documentation hashes. [run-artifacts.json](run-artifacts.json) records local
  model/array paths, sizes, and SHA256 hashes; large arrays are not duplicated
  in this compact bundle. [manifest.json](manifest.json) hashes the bundle.

## Replay

From the Glassbox checkout using the existing virtual environment:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python scripts/experiment_history_confounding.py --plan docs/investigations/history-confounding/plan.json --prior ../artifacts/sequence-diagnostics/comparison-01 --output ../artifacts/history-confounding/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_horizon_scaling.py --plan docs/investigations/history-confounding/objective-ablation-plan.json --previous ../artifacts/history-confounding/comparison-replay --output ../artifacts/history-confounding/objective-comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_history_confounding.py --run ../artifacts/history-confounding/comparison-replay --ablation ../artifacts/history-confounding/objective-comparison-replay --output ../artifacts/history-confounding/report-replay
```

The executed trees are `comparison-01`, `objective-comparison-01`, and final
`report-02` under `../artifacts/history-confounding`. Each main experiment/auditor
requires a new output directory. Prior delay/noise model paths are recorded in
the previous investigation and remain necessary for exact replay. Calibration
and evaluation recordings regenerate from fixed seeds and archived generators.

The standalone read-only objective decomposition can be rerun on any reproduced
cycle baseline tree from the same checkout:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python docs/investigations/history-confounding/objective_probe.py --run ../artifacts/history-confounding/comparison-01 --output ../artifacts/history-confounding/objective-audit-replay.json
JAX_ENABLE_X64=1 ../.venv/bin/python -m pytest -q tests/test_history_confounding.py tests/test_sequence_diagnostics.py tests/test_model_qualification.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py
```

The normalized selection losses differ between objectives and must not be
compared as a common accuracy metric. The report uses paired physical-coordinate
errors instead. The objective intervention changes optimization and checkpoint
selection together. Audits verify arithmetic and evidence separation; they
share data generators and do not independently rerun optimization. Repeated
cycle observations and three data seeds are not independent uncertainty bounds.
