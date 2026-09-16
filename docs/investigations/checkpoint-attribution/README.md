# Checkpoint attribution and selection coverage evidence

The [report](../../checkpoint-attribution.md) describes 51 optimizer runs and a
subsequent selection-only comparison. All examples are generic synthetic
systems; no library recipe or consumer configuration changed.

## Contents

- [plan.json](plan.json): initial crossed optimization/selection protocol,
  known and fresh seeds, fixed final-checkpoint comparison, and scoring rules.
- [selection-pools-plan.json](selection-pools-plan.json) and
  [coverage-confirmation-path-plan.json](coverage-confirmation-path-plan.json):
  follow-up protocols frozen after the first comparison, before pool scoring
  or the six new confirmation fits. Dataset metadata is inherited; each path
  plan's `stages` field specifies the actual families and seeds to execute.
- [results.zip](results.zip): every path result, full checkpoint trace,
  development choice, coverage variant, and per-recording/horizon/channel score.
  The two objectives' selections are crossed on each archived optimizer path.
- [known-summary.json](known-summary.json), [fresh-summary.json](fresh-summary.json),
  and [coverage-paths-summary.json](coverage-paths-summary.json): all four
  optimizer/selector cells, both orders of paired contrasts, their interaction,
  fixed-final scores, selection steps, and per-recording exclusion diagnostics.
- [coverage-summary.json](coverage-summary.json): all 720 selected-model/regime
  comparisons, counts versus the original development-selected models, and
  direct nested two-to-16-recording comparisons. Pools share optimizer paths
  and evaluation; these are not independent trials.
- [known-audit.json](known-audit.json), [fresh-audit.json](fresh-audit.json),
  [coverage-paths-audit.json](coverage-paths-audit.json), and
  [coverage-audit.json](coverage-audit.json): independent window ranking/time
  indexing, loss and selection reconstruction, NumPy model replay, and all
  reported score checks. Generators and artifact loaders are shared.
- [known-regression.png](known-regression.png): both development criteria and
  the matched/shifted evaluation curves for the exactly reproduced failure.
- [metric_consistency.py](metric_consistency.py) and
  [metric-consistency.json](metric-consistency.json): a clearly labeled post-hoc
  check showing that two reported failures also worsen the same selection loss
  on evaluation data. This does not select a checkpoint or change a model.
- `known-`, `fresh-`, `coverage-paths-`, and `coverage-` prefixed
  `environment.json`, `sources.json`, and `executed-sources.zip`: exact sources
  and versions at each execution. All listed files match their final versions.
  Later auditors and coverage scripts first appear in subsequent snapshots;
  the complete final implementation is included separately.
- [coverage-source-artifacts.json](coverage-source-artifacts.json): hashes of
  frozen input model, evaluation, and result files used by the coverage runner.
  Relative paths resolve from the Glassbox checkout.
- [final-implementation.zip](final-implementation.zip) and
  [implementation-sources.json](implementation-sources.json): all 82 library
  Python files, research scripts, the nine tested modules and test configuration,
  and build/dependency metadata. This pins the dirty checkout's actual sources.
- [source-pytest.xml](source-pytest.xml), [validation.json](validation.json), and
  [starting-hashes.json](starting-hashes.json): 76 passing focused tests,
  unchanged library hashes, and lint/format/document-link checks. The parity
  test checks exact selected-model fingerprints and complete optimizer traces
  against the library fitter on two small fixtures.
- [run-artifacts.json](run-artifacts.json): absolute local paths, sizes, and
  SHA256 hashes for raw runs and final reports. Large checkpoint/evaluation
  arrays remain in those raw directories rather than being duplicated here.
  [manifest.json](manifest.json) hashes every bundle file except itself.

## Replay

Use the existing virtual environment from the Glassbox checkout. All experiment
and report output directories must be new. Work used CPU float64, Python 3.12,
JAX 0.11.1, and NumPy 2.5.3; full versions are recorded. Bitwise optimizer
reproducibility on different runtimes or hardware is not promised.

The known-case parity check requires the prior raw run at
`../artifacts/horizon-generalization/first-step-confirmation-01`. If absent,
regenerate it with the [previous bundle's replay](../horizon-generalization/README.md).
The following path-study commands use an example new parent directory:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_checkpoint_attribution.py --plan docs/investigations/checkpoint-attribution/plan.json --stage known --previous ../artifacts/horizon-generalization/first-step-confirmation-01 --output ../artifacts/checkpoint-replay/known-01
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_checkpoint_attribution.py --plan docs/investigations/checkpoint-attribution/plan.json --stage fresh --previous ../artifacts/horizon-generalization/first-step-confirmation-01 --output ../artifacts/checkpoint-replay/fresh-01
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_checkpoint_attribution.py --plan docs/investigations/checkpoint-attribution/coverage-confirmation-path-plan.json --stage fresh --previous ../artifacts/horizon-generalization/first-step-confirmation-01 --output ../artifacts/checkpoint-replay/coverage-confirmation-paths-01
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/experiment_selection_coverage.py --plan docs/investigations/checkpoint-attribution/selection-pools-plan.json --artifacts ../artifacts/checkpoint-replay --output ../artifacts/checkpoint-replay/selection-coverage-01
```

Audit each of the three path runs with `report_checkpoint_attribution.py --run
RUN --output NEW_REPORT_DIRECTORY`, using the same environment variables. Audit
coverage and reproduce the post-hoc check with:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_selection_coverage.py --run ../artifacts/checkpoint-replay/selection-coverage-01 --artifacts ../artifacts/checkpoint-replay --output ../artifacts/checkpoint-replay/coverage-report
../.venv/bin/python docs/investigations/checkpoint-attribution/metric_consistency.py --artifacts ../artifacts/checkpoint-replay --output ../artifacts/checkpoint-replay/metric-consistency.json
```

The focused test command was:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python -m pytest -q tests/test_checkpoint_attribution.py tests/test_selection_coverage.py tests/test_horizon_generalization.py tests/test_history_confounding.py tests/test_sequence_diagnostics.py tests/test_model_qualification.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py
```

The evidence rejects a simple general fix. Additional development recordings
resolve one inspected case but retain confirmation regressions. Fresh seeds use
selected synthetic equations; this is not real-platform qualification. No full
repository suite or wheel was repeated for research-only changes.
