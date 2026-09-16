# Recording selection and observation-budget evidence

The [research report](../../recording-selection.md) describes the generic
interfaces, selection tradeoffs, and fixed-row coverage comparison. The plan was
written on 2026-09-14 before computing this iteration's results. All source
recordings had already been inspected in earlier work.

## Contents

- [plan.json](plan.json): fixed questions, policies, datasets, budgets, seeds,
  and limitations.
- [selection-summary.json](selection-summary.json): evaluation and
  leave-one-development-recording-out results, with equal-recording aggregation,
  chosen models, choice-change counts, and physical errors relative to hold.
- [coverage-summary.json](coverage-summary.json): complete per-family metrics
  and sampled coverage for three new 337-row training draws.
- [case-reports.zip](case-reports.zip): all 108 decisions, all candidate recording
  scores, original prediction-file hashes, the 126 newly fitted candidates'
  scores, and exact crop/window source identities. Selection scores and guards
  use standardized errors; the reporting metrics use physical vector RMSE.
- [audit.json](audit.json): 108 independently replayed choices, 126 model replays,
  324 direct-head checks, 501 reconstructed windows, and numerical tolerances.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json): the
  exact experiment source at execution, including imported experimental modules.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, selected tests, and configuration. The final collection/evidence
  dataclasses add constructor validation and defensive copies after fitting;
  numerical behavior of the recorded inputs is unchanged. The audit script was
  added after the fits. These source snapshots are intentionally distinct.
- [environment.json](environment.json), [validation.json](validation.json),
  [source-pytest.xml](source-pytest.xml), and [wheel-pytest.xml](wheel-pytest.xml):
  recorded runtime and focused source/wheel checks.
- [recording-selection.png](recording-selection.png) /
  [recording-selection.svg](recording-selection.svg): per-recording body-rate
  comparison, with labels in [figure-recordings.json](figure-recordings.json).
- [starting-hashes.json](starting-hashes.json): preexisting workspace file hashes.
- [manifest.json](manifest.json): byte counts and hashes of bundle members,
  excluding the manifest itself.

## Replay

From the Glassbox repository, the full local tree is
`../artifacts/recording-selection`. It holds `comparison-01` with all new fitted
models, samples, predictions, and selection evidence; `report-02` is the final
independent audit. The compact bundle avoids duplicating these arrays or raw
recordings.

The experiment additionally uses the preserved
`../artifacts/representation-study` tree and
`../artifacts/forecast-diagnosis/comparison-01`. Their corpus identities, clocks,
preprocessing, and earlier audits are described in the
[representation evidence](../forecast-representation/README.md) and
[forecast-diagnosis evidence](../forecast-diagnosis/README.md). The coverage
experiment reuses the same pinned Crazyflie log10 and keeps the existing
log15/log16 windows unchanged. No external data was added in this iteration.

To replay the audit against the retained local artifacts:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_recording_selection.py --output ../artifacts/recording-selection/report-replay
```

To rerun the experiment into a new directory using the same prior artifacts:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python scripts/experiment_recording_selection.py --plan docs/investigations/recording-selection/plan.json --output ../artifacts/recording-selection/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_recording_selection.py --run ../artifacts/recording-selection/comparison-replay --output ../artifacts/recording-selection/report-replay-2
```

Both scripts reject existing output directories. `--previous` selects a
different representation-study tree; the experiment's `--original` selects the
prior forecast-diagnosis tree. Selection provenance records absolute source
paths, so relocating saved runs also requires making those sources available or
regenerating the comparison against the relocated trees. Regeneration does not
turn these recordings into new untouched evaluation evidence.

Coverage budget counts refer to rows after time-grid sampling. Equal state-row
counts do not imply equal complete-window counts, transition counts, or elapsed
time span. Repeated seeds and overlapping windows are not independent flights.
The selector supplies empirical comparisons, not calibrated uncertainty or
validated control behavior.
