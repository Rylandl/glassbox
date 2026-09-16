# Accuracy requirement evidence

The [report](../../accuracy-requirements.md) distinguishes task tolerances from
model prediction errors and records empirical forecast allowances. This work
ran on 2026-09-14 on `experiment/generic-transition-support`. It reuses the
frozen [default-recipe predictions](../default-recipe/README.md) without fitting,
selecting, or changing a model. No vehicle-control trial was added.

## Contents

- [plan.json](plan.json): pre-replay definitions, descriptive 95% fraction,
  illustrative physical limits, references, and limitations.
- [summary.json](summary.json): seven cases, 27 arms, final-horizon allowances,
  simultaneous empirical fractions and measured qualifying horizons.
- [reports.zip](reports.zip): complete `reports.json`, including per-recording
  endpoint RMS, endpoint and prefix empirical percentiles, maxima, joint counts,
  model identities, channel contracts, and original evaluation origins.
- [source-artifacts.json](source-artifacts.json): hashes and original absolute
  paths of the 21 input reports, predictions, and provenance files. These were
  also checked against the earlier default-recipe artifact manifest.
- [feedback-example.json](feedback-example.json): four synthetic scalar error
  recurrences with equal disturbance RMS, including complete time series and
  analytic asymptotic amplitudes. These are stipulated feedback dynamics, not
  new learned vehicle models or fits to the recorded forecasts.
- [independent-trial-requirements.json](independent-trial-requirements.json):
  exact zero-failure independent-trial counts for three success probabilities.
  This calculation is not applied to the correlated recording windows.
- [audit.json](audit.json): independent vector-error, empirical-rank, prefix,
  joint-fraction, identity and horizon checks; convolution replay of the scalar
  recurrence; and the trial-count calculation.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json):
  the measurement helper, replay script and tests when the final replay ran.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, focused tests, and test configuration. The independent audit script
  was added after replay. No preexisting fitted model code changed.
- [forecast-allowance.png](forecast-allowance.png) /
  [forecast-allowance.svg](forecast-allowance.svg): per-horizon marginal physical
  allowances for the latest saved revisions. Each panel is a separate metric.
- [environment.json](environment.json), [validation.json](validation.json),
  [source-pytest.xml](source-pytest.xml), [wheel-pytest.xml](wheel-pytest.xml):
  environment, focused checks, and isolated-package validation.
- [starting-hashes.json](starting-hashes.json): preexisting workspace hashes.
- [sources-reviewed.json](sources-reviewed.json): primary research/statistics
  references supporting the task-dependent framing and trial calculation.
- [manifest.json](manifest.json): hashes and sizes of bundle files, excluding
  the manifest itself.

## Replay

From the Glassbox repository, input predictions remain in
`../artifacts/default-recipe/comparison-01`. The final measurement replay is
`../artifacts/accuracy-requirements/comparison-02`. It adds rejection of complex
values and byte-string recording IDs relative to the first replay; the measured
real-valued results are identical. Every input was inspected in earlier work.

```sh
MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/quantify_forecast_accuracy.py --plan docs/investigations/accuracy-requirements/plan.json --output ../artifacts/accuracy-requirements/comparison-replay
../.venv/bin/python scripts/audit_forecast_accuracy.py --run ../artifacts/accuracy-requirements/comparison-replay --output ../artifacts/accuracy-requirements/audit-replay.json
```

The replay rejects existing output directories. `--run` chooses an alternative
default-recipe comparison tree; the auditor's `--forecasts` must point to the
same tree. Absolute paths in source provenance must remain available for hash
checking, or the replay must be regenerated against relocated sources. The
compact bundle does not duplicate the original raw arrays or dependencies.

Observed fractions are proportions of existing overlapping windows. Recording
names do not establish independent trials, identical distributions, or support
outside the observed regimes. Reported 95% allowances are descriptive order
statistics, not conformal bounds, confidence intervals, or universal task limits.
The scalar example illustrates why temporal structure and feedback matter; its
residual bound cannot be filled in with a multi-step forecast RMSE.
