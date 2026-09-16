# Forecast diagnosis evidence

See [the investigation](../../forecast-diagnosis.md) for methods and limits.

- `plan.json`, `environment.json`, `sources.json`, `executed-sources.zip`:
  the frozen full-run recipe and code identities.
- `case-reports.zip`: per-model metrics, candidate selections and exact
  per-recording training/development/evaluation window origins.
- `summary.json`, `case-metrics.json`: family-selected results and seed ranges.
- `diagnostics.json`: per-case/horizon/output support bins, counts, errors,
  hold-current errors and rank associations with support, age and motion.
- `support-composition.json`: feature-block contributions to distance, not
  an attribution of prediction error to those blocks.
- `audit.json`: independent prediction replay, direct normal equations,
  support-distance reconstruction and selection checks.
- `prior-timing-audit.json`: previous independent check of the reused raw-clock
  extraction. The raw/prepared sources and their limits are documented in
  [sequence transfer](../../sequence-transfer.md).
- `arp-forecast.png` / `.svg`: 240 ms conditional predictions across four
  held-out recordings. Error bars are sampling-seed ranges.
- `final-implementation.zip`: source package, scripts, relevant tests and
  package configuration after final formatting and validation.
- `starting-hashes.json`: identities of preexisting repository files.
- `validation.json`, `source-pytest.xml`, `wheel-pytest.xml`: final source/wheel
  verification and environment.
- `manifest.json`: hashes and sizes of the portable evidence.

Large sample, fitted-model, support and prediction arrays remain in
`artifacts/forecast-diagnosis/comparison-01` in the parent workspace. Prepared
X8/ARP recordings and raw sources remain in `artifacts/sequence-transfer`;
Nano samples also depend on previous investigations. This bundle does not
duplicate the flight corpora.

The expanded ARP splits reuse four existing recordings. No new pristine
holdout, uncertainty calibration or controller result is claimed. A direct
forecast head is not a recursive step model. Methods were selected using
development observations; independent replay does not make that selection
valid under an arbitrary future distribution shift.
