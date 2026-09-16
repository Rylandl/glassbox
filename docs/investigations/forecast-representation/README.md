# Forecast representation evidence

See the [research report](../../forecast-representation.md) for results and limits.
This bundle covers the 2026-09-14 iteration: 756 fits on reused ARP/Nano/X8 windows,
42 frozen fits for a new Crazyflie sensor evaluation, and 42 adaptive fits after
expanding only training-segment coverage.

## Contents

- [summary.json](summary.json): per-fold mean RMSE across three training seeds on
  reused recordings, including all representation families and reference arms.
- [case-metrics.json](case-metrics.json): per-case metrics, including adaptive
  Crazyflie results; [reserved-evaluation.json](reserved-evaluation.json) keeps the
  first reserved test separate.
- [case-reports.zip](case-reports.zip): complete candidate scores, chosen models,
  exact Crazyflie origins/segments, and prepared-recording metadata.
- [frozen.json](frozen.json): hashes of all original fit artifacts, fixed selection,
  reporting arms, and freeze time. It references arrays retained in the full local
  run directory, not included in this compact bundle.
- The `ablation-01-*`, `cf-frozen-01-*`, and `cf-expanded-01-*` files preserve each
  run's plan, source hashes, and executed source archive. The first two fit
  archives precede the adaptive follow-up implementation. The later switch from
  `timezone.utc` to the equivalent `UTC` alias does not alter fitting.
- [plan-errata.json](plan-errata.json): a stale budget description in the adaptive
  plan; the executed sampler and origin records are authoritative. Archived plans
  and fits have not been rewritten.
- [corpus-roles.json](corpus-roles.json), [corpus-sources.json](corpus-sources.json),
  and [corpus-decoder-source.json](corpus-decoder-source.json): identities and
  pinned URLs of the new source logs and reference decoder.
- [segmentation-diagnostics.json](segmentation-diagnostics.json): complete-record
  time-grid, age, and powered-interval counts, including discarded intervals.
- [audit.json](audit.json): independent replay and provenance results.
- [validation.json](validation.json), [source-pytest.xml](source-pytest.xml), and
  [wheel-pytest.xml](wheel-pytest.xml): focused source/wheel checks and environment.
- [final-implementation.zip](final-implementation.zip) and
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, selected tests, and configuration; includes the independent audit
  script and corrected plan writer. These are distinct from executed fit sources.
- [manifest.json](manifest.json): byte counts and SHA256 for this bundle, excluding
  the manifest itself. [starting-hashes.json](starting-hashes.json) records the
  preexisting workspace content so unrelated edits can be checked.
- [PNG](crazyflie-representation.png) / [SVG](crazyflie-representation.svg):
  reserved sensor evaluation and adaptive coverage follow-up.

The corpus repository is pinned to
`arplaboratory/data-driven-system-identification@2d267dd07b4262f579ee223d20b26a6dc9d17147`.
The Bitcraze reference decoder is pinned to
`crazyflie-firmware@349555f4dc739eb66fbc64f7d009f9b63f6a2b14`.
The independent decoder is our experiment script; the external reference decoder
is not bundled into the library. No raw recordings are duplicated here.

## Replay

The full local experiment tree is `../artifacts/representation-study` relative to
the Glassbox repository. It contains saved models, sampled arrays, predictions,
raw downloads, the built wheel, and audit output `report-02`. The ablation also
uses `../artifacts/forecast-diagnosis/comparison-01`, whose data provenance is
documented in the [prior investigation](../forecast-diagnosis/README.md).

With that tree available, independently replay all fits and regenerate the
report into a new directory:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_forecast_representation.py --output ../artifacts/representation-study/report-replay
```

To rerun the experiment, use a fresh output tree with the same subdirectory
names. Download the exact files in `corpus-sources.json` and
`corpus-decoder-source.json` into its `new-corpus` directory and verify their
hashes. Copy those two identity records into that directory as `sources.json`
and `decoder-source.json` for the audit. The scripts reject existing output
directories. Run in this order:

1. `prepare_crazyflie_reference.py --raw <root>/new-corpus --output <root>/cf-development`
   prepares only training and development logs.
2. `experiment_forecast_representation.py --phase ablation --output <root>/ablation-01`
   reuses prior matched windows. Supply `--previous` if they are elsewhere.
3. `experiment_forecast_representation.py --phase freeze-new --prepared-cf <root>/cf-development --output <root>/cf-frozen-01`
   fits and freezes selection before evaluation observations are loaded.
4. `prepare_crazyflie_reference.py --raw <root>/new-corpus --include-evaluation --output <root>/cf-complete`
   decodes evaluation after the freeze.
5. `experiment_forecast_representation.py --phase evaluate-new --prepared-cf <root>/cf-complete --frozen <root>/cf-frozen-01 --output <root>/cf-evaluation-01`
   evaluates the fixed reporting arms.
6. `experiment_forecast_representation.py --phase expand-training --raw-cf <root>/new-corpus --prepared-cf <root>/cf-complete --frozen <root>/cf-frozen-01 --output <root>/cf-expanded-01`
   performs the explicitly adaptive coverage follow-up.
7. `report_forecast_representation.py --root <root> --output <root>/report-02`
   audits source/model/data evidence and generates summaries and figures.

All script names above are under `scripts/`; use the recorded Python environment
and `JAX_ENABLE_X64=1` for fitting/replay. Different JAX/NumPy builds can change
floating-point results. Rerunning does not create a new untouched evaluation
recording: log16 has now been inspected. The compact bundle supports review of
the complete scores and provenance; numerical replay additionally needs the
saved arrays/full run tree or regeneration from the pinned data.

These are conditional observation forecasts. Source/decoder agreement verifies
extraction, not independent physical truth. Neither the local empirical-error
selector nor the representation family comparisons provide calibrated
uncertainty or a validated control model.
