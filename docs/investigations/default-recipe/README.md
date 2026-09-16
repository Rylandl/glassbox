# Fixed generic recipe evidence

The [research report](../../default-recipe.md) describes the one-model consumer
workflow, automatic data handling, batch updates, and remaining prediction
failures. The plan was written on 2026-09-14 before these fits; every evaluation
recording had already been inspected in earlier work.

## Contents

- [plan.json](plan.json), [recipe.json](recipe.json): the frozen experiment and
  exact implementation constants. The implementation hashes JSON-encoded IDs;
  the plan's shorthand says SHA256 of recording IDs.
- [summary.json](summary.json): physical errors at the final horizon, selected
  checkpoints, actual training coverage, and fit/update timings for seven cases.
- [case-reports.zip](case-reports.zip): full per-horizon physical errors, all
  development channel errors and optimization traces, recording contracts,
  sources and roles, evaluation origins, and revision fingerprints. Seven
  initial fits and six updates were executed. ARP folds reuse recordings; two
  initial fits use identical data, so revisions are not independent experiments.
- [audit.json](audit.json): replay of 13 model revisions, automatic split and
  cache selection, 8,256 training/development windows, evaluation sampling and
  extraction, selected losses, and seven affine normal equations. This is an
  implementation audit, not independent physical ground truth or retraining of
  every optimizer step.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json):
  exact library and script sources when the experiment started.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final library,
  scripts, selected tests, and test configuration. The audit script was added
  after fitting; the fitted learner code matches the executed source.
- [run-artifacts.json](run-artifacts.json): hashes and byte counts for every
  file in the retained comparison tree, including full source arrays, sampled
  windows, serialized revisions, affine references, and evaluation predictions.
  Those large arrays are not duplicated in this compact bundle.
- [default-recipe.png](default-recipe.png) /
  [default-recipe.svg](default-recipe.svg): the four ARP cases before and after
  one batch update, relative to hold-current, at 240 ms. Labels abbreviate the
  evaluation record's log number, with exact names in the case reports.
- [environment.json](environment.json), [validation.json](validation.json),
  [source-pytest.xml](source-pytest.xml), [wheel-pytest.xml](wheel-pytest.xml):
  runtime and focused float64 source / float32 isolated-wheel test records.
- [starting-hashes.json](starting-hashes.json): preexisting workspace hashes.
- [manifest.json](manifest.json): byte counts and SHA256 hashes of bundle files,
  excluding the manifest itself.

## Replay

From the Glassbox repository, the full local run is
`../artifacts/default-recipe/comparison-01`; the final audit is `report-02`.
The sources remain at `../artifacts/real-transition/corpus` for Nano,
`../artifacts/sequence-transfer/prepared-02` for X8/ARP, and
`../artifacts/representation-study/cf-complete` for the retained Crazyflie
sensor intervals. No new data was acquired. Prior raw-clock and adapter audits
remain in the [sequence-transfer evidence](../sequence-transfer/README.md) and
[representation evidence](../forecast-representation/README.md).

Replay the audit against the retained arrays:

```sh
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_default_model.py --output ../artifacts/default-recipe/report-replay
```

Rerun the fixed recipe using those same data sources into a fresh directory:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python scripts/experiment_default_model.py --plan docs/investigations/default-recipe/plan.json --output ../artifacts/default-recipe/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_default_model.py --run ../artifacts/default-recipe/comparison-replay --output ../artifacts/default-recipe/report-replay-2
```

Both scripts reject existing output directories. The experiment accepts source
locations through `--nano`, `--prepared`, and `--cf`; they locate data and do not
change the learner recipe. Provenance records absolute source paths, so moving
an archived run requires retaining those locations or regenerating with the
new source paths. The source snapshots preserve executable code, not a portable
copy of all original recordings or runtime dependencies.

Observed coverage counts refer to uniformly sampled rows and edges; overlapping
windows count source rows once. Equal window budgets do not imply equal unique
observations or durations. Timings include fit/save work and mixed compilation
and caching effects; they are not a real-time benchmark. Forecasts condition on
future recorded inputs. No calibrated uncertainty, closed-loop performance, or
new-platform generalization claim follows from this experiment.
