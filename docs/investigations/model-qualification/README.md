# Generic model qualification evidence

The [report](../../model-qualification.md) records 27 platform-neutral synthetic
fits on 2026-09-14, on `experiment/generic-transition-support`. No vehicle,
controller, or learner implementation was changed. The investigated model is
the existing `generic-history-v1-prototype` candidate.

## Contents

- [plan.json](plan.json): protocol recorded before fitting, including data
  seeds, observation budgets, counterfactual command probes, hidden-memory
  system, and encoding transforms. Every planned fit is reported.
- [summary.json](summary.json): common command-response probes, paired
  hidden-memory errors and exact floors, and decoded encoding comparisons.
- [fit-results.zip](fit-results.zip): all 27 full results, fit reports, model
  fingerprints, selected checkpoints, training/development window identities,
  and coverage. Fits are not selected based on the qualification results.
- [identifiability-witnesses.json](identifiability-witnesses.json): distinct
  scalar dynamics coefficients fitting the same unexcited feedback transitions.
- [audit.json](audit.json): 69,450 numerical checks and 17,280 reconstructed
  cached windows; independent NumPy prediction replay, closed-form linear
  responses, paired-error identities, and common-coordinate scoring.
- [model-qualification.png](model-qualification.png) /
  [model-qualification.svg](model-qualification.svg): the three experiments.
- [environment.json](environment.json): software and float64 CPU settings.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json):
  the unchanged library, experiment script, and tests when the fits ran.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): library, scripts,
  focused tests, and test/build configuration, including the subsequent auditor.
- [run-artifacts.json](run-artifacts.json): paths, sizes, and SHA256 hashes of
  raw saved models, prediction arrays, and other run artifacts. These remain in
  the local artifact tree rather than being duplicated in this compact bundle.
- [validation.json](validation.json), [source-pytest.xml](source-pytest.xml),
  [qualification-pytest.xml](qualification-pytest.xml): 37 focused passing tests,
  including eight experiment-specific tests run before the 27 fits. The full
  repository suite and another wheel build were not run.
- [starting-hashes.json](starting-hashes.json): the 81 preexisting library
  source hashes and documentation index hash. Only the index changed among
  those files, to link the new report.
- [manifest.json](manifest.json): bundle hashes and sizes, excluding itself.

## Replay

From the Glassbox checkout with the existing virtual environment:

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python scripts/experiment_model_qualification.py --plan docs/investigations/model-qualification/plan.json --output ../artifacts/model-qualification/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_model_qualification.py --run ../artifacts/model-qualification/comparison-replay --output ../artifacts/model-qualification/report-replay
```

Both scripts reject an existing output directory. The recorded fit tree is
`../artifacts/model-qualification/comparison-01`; the final report is `report-02` in
the same parent. Point the auditor at the recorded tree to replay without
fitting again. Raw calibration trajectories regenerate deterministically from
the archived source and seeds; retained windows and predictions are also saved.
The final experiment source is unchanged from execution.

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python -m pytest -q tests/test_model_qualification.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py
../.venv/bin/ruff check scripts/experiment_model_qualification.py scripts/report_model_qualification.py tests/test_model_qualification.py
../.venv/bin/ruff format --check scripts/experiment_model_qualification.py scripts/report_model_qualification.py tests/test_model_qualification.py
```

The synthetic transitions and information ambiguities are stipulated and known
exactly. Results diagnose the frozen recipe on those examples. They neither
identify the unique cause of the earlier Cascade failure nor supply confidence
intervals, real-platform accuracy bounds, or an automatic control-admission rule.
Three data replications use the same internal learner seed; they do not measure
variation over neural initialization seeds.
