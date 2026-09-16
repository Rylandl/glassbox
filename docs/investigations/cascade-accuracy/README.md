# Cascade tracking accuracy evidence

The [report](../../cascade-accuracy.md) records 33 offline ordinary cruise
tracking trials on 2026-09-14, on `experiment/generic-transition-support`.
One frozen controller compares the unchanged generic learner with a simulator-
equation oracle and prescribed forecast-error families. The oracle passes
3/3; the learned model passes 0/3. There is no operational deployment or new
learner configuration.

## Contents

- [design-plan.json](design-plan.json): task and data definitions before
  controller development.
- [controller-development.zip](controller-development.zip): the first,
  successful oracle development trial, its arrays, parameters, platform stamp,
  and development source snapshot. No controller retuning followed this trial.
- [evaluation-plan.json](evaluation-plan.json): frozen controller, learner
  recipe, 11 predictor arms, three initial-condition seeds, and scoring rules,
  recorded before the comparison fit and trials.
- [diagnostic-plan.json](diagnostic-plan.json): post-result derivative and
  command-coverage questions, recorded before these diagnostics.
- [platform.json](platform.json), [environment.json](environment.json): Cascade
  model/spec identities, initial state/command, software and numerical settings.
- [fit-report.json](fit-report.json), [model-identity.json](model-identity.json):
  unchanged recipe, automatic split and selection, cached observation budgets,
  and saved model fingerprint.
- [forecast-errors.json](forecast-errors.json): per-recording physical errors
  at each 50–250 ms horizon, for every predictor on common reserved windows.
- [tracking-results.json](tracking-results.json), [summary.json](summary.json):
  every trial, failures and truncations included, and the pooled comparison.
- [command-coverage.json](command-coverage.json): completed-command marginal
  range checks. These are not joint support or uncertainty estimates.
- [initial-command-jacobians.json](initial-command-jacobians.json): shared-history
  local derivatives and relative errors, using normalized command coordinates.
- [audit.json](audit.json): 2,946 numerical checks of cached windows, forecasts,
  scoring, bounds, and replay of every saved command trajectory. The maximum
  absolute difference is 8.85e-13. Learned predictions are independently
  implemented in NumPy; Cascade replay uses the public simulator core and
  cannot independently validate its physics. Optimizer iterations are not
  independently re-solved.
- [accuracy-vs-tracking.png](accuracy-vs-tracking.png) /
  [accuracy-vs-tracking.svg](accuracy-vs-tracking.svg): common forecast errors
  versus trial fractions, and an equal-RMS temporal-structure comparison.
- [run-artifacts.json](run-artifacts.json): hashes, sizes, and original paths
  of all comparison artifacts, including raw arrays and the saved model.
- [executed-sources.zip](executed-sources.zip) / [sources.json](sources.json):
  all Glassbox library/example Python and Cascade Python used in the comparison.
- [final-implementation.zip](final-implementation.zip) /
  [implementation-sources.json](implementation-sources.json): final Glassbox
  library, scripts, examples, focused tests, and build/test configuration.
- [validation.json](validation.json), [wheel-check.json](wheel-check.json),
  [source-pytest.xml](source-pytest.xml), [wheel-pytest.xml](wheel-pytest.xml):
  source and isolated-wheel checks. Both focused suites pass 34 tests; all
  81 packaged Python files match source. The full repository suite was not run.
- [starting-hashes.json](starting-hashes.json): hashes of the 1,348 preexisting
  workspace files. Only the documentation index and earlier accuracy discussion
  were changed among those files in this iteration.
- [manifest.json](manifest.json): hashes and sizes of bundle files, excluding
  the manifest itself.

The final example adds JSON `null` handling for nonfinite local RMSE/replay
diagnostics and binds a closure variable for lint after the comparison. All
recorded values to which those guards apply were finite. Predictor/controller
math and constants are unchanged. The audit script was added after the trials;
its second report run only fixes closure lint and scatter-marker overlap.
Both report runs produce byte-identical numerical JSON results.

## Replay

From the Glassbox repository, use the existing virtual environment with the
adjacent Cascade checkout installed. Recorded versions are Python 3.12.12,
JAX/JAXlib 0.11.1, NumPy 2.5.3, and Cascade 0.2.0. The exact Cascade Python
sources are archived alongside the model/spec hashes. The dependency checkout
and its non-Python assets must also remain available; the archives are evidence,
not self-contained environment installers.

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python examples/cascade_accuracy.py --plan docs/investigations/cascade-accuracy/evaluation-plan.json --output ../artifacts/cascade-accuracy/comparison-replay
JAX_ENABLE_X64=1 MPLCONFIGDIR=/tmp/glassbox-transfer-mpl ../.venv/bin/python scripts/report_cascade_accuracy.py --run ../artifacts/cascade-accuracy/comparison-replay --output ../artifacts/cascade-accuracy/report-replay
```

Both scripts reject an existing output directory. To audit the recorded run
without fitting or running a controller again, point `--run` to
`../artifacts/cascade-accuracy/comparison-01` and choose a fresh report directory.
The recorded final report is `../artifacts/cascade-accuracy/report-02`.
An oracle-only development run is available with `--develop` in place of
`--plan`; it is not needed to audit the comparison.

The 7.2 MB of generated comparison NPZ files remain in the local artifact tree;
the compact repository bundle records their hashes rather than duplicating
them. Reproduction at another location requires copying those artifacts or
regenerating them. Floating-point and solver behavior can depend on the
recorded dependency versions and numerical settings.

```sh
JAX_ENABLE_X64=1 ../.venv/bin/python -m pytest -q tests/test_cascade_accuracy.py tests/test_default_model.py tests/test_sequence_collection.py tests/test_public_api.py
../.venv/bin/ruff check examples/cascade_accuracy.py scripts/report_cascade_accuracy.py tests/test_cascade_accuracy.py
../.venv/bin/ruff format --check examples/cascade_accuracy.py scripts/report_cascade_accuracy.py tests/test_cascade_accuracy.py
```

The wheel test uses an unpacked built package outside the source tree, copies
these four test modules and the examples into that workspace, and runs with
float64 disabled. Cascade remains an installed optional dependency. This checks
packaged library imports and the research harness, not inclusion of the harness
in the wheel.

Three seeds are descriptive outcomes on this simulator, not a reliability
estimate. The injected output errors and oracle/learned forecast mixtures are
diagnostic constructs, not physical platform models or generic error bounds.
