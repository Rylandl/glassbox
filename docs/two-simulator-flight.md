# Controlled Crazyflow and Cascade model evaluation

The [frozen protocol](harness/two-simulator-flight-v1.json) compares the unchanged
public generic learner, a freshly fitted structured reference and hold-current.
The two simulators each have 24 calibration/primary conditions and 18 declared
shift conditions, spanning heading, speed, maneuver intensity and wind. Whole
parent recordings separate training, development and testing. This is a direct
prediction and finite command-response baseline, not controller qualification.

Crazyflow uses the pinned Dart free-flight equations and 100 Hz recordings.
Cascade uses clean committed source `e8f6ba6`, native integration and 20 Hz
recordings. Native hidden state is saved for truth replay and branching; the
learner receives only its declared observed channels and issued commands.
Current Cascade physics differs from Dart's earlier comparison. A new result
does not erase the earlier saved-data failures.

Use the interpreter and package versions declared in the protocol. Set
`SCIPY_ARRAY_API=1`, `JAX_ENABLE_X64=1`, and `PYTHONPATH` to this checkout's `src`
and the clean Cascade archive's `src`. The module
`glassbox.experimental.two_simulator_flight` provides `generate`, `fit`,
`evaluate`, `finalize` and `replay` stages. Every invocation takes a simulator
name and `--output`. Fit additionally takes `--arm generic` or
`--arm structured`. Generate and seal both simulator datasets before fitting.
Finalize binds every data, model and evaluation artifact. Replay requires
`--expected-bundle-sha` from the committed result record, regenerates physical
parents and branches, and predicts from saved models without refitting.

Every planned test slot remains represented. Reports distinguish requested
conditions, achieved valid-prefix motion, invalid-tail diagnostics, missing
truth, unavailable models and nonfinite predictions. Errors are averaged over
origins, parents and cells before taking the square root; shift families remain
separate. The two fixed fitting workflows have different priors, history,
objectives and window/optimizer budgets. Their comparison does not isolate
architecture at equal compute.
