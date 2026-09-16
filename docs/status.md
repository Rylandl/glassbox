# Status: gap against the charter

Measured on 2026-09-16 at commit `a38ef76`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | Met. One recipe (`generic-memory-v2-prototype`) in `experimental/default_model.py`, one saved format, six experimental modules, one harness with a frozen manifest. `fit(recordings)`, `predict`, `update(recordings)` take no options. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | This iteration deleted v1, its fixture, and the second runner. |
| Accuracy | v2 not measured on any platform corpus. Superseded v1 at 250 ms (velocity m/s / body rate rad/s): Nano 0.200/0.885, X8 0.440/0.246, ARP log63 0.336/1.139, log66 2.381/5.778 versus hold-current 0.266/0.648, Crazyflie 0.205 g/2.255 rad/s versus hold 0.153/1.895. Structured model on the same rows: not measured. | Every pinned corpus, whole recordings held out: v2 error at or below the structured model on the same rows, and inside the allowance (Nano 0.696/3.706, X8 1.601/0.764, ARP log63 0.583/1.147, log64 0.846/2.329, log65 0.654/1.624, log66 0.709/2.864, Crazyflie 0.535 g/4.355 rad/s). | None on v2. |
| Capability | Harness v1 accepted, 27 of 27 cases inside their caps in both regimes; witness paired-probe first step 0.0028 to 0.0040 against a 0.2 blind floor; worst margin coupled_nonlinear shifted at 0.186 of a 0.25 cap. 24 s on CPU. | Pass the harness caps on every run. | This iteration collapsed the M1 and M2 runners into harness v1. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | The learner reports development errors per recording against a hold-current reference, and the harness reports per-horizon and per-recording scaled errors. No forecast carries an envelope, and nothing calibrates one. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Generic half met. Structured core still present (dynamics, identification, fitting, five belief modules) with the recovery and refinement scripts and pages that CONTRIBUTING and the onboarding guide run: 24 scripts, 17 docs pages, no investigation archives. | Learner, harness, telemetry adapters, controller. | This iteration deleted the generic research scripts, tests, archives and pages. |

## Next iteration

Accuracy: measure the one recipe against the structured model on the pinned
platform corpora with whole recordings held out, on the same rows, and against
the task allowance in [accuracy-requirements](accuracy-requirements.md). That
row blocks the structured-model deletion in "Lean" and everything downstream
of it.
