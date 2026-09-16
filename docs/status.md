# Status: gap against the charter

Measured on 2026-09-16 at commit `c5e84ab`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v2-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options. Seven experimental modules, 2,267 lines. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | v2 not measured on any platform corpus. Superseded v1 at 250 ms (velocity m/s / body rate rad/s): Nano 0.200/0.885, X8 0.440/0.246, ARP log63 0.336/1.139, log66 2.381/5.778 versus hold-current 0.266/0.648, Crazyflie 0.205 g/2.255 rad/s versus hold 0.153/1.895. Structured model on the same rows: not measured. | Every pinned corpus, whole recordings held out: v2 error at or below the structured model on the same rows, and inside the allowance (Nano 0.696/3.706, X8 1.601/0.764, ARP log63 0.583/1.147, log64 0.846/2.329, log65 0.654/1.624, log66 0.709/2.864, Crazyflie 0.535 g/4.355 rad/s). | None on v2. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | The learner reports development errors per recording against a hold-current reference, and the harness reports per-horizon and per-recording scaled errors. No forecast carries an envelope, and nothing calibrates one. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Generic track done: 48 research scripts, 24 test modules, 11 experimental modules, 85 MB of archives and 18 research pages deleted. Structured core still present (dynamics, identification, fitting, five belief modules); 25 scripts and three structured-evidence pages remain for it. | Learner, harness, telemetry adapters, controller. | Lean-down merged at `c5e84ab`. |

## Next iteration

Accuracy measurement. Extend the harness with a platform tier that, for every
pinned corpus on disk, holds out the corpus's declared evaluation recordings,
fits the generic recipe on the rest, fits the structured model on the same
training recordings, scores both at 250 ms on exactly the same held-out rows,
and reports the task allowance beside them. This iteration measures; the gate
that the generic error must be at or below the structured error and inside the
allowance is frozen in the platform manifest before the first accuracy change.
