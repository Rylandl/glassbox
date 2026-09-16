# Status: gap against the charter

Measured on 2026-09-16 at commit `330ab76`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | Two recipes in `experimental/default_model.py` (`generic-history-v1-prototype` retained, `generic-memory-v2-prototype` default), a golden v1 fixture, 16 experimental modules, two acceptance runners, 73 scripts. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | M2 versioned the recipe and kept v1. |
| Accuracy | v2 not measured on any platform corpus. Superseded v1 at 250 ms (velocity m/s / body rate rad/s): Nano 0.200/0.885, X8 0.440/0.246, ARP log63 0.336/1.139, log66 2.381/5.778 versus hold-current 0.266/0.648, Crazyflie 0.205 g/2.255 rad/s versus hold 0.153/1.895. Structured model on the same rows: not measured. | Every pinned corpus, whole recordings held out: v2 error at or below the structured model on the same rows, and inside the allowance (Nano 0.696/3.706, X8 1.601/0.764, ARP log63 0.583/1.147, log64 0.846/2.329, log65 0.654/1.624, log66 0.709/2.864, Crazyflie 0.535 g/4.355 rad/s). | None on v2. |
| Capability | v2 passes every M2 cap: delayed-input ratio 0.102, witness ratio 0.051, paired-probe first step 0.0025 to 0.0037 against a 0.2 floor; ordinary families 0.924 aggregate, hysteresis 1.072 inside cap. | Pass the harness caps on every run. | M2 replaced the incumbent. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | v2 reports development errors per recording only. Split-conformal error calibration exists as an experiment and is wired to neither `predict` nor the controller. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Structured core present (dynamics, identification, fitting, five belief modules), 73 scripts, 85 MB of investigation archives, 31 research pages, two acceptance runners. | Learner, harness, telemetry adapters, controller. | None. |

## Next iteration

Lean-down of the generic track, addressing "One recipe" and the generic half
of "Lean": delete the v1 recipe, its fixture and the archived-script shims;
delete the research scripts, their tests, the investigation archives and the
research pages; collapse the M1 and M2 runners into one harness with a
versioned manifest that carries the synthetic families, the delayed witness,
and the absolute caps forward as harness v1. Structured-model deletion waits
for the accuracy and control rows.
