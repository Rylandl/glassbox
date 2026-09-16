# Status: gap against the charter

Measured on 2026-09-16 at commit `4a3f2b0`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v2-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options. Seven experimental modules; the harness now has a synthetic and a platform tier. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | **Measured, not met.** Platform tier v1 (`docs/harness/platform-v1.json`, digest `8d4705d8`), whole recordings held out, both models scored on identical rows at the recipe's horizon; final-step velocity m/s / body rate rad/s, generic versus best structured arm: nanodrone 0.136/0.543 vs 0.179/0.597; x8 0.222/0.132 vs 0.287/0.187; idf 0.158/0.122 vs 0.554/0.174; epfl 0.146/0.070 vs 0.526/0.217; **arp 0.176/0.715 vs 0.174/0.285**, and hold-current 0.149/0.362. Rule met on four of five corpora; inside every declared allowance. The arp failure is roll and pitch rate: worse than hold-current from the first 20 ms step on the development recording, growing linearly to 0.85/0.89 rad/s at 240 ms with a -0.2 rad/s roll-rate bias, while yaw rate beats the structured model (0.154 vs 0.323). | Every pinned corpus, whole recordings held out: generic error at or below the structured model on the same rows, and inside the allowance (nano 0.696/3.706, X8 1.601/0.764, ARP log66 0.709/2.864). | Platform tier merged at `4a3f2b0`; no recipe change yet. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | The learner reports development errors per recording against a hold-current reference, and the harness reports per-horizon and per-recording scaled errors. No forecast carries an envelope, and nothing calibrates one. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Generic track done: 48 research scripts, 24 test modules, 11 experimental modules, 85 MB of archives and 18 research pages deleted. Structured core still present (dynamics, identification, fitting, five belief modules); 25 scripts and three structured-evidence pages remain for it. | Learner, harness, telemetry adapters, controller. | Lean-down merged at `c5e84ab`. |

## Next iteration

Fix the arp body-rate failure with one recipe change, gated. Before any fit:
freeze `docs/harness/platform-v2.json` with the rule enforced and a per-corpus
regression reference taken from the merged platform-v1 run (generic final-step
numbers; no corpus may regress past reference times 1.05 plus 0.005), commit
it, and confirm the synthetic tier's reference gate still applies. Then
diagnose the named failure on arp (roll and pitch rate worse than hold-current
at one step on the development recording, linear growth, roll-rate bias) with
the frozen artifacts, name the mechanism, change one thing in the recipe, run
both tiers, and report the numbers whether or not the gate passes. Thresholds
do not move after scores are seen.
