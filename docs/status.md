# Status: gap against the charter

Measured on 2026-09-16 at commit `783922e`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v2-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options. Seven experimental modules; the harness now has a synthetic and a platform tier. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | **Measured, not met.** Platform tier v1 (`docs/harness/platform-v1.json`, digest `8d4705d8`), whole recordings held out, both models scored on identical rows at the recipe's horizon; final-step velocity m/s / body rate rad/s, generic versus best structured arm: nanodrone 0.136/0.543 vs 0.179/0.597; x8 0.222/0.132 vs 0.287/0.187; idf 0.158/0.122 vs 0.554/0.174; epfl 0.146/0.070 vs 0.526/0.217; **arp 0.176/0.715 vs 0.174/0.285**, and hold-current 0.149/0.362. Rule met on four of five corpora; inside every declared allowance. The arp failure is roll and pitch rate: worse than hold-current from the first 20 ms step on the development recording, growing linearly to 0.85/0.89 rad/s at 240 ms with a -0.2 rad/s roll-rate bias, while yaw rate beats the structured model (0.154 vs 0.323). | Every pinned corpus, whole recordings held out: generic error at or below the structured model on the same rows, and inside the allowance (nano 0.696/3.706, X8 1.601/0.764, ARP log66 0.709/2.864). | Gate enforced at `783922e` (`platform-v2.json`, digest `f4796e1a`, regression reference `platform-reference.json`). Attempt 1 rejected, see below. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | The learner reports development errors per recording against a hold-current reference, and the harness reports per-horizon and per-recording scaled errors. No forecast carries an envelope, and nothing calibrates one. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Generic track done: 48 research scripts, 24 test modules, 11 experimental modules, 85 MB of archives and 18 research pages deleted. Structured core still present (dynamics, identification, fitting, five belief modules); 25 scripts and three structured-evidence pages remain for it. | Learner, harness, telemetry adapters, controller. | Lean-down merged at `c5e84ab`. |

## Diagnosis on record

The arp failure is in the affine start, not the optimizer. Rebuilt from the
saved artifact, checkpoint zero is already worse than hold-current at the
first 20 ms step of the development log (roll/pitch rate 0.130/0.230 versus
hold 0.096/0.169) while 3.3 times better than hold on the two training logs;
Adam then barely moves it (development MSE 0.493 at step 0, 0.451 at the
selected step 100). About 65% of the roll-rate and 62% of the pitch-rate
coefficient mass sits on the 75 explicit history-difference columns; zeroing
only those columns makes the held-out first step better than hold (0.094/0.136)
while worsening training error. That is variance from an almost unregularized
ridge (1% of the design scale), and every leave-one-log-out split shows it.
Loss scaling and window coverage are refuted: the failure exists before any
weighting, and the 384 windows cover every row of both training logs.

**Attempt 1, rejected.** A development-selected ridge ladder (0.01 to 100)
replacing the fixed fraction. Gate frozen at `783922e` 06:01Z, first candidate
fit 06:21Z. Synthetic tier accepted (24 of 27 cases bit-identical, the rest
inside the gate). Platform tier rejected for two measured reasons: arp body
rate 0.343 against the structured 0.285 (down from 0.715; roll/pitch rate
0.363/0.411, now at hold-current rather than below the comparator, and yaw
rate gave ground, 0.154 to 0.231), while arp velocity passed for the first time
(0.124 against 0.174); and epfl body rate regressed 15% past its reference
limit (0.0805 against 0.0783) because its few development windows chose more
shrinkage. Global shrinkage chosen by sparse development evidence is the wrong
lever: it over-shrinks channels that were fine.

## Next iteration

Attempt 2 on the same named gap, from `783922e`. One change to the affine
start that targets the history-difference variance specifically and does not
rest on a discrete choice made from a handful of development windows: for
example a structural rule that shrinks the difference block relative to the
current-state and command columns, or an affine start without explicit
difference columns, leaving history to the memory and the correction. Both
tiers gate it; thresholds do not move. If the start is fixed and arp roll and
pitch rate still sit at hold-current at 240 ms, the remaining gap is the
optimizer barely improving on the start, which is the next named gap, not
this iteration's.
