# Status: gap against the charter

Measured on 2026-09-16 at commit `783922e`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v2-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options. Seven experimental modules; the harness now has a synthetic and a platform tier. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | **Measured, not met.** Platform tier v1 (`docs/harness/platform-v1.json`, digest `8d4705d8`), whole recordings held out, both models scored on identical rows at the recipe's horizon; final-step velocity m/s / body rate rad/s, generic versus best structured arm: nanodrone 0.136/0.543 vs 0.179/0.597; x8 0.222/0.132 vs 0.287/0.187; idf 0.158/0.122 vs 0.554/0.174; epfl 0.146/0.070 vs 0.526/0.217; **arp 0.176/0.715 vs 0.174/0.285**, and hold-current 0.149/0.362. Rule met on four of five corpora; inside every declared allowance. The arp failure is roll and pitch rate: worse than hold-current from the first 20 ms step on the development recording, growing linearly to 0.85/0.89 rad/s at 240 ms with a -0.2 rad/s roll-rate bias, while yaw rate beats the structured model (0.154 vs 0.323). | Every pinned corpus, whole recordings held out: generic error at or below the structured model on the same rows, and inside the allowance (nano 0.696/3.706, X8 1.601/0.764, ARP log66 0.709/2.864). | Gate enforced at `783922e` (`platform-v2.json`, digest `f4796e1a`, regression reference `platform-reference.json`). Three attempts below, none accepted. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | No generic model has run in Cascade tracking. The last learned predictor tried there diverged in 3 of 3 trials; the simulator-equation predictor passed 3 of 3. | Meet or beat the structured model on the matched Cascade trial set. | None. |
| Live improvement | Streaming transport and the background refinement worker exist for the structured belief only. No generic refit-and-swap. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | None. |
| Evidence | The learner reports development errors per recording against a hold-current reference, and the harness reports per-horizon and per-recording scaled errors. No forecast carries an envelope, and nothing calibrates one. | Every forecast carries an envelope with held-out coverage in a declared band, consumed by the controller's robustness terms. | None. |
| Lean | Generic track done: 48 research scripts, 24 test modules, 11 experimental modules, 85 MB of archives and 18 research pages deleted. Structured core still present (dynamics, identification, fitting, five belief modules); 25 scripts and three structured-evidence pages remain for it. | Learner, harness, telemetry adapters, controller. | Lean-down merged at `c5e84ab`. |

## Diagnosis on record

The first diagnosis put the arp failure in the affine start; attempt 2 refuted
that and attempt 3 refuted the optimizer behind it. As measured then: rebuilt
from the saved artifact, checkpoint zero is already worse than hold-current
at the first 20 ms step of the development log (roll/pitch rate 0.130/0.230
versus hold 0.096/0.169) while 3.3 times better than hold on the two training
logs; Adam then barely moves it (development MSE 0.493 at step 0, 0.451 at the
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

**Attempt 2, no candidate.** Both start-side directions were refuted before a
gate run. A start re-solved without the explicit difference columns gives arp
log 66 body rate 0.684 at checkpoint zero and 0.670 trained, against 0.780 and
0.647 with them: the variance is not localized in that block, because the
current-state and command columns rebuild the same amplification once it is
removed (the same solve with those coefficients merely zeroed gives 0.367, so
attempt 1's ablation was not predictive of a refit). No design-derived quantity
orders the corpora the way the needed shrinkage does: nanodrone's difference
block is the most collinear of the five (median variance inflation 5.9e6
against arp's 6.8e3) and needs none, and penalties proportional to column gain,
inflation, or block width leave arp unchanged or cost idf, nanodrone, and the
synthetic families 30% to 880% of development error. Sweeping the global ridge
shows the best body rate any affine start reaches on log 66 is **0.323**,
against hold-current 0.366 and the structured 0.285.

**Attempt 3, no candidate: the gap is not the optimizer either.** The recipe's
arp fit trains on logs 64 and 65 (54 s and 58 s), develops on log 63 (28 s) and
is scored on log 66 (76 s), with 384 training windows, a 25-step context, a
12-step horizon and 7,245 parameters. Every named suspect is inert or refuted,
and the whole training procedure has a measured ceiling above the gate.

*The clip is inert.* Over 1,000 steps the gradient norm before clipping is
median 0.585, p95 1.47, max 3.07 against a threshold of 5: it binds on 0 of
1,000 steps, and raising it to 50 returns a bit-identical model.

*Training does reduce the rollout loss.* Hold-scaled training MSE falls 0.1364
to 0.0294 over 1,000 steps, a 4.6x reduction, while development moves 0.4933 to
0.4512, 8.5%. The 11 development checkpoints span 0.4512 to 0.4704, a 4.3%
spread against an 8.5% total improvement, and the 0.3% that separates step 100
from step 400 chooses between held-out body rates of 0.647 and 0.563. Selection
is reading noise, but selecting perfectly does not help: an oracle that scores
every checkpoint on log 66 itself stops at 0.513.

*Loss weighting is refuted.* At the start the nine rotation entries take 54.4%
of the training loss, the three rate channels 36.5% (yaw alone 23.4%, roll and
pitch together 13.1%) and velocity 9.1%, with the floor binding only on `R00`
and `R11` at short horizons; per-horizon shares run 3.9% at 20 ms to 13.1% at
240 ms, so the hold scaling already equalizes horizons. Giving roll, pitch and
yaw the entire loss moves log 66 body rate only from 0.647 to 0.593.

*The failure is compounding, not the map.* On log 66 the fitted model beats
hold-current at the first step and loses after it: pooled body rate 0.0453
against 0.0510 at 20 ms, crossing at 60 ms (0.1362 against 0.1342) and reaching
0.6472 against 0.3658 at 240 ms. Roll and pitch-rate error grows 1.282 per step
against the process's own 1.194; on the training windows the same excess is
1.223 against 1.197. The out-of-distribution excess, 1.282/1.194 over the
remaining 11 steps, is 2.18x, which is the whole gap. Only 5% to 10% of the
final-step error is bias.

*The data, not the fit.* On the 3,712 origins of logs 64 and 65 that the 384
windows never sampled, body rate is 0.387 against hold 0.787. Training on the
first half of log 66 and scoring its second half gives 0.174 against hold 0.281.
Fitting the same recipe on log 66 itself and scoring the manifest's own 1,892
held-out rows gives 0.115 at the selected checkpoint and 0.105 at step 1,000,
against the structured comparator's 0.285. The model
class, the features and this Adam all reach arp's rotational dynamics; two
flights whose rate RMS is 0.65 and 0.59 rad/s (p99 3.24 and 3.16) do not
determine them for a third whose rate RMS is 0.41 (p99 1.66).

*The ceiling.* On the manifest's own 1,892 held-out rows the gate needs body
rate at or below 0.285 and hold-current is 0.3623. Selected checkpoints across
learning rates 0.002 to 0.02, 1,000 and 5,000 steps, batches 16 and 64 and
checkpoint grids of 10 and 100 steps land between 0.525 and 0.729, and the last
step of each between 0.514 and 0.651. On a 256-window sample of log 66, where
hold-current is 0.366, the rest of the sweep agrees: learning rates 0.0005 and
0.001, full batch, and clips 0.5 and 50 give 0.474 to 0.724; 768 to 3,072
training windows reach 0.496 under an oracle; spending the development recording
on training instead reaches 0.550. Folding an oracle scalar gain into the
trained delta, chosen knowing log 66, reaches 0.301 for the recipe and
**0.2824** at its very best over every optimizer variant tried (learning rate
0.02 at gain 0.25) — 0.9% under the threshold, and only with two choices made
by reading the held-out log. The best rule that reads no held-out row, a
closed-form per-channel calibration slope of the one-step prediction on the
development windows, gives arp 0.1271/0.3300: velocity below the comparator
0.1744 for the first time and body 54% below the reference 0.7155 and below
hold-current, but still 16% above the gate, and it leaves x8 at 0.2372 against a
0.2385 regression limit and nanodrone body at 0.5487. No candidate was fitted.

**Reviewer decision after three attempts.** Accuracy holds on four of five
corpora and every declared allowance. On arp the remaining lever is
structural: a recursion whose gain is carried by the fit. That attempt is
queued, not abandoned. The largest gaps in this table are now the three rows
with no measurement at all, so the loop rotates to Control, then Evidence,
then Live improvement, and returns to arp with the structural change after
each has a first measurement. If arp's body rate proves to be evidence-limited
by a four-flight corpus, that is a charter question for the owner, not a
threshold to move.

## Next iteration

Control measurement. Present the generic learner to the existing NMPC seam as
a `PlanModel` (`glassbox.control.plan`): the learner's 15 observed channels
map to the controller's rigid-body state by integrating position from
predicted world velocity and projecting the predicted rotation entries onto a
quaternion; the memory is carried from the observed history. No covariance is
claimed: run under the explicit no-evidence path the seam already provides,
and record that. Freeze `docs/harness/control-v1.json` before any trial: the
Cascade X8 plant and cruise reference, calibration recordings, duration,
seeds, controller policy and trial pairs exactly as `examples/cascade_refinement.py`
runs the structured belief's frozen arm, and the rule that the generic arm's
position and attitude tracking RMSE are at or below the structured arm's on
the same trials, with no terminated trial. This iteration measures both arms
on the same trials and reports; the rule gates merges from the first recipe
change that follows. Recipe and learner arithmetic unchanged; the synthetic
and platform tiers must reproduce their references.
