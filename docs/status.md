# Status: gap against the charter

Measured on 2026-09-16 at commit `3f025f3`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v3-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options; recordings may declare the excitation the caller injected; and `envelope(horizon_steps)` beside them reporting what every forecast already carries. Seven experimental modules; the harness has synthetic, platform, and control tiers, and `experimental/learned_plan.py` presents the learner to the NMPC seam. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | **Measured, not met.** Platform tier v3 (`docs/harness/platform-v3.json`; the rule gates only where the reference meets it, no metric may regress past its reference), whole recordings held out, both models scored on identical rows at the recipe's horizon; final-step velocity m/s / body rate rad/s, generic versus best structured arm: nanodrone 0.136/0.543 vs 0.179/0.597; x8 0.222/0.132 vs 0.287/0.187; idf 0.158/0.122 vs 0.554/0.174; epfl 0.146/0.070 vs 0.526/0.217; **arp 0.176/0.715 vs 0.174/0.285**, and hold-current 0.149/0.362. Rule met on four of five corpora; inside every declared allowance. The arp failure is roll and pitch rate: worse than hold-current from the first 20 ms step on the development recording, growing linearly to 0.85/0.89 rad/s at 240 ms with a -0.2 rad/s roll-rate bias, while yaw rate beats the structured model (0.154 vs 0.323).. Re-measured at `3c149fa` under the enforced manifest: every corpus reproduces the regression reference to every digit, no reference regression, and the run is rejected on arp for both metrics (0.17586 against the comparator's 0.17437, 0.71546 against 0.28497). | Every pinned corpus, whole recordings held out: generic error at or below the structured model on the same rows, and inside the allowance (nano 0.696/3.706, X8 1.601/0.764, ARP log66 0.709/2.864). | Gate enforced at `783922e` (`platform-v2.json`, digest `f4796e1a`, regression reference `platform-reference.json`). Four attempts below, none accepted; the fourth bounded the recursion's gain inside the fit and the synthetic gate rejected it. arp is recorded as evidence-limited. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. It is the tier that has rejected the last two candidates: the recursion bound on hidden_hysteresis, and Control attempt 3 on hidden_hysteresis's caps and nineteen shifted-regime references. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | **Measured, not met.** Control tier v4 (`docs/harness/control-v4.json`, digest `a2ec4bea`, the same protocol as v3 computed in simulated time; two runs under a load average of 17 are byte-identical and the incumbent numbers are v3's to every digit): the tracking task of `docs/cascade-accuracy.md` (lateral sin(0.35 t) m, altitude 100 + 0.75 sin(0.3 t) m, 16 s, seeds 101 and 102), calibration whose setpoint variation moves every command at least 10% of its range (throttle now 12 to 14%, was 2 to 3%), both arms through the same bounded solver under the no-evidence override. Generic arm 60.80 m / 97.2 deg and 49.31 m / 86.5 deg; structured arm 1.18 m / 1.32 deg and 1.18 m / 1.28 deg; no terminated trial. The structured arm holds lateral position to 0.27 m but settles about 2 m high, so it does not pass the page's 0.5 m criterion either (reported, not gated). Holding trim scores 1.22 m / 0.75 deg and 1.73 m / 1.90 deg, so the task now demands authority on position and mostly on attitude. Excitation did not fix the generic arm: it still drives throttle far below trim and rests roll near its bound; the seam prices none of the learner's ignorance because the learner declares none. Attempt 2 diagnosed it and fitted no candidate: the ceiling of a perfect command response is 9.04 m, of perfect command confinement 26.96 m, and of claiming nothing 1.22 m, because the model predicts a 0.172 m/s sink out of a plant resting exactly at trim.  Control-v5 (`control-v5.json`, digest `c87b40e1`) now supplies the calibration pilot's known additive dither to the learner's recordings as declared excitation, 5.7 / 2.5 / 2.3% of each command's range against the 12 to 38% the whole command moves; the recipe ignores it, so the numbers are v4's to every digit. Excitation attempt 1 then diagnosed that dither and fitted no candidate: it identifies no channel's one-step response (mean over its own standard error 1.43 / 3.60 / 0.67 between the three recordings, against 2.68 / 7.05 / 4.94 for the whole command), and holding the affine block's command columns to what it does identify flies at 88.19 m against a 63.84 m regression ceiling. Control attempt 3 then held the affine block to what the *whole* command identifies -- roll and pitch at 0.997 / 0.981 against the plant, throttle not identified and left to the ridge -- and was rejected by the synthetic gate before the control tier ran; measured as a diagnostic rather than a gate run, that candidate flies repetition 0 at **14.306 m / 63.492 deg** against the 60.797 / 97.180 reference, still 12 times the structured arm, and its trained one-step command Jacobian is not the response its own affine columns are held to (roll -0.86 against the plant, -0.89 against the held response). | Meet or beat the structured model on the matched Cascade trial set. | control-v3 reference merged at `35b85ec`; the envelope (`7224ace`) left tracking identical. Attempt 2 and excitation attempt 1 below, no candidate from either; attempt 3 rejected at `0b57954`. |
| Live improvement | **Measured, not met.** Live tier v2 (`docs/harness/live-v2.json`, digest `39394675`): control-v3's plant, task, calibration and arms; the structured belief flies from the start; the generic learner refits on the trial's own 40-interval blocks through the existing transition buffer and refinement worker, and is offered when its held-out forecast error on the newest block beats the structured belief's. The trajectory is computed in simulated time (no wall-clock fallbacks, refits released a declared 40 intervals after their block, worker driven synchronously); two runs are byte-identical. Every refit took 0.86 to 0.91 s inside a 4.0 s budget with no overrun or dropped block. The swap happened at interval 140 (7.0 s) in both adopting trials because the candidate's forecast error (0.032 / 0.024 m/s, rad/s) beat the structured belief's (0.158 / 0.080), and tracking then went from 0.87 m / 1.7 deg to 19.4 m / 86 deg and from 0.90 m / 1.6 deg to 5.7 m / 14.9 deg. Hold-current scores 0.021 m/s / 0.0014 rad/s on the decision block, better than both models: a held-out forecast comparison in a quiet regime reads no command authority.  Live-v3 (`live-v3.json`, digest `4d39b495`) adds a seeded declared dither to the active controller's command on both arms; two runs are byte-identical; the structured arm before the swap moves to 0.80 m / 2.11 deg and 0.98 m / 1.92 deg; the swap comes at interval 140 and 220; tracking after it is 36.1 m / 81.6 deg and 11.8 m / 85.8 deg, because the recipe still ignores the excitation. A block cannot be made to carry the command response either: two seconds holds 0.41 / 0.67 / 0.54 of a cycle of each channel's dither, and over a frozen arm's seven blocks the identified response is 0.15 to 1.28 times its own standard error at direction cosines of -0.911 to +0.028. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | Live tier merged at `94ea90e`; mechanics met, rule not met. Excitation attempt 1 below, no candidate. |
| Evidence | **Measured, not met.** Evidence tier v2 (`docs/harness/evidence-v2.json`, band 85 to 95% and now `enforced: true`; `evidence-v1` was the same contract with the flag false for its own first measurement and is deleted): every forecast carries a split-conformal 90% half-width per horizon step and channel, calibrated only on the recipe's own development windows, stored in the artifact and read through `envelope()`; the plan model maps it into the controller's diagonal tangent covariance and `uncertainty_available` is true. Held-out coverage against the band at the recipe's horizon, world velocity / body rate / rotation entries: nanodrone 0.906 / 0.858 / 0.886, x8 0.918 / 0.929 / 0.906, idf 0.890 / 0.872 / 0.897, epfl 0.894 / 0.856 / 0.873, **arp 0.755 / 0.689 / 0.717**, control reserved recording 0.833 / 0.797 / 0.799; synthetic matched regimes 0.88 to 0.94, **shifted regimes 0.13 to 0.82**. The development windows of one set of recordings are not exchangeable with a different flight or command regime, and nothing is widened. The seam charges the envelope, but with no resolved parameter direction the charge is the same for every plan and cannot move a command: the generic arm's tracking is identical to the incumbent's. One attempt below, rejected: widening every half-width by the support of its query repaired arp and the control tier's reserved recording and pushed x8 from 30 of 30 band cells inside the band to 0 of 30 above it. | Every forecast carries an envelope whose held-out coverage lands in the declared band, and the controller's robustness terms consume it. | Evidence tier and `generic-memory-v3-prototype` merged at `7224ace`; the band enforced as `evidence-v2` after attempt 1. |
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

**Attempt 4, rejected: the recursion's gain is bounded, and it is not arp's.**
The queued structural attempt, fitted as `generic-memory-v4-prototype` at
`8a31c72` 18:05:00Z and reverted at `880bdeb`. The change carried a bound on the
recursion's gain inside the fit: past the first horizon step, an error the size
of the process's own one-step motion may not come out of the recursion larger
than the process's own motion has grown by that step, both sides read off
hold-current on the training windows, the ceiling floored at no amplification so
a factor of zero — the plant's own hold-and-shift, gain one at every step —
always meets it, and the paths from an observed channel back into the next
prediction scaled by the largest factor that does, by bisection, before the
first step and after every step. No target, development row or held-out row
enters it, and no constant. The synthetic gate rejected it and the run stopped
there; the platform, control and live tiers were not run.

*Where the gain lives, on arp's own training origins.* The learned one-step map
on the augmented state has spectral radius above 1 at 100% of origins, median
1.0264 and maximum 1.0331, the largest median of the five corpora. It is a
transient, not an instability: 78.4% of the leading eigenvector's mass sits on
the explicit history-difference block, and removing that block's feature rows
collapses the 12-step gain by 5.4 times, from 0.745 to 0.138, while leaving the
radius at 1.0236. Removing the current-state rows moves the radius to 1.0126 and
raises the gain to 0.800; removing the memory rows gives 1.0193 and 0.726; and
removing either command block changes nothing at all — 1.0259 and 1.0264, gain
0.745 and 0.744 — because the command rows never enter the map's Jacobian with
respect to the state. By channel, holding the rotation entries cuts the gain to
0.363 and holding roll and pitch rate to 0.684, against 0.742 for yaw alone.

*The plant's own gain on the same origins, and what it says.* Hold-current's
motion grows 7.664 times over arp's 12 steps, a per-step 1.204 that matches the
1.194 the record names for roll and pitch; the other corpora are 12.145 over 25
steps, 5.505 over 10, 5.580 over 12 and, on epfl, 1.000 because its 0.2 s
horizon is one step and no recursion runs at all. A least-squares one-step
linearization of the plant on the same augmented rows is ill-conditioned and
gives radii of 5.08, 1.59, 3.58, 2.24 and 1.04; reported, not used. Perturbing
the forecast origin by hold-current's own first-step error, arp's recursion
returns 1.581 after one step and 6.740 after twelve — **0.879 of the plant's own
7.664**. Measured on everything the fit can read, arp's recursion amplifies less
than the plant's own motion grows. The excess is one step wide: 1.144 at the
second step and below one from the fourth on.

*And it is not arp's alone.* The same maximum past the first step, over the
process's own growth: nanodrone 1.055, x8 0.982, **arp 1.144**, idf 2.017,
control's calibration 2.680, epfl not applicable. Every one of the five corpora
has a map whose radius is above 1 at every origin — median 1.0117, 1.0080,
1.0264, 1.0014, 1.0218, and 1.0963 on the control calibration — while eight of
the nine synthetic families are inside the bound, at 0.30 to 0.81, and every one
of the nine is at or below radius 1 at the median, 0.70 to 0.99. hidden_hysteresis
is the one family above the bound, at 1.32 to 1.38, and the only other family
with any origin above radius 1 is near_periodic, 9% of them on one seed. No gain
quantity measurable on the fit's own data singles arp out.

*Two other forms of the bound are refuted by measurement.* An error-growth
bound — the model's error at the last horizon step at most its error at the
first, in hold-current's own per-step scale — is infeasible on arp: at a
recursion factor of zero the ratio is still 1.0362, the bisection returns zero,
and the fitted candidate scores 0.1969/0.5256 on log 66, a velocity regression
past the 0.1897 limit. A spectral-radius bound is degenerate: the radius is one
plus the factor times a positive quantity, so arp's 1.0304 falls to 1.0255 at
factor 0.98, 1.0050 at 0.2 and reaches 1.0000 only at 0, on every corpus.

*What the gate said.* 45 of the 51 synthetic case-and-regime scores reproduce
`reference.json` to every digit, because the bound never bound there: zero
bounded steps, factor one, parameters untouched. hidden_hysteresis bound on 990
to 1,001 of its 1,001 steps at factors down to 0.730 and breached both gates on
all three seeds — overall scaled RMSE 0.0451/0.0476/0.0373 matched against
limits 0.0253/0.0223/0.0276 and 0.0940/0.1061/0.0998 shifted against
0.0632/0.0547/0.0621, and horizon scaled RMSE 0.0714/0.0751/0.0588 against the
0.05 cap and 0.1464/0.1639/0.1543 against the 0.12 cap. The witness paired probe
was unchanged at 0.0029/0.0028/0.0040. Verify replayed all 27 cases in 54
replays to a maximum difference of 7.8e-16 and reproduced the same rejection.
The evidence measurement folded into that run accepted, with the band still
unenforced, 253 band breaches none of them gating, and its three reference
regressions all hidden_hysteresis. The whole run took 54.3 s.

*What the platform tier would have said, measured before the commit and not a
gate run.* Fitting the candidate corpus by corpus through the platform tier's
own split and scoring the manifest's own held-out rows: x8 and epfl reproduce
their references to every digit (0.2223/0.1325 and 0.1456/0.0698), because the
bound never binds on x8 and epfl's one-step horizon has no recursion; nanodrone
0.1432/0.5608 against a reference of 0.1360/0.5432 and limits of 0.1478/0.5753,
inside; **arp 0.1699/0.6194** against 0.1759/0.7155, both improving, with arp's
velocity below the structured comparator's 0.1744 for the first time and its
body rate still 2.2 times the comparator's 0.2850 and above hold-current's
0.3623; and **idf 0.1818/0.1270** against 0.1575/0.1221 and limits of
0.1704/0.1332, a velocity regression that would have rejected the tier
independently of hidden_hysteresis.

**arp should be recorded as evidence-limited.** The gain is now measured, and
bounding it is not the lever. arp's recursion is within the plant's own growth
on every row the fit can read: its development recording, log 63, has the
model's error *shrinking* over the horizon, 0.611 of its first-step error, while
the held-out log 66 has it growing 1.282 per step against the process's 1.194.
The training windows show 1.473. There is no quantity on logs 63, 64 and 65 that
the log 66 failure is visible in, which is the same finding attempt 3 reached
from the optimizer side and the record's own ceilings state from the outcome
side: an oracle scalar gain reaches 0.2824 against the 0.285 comparator only by
reading log 66, the best rule that reads no held-out row reaches 0.3300, and the
same recipe fitted on log 66 itself reaches 0.105. Two flights of rate RMS 0.65
and 0.59 do not determine a third of 0.41. That is a charter question for the
owner, not a threshold to move and not a fifth attempt.

**Reviewer decision after four attempts.** Accuracy holds on four of five
corpora and every declared allowance. The structural lever the first three
attempts left is now measured and spent: the recursion's gain is bounded above
by the plant's own growth on every row arp's fit can read, the one attempt that
carried that bound inside the fit was rejected by the synthetic gate, and the
one thing the bound did move on arp -- velocity below the structured comparator
for the first time -- it moved while idf's velocity regressed. arp is
evidence-limited by its four flights, which is a charter question for the owner
and not a threshold to move; no fifth attempt is queued. The remaining named
mechanisms all belong to the questions below.

## Control attempt 2, no candidate: the regime, not the commands

Everything below is a labelled diagnostic fit or an edited model flown through
the same seam on the merged `control-v3` artifacts. No candidate was fitted, so
the recipe, the artifact format and all eight frozen gate files are unchanged,
and no gate was run.

*Excitation identified roll and pitch, and did not identify throttle.* Against
the plant's own final-step response to a +0.05 command step, averaged over three
origins of the reserved recording, the selected checkpoint is throttle
0.161 / 17.78, roll 0.833 / 0.736 and pitch 0.911 / 1.119 against the structured
arm's 0.974 / 1.576, 0.843 / 0.795 and 0.892 / 0.759; the affine start is
0.103 / 9.01, 0.864 / 1.255 and 0.905 / 1.104. Under `control-v2`'s unexcited
calibration the same throttle numbers were -0.056 / 26.9 and -0.048 / 25.2, so
four to six times the throttle excitation moved its direction cosine from -0.06
to 0.16 and no further. Training degrades what the start had: the one-step
command Jacobian at trim goes from cosine +0.928 to **-0.974** on roll — the
sign inverts — and from 38.0 to 102.3 in throttle magnitude, while pitch holds
at 0.998. Across 1,000 Adam steps the development rollout MSE falls 0.2081 to
0.1564, 25%, so nothing in the objective notices.

*The recordings are not what limits roll and pitch.* Each command column is
98.9% to 99.3% explained by the rest of the fit's own design (throttle 0.9930,
roll 0.9893, pitch 0.9913), and the declared exogenous dither is only 0.3% to
4.4% of each command's variance, so the calibration is overwhelmingly
closed-loop. Even so, the residualized partial one-step slope those recordings
identify scores **0.997** on roll and **0.981** on pitch against the plant's own
one-step Jacobian, at magnitude 0.85 and 0.71. Throttle is identified by
nothing: the same partial slope scores -0.119 at 32.9 times the plant's
magnitude, because the plant's whole one-step throttle response at trim has norm
0.019 across the fifteen channels.

*The flight departs at the first interval a forecast exists.* On trial 0 the
generic arm holds the initial command for two intervals while its context
fills, and on the third, `k=2` at 0.10 s, it commands throttle 0.089 against a
trim of 0.437 — a 0.35 cut, 2.6 calibration standard deviations, on its first
solve. Throttle reaches a bound at `k=7` (0.35 s) and sits at one on 76% of the
trial's intervals, where the structured arm touches no bound at all. The two
flights separate by 0.05 m at `k=11` (0.55 s), 0.25 m at `k=17` (0.85 s) and
1 m at `k=62` (3.10 s). The solver's commands leave the box the recordings
covered on 92.5% of throttle intervals, 75.3% of roll and 75.9% of pitch, with
z-scores to 4.5, and by `k=100` the model's own five-step forecast under its own
applied commands returns a rotation-matrix entry of -128 and a velocity of
43 m/s against the plant's 5.6.

*Confinement is measured and is not the remedy.* Flown again on the same trial,
one repetition each: the recipe 60.797 m / 97.180 deg; confined to the pooled
calibration command box 50.984 / 86.204; to throttle's part of it alone
65.854 / 97.579; to one calibration standard deviation about its mean
63.517 / 93.659; to half of one 46.088 / 73.227; to a quarter 43.495 / 84.106;
and confined to an **oracle** box drawn from the structured arm's own applied
commands, padded by 0.02, **26.963 / 65.245**. Forcing the solver into the exact
region where the flyable solution lives leaves the generic arm 23 times worse
than the structured arm. Checkpoint zero, the affine start flown through the
same seam, is **94.431 / 109.549** — worse than the trained model, not better,
where under `control-v2` it was 10.6 m against 60.3 m.

*Neither is the command response.* Editing the trained model's command columns
by hand and flying each: throttle muted 61.037 / 96.022; throttle and roll muted
13.354 / 20.960; every command muted **1.217 / 0.753**, which is holding trim to
every printed digit. Replacing every command column with the response the
recordings themselves identify — the partial slope above, roll at 0.997 against
the plant — gives **9.044 / 8.261**, still 7.7 times the structured arm's
1.179 m and 7.4 times holding trim. The realizable in-scope version of that
graft, a training-objective term charging the model's own one-step command
Jacobian against that measured slope, was prototyped per window and as a batch
mean over weights 0.001 to 1.0: the best is 52.156 m at weight 0.01, weight 0.1
collapses roll's magnitude to 0.071 while lifting its cosine to 0.35, and weight
1.0 stops training improving at all and returns the affine start.

*What is actually wrong is the regime.* Held at trim with the trim command, the
plant does not move: hold-current's 0.25 s error is 0.0000 on every channel. The
selected checkpoint predicts a 0.25 s drift of **-0.172 m/s** of vertical
velocity, **+0.039 rad/s** of roll rate and **-0.064 rad/s** of pitch rate out
of a stationary aircraft, and the affine start 0.394, the grafted model 0.168
and the command-muted model 0.339 in the same norm, so the drift belongs to the
design and the calibration rather than to the optimizer or to any command
column. Every one of those six errors is *inside* the model's own 90%
half-width — 0.172 against 0.383, 0.039 against 0.119, 0.064 against 0.259 —
because the envelope is calibrated on the excited development windows, so the
Evidence row's instrument prices none of it. On the reserved recording, split
into five buckets by how far the plant actually moves over 0.25 s, the model's
final-step error is flat — velocity 0.154 to 0.222 m/s, body rate 0.121 to
0.154 rad/s — while hold-current's scales with the motion, 0.135 to 0.602 and
0.185 to 0.434. In the quietest bucket, mean motion 0.390, the model is already
1.28 times *worse* than hold-current on velocity. The trial is quieter than that
bucket by an order of magnitude: the structured arm's own 0.25 s motion is 0.030
at the median and 0.317 at p95.

**No learner change makes this arm fly from these recordings, and the ladder
says so with numbers.** The ceiling of a perfect command response is 9.044 m,
the ceiling of perfect confinement 26.963 m, the ceiling of claiming nothing
1.217 m — and that last still loses position to the structured arm's 1.179 m,
which is the task working as `control-v3` intended. The calibration excites the
plant an order of magnitude harder than the task flies it, so the fit's error
floor is larger than the motion the controller has to resolve. Excitation was
added to identify the commands and it did, for roll and pitch; it moved the
calibration further from the regime being controlled at the same time. Those two
requirements are in tension and this calibration satisfies one of them.

## Excitation attempt 1, no candidate: the declaration carries no information

Everything below is a labelled diagnostic fit or an edited model flown through
the same seam on the merged `control-v5` and `live-v3` artifacts. No candidate
was fitted, so the recipe, the artifact format and all ten frozen gate files --
five manifests and the five references beside them -- are unchanged, and no gate
was run.

*The instrument is checked against the record and against the tier.* The
partial one-step slope on the whole command, over the `control-v5` fit's own
292 training windows, is throttle -0.119 at 32.79 times the plant's one-step
magnitude, roll **0.997** at 0.813 and pitch **0.981** at 0.682: the numbers
this page already carries for these recordings. The untouched `control-v5`
artifact flown through the tier's own trial code reproduces its reference to
every printed digit, 60.797 m and 97.180 degrees. The plant's own one-step
Jacobian at trim is the comparator throughout, with norms 0.0197, 1.650 and
1.495 per unit command.

*Reading the declared excitation is worse than reading the command whole.*
Splitting each applied command into the declared dither and the rest, and
taking the partial slope of the one-step state change on the dither given the
state context and the non-dither part of the command -- this iteration's
quantity -- gives throttle **-0.286** at 57.66, roll **0.828** at 0.666 and
pitch **-0.166** at 0.849. The reason is algebra before it is noise. The
applied command is the non-dither part plus the dither identically, so a design
holding the non-dither command and the dither beside it spans every command
column the recipe's own design spans; the only thing the split adds is the
three dither columns themselves, and what it takes away is the dither's earlier
rows, which it subtracts from the command-difference columns. Those two are the
same signal at these rates -- the dither's lag-1 autocorrelation is 0.9939,
0.9931 and 0.9921 -- so the slope on the dither is the coefficient on the
command plus a mixing with the dither's own lags that is nearly singular. Stated exactly, as a contemporaneous response with the
dither's own two earlier rows held, 0.49%, 0.48% and 0.42% of the dither
survives the design and the estimate returns 7,521, 72.0 and 169.7 times the
plant's magnitude.

*It is not identified, and the standard error is the spread between the
recordings.* The row-wise standard error understates, because the one-step
residual of this design is a smooth function of time and the dither is a smooth
function of time. The honest one is the spread between the three training
recordings, each of which is an independent 8-second flight. Mean over the
three, its standard error, and their ratio:

| Command | Declared excitation | Whole command |
| --- | --- | --- |
| throttle | 1.492 / 1.046, **1.43** | 1.204 / 0.449, 2.68 |
| roll | 2.450 / 0.681, **3.60** | 2.503 / 0.355, 7.05 |
| pitch | 1.078 / 1.606, **0.67** | 1.121 / 0.227, 4.94 |

Pitch's response to the declared excitation is smaller than its own standard
error. Recording by recording the direction says the same: the whole command
gives roll 0.931, 0.911 and 0.920 and pitch 0.950, 0.982 and 0.994 on the three
flights separately, and the declared dither gives roll 0.351, 0.820 and 0.768
and pitch 0.066, 0.787 and 0.594.

*A dither that was never injected does as well.* Against 32 placebo draws of
the declared form -- the same amplitudes, the same rates, the same ramp, phases
that were never applied -- the declared draw's slope is matched or exceeded in
size by 7, 24 and 15 of them, and the placebo's median direction cosine against
the plant is **better** than the declared draw's on two of the three channels,
0.962 against 0.828 on roll and 0.904 against -0.166 on pitch. That is what a
redundant regressor looks like: the estimator is reading the command, and which
smooth signal at these frequencies is called the excitation changes the answer
by as much as the answer.

*The recordings do not contain enough dither to fix that.* Each 8-second
calibration recording holds 1.66, 2.67 and 2.16 cycles of its channel's dither,
and the three training recordings hold 4.97, 8.02 and 6.49 between them. A
`live-v3` streamed block, two seconds, holds **0.41, 0.67 and 0.54 of a
cycle** -- less than one period, so no block can resolve a gain and a phase at
all, and the seven blocks of a frozen arm measure it at a mean over a
between-block standard error of 0.17, 1.28 and 1.26 on the first repetition and
0.15, 0.91 and 0.30 on the second, with direction cosines of -0.873, -0.004 and
0.028 and -0.911, 0.000 and 0.008 and magnitude ratios of 7.1 to 106.3.

*The candidate was measured rather than argued.* Holding the affine block's
command columns to the response the declared excitation identifies, and flying
the result through the same seam on the first repetition:

| The command columns held to | Position RMSE (m) | Attitude RMSE (deg) |
| --- | --- | --- |
| nothing; the recipe as it stands | 60.797 | 97.180 |
| what the **declared excitation** identifies | **88.189** | **106.821** |
| what the whole command identifies | 9.577 | 8.363 |
| the plant's own one-step Jacobian | 16.148 | 8.767 |

The queued candidate is a 45% regression against a reference of 60.797 and a
regression ceiling of 63.842 m and 102.044 degrees: the control tier would
reject it, and it would reject it for the measured reason that the response it
holds the columns to has pitch pointing the wrong way. The two rows under it
are the ceilings that row already carries, remeasured here through the tier's
own trial code.

*What amplitude would.* The standard error falls as one over the amplitude and
as one over the square root of the number of dither cycles. For a ten-sigma
reading of the same quantity at the same 8-second duration, throttle needs
**40.0%** of its declared range, roll **6.9%** and pitch **34.3%**, against the
5.72%, 2.47% and 2.30% declared now -- and against the 12.1% to 14.2%, 13.7% to
20.3% and 37.5% to 38.1% that the **whole command** moves in these recordings
today. At the amplitude now declared, the same reading needs 1,174, 185 and
5,346 seconds of calibration. Only roll is reachable as a dither. A throttle or
pitch dither that identified its channel would be as large as, or larger than,
everything the command does, which is a different calibration protocol rather
than a known additive component of one.

**The reviewer's first decision stands and its premise does not.** Declared
excitation is a legitimate signal and the contract that carries it is measured,
replayed and frozen. What this iteration measures is that at this amplitude the
declaration carries no information the commands did not already carry, and that
a fit made accountable for it would be held to a worse response than the one
the commands identify. No learner change follows from that. The protocol
amplitude is the next decision and it belongs to the harness, which is what the
previous "Next iteration" said would happen in this case.

## Control: the named mechanism, confirmed and refuted

Gate frozen at `3c149fa` 09:45Z as `control-v2.json`, the protocol of
`control-v1` constant for constant with the rule enforced. Everything below is a
labelled diagnostic solve or an edited model flown through the same seam, not a
candidate; no candidate was fitted, so the recipe and the artifact format are
unchanged.

*The arithmetic is confirmed.* Across the three calibration recordings the pilot
moves throttle with a standard deviation of 0.0225 to 0.0312 of a declared 1.0
range, roll 0.1117 to 0.1727 of 0.7, and pitch 0.2598 to 0.2671 of 0.7. The
affine start standardizes each command by that sample standard deviation and
then by the design's own column scale, so the recipe's ridge of 14.6 charges,
per full declared-range move, 0.0127 on the throttle level column against 0.4845
on roll and 2.126 on pitch, and 2.0e-4 and 6.6e-5 on the throttle difference
columns. The throttle column's penalty in physical units is 168 times weaker
than pitch's. Against the plant's own final-step response to a +0.05 step,
averaged over three origins of the reserved recording, the throttle column is
-0.048 / 25.17 at the affine start and -0.056 / 26.90 at the selected
checkpoint, against the structured arm's 0.968 / 1.394; roll is 0.764 / 1.178
then 0.594 / 0.576 against 0.798 / 0.598, and pitch 0.946 / 1.025 then
0.969 / 1.124 against 0.951 / 1.030. Stated as a full declared-range move on the
training windows, in hold-current error units, the selected checkpoint's forecast
moves 22.2 for throttle, 2.0 for roll and 4.6 for pitch, and the affine block
carries all of it: with the tanh residual zeroed the same numbers are 24.6, 2.2
and 5.3.

*The remedy is refuted, twice.* Restating the throttle column's scale as the
declared command range and restating its penalty per that range are the same
solve to every printed digit, and both do to the affine start exactly what the
diagnosis predicted: throttle -0.048 / 25.17 becomes 0.259 / 0.127. Neither
survives the fit. The ridge lives only in the initializer; the training
objective never charges for the command Jacobian, so Adam rebuilds it, and the
trained models are worse than the recipe's, at -0.062 / 52.80 with the penalty
on the command levels and 0.036 / 33.30 with it on every command column. Nor
would carrying the penalty into the objective help, because shrinkage is all any
penalty can buy: sweeping the throttle columns' ridge from 1e-4 to 1e+8 times the
recipe's moves the magnitude ratio from 25.42 to 0.0000 and never lifts the
direction cosine above **-0.044**. Roll's best is 0.764 at ratio 1.178 and
pitch's 0.965 at 1.298, both at or below the recipe's own ridge: those two
columns are already as identified as this design can make them, and the throttle
column is not identified at all.

*And small is not safe.* A bounded shooting solver that believes a command is
weak spends more of it. Flown through the same seam on the same trial, one
repetition each: the recipe 60.347 m / 121.874 deg; its affine start alone
10.557 / 66.433; the penalty per declared range on the command levels 53.686 /
117.814 and on every command column 56.417 / 121.104, the latter resting roll at
-0.294 against its -0.35 bound; the penalized affine start 45.275 / 103.335. The
solver confined to the command box the recordings actually covered is unchanged
at 60.097 / 121.448, because that box is the declared one for roll and pitch and
only narrows throttle to [0.376, 0.488]; narrowing throttle alone gives 54.583 /
117.453.

*What would fly, and what that says.* Editing the trained model's command
columns by hand: throttle muted, 51.041 / 109.790 — still lost. Throttle and
roll muted, 2.030 / 2.300. Every command muted, **0.109 / 0.000**, because this
trial's reference is the trim trajectory the plant is already on, so a model with
no command authority outscores both arms. The generic arm does not fail in one
column; at flight amplitude it misuses two, and the gate it has to clear is
cleared by claiming nothing.

*The horizon is ruled out.* The structured arm replanned at the generic arm's own
0.25 s horizon tracks at 1.738 m / 1.371 deg, 44% and 34% worse than at 0.80 s
and 35 and 89 times better than the generic arm. Its applied throttle stays in
[0.437, 0.553] and its roll in [+0.004, +0.007], never at a bound, where the
generic arm's throttle sits at a bound on 79% of intervals.

## Gate semantics, since `3416a7e`

Every tier accepts a run only when no metric regresses past its reference
times 1.05 plus 0.005, the tier's rule holds on every case where the reference
already meets it, and nothing structural fails. Rule breaches are always
reported with whether they gate; a row of this table is met only when the rule
holds everywhere, which is stronger than an accepted run. This removed the
defect where an enforced rule the incumbent already failed blocked every
change. The control task was replaced because the cruise reference certified a
model with no command authority (0.109 m / 0.000 deg); under the new task
holding trim scores 1.22 m / 0.75 deg on one seed and still wins that seed's
attitude metric against the structured arm, so attitude alone does not yet
demand authority and the pass criterion is the honest readout.

## Evidence: what the first measurement says

Gate frozen and committed at `8a130dd` 11:52:46Z, first candidate fit 12:01Z
and its first saved artifact 12:02:08Z. All three tiers accepted the candidate.
The mean fit is untouched by construction — the envelope is calibrated after
training, on windows the optimizer never saw — and the measurement confirms it:
all 51 synthetic case-and-regime overall scaled
RMSEs and all ten platform final-step RMSEs reproduce their references to every
digit, and both control repetitions reproduce theirs. Only the fingerprints
moved, because the artifact now carries the envelope.

*The robustness terms are nonzero and change nothing.* The learner resolves no
parameter direction, so there is no covariance factor and no plan-dependent
`J C J.T`; the tangent covariance it hands the seam is the same at every plan.
The tracking and terminal spread terms therefore add a constant to the
objective and leave its minimizer alone, and the generic arm's tracking is
identical to the incumbent's to every printed digit: 60.797 m / 97.180 deg and
49.308 m / 86.514 deg. Making those terms move a command needs a spread that
depends on the plan, which is what a resolved parameter direction would supply.
Meanwhile `maximum_normalized_uncertainty` is reported as infinite by the
solver, because it withholds that margin until `uncertainty_complete`, and that
stays false.

*Coverage transfers to the calibration distribution and not past it.* The
matched synthetic regime, which draws commands the way the calibration
recordings were drawn, lands inside the band or close to it on every family:
pooled 0.884 to 0.944. The shifted regime, whose command persistence drops from
0.65 to 0.25 and whose amplitude rises to 1.25, collapses to 0.132 to 0.817.
All 40 platform cells outside the band are below it and all but four are arp's:
arp runs 0.686 to 0.839 over its 36 cells, the same held-out flight whose body
rate is the accuracy row's one failure, and idf's four are its first-step
world velocity, 0.752 rising to 0.890 by the final step. nanodrone, x8 and epfl
are inside the band on all of theirs. The control tier's reserved recording, drawn
from the same pilot and seeds as the three that fit the arms, is 0.797 to 0.833:
below the band on 14 of its 15 cells, by two to five points. Nothing here was
widened to reach the band. The gap is not the estimator — the conformal
half-width covers its calibration windows at 90% by construction — it is that
the development windows of a recording are not exchangeable with a different
flight, a different command distribution, or even a fourth recording of the same
pilot.

## One root cause across Control, Live and Evidence

Every measured failure of the generic learner in the loop traces to one fact:
its calibration and its streamed recordings are closed-loop, so commands are
98.9% to 99.3% explained by the state, and open-loop forecast error under the
flown policy is blind to command response. Excitation identifies roll and
pitch at one step (0.997 / 0.981 against the plant) but training and
development selection read only forecast error and degrade that response
(roll's sign inverts); the envelope is calibrated on the same recordings and
cannot price a command direction they never varied; the seam's charge is the
same for every plan; and the live swap gate, a forecast comparison, admits a
model that has learned the regime and not the commands. The structured model
escapes because its physics fixes how commands enter. A generic learner needs
the identifying variation to be declared: either the caller supplies the
exogenous excitation it injected as a signal (a data fact, like units), or the
calibration protocol is required to contain it and the learner is made
accountable for the response it shows. Both are charter decisions for the
owner, listed below; neither is a threshold to move.

## Evidence attempt 1, rejected: widening by distance is not widening by need

The queued candidate, fitted as `generic-memory-v4-prototype` at `21eb664`
19:02:10Z with the first gate fit at 19:02:17Z, and reverted. It carried one
change to the learner's evidence: every half-width was the calibrated table
times `max(1, d / m)`, where `d` is the standardized distance of the query's own
origin features from the training windows -- the current observation, the two
earlier observations as differences from it, and the two commands applied before
it as differences from the last of them, each in the model's own normalized
units, standardized by the training windows' mean and spread and pooled as a
root mean square -- and `m` is the median of that same distance over the
development windows the table is calibrated on. No new constant, no caller
option, no platform branch. What it measured is recorded here; the candidate
itself is not kept, the learner is unchanged at `generic-memory-v3-prototype`,
and the four references are the ones it was measured against. The band is
enforced from `evidence-v2` so that the rule this was judged by is the rule the
manifest states.

*The support statistic separates three of the four named cases.* Median query
distance over the development median: every shifted synthetic regime 1.9 to 3.4
against its own matched regime's 1.00 to 1.15, with no matched case above 1.15
and no shifted case below 1.9; arp 1.74 against nanodrone 1.00, idf 1.00, epfl
1.00 and x8 1.25; the control tier's reserved recording 1.17. Within a case the
distance orders the error too: the Spearman correlation of the distance against
the row's mean absolute error over its half-width is 0.83 on arp, 0.87 on x8,
0.84 on nanodrone, 0.90 on epfl, 0.66 on idf and 0.36 on the reserved recording,
and coverage in the closest fifth of each corpus's rows against the farthest
fifth is 0.965 against 0.377 on arp, 0.998 against 0.625 on x8, 0.991 against
0.728 on nanodrone, 1.000 against 0.548 on epfl, 0.975 against 0.772 on idf and
0.841 against 0.669 on the reserved recording.

*The fourth case it does not reach.* The held-at-trim origin whose forecast
drifts 0.172 m/s of vertical velocity out of a stationary aircraft sits at 0.39
of the development median distance, and the structured arm's own flown
trajectory -- the regime the control trial actually flies -- at 0.44, with a
maximum widening of 1.20 over 612 origins. That phantom sink is inside the
data, not outside it, and no support statistic reaches it.

*What the five tiers measured.* All five ran and each `verify` replayed its own
run: synthetic 25.0 s, platform 1,119.8 s, control 101.8 s, live 112.4 s. The
mean fit is untouched by construction and the runs said so -- all 51 synthetic
case-and-regime overall scaled RMSEs and all ten platform final-step RMSEs
reproduced their references to every digit, the witness paired probe was
unchanged, and the live tier swapped at interval 140 in both adopting trials on
identical block scores. The one metric that moved at all was the control tier's
second generic trial, 49.2902 m / 86.5264 deg against 49.3076 / 86.5140, 0.04%
and 0.01%, because the seam's spread term is a larger constant at some origins
and the solver's relative-improvement test reads a constant.

Coverage at the recipe's horizon, world velocity / body rate / rotation entries,
incumbent -> candidate:

| Case | Incumbent | Candidate | Band cells inside 85 to 95% |
| --- | --- | --- | --- |
| nanodrone | 0.905 / 0.858 / 0.886 | 0.936 / 0.888 / 0.913 | 75 of 75 -> 71 of 75 |
| x8 | 0.918 / 0.929 / 0.906 | **0.998 / 0.995 / 0.990** | **30 of 30 -> 0 of 30** |
| idf | 0.890 / 0.872 / 0.897 | 0.937 / 0.923 / 0.947 | 32 of 36 -> 28 of 36 |
| epfl | 0.894 / 0.856 / 0.873 | 0.950 / 0.919 / 0.943 | 3 of 3 -> 2 of 3 |
| arp | 0.755 / 0.689 / 0.717 | **0.939 / 0.921 / 0.933** | **0 of 36 -> 28 of 36** |
| control reserved | 0.833 / 0.797 / 0.799 | **0.893 / 0.890 / 0.874** | **1 of 15 -> 15 of 15** |
| synthetic matched, pooled over families | 0.826 to 0.981 | 0.890 to 1.000 | 145 of 210 -> 120 of 210 |
| synthetic shifted, pooled over families | 0.105 to 0.869 | 0.209 to 0.972 | 7 of 195 -> 19 of 195 |

*Why it is rejected.* Under the band the tier declares, a candidate may not
trade a case that holds it for one that does not. The platform tier would have
41 gating breaches, every one of them above the band and 30 of them x8's, and
the synthetic tier 47, every one above the band and 41 of them matched regimes
that were inside it. The run recorded them as `gating` and the flag in the
manifest, not the rule, is why nothing rejected it at the time.

*What the numbers say the shape is wrong.* The widening is applied by distance
and the band is a statement about need. x8's development recording happens to
sit unusually close to its training recordings -- median distance 0.418, against
nanodrone's 0.730, idf's 0.781 and epfl's 1.344 -- so its held-out queries are
far in those units and the rule widened them by a median of 1.25 and a mean of
2.05 although its envelope already covered them. arp's development recording
sits at 0.493 and its held-out flight genuinely is elsewhere, so the same rule
widened it by a median of 1.74 and repaired it. One reference, measured from
whichever recording the fit happened to hold out, cannot separate those two.
And the shifted regimes are improved rather than repaired for the complementary
reason: the widening the statistic asks for there is 2 to 3 times while the
error at those origins is 3 to 8 times the half-width. A support statistic
states how far a query is from the calibration; what the band needs is how
fast this model's error grows with that distance, which is a quantity no
attempt has yet measured. That is the next mechanism if the owner wants one,
and it is not a threshold to move.

The rejected candidate is `21eb664` and its measurement record is this section.

## Where the loop stands

Every row has a first measurement and every candidate the charter allows
without an owner decision has been tried, gated and either merged or rejected
with numbers. Merged: the lean-down, the four harness tiers and their frozen
gates, the memory recipe (v2) and the envelope (v3), simulated-time control
and live tiers, and the recording contract that carries a declared excitation.
Rejected by their own gates: four arp attempts (start, start again, optimizer,
recursion gain), two control attempts (unexcited then excited calibration),
control attempt 3 (holding the affine block to the identified command
response, rejected by the synthetic tier), one evidence attempt (support
widening). Diagnosed to no candidate: control
attempt 2 (the regime, not the commands) and excitation attempt 1 (the
declaration carries no information at this amplitude). Met: One recipe,
Capability. Not met: Accuracy (arp only), Control, Live improvement, Evidence,
Lean (structured side). Every harness measurement is reproducible run to run
and every saved run replays and rejects tampering.

## Reviewer decisions, pending owner reversal

The loop cannot stop with five rows unmet, and the charter forbids moving a
threshold, so the reviewer takes the two decisions the charter already
permits and records them here for the owner to reverse:

1. **Declared excitation is a signal.** The charter has the caller supply
   signals, units, timing and recording boundaries. An excitation the caller
   injected into its commands is such a signal: platform-agnostic, a data fact
   with a declared meaning like "command" or "measurement". Recordings may
   therefore carry, per applied command, the exogenous component the caller
   injected, and the learner may be made accountable for the response those
   recordings show to it. This is a structural assumption (an exogenous
   component of the command), not a platform one.
2. **Trials excite.** A protocol is the reviewer's to define when it is frozen
   before any candidate and both arms receive the same data. The control and
   live tiers' calibration already inject a known command dither; the live
   tier's active controller will inject a small declared dither during the
   trial too, so streamed recordings carry identifying variation in the regime
   being flown. Both arms, both trials, the same seeded sequence.

Not decided by the reviewer: the Evidence band stays as declared (85 to 95%
on every measured case) and arp stays recorded as evidence-limited.

## Control attempt 3, rejected: the affine block is not the command response

The queued candidate, fitted as `generic-memory-v4-prototype` at `d774042`
00:23:06Z with the first gate fit at 00:23:21Z, and reverted at `0b57954`. It
carried one change to the learner. On the training windows the recipe took the
partial regression of the one-step next-state change on the applied command
given the observed context -- the affine start's own design with that command's
level columns taken out and used as the regressor, so what conditions it is the
current observation, the explicit history differences and the command
differences -- with the spread of the same slope between the training
recordings as its standard error. A command channel whose whole response was
larger than its own standard error had the affine block's columns for it held
for every one of the recipe's 1000 steps: the level column the whole response,
the difference columns zero, the rest of the block solved by the same ridge
around them, no gradient and restored after every update, while the nonlinear
correction and the memory kept their own rows of that command. A channel
identified no better than that kept the ridge's estimate. No new constant: the
least-squares cutoff is numpy's own, referred to the command's own variation
rather than to what survived the conditioning, so a command the observed
context already accounts for to within floating point identifies nothing rather
than dividing by nothing. The assumption was stated in the module docstring:
the command's variation given the observed context is exogenous to unobserved
disturbance. What it measured is recorded here; the candidate is not kept, the
learner is unchanged at `generic-memory-v3-prototype`, and the five references
are the ones it was measured against.

*The instrument reproduces this page.* On `control-v5`'s own training windows
-- the recipe reserves one of the three calibration recordings for development,
so two identify -- the partial slope is throttle **-0.1192** at 32.79 times the
plant's one-step magnitude, roll **+0.9974** at 0.813 and pitch **+0.9810** at
0.682, the numbers this page already carries. Response over its own standard
error is 0.86, 2.24 and 2.84, so roll and pitch are held and throttle is not.
Recording by recording the direction says the same: throttle -0.008 and -0.159,
roll 0.931 and 0.911, pitch 0.950 and 0.982.

*What it holds everywhere else, measured before the gate ran.* Platform, as
response over standard error: nanodrone 1.47 / 1.04 / 2.06 / 1.38, all four
motors held; arp 2.23 / 5.62 / 2.29 / 12.53, all four; epfl 1.18 / 1.60 /
2.94 / 1.15, all four; x8 0.58 / 0.86 / 0.64, none; and idf none, because its
83 training recordings each get too few of the 384 windows to state a slope of
their own and fewer than two can, so there is no spread. Synthetic: every case
of seven families holds every channel, at 2.9 to 24.0 times its standard error;
`stable_affine` holds none, because a noiseless linear plant whose command is
recoverable from the observed state and the design's own command differences
leaves no command variation to read; and the delayed-input witness holds none,
at 0.29 to 0.44.

*What the gate said.* The synthetic run took 25.1 s and rejected. Two absolute
cap breaches, both hidden_hysteresis shifted horizon scaled RMSE: **0.1476** on
seed 6101 and **0.1591** on 6303 against the 0.12 cap. Nineteen reference
regressions, every one of them a shifted regime -- hidden_hysteresis
0.1079 / 0.0801 / 0.1163 against limits 0.0632 / 0.0547 / 0.0621, near_periodic
0.0522 / 0.0490 / 0.0405 against 0.0380 / 0.0336 / 0.0190, off_periodic
0.0458 / 0.0350 / 0.0364 against 0.0277 / 0.0332 / 0.0333, deadzone_saturation
0.3759 / 0.3312 / 0.3285 against 0.3556 / 0.3175 / 0.3163, noisy_observation
0.3332 / 0.3684 / 0.3426 against 0.3084 / 0.3544 / 0.3254, delayed_nonlinear
0.2211 and 0.2470 against 0.2089 and 0.2314, and coupled_nonlinear 0.1701 and
0.1546 against 0.1601 and 0.1435. Nine of the 51 case-and-regime scores
reproduce `reference.json` to every digit and they are exactly the nine where
nothing was identified: `stable_affine`'s six and the witness's three. The
witness paired probe was unchanged at 0.0029 / 0.0028 / 0.0040.

**The matched regimes are not the failure.** Of the 27 matched scores, 6 are
identical, 7 better and 14 worse, at a median 1.005 and a maximum 1.199 of
their reference, inside the allowance on every case. Of the 24 shifted scores,
3 are identical, 1 better and 20 worse, at a median 1.152 and a maximum 3.046.
The evidence measurement folded into the run rejected as well: 241 band
breaches, every one of them below the band, 20 of them gating, and 125
reference regressions. `verify` replayed all 27 cases in 54 replays to a
maximum difference of 8.3e-16 and reproduced the rejection. The run stopped
there, so the platform, control and live tiers were not run.

*What the control tier would have said, measured as a labelled diagnostic and
not a gate run.* Fitting the same candidate on `control-v5`'s own three
calibration recordings takes 6.2 s and holds roll and pitch and not throttle,
as above. Flown through the tier's own trial code on repetition 0 it tracks at
**14.306 m and 63.492 degrees**, against the incumbent's 60.797 / 97.180 and a
regression ceiling of 63.842 / 102.044: a 4.2 times improvement the control
tier's no-regression semantics would have accepted, and still 12 times the
structured arm's 1.179 m. Mean applied throttle is 0.442 against a trim of
0.437, where the incumbent's is 0.137.

**And the mechanism it exposes is that holding the affine block does not hold
the command response.** The trained model's own one-step command Jacobian at
trim, against the plant: throttle +0.0365 at 33.07, roll **-0.8625** at 0.548,
pitch +0.9923 at 0.368. Against the response its own affine columns are held
to: throttle +0.2924, roll **-0.8873**, pitch +0.9502. Roll's sign inverts
exactly as it did with the columns free. The affine columns are the identified
response to the last digit at checkpoint zero and at every step after it, and
the nonlinear correction and the memory -- which keep their own rows of the
command -- rebuild a different response around them. The two edits that did fly
held every path the command takes, the residual's and the memory's rows
included: 9.577 m holding all three channels and 13.584 m holding the two this
rule selects.

**What the numbers say the trade is.** A single held linear command response
costs nothing where the response is linear, costs nothing at all where nothing
is identified, and is charged on the shifted regimes of the families whose
command response is genuinely state-dependent -- a hysteresis, a deadzone, a
saturation -- because it is identified on one command distribution and applied
to another. That is the Capability row's absolute caps and the Control row's
command authority asking for opposite things out of the same columns, and this
candidate is the measurement of it rather than an argument about it.

## Reviewer reading of excitation attempt 1

The declaration carries no information at this amplitude, and the reviewer's
second decision (trials excite) bought nothing: a 2 to 6% dither at these
rates is collinear with the command's own variation (lag-1 autocorrelation
0.99, half a percent of it survives the design), a placebo dither identifies
as well, and a two-second block holds less than one cycle. The identifying
variation was already in the whole command: its partial response given the
state context identifies roll and pitch at 0.997 / 0.981 against the plant on
every calibration recording, and holding the affine block's command columns to
that response flew repetition 0 at 9.58 m / 8.36 deg against the incumbent's
60.80 m / 97.18 deg, a diagnostic flight that the control gate's no-regression
semantics would accept. That is the candidate. live-v3's dither stays in place
because its reference exists and removing it is churn until a candidate needs
it; it costs the structured arm about 0.4 deg of attitude and identifies
nothing, which is recorded.

## Next iteration

None in scope. Every direction left changes a frozen gate's meaning or the
charter's identification assumption, so the reviewer stops spawning
iterations and hands the record back. In order of consequence:

1. **Control-affine structure.** The only edit that flies (9.6 m against
   60.8 m) holds every command path, which is the assumption that commands
   enter the dynamics linearly. The synthetic suite is built to charge exactly
   that on dead zones, saturation and coupled responses, and rejects it. The
   owner decides whether "commands enter affinely" is a structural assumption
   this learner makes, in which case the synthetic caps for those families
   are reissued as a new manifest version with that assumption declared, or
   whether the learner must keep a state-dependent command response and
   identify it, which no closed-loop recording measured so far supports.
2. **Declared excitation.** The reviewer's interim decision stands but is
   measured to buy nothing at 2 to 6% of range; identifying throttle and
   pitch by dither would need 40% and 34% of range, which is a different
   calibration, and the live tier's two-second blocks hold less than one
   cycle. Whether the calibration protocol should demand that is the owner's.
3. **Evidence band** on matched regimes only, or an envelope that carries
   support without overshooting where coverage was already inside the band.
4. **arp** as evidence-limited, or more recordings.

