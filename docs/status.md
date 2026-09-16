# Status: gap against the charter

Measured on 2026-09-16 at commit `3f025f3`. One row per criterion in
[`charter.md`](charter.md). "Current" is what the harness or the recorded
artifacts actually measured; "not measured" means exactly that.

| Criterion | Current | Target | Last change |
| --- | --- | --- | --- |
| One recipe | **Met.** One recipe (`generic-memory-v3-prototype`) in `experimental/default_model.py`, one model kind, one harness in `experimental/harness.py`; `fit`, `predict`, `update` with no options, and `envelope(horizon_steps)` beside them reporting what every forecast already carries. Seven experimental modules; the harness has synthetic, platform, and control tiers, and `experimental/learned_plan.py` presents the learner to the NMPC seam. | One recipe, one module, one harness; `fit`, `predict`, `update` only. | Lean-down merged at `c5e84ab`. |
| Accuracy | **Measured, not met.** Platform tier v3 (`docs/harness/platform-v3.json`; the rule gates only where the reference meets it, no metric may regress past its reference), whole recordings held out, both models scored on identical rows at the recipe's horizon; final-step velocity m/s / body rate rad/s, generic versus best structured arm: nanodrone 0.136/0.543 vs 0.179/0.597; x8 0.222/0.132 vs 0.287/0.187; idf 0.158/0.122 vs 0.554/0.174; epfl 0.146/0.070 vs 0.526/0.217; **arp 0.176/0.715 vs 0.174/0.285**, and hold-current 0.149/0.362. Rule met on four of five corpora; inside every declared allowance. The arp failure is roll and pitch rate: worse than hold-current from the first 20 ms step on the development recording, growing linearly to 0.85/0.89 rad/s at 240 ms with a -0.2 rad/s roll-rate bias, while yaw rate beats the structured model (0.154 vs 0.323).. Re-measured at `3c149fa` under the enforced manifest: every corpus reproduces the regression reference to every digit, no reference regression, and the run is rejected on arp for both metrics (0.17586 against the comparator's 0.17437, 0.71546 against 0.28497). | Every pinned corpus, whole recordings held out: generic error at or below the structured model on the same rows, and inside the allowance (nano 0.696/3.706, X8 1.601/0.764, ARP log66 0.709/2.864). | Gate enforced at `783922e` (`platform-v2.json`, digest `f4796e1a`, regression reference `platform-reference.json`). Four attempts below, none accepted; the fourth bounded the recursion's gain inside the fit and the synthetic gate rejected it. arp is recorded as evidence-limited. |
| Capability | **Met.** Harness v1 (manifest digest `1ba15b3f`) accepts 27 of 27 cases end to end through `fit(recordings)`: every M2 cap holds, witness paired-probe first step 0.0029/0.0028/0.0040 against a 0.05 limit and a 0.2 blind floor, tightest cap margin 0.74 of cap. Reference scores frozen in `docs/harness/reference.json`. | Pass the harness caps on every run. | Lean-down merged at `c5e84ab`. |
| Control | **Measured, not met.** Control tier v3 (`docs/harness/control-v3.json`, digest `69cb4d99`): the tracking task of `docs/cascade-accuracy.md` (lateral sin(0.35 t) m, altitude 100 + 0.75 sin(0.3 t) m, 16 s, seeds 101 and 102), calibration whose setpoint variation moves every command at least 10% of its range (throttle now 12 to 14%, was 2 to 3%), both arms through the same bounded solver under the no-evidence override. Generic arm 60.80 m / 97.2 deg and 49.31 m / 86.5 deg; structured arm 1.18 m / 1.32 deg and 1.18 m / 1.28 deg; no terminated trial. The structured arm holds lateral position to 0.27 m but settles about 2 m high, so it does not pass the page's 0.5 m criterion either (reported, not gated). Holding trim scores 1.22 m / 0.75 deg and 1.73 m / 1.90 deg, so the task now demands authority on position and mostly on attitude. Excitation did not fix the generic arm: it still drives throttle far below trim and rests roll near its bound; the seam prices none of the learner's ignorance because the learner declares none. Attempt 2 diagnosed it and fitted no candidate: the ceiling of a perfect command response is 9.04 m, of perfect command confinement 26.96 m, and of claiming nothing 1.22 m, because the model predicts a 0.172 m/s sink out of a plant resting exactly at trim. | Meet or beat the structured model on the matched Cascade trial set. | control-v3 reference merged at `35b85ec`; the envelope (`7224ace`) left tracking identical. Attempt 2 below, no candidate. |
| Live improvement | **Measured, not met.** Live tier v2 (`docs/harness/live-v2.json`, digest `39394675`): control-v3's plant, task, calibration and arms; the structured belief flies from the start; the generic learner refits on the trial's own 40-interval blocks through the existing transition buffer and refinement worker, and is offered when its held-out forecast error on the newest block beats the structured belief's. The trajectory is computed in simulated time (no wall-clock fallbacks, refits released a declared 40 intervals after their block, worker driven synchronously); two runs are byte-identical. Every refit took 0.86 to 0.91 s inside a 4.0 s budget with no overrun or dropped block. The swap happened at interval 140 (7.0 s) in both adopting trials because the candidate's forecast error (0.032 / 0.024 m/s, rad/s) beat the structured belief's (0.158 / 0.080), and tracking then went from 0.87 m / 1.7 deg to 19.4 m / 86 deg and from 0.90 m / 1.6 deg to 5.7 m / 14.9 deg. Hold-current scores 0.021 m/s / 0.0014 rad/s on the decision block, better than both models: a held-out forecast comparison in a quiet regime reads no command authority. | Bounded refit on streamed recordings and a threshold swap during a Cascade run; tracking after the swap no worse. | Live tier merged at `94ea90e`; mechanics met, rule not met. |
| Evidence | **Measured, not met.** Evidence tier v1 (`docs/harness/evidence-v1.json`): every forecast now carries a split-conformal 90% half-width per horizon step and channel, calibrated only on the recipe's own development windows, stored in the artifact and read through `envelope()`; the plan model maps it into the controller's diagonal tangent covariance and `uncertainty_available` is true. Held-out coverage against the 85 to 95% band at the recipe's horizon, world velocity / body rate / rotation entries: nanodrone 0.906 / 0.858 / 0.886, x8 0.918 / 0.929 / 0.906, idf 0.890 / 0.872 / 0.897, epfl 0.894 / 0.856 / 0.873, **arp 0.755 / 0.689 / 0.717**, control reserved recording 0.833 / 0.797 / 0.799; synthetic matched regimes 0.88 to 0.94, **shifted regimes 0.13 to 0.82**. The development windows of one set of recordings are not exchangeable with a different flight or command regime, and nothing was widened. The seam charges the envelope, but with no resolved parameter direction the charge is the same for every plan and cannot move a command: the generic arm's tracking is identical to the incumbent's. | Every forecast carries an envelope whose held-out coverage lands in the declared band, and the controller's robustness terms consume it. | Evidence tier and `generic-memory-v3-prototype` merged at `7224ace`. |
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

## Next iteration

Nothing on the Accuracy row. arp's structural attempt has been made and
measured; the corpus is recorded as evidence-limited above, and the charter
question that follows is the owner's. Every other unmet row -- Control, Live
improvement, Evidence -- traces to the one root cause named above, and none of
them can be moved without one of the owner decisions below, because the
identifying variation the generic learner needs is not in the recordings it is
given and no change to the learner can put it there.

Questions for the owner, in order of consequence: whether the caller may
declare exogenous excitation as a signal so command response can be identified
from closed-loop recordings; whether the calibration protocol must contain the
regime a trial flies as well as command excitation; whether the Evidence band
should be declared on matched command regimes only; and whether arp's
four-flight corpus, now measured as evidence-limited, is accepted as such or
re-collected.
