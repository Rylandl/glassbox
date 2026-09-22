# Accumulator migration

**Adopted on 2026-09-22 as the single maintained implementation.** The accumulator
roughly halves quad online update cost, slightly improves online accuracy, and
improves aggregate offline forecasts and command responses. The public workflow
and generic physical assumptions remain unchanged. There is no legacy-model
selector or vehicle-family branch.

The user clarified that adoption should weigh the whole result rather than make
every regression a veto. This decision retains all measured losses and frozen
verdicts. The original failed accumulator screen remains failed; the corrected
candidate has its own prospective evidence.

## What changed

Eight stable exponential accumulators replace nonlinear recurrent hidden-state
feedback. Their input drive is nonlinear, and their positive time constants are
learned. Observed history uses a parallel weighted reduction; future memory
advances once per sample. The model retains all 100 ms lag inputs, 500 ms observed
context, the 32-unit nonlinear acceleration head, command filters and shared
rigid-body mechanics. Online fitting retains its 16-iteration PCG budget.

The memory projection keeps the original full-head feature-count initialization
scale. Merely reducing the matrix row count had previously strengthened the
initial drive and confounded the architecture comparison. This restores matched
scale; it does not establish a universally optimal initializer.

Parameters fall from 10,130 to **8,714** for four commands sampled at 10 ms and
3,399 to **3,103** for three commands at 50 ms. Shapes follow the observations and
command dimension, not platform names.

## Online comparison

All six streams and 3,137 targets are scored before assimilation. Alternating
exclusive timing blocks compare the two architectures without concurrent compute;
the v8 arm exactly reproduces the original predictions, revisions and reports.

| Measure | Accumulator versus v8 |
| --- | ---: |
| Equal-family velocity/rate error | **4.11% lower** |
| Velocity error | 4.22% lower |
| Body-rate error | 4.01% lower |
| Worst-decile error | 5.33% lower |
| Quad median / p95 whole-update time | **47.63% / 49.76% lower** |
| Fixed-wing median / p95 whole-update time | 16.60% / 16.01% lower |

Every frozen online accuracy/speed check passes. Quad updates remain roughly
40 ms against 10 ms observations; this is not real-time qualification or a live
closed-loop identification trial.

## Offline comparison

Six fits completed the full unchanged 1,000-attempt production budget: one fresh
v8 and one accumulator fit each for Dart, Crazyflow and Cascade. Both arms use
identical retained training/development windows. Checkpoint selection sees only
development data. All 8,064 saved flight queries, their eligibility rules and
factual/counterfactual commands are retained.

Forecast errors compare predicted and recorded states. Command-response errors
compare counterfactual-minus-factual changes in prediction with the same changes
in truth. Metrics are pooled within each cohort/scope/horizon, then weighted
equally across those groups, equally across families and equally between velocity
and angular-rate ratios. This is an aggregate relative-error measure, not a
fraction of successful flights.

| Reference | Forecast error | Command-response error |
| --- | ---: | ---: |
| Fresh v8, identical fitting budget | **8.29% lower** | **9.78% lower** |
| Previously deployed v8 revisions | **6.19% lower** | **2.23% lower** |

The differences are not uniformly positive:

| Family and reference | Forecast primary | Response primary |
| --- | ---: | ---: |
| Crazyflow / fresh v8 | 16.73% lower | 15.73% lower |
| Cascade / fresh v8 | 1.01% higher | 3.40% lower |
| Crazyflow / deployed v8 | 11.34% lower | 11.14% lower |
| Cascade / deployed v8 | 0.74% lower | 7.57% higher |

Cascade response velocity is 18.02% worse than the deployed revision, but 0.55%
better than freshly fitted v8. All frozen flight-aggregate checks pass. Every
per-group value, count and tail remains in the authenticated evaluation packs;
no unsuccessful cells were discarded.

All **60 numerical derivative directions** across the three arms pass centered
finite-difference checks on both simulators and held-out Dart queries. This
establishes mathematical derivative fidelity at those queries, not physical
control-response accuracy everywhere. The maintained package exactly replays all
17 saved forecast/availability arrays for its three revisions, without fitting
or optimizer calls.

## Dart and the fitting-procedure gap

Both new Dart trials use the same saved initial observations, seed, plant,
controller objective, optimizer budget and strict 1 mm target as the earlier
precision trial. Only the loaded fitted revision changes.

| Revision | Contact miss | Axis error | Finite gradient calls |
| --- | ---: | ---: | ---: |
| Historical refined v8 | 0.720 mm | 0.930° | 9,874 |
| Fresh equal-budget v8 | **3.597 mm** | 0.866° | 9,155 |
| Fresh accumulator | **3.669 mm** | 1.013° | 9,177 |

Both fresh models pass speed, attitude and deadline limits, with all callbacks
finite. Both **fail the strict 1 mm criterion**. The 0.072 mm matched architecture
difference is small compared with the shared loss against the older refinement.
It supports a fitting-procedure explanation; it does not isolate which earlier
training stage produced the difference. The conditional fine-grid precision
audit was not run because neither fresh trial passed the nominal 1 mm gate.

The deployed weights came through an earlier refinement pipeline. The maintained
production fit uses the saved 250 ms training horizon; some earlier experimental
training used longer task horizons. The two comparisons answer different
questions: equal-budget fits compare architectures, while the deployed model
measures a practical capability to recover.

The held-out Dart recordings also expose the long-horizon gap. On the same twelve
1.2 s forecasts, velocity/rate RMSE is **8.785 m/s / 14.643 rad/s** for the
accumulator, **8.731 / 13.627** for fresh v8 and **7.927 / 9.892** for the deployed
revision. Shorter horizons are recorded separately. These open-loop errors
remain improvement work; nominal feedback control is not evidence that all
long forecasts are accurate.

The offline fits and Dart trials ran alongside other work. Their wall times are
not a model speed comparison; use the exclusive online timings above.

## Reproduction and verification

The [machine-readable index](accumulator-migration.json) pins the fit pack,
evaluation packs, both Dart trials, model fingerprints and verification results.
[The development guide](../CONTRIBUTING.md) gives saved-data and no-fit replay
commands. The principal scientific commits are:

- `dbc254d`: frozen corrected online comparison; `76f06f9`: corrected model.
- `45834a3`: frozen matched offline fits and physical query set.
- `acac850`: offline verification and current-model replay entry points.
- `df44d31`: model-bound accumulator Dart trial.
- `43990ee`: fresh-v8 Dart control after the candidate miss.
- `a23a2f7`: single-implementation migration.

All 471 model/package/harness tests passed on the corrected candidate; five
additional offline/model-binding tests and all 25 installed-wheel lifecycle
checks passed afterward. Ruff passes. A redundant full-suite repeat was stopped
after 94 passes to release compute; it is recorded as interrupted, not completed.

Current package source matches the qualified accumulator source. Model and
streaming-session archive formats have changed. Historical v8 archives cannot
be converted by relabeling metadata; use their historical checkout or refit.
The live external Dart launcher was not rewritten: qualification uses the
preserved controller snapshot and Glassbox's maintained motion adapter.

The next iteration is the focused computation-reuse experiment described in
[the network review](network-review.md), not another architecture sweep or an
attempt to make every recorded benchmark cell improve.
