# Bounded Gauss–Newton suffix repair

One GN-SQP update repairs the known fifth request and returns a checked feasible
`SolveResult`. Two updates complete the prescribed 36-interval continuation.
This improves optimization work within the already selected suffix formulation;
it does not establish a production recovery or deadline qualification.

The [artifact](investigations/fast-suffix.json) uses the verified shared belief
and tick4 fixture at baseline `e15c088`. Fixture provenance traces the scenario
to `4f5cddb`; no identification was repeated. Source and fixture SHA-256 hashes,
the manifest, and the resolved research-checkout import are recorded.

The adapter freezes the first 24 physical commands exactly, optimizing six
independent four-component commands. Its 24 variables differ from the maintained
40-variable ten-block space. The complete 30-stage, 0.6-second forecast retains
actuator lag, covariance, original cost weights and nonlinear support margins.
The prefix travels through dynamic model values, so shifting it does not embed a
new prefix into compiled code. Freezing 24 commands also prevents this restricted
solver from changing its immediate action in response to a new disturbance:
its first adjustable command is 0.48 s away. It is a terminal-repair diagnostic,
not a complete feedback parameterization. Production files and defaults are unchanged.

Both fixed-work comparisons start from the identical already-shifted held-tail
waveform (hash `2e423fdf44ac5ee0b07d9013245cbac4c826049bc204ed95c846b812bc950c07`).
There is only one seed. The adapter places that seed in the common solver's
`cold_blocks` slot and disables its optional alternate warm seed. Consequently
`warm_start_used=false` is an API-slot observation: the physical seed comes from
the previous solved waveform, not from a cold-start hover search. There is no
optimizer history reuse. Cold startup is not studied.

| Same known request | Updates | Value calls | Linearizations | Finalizers | Cost | Maximum robust utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior SLSQP reference | 50 | 68 | 100 derivative calls | independent recheck | 28.630417 | 0.980752945 |
| Existing GN-SQP on suffix | 1 | 2 | 1 | 1 | 28.664114 | 0.980752945 |
| Existing GN-SQP on suffix | 2 | 3 | 2 | 1 | 28.632198 | 0.980752945 |

The original shifted waveform has cost 44.851501 and maximum utilization
1.277239 in the prior reference. The SLSQP row is the preserved earlier artifact,
not a new solve. GN counts include the one seed value call, nonlinear trials,
seed and subsequent linearizations, and the selected finalizer. Each GN update
uses the existing whitened QP capped at 20 inner iterations and a merit search
capped at 12 trials; exact QP inner iterations are not instrumented. All recorded
usable two-update solves use three value calls, two linearizations and one
finalizer. These counts exclude the independent reporting rollout.

The unchanged SQP backend retains candidates only after checking the full
nonlinear inequalities. It prepares or finalizes a prediction and attaches its
`NonlinearFeasibility` to the normal solve result, which still checks finite
values, physical bounds and elapsed deadlines. Both known repairs report
`stalled` (no convergence assertion), zero violation across 186 constraints,
and tolerance `1e-6`. An independent rollout of the returned physical commands
checks finite states, actuator states and covariance, rechecks margins, and
asserts bitwise command/prefix equality. State recomputation uses `2e-6`
absolute/relative comparison because the float32 fused and separate kernels can
round differently; this does not relax the `1e-6` nonlinear acceptance threshold.

The prescribed two-update arm was continued once for 36 intervals. Every solve
and applied interval passed; peak actual support utilization was 0.961492419 and
final utilization 0.590978146. The command applied at continuation tick24 is
asserted equal to the initial repair's command24. Thus twelve applied intervals
originate in optimized suffix commands. This remains a short continuation,
not completed recovery, terminal invariance or recursive feasibility.

Two single-request controls use the same two-update solver and an explicit
held-hover waveform. The small disturbance returns feasible support utilization
0.322756171. The state at initial roll-rate support utilization1.1 fails the QP
and returns `line_search_failed` with `not_assessed` output feasibility. Its hold
is never applied. Neither a linearized QP nor a failure hold is labeled a
feasible nonlinear plan.

The parallel feasibility study omitted timing while calibration was active.
Its artifact omits backend timing fields and uses no deadlines, so all
`deadline_met` values are null. Existing cooperative estimates can be supplied
explicitly; they are not suffix-specific execution-time bounds. The final common
boundary remains authoritative for late rejection. A controlled-clock adapter
test confirms an expired budget returns an unassessed deadline failure before
any kernel evaluation. Existing SQP tests exercise prepared-result retention,
unchecked quadratic steps, slow evaluations and output deadlines.

The probe initially stopped on a reporting attribute typo after its first solve.
A second attempt stopped on an overly strict `1e-12` state recomputation check
(max discrepancy `5.96e-8`). After correcting reporting and precision-aware
comparison, the prescribed configuration ran once to completion. No weights,
suffix lengths, seeds or budgets were searched. Test development also caught a
point-model fixture's unresolved-policy and wrong-horizon covariance setup;
the clock test now rejects before model evaluation with explicit admission
estimates. These instrumentation failures do not disappear into a success-only
account.

Reproduction (reuse the [shared fixture](terminal-suffix-investigation.md#reproduction-and-next-decision)
at `/tmp/glassbox-nmpc-fixture`):

```sh
uv run python scripts/investigate_fast_suffix.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/fast-suffix.json
uv run pytest \
  tests/test_fast_suffix_investigation.py \
  tests/test_sqp_recovery_investigation.py \
  tests/test_terminal_suffix_investigation.py -q
```

The fixed-grid warm2 baseline remains unchanged; its 120/120 deadline-free
recovery is not replaced by this 36-interval restricted experiment.

## Serialized request deadlines

The runtime wrapper starts its clock before waveform shifting, validation,
device transfer and `set_seed`. It passes the remaining budget into the common
solve, synchronizes outputs, and checks again after result assembly. The
simulation caller independently checks elapsed time across the complete wrapper
return. Any unusable or late result stops the experiment without applying its
hold. Construction, compilation, plant simulation and report logging are
explicitly outside the measured warm request. Controlled-clock tests cover
expired preparation, late result assembly, invalid seeds and invalid deadlines.

Two serialized runs of the same two-update configuration are preserved:

| Run | Prewarming | Applied intervals | Outcome |
| --- | --- | ---: | --- |
| [Initial](investigations/fast-suffix-runtime.json) | Two successful requests; shift and failure branches omitted | 1 of 36 | Second request exceeded the budget before optimization; its unassessed hold was not applied. |
| [Corrected](investigations/fast-suffix-runtime-prewarmed.json) | Two requests through the exact shifted-waveform path, plus failure buffers | 36 of 36 | Every returned plan was checked feasible and met both deadline gates. |

The first run exposed missing prewarming in the harness: waveform shifting first
ran at continuation tick1, and its eventual failure also constructed the hold
arrays for the first time. Its recorded duration includes failure assembly;
there is insufficient phase evidence to assign the miss between those cold paths
and host scheduling. The [executed initial script](investigations/fast-suffix-runtime-initial-source.py)
is archived with bytes matching its recorded source hash.

The corrected harness prewarms those branches without advancing the plant or
improving the initial seed. It additionally records each value, derivative and
finalization call, seed preparation, and compilation messages. The measured
request log emitted [no compilation messages](investigations/fast-suffix-runtime-prewarmed-compilation.log).
All calls used three evaluations, two linearizations and one finalizer. The
maximum complete caller duration used approximately 73% of the prescribed
budget (`max(requests[*].caller_elapsed_s) / deadline_s`). Its actual support
trajectory matches the deadline-free continuation, including the twelve
suffix-origin commands. No solver, horizon, covariance, weight or budget tuning
occurred between these runs; admission estimates remain disabled, and final
elapsed-time rejection remains authoritative.

This is one successful warm continuation on this host, not a hard real-time
guarantee or a measurement of cold recovery startup. The initial miss remains
part of the evidence; the corrected replay does not identify its exact cause.
Reproduce the current harness separately from other fitting or test jobs:

```sh
uv run python scripts/investigate_fast_suffix_runtime.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/fast-suffix-runtime.json \
  2> /tmp/fast-suffix-runtime-compilation.log
uv run pytest tests/test_fast_suffix_runtime.py -q
```

The frozen-prefix restriction cannot be promoted as a general feedback
controller solely because its warm request fits this budget. The following
experiment adds immediate command freedom within the same NMPC solve.

## Immediate feedback with head and suffix freedom

The fixed follow-up frees commands `[0:4]` and `[24:30]`, preserving the supplied
middle waveform `[4:24]` bitwise. These ten four-component commands give forty
variables, equal in count to the maintained ten-block problem but in a different
space. The adapter bypasses uniform block expansion and averaging. Its distinct
compilation signature and dynamic middle waveform prevent a previous request's
commands from being embedded in the kernel. Every new request clears the existing
SQP seed/checkpoint cache. Two updates, the original objective, full covariance,
support margins, actuator dynamics and 0.6 s horizon were fixed before testing.

The [feasibility report](investigations/feedback-suffix.json) records one nominal
request and two matched initial roll-rate perturbations, followed by one
36-interval nominal continuation. The perturbations are ±0.02 of the support
half-width, or ±0.021583667 rad/s. Seed waveform, actuator state, previous command,
reference and belief are identical across the three requests. All return feasible
plans, with independent whole-horizon robust utilization below one:

| Request | Returned second-motor command | Independent cost | Maximum robust utilization |
| --- | ---: | ---: | ---: |
| Nominal tick4 | 0.762277365 | 27.439175 | 0.999990404 |
| Negative perturbation | 0.736622453 | 27.293583 | 0.999990284 |
| Positive perturbation | 0.808165908 | 27.588997 | 0.999990284 |

The other three immediate motor commands remain at `[0, 1, 0]`. The paired
difference of 0.071543455 exceeds the predeclared normalized command threshold
`1e-5`; this demonstrates local state dependence of the command actually returned
for immediate application. It is stronger evidence than merely allowing early
variables or changing a later planned command. These matched probes disable
deadlines; they do not establish the same response under every deadline stop.

The nominal continuation completes 36/36 intervals, with actual support peaking
at 0.999860466 and ending at 0.999858320. It uses substantially more of the
declared envelope than the suffix-only trajectory. Initial command24 enters
the adjustable head at continuation tick21 and may be revised before application,
so the previous experiment's assertion that twelve original suffix commands were
applied unchanged no longer applies. Only the current frozen middle is checked
for exact preservation. This remains a short local continuation, not completed
recovery or recursive feasibility.

Two subsequent serialized timing runs preserve the negative result and a bounded
follow-up using the **existing unchanged** SQP work-admission estimates:

| Run | Applied intervals | Returned-plan evidence |
| --- | ---: | --- |
| [Full two-update work without admission](investigations/feedback-suffix-runtime.json) | 0 of 36 | The first solve found a feasible plan but exceeded the deadline during prediction diagnostics. Its unassessed hold was not applied. |
| [Up to two updates with existing admission checks](investigations/feedback-suffix-budget.json) | 36 of 36 | Seven requests returned a checked linearization checkpoint; 29 finalized the latest candidate. Every applied result passed both elapsed deadline gates. |

Both prewarm two identical shifted requests and the failure buffers without
advancing the plant. Both include shifting, seed preparation, device transfer,
the complete solve and result assembly within a 20 ms request budget. Independent
validation and simulated plant steps are outside that budget. Their compilation
logs ([first](investigations/feedback-suffix-runtime-compilation.log),
[budgeted](investigations/feedback-suffix-budget-compilation.log)) contain no
messages during measured requests.

Budgeted requests that returned checkpoints used two value calls and two
linearizations, with no finalizer. Their reported iteration count of two counts
optimizer rounds; it does not imply that a second nonlinear trial was accepted.
The second QP may already have been computed before admission stops its trial
evaluation. The returned checkpoint is the earlier evaluated waveform, whose
prediction and feasibility are already available. An unchecked QP step is never
applied. The largest complete caller duration used approximately 97% of the
budget (`max(requests[*].caller_elapsed_s) / deadline_s`), leaving narrow headroom.
Actual support remains inside the declared envelope throughout this run.

No work estimates, iteration limits, weights, support or covariance were tuned
between the two runtime cases. Host variation prevents treating this pair as a
latency distribution or attributing every timing difference to admission. The
seven recorded checkpoint returns do directly show the intended budget mechanism
operating with feasible results. The estimates are not worst-case execution-time
bounds; the initial rejected run remains part of the evidence.

Independent physical-waveform evaluation checks mean states, actuator states,
cost, finite covariance and every original nonlinear margin against the returned
result. Actuator-state and cost equality assertions were added after the first
feasibility run; both timing runs use the stronger checks, including all 36
applied budgeted results. Six adapter/checker tests verify the exact command
Jacobian, changing dynamic middle values, shifted seed reconstruction, atomic
rejection of invalid seeds, and rejection of inconsistent returned actuator
predictions or costs. Existing SQP and waveform-boundary tests cover candidate
retention and late-result rejection.

The reports preserve source and fixture hashes at baseline `7cc3541`. Each
executed main script is archived beside its JSON as `.source.py`; the initial
feasibility snapshot therefore retains the pre-check-strengthening source, and
the initial runtime snapshot retains the version before adding `--admission`.
No identification was repeated. Reproduce with the same shared fixture:

```sh
uv run python scripts/investigate_feedback_suffix.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/feedback-suffix.json
uv run python scripts/investigate_feedback_suffix_runtime.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/feedback-suffix-runtime.json \
  2> /tmp/feedback-suffix-runtime-compilation.log
uv run python scripts/investigate_feedback_suffix_runtime.py --admission \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/feedback-suffix-budget.json \
  2> /tmp/feedback-suffix-budget-compilation.log
```

The next validation extends this layout to cold startup and full recovery,
including a disturbance after control has begun. The results below address that
formulation question while retaining the unresolved runtime boundary.

## Cold startup and full recovery

The [full-recovery study](investigations/feedback-recovery/report.json) uses the
same four-command head, six-command suffix, uncertainty and support limits. It
starts with thirty repetitions of the actual previous command, without any
solved waveform. An explicit `SeedRequest` creates that hold or shifts a warm
waveform inside the existing timed preparation boundary. Startup uses the
reference backend's eight-update allowance; subsequent requests use two.
Simply omitting a warm start from the earlier adapter would still give two
updates, so cold/warm mode and the recorded iteration budget are checked explicitly.

The original initial state, actuator state and previous command come directly
from the verified tick0 fixture. The small case changes only the physical
disturbance. Every arm ends at absolute tick120, or 2.4 s, and stops on an
unusable solve or actual support exit. The predeclared tick4 continuation would
have used 116 intervals if original cold startup failed; it was unnecessary.

All three deadline-free cases complete the full interval count and finish
within **all twelve** local-state tolerances:

| Case | Applied intervals | Final-20-sample normalized tracking RMS | Maximum actual support utilization |
| --- | ---: | ---: | ---: |
| Original cold start | 120/120 | 0.260316610 | 0.999870181 |
| Small cold start | 120/120 | 0.009645335 | 0.339922905 |
| Original with roll-rate kick | 120/120 | 0.260532320 | 0.999870181 |

The kick adds 0.02 of the roll-rate support half-width at absolute tick60,
or 1.2 s. It changes only the current physical roll rate; actuator state,
previous command and incoming waveform remain identical. State, actuator and
seed histories match the nominal case up to that event. The
[saved-data audit](investigations/feedback-recovery-audit.json) confirms the
immediate command changes by up to 0.037419528 of its physical range. This
establishes an in-flight feedback response in the deterministic known-state
simulation, with full uncertainty still present in each NMPC forecast.

Tracking RMS follows the existing benchmark's final twenty samples and twelve
normalized coordinates. Completing the interval count, meeting attitude/rate
tolerances and meeting all state tolerances are separate report fields. Truncated
or support-exiting runs receive no completed-recovery score. The audit recomputes
every score from the saved trajectories and records the tolerance scale. These
finite-horizon results do not establish asymptotic stability or robustness to
larger disturbances, state-estimation error or other plants.

The [serialized runtime study](investigations/feedback-recovery-runtime/report.json)
uses the existing admission estimates with a declared 100 ms startup deadline
and 20 ms thereafter. Compilation is prewarmed before the run; hold creation,
shifting, validation, transfer, complete solve and synchronized result assembly
are inside the command budget. Prewarming exercises all seed modes and failure
buffers, then discards its outputs. The actual startup is again seeded by the
held previous command. Independent waveform validation, simulated plant steps
and logging are excluded from the measured solver request.

| Runtime case | Applied intervals | Outcome |
| --- | ---: | --- |
| Original cold start | 120/120 | All deadlines pass; all final-state tolerances pass. |
| Small cold start | 120/120 | All deadlines pass; all final-state tolerances pass. |
| Original with scheduled kick | 10/120 | Request10 exceeds its deadline during seeding. Its unassessed hold is not applied; the tick60 kick is never reached. |

The largest caller durations in the two completed cases consume approximately
93% and 90% of their respective request budgets. In the stopped run, request10
consumes approximately 322% of its budget before returning a failure. These are
`caller_elapsed_s / deadline_s` ratios, not execution-time bounds.
The [compilation log](investigations/feedback-recovery-compilation.log) contains
no messages during measured requests.

The audit verifies that failed request10 has **identical** physical state,
actuator state, previous command and seed to successful request10 in the original
runtime case; their preceding state, actuator and seed histories also match
bitwise. The successful call used approximately 90% of its budget. Thus the
recorded failure precedes the disturbance and does not demonstrate loss of
feasibility caused by the kick. It also prevents claiming an end-to-end timed
disturbance recovery. One value call and one linearization were attempted before
rejection; no optimizer report was produced. The recordings cannot separate
kernel execution, synchronization, Python overhead or host scheduling as the
cause of that delay. No timing case was repeated to replace this negative result.

Six harness tests cover explicit eight/two budgets, cold-seed replacement,
exact one-component kick injection, final-twenty-sample scoring, rejected-command
handling, and failed-kick/nonfinite diagnostics. A stopped kick preserves the
post-kick state as the trace endpoint. Nonfinite JSON diagnostics become null
while raw arrays remain in the NPZ, so intended failure reports remain writable.
An initial harness launch stopped before any solve on strict sample-period
float equality; the guard now permits representation error within `1e-8` s.
The physical integration period was unchanged.

Each study directory contains a predeclared design, report, compact trajectory
arrays, hashes and the exact executed main source at baseline `0723e96`.
No fitting was repeated. Reproduce the formulation first, then time its completed
cases in isolation:

```sh
uv run python scripts/investigate_feedback_recovery.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/feedback-recovery
uv run python scripts/investigate_feedback_recovery.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --formulation-report /tmp/feedback-recovery/report.json \
  --output /tmp/feedback-recovery-runtime \
  2> /tmp/feedback-recovery-compilation.log
```

The audit command below checks the committed observations, including their
specific recorded deadline failure; a new timing run may have different stops:

```sh
uv run python scripts/audit_feedback_recovery.py \
  --formulation docs/investigations/feedback-recovery \
  --runtime docs/investigations/feedback-recovery-runtime \
  --belief /tmp/glassbox-nmpc-fixture/rich-belief.json \
  --output /tmp/feedback-recovery-audit.json
```

Cold-start recovery and the small in-flight perturbation now have formulation
evidence. The following diagnostic investigates the seed-stage timing variability
before considering promotion to the maintained runtime.

## Seed timing and process history

The [fixed replay](investigations/seed-timing/report.json) resets every request
to the recorded failed tick10 inputs, including the original unshifted incoming
waveform. Its physical state, actuator state, previous command and shifted seed
hashes match both the earlier successful and failed requests. It uses the same
two-update budget, admission estimates, uncertainty, support and objective.
No commands are applied and no fitting is repeated.

The predeclared design has 64 blocks of baseline/traced/traced/baseline requests.
Baseline disables the observation wrappers. Traced requests record wall, process
CPU and caller-thread CPU clocks around validation, preparation, kernel dispatch,
seed materialization, host arithmetic and result/failure construction. The seed's
existing NumPy conversion is split into native-dtype materialization and a host
float conversion, without synchronizing additional prediction leaves. A separate
deadline-free parity pair checks bitwise-identical returned commands, physical
and actuator forecasts, cost and feasibility. GC callbacks observe both arms;
GC policy and BLAS configuration remain unchanged.

| Replay arm | Accepted requests | Median caller budget fraction | Maximum caller budget fraction |
| --- | ---: | ---: | ---: |
| Baseline | 128/128 | 0.879786 | 0.956477 |
| Traced | 128/128 | 0.886784 | 0.991092 |

All 256 returned waveforms have the same hash and independently pass the physical
rollout, frozen-middle and nonlinear-feasibility checks. Each performs two
optimizer iterations and uses its finalizer. The median traced/baseline ratio
is 1.007954. The original long spike is **not reproduced** by this bounded replay.
The added tracing consumes budget and allocates records, so its median overhead
is not a bound on its effect on later requests.

The [saved-data audit](investigations/seed-timing-audit.json) separates the two
evaluation materializations from the four linearization materializations.
Their medians consume approximately 1.75% and 23.08% of the warm deadline.
The post-linearization host gradient calculation consumes a median 0.127% and
maximum 0.171% of that budget. The 25 observed GC collections are generation0
or generation1, with the longest consuming 2.94% of the budget. Neither arm's
slowest request overlaps GC. These observations do not identify the cause of
the original outlier.

One separately declared [scenario-order pass](investigations/seed-timing-history/report.json)
then enables the same tracing during the original, small and scheduled-kick
cases, retaining their prewarming order and original deadlines. Every case still
stops at its first rejected request. This preserves scenario order, but tracing,
earlier stops and deadline-dependent checkpoint choices change the subsequent
process and numerical histories. It is not an exact reproduction of the prior
runtime process.

| Case | Applied intervals | Failed request's budget fraction | Reason |
| --- | ---: | ---: | --- |
| Original | 4/120 | 1.021142 | Feasible optimizer candidate rejected after prediction diagnostics. |
| Small | 68/120 | 1.156917 | Deadline exceeded during seeding, before optimization. |
| Original with scheduled kick | 11/120 | 0.858938 | Further work refused before a feasible candidate was retained. The kick is never reached. |

All three returned failures are unassessed holds and none is applied. None
receives a completed-recovery score. The third case illustrates why deadline
completion and feasible-plan availability are separate: it returns within the
deadline, but cannot provide a usable command. It does not establish that the
underlying optimization problem is infeasible.

The small case localizes a **new** seed overrun: 91.24% of its seed time is
inside array materialization. Evaluation materialization is 10.34 times the
median of that case's preceding accepted warm requests, and linearization
materialization is 2.77 times its corresponding median. The host gradient
calculation consumes only 0.251% of the request budget. No GC collection overlaps
this failure. An earlier accepted warm request had a longer linearization
materialization alone; the combined request cost determines the rejection.
These spans include waiting for JAX results and host access. They do not separate
kernel execution from worker scheduling or other host delays. Process CPU includes
native workers, and the three clocks are sampled sequentially. Nested phase
totals must not be summed as disjoint contributions.

The audit verifies matching request inputs through tick4 in the original case,
tick39 in the small case and tick9 in the kick case. Earlier checkpoint selection
changes small-case inputs from tick40 and kick-case inputs from tick10. Therefore
neither later failure reproduces the original saved tick10 request. Both
[replay](investigations/seed-timing-compilation.log) and
[scenario-order](investigations/seed-timing-history-compilation.log) logs contain
no JAX compilation messages during measurement. The replay log also preserves
two setup warnings about the optional configuration-output formatter.

The directories archive the exact executed sources at baseline `f69c80e`, and
the audit verifies their hashes, input identities and forecast artifacts.
Subsequent harness cleanup restores temporary class-method wrappers without
leaving bound-method aliases on the instance; a regression test covers it.
Neither timing experiment was repeated after that cleanup. Reproduce into new
output directories:

```sh
uv run python scripts/investigate_seed_timing.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --runtime docs/investigations/feedback-recovery-runtime \
  --output /tmp/seed-timing \
  2> /tmp/seed-timing-compilation.log
uv run python scripts/investigate_feedback_recovery.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --formulation-report docs/investigations/feedback-recovery/report.json \
  --seed-trace --output /tmp/seed-timing-history \
  2> /tmp/seed-timing-history-compilation.log
uv run python scripts/audit_seed_timing.py \
  --replay docs/investigations/seed-timing \
  --history docs/investigations/seed-timing-history \
  --prior docs/investigations/feedback-recovery-runtime \
  --output /tmp/seed-timing-audit.json
```

Admission estimates cannot preempt a dispatched kernel. The next implementation
slice below reduces repeated rollout work in the single-waveform seed path by
using the first linearization's already-computed residuals and margins.

## Reusing a single seed's linearization values

`GaussNewtonReference(..., reuse_single_seed=True)` now skips the standalone
value evaluation when exactly one candidate remains after warm-start compatibility
checks. The mandatory first linearization already returns that candidate's
residuals, inequality margins and prepared prediction. With multiple candidates,
the existing feasibility/violation/cost ranking and warm-seed tie rule remain.
Legacy seeding bypasses this path. This is an experimental opt-in; the default
and maintained production solver are unchanged.

The path admits the linearization and existing output reserve without charging
for the removed evaluation. It still rejects nonfinite residuals, margins,
Jacobians and squared cost before populating its caches. Finite residual entries
can have an overflowing squared norm, so checking their entries alone would lose
a guard previously provided by the standalone evaluation. Caches reset on every
request. A late linearization still fails at the common solve boundary.

The [fixed comparison](investigations/single-seed-reuse/report.json) first checks
26 saved requests without deadlines: eight predeclared ticks in each of the
original, small and perturbed full-recovery histories, plus the two recorded
runtime failures. All 26 pass **bitwise seed and returned-array parity**, along
with equal objective, nonlinear feasibility, status and iteration budgets.
Each seed uses one linearization and zero standalone evaluations instead of one
of each. Full-solve linearization and finalizer counts also remain equal, with
exactly one fewer evaluation. Original uncertainty, model support, objective,
horizon and physical command layout are retained.

Only after that gate passes does the study time 32 baseline/reused/reused/baseline
blocks for each of the two saved failed requests. Every call resets the exact
recorded inputs and applies no command. Both variants share the same compiled
kernels and admission constants. Prewarming covers both seed paths and discards
its outputs. Independent waveform checks remain outside each request timer.

| Saved request | Baseline accepted | Reused accepted | Baseline median budget fraction | Reused median budget fraction |
| --- | ---: | ---: | ---: | ---: |
| Original tick10 | 36/64 | 49/64 | 0.973746 | 0.931367 |
| Small tick68 | 36/64 | 60/64 | 0.992999 | 0.890118 |

Acceptance totals increase from 72/128 to 109/128 in this run. The
[saved-data audit](investigations/single-seed-reuse-audit.json) verifies input,
forecast and source hashes, the paired order, evaluation counts and timing
summaries. It also identifies one result per variant that remained usable at
the inner boundary but failed the caller's final elapsed-time check.

These acceptance counts do not imply that two optimizer updates completed.
Of the accepted outputs, 60 baseline and 93 reused requests performed zero
optimizer iterations and returned a currently checked seed checkpoint. Other
requests attempted work before returning a checkpoint. Under the predeclared
comparison restricted to blocks where all four calls are accepted and return
the same waveform with the same iteration count, 4 tick10 blocks and 12 small68
blocks qualify. Their median ratios of mean reused/baseline duration are
0.964206 and 0.965994. These conditional summaries describe that subset; they
are not general backend speedup estimates.

The same process then runs one separately initialized cold-start pass per variant
for each of the three scenarios, in a predeclared, partly counterbalanced order.
The original startup and warm deadlines and eight/two iteration limits apply.
All six arms stop:

| Case | Baseline applied intervals | Reused applied intervals |
| --- | ---: | ---: |
| Original | 0/120 | 3/120 |
| Small | 0/120 | 0/120 |
| Original with scheduled kick | 0/120 | 0/120 |

Five failures occur at startup after a feasible optimizer candidate exists but
cannot return within the deadline. The original/reused arm completes startup,
then fails at tick3 before finding a feasible candidate within the remaining
work budget. All rejected holds remain unapplied and no case receives a
completed-recovery score. The kick is never reached. Neither variant establishes
timed recovery in this recording.

At these six failed requests, measured seed-linearization duration is 2.06 to
2.83 times the configured linearization admission estimate. This duration includes
materialization, not just dispatch. The records do not establish why the host
costs differ from earlier runs, and removing an evaluation does not explain the
shared slowdown. The [compilation log](investigations/single-seed-reuse-compilation.log)
contains no messages during timed requests. This is one recorded comparison;
no repetition replaces its negative results.

The directory archives the executed sources at baseline `41a2293`. The audit
checks these particular saved outcomes without optimizing again. Reproduce the
comparison into a new directory, or audit the committed observations:

```sh
uv run python scripts/investigate_single_seed.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --records docs/investigations \
  --output /tmp/single-seed-reuse \
  2> /tmp/single-seed-reuse-compilation.log
uv run python scripts/audit_single_seed.py \
  --directory docs/investigations/single-seed-reuse \
  --output /tmp/single-seed-reuse-audit.json
```

## Request-local work admission

`GaussNewtonReference(..., use_observed_linearization_cost=True)` now uses the
larger of the configured linearization estimate and the current seed's measured
linearization duration before starting another linearization. The measurement
includes completed array materialization. This option is off by default and
applies only when work estimates and an explicit deadline are both supplied.
It neither changes the configured estimates nor carries observations into later
requests. Missing, nonfinite or nonpositive measurements cannot raise the floor.

The first optimizer round can still use the seed's cached derivative. Quadratic
step and value-evaluation admission, output reserve, nonlinear feasibility checks
and final elapsed-time rejection retain their existing behavior. If another
linearization no longer fits, a prepared feasible candidate can be returned;
without one, the request must reject. This scheduling hint cannot rescue an
already late seed or guarantee future execution time.

Deterministic tests use simulated costs at three time scales to check admission,
feasible checkpoint retention and explicit rejection. They also check request
reset, invalid measurements, materialization accounting and disabled modes.
These are scheduling contract tests with no dependency on machine speed.
The [deadline-free verification](investigations/observed-admission-parity/report.json)
replays the preceding comparison's 26 saved requests with the option off and on.
All pass bitwise returned-array parity and equal objective, feasibility, status
and work counts. Both outputs receive independent nonlinear waveform checks.
The floor stays inactive throughout because these solves have no deadline.

The verification archives its executed sources at baseline `7c6e88a`. Reproduce
it using the existing fitted fixture, without a timing acceptance criterion:

```sh
uv run python scripts/verify_observed_admission.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --records docs/investigations \
  --output /tmp/observed-admission-parity
```

No new host timing study accompanies this option. Earlier deadline outcomes are
observations under their recorded budgets and host conditions, not portable
requirements on the model or optimization formulation. Application-specific
budgets remain a deployment choice. Further performance work should target
computational cost and repeated work, with numerical correctness assessed
separately from hardware timing. This change makes no speedup or timed-recovery
claim and leaves the control formulation unchanged.

## Sharing the nominal rollout during differentiation

`FusedLinearizationSolver` now provides an experimental implementation of the
same recovery formulation. The baseline predicts a nominal trajectory and then
predicts it again inside the parameter directional derivative used for covariance.
The new path returns that derivative's nominal states and actuator states as
auxiliary outputs, removing the separate nominal pass during linearization.
Trial evaluation, finalization and the common solver kernels use the baseline
implementation. The default research solver and maintained production model are
unchanged.

The parameter state tangents are projected through the existing local-error
function's actual-state argument. Its reference is fixed for that inner
parameter derivative, while outer command derivatives still include the moving
local attitude frame. Actuator tangents remain in the dynamics carry. Forecast
error, parameter directions, physical commands and the horizon retain their
definitions. The alternate rollout receives each request's model values and
frozen middle explicitly; it does not reuse an earlier request's trajectory.

The [derivative-only comparison](investigations/fused-linearization/report.json)
passes all 26 saved requests with **bitwise equality** of seed Jacobians, seed
values and prepared predictions, complete returned arrays, and final objective.
Optimizer status, nonlinear feasibility and solver work counts also agree.
Both variants' outputs receive independent physical-waveform checks. These
are deadline-free solves using the existing fitted fixture; no new fitting or
host timing acceptance criterion is involved.

The compiler retains the separate scans in the baseline. The recorded compiled
loop sites confirm the bounded change:

| Kernel | Baseline | Derivative-only fusion |
| --- | ---: | ---: |
| Physical-command rollout | 2 | 2 |
| Residual and constraint linearization | 2 | 1 |
| Final scoring and gradient | 4 | 4 |

These counts describe loop structure, not a proportional runtime reduction.
The compiler's arithmetic estimate for linearization falls by 0.29%, and its
estimated bytes accessed by 1.84%. Those compiler/backend estimates are neither
measured request costs nor portable speedup claims. The structural saving is
real but modest in arithmetic terms; further performance work should examine
command and parameter sensitivity propagation.

An earlier [full-rollout fusion comparison](investigations/fused-covariance/report.json)
also changes trial evaluation and finalization. All 26 seed Jacobians and prepared
predictions are bitwise equal, all final objectives agree exactly, and all returned
arrays meet the predeclared numerical tolerance. Only 20 returned-array pairs are
bitwise equal. One request, `cold_small_39`, takes seven rather than ten value
evaluations, so that comparison **fails its equal-work-count criterion**. The
negative result remains recorded; its criterion was not relaxed.

The [separate branch trace](investigations/fused-covariance-branch/report.json)
reproduces those counts. Both QP rounds have identical variables, gradients,
steps and multipliers. At the second round's step fraction of 1/32, identical
candidate commands produce residuals differing by at most 1.49e-8. The merit
minus Armijo threshold changes from +2.364e-9 to -1.540e-10, flipping acceptance.
This localizes the branch change to floating-point trial evaluation. It motivates
limiting fusion to linearization; neither the Armijo rule nor its tolerances
change. The trace saves the compared arrays, not just their summaries.

Twelve additional mathematical tests cover multirotor and fixed-wing dynamics,
moving attitude, actuator lag, absent/empty/multiple covariance directions,
dynamic model values, command Jacobians of covariance/residuals/margins, reverse
cost gradients and bounded finite differences. The comparisons archive their
executed sources at baseline `52a1537`. Reproduce the selected variant into a new
directory:

```sh
uv run python scripts/investigate_fused_covariance.py \
  --scope linearization \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --records docs/investigations \
  --output /tmp/fused-linearization
```

Use `--scope full` for the broader diagnostic variant; its recorded comparison
fails as described above. To reproduce the branch trace against that recording,
load its archived research imports before the current scripts:

```sh
PYTHONPATH=docs/investigations/fused-covariance/executed-sources:scripts \
  uv run python -c 'import runpy; runpy.run_path("scripts/trace_fused_covariance.py", run_name="__main__")' \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --output /tmp/fused-covariance-branch
```

The trace driver now resolves the actual imported source paths for that provenance
check. Its archived executed source predates this import-resolution improvement;
the numerical trace was not repeated after it. No new closed-loop recovery or
timing qualification follows from the saved-request comparisons.

## Stopping backtracking at finite precision

`GaussNewtonReference(..., precision_stopping=True)` adds two experimental
stopping checks. The option remains off by default and does not change Armijo
acceptance, nonlinear feasibility tolerances, the objective or the control
formulation. The recorded comparison uses the preceding derivative-only solver
with single-seed reuse enabled in both arms.

The exact check stops when a clipped trial, converted to the evaluator's actual
dtype, has the same bytes as the current point. Further halving cannot leave that
point's rounding cell. Equality with a previous trial is insufficient: later
halving can leave that intermediate plateau, and the Armijo threshold changes
with step fraction. Such trials still receive the existing acceptance test.

The additional heuristic stops only **after a finite, feasible trial fails
Armijo**. It requires a prepared checkpoint at the current point, no current
constraint violation, a successful usable QP, and a full step inside the command
box without clipping. Its nonnegative full linear predicted decrease, `-g.T p`,
must be smaller than `spacing(checkpoint.value)` in the reported objective's
own floating-point dtype. There is no absolute cost floor or host time threshold.
Tests cover objective dtype and power-of-two cost scaling.

For the positive-definite regularized GN model, the decrease along the remaining
ray is bounded above by `-g.T p` for step fractions between zero and one. This is
a statement about that local quadratic model, not a bound on nonlinear
improvement. A useful full step still gets evaluated and may be accepted.
Infeasible or nonfinite rejected trials keep backtracking. Either stop returns
only an already checked feasible candidate; otherwise the solve rejects.
Neither stop asserts constrained convergence.

The first [pretrial stopping experiment](investigations/precision-stopping/report.json)
fails two of its 26 saved-request checks. It stops before evaluating the full
step: `cold_small_60` and `cold_small_68` retain objectives within one scalar
increment, but their maximum returned-array differences are 3.110e-4 and
2.518e-4, exceeding the predeclared comparison tolerance. Its planned closed-loop
extension does not run. This result demonstrates why tiny predicted objective
improvement alone is insufficient to discard a potentially useful trial.

The narrower [backtracking comparison](investigations/precision-backtracking/report.json)
passes all 26 fixed requests under the same numerical tolerances. All 20 requests
without a precision stop retain bitwise outputs, objectives and equal work
counts. Five stopped requests also return identical outputs and objectives.
The remaining `cold_small_39` request reduces value evaluations from ten to two
and needs no finalizer; its largest returned-array difference is 1.311e-6 and its
objective differs by one scalar increment, 7.451e-9. Across the saved requests,
linearizations fall from 70 to 66, value evaluations from 78 to 61, and finalizers
from 21 to 20. All outputs receive independent nonlinear waveform checks.

Both variants then complete all three deadline-free recovery scenarios: **six
arms of 120 intervals**, all actual states inside support and all existing
terminal full-state tolerances satisfied. The scheduled kick is reached, and
each kick arm's preceding history matches its corresponding unperturbed arm.

| Recovery case | Baseline value evaluations | Precision value evaluations | Precision stops |
| --- | ---: | ---: | --- |
| Original | 246 | 245 | 1 representable-step stop |
| Small | 349 | 222 | 22 representable-step and 14 model-resolution stops |
| Original with kick | 246 | 245 | 1 representable-step stop |

The original and kick arms retain identical state, actuator and forecast arrays.
The small case develops different request histories: maximum state and actuator
differences are 1.889e-5 and 1.955e-5, while its maximum forecast-command
difference is 3.553e-4. Its normalized tracking RMS ratio is 0.999960 relative
to baseline. Closed-loop checks use support, completion and existing terminal
tolerances; they do not require identical evolving forecasts. These counts
establish avoided model calls, without a runtime speedup or stability claim.
The [saved-data audit](investigations/precision-backtracking-audit.json) checks
source and fixture hashes, request/forecast/seed identities, applied counts,
recomputed support and recovery scores, and the matched kick histories. It
retains the rejected pretrial design and performs no solves or refits.

[Hager and Zhang's line-search analysis](https://people.clas.ufl.edu/hager/files/cg_descent.pdf)
discusses reduced accuracy of sufficient-decrease tests near a local minimum
and develops derivative-based approximate Wolfe conditions for smooth
unconstrained optimization. That is a separate research path; the stopping
heuristic here does not inherit their convergence results.

The experiment archives its executed sources at baseline `ac0ff55`. Twenty-five
additional regression cases cover dtype/scaling, checkpoint and feasibility
guards, accepted full steps, infeasible/nonfinite trial rejection, and both
precision settings at the common late-result boundary.
Reproduce the selected comparison into a new directory using the existing fitted
fixture:

```sh
uv run python scripts/investigate_precision_stopping.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --records docs/investigations \
  --output /tmp/precision-backtracking
uv run python scripts/audit_precision_stopping.py \
  --fixtures /tmp/glassbox-nmpc-fixture \
  --records docs/investigations \
  --output /tmp/precision-backtracking-audit.json
```
