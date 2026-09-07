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

The next validation target is a full recovery including startup and a disturbance
after the controller has begun operating. That should precede promoting this
restricted command layout into the maintained controller. The first four commands
now provide immediate freedom, but the fixed middle and tight timing headroom
still require evaluation beyond this local case.
