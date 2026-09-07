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

The next formulation step is to retain suffix freedom while also permitting
changes to near-term commands in the primary NMPC solve. The frozen-prefix
restriction cannot be promoted as a general feedback controller solely because
its warm request fits this budget.
