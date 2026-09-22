# Online update cost

The measured cost points to repeated derivative calculations in the numerical
proposal. The quad proposal takes roughly **75–76 ms**, against a public update
of **77–79 ms**; the public model snapshot takes only **0.054 ms**. This is a cost
diagnosis, with no faster learner implemented or accuracy change claimed.

The [frozen saved-session profile](harness/online-cost-profile-v1.json) diagnoses
the adopted v8 update without changing any of the fifteen package files. It uses
35 predetermined checkpoints: 29 genuine next observations on disposable session
copies and six final-cache probes without a fabricated observation. Each scope
has one qualification call, three warmups and 21 retained timing samples.

## Qualification failure retained

The first attempt, source `09cef42`, stopped at the first quad checkpoint after
one exact native replay and twelve component calls. Only the standalone
solver-prefix delta failed the frozen component tolerance (`rtol=2e-8`,
`atol=2e-10`): maximum absolute difference 7.90144e-8, maximum scaled error
351.6096, and 2,164 of 10,130 coordinates outside tolerance. Full instrumented and
native proposals matched exactly, as did the trust-stage arrays; the trust shrink
was one. The setup gradient differed by at most 3.55e-15. Different compiler
fusion/evaluation amplified through PCG is a plausible explanation, not a proved
root cause. This does not establish a regression in the unchanged learner.

The failed pack remains sealed at
`artifacts/online-cost-profile-v1/profile`, authority
`230d06d6c8149c5a09321d2c075c1f273d6022f705d7c55d4497bbcde67c4d39`.
It contains qualification/first-call timings only, with no warmed timing samples.

The continuation changes failure collection only: finite floating-point tolerance
violations remain failed checks while the unchanged roster and repetitions are
measured. Source/input integrity, shape, dtype, nonfinite-mask, discrete-decision
and native-replay failures remain fatal. No threshold, kernel, parameter, fit
budget or point-selection rule changes. Measurement completion and scientific
qualification are reported separately; a complete timing pack cannot erase a
failed numerical check. Unqualified diagnostic timings cannot support a claimed
production decomposition.

## Completed measurements

The continuation, source `9f27b54`, completed every point and all 725 exact native
replays. Its authority is
`325291f6d4c129349b63021f9ab59bca47cf47eb84f51c103461a9e1ea9a596c`, at
`artifacts/online-cost-profile-v1/profile-retained`. Across both attempts there
were 726 native replays, 10,512 component dispatches and zero initializations.

Values below are **medians of per-point medians, in milliseconds**. Initial and
captured checkpoints replay real next observations. Final checkpoints measure
retained caches only, so no public-update value exists. The 23 fixed-wing captures
are kept separate from endpoints rather than weighted as a representative stream.

| Checkpoints | Count | Native proposal | Public update | Cached curvature product | Setup prefix | Trust prefix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| quad initial | 4 | 75.444 | 77.384 | 4.676 | 5.211 | 73.787 |
| quad final | 4 | 75.456 | — | 4.664 | 5.067 | 74.049 |
| fixedwing initial | 2 | 9.097 | 10.076 | 0.660 | 0.753 | 8.646 |
| fixedwing final | 2 | 9.114 | — | 0.651 | 0.751 | 8.594 |
| fixedwing capture | 23 | 9.125 | 10.084 | 0.658 | 0.753 | 8.630 |

Native proposal/public median ratios are 0.967–0.976 for quad initial checkpoints
and 0.898–0.913 across genuine fixed-wing checkpoints. These independently timed
scopes cannot be subtracted to infer host overhead. The public observe/snapshot
split uses nested timestamps and is additive; snapshot medians span only
0.048–0.055 ms across all genuine points.

The passing setup/trust diagnostics and cached curvature actions consistently
point to repeated derivative applications after setup. Each proposal executes
sixteen curvature products, plus the initial gradient transpose and trust
projection. Native v8 already retains one nonlinear linearization per proposal;
recomputing that primal linearization is not the repeated work. The individual
scopes have different compiler fusion/materialization and are **not additive
production shares**. The history-specific fraction remains unmeasured.

Full component qualification still **fails**. Only the standalone solver prefix
fails, at all four initial quad checkpoints. Maximum absolute delta differences
range from 7.90e-8 to 2.55e-6; maximum scaled errors range from 351.61 to 10619.26.
All other checks pass at all 35 points; the worst passing scaled error is
6.55e-7, far below the acceptance threshold of one. Full instrumented/native
outputs and trust-stage arrays match exactly. The failing solver-prefix timings
are excluded from these conclusions; their values remain in the raw pack.

The independent saved-data audit authenticates 257 profile payloads and 400
parent payloads, reconstructs all 725 post-update sessions and validates 12,675
raw timing samples without model or optimizer calls. All fifteen package files
remain byte-identical to v8. **464 tests pass** (414 existing and 50 profile tests),
and Ruff passes. The previous exact offline/Dart baseline replay remains
applicable because package source did not change. The [machine-readable
index](online-cost-profile.json) binds both attempts, runtime, checks and results.

These repeated snapshots do not requalify real-time operation or establish
broader generalization. They add no new forecast-accuracy evidence or closed-loop recovery trials.

## Next iteration

Test one implementation change: cache the small learned-memory transport matrices
and batch the history contribution to parameter JVPs and VJPs. This could replace
repeated sequential small matrix multiplies through recorded history, without a
dense forecast Jacobian. It is a hypothesis informed by this profile, not an established
history bottleneck or promised speedup.

Freeze numerical, forecast and latency qualification before implementing it.
Preserve the model, prior, conditioning, sixteen solver steps, trust bound and
backtracking rules. Reordered contractions can change floating-point results, as
this profile demonstrates; mathematical equivalence alone is insufficient. Measure
setup cost and whole-proposal time as well as individual derivative actions. The
10 ms quad cadence remains a target, with no claim it is reached or guaranteed.
