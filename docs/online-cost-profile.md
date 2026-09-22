# Online update cost

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
